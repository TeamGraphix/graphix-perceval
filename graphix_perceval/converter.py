from __future__ import annotations
import graphix
import numpy as np
import perceval as pcvl
from perceval import components as comp
import sympy as sp

from graphix_perceval.experiment import PercevalExperiment, Photon, PhotonType
from graphix_perceval.extraction import (
    Cluster,
    ClusterType,
    extract_clusters_from_graph,
)
from graphix_perceval.clifford import CLIFFORD_TO_PERCEVAL_POLAR


def pattern2graphstate(pattern: graphix.Pattern) -> tuple[graphix.GraphState, dict[int, float], list[int]]:
    """Create a graph state from a MBQC pattern.

    Parameters
    ----------
    pattern : :class:`graphix.Pattern` object
        MBQC pattern to be run on the device

    Returns
    -------
    graph_state : :class:`graphix.GraphState` object
        Graph state corresponding to the pattern.
    phasedict : dict
        Dictionary of phases for each node.
    output_nodes : list
        List of output nodes.
    """
    nodes, edges = pattern.get_graph()
    vop_init = pattern.get_vops()
    graph_state = graphix.GraphState(nodes=nodes, edges=edges, vops=vop_init)
    phasedict = {}
    for command in pattern.get_measurement_commands():
        phasedict[command[1]] = command[3]

    output_nodes = pattern.output_nodes
    return graph_state, phasedict, output_nodes


def to_perceval(pattern: graphix.Pattern) -> PercevalExperiment:
    """Convert a graphix.Pattern to a perceval.Circuit.

    Parameters
    ----------
    pattern : graphix.Pattern
        GraphState to be converted to a perceval.Circuit

    Returns
    -------
    experiment : PercevalExperiment
        :class:`graphix_perceval.experiment.PercevalExperiment` object
    """
    if not isinstance(pattern, graphix.Pattern):
        raise TypeError("pattern must be a graphix.Pattern object")
    graph_state, phasedict, output_nodes = pattern2graphstate(pattern)
    clusters = extract_clusters_from_graph(graph_state)
    vops = pattern.get_vops()

    pcc = PercevalCircuitConstructor()
    for cluster in clusters:
        pcc.add_cluster(cluster, phasedict, output_nodes)

    pcc.add_fusions()

    pcc.apply_local_clifford(vops)

    perceval_circuit = pcc.setup_perceval_circuit()

    exp = PercevalExperiment(perceval_circuit, pcc.photons)

    return exp


class PercevalCircuitConstructor:
    def __init__(self):
        self.num_photons = 0
        self.clusters: list[Cluster] = []
        self.fusion_pairs: list[tuple[Photon, Photon]] = []
        self.photons: list[Photon] = []
        self.node_id2photon_ids: dict[int, list[int]] = {}
        self.clifford_comps: list[comp.BS] = []
        self._is_fused: bool = False
        self._clifford_applied: bool = False

    def add_cluster(self, cluster: Cluster, phasedict: dict[int, float], readouts: list) -> None:
        if self._is_fused:
            raise RuntimeError("Cannot add cluster after fusion")
        for node_id in cluster.graph.nodes:
            if node_id in readouts:
                ph = Photon(
                    exp_id=self.num_photons, type=PhotonType.READOUT, node_id=node_id, angle=phasedict.get(node_id)
                )
            else:
                ph = Photon(
                    exp_id=self.num_photons, type=PhotonType.COMPUTE, node_id=node_id, angle=phasedict.get(node_id)
                )
            if self.node_id2photon_ids.get(node_id) is not None:
                self.node_id2photon_ids[node_id].append(self.num_photons)
            else:
                self.node_id2photon_ids[node_id] = [self.num_photons]
            self.num_photons += 1
            self.photons.append(ph)

        if cluster.type in (ClusterType.GHZ, ClusterType.LINEAR):
            self.clusters.append(cluster)
        else:
            raise TypeError(f"ClusterType {cluster.type} is not supported")

    def get_readouts(self) -> list[Photon]:
        return [ph for ph in self.photons if ph.type == PhotonType.READOUT]

    def get_computes(self) -> list[Photon]:
        return [ph for ph in self.photons if ph.type == PhotonType.COMPUTE]

    def get_witnesses(self) -> list[Photon]:
        return [ph for ph in self.photons if ph.type == PhotonType.WITNESS]

    def add_fusions(self) -> None:
        """Find edges that connects two clusters.
        If the two clusters share a same node id, then they are fused.
        Type-1 fusion.
        """
        for _, photon_ids in self.node_id2photon_ids.items():
            fusing_photons = sorted(photon_ids)
            for idx in range(len(fusing_photons) - 1):
                self.fusion_pairs.append((self.photons[fusing_photons[idx]], self.photons[fusing_photons[idx + 1]]))
                # Note that the photon with the larger index is the witness
                self.photons[fusing_photons[idx + 1]].type = PhotonType.WITNESS

        self._is_fused = True

    def get_all_clusters(self) -> list[Cluster]:
        return self.ghz_clusters | self.linear_clusters

    def setup_perceval_circuit(self, name: str | None = None, merge: bool = False) -> pcvl.Circuit:
        if not self._is_fused:
            raise RuntimeError("Must fuse before setting up perceval circuit")
        if not self._clifford_applied:
            raise RuntimeError("Must apply local clifford before setting up perceval circuit")
        circ = pcvl.Circuit(self.num_photons * 2, name=name)
        # Create circuits for all the clusters
        photon_idx = 0
        for cl in self.clusters:
            if cl.type == ClusterType.GHZ:
                circ.add(
                    [idx for idx in range(photon_idx, photon_idx + len(cl.graph.nodes))],
                    ghz_circuit(len(cl.graph.nodes)),
                    merge,
                )
            elif cl.type == ClusterType.LINEAR:
                circ.add(
                    [idx for idx in range(photon_idx, photon_idx + len(cl.graph.nodes))],
                    linear_circuit(len(cl.graph.nodes)),
                    merge,
                )
            photon_idx += len(cl.graph.nodes)

        circ.add(0, comp.PERM(list(range(self.num_photons))))  # work as a barrier

        # Create circuits for all the Fusions
        for ph1, ph2 in self.fusion_pairs:
            circ.add(list(range(ph1.id, ph2.id + 1)), fusion_circuit(ph1, ph2), merge)

        circ.add(0, comp.PERM(list(range(self.num_photons))))  # work as a barrier

        # Add local clifford
        for photon_id, clifford_comp in self.clifford_comps:
            circ.add(photon_id, clifford_comp)

        circ.add(0, comp.PERM(list(range(self.num_photons * 2))))  # work as a barrier

        # Convert measurement basis
        for ph in self.photons:
            if ph.type == PhotonType.COMPUTE:
                circ.add(ph.id, comp.QWP(ph.angle[0]))
                circ.add(ph.id, comp.HWP(ph.angle[1]))

        # Currently, Perceval does not support measurement in polarization, so we need to convert it to dual-rail encoding.
        # |{P:H},0> -> |0,1> = |0> (this is the opposite of the definition in perceval)
        # |{P:V},0> -> |1,0> = |1> (this is the opposite of the definition in perceval)
        circ.add(
            0,
            comp.PERM(
                sum(
                    list([2 * i] for i in range(0, self.num_photons))
                    + list([2 * i + 1] for i in range(0, self.num_photons)),
                    [],
                )
            ),
        )
        for i in range(0, self.num_photons):
            circ.add(i * 2, comp.PBS())

        return circ

    def apply_local_clifford(self, vops: dict[int, int]) -> None:
        if not self._is_fused:
            raise RuntimeError("Must fuse before applying local clifford")
        for node_id, cid in vops.items():
            for ph_id in self.node_id2photon_ids[node_id]:
                if self.photons[ph_id].type != PhotonType.WITNESS:
                    self.clifford_comps.append((ph_id, local_clifford_circuit(cid)))

        self._clifford_applied = True


def local_clifford_circuit(clifford_id: int) -> pcvl.Circuit:
    """Create a Perceval Circuit for a local clifford.

    Parameters
    ----------
    mode_id : int
        Mode id.
    clifford_id : int
        Clifford id.

    Returns
    -------
    perceval.Circuit
        Perceval Circuit for a local clifford.
    """
    if not 0 <= clifford_id <= 23:
        raise ValueError("clifford_id must be in [0, 23]")
    circ = pcvl.Circuit(m=1, name="LOCAL CLIFFORD ID:" + str(clifford_id))
    for comps in CLIFFORD_TO_PERCEVAL_POLAR[clifford_id]:
        circ.add(0, comps)
    return circ


def fusion_circuit(ph1: Photon, ph2: Photon) -> pcvl.Circuit:
    """Create a Perceval Circuit for fusing two photons.

    Parameters
    ----------
    ph1 : Photon
        First photon.
    ph2 : Photon
        Second photon.

    Returns
    -------
    perceval.Circuit
        Perceval Circuit for fusing two photons.
    """
    if not isinstance(ph1, Photon) or not isinstance(ph2, Photon):
        raise TypeError("ph1 and ph2 must be Photon objects")
    if ph1.type == PhotonType.WITNESS and ph1.id > ph2.id:
        ph1, ph2 = ph2, ph1
    if ph2.type != PhotonType.WITNESS:
        raise ValueError("The second photon must be a witness")
    l = ph2.id - ph1.id
    circ = pcvl.Circuit(m=l + 1, name="FUSE " + str(ph1.id) + "-" + str(ph2.id))
    # If the photons are not neighbors, we swap the ph2 and the photon next to ph1,
    # do the fusion and swap back.
    if l > 1:
        a, *b, c = list(range(0, l))
        perm = comp.PERM([c, *b, a])
        circ.add(1, perm)
    circ.add((0, 1), comp.PBS())
    circ.add(1, comp.HWP(sp.pi / 8))
    if l > 1:
        circ.add(1, perm)
    return circ


def linear_circuit(num_photons: int, name: str = "") -> pcvl.Circuit:
    """Create a Perceval Circuit for a linear cluster.

    Parameters
    ----------
    num_photons : int
        Number of photons.

    Returns
    -------
    perceval.Circuit
        Perceval Circuit for a linear cluster.
    """
    if not isinstance(num_photons, int):
        raise TypeError("num_photons must be an integer")
    circ = pcvl.Circuit(m=num_photons, name="LINEAR " + name)
    for i in range(num_photons):
        circ.add(i, comp.HWP(sp.pi / 8))

    for i in range(num_photons - 1):
        circ.add((i, i + 1), comp.PBS())
        if i >= 1 and i != num_photons - 2:
            circ.add(i + 1, comp.HWP(sp.pi / 8))
    circ.add(0, comp.PERM(list(range(num_photons))))  # work as a barrier
    circ.add(0, comp.HWP(sp.pi / 8))
    circ.add(num_photons - 1, comp.HWP(sp.pi / 8))

    return circ


def ghz_circuit(num_photons: int, name: str = "") -> pcvl.Circuit:
    """Create a Perceval Circuit for a GHZ cluster.

    Parameters
    ----------
    num_photons : int
        Number of photons.

    Returns
    -------
    perceval.Circuit
        Perceval Circuit for a GHZ cluster.
    """
    if not isinstance(num_photons, int):
        raise TypeError("num_photons must be an integer")
    circ = pcvl.Circuit(m=num_photons, name="GHZ " + name)
    for i in range(num_photons):
        circ.add(i, comp.HWP(sp.pi / 8))

    for i in range(num_photons - 1):
        circ.add((i, i + 1), comp.PBS())

    for i in range(1, num_photons):
        circ.add(i, comp.HWP(sp.pi / 8))  # Hadamard

    return circ
