from __future__ import annotations
from _collections_abc import dict_items
import itertools

from perceval.algorithm import Sampler
from perceval.utils import PostSelect
import perceval as pcvl
import sympy as sp
import warnings
from enum import Enum
from tabulate import tabulate

import sys

IS_NOTEBOOK = "ipykernel" in sys.modules
if IS_NOTEBOOK:
    from IPython.display import HTML, display  # type: ignore


class PhotonType(Enum):
    READOUT = "READOUT"
    COMPUTE = "COMPUTE"
    WITNESS = "WITNESS"
    LOSS = "LOSS"
    NONE = None

    def __str__(self) -> str:
        return self.name


class Photon:
    def __init__(self, exp_id: int, type: PhotonType, node_id: int = 0, angle: float | None = None):
        self.id = exp_id
        self.type = type
        self.node_id = node_id
        # angle for QWP and HWP
        if angle:
            self.angle = [sp.pi / 4, (sp.pi - (2 * angle * sp.pi)) / 8]
        else:
            self.angle = [sp.pi / 4, sp.pi / 8]  # X-basis measurement

    def __str__(self) -> str:
        return f"Photon(ID:{str(self.id)} Node:{str(self.node_id)} ({str(self.type)}))"

    def __repr__(self) -> str:
        return self.__str__()


class PercevalExperiment:
    """PercevalExperiment class for running MBQC patterns on Perceval simulators and Quandela devices.

    Attributes
    ----------
    pattern: :class:`graphix.Pattern` object
        MBQC pattern to be run on the device
    circ: :class:`perceval.Circuit` object
        Perceval circuit corresponding to the pattern.
    backend : str
        Name of a Perceval simulator or Quandela device
    """

    def __init__(self, circuit: pcvl.Circuit, photons: list[Photon]):
        """

        Parameters
        ----------
        pattern: :class:`graphix.Pattern` object
            MBQC pattern to be run on the Quandela device or Perceval simulator.
        """
        self.circ = circuit
        self.photons = photons
        self.processor = None
        self.input_state = None
        self.output_states: dict[str, str] | None = None

    def set_local_processor(self, backend: str, source: pcvl.Source = pcvl.Source(), name: str = None):
        """Set the local computing backend.

        Parameters
        ----------
        backend : str
            Name of a local backend.
        source : :class:`perceval.Source` object, optional
            Setting of single-photon source.
        name : str, optional
            Name for the processor.
        """
        if self.circ is None:
            warnings.warn("The circuit has not been converted to Perceval circuit. It will be converted automatically.")
            self.to_perceval()
        if self.processor is not None:
            warnings.warn("The processor has already been set. The previous processor will be overwritten.")
        self.processor = pcvl.Processor(backend=backend, m_circuit=self.circ, source=source, name=name)
        self.backend = backend

        self.set_input_state()
        self.set_output_states()

    def set_remote_processor(self, backend: str, token: str):
        """Set the remote computing backend.

        Parameters
        ----------
        backend : str
            Name of a remote backend.
        token : str
            Token for the remote processor.
        """
        if self.circ is None:
            warnings.warn("The circuit has not been converted to Perceval circuit. It will be converted automatically.")
            self.to_perceval()
        if self.processor is not None:
            warnings.warn("The processor has already been set. The previous processor will be overwritten.")
        self.processor = pcvl.RemoteProcessor(name=backend, token=token)
        self.processor.set_circuit(self.circ)
        self.backend = backend

        self.set_input_state()
        self.set_output_states()

    def set_input_state(self):
        """Set the input states for the processor.
        The default input state is |{P:H}> for each photon and |0> for each ancillary mode.
        """
        if self.processor is None:
            raise Exception(
                "No processor has been set. Please set a processor by `set_local_processor` or `set_remote_procesor` before running the experiment."
            )

        input_state = "|"
        input_state = input_state + ",".join([r"{P:H}" for _ in range(len(self.photons))])
        input_state = input_state + "," + ",".join(["0"] * len(self.photons))
        input_state = input_state + ">"

        self.input_state = pcvl.BasicState(input_state)
        self.processor.with_polarized_input(self.input_state)  # not with_input (it will not work for polarized input)

    def set_output_states(self):
        r"""Set the output states.
        Currently, Perceval does not support feed-forward opetations,
        so we postselect the output states where

        - The witness photons are in |{P:H}> and translated to |0,1>
        - The computing photons are in |{P:H}> and translated to |0,1>
        - The readout photons are in |{P:H}> or |{P:V}>
        """
        if self.processor is None:
            raise Exception(
                "No processor has been set. Please set a processor by `set_local_processor` or `set_remote_procesor` before running the experiment."
            )
        (readouts, witnesses, comps) = (
            self.get_readout_photons(),
            self.get_witness_photons(),
            self.get_compute_photons(),
        )
        out_states = {}
        x = 0
        (zero, one) = ([0, 1], [1, 0])
        for st in itertools.product([zero, one], repeat=len(readouts)):
            basic_out_state = [[]] * len(self.photons)
            for w in witnesses:
                basic_out_state[w.id] = zero
            for c in comps:
                basic_out_state[c.id] = zero
            for i in range(len(readouts)):
                basic_out_state[readouts[i].id] = st[i]
            out_states[
                str(pcvl.BasicState(list(itertools.chain.from_iterable(basic_out_state))))
            ] = f"|{x:0{len(readouts)}b}>"
            x = x + 1
        self.output_states = out_states

    def get_probability_distribution(self, format_result=True, postselection=True):
        if self.processor is None:
            raise Exception(
                "No processor has been set. Please set a processor by `set_local_processor` or `set_remote_procesor` before running the experiment."
            )
        if postselection:
            self.set_postselection()

        sampler = Sampler(self.processor)
        probs = PhotonDistribution(sampler.probs()["results"])

        if format_result:
            probs = self.format_result(probs)

        return probs

    def sample(self, num_samples=1024, format_result=True, postselection=True):
        """Run the MBQC pattern on IBMQ devices

        Parameters
        ----------
        num_samples : int, optional
            the number of samples.
        format_result : bool, optional
            whether to format the result so that only the result corresponding to the output qubit is taken out.

        Returns
        -------
        result : dict
            the measurement result.
        """
        if self.processor is None:
            raise Exception(
                "No processor has been set. Please set a processor by `set_local_processor` or `set_remote_procesor` before running the experiment."
            )
        if postselection:
            self.set_postselection()

        sampler = Sampler(self.processor)
        sample_result = PhotonDistribution(sampler.samples(num_samples)["results"])

        if format_result:
            sample_result = self.format_result(sample_result)

        return sample_result

    def format_result(self, result: PhotonDistribution) -> PhotonDistribution:
        """Format the result to replace the dual-rail encoded qubit with logical qubit.

        Returns
        -------
        masked_results : dict
            Dictionary of formatted results.
        """
        masked_results = PhotonDistribution()
        # Iterate over original measurement results
        for key, value in result.items():
            if str(key) not in self.output_states:
                continue
            masked_results[self.output_states[str(key)]] = value
        return masked_results

    def set_postselection(self):
        """Postselect the results according to the pattern."""
        ps = PostSelect()
        for ph in self.get_readout_photons():
            ps.eq([2 * ph.id, 2 * ph.id + 1], 1)
        for ph in self.get_compute_photons():
            ps.eq([2 * ph.id], 0).eq([2 * ph.id + 1], 1)
        for ph in self.get_witness_photons():
            ps.eq([2 * ph.id], 0).eq([2 * ph.id + 1], 1)

        self.processor.set_postselection(ps)

    def get_readout_photons(self):
        return [ph for ph in self.photons if ph.type == PhotonType.READOUT]

    def get_compute_photons(self):
        return [ph for ph in self.photons if ph.type == PhotonType.COMPUTE]

    def get_witness_photons(self):
        return [ph for ph in self.photons if ph.type == PhotonType.WITNESS]

    def run(self):
        if self.processor is None:
            raise Exception(
                "No processor has been set. Please set a processor by `set_local_processor` or `set_remote_procesor` before running the experiment."
            )
        if self.input_state is None:
            self.set_input_state()
        if self.output_states is None:
            self.set_output_states()

        ca = pcvl.algorithm.Analyzer(self.processor, input_states=self.input_state, output_states=self.output_states)
        return ca


class PhotonDistribution(dict):
    """PhotonDistribution class for storing the probability distribution of the measurement results.

    perceval.BSDistribution does not seem to show fock state with one qubit properly."""

    def __init__(self, distribution: dict[str, float] = {}):
        # TODO: use sympy.physics.secondquant.FockStateBosonBra?
        if not isinstance(distribution, dict):
            raise TypeError("distribution must be a dictionary.")
        super().__init__()
        self.distribution = dict(distribution)

    def __str__(self) -> str:
        return str(self.distribution)

    def __getitem__(self, key: str) -> float:
        if not isinstance(key, str):
            raise TypeError("key must be a string.")
        return self.distribution[key]

    def __setitem__(self, key: str, value: float):
        if not isinstance(key, str):
            raise TypeError("key must be a string.")
        if not isinstance(value, float):
            raise TypeError("value must be a float.")
        self.distribution[key] = value

    def items(self) -> dict_items:
        return self.distribution.items()

    def draw(self, sort: bool = True):
        """Draw the probability distribution in a table.
        If the code is run in a Jupyter notebook, the table will be displayed in HTML format.
        If the code is run in a terminal, the table will be displayed in ASCII format.

        Parameters
        ----------
        sort : bool, optional
            Whether to sort the distribution by the key.
        """
        headers = ["state", "probability"]
        d = []
        for key, value in self.distribution.items():
            d.append([str(key), value])
        if sort:
            d.sort()
        if IS_NOTEBOOK:
            table = tabulate(d, headers=headers, tablefmt="html")
            display(HTML(table))
        else:
            table = tabulate(d, headers=headers, tablefmt="pretty")
            print(table)
