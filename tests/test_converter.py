import unittest

import numpy as np
from graphix import Circuit

from graphix_perceval.converter import to_perceval
from graphix_perceval.experiment import PhotonDistribution


class TestConverter(unittest.TestCase):
    def test_sampling_circuit_wo_postselection(self):
        circuit = Circuit(2)
        circuit.h(1)
        circuit.cnot(0, 1)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()

        exp = to_perceval(pattern)
        exp.set_local_processor("SLOS")
        dist = exp.sample(num_samples=1000, format_result=False, postselection=False)
        counts = 0
        for k, v in dist.items():
            counts += v
        self.assertEqual(counts, 1000)

    def test_sampling_circuit_w_postselection(self):
        circuit = Circuit(2)
        circuit.h(1)
        circuit.cnot(0, 1)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()

        exp = to_perceval(pattern)
        exp.set_local_processor("SLOS")
        dist = exp.sample(num_samples=1000)
        counts = 0
        for k, v in dist.items():
            self.assertTrue(k in ["|00>", "|11>"])
            counts += v
        self.assertEqual(counts, 1000)

    def test_zero_state_creation_wo_pauli_meas(self):
        circuit = Circuit(1)  # initialize with |+>
        circuit.h(0)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()

        exp = to_perceval(pattern)
        exp.set_local_processor("SLOS")
        dist = exp.get_probability_distribution()

        ans_dist = PhotonDistribution({"|0>": 1.0})
        self.assertAlmostEqual(dist["|0>"], ans_dist["|0>"])
        self.assertTrue(all(k in dist.keys() for k in ans_dist.keys()))

    def test_one_state_creation_wo_pauli_meas(self):
        circuit = Circuit(1)  # initialize with |+>
        circuit.h(0)
        circuit.x(0)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()

        exp = to_perceval(pattern)
        exp.set_local_processor("SLOS")
        dist = exp.get_probability_distribution()

        ans_dist = PhotonDistribution({"|1>": 1.0})
        self.assertAlmostEqual(dist["|1>"], ans_dist["|1>"])
        self.assertTrue(all(k in dist.keys() for k in ans_dist.keys()))

    def test_rotated_one_qubit_state_creation_wo_pauli_meas(self):
        circuit = Circuit(1)  # initialize with |+>
        circuit.h(0)
        circuit.rx(0, np.pi / 1.23)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()

        exp = to_perceval(pattern)
        exp.set_local_processor("SLOS")
        dist = exp.get_probability_distribution()

        ans_dist = PhotonDistribution({"|0>": np.cos(np.pi / 1.23 / 2) ** 2, "|1>": np.sin(np.pi / 1.23 / 2) ** 2})
        self.assertAlmostEqual(dist["|0>"], ans_dist["|0>"])
        self.assertAlmostEqual(dist["|1>"], ans_dist["|1>"])
        self.assertTrue(all(k in dist.keys() for k in ans_dist.keys()))

    def test_bell_state_phi_plus_creation_wo_pauli_meas(self):
        circuit = Circuit(2)  # initialize with |+> \otimes |+>
        circuit.h(1)
        circuit.cnot(0, 1)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()

        exp = to_perceval(pattern)
        exp.set_local_processor("SLOS")
        dist = exp.get_probability_distribution()

        ans_dist = PhotonDistribution({"|00>": 0.5, "|11>": 0.5})
        self.assertAlmostEqual(dist["|00>"], ans_dist["|00>"])
        self.assertAlmostEqual(dist["|11>"], ans_dist["|11>"])
        self.assertTrue(all(k in dist.keys() for k in ans_dist.keys()))

    def test_bell_state_phi_plus_creation_with_pauli_meas(self):
        circuit = Circuit(2)  # initialize with |+> \otimes |+>
        circuit.h(1)
        circuit.cnot(0, 1)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()

        exp = to_perceval(pattern)
        exp.set_local_processor("SLOS")
        dist = exp.get_probability_distribution()

        ans_dist = PhotonDistribution({"|00>": 0.5, "|11>": 0.5})
        self.assertAlmostEqual(dist["|00>"], ans_dist["|00>"])
        self.assertAlmostEqual(dist["|11>"], ans_dist["|11>"])
        self.assertTrue(all(k in dist.keys() for k in ans_dist.keys()))

    def test_ghz_state_creation_with_pauli_meas(self):
        circuit = Circuit(3)  # initialize with |+> \otimes |+> \otimes |+>
        circuit.h(1)
        circuit.h(2)
        circuit.cnot(0, 1)
        circuit.cnot(1, 2)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()

        exp = to_perceval(pattern)
        exp.set_local_processor("SLOS")
        dist = exp.get_probability_distribution()

        ans_dist = PhotonDistribution({"|000>": 0.5, "|111>": 0.5})
        self.assertAlmostEqual(dist["|000>"], ans_dist["|000>"])
        self.assertAlmostEqual(dist["|111>"], ans_dist["|111>"])
        self.assertTrue(all(k in dist.keys() for k in ans_dist.keys()))

    def test_bell_state_and_ry_with_pauli_meas(self):
        circuit = Circuit(2)  # initialize with |+> \otimes |+> \otimes |+>
        circuit.h(1)
        circuit.cnot(0, 1)
        circuit.ry(1, np.pi / 4)
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()

        exp = to_perceval(pattern)
        exp.set_local_processor("SLOS")
        dist = exp.get_probability_distribution()

        ans_dist = PhotonDistribution(
            {
                "|00>": (np.cos(np.pi / 8) / np.sqrt(2)) ** 2,
                "|01>": (np.sin(np.pi / 8) / np.sqrt(2)) ** 2,
                "|10>": (np.sin(np.pi / 8) / np.sqrt(2)) ** 2,
                "|11>": (np.cos(np.pi / 8) / np.sqrt(2)) ** 2,
            }
        )
        self.assertAlmostEqual(dist["|00>"], ans_dist["|00>"])
        self.assertAlmostEqual(dist["|01>"], ans_dist["|01>"])
        self.assertAlmostEqual(dist["|10>"], ans_dist["|10>"])
        self.assertAlmostEqual(dist["|11>"], ans_dist["|11>"])
        self.assertTrue(all(k in dist.keys() for k in ans_dist.keys()))
