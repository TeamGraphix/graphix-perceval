import unittest

import numpy as np
import perceval as pcvl
from graphix.clifford import CLIFFORD
from graphix_perceval.clifford import (
    CLIFFORD_TO_PERCEVAL_BS,
    CLIFFORD_TO_PERCEVAL_POLAR,
)
from sympy import matrix2numpy


class TestConverter(unittest.TestCase):
    def test_clifford_conversion_BS(self):
        for idx in range(len(CLIFFORD_TO_PERCEVAL_BS)):
            circ = pcvl.Circuit(2)
            for component in CLIFFORD_TO_PERCEVAL_BS[idx]:
                circ.add(0, component)
            self.assertTrue(np.allclose(matrix2numpy(circ.U, dtype=complex), CLIFFORD[idx].astype(complex)))
        assert len(CLIFFORD_TO_PERCEVAL_BS) == len(CLIFFORD)

    def test_clifford_conversion_polar(self):
        for idx in range(len(CLIFFORD_TO_PERCEVAL_POLAR)):
            circ = pcvl.Circuit(1)
            for component in CLIFFORD_TO_PERCEVAL_POLAR[idx]:
                circ.add(0, component)
            self.assertTrue(np.allclose(matrix2numpy(circ.U, dtype=complex), CLIFFORD[idx].astype(complex)))
        assert len(CLIFFORD_TO_PERCEVAL_POLAR) == len(CLIFFORD)
