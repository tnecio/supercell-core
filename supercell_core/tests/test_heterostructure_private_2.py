import unittest as ut
from os import path

import numpy as np

import supercell_core as sc


class MockHeterostructureMoire(sc.Heterostructure):
    def _get_dt_As(self, max_el: int) -> np.ndarray:
        return np.array([
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[1, 1], [2, 2], [3, 3], [4, 4]],
            [[5, 5], [6, 6], [7, 7], [8, 8]],
            [[-1, 1], [-9, 2], [3, 5], [4, -3]],
        ])

class TestHeterostructurePrivate2(ut.TestCase):
    def test_moire_1(self):
        h = MockHeterostructureMoire()

        theta = 0
        ABr = np.identity(2)
        A_inv = h._prepare_A_inv(theta, ABr)
        self.assertTrue(np.allclose(A_inv, np.identity(2)))
        qtys = h._moire(10, A_inv)
        self.assertTrue(np.allclose(qtys, np.zeros((4, 4))))

        theta = np.arctan(4 / 3)
        ABr = np.identity(2)
        A_inv = h._prepare_A_inv(theta, ABr)
        self.assertTrue(np.allclose(A_inv, np.array([
            [3 / 5, 4 / 5],
            [-4 / 5, 3 / 5]
        ])))
        qtys = h._moire(10, A_inv)
        #self.assertTrue(np.allclose(qtys[0, 2], 0))
        print(qtys)