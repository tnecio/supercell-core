import unittest as ut

from ..calc import *


class TestCalc(ut.TestCase):
    def test_inv(self):
        # this also tests det since inv uses det internally
        a = 3
        m = np.random.random((a, a, a, a, 2, 2))
        expected = np.empty((a, a, a, a, 2, 2))
        for i in range(a):
            for j in range(a):
                for k in range(a):
                    for l in range(a):
                        expected[i, j, k, l] = np.linalg.inv(m[i, j, k, l])
        self.assertTrue(np.allclose(expected, inv(m)))

    def test_matvecmul(self):
        a = 3
        m1 = np.random.random((a, a, a, a, 2, 2))
        m2 = np.random.random((a, a, a, a, 2))
        expected = np.empty((a, a, a, a, 2))
        for i in range(a):
            for j in range(a):
                for k in range(a):
                    for l in range(a):
                        expected[i, j, k, l] = m1[i, j, k, l] @ m2[i, j, k, l]
        actual = matvecmul(m1, m2)
        self.assertTrue(np.allclose(expected, actual))

    def test_rotate(self):
        a = 3
        theta = np.random.random()
        m = np.random.random((a, a, a, a, 2, 2))
        expected = np.empty((a, a, a, a, 2, 2))
        for i in range(a):
            for j in range(a):
                for k in range(a):
                    for l in range(a):
                        expected[i, j, k, l] = np.array([[np.cos(theta), -np.sin(theta)],
                                                         [np.sin(theta), np.cos(theta)]]) @ m[i, j, k, l]
        actual = rotate(m, theta)
        self.assertTrue(np.allclose(expected, actual))

    def test_matnorm(self):
        a = 3
        theta = np.random.random()
        m = np.random.random((a, a, a, a, 2, 2))
        expected1 = np.empty((a, a, a, a))
        expected2 = np.empty((a, a, a, a))
        for i in range(a):
            for j in range(a):
                for k in range(a):
                    for l in range(a):
                        expected1[i, j, k, l] = np.max(np.sum(np.abs(m[i, j, k, l])))
                        expected2[i, j, k, l] = np.linalg.norm(m[i, j, k, l])
        actual1 = matnorm(m, 1, 1)
        actual2 = matnorm(m, 2, 2)
        self.assertTrue(np.allclose(expected1, actual1))
        self.assertTrue(np.allclose(expected2, actual2))
