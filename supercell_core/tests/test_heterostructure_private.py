"""
This file contains unit tests for Heterostructure private methods, since
they can get pretty complex in some places
"""

import unittest as ut
from os import path

import supercell_core as sc

from ..heterostructure import *


class TestHeterostructurePrivate(ut.TestCase):
    def test_opt_aux(self):
        # test 1: graphene-graphene magic angle 21.8, theta range 16-24-0.1
        max_el = 4
        theta_ranges = [(16 * DEGREE, 24 * DEGREE, 0.1 * DEGREE)]

        graphene = sc.read_POSCAR(
            path.join(path.dirname(__file__), "../resources/vasp/graphene/POSCAR"),
            ["C"])
        h = heterostructure()
        h.set_substrate(graphene)
        h.add_layer(graphene, theta=(0, 15 * DEGREE, 1 * DEGREE))

        actual_res = h._Heterostructure__opt_aux(
            (1, 1),
            max_el, theta_ranges
        )

        expected_res = ([21.8 * DEGREE],
                        np.array([[2, 3], [-3, -1]]))

        for theta1, theta2 in zip(actual_res[0], expected_res[0]):
            self.assertAlmostEqual(theta1, theta2)
        actual_max_strain = h.calc(actual_res[1], expected_res[0]).max_strain()
        expected_max_strain = h.calc(expected_res[1], expected_res[0]).max_strain()
        self.assertTrue(np.allclose(actual_max_strain, expected_max_strain))

    def test_opt_aux_2(self):
        # test 2: graphene-graphene magic angle 21.8, theta range 20.8-22.8-1
        max_el = 4
        theta_ranges = [(20.8 * DEGREE, 22.8 * DEGREE, 1 * DEGREE)]

        graphene = sc.read_POSCAR(
            path.join(path.dirname(__file__), "../resources/vasp/graphene/POSCAR"),
            ["C"])
        h = heterostructure()
        h.set_substrate(graphene)
        h.add_layer(graphene, theta=(0, 15 * DEGREE, 1 * DEGREE))

        actual_res = h._Heterostructure__opt_aux(
            (1, 1),
            max_el, theta_ranges
        )

        expected_res = ([21.8 * DEGREE],
                        np.array([[2, 3], [-3, -1]]))

        for theta1, theta2 in zip(actual_res[0], expected_res[0]):
            self.assertAlmostEqual(theta1, theta2)
        actual_max_strain = h.calc(actual_res[1], expected_res[0]).max_strain()
        expected_max_strain = h.calc(expected_res[1], expected_res[0]).max_strain()
        self.assertTrue(np.allclose(actual_max_strain, expected_max_strain))

    def test_opt_aux_3(self):
        # 3. # TODO: reference that book
        max_el = 2

        h = heterostructure()
        h.set_substrate(sc.lattice().set_vectors([1, 0], [0, 1]))
        h.add_layer(sc.lattice().set_vectors([1.41, 0], [0, 1.41]))

        actual_res = h._Heterostructure__opt_aux(
            (1, 1),
            max_el, [(0, 90 * DEGREE, 45 * DEGREE)]
        )

        expected_res = ([45 * DEGREE],
                        [[1, 1], [-1, 1]])

        for theta1, theta2 in zip(actual_res[0], expected_res[0]):
            self.assertAlmostEqual(theta1, theta2)
        actual_max_strain = h.calc(actual_res[1], [45 * DEGREE]).max_strain()
        expected_max_strain = h.calc(expected_res[1], [45 * DEGREE]).max_strain()
        self.assertTrue(np.allclose(actual_max_strain, expected_max_strain))

    def test_opt_aux_4(self):
        # test 4: graphene-NiPS3 low-strain angle 21.9, theta range 16-30-0.1
        # Note: Takes a while
        max_el = 12
        theta_ranges = [(16 * DEGREE, 30 * DEGREE, 0.1 * DEGREE)]

        graphene = sc.read_POSCAR(
            path.join(path.dirname(__file__), "../resources/vasp/graphene/POSCAR"),
            ["C"])
        nips3 = sc.read_POSCAR(
            path.join(path.dirname(__file__), "../resources/vasp/NiPS3/POSCAR"),
            ["Ni", "P", "S"]
        )
        h = heterostructure()
        h.set_substrate(graphene)
        h.add_layer(nips3, theta=theta_ranges[0])

        actual_res = h._Heterostructure__opt_aux(
            (1, 1),
            max_el, theta_ranges
        )

        expected_res = ([21.9 * DEGREE],
                        np.array([[7, 9], [10, -8]]))

        for theta1, theta2 in zip(actual_res[0], expected_res[0]):
            self.assertAlmostEqual(theta1, theta2)
        actual_max_strain = h.calc(actual_res[1], expected_res[0]).max_strain()
        expected_max_strain = h.calc(expected_res[1], expected_res[0]).max_strain()
        self.assertTrue(np.allclose(actual_max_strain, expected_max_strain))

    def test_get_strain_tensor_opt(self):
        # TODO: redo this test without prepare_opt
        h = sc.heterostructure()
        sub = sc.lattice()
        sub.set_vectors(*np.identity(2))  # A = X
        h.set_substrate(sub)

        # 1. Stretch
        theta = 0
        stretch_lay = sc.lattice()
        stretch_lay.set_vectors([1.02, 0.01], [0.05, 0.99])
        h.add_layer(stretch_lay)
        XA, XB_lay = h._Heterostructure__get_lattice_matrices()
        XB = XB_lay[0]
        dt_As = h._Heterostructure__get_dt_As(1)
        res = h._Heterostructure__get_strain_tensor_opt(
            theta, XA, XB, dt_As)[1, 0, 0, 1]
        # expected_res = BtB - 1 = BtXt @ XtX @ XB - 1 = BX @ XtX # XB - 1
        XtX = np.array([[1.02, 0.05], [0.01, 0.99]])
        expected_res = inv(XB) @ XtX @ XB - np.identity(2)

        self.assertTrue(np.allclose(res, expected_res))
        h.remove_layer(0)

        # 2. Rotation
        theta = -60 * DEGREE
        rot_layer = sc.lattice()
        rot_layer.set_vectors([0.5, np.sqrt(3) * 0.5], [np.sqrt(3) * 0.5, -0.5])
        h.add_layer(rot_layer)
        XA, XB_lay = h._Heterostructure__get_lattice_matrices()
        XB = XB_lay[0]
        dt_As = h._Heterostructure__get_dt_As(1)
        res = h._Heterostructure__get_strain_tensor_opt(
            theta, XA, XB, dt_As)[1, 0, 0, 1]
        # expected_res = id - 1
        expected_res = np.zeros((2, 2))

        self.assertTrue(np.allclose(res, expected_res))

    def test_get_dt_As(self):
        dt_As = Heterostructure._Heterostructure__get_dt_As(2)

        for i in range(-2, 3):
            for j in range(-2, 3):
                self.assertEqual(dt_As[i, j][0], i)
                self.assertEqual(dt_As[i, j][1], j)

    def test_update_opt_res(self):
        old_res = ([21.4], 0.12, np.array([[2, 2], [0, 2]]), 0)

        worse_qty_res = ([32.1], 0.24, np.array([[2, 0], [0, 2]]), "whatver")
        worse_size_res = ([12.9], 0.12, np.array([[4, 0], [0, 4]]), 1)
        worse_shape_res = ([7.9], 0.12, np.array([[4, 0], [4, 1]]), -1)

        better_qty_res = ([32.1], 0.10, np.array([[2, 0], [0, 2]]), "z")
        better_size_res = ([12.9], 0.12, np.array([[3, 0], [0, 1]]), 1.2)
        better_shape_res = ([12.9], 0.12, np.array([[2, 0], [0, 2]]), [])

        worse_results = [worse_qty_res, worse_size_res, worse_shape_res]
        better_results = [better_qty_res, better_size_res, better_shape_res]

        for worse_res in worse_results:
            res = Heterostructure._Heterostructure__update_opt_res(
                worse_res, *old_res)
            self.assertEqual(res[0], old_res[0])
            self.assertEqual(res[1], old_res[1])
            self.assertTrue(np.allclose(res[2], old_res[2]))
            self.assertEqual(res[3], old_res[3])

        for better_res in better_results:
            res = Heterostructure._Heterostructure__update_opt_res(
                better_res, *old_res)
            self.assertEqual(res[0], better_res[0])
            self.assertEqual(res[1], better_res[1])
            self.assertTrue(np.allclose(res[2], better_res[2]))
            self.assertEqual(res[3], better_res[3])

    def test_get_XXt(self):
        # TODO: a deterministic test on values calculated by hand
        dt_xs = np.random.random((2, 2, 2)) * 10 + 1
        # assuming Br = X
        d_xs = np.round(dt_xs)
        XD = d_xs[0].T
        XDt = dt_xs[0].T
        DX = np.linalg.inv(XD)
        expected = XDt @ DX
        actuals = Heterostructure._Heterostructure__get_XXt(d_xs, dt_xs)
        self.assertTrue(np.allclose(expected, actuals[0, 1, 0, 0]))

    def test_get_strain_tensor_wiki(self):
        span = 5
        XXt = [[[[[[np.random.random()
                    for col in range(2)]
                   for row in range(2)]
                  for y2 in range(span)]
                 for x2 in range(span)]
                for y1 in range(span)]
               for x1 in range(span)]

        actual = Heterostructure._Heterostructure__get_strain_tensor_wiki(XXt)

        expected = [[[[[[(XXt[x1][y1][x2][y2][row][col]
                          + XXt[x1][y1][x2][y2][col][row]) / 2
                         - (1 if row == col else 0)
                         for col in range(2)]
                        for row in range(2)]
                       for y2 in range(span)]
                      for x2 in range(span)]
                     for y1 in range(span)]
                    for x1 in range(span)]

        self.assertTrue(np.allclose(actual, expected))

    def test_get_lattice_matrixes(self):
        h = sc.heterostructure()
        sub = sc.lattice()
        sub.set_vectors(*np.identity(2))  # A = X
        h.set_substrate(sub)

        stretch_lay = sc.lattice()
        stretch_lay.set_vectors([1.02, 0.01], [0.05, 7.99])
        h.add_layer(stretch_lay)

        rot_layer = sc.lattice()
        rot_layer.set_vectors([0.5, np.sqrt(3) * 0.5], [np.sqrt(3) * 0.5, -0.5])
        h.add_layer(rot_layer)

        actual_XA, actual_XB_lay = h._Heterostructure__get_lattice_matrices()
        expected_XA = np.identity(2)
        expected_XB_lay = [
            np.array([[1.02, 0.05], [0.01, 7.99]]),
            np.array([[0.5, np.sqrt(3) * 0.5], [np.sqrt(3) * 0.5, -0.5]])
        ]
        for m1, m2 in zip(actual_XB_lay + [actual_XA],
                          expected_XB_lay + [expected_XA]):
            self.assertTrue(np.allclose(m1, m2))

    def test_get_d_xs(self):
        AX = np.identity(2)
        XB = np.sqrt(2 * np.identity(2))  # ((1.4, 0), (0, 1.4))
        max_el = 3
        span_range = range(-max_el, max_el + 1)
        dt_As = [[(x, y) for x in span_range] for y in span_range]
        theta = 45 * DEGREE

        actual_d_xs = Heterostructure._Heterostructure__get_d_xs(
            AX, XB, np.array(dt_As), theta
        )

        XBr = rotate(XB, theta)  # [[1, -1], [1, 1]]
        # don't write [[1, -1], [1, 1]] out explicitly because
        # it will cause different floating-point error to propagate
        self.assertTrue(np.allclose(np.array([[1, -1], [1, 1]]), XBr))

        BrA = np.linalg.inv(AX @ XBr)
        dt_Brs = [[BrA @ np.array(v_xy)
                   for v_xy in v_x]
                  for v_x in dt_As]
        actual_dt_Brs = np.einsum('ij,xyj->xyi', BrA, np.array(dt_As))

        self.assertTrue(np.allclose(np.array(dt_Brs), actual_dt_Brs))

        dt_Btrs = [[np.round(v_xy)
                    for v_xy in v_x]
                   for v_x in dt_Brs]
        actual_dt_Btrs = np.round(actual_dt_Brs)

        self.assertTrue(np.allclose(np.array(dt_Btrs), actual_dt_Btrs))

        expected_d_xs = np.array([
            [XBr @ v_xy
             for v_xy in v_x]
            for v_x in dt_Btrs
        ])

        self.assertTrue(np.allclose(actual_d_xs, expected_d_xs))
