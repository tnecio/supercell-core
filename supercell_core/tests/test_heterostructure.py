import unittest as ut
from os import path

import numpy as np

import supercell_core as sc


class TestHeterostructure(ut.TestCase):
    """
    Test of the Heterostructure and Result classes
    """

    def test_substrate(self):
        # Tests `substrate` and `set_substrate` methods
        h = sc.heterostructure()
        sub = sc.lattice()
        h.set_substrate(sub)
        self.assertEqual(h.substrate(), sub)

        h = sc.heterostructure()
        with self.assertRaises(AttributeError):
            h.substrate()

    def test_layers(self):
        # Tests `layers`, `add_layer`, `add_layers`, `remove_layer`,
        # and `get_layer` methods
        h = sc.heterostructure()
        lay = sc.lattice()
        lay.add_atom("H", (0, 0, 0))
        h.add_layer(lay, theta=1 * sc.DEGREE)

        lay2 = sc.lattice()
        lay2.add_atom("He", (0.1, 0.1, 0.1))
        h.add_layer(lay2)

        lay3 = sc.lattice()
        lay3.add_atom("Li", (0.2, 0.2, 0.2))
        h.add_layer(lay3, theta=(0, 45 * sc.DEGREE, 0.2 * sc.DEGREE))

        lay4 = sc.lattice()
        lay4.add_atom("Be", (0.3, 0.3, 0.3))
        h.add_layer(lay4, pos=1)

        self.assertEqual(
            [lay, lay4, lay2, lay3],
            h.layers()
        )

        h.remove_layer(1)

        self.assertEqual(
            [lay, lay2, lay3],
            h.layers()
        )

        h = sc.heterostructure()
        h.add_layers([
            (lay, 1 * sc.DEGREE),
            lay2,
            (lay3, (0, 45 * sc.DEGREE, 0.2 * sc.DEGREE)),
            lay4
        ])

        self.assertEqual(
            [lay, lay2, lay3, lay4],
            h.layers()
        )

        got_lay = h.get_layer(0)

        self.assertEqual(lay, got_lay[0])
        self.assertEqual((1 * sc.DEGREE, 1 * sc.DEGREE, 1.0), got_lay[1])

        with self.assertRaises(IndexError):
            h.get_layer(10)

    def test_opt(self, algorithm="fast"):
        # graphene-NiPS3 low-strain angle 21.9, theta range 16-30-0.1
        graphene = sc.read_POSCAR(
            path.join(path.dirname(__file__), "../resources/vasp/graphene/POSCAR"),
            atomic_species=["C"])
        nips3 = sc.read_POSCAR(
            path.join(path.dirname(__file__), "../resources/vasp/NiPS3/POSCAR"),
            atomic_species=["Ni", "P", "S"]
        )

        h = sc.heterostructure()
        h.set_substrate(graphene)
        h.add_layer(nips3)

        res = h.opt(max_el=11,
                    thetas=np.arange(16 * sc.DEGREE, 30 * sc.DEGREE, 0.1 * sc.DEGREE),
                    algorithm=algorithm)

        self.assertTrue(np.allclose(res.M(), np.array([[7, 9], [10, -8]]))
                        or np.allclose(res.M(), np.array([[9, 7], [-8, 10]])))
        self.assertTrue(np.allclose(res.layer_Ms()[0], np.array([[6, -1], [-1, -2]]))
                        or np.allclose(res.layer_Ms()[0], np.array([[-1, 6], [-2, -1]])))
        self.assertAlmostEqual(res.thetas()[0], 21.9 * sc.DEGREE)
        self.assertAlmostEqual(res.max_strain(), 0.000608879275296, places=5)
        self.assertEqual(res.atom_count(), 552)

    def test_opt_direct(self):
        return self.test_opt(algorithm="direct")

    def test_opt_graphene_fast(self):
        graphene = sc.read_POSCAR(
            path.join(path.dirname(__file__), "../resources/vasp/graphene/POSCAR"),
            atomic_species=["C"])

        h = sc.heterostructure()
        h.set_substrate(graphene)
        h.add_layer(graphene)

        res = h.opt(max_el=20, thetas=np.arange(5.5 * sc.DEGREE, 7 * sc.DEGREE, 0.001 * sc.DEGREE))
        self.assertAlmostEqual(res.max_strain(), 0, places=6)
        self.assertEqual(res.atom_count(), 364)

    def test_homogeneous_trilayer(self):
        graphene = sc.read_POSCAR(
            path.join(path.dirname(__file__), "../resources/vasp/graphene/POSCAR"),
            atomic_species=["C"])

        h = sc.heterostructure().set_substrate(graphene).add_layer(graphene).add_layer(graphene)
        res = h.opt(max_el=4, thetas=[np.arange(0, 10*sc.DEGREE, 1 * sc.DEGREE)]*2)

        self.assertEqual(res.atom_count(), 3 * len(graphene.atoms()))

    def test_bad_arg_thetas(self):
        graphene = sc.read_POSCAR(
            path.join(path.dirname(__file__), "../resources/vasp/graphene/POSCAR"),
            atomic_species=["C"])

        h = sc.heterostructure().set_substrate(graphene).add_layer(graphene).add_layer(graphene)
        with self.assertRaises(TypeError):
            h.opt(max_el=4, thetas=[np.arange(0, 10 * sc.DEGREE, 1 * sc.DEGREE)])
