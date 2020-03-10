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

    def test_opt(self):
        # graphene-NiPS3 low-strain angle 21.9, theta range 16-30-0.1
        # Note: Takes a while
        graphene = sc.read_POSCAR(
            path.join(path.dirname(__file__), "../resources/vasp/graphene/POSCAR"),
            ["C"])
        nips3 = sc.read_POSCAR(
            path.join(path.dirname(__file__), "../resources/vasp/NiPS3/POSCAR"),
            ["Ni", "P", "S"]
        )

        sub = graphene
        lay = nips3

        h = sc.heterostructure()
        h.set_substrate(sub)
        h.add_layer(lay, theta=(16 * sc.DEGREE, 30 * sc.DEGREE, 0.1 * sc.DEGREE))

        res = h.opt(max_el=11)

        self.assertTrue(np.allclose(res.M(), np.array([[7, 9], [10, -8]])))
        self.assertTrue(np.allclose(res.layer_Ms()[0], np.array([[6, -1], [-1, -2]])))
        self.assertAlmostEqual(res.thetas()[0], 21.9 * sc.DEGREE)
        self.assertAlmostEqual(res.max_strain(), 0.000608879275296, places=5)
        self.assertEqual(res.atom_count(), 552)

    def test_opt2(self):
        # Defining unit cell of graphene
        graphene = sc.lattice()
        graphene.set_vectors([2.13, -1.23], [2.13, 1.23])
        # "C" (carbon) atoms in the unit cell in either
        # angstrom or direct coordinates
        graphene.add_atom("C", (0, 0, 0)) \
            .add_atom("C", (2 / 3, 2 / 3, 0),
                      unit=sc.Unit.Crystal)

        # Combining graphene layers
        h = sc.heterostructure().set_substrate(graphene) \
            .add_layer(graphene)
        for theta in np.arange(0, 1 * sc.DEGREE, 0.25 * sc.DEGREE):
            h.opt(max_el=8, thetas=[theta])


    def test_calc(self):
        # `calc` is called by `opt` so for now we can do without a separate test
        pass

