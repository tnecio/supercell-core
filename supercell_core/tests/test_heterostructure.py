import unittest as ut

import supercell_core as sc


class TestHeterostructure(ut.TestCase):
    """
    Test of the Heterostructure class
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
        # TODO: add reference to that book
        sub = sc.lattice()
        sub.set_vectors((1, 0), (0, 1))

        lay = sc.lattice()
        lay.set_vectors((1.41, 0), (0, 1.41))

        h = sc.heterostructure()
        h.set_substrate(sub)
        h.add_layer(lay)

        res = h.opt()
        print(res.layers[0].strain_tensor)

    def test_calc(self):
        pass

    def test_plot(self):
        pass

    # TODO: integration test: plot -> PlotResult.get_calc_params -> calc
    # TODO: documentation: use-case: plot min strain (theta) in matplotlib
