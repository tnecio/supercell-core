import unittest as ut

import supercell_core as sc

from ..errors import *
from ..physics import *

class TestPhysics(ut.TestCase):
    def test_element_symbol(self):
        self.assertEqual(sc.element_symbol(1), "H")
        self.assertEqual(sc.element_symbol(8), "O")
        with self.assertRaises(IndexError):
            sc.element_symbol(0)
            sc.element_symbol(200)
        with self.assertRaises(TypeError):
            sc.element_symbol("Helium")

    def test_atomic_number(self):
        self.assertEqual(atomic_number("Fe"), 26)
        with self.assertRaises(ValueError):
            atomic_number("Zinc")
            atomic_number("E")
            atomic_number("1")