import unittest as ut

import supercell_core as sc
import os

from ..errors import *
from ..input_parsers import *


class TestInputParsers(ut.TestCase):
    def test_read_POSCAR_1(self):
        example_poscar = """whatever
1.0
1 2 3
0.5 0.7 0.91
3 1 0
1 3
Direct
-21.84 72 -4.72
17.38 -54 3.54
2.66 -8 0.48
0 0 0"""

    def test_read_POSCAR_2(self):
        example_poscar = """
        """

    def test_supercell_in(self):
        pass
