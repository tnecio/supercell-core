import unittest as ut

import supercell_core as sc
import os

from ..errors import *
from ..input_parsers import *


class TestInputParsers(ut.TestCase):
    def test_read_POSCAR_IO_fail(self):
        with self.assertRaises(IOError):
            sc.read_POSCAR("csnadkjndksca")

    def test_parse_POSCAR_1(self):
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

        lay = sc.parse_POSCAR(example_poscar, ["Fe", "Zn"], magmom="2*-1 1 1*0")
        # a hack around np.ndarray's lack of test-friendly __eq__
        self.assertEqual([lay.vectors()[j][i] for j in range(3) for i in range(3)],
                         [1, 2, 3, 0.5, 0.7, 0.91, 3, 1, 0])

        atoms = [
            sc.Atom("Fe", (0, 2, 0), spin=(0, 0, -1)),
            sc.Atom("Zn", (1, 0.5, 3), spin=(0, 0, -1)),
            sc.Atom("Zn", (0.1, 0.2, 0.7), spin=(0, 0, 1)),
            sc.Atom("Zn", (0, 0, 0), spin=(0, 0, 0))
        ]

        for a1, a2 in zip(atoms, lay.atoms()):
            self.assertEqual(a1, a2)

        self.assertEqual(len(lay.atoms()), 4)

    def test_parse_POSCAR_2(self):
        # This time with specie names, selective dynamics and initial velocities
        example_poscar = """whatever
1.0
1 2 3
0.5 0.7 0.91
3 1 0
1 3
Selective dynamics 
d
-21.84 72 -4.72 T T F
17.38 -54 3.54 F T F
2.66 -8 0.48   F F T
0 0 0 F F F

0 0 0 
1 2 3
1.00000 2.000000 3.000000
3.0 1 0.000"""

        lay = sc.parse_POSCAR(example_poscar, ["Fe", "Zn"], magmom="2*-1 1 1*0")
        # a hack around np.ndarray's lack of test-friendly __eq__
        self.assertEqual([lay.vectors()[j][i] for j in range(3) for i in range(3)],
                         [1, 2, 3, 0.5, 0.7, 0.91, 3, 1, 0])

        atoms = [
            sc.Atom("Fe", (0, 2, 0), spin=(0, 0, -1)),
            sc.Atom("Zn", (1, 0.5, 3), spin=(0, 0, -1)),
            sc.Atom("Zn", (0.1, 0.2, 0.7), spin=(0, 0, 1)),
            sc.Atom("Zn", (0, 0, 0), spin=(0, 0, 0))
        ]

        for a1, a2 in zip(atoms, lay.atoms()):
            self.assertEqual(a1, a2)

        self.assertEqual(len(lay.atoms()), 4)

    def test_read_POSCAR_2(self):
        example_poscar = """ Si                                     
   1.00000000000000     
     5.4365330000000000    0.0000000000000000    0.0000000000000000
     0.0000000000000000    5.4365330000000000    0.0000000000000000
     0.0000000000000000    0.0000000000000000    5.4365330000000000
     1
Cartesian
  0.8750000000000000  0.8750000000000000  0.8750000000000000"""

        lay = sc.parse_POSCAR(example_poscar, ["Si"])
        self.assertEqual(lay.atoms()[0], sc.Atom("Si", (0.875, 0.875, 0.875)))
        self.assertEqual(len(lay.atoms()), 1)

    # TODO: test bad POSCARs
    # TODO: test magmoms
    # TODO: test correct IO

    def test_supercell_in(self):
        pass
