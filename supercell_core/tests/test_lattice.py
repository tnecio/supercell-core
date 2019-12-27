import os
import numpy as np
import unittest as ut

from io import StringIO
from unittest.mock import patch

import supercell_core as sc

from ..errors import *


class TestLattice(ut.TestCase):
    """
    Test Lattice object
    """

    def test_vectors_good(self) -> None:
        """
        Tests methods: set_vectors, and vectors
        (cases where it succeeds)
        """
        lay = sc.lattice()

        # Testing default values
        self.assertEqual(lay.vectors(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Setting two 2D elementary cell vectors, omitting z axis as irrelevant
        lay.set_vectors([2, 3], [1, 4])
        self.assertEqual(lay.vectors(), [[2, 3, 0], [1, 4, 0], [0, 0, 1]])

        # Set two 3D elementary cell vectors while omitting the 3rd one is
        # nonsensical, unless you give '0' as z-component of these vectors
        lay.set_vectors([2, 3, 0], [1, 5, 0])
        self.assertEqual(lay.vectors(), [[2, 3, 0], [1, 5, 0], [0, 0, 1]])

        # When z-component is non '0' but the 3rd vector is left default,
        # (which means it's unclear whether 'z' direction is important for
        # the user) a warning should be issued
        with self.assertWarns(UserWarning, msg=WarningText.ZComponentWhenNoZVector):
            lay.set_vectors([2, 3, 0], [1, 5, 0.2])
            self.assertEqual(lay.vectors(), [[2, 3, 0], [1, 5, 0.2], [0, 0, 1]])

        # Set three 3D vectors
        lay.set_vectors([2, 3, 0], [1, 6, 1], [7, 8, 9])
        self.assertEqual(lay.vectors(), [[2, 3, 0], [1, 6, 1], [7, 8, 9]])

        # Change only part of the vector – are you sure z-component is what you
        # think it is?
        lay = sc.lattice()
        lay.set_vectors([1, 0, 1], [0, 1, 1], [0, 0, 1])
        with self.assertWarns(UserWarning, msg=WarningText.ReassigningPartOfVector):
            lay.set_vectors([2, 0], [0, 3])

    def test_vectors_bad(self) -> None:
        """
        Tests methods: set_vectors, and vectors
        (cases where they should fail)
        """
        lay = sc.lattice()

        # Test bad type of values passed to the function
        with self.assertRaises(TypeError):
            # Too few arguments – less than two
            lay.set_vectors([1, 2])

            # Too many arguments – more than three
            lay.set_vectors([1, 2, 0], [3, 4, 0], [5, 6, 7], [8, 9, 0])

            # Too short vectors
            lay.set_vectors([1], [5])

            # Too long vectors
            lay.set_vectors([1, 2, 3, 4], [5, 6, 7, 8])

            # Mismatched vectors length
            lay.set_vectors([1, 2], [1, 2, 3])

            # Wrong type
            lay.set_vectors([1, 2, 3], ["5", "6", "7"])

            # Any arguments passed to vectors
            lay.vectors("whatever")

        # Test incorrect data passed to function
        with self.assertRaises(LinearDependenceError):
            # Two linearly dependent vectors
            lay.set_vectors([1, 2], [2, 4])

            # One vector is linear combination of the second and the default
            # third vector ([0, 0, 1])
            lay.set_vectors([0, 1, 1], [0, 1, 0])

    def test_add_atoms_good(self) -> None:
        """
        Tests methods: add_atom, add_atoms, and atoms
        """
        lay = sc.lattice()

        # Test default (no atoms in an elementary cell)
        self.assertEqual(lay.atoms(), [])

        # Test add_atom Helium (default unit: angstrom)
        he = sc.Atom("He", (0.1, 0.2, 0.3))
        lay.add_atom(he.element, he.pos)
        self.assertEqual(lay.atoms()[0], he)

        # Test add_atoms: Hydrogen and Lithium
        # Retaining element order is expected
        h, li = sc.Atom("H", np.array([0, 0, 0])), \
                sc.Atom("Li", np.array([0.9, 0.9, 0.9]), spin=(0, 1, 2))
        lay.add_atoms([h, li])
        for a1, a2 in zip(lay.atoms(), [he, h, li]):
            self.assertEqual(a1, a2)

        # Add atom using 2D position vector
        be = sc.Atom("Be", (0.5, 0.5), spin=(0, 0, 1))
        lay.add_atoms([be])
        self.assertEqual(lay.atoms()[-1],
                         sc.Atom("Be", np.array([0.5, 0.5, 0]), spin=(0, 0, 1)))

        # When atom is outside the elementary cell, a warning should be logged
        with self.assertWarns(UserWarning, msg=WarningText.AtomOutsideElementaryCell):
            lay.add_atom("C", (2, 0, 0))

        # Element symbol not in the periodic table
        with self.assertWarns(UserWarning, msg=WarningText.UnknownChemicalElement):
            lay.add_atom("Helium", (1, 0, 0))  # should be: "He"

        # Add atom using crystal units
        lay = sc.lattice()
        lay.set_vectors([2, 0, 0], [2, 2, 0], [0, 0, 3])
        lay.add_atom("Na", (0.5, 0.5, 0.5), unit=sc.Unit.Crystal)
        self.assertEqual(lay.atoms()[0], sc.Atom("Na", (2, 1, 1.5)))

        # Change lattice vectors after adding atoms
        lay.set_vectors([4, 0, 0], [4, 4, 0], [0, 0, 6],
                        atoms_behaviour=sc.Unit.Crystal)
        self.assertEqual(lay.atoms()[0], sc.Atom("Na", (4, 2, 3)))

        lay.set_vectors([8, 0, 0], [8, 8, 0], [0, 0, 12],
                        atoms_behaviour=sc.Unit.Angstrom)
        self.assertEqual(lay.atoms()[0], sc.Atom("Na", (4, 2, 3)))

        # List atoms using CRYSTAL units
        lay = sc.lattice()
        lay.set_vectors([2, 0, 0], [2, 2, 0], [0, 0, 3])
        lay.add_atom("Na", (0.5, 0.5, 0.5), unit=sc.Unit.Crystal)
        self.assertEqual(lay.atoms()[0], sc.Atom("Na", (2, 1, 1.5)))
        self.assertEqual(lay.atoms(unit=sc.Unit.Crystal)[0],
                         sc.Atom("Na", (0.5, 0.5, 0.5),
                                 pos_unit=sc.Unit.Crystal))

    def test_add_atoms_bad(self) -> None:
        """
        Tests methods: add_atom, and add_atoms, where they should fail
        """
        lay = sc.lattice()

        # Adding atoms before changing elementary cell vectors:
        # (we don't know what to do with the atomic positions,
        # so we refuse the temptation to guess and raise an error)
        # Note: set_vectors should have a behaviour flag allowing
        # specifying what to do with the atomic positions
        lay.add_atom("Na", (0.5, 0.5, 0.5), unit=sc.Unit.Crystal)
        with self.assertRaises(UndefinedBehaviourError):
            lay.set_vectors([4, 0, 0], [4, 4, 0], [0, 0, 6])

        # Test bad type of values passed to function
        with self.assertRaises(TypeError):
            # Bad length of atomic position vector
            lay.add_atom("O", 0)
            lay.add_atom("O", (1, 2, 3, 4))

            # Bad number of arguments to add atom
            lay.add_atom("N")
            lay.add_atom("N", 0, 0)

            # Non-atom passed to add_atoms
            lay.add_atoms(["whatever"])

            # Non-unit passed as unit
            lay.add_atom("N", (0, 0), unit="meter")

    def test_save_POSCAR(self):
        os.system("mkdir -p tmp")
        fn = "tmp/test_POSCAR"
        lay = sc.lattice()
        lay.set_vectors([1, 2, 3], [0.5, 0.7, 0.91], [3, 1, 0])
        lay.add_atoms([
            sc.Atom("Fe", (0, 2)),
            sc.Atom("Zn", (1, 0.5, 3), spin=(0, 1, -1)),
            sc.Atom("Zn", (0.1, 0.2, 0.7)),
            sc.Atom("Zn", (0, 0, 0))
        ])
        lay.add_atoms([
            sc.Atom("Zn", (0.1, 0, 0), pos_unit=sc.Unit.Crystal, spin=(0, 0, 1)),
            sc.Atom("Zn", (0.2, 0, 0), pos_unit=sc.Unit.Crystal, spin=(0, 0, -1)),
            sc.Atom("Zn", (0.3, 0, 0), pos_unit=sc.Unit.Crystal, spin=(0, 0, 1))
        ])

        # first goes Fe (-21.84...)
        # then Zn, starting with z-spin=1 and in order of adding,
        # so: z-spin=1: (0.1, 0, 0), (0.3, 0, 0) etc.
        expected_poscar = """supercell_generated_POSCAR
1.0
1 2 3
0.5 0.7 0.91
3 1 0
1 6
Direct
-21.84 72 -4.72
0.1 0 0
0.3 0 0
2.66 -8 0.48
0 0 0
17.38 -54 3.54
0.2 0 0
"""

        lay.save_POSCAR(filename=fn)
        with open(fn, 'r') as f:
            poscar = f.read()
        os.system("rm -f " + fn)

        self.assertEqual(expected_poscar, poscar)

        # test stdout
        names = ["Fe", "Zn"]
        # https://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python
        with patch('sys.stdout', new=StringIO()) as fakeOutput:
            lay.save_POSCAR()
            self.assertEqual(fakeOutput.getvalue(), expected_poscar + "\n" + \
                             "Note: The order of the atomic species in this generated POSCAR " + \
                             "file is as follows:\n" + " ".join(names) + "\n" + \
                             "MAGMOM flag: 0 2*1 2*0 2*-1\n")
            # TODO: test sorting by z-spin in atomic species

    def test_save_xsf(self):
        os.system("mkdir -p tmp")
        fn = "tmp/test.xsf"
        lay = sc.lattice()
        lay.set_vectors([1, 2, 3], [0.5, 0.7, 0.91], [3, 1, 0])
        lay.add_atoms([
            sc.Atom("Fe", (0, 2)),
            sc.Atom("Zn", (1, 0.5, 3), spin=(0, 1, -1)),
            sc.Atom("Zn", (0.1, 0.2, 0.7)),
            sc.Atom("Zn", (0, 0, 0))
        ])

        expected_xsf = """CRYSTAL

PRIMVEC
1 2 3
0.5 0.7 0.91
3 1 0

PRIMCOORD
4 1
26 0 2 0 0 0 0
30 1 0.5 3 0 1 -1
30 0.1 0.2 0.7 0 0 0
30 0 0 0 0 0 0
"""

        lay.save_xsf(filename=fn)
        with open(fn, 'r') as f:
            xsf = f.read()
        os.system("rm -f " + fn)

        self.assertEqual(expected_xsf, xsf)

        # test stdout
        # https://stackoverflow.com/questions/4219717/how-to-assert-output-with-nosetest-unittest-in-python
        with patch('sys.stdout', new=StringIO()) as fakeOutput:
            lay.save_xsf()
            self.assertEqual(fakeOutput.getvalue(), expected_xsf)

    def load_graphene(self):
        return sc.read_POSCAR(
            os.path.join(os.path.dirname(__file__),
                         "../resources/vasp/graphene/POSCAR"), ["C"])

    def test_translate(self):
        l = sc.lattice().set_vectors([2, 0], [0, 2])
        l.add_atom("H", (0.0, 1.0))
        l.add_atom("He", (1.0, 3.0))

        l.translate_atoms((-1, -2))
        self.assertEqual(l.atoms()[0], sc.Atom("H", (-1, -1)))
        self.assertEqual(l.atoms()[1], sc.Atom("He", (0, 1)))

        l.translate_atoms((0.5, 0.5), unit=sc.Unit.Crystal)
        self.assertEqual(l.atoms()[0], sc.Atom("H", (0, 0)))
        self.assertEqual(l.atoms()[1], sc.Atom("He", (1, 2)))

    def test_draw(self):
        graphene = self.load_graphene()

        nips3 = sc.read_POSCAR(
            os.path.join(os.path.dirname(__file__),
                         "../resources/vasp/NiPS3/POSCAR"), ["Ni", "P", "S"],
            magmom=""
        )

        fig, ax = graphene.draw()
        fig.show()

        fig, ax = nips3.draw()
        fig.show()

        # Two at the same time
        fig, ax = graphene.draw()
        fig, ax = nips3.draw(ax)
        fig.show()

