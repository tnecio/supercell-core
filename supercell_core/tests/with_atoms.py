import unittest as ut
import numpy as np

class TestCaseWithAtoms(ut.TestCase):
    """
    Special extension to unittest.TestCase to handle comparisons wiht atoms
    If Atom becomes a class in its own right with __eq__ implemented, this
    TestCase subclass might be removed
    """
    def assertAtomsEqual(self, a1, a2):
        if len(a1) == 3:
            el1, pos1, spin1 = a1
        elif len(a1) == 2:
            el1, pos1 = a1
            spin1 = (0, 0, 0)
        else:
            self.assertTrue(False, "bad atom")

        if len(a2) == 3:
            el2, pos2, spin2 = a2
        elif len(a2) == 2:
            el2, pos2 = a2
            spin2 = (0, 0, 0)
        else:
            self.assertTrue(False, "bad atom")

        self.assertEqual(el1, el2)
        self.assertEqual(spin1, spin2)
        self.assertTrue(np.all(pos1 == pos2))

    def testSelf(self):
        a = ("H", (0, 1, 0))
        b = ("H", (0, 1, 0), (0, 0, 0))

        self.assertAtomsEqual(a, a)
        self.assertAtomsEqual(a, b)