import unittest as ut

import supercell_core as sc

from ..errors import *

from ..calc import *

class TestCalc(ut.TestCase):
    def test_flatten_rect_array(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(flatten_rect_array(m),
                         [i for i in range(1, 10)])