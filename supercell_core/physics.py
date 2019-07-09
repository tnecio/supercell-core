"""
This file contains names of various physical qties, objects, units etc.
and their mapping to values used in the calculations
"""

from typing import Union, Sequence
from enum import Enum, auto
import numpy as np

# Type alias for acceptable numeric values
Number = Union[int, float]

# Type alias for spatial vectors that can be input of a public function
# No check for the number of items, because support for this in Python
# typing is bad
VectorLike = Sequence[Number]

# Type aliases for spatial vectors returned from public functions
VectorNumpy = np.ndarray

DEGREE = (2 * np.pi) / 360

PERIODIC_TABLE = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
    "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
    "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl",
    "Mc", "Lv", "Ts", "Og"
)

# constants that might be useful
Z_SPIN_UP = (0, 0, 1)
Z_SPIN_DOWN = (0, 0, -1)

def element_symbol(atomic_no: int) -> str:
    """
    Returns symbol of a chemical element given its atomic number

    Parameters
    ----------
    atomic_no : int
        atomic number of an element

    Returns
    -------
    str
        symbol of that element

    Raises
    ------
    IndexError or TypeError
        If there is no chemical element with such atomic number
    """
    return PERIODIC_TABLE[atomic_no - 1]


def atomic_number(element_symbol: str) -> int:
    """
    Inverse function of `element_symbol`
    Parameters
    ----------
    element_symbol : str
        must be a valid symbol of chemical element, can not be its full name

    Returns
    -------
    int
        atomic number of the element

    Raises
    ------
    ValueError or TypeError
        If there is no chemical element with such symbol
    """
    return PERIODIC_TABLE.index(element_symbol) + 1


class Unit(Enum):
    """
    Enumeration representing different ways in which lattice vectors
    and atomic positions can be specified

    Attributes
    ----------
    Angstrom
        angstrom (1e-10 m)
    Crystal
        representation of vector in the base of elementary cell vectors
        of a given lattice
    """
    Angstrom = auto()
    Crystal = auto()


class Quantity(Enum):
    """
    Enumeration representing different quantities measuring the quality of
    a heterostructure supercell

    Attributes
    ----------
    Strain
        :math:`||\varepsilon - I||_2` where :math:`\varepsilon` is strain tensor
    MaxStrainElement
        :math:`\max_{ij} |\varepsilon - I|_{ij}` where :math:`\varepsilon`
         is strain tensor
    """
    Strain = auto()
    MaxStrainElement = auto()
