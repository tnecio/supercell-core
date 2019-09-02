from typing import Tuple, Optional

import numpy as np

from .physics import VectorLike, Unit, Matrix2x2


class Atom:
    """
    Class representing an atom in a crystal lattice.
    This is a container class, and doesn't implement any methods except for
    basis change.
    Access its properties directly.

    Properties
    ----------
    element : string
        Symbol of the chemical element
    pos : (float, float, float)
        Position in a cell
    pos_unit : Unit
        Unit of `pos` vector (Crystal or Angstrom)
    velocity : (float, float, float), optional
        Velocity in angstrom/fs. When read from / written to VASP POSCAR file
        this is the atoms' intitial velocity
    spin: (int, int, int)
        Magnetic moment of the atom. Usually you are interested in the
        z-component, i.e. spins are of the form (0, 0, int).
        Default: (0, 0, 0)
    selective_dynamics: (bool, bool, bool)
        Specifies whether  the respective coordinates of the atom were / will be
        allowed to change during the ionic relaxation in VASP calculations
    """
    element: str
    pos: VectorLike
    pos_unit: Unit
    velocity: Optional[VectorLike]
    spin: VectorLike
    selective_dynamics: Tuple[bool, bool, bool]

    def __init__(self,
                 element: str,
                 pos: VectorLike,
                 pos_unit: Unit = Unit.Angstrom,
                 velocity: VectorLike = None,
                 spin: VectorLike = (0, 0, 0),
                 selective_dynamics=(True, True, True)):
        """
        Constructor of the Atom class.

        Parameters
        ----------
        element : string
            Symbol of the chemical element
        pos : (float, float, float)
            Position in a cell
        pos_unit : Unit, optional
            Unit of `pos` vector (Crystal or Angstrom)
            Default: Unit.Angstrom
        velocity : (float, float, float), optional
            Velocity in angstrom/fs. When read from / written to VASP POSCAR
            file this is the atoms' intitial velocity
            Default: None
        spin: (int, int, int), optional
            Magnetic moment of the atom. Usually you are interested in the
            z-component, i.e. spins are of the form (0, 0, int).
            Default: (0, 0, 0)
        selective_dynamics: (bool, bool, bool)
            Specifies whether  the respective coordinates of the atom were /
            will be allowed to change during the ionic relaxation in VASP
            calculations
            Default: (True, True, True)
        """
        self.element = element
        self.pos = pos if len(pos) == 3 else (pos[0], pos[1], 0)
        self.pos_unit = pos_unit
        self.velocity = velocity
        self.spin = spin
        self.selective_dynamics = selective_dynamics

    def __eq__(self, other: "Atom") -> bool:
        return self.element == other.element \
               and np.allclose(self.pos, other.pos) \
               and self.pos_unit == other.pos_unit \
               and (self.velocity == other.velocity
                    or np.allclose(self.velocity, other.velocity)) \
               and np.allclose(self.spin, other.spin) \
               and np.allclose(self.selective_dynamics, other.selective_dynamics)

    def __str__(self):
        sd = " ".join(["T" if x else "F" for x in self.selective_dynamics])

        return self.element + " at: " + str(self.pos) + " (" \
    + ("A" if self.pos_unit == Unit.Angstrom else "crystal coord.") + ")" \
    + (f" (velocity: {self.velocity})" if self.velocity is not None else "") \
    + (f" (spin: {self.spin})" if np.any([x != 0 for x in self.spin]) else "") \
    + (" " + sd if sd != "T T T" else "")

    def __repr__(self):
        return str(self)

    def basis_change(self,
                     M: Matrix2x2,
                     new_unit: Unit) -> "Atom":
        """
        Creates new atom with values in different basis.
        Note: currently, this has no effect on the value of spin parameter!

        Parameters
        ----------
        M : Matrix2x2
            basis change matrix
        new_unit : Unit
            unit of `pos` of the new atom

        Returns
        -------
        Atom
        """
        new_pos = M @ np.array(self.pos)

        if self.velocity is not None:
            new_velocity = M @ np.array(self.velocity)
        else:
            new_velocity = None

        return Atom(self.element,
                    new_pos,
                    new_unit,
                    velocity=new_velocity,
                    spin=self.spin
                    )
