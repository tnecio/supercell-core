from typing import List, Optional, Tuple, Union

from .errors import *
from .calc import *
from .physics import Unit, VectorLike, VectorNumpy, PERIODIC_TABLE, atomic_number

# Type alias for specifying atom element symbol (should be one of the strings
# from PERIODIC_TABLE), position in a lattice, and optionally its spin
# in x, y, and z directions
# (maybe move to its own class?)
Atom = Union[Tuple[str, VectorLike], Tuple[str, VectorLike, VectorLike]]


# noinspection PyPep8Naming
class Lattice:
    """
    Object representing a 2D crystal lattice (or rather its unit cell)

    Initially set with default values: no atoms, unit cell vectors
    equal to unit vectors (of length 1 angstrom each) along x, y, z axes.

    Elementary cell vectors are here and elsewhere referred to as b_1, b_2, b_3.
    """
    __atoms: List[Atom]
    __XA: np.ndarray

    def __init__(self):
        # set initial values of attributes

        # store unit cell vectors as a numpy array, unit: angstrom
        # vectors in columns; note: this means that this array is also
        # a base change matrix from the lattice vectors basis
        # to the x-y-z basis
        # default value: identity matrix
        self.__XA = np.identity(3)

        # store atoms in one unit cell as a list of values of type Atom
        # since what we will be doing with them most is adding and iterating
        # (when we build a heterostructure unit cell)
        # position unit: Crystal
        self.__atoms = []

    def set_vectors(self, *args: VectorLike,
                    atoms_behaviour: Optional[Unit] = None) -> None:
        """
        Sets (or changes) lattice vectors to given values

        Parameters
        ----------
        args : 2 2D vectors, or 3 3D vectors
            Two or three vectors describing unit cell. Since this program
            is used to calculate values regarding 2D lattices, it is not
            necessary to specify z-component of the vectors. In that case,
            the z-component defaults to 0 for the first two vectors
            (unless it was set by previous invocation of `set_vectors`;
            this issues a warning but works).
            All vectors must be the same length.
            Unit : angstrom (1e-10 m)
        atoms_behaviour : Unit, optional
            Described how atomic positions of the atoms already added to
            the lattice should behave. They are treated as if they were
            specified with the unit `atomic_behaviour`. This argument is
            optional if the lattice does not contain any atoms, otherwise
            necessary.

        Returns
        -------
        None

        Raises
        -----
        TypeError
            If supplied values don't match expected types
        LinearDependenceError
            If supplied vectors are not linearly independent
        UndefinedBehaviourError
            If trying to change unit cell vectors of a lattice that
            contains atoms, without specifying what to do with the atomic
            positions in the unit cell

        Warns
        -----
        UserWarning
            When unit cell vectors are reset in a way that disregards
            previous change of the values of z-component or the third vector
            (suggesting that user might not be aware that their values are
            not default)
        """
        # First, check if we should warn user about something, and if the data
        # are correct
        if len(args) < 2 or len(args) > 3:
            raise TypeError("Expected 2 or 3 vectors")
        if len(set(map(len, args))) != 1:
            # len(set) tells us about # of unique elements;
            # in this case set contains lengths of the supplied vectors
            raise TypeError("Different lengths of supplied vectors")
        for i, v in enumerate(args):
            if len(v) < 2 or len(v) > 3:
                raise TypeError("Expected 2D or 3D vectors")
            if len(v) == 2 and self.__XA[2, i] != 0:
                warnings.warn(Warning.ReassigningPartOfVector.value)
            if len(v) == 3 and v[2] != 0 and len(args) == 2:
                warnings.warn(Warning.ZComponentWhenNoZVector.value)
        if (not isinstance(atoms_behaviour, Unit)) and len(self.__atoms) > 0:
            raise UndefinedBehaviourError

        # Then, convert received values into a numpy ndarray and replace
        # the relevant parts of self.__XA
        old_XA = np.copy(self.__XA)

        new_vectors = np.array(args).T
        self.__XA[0:len(args[0]), 0:len(args)] = new_vectors
        if np.isclose(np.linalg.det(self.__XA), 0):
            self.__XA = old_XA
            raise LinearDependenceError

        if atoms_behaviour == Unit.Angstrom:
            for i, a in enumerate(self.__atoms):
                el, old_pos, spin = a
                self.__atoms[i] = (el, self.__to_crystal_base(old_XA @ old_pos),
                                   spin)

    def vectors(self) -> List[VectorNumpy]:
        """
        Lists lattice vectors

        Returns
        -------
        List[Vec3D]
            List of unit cell vectors (in angstrom)
        """
        return self.__XA.T.tolist()

    def __to_angstrom_base(self, pos: np.ndarray) -> np.ndarray:
        """
        Convert from crystal base to angstrom xyz base
        """
        return self.__XA @ pos

    def __to_crystal_base(self, pos: np.ndarray) -> np.ndarray:
        """
        Convert from angstrom xyz base to crystal base
        """
        return np.linalg.inv(self.__XA) @ pos

    def add_atom(self, element: str, pos: VectorLike, spin: VectorLike = (0, 0, 0),
                 unit: Unit = Unit.Angstrom) -> None:
        """
        Adds a single atom to the unit cell of the lattice
        Parameters
        ----------
        element : str
            Symbol of the chemical element (does NOT accept full names)
            If unknown symbol is passed, warns the user and defaults to hydrogen
        pos : 2D or 3D vector
            Position of the atom in the unit cell. If atomic position is
            not within the parallelepiped described by unit cell vectors,
            position is accepted but a warning is issued.
        spin : 3D vector, optional
            Describes spin of the atom (s_x, s_y, s_z), default: (0, 0, 0)
        unit : Unit
            Gives unit in which `pos` was given (must be either Unit.ANGSTROM
            or Unit.CRYSTAL)

        Returns
        -------
        None

        Warns
        -----
        UserWarning
            If an unknown chemical element symbol is passed as `element`
            If the atomic position is outside the box defined by unit cell
             vectors.
        """
        # check correctness of input data
        if element not in PERIODIC_TABLE:
            warnings.warn(Warning.UnknownChemicalElement.value)

        if len(pos) == 2:
            pos = [pos[0], pos[1], 0]
        pos = np.array(pos)
        if unit == Unit.Angstrom:
            pos = self.__to_crystal_base(pos)

        if not ((0 <= pos).all() and (pos < 1).all()):
            warnings.warn(Warning.AtomOutsideElementaryCell.value)

        self.__atoms.append((element, pos, spin))

    def add_atoms(self, atoms: List[Atom], unit: Unit = Unit.Angstrom) -> None:
        """
        Adds atoms listed in `atoms` to the unit cell

        Parameters
        ----------
        atoms : List[Atom]
            List of atoms to add to the lattice unit cell
        unit : Unit
            Unit in which atomic positions of the `atoms` are specified

        Returns
        -------
        None
        """
        for atom in atoms:
            self.add_atom(*atom, unit=unit)

    def atoms(self, unit: Unit = Unit.Angstrom) -> List[Atom]:
        """
        Lists atoms in the unit cell

        Parameters
        ----------
        unit : Unit
            Unit in which to return atomic positions

        Returns
        -------
        List[Atom]
            List of atoms in an unit cell of the lattice
        """
        if unit == Unit.Crystal:
            return self.__atoms
        if unit == Unit.Angstrom:
            return [(el, self.__to_angstrom_base(pos), spin) for (el, pos, spin)
                    in self.__atoms]

    def save_POSCAR(self, filename: Optional[str] = None) -> None:
        """
        Saves lattice structure in VASP POSCAR file.
        Order of the atomic species is the same as order of their first
        occurence in the list generated by `atoms` method of this object.

        Parameters
        ----------
        filename : str, optional
            if not provided, writes to stdout

        Returns
        -------
        None
        """
        # let's use 1.0 as scaling constant for simplicity
        s = "supercell_generated_POSCAR\n1.0\n"

        # lattice vectors
        lattice_vectors = flatten_rect_array(self.__XA.T)
        for i, val in enumerate(lattice_vectors):
            s += "{:.5g}".format(val) + " "
            if i % 3 == 2:
                s = s[:-1] + '\n'

        # counts of each 'atomic species' in one line
        atomic_species = {}
        names = []

        for a in self.__atoms:
            try:
                atomic_species[a[0]].append(a)
            except KeyError:
                atomic_species[a[0]] = [a]
                names.append(a[0])

        for name in names:
            atoms_list = atomic_species[name]
            s += "{} ".format(len(atoms_list))
        s = s[:-1] + '\n'

        # coordinate system
        s += "Direct\n"

        # for each atom write down the coordinates in Crystal coordinates
        for name in names:
            for atom in atomic_species[name]:
                s += "{:.5g} {:.5g} {:.5g}\n".format(*atom[1])

        # saving
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(s)
        else:
            print(s)

        print("Note: Order of the atomic species in this generated POSCAR " + \
              "file is as follows:\n" + " ".join(names))

    def save_xsf(self, filename: Optional[str] = None) -> None:
        """
        Saves lattice structure in XCrysDen XSF file
        Parameters
        ----------
        filename : str, optional
            if not provided, writes to stdout

        Returns
        -------
        None
        """
        s = "CRYSTAL\n\nPRIMVEC\n"

        # lattice vectors
        lattice_vectors = flatten_rect_array(self.__XA.T)
        for i, val in enumerate(lattice_vectors):
            s += "{:.5g}".format(val) + " "
            if i % 3 == 2:
                s = s[:-1] + '\n'

        s += "\nPRIMCOORD\n"

        # number of atoms and '1'
        s += "{} 1\n".format(len(self.__atoms))

        # atoms: 'atomic_number pos in angstroms (x y z) spin (x y z)
        for atom in self.atoms(unit=Unit.Angstrom):
            s += "{} {:.5g} {:.5g} {:.5g} {} {} {}\n".format(
                atomic_number(atom[0]),
                *atom[1],
                *atom[2]
            )

        # saving
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(s)
        else:
            print(s, end='')


def lattice():
    """
    Creates a Lattice object

    Returns
    -------
    Lattice
        a new Lattice object
    """
    return Lattice()
