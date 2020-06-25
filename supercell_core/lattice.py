from typing import List, Optional, Tuple

from .errors import *
from .calc import *
from .physics import Unit, Number, VectorLike, VectorNumpy, PERIODIC_TABLE, \
    atomic_number
from .atom import Atom


# noinspection PyPep8Naming
class Lattice:
    """
    Object representing a 2D crystal lattice (or rather its unit cell)

    Initially set with default values: no atoms, unit cell vectors
    equal to unit vectors (of length 1 angstrom each) along x, y, z axes.

    Elementary cell vectors are here and elsewhere referred to as b_1, b_2, b_3.
    """
    # Let's keep atoms in a list not set, because most operations will be done
    # probably on all atoms together anyway, except maybe for adding atoms;
    # list doesn't really create any significant disadvantages over a set
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
                    atoms_behaviour: Optional[Unit] = None) -> "Lattice":
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
        Lattice
            for chaining

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
                warnings.warn(WarningText.ReassigningPartOfVector.value)
            if len(v) == 3 and v[2] != 0 and len(args) == 2:
                warnings.warn(WarningText.ZComponentWhenNoZVector.value)
        if (not isinstance(atoms_behaviour, Unit)) and len(self.__atoms) > 0:
            raise UndefinedBehaviourError

        # Then, convert received values into a numpy ndarray and replace
        # the relevant parts of self.__XA
        old_XA = np.copy(self.__XA)

        new_vectors = np.array(args).T
        self.__XA[0:len(args[0]), 0:len(args)] = new_vectors
        if np.isclose(np.linalg.det(self.__XA), 0):
            bad_XA = self.__XA
            self.__XA = old_XA
            raise LinearDependenceError("{}".format(bad_XA))
        
        if atoms_behaviour == Unit.Angstrom:
            for i, a in enumerate(self.__atoms):
                self.__atoms[i] = Atom(a.element,
                                       self.__to_crystal_base(old_XA @ a.pos),
                                       Unit.Crystal,
                                       spin=a.spin)

        return self

    def rotate(self, theta: float):
        """
        Rotates the lattice around the origin.

        Parameters
        ----------
        theta : float
            Angle in radians

        Returns
        -------
        Lattice
            for chaining
        """
        # TODO: tests
        vecs = self.vectors()
        v1 = vecs[0][0:2]
        v2 = vecs[1][0:2]
        v1 = rotate(v1, theta)
        v2 = rotate(v2, theta)
        self.set_vectors(v1, v2, atoms_behaviour=Unit.Crystal)
        return self

    def vectors(self) -> List[VectorNumpy]:
        """
        Lists lattice vectors

        Returns
        -------
        List[Vec3D]
            List of unit cell vectors (in angstrom)
        """
        return self.__XA.T.tolist()

    def basis_change_matrix(self) -> np.ndarray:
        """
        Returns basis change matrix from basis of lattice vectors (Direct) to
        Cartesian basis.

        Returns
        -------
        np.ndarray, shape (3,3)
        """
        return self.__XA

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

    def add_atom(self,
                 element: str,
                 pos: VectorLike,
                 spin: VectorLike = (0, 0, 0),
                 unit: Unit = Unit.Angstrom,
                 normalise_positions: bool = False) -> "Lattice":
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
            Gives unit in which `pos` was given (must be either `Unit.Angstrom`
            or `Unit.Crystal`)
        normalise_positions : bool
            If True, atomic positions are moved to be within the elementary cell
            (preserving location of atoms in the whole crystal)
            Default: False

        Returns
        -------
        Lattice
            for chaining

        Warns
        -----
        UserWarning
            If an unknown chemical element symbol is passed as `element`
            If the atomic position is outside the elementary cell defined by
            lattice vectors, and `normalise_positions` is False

        Raises
        ------
        TypeError
            If supplied arguments are of incorrect type
        """
        # TODO: replace with Atom class constructor
        # check correctness of input data
        if element not in PERIODIC_TABLE:
            warnings.warn(WarningText.UnknownChemicalElement.value)

        if len(pos) == 2:
            pos = [pos[0], pos[1], 0]
        elif len(pos) == 3:
            pass
        else:
            raise TypeError("Bad length of atomic position vector")

        if len(spin) != 3:
            raise TypeError("Bad length of spin vector. Must be a triple")

        pos = np.array(pos)
        if unit == Unit.Angstrom:
            pos = self.__to_crystal_base(pos)

        if (not ((0 <= pos).all() and (pos < 1).all())) \
                and not normalise_positions:
            warnings.warn(WarningText.AtomOutsideElementaryCell.value)

        if normalise_positions:
            # move all positions to be within the elementary cell
            pos %= 1.0

        self.__atoms.append(Atom(element, pos, Unit.Crystal, spin=spin))
        return self

    def add_atoms(self, atoms: List[Atom]) -> "Lattice":
        """
        Adds atoms listed in `atoms` to the unit cell

        Parameters
        ----------
        atoms : List[Atom]
            List of atoms to add to the lattice unit cell

        Returns
        -------
        Lattice
            for chaining
        """
        # TODO: use Atom constructor
        for atom in atoms:
            self.add_atom(atom.element, atom.pos, spin=atom.spin, unit=atom.pos_unit)
        return self

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
            return [atom.basis_change(self.__XA, Unit.Angstrom)
                    for atom in self.__atoms]

    def save_POSCAR(self,
                    filename: Optional[str] = None,
                    silent: bool = False) -> "Lattice":
        """
        Saves lattice structure in VASP POSCAR file.
        Order of the atomic species is the same as order of their first
        occurence in the list generated by `atoms` method of this object.
        This order is printed to stdout.
        If atoms have non-zero z-spins, the MAGMOM flag is also printed
        to stdout.

        Parameters
        ----------
        filename : str, optional
            if not provided, writes to stdout
        silent : bool, default: False
            if True, the atomic order and MAGMOM flag are not printed to stdout

        Returns
        -------
        Lattice
            for chaining
        """
        # TODO: ensure proper chirality
        # let's use 1.0 as scaling constant for simplicity
        s = "supercell_generated_POSCAR\n1.0\n"

        # lattice vectors
        lattice_vectors = self.__XA.T.flatten().tolist()
        for i, val in enumerate(lattice_vectors):
            s += "{:.5g}".format(val) + " "
            if i % 3 == 2:
                s = s[:-1] + '\n'

        # counts of each 'atomic species' in one line
        atomic_species = {}
        names = []

        for a in self.__atoms:
            try:
                atomic_species[a.element].append(a)
            except KeyError:
                atomic_species[a.element] = [a]
                names.append(a.element)

        # New in VASP December 2019 – atomic species names
        for name in names:
            s += "{} ".format(name)
        s = s[:-1] + '\n'

        for name in names:
            atoms_list = atomic_species[name]
            s += "{} ".format(len(atoms_list))
        s = s[:-1] + '\n'

        # coordinate system
        s += "Direct\n"

        # z-spins counting for magmom
        # elements of type (z_spin, atoms count)
        magmom: List[Tuple[Number, int]] = []

        # for each atom write down the coordinates in Crystal coordinates
        for name in names:
            # while we're at it, we can also already sort atomic_species[name]
            # by their atoms' z-spin (useful for calculating the MAGMOM flag)
            # reverse=True, so spin up (1) is before down (-1), matter of pref.
            atomic_species[name].sort(key=self.__z_spin, reverse=True)

            for atom in atomic_species[name]:
                try:
                    if magmom[-1][0] == self.__z_spin(atom):
                        magmom[-1] = (magmom[-1][0], magmom[-1][1] + 1)
                    else:
                        magmom.append((self.__z_spin(atom), 1))
                except IndexError:
                    # happens for the first atom only
                    magmom.append((self.__z_spin(atom), 1))

                # print atom position
                s += "{:.5g} {:.5g} {:.5g}\n".format(*atom.pos)

        # saving
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(s)
        else:
            print(s)

        if not silent:
            # magmom
            magmom_str = " ".join([((str(x[1]) + "*" if x[1] > 1 else "")
                                    + str(x[0])) for x in magmom])
            print("MAGMOM flag: " + magmom_str)

        return self

    def save_xsf(self, filename: Optional[str] = None) -> "Lattice":
        """
        Saves lattice structure in XCrysDen XSF file

        Parameters
        ----------
        filename : str, optional
            if not provided, writes to stdout

        Returns
        -------
        Lattice
            for chaining
        """
        s = "CRYSTAL\n\nPRIMVEC\n"

        # lattice vectors
        lattice_vectors = self.__XA.T.flatten().tolist()
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
                atomic_number(atom.element),
                *atom.pos,
                *atom.spin
            )

        # saving
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(s)
        else:
            print(s, end='')

        return self

    @staticmethod
    def __z_spin(atom: Atom) -> Number:
        return atom.spin[2]

    def translate_atoms(self, vec: VectorLike, unit: Unit = Unit.Angstrom):
        """
        Moves all atoms in the unit cell by some vector `vec`

        Parameters
        ----------
        vec : VectorLike
            Translation vector. If len(vec) == 2, third argument is set to zero.
        unit : Unit, optional
            Unit of `vec`, by default: angstroms

        Returns
        -------
        None
        """
        if len(vec) == 2:
            vec = (vec[0], vec[1], 0)

        atoms = self.atoms(unit=unit)
        for a in atoms:
            a.pos = tuple(np.array(a.pos) + np.array(vec))
        self.__atoms = []
        self.add_atoms(atoms)

    def draw(self, ax=None, cell_coords=None, scatter_kwargs=None, border_kwargs=None):
        """
        Requires matplotlib. Creates an image of the lattice elementary cell
        as a matplotlib plot.

        Parameters
        ----------
        cell_coords : List[Vector2D], optional
            Each point in `cell_coords` is an offset in Lattice vector basis.
            For each of those offsets, all atoms in the cell will be drawn with
            their positions moved by this offset. Use integer values to get
            consistent drawings. Note: You can use things like [(-0.5, -0.5),
             (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)] to get a picture of a translated
             elementary cell.
            Default: [(0, 0)] (draws just one elementary cell)
        ax : matplotlib axes.Axes object, optional
            If given, will draw the elementary cell to ax

        Returns
        -------
        Figure, Axes

        Notes
        -----
        Resulting axes has points labeled; use
        `ax.legend(loc='best')` to turn on the legend.
        """
        if cell_coords is None:
            cell_coords = [np.array([0, 0])]
        else:
            cell_coords = [np.array(pt) for pt in cell_coords]

        if ax is None:
            import matplotlib.pyplot as plt
            plt.clf()
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if scatter_kwargs is None:
            scatter_kwargs = {
                "marker": '.',
                "s": 2
            }
        if border_kwargs is None:
            border_kwargs = {
                "linestyle": '--',
                "color": (55 / 255, 126 / 255, 184 / 255),  # orig. "gray"
                "linewidth": 2
            }

        species = set()
        for a in self.atoms():
            species.add(a.element)
        species = sorted(species, key=atomic_number)

        # A set of nice colours from ColorBrewer2.org
        colors = [
            (228 / 255, 26 / 255, 28 / 255),
            #(55 / 255, 126 / 255, 184 / 255),
            (77 / 255, 175 / 255, 74 / 255),
            (152 / 255, 78 / 255, 163 / 255),
            (255 / 255, 127 / 255, 0 / 255),
            (255 / 255, 255 / 255, 51 / 255),
            (166 / 255, 86 / 255, 40 / 255),
            (247 / 255, 129 / 255, 191 / 255),
            (153 / 255, 153 / 255, 153 / 255)
        ]

        # Hack if there is less colors than species
        colors = colors * (len(species) // len(colors) + 1)

        for color, specie in zip(colors, species):
            atoms = [a for a in self.atoms() if a.element == specie]
            positions = [a.pos for a in atoms]
            v1, v2 = self.__XA[0:2, 0], self.__XA[0:2, 1]
            for cell_coord in cell_coords:
                positions_x = [p[0] + v1[0] * cell_coord[0] + v2[0] * cell_coord[1]
                               for p in positions]
                positions_y = [p[1] + v1[1] * cell_coord[0] + v2[1] * cell_coord[1]
                               for p in positions]
                ax.scatter(positions_x,
                           positions_y,
                           label=specie,
                           color=color,
                           **scatter_kwargs)

                for a, px, py in zip(atoms, positions_x, positions_y):
                    if a.spin[2] > 0:
                        ax.annotate('↑', (px, py))
                    elif a.spin[2] < 0:
                        ax.annotate('↓', (px, py))

        # Draw cell boundary
        vecs = np.array([v[0:2] for v in self.vectors()[0:2]])
        pts = np.array([(0, 0), vecs[0], vecs[0] + vecs[1], vecs[1], (0, 0)]).T

        ax.plot(pts[0], pts[1], **border_kwargs)

        ax.set_aspect('equal', adjustable='box')
        return fig, ax


def lattice():
    """
    Creates a Lattice object

    Returns
    -------
    Lattice
        a new Lattice object
    """
    return Lattice()
