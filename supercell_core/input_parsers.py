from typing import List, Optional

from .errors import *
from .physics import Unit
from .lattice import lattice, Lattice
from .heterostructure import Heterostructure


def read_POSCAR(filename: str,
                atomic_species: List[str],
                magmom: Optional[str] = None,
                normalise_positions: Optional[bool] = False) -> Lattice:
    """
    Reads VASP input file "POSCAR"

    Parameters
    ----------
    filename : str
    atomic_species : List[str]
        Contains symbols of chemical elements in the same order as in POTCAR
        One symbol per atomic species
    magmom : str, optional
        Contents of the MAGMOM line from INCAR file.
        Default: all spins set to zero
    normalise_positions : bool, optional
        If True, atomic positions are moved to be within the elementary cell
        (preserving location of atoms in the whole crystal)
        Default: False

    Returns
    -------
    Lattice
        object representing the same lattice as the VASP input files

    Raises
    ------
    IOError
        If something is wrong with I/O on the file
    ParseError
        If supplied file is not a valid POSCAR, or if the arguments supplied
        cannot be reasonably paired with the input file in a way that makes
        a proper Lattice
    """
    with open(filename, 'r') as f:
        # I don't think there is any reasonable POSCAR with > 1000 atoms
        # so s will be at most a few tens of thousands of kB, so just read() it
        s = f.read()
    return parse_POSCAR(s, atomic_species, magmom,
                        normalise_positions=normalise_positions)


# Helper functions for parser to make the code read better

def eat_line(s: List[str]) -> List[str]:
    # Takes a list of lines, returns list without the first one
    return s[1:]


def get_line(s: List[str]) -> str:
    # Returns first line in list
    return s[0].strip()


def iter_magmom(magmom: str):
    # TODO: Type Hint for generator
    # Yields values of spin from magmom string
    s = [x.strip() for x in magmom.splitlines()][0].split()
    try:
        for el in s:
            y = el.split('*')
            if len(y) == 2:
                z_spin = int(y[1])
                count = int(y[0])
                for i in range(count):
                    yield (0, 0, z_spin)
            elif len(y) == 1:
                yield (0, 0, int(y[0]))
            else:
                raise ParseError("Ambiguous spin description using '*'")
    except Exception:
        raise ParseError("Invalid MAGMOM string supplied")


def parse_POSCAR(poscar: str,
                 atomic_species: List[str],
                 magmom: Optional[str] = None,
                 normalise_positions: Optional[bool] = False) -> Lattice:
    """
    Reads lattice data from a string, treating it as VASP POSCAR file
    Format documentation: https://cms.mpi.univie.ac.at/wiki/index.php/POSCAR

    Parameters
    ----------
    poscar : str
        Contents of the POSCAR file
    atomic_species : List[str]
        Contains symbols of chemical elements in the same order as in POTCAR
        One symbol per atomic species
    magmom : str, optional
        Contents of the MAGMOM line from INCAR file.
        Default: all spins set to zero
    normalise_positions : bool, optional
        If True, atomic positions are moved to be within the elementary cell
        (preserving location of atoms in the whole crystal)
        Default: False

    Returns
    -------
    Lattice
        object representing the same lattice as the VASP would-be input

    Raises
    ------
    IOError
        If something is wrong with I/O on the file
    ParseError
        If supplied file is not a valid POSCAR, or if the arguments supplied
        cannot be reasonably paired with the input file in a way that makes
        a proper Lattice
    """
    # Build the lattice the usual way
    res = lattice()

    try:
        s = poscar.splitlines()

        # 1: system name, irrelevant for us
        s = eat_line(s)

        # 2: scale factor
        scale = float(get_line(s))
        s = eat_line(s)

        # 3, 4, 5: lattice vectors in angstroms
        vecs = []
        for i in range(3):
            vecs.append([scale * float(x.strip()) for x in get_line(s).split()])
            if len(vecs[-1]) != 3:
                raise ParseError("Vector length different than 3")
            s = eat_line(s)

        res.set_vectors(*vecs)

        # 6: atomic species counts
        line = get_line(s).split()
        as_counts = []
        if len(line) != len(atomic_species):
            raise ParseError("Number of atomic species doesn't match ({} != {})".format(
                len(get_line(s).split()), len(atomic_species)
            ))

        # Note: VASP will output here (and read correctly) a line with names of
        # the atomic species; This is an undocumented feature of VASP
        # so files written by supercell_core don't contain this line; However,
        # we must check if this line exist and if so, ommit it.
        # (in the future we might check its contents against `atomic_species`)
        try:
            int(line[0])
        except ValueError:
            s = eat_line(s)

        for x in get_line(s).split():
            as_counts.append(int(x.strip()))
        s = eat_line(s)

        # 7: possibly Selective Dynamics, then remember to ignore Ts and Fs
        # at the ends of positions
        selective_dynamics = False
        if get_line(s)[0] in "Ss":
            s = eat_line(s)
            selective_dynamics = True

        # 8: Cartesian or Direct
        unit = Unit.Crystal
        if get_line(s)[0] in "CcKk":
            unit = Unit.Angstrom
        s = eat_line(s)

        # 9+: atomic positions
        if magmom:
            spins = iter_magmom(magmom)
        else:
            # default spin: 0
            spins = iter([(0, 0, 0)] * sum(as_counts))

        def letter_to_sd_bool(letter: str):
            if letter == 'T':
                return True
            elif letter == 'F':
                return False
            raise ParseError("Bad selective dynamics flag")

        for specie, count in zip(atomic_species, as_counts):
            for i in range(count):
                splitted = get_line(s).split()
                vec = [float(x) for x in splitted[0:3]]
                if selective_dynamics:
                    if len(splitted) != 6:
                        raise ParseError("Vector length different than 3, "
                                        + "or bad number of selective dynamics "
                                        + "flags")
                    sd = tuple(map(letter_to_sd_bool,
                                   [x.strip() for x in splitted[3:6]]))

                else:
                    if len(splitted) != 3:
                        raise ParseError("Vector length different than 3")
                res.add_atom(specie, vec, next(spins), unit=unit,
                             normalise_positions=normalise_positions)
                s = eat_line(s)

    except Exception as e:
        raise ParseError(str(e))
    return res


def read_supercell_in(filename: str) -> Heterostructure:
    pass
