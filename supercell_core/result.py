from typing import Tuple, List
import numpy as np

from .lattice import Lattice
from .physics import Quantity, Angle, Matrix2x2, VectorNumpy
from .calc import inv

class Result:
    """
    A class containing results of supercell calculations.

    Notes
    -----
    Terminology:

    MN – basis change matrix from N to M
         (the order of the letters makes it easy to combine these:
          MM' @ M'N == MN)
    v_M – vector v (unit vector of basis V) in basis M
    v_Ms – an array of vectors v in basis M
    stg_lay – list of "stg" for each of the layers
              (len(stg_lay) == len(self.layers()))

    A, a – basis of lattice vectors of the substrate
    B, b – basis of lattice vectors of a given layer
    Br, br – basis B rotated by theta angle
    Btr, btr – basis of lattice vectors of a given layer when embedded
          in the heterostructure (rotated – r, and stretched
          due to strain – t)
    D, d – basis of vectors composed of integer linear combinations of
           the B basis vectors
    Dt, dt – basis of vectors composed of the integer linear combinations
             of the A basis vectors, represents possible supercell lattice
             vectors
    X, x – cartesian basis (unit vectors: (1 angstrom, 0),
                                       (0, 1 angstrom))

    Note that the vector space of all the mentioned objects is R^2
    """
    __heterostructure : "Heterostructure"
    __superlattice : Lattice
    __thetas : List[Angle]
    __strain_tensors : List[Matrix2x2]

    def __init__(self,
                 heterostructure: "Heterostructure",
                 superlattice: Lattice,
                 thetas: List[Angle],
                 strain_tensors: List[Matrix2x2],
                 strain_tensors_wiki: List[Matrix2x2]):
        """

        Parameters
        ----------
        heterostructure : Heterostructure
        superlattice : Lattice
        thetas : List[float]
        strain_tensors : List[Matrix2x2]
            in cartesian basis
        strain_tensors_wiki : List[Matrix2x2]
        """
        self.__heterostructure = heterostructure
        self.__superlattice = superlattice
        self.__thetas = thetas
        self.__strain_tensors = strain_tensors
        self.__strain_tensors_wiki = strain_tensors_wiki

    def strain_tensors(self, wiki_definition=False) -> \
            List[Matrix2x2]:
        """
        Returns list of strain tensors for each of the heterostructure's layers.

        Parameters
        ----------
        wiki_definition : optional, bool
            Default: False.
            If True, definition from [1] will be used instead of the default
            (see docs of `Heterostructure.calc` for details)

        Returns
        -------
        List[Matrix2x2]

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Infinitesimal_strain_theory
        """
        if wiki_definition:
            return self.__strain_tensors_wiki
        return self.__strain_tensors

    def max_strain(self):
        """
        Returns absolute maximum value of sum of strains on all layers.

        Returns
        -------
        float

        Notes
        -----
        The returned value is minimised in `Heterostructure.opt` calculations.
        """
        return np.max(np.abs(np.sum(self.strain_tensors())))

    def BtrBrs(self) -> Matrix2x2:
        """

        Returns
        -------

        """
        # TODO
        pass

    def BtrDts(self) -> Matrix2x2:
        """

        Returns
        -------

        """
        # TODO
        pass

    def atom_count(self) -> int:
        """

        Returns
        -------

        """
        # TODO doc
        return len(self.superlattice().atoms())

    def superlattice(self) -> Lattice:
        """

        Returns
        -------

        """
        # TODO doc
        return self.__superlattice

    def layers_lattice_vectors(self) -> \
        List[Tuple[VectorNumpy, VectorNumpy, VectorNumpy]]:
        """

        Returns
        -------

        """
        # TODO
        pass

    def thetas(self) -> List[Angle]:
        """

        Returns
        -------

        """
        # TODO doc
        return self.__thetas
