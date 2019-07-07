from typing import Optional, Union, List, Tuple
import numpy as np

from .lattice import Lattice
from .physics import Number, Quantity, Unit, DEGREE
from .result import *

# Type alias for an angle
Angle = Number

# Type aliases for range of angles
AngleRange = Tuple[Angle, Angle, Angle]

# Type alias for immutable representation of 2x2 matrix
TupleMatrix2x2 = Tuple[Tuple[Number, Number], Tuple[Number, Number]]

# Type alias for mutable represenation of 2x2 matrix
ListMatrix2x2 = List[List[Number]]

# Type alias for Numpy 2x2 array (used internally to represent 2D square matrices)
# note: no support on dimensionality and ndarray dtype typing available yet
Matrix2x2 = np.ndarray

# Type alias for a description of 2D square matrix that can be entered into
# a public method
InMatrix2x2 = Union[TupleMatrix2x2, Matrix2x2, ListMatrix2x2]


class Heterostructure:
    """
    Class describing a system of a number of 2D crystals deposited
    on a substrate.

    Elementary cell vectors of the substrate are hereafter described as
    a_1 through a_3, of the 2D layers above as b_i_1 through b_i_3 (where
    i is layer number, counting from the substrate up, starting at 1),
    elementary cell of the whole system (where 2D layers have been changed
    due to strain) is called heterostructure _supercell_ and its vectors
    are referred to as c_1, c_2, and c_3.

    Do not confuse `Heterostructure` with a class representing heterostructure
    lattice. `Heterostructure` describes a collection of crystals that build
    up the structure; to obtain the crystal lattice resulting from joining
    these crystals you must first define a way in which these crystals come
    together (angles, etc. – use `opt`, `res`, or `plot` methods to do this),
    and the resulting lattice will be available as `supercell` attribute
    of the `Result` class (the `supercell` attribute will have `Lattice` class)

    """

    def set_substrate(self, substrate: Lattice) -> None:
        """
        Defines substrate layer of the heterostructure

        Parameters
        ----------
        substrate : Lattice
            Lattice object describing elementary cell of the substrate on which
            2D crystals will be laid down

        Returns
        -------
        None
        """
        pass

    def add_layer(self, layer: Lattice, pos: Optional[int] = None,
                  theta: Union[Angle, AngleRange] =
                  (0, 180 * DEGREE, 0.1 * DEGREE)) -> None:
        """
        Adds a 2D crystal to the system

        Parameters
        ----------
        layer : Lattice
            Lattice object describing elementary cell of the crystal to add to
            the system
        pos : int, optional
            Position of the layer in the stack, counting from the substrate up
            (first position is 1). If not specified, layer will be added at the
            top of the stack
        theta : Angle or AngleRange, optional
            If specified, supplied value of `theta` will be used in place of
            the default values in calculations such as `opt`.
            If a single angle is passed then the `theta` angle of that layer
            is fixed to that value. If an AngleRange (three floats) is passed
            then it is treated as a range of possible values for the `theta`
            value (start, stop, step).
            Unit: radians
            Default: (0, pi, 0.1 * DEGREE)

        Returns
        -------
        None
        """

    def add_layers(self, layers: List[Union[Lattice,
                                            Tuple[Lattice,
                                                  Union[Angle, AngleRange]]]]):
        pass

    def substrate(self) -> List[Lattice]:
        pass

    def layers(self) -> List[Lattice]:
        pass

    def remove_layer(self, layer_no: int) -> None:
        pass

    # TODO: add parameter for the heuristic, how many best guesses to check

    def calc(self,
             qty: Quantity,
             M: InMatrix2x2 = ((1, 0), (0, 1)),
             thetas: Optional[List[Union[Angle, None]]] = None) -> Result:
        """
        Calculates specified quantity for a given system under given constraints

        Parameters
        ----------
        qty : Quantity
            Enumerated value describing function to calculate
        M : 2x2 matrix
            Base change matrix from the base of the supercell lattice
            elementary cell vectors to the base of the substrate elementary cell
            vectors (<c_1, c_2> -> <a_1, a_2>), or, in other words,
            c_1 = M_11 * a_1 + M_21 * a_2, and c_2 = M_12 * a_1 + M_22 * a_2.
        thetas : List[Angle|None], optional
            Required if `theta` parameter is not fixed for all the layers
            in the system.
            If specified, it will be zipped with the layers list (starting from
            the layer closest to the substrate), and then, if the value
            corresponding to a layer is not None then that layer's `theta`
            angle will be set to that value.
            Example:
                >>> h = heterostructure()
                >>> h.set_substrate(lattice())
                >>> lay1, lay2, lay3 = lattice(), lattice(), lattice()
                >>> h.add_layers([(lay1, 180 * DEGREE), lay2, lay3])
                >>> h.calc(Quantity.Strain, M=((1, 2), (3, 4)), \
                    thetas = [None, 45 * DEGREE, 90 * DEGREE) \
                    # will be set to (180, 45, 90) degrees for layers \
                    # (lay1, lay2, lay3) repsectively

        Returns
        -------
        Result
            Result object containing the results of the calculation.
            For more information see documentation of `Result`
        """
        pass

    def opt(self,
            qty: Quantity = Quantity.MaxStrainElement,
            max_el: int = 6,
            max_supercell_size: Optional[float] = None) -> Result:
        """
        Minimises strain measure quantity, and calculates its value

        Parameters
        ----------
        qty : Quantity
            Enumerated value describing function to minimise
            Default: Quantity.MaxStrainElement
        max_el : int, optional
            Let M be a base change matrix from the base of the supercell lattice
            elementary cell vectors to the base of the substrate elementary cell
            vectors (<c_1, c_2> -> <a_1, a_2>), or, in other words,
            c_1 = M_11 * a_1 + M_21 * a_2, and c_2 = M_12 * a_1 + M_22 * a_2.
            Then `max_el` defines maximum absolute value of the M matrix element
            The opt calculation is O(`max_el`^2).
            Default: 6.
        max_supercell_size : float, optional
            Defines maximal area of the supercell lattice elementary cell
            (x-y directions) in square angstroms (1e-20 m^2)
            If not set, then the maximum supercell area is left unbounded.

        Returns
        -------
        Result
            Result object containing the results of the optimisation.
            For more information see documentation of `Result`
        """
        pass

    def plot(self,
             qty: Quantity,
             max_el: int,
             Ms: List[Union[InMatrix2x2, None]],
             thetas: List[Union[InMatrix2x2, None]]
             ) -> List[(float)]:
        raise NotImplementedError


def heterostructure() -> Heterostructure:
    """
    Creates a Heterostructure object

    Returns
    -------
    Heterostructure
        a new Heterostructure object
    """
    return Heterostructure()
