from typing import Optional, List
import itertools

from .lattice import Lattice
from .physics import *
from .result import *
from .calc import *
from .heterostructure_opt import *

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
    together (angles, etc. – use `opt` or `res` methods to do this),
    and the resulting lattice will be available as `supercell` attribute
    of the `Result` class (see documentation of the relevant methods)
    """

    # Terminology used throughout the implementation:
    #
    # MN – basis change matrix from N to M
    #      (the order of the letters makes it easy to combine these:
    #       MW @ WN == MN for any W)
    # v_M – vector v (unit vector of basis V) in basis M
    # v_Ms – an array of vectors v in basis M
    # stg_lay – list of "stg" for each of the layers
    #           (len(stg_lay) == len(self.layers()))
    #
    # A, a – basis of lattice vectors of the substrate
    # B, b – basis of lattice vectors of a given layer
    # Br, br – basis B rotated by theta angle
    # Btr, btr – basis of lattice vectors of a given layer when embedded
    #       in the heterostructure (rotated – r, and stretched
    #       due to strain – t)
    # D, d – basis of vectors composed of integer linear combinations of
    #        the B basis vectors
    # Dt, dt – basis of vectors composed of the integer linear combinations
    #          of the A basis vectors, represents possible supercell lattice
    #          vectors
    # X, x – cartesian basis (unit vectors: (1 angstrom, 0),
    #                                    (0, 1 angstrom))
    # Xt, xt – Transformation corresponding to Br - Btr transformation, but in
    # cartesian basis
    #
    # Note that the vector space of all the mentioned objects is R^2

    __layers: List[Tuple[Lattice, AngleRange]]
    __substrate: Lattice

    def __init__(self):
        # We store data on preferred theta (or theta range) as second element
        # of the pair, alongside with the Lattice itself
        self.__layers = []

    ### LATTICES METHODS

    def set_substrate(self, substrate: Lattice) -> "Heterostructure":
        """
        Defines substrate layer of the heterostructure

        Parameters
        ----------
        substrate : Lattice
            Lattice object describing elementary cell of the substrate on which
            2D crystals will be laid down

        Returns
        -------
        Heterostructure
            for chaining
        """
        self.__substrate = substrate
        return self

    def substrate(self) -> Lattice:
        """
        Getter for the substrate of the heterostructure

        Returns
        -------
        Lattice

        Raises
        ------
        AttributeError
            if substrate wasn't set yet
        """
        return self.__substrate

    def add_layer(self,
                  layer: Lattice,
                  pos: Optional[int] = None,
                  theta: Union[Angle, AngleRange] =
                  (0, 180 * DEGREE, 0.1 * DEGREE)) -> "Heterostructure":
        """
        Adds a 2D crystal to the system

        Parameters
        ----------
        layer : Lattice
            Lattice object describing elementary cell of the crystal to add to
            the system
        pos : int, optional
            Position of the layer in the stack, counting from the substrate up
            (first position is 0). If not specified, layer will be added at the
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
        Heterostructure
            for chaining
        """
        try:
            # if theta is an Angle `float` will work
            theta = (float(theta), float(theta), 1.0)
        except TypeError:
            assert len(theta) == 3

        if pos is None:
            self.__layers.append((layer, theta))
        else:
            self.__layers.insert(pos, (layer, theta))

        return self

    def add_layers(self, layers: List[Union[Lattice,
                                            Tuple[Lattice,
                                                  Union[Angle, AngleRange]]]]) \
            -> "Heterostructure":
        """
        Adds a lot of layers to the heterostructure at once

        Parameters
        ----------
        layers : List[
            Lattice,res.set_vectors()
            or (Lattice, float),
            or (Lattice, (float, float, float))
        ]
            List of layers to add to the structure.
            If list element is a tuple, the second element serves the same way
            as `theta` parameter in `add_layer`

        Returns
        -------
        Heterostructure
            for chaining
        """
        for el in layers:
            # if el is a Lattice
            if type(el) is Lattice:
                self.add_layer(el)
            # if el is (Lattice, Angle) or (Lattice, AngleRange)
            else:
                self.add_layer(el[0], theta=el[1])

        return self

    def layers(self) -> List[Lattice]:
        """
        Returns layers in the heterostructure

        Returns
        -------
        List[Lattice]
            List of layers on the substrate as Lattice objects
        """
        return [el[0] for el in self.__layers]

    def remove_layer(self, pos: int) -> "Heterostructure":
        """
        Removes layer in position `pos`

        Parameters
        ----------
        pos : int
            Position of the layer in the stack, counting from the substrate up
            (first position is 0).

        Returns
        -------
        Heterostructure

        Raises
        ------
        IndexError
            If there's less layers in the stack than `pos`
        """
        self.__layers.pop(pos)
        return self

    def get_layer(self, pos: int) -> Tuple[Lattice, AngleRange]:
        """
        Get Lattice object describing layer at position `pos` and information
        on preferred theta angles for that layer

        Parameters
        ----------
        pos : int
            Position of the layer in the stack, counting from the substrate up
            (first position is 0).

        Returns
        -------
        Lattice
            Lattice object describing the layer
        (float, float, float)
            (start, stop, step) (radians) – angles used in `calc` and `opt`
            If a specific angle is set, returned tuple is (angle, angle, 1.0)

        Raises
        ------
        IndexError
            If there's less layers in the stack than `pos`
        """
        return self.__layers[pos]

    ### STRAIN TENSOR DEFINITION

    @staticmethod
    def __calc_strain_tensor(XBr: Matrix2x2, XXt: Matrix2x2) -> Matrix2x2:
        """
        Calculate strain tensor.

        See docs of `Heterostructure.calc` for definition of strain tensor.

        Parameters
        ----------
        XBr : Matrix 2x2
        XXt : Matrix 2x2

        Returns
        -------
        Matrix 2x2
        """
        BrX = inv(XBr)
        BrBtr = BrX @ XXt @ XBr
        return inv(BrBtr) - np.identity(2)

    ### CALC METHODS

    def calc(self,
             M=((1, 0), (0, 1)),
             thetas=None) -> Result:
        """
        Calculates strain tensor and other properties of the system in given
        circumstances

        !! See Notes for the definition of strain tensor used here.

        Parameters
        ----------
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
                >>> from supercell_core import *
                >>> h = heterostructure()
                >>> h.set_substrate(lattice())
                >>> lay1, lay2, lay3 = lattice(), lattice(), lattice()
                >>> h.add_layers([(lay1, 180 * DEGREE), lay2, lay3])
                >>> h.calc(M=((1, 2), (3, 4)), \
                    thetas = [None, 45 * DEGREE, 90 * DEGREE) \
                    # will be set to (180, 45, 90) degrees for layers \
                    # (lay1, lay2, lay3) repsectively

        Returns
        -------
        Result
            Result object containing the results of the calculation.
            For more information see documentation of `Result`

        Notes
        -----
        Let ei be strain tensor of layer i. Let ai_1 and ai_2 be lattice vectors
        of layer i when not under strain. Then:
        :math:`\sum_{k=1}^2 (ei + I)_j^k ai_k = a'i_j`, where
        a'i_j is j-th lattice vector of a when embedded in the heterostructure.

        This definition is different than the one given in [1].
        To calculate strain tensor as defined in [1], use
        `wiki_definition=True` in relevant `Result` methods.

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Infinitesimal_strain_theory

        """
        if thetas is not None:
            thetas = [arg if arg is not None else lay_desc[1][0]
                      for arg, lay_desc in zip(thetas, self.__layers)]
        else:
            thetas = [lay_desc[1][0] for lay_desc in self.__layers]

        ADt = np.array(M)
        return self.__calc_aux(ADt, thetas)

    def __calc_strain_tensor_from_ADt(self, ADt: Matrix2x2, XBr: Matrix2x2) -> Matrix2x2:
        """
        Calculate strain tensor.

        See docs of `Heterostructure.calc` for definition of strain tensor.

        Parameters
        ----------
        ADt : Matrix 2x2
        XBr : Matrix 2x2

        Returns
        -------
        Matrix 2x2
        """
        BrDt, BtrBr, XXt = self.__get_basis_change_matrices(ADt, XBr)
        return Heterostructure.__calc_strain_tensor(XBr, XXt)

    def __get_basis_change_matrices(self, ADt, XBr):
        """
        Helper function to retrieve BrDt, BtrBr, XXt matrices

        Parameters
        ----------
        ADt : Matrix 2x2
        XBr : Matrix 2x2

        Returns
        -------
        BrDt : Matrix 2x2
        BtrDt : Matrix 2x2
        XXt : Matrix 2x2
        """
        XA = self.__substrate.basis_change_matrix()[0:2, 0:2]
        BrDt = inv(XBr) @ XA @ ADt
        BtrBr = Heterostructure.__get_BtrBr(BrDt)
        XXt = XBr @ inv(BtrBr) @ inv(XBr)
        return BrDt, BtrBr, XXt

    def __calc_aux(self,
                   ADt: InMatrix2x2,
                   thetas: List[Angle]) -> Result:
        """
        Calculates strain tensor and other properties of the system in given
        circumstances

        Parameters
        ----------
        ADt : Matrix2x2
            AC matrix
        thetas : List[float]
            List of theta angles of the layers (length must be equal to length
            of self.__layers)

        Returns
        -------
        Result
            see `Result` documentation for details
        """
        XA = self.__substrate.basis_change_matrix()[0:2, 0:2]
        XBrs = [rotate(lay_desc[0].basis_change_matrix()[0:2, 0:2], theta)
                for theta, lay_desc in zip(thetas, self.__layers)]
        strain_tensors = [self.__calc_strain_tensor_from_ADt(ADt, XBr)
                          for XBr in XBrs]
        BrDts = []
        BtrBrs = []
        XXts = []
        for XBr in XBrs:
            BrDt, BtrBr, XXt = self.__get_basis_change_matrices(ADt, XBr)
            BrDts.append(BrDt)
            BtrBrs.append(BtrBr)
            XXts.append(XXt)

        ABtrs = [inv(XA) @ XBr @ inv(BtrBr) for XBr, BtrBr in zip(XBrs, BtrBrs)]

        # also add the alternative strain tensor definition to the Result
        strain_tensors_wiki = [Heterostructure.__get_strain_tensor_wiki(XXt)
                               for XXt in XXts]

        superlattice = self.__build_superlattice(XA @ ADt, ADt, BrDts)

        return Result(
            self,
            superlattice,
            thetas,
            strain_tensors,
            strain_tensors_wiki,
            ADt,
            ABtrs
        )

    @staticmethod
    def __get_strain_tensor_wiki(XXt: Matrix2x2) -> np.ndarray:
        """
        Calculates strain tensor as defined in [1].

        Parameters
        ----------
        XXt : Matrix2x2

        Returns
        -------
        Matrix 2x2

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Infinitesimal_strain_theory#Infinitesimal_strain_tensor
        """
        # Notice that XXt = deformation gradient tensor (F)
        # [1]
        # https://en.wikipedia.org/wiki/Finite_strain_theory#Deformation_gradient_tensor
        F = np.array(XXt)

        # replace F with strain tensor
        F += F.swapaxes(-1, -2)  # F + F^T
        F /= 2
        F -= np.identity(2)
        return F

    def __build_superlattice(self, XDt: Matrix2x2, ADt: Matrix2x2,
                             BrDts: List[Matrix2x2]) -> Lattice:
        """
        Creates Lattice object describing superlattice (supercell)
        of the heterostructure

        Parameters
        ----------
        XDt : Matrix2x2
        ADt : Matrix2x2
        BrDts : List[Matrix2x2]

        Returns
        -------
        Lattice
        """
        res = Lattice()
        # We also need to include substrate here
        lattices = [self.__substrate] + [lay_desc[0] for lay_desc in self.__layers]
        MDts = [ADt] + BrDts

        # lattice vectors
        full_XDt = np.zeros((3, 3))
        full_XDt[0:2, 0:2] = XDt
        full_XDt[0:3, 2] = sum([lay.basis_change_matrix()[2, 0:3] for lay in lattices])
        res.set_vectors(*full_XDt.T.tolist())

        # atoms
        for i, lay, MDt in zip(range(len(lattices)), lattices, MDts):
            # We stack the layers one above the other so atoms must be moved up
            # by the sum of z-sizes of all elementary cells of layers below
            z_offset = self.__get_z_offset(i)
            z_offset /= full_XDt[2, 2]  # angstrom -> Direct coordinates
            Heterostructure.__superlattice_add_atoms(res, lay, MDt, z_offset)

        return res

    @staticmethod
    def __superlattice_add_atoms(superlattice, lay, MDt, z_offset):
        """
        Adds atoma from a given heterostructure-embedded lattice to the super-
        lattice. To do this, atomic positions must be transformed to occupy 
        the same positions in the "stretched" elementary cell; and atoms 
        themselves must be copied a few times because one superlattice cell is 
        composed of a few layer cells.

        Parameters
        ----------
        superlattice : Lattice
            Superlattice to modify
        lay : Lattice
            Layer of the heterostr.; Source of the atoms to add to superlattice
        MDt : Matrix 2x2
        z_offset : float
            Describes how much above 0 in the supercell should be the layer
            `lay`. This value will be added to z-component of all atomic
            positions of atoms from this layer. (Unit: angstroms)

        Returns
        -------
        None
        """
        # Safe upper bound on the size of the supercell in any direction
        cell_upper_bound = 2 * int(round(np.max(np.abs(MDt))) + 1)

        atoms = lay.atoms(unit=Unit.Crystal)
        atomic_pos_Dt_basis = [inv(MDt) @ atom.pos[0:2] for atom in atoms]

        # vecs = 2x2 columns of DtBrt
        DtBrt = inv(MDt) @ inv(Heterostructure.__get_BtrBr(MDt))
        vecs = DtBrt.T[0:2, 0:2]

        # We will copy atoms many times, each time translated by some integer
        # linear combination of layer vecs, and those atoms that will be inside
        # the supercell will be kept
        # Those integer linear combinations are called here 'nodes'
        get_node = lambda j, k: j * vecs[0] + k * vecs[1]
        # 10 is arbtrary; hopefully it's enough
        epsilon = 10 * np.finfo(np.dtype('float64')).eps
        nodes = [get_node(j, k)
                 for j in range(-cell_upper_bound, cell_upper_bound + 1)
                 for k in range(-cell_upper_bound, cell_upper_bound + 1)
                 if 0 <= get_node(j + 0.1, k + 0.1)[0] < 1 - epsilon
                 and 0 <= get_node(j + 0.1, k + 0.1)[1] < 1 - epsilon]

        for pos, a in zip(atomic_pos_Dt_basis, atoms):
            for node in nodes:
                superlattice.add_atom(a.element,
                                      (pos[0] + node[0],
                                       pos[1] + node[1],
                                       a.pos[2] + z_offset),
                                      spin=a.spin,
                                      unit=Unit.Crystal,
                                      normalise_positions=True)

    def __get_z_offset(self, pos: int) -> float:
        """
        Return sum of z-sizes of lattices below specified position (in angstrom)
        for `Heterostructure.__build_superlattice`

        Parameters
        ----------
        pos : int
            Position of the layer in the stack (lowest is 1)

        Returns
        -------
        float
            z-offset in Angstrom
        """
        offset = 0
        # We need to include substrate since we are also adding atoms from it
        # ld == LayerDescription (Lattice, thetas)
        lays = [self.__substrate] + [ld[0] for ld in self.__layers]
        for lay in lays[:pos]:
            offset += lay.vectors()[2][2]
        return offset

    @staticmethod
    def __get_BtrBr(BrDt):
        """
        BrDt -> BtrBr
        """
        # see why round in implementation of `Heterostructue.__get_d_xs`
        BtrDt = np.round(BrDt)
        return BtrDt @ inv(BrDt)

    ### OPT METHODS

    def opt(self,
            max_el: int = 6,
            thetas: Optional[List[Optional[List[float]]]] = None
            ) -> Result:
        """
        Minimises strain, and calculates its value.
        Note: definition of strain tensor used here is the same as in
        documentation of `Heterostructure.calc`

        Parameters
        ----------
        max_el : int, optional
            Defines maximum absolute value of the strain tensor element
            The opt calculation is O(`max_el`^2).
            Default: 6

        thetas : List[List[float]|None] | List[float], optional
            Allows to override thetas specified for the layers.
            If specified, it must be equal in length to the number of layers.
            For a given layer, the list represents values of theta to check.
            If None is is passed instead of one of the inner lists, then default
            is not overriden.
            If there are only two layers, the argument can be passed just as a list
            of floats, without the need for nesting.
            All angles are in radians.

        Returns
        -------
        Result
            Result object containing the results of the optimisation.
            For more information see documentation of `Result`
        """
        # TODO: the whole routine is pretty complex;
        #   maybe separate opt numerics into a class "MoireSolver" or stg.
        #   so we don't have to deal with passing XA, XB etc. all the time,
        #   and the mess in Heterostructure

        # Prepare ranges of theta values
        def is_iterable(x):
            try:
                iter(x)
                return True
            except Exception:
                return False

        if thetas is not None:
            if is_iterable(thetas[0]):
                thetas_in = [arg if arg is not None else np.arange(*lay_desc[1])
                             for arg, lay_desc in zip(thetas, self.__layers)]
            elif len(self.__layers) == 1:
                thetas_in = [thetas]
            else:
                raise TypeError("Bad argument thetas")
        else:
            thetas_in = [np.arange(*lay_desc[1]) for lay_desc in self.__layers]

        # Using (1, 1) norm since we are usually interested in minimising
        # |strain_ij| over i, j (where 'strain' is strain tensor)

        config = OptSolverConfig()
        config.max_el = max_el
        config.ord = (1, 1)

        XA, XBs = self.__get_lattice_matrices()
        thetas, ADt = MoireFinder(XA, XBs, thetas_in, config).solve()

        return self.calc(ADt, thetas)

    def __opt_aux(self,
                  ord: Tuple[int, int],
                  max_el: int,
                  thetas_in: List[List[Angle]]) \
            -> Tuple[List[Angle], Matrix2x2]:
        """
        This routine calculates optimal supercell lattice vectors, layers'
        rotation angles and their strain tensors. Here, optimal means
        values that result in L_{1, 1} strain tensor norm to be smallest
        :math:`L_{11}(\epsilon) = \max_{ij} |\epsilon_{ij}\` [1]

        Parameters
        ----------
        ord : (int, int)
            (p, q) for calculating L_{p, q} norm of the strain tensor
        max_el : int
            Maximum absolute value of ADt matrix element
        thetas_in : List[List[float]]
            Must have length equal to the number of layers in the heterostructure.
            Elements are lists or arrays containing possible values of thata for
            a given layer.

        Returns
        -------
        List[float]
            List of optimal theta values corresponding to the Heterostructure
            layers. Length is equal to the number of layers.
        Matrix2x2
            Best ADt matrix found
        """

        # prepare layer matrices
        pass


    @staticmethod
    def __is_ADt_acceptable(ADt: Matrix2x2, XA: Matrix2x2, XBs: List[Matrix2x2], thetas: List[float]) -> bool:
        """
        Checks if given ADt can be a solution

        Parameters
        ----------
        ADt : Matrix2D
        XA : Matrix2D
        XBs : List[Matrix2D]
            corresponds to layers of the heterostructure
        thetas : List[float]
            corresponds to layers of the heterostructure

        Returns
        -------
        bool
        """
        # 1. Is det != 0 ?
        if np.isclose(np.linalg.det(ADt), 0):
            return False

        # 2. Is det in any other basis == 0?
        # This can happen when the solution to the equation in `_moire` is bad and gives
        # e.g. values close to zero; this translates to impossible stretching of Br
        XBrs = [rotate(XB, theta) for XB, theta in zip(XBs, thetas)]
        BrDts = [inv(XBr) @ XA @ ADt for XBr in XBrs]
        BtrDts = [np.round(BrDt) for BrDt in BrDts]
        zero_dets = [x[0][0] * x[1][1] == x[0][1] * x[1][0] for x in BtrDts]
        if any(zero_dets):
            ADt[:, 1] = 0
            return False

        return True

    def _moire(self, dt_As: np.ndarray, XA: Matrix2x2, XB: Matrix2x2, theta: float) -> np.ndarray:
        """
        Calculates qualities of each dt vector in dt_As,
        where the measure of quality is a heuristic of
        how much will the d vector be stretched for given dt
        vector; the higher the quality the better.

        For now as a measure of quality we use -|d - dt|^2

        Parameters
        ----------
        dt_As : np.ndarray, shape (span, span, 2)
            All possible potential dt vetors in A basis
        XA : Matrix2D
        XB : Matrix2D
        theta : float

        Returns
        -------
        qtys : np.ndarray, shape (span, span)
            Quality of corresponding dt vectors
        """
        # T = rotation_matrix(theta) in A basis @ AB = AX @ rotation_matrix(theta) @ XA @ AB
        # T_inv to solve equation [n1, n2] == T @ [m1, m2],
        # where dt_i == n1 a_1 + n2 a_2 == m1 b_1 + m2 b_2
        # this is a 2D matrix, so solving by inv is OK
        T_inv = inv(XB) @ np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ]) @ XA

        solution_vecs = matvecmul(T_inv, dt_As) # dt_B
        qtys = -vecnorm((solution_vecs - np.round(solution_vecs)), 1) / np.sqrt(vecnorm(solution_vecs, 2)) # TODO: better measure
        # TOO self.__calc_strain_tensor_from_ADt(ADt, XBr)
        return qtys

    def _get_dt_As(self, max_el: int) -> np.ndarray:
        """
        Prepares dt_As for __opt_aux calculation.

        Parameters
        ----------
        max_el : int
            Maximum absolute value of ADt matrix element

        Returns
        -------
        dt_As : np.ndarray, shape (span, span, 2)
        """
        # Let's imagine all possible superlattice vectors (dt vectors). At the
        # very least, they need to be able to recreate the substrate lattice.
        # We assume substrate lattice to be constant.
        # Thus every grid point in substrate basis (A) is a valid dt vector
        span_range = np.arange(0, 2 * max_el + 1)
        span_range[(max_el + 1):] -= 2 * max_el + 1
        dt_As = np.transpose(np.meshgrid(span_range, span_range))

        dt_As[0, 0] = [0, 1]  # Hack to remove "[0, 0]" vector
        return dt_As

    @staticmethod
    def _update_opt_res(res: Tuple[List[Angle], float, Matrix2x2, List[Matrix2x2]],
                        thetas: Tuple[Angle, ...],
                        min_qty: float,
                        XDt: np.ndarray,
                        min_st: List[Matrix2x2]
                        ) -> Tuple[List[Angle], float, Matrix2x2, List[Matrix2x2]]:
        """
        Checks if newly calculated result is better than the previous one
        (this usually means that the strain measure `qty` is smaller),
        and if so updates it accordingly

        Parameters
        ----------
        res : List[float], float, Matrix2x2, List[Matrix2x2]
        thetas : Collection[float]
        min_qty : float
        XDt : Matrix2x2
        min_st : List[Matrix2x2]

        Returns
        -------
        List[float]
        float
        Matrix2x2
        List[Matrix2x2]
        """

        # 1. Check for smaller supercell quality function

        if min_qty - ABS_EPSILON > res[1]:
            return res

        thetas = list(thetas)
        new_res = (thetas, min_qty, XDt, min_st)

        # Here it is most efficient to check whether we aren't at the beginning
        # (res[0] is None in that case)
        if (min_qty + ABS_EPSILON < res[1]) or (res[0] is None):
            return new_res

        # 2. If qties are almost equal, choose smaller elementary cell

        old_size = np.abs(np.linalg.det(res[2]))
        new_size = np.abs(np.linalg.det(XDt))

        if new_size < old_size - ABS_EPSILON:
            return new_res
        elif old_size < new_size - ABS_EPSILON:
            return res

        # 3. If even the sizes are equal, return the cell more 'square-y'
        # here it will mean the cell with smaller |max_lattice_vector_element|

        if np.max(np.abs(XDt)) < np.max(np.abs(res[2])):
            return new_res
        return res

    def __get_lattice_matrices(self):
        """
        Calculates basis change vectors between Cartesian basis and
        basis of lattice vectors of lattices in the Heterostructure

        Returns
        -------
        XA : Matrix2x2
        XBs : List[Matrix2x2]
            Length of list is equal to the number of layers
        """
        XA = np.transpose(np.array(self.substrate().vectors())[0:2, 0:2])
        XBs = [np.transpose(np.array(lay_desc[0].vectors())[0:2, 0:2])
               for lay_desc in self.__layers]
        return XA, XBs


def heterostructure() -> Heterostructure:
    """
    Creates a Heterostructure object

    Returns
    -------
    Heterostructure
        a new Heterostructure object
    """
    return Heterostructure()
