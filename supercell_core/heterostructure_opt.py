from typing import List, Iterable
import itertools
from abc import abstractmethod

try:
    import pandas as pd
except ImportError:
    pd = None

from .physics import *
from .result import *
from .calc import *


class OptSolverConfig:
    """
    TODO

    Attributes
    ----------
    ord : (int, int)
        (p, q) for calculating L_{p, q} norm of the strain tensor
    max_el : int
        Maximum absolute value of ADt matrix element
    log : bool
        Whether to save a log or not. Setting to `True` requires `pandas`
    """
    ord: Tuple[int, int]
    max_el: int
    log: bool  # log really ought to be a pandas.DataFrame

    def __init__(self):
        # Some default values
        self.ord = (1, 1)
        self.max_el = 10
        self.log = False


class OptSolver:
    """
    TODO
    """
    XA: Matrix2x2
    XBs: List[Matrix2x2]
    thetas: List[np.ndarray]  # List of arrays of floats, corresponding to XBs
    config: OptSolverConfig

    def __init__(self,
                 XA: Matrix2x2,
                 XBs: List[Matrix2x2],
                 thetas: List[np.ndarray],
                 config: OptSolverConfig):
        self.XA = XA
        self.XBs = XBs
        self.thetas = thetas
        self.config = config

    def _calculate_strain_tensor(self, ADt: Matrix2x2, XBr: Matrix2x2) -> Matrix2x2:
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
        BrDt = inv(XBr) @ self.XA @ ADt
        BtrDt = np.round(BrDt)
        BtrBr = BtrDt @ inv(BrDt)
        return BtrBr - np.identity(2)

    @staticmethod
    def _get_strain_tensor_wiki(XXt: Matrix2x2) -> np.ndarray:
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

    @abstractmethod
    def solve(self) -> Tuple[List[float], Matrix2x2]:
        pass


class StrainOptimisator(OptSolver):
    """
    TODO: copy from supercell_core 0.0.6
    """
    def solve(self) -> Result:
        pass


class MoireFinder(OptSolver):
    """
    TODO
    """

    def get_result(self) -> Tuple[List[float], Matrix2x2]:
        """
        TODO
        Returns:
        ADt, thetas
        """
        thetas, min_qty, XDt, strain_tensors = self.res
        ADt = inv(self.XA) @ XDt
        return thetas, ADt

    def _update_opt_res(self,
                        thetas: Tuple[Angle, ...],
                        ADt: np.ndarray,
                        min_st: List[Matrix2x2]
                        ) -> None:
        """
        Checks if newly calculated result is better than the previous one
        (this usually means that the strain measure `qty` is smaller),
        and if so updates it accordingly

        Parameters
        ----------
        thetas : Collection[float]
        ADt : Matrix2x2
        min_st : List[Matrix2x2]

        Returns
        -------
        None
        """
        min_qty = sum([matnorm(st, *self.config.ord) for st in min_st])
        XDt = self.XA @ ADt

        # 1. Check for smaller supercell quality function
        if min_qty - ABS_EPSILON > self.res[1]:
            return

        thetas = list(thetas)
        new_res = (thetas, min_qty, XDt, min_st)

        # Here it is most efficient to check whether we aren't at the beginning
        # (res[0] is None in that case)
        if (min_qty + ABS_EPSILON < self.res[1]) or (self.res[0] is None):
            self.res = new_res

        # 2. If qties are almost equal, choose smaller elementary cell
        old_size = np.abs(np.linalg.det(self.res[2]))
        new_size = np.abs(np.linalg.det(XDt))

        if new_size < old_size - ABS_EPSILON:
            self.res = new_res
        elif old_size < new_size - ABS_EPSILON:
            return

        # 3. If even the sizes are equal, return the cell more 'square-y'
        # here it will mean the cell with smaller |max_lattice_vector_element|
        if np.max(np.abs(XDt)) < np.max(np.abs(self.res[2])):
            self.res = new_res

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Prepare all possible dt vectors (in A basis)
        self.dt_As = np.array([])
        self.prepare_dt_As()

        # Dummy start value for classic O(n) find min algorithm, we want to find
        # values for which sum of `quality_fun`(strain tensor) is minimal
        # (thetas, min_qty, XDt, strain_tensors)
        self.res: Tuple[List[Angle], float, Matrix2x2, List[Matrix2x2]] = \
            ([], np.inf, np.identity(2), [])

    def prepare_dt_As(self) -> None:
        """
        Prepares dt_As for the calculation.

        Returns
        -------
        None
        """
        # Let's imagine all possible superlattice vectors (dt vectors). At the
        # very least, they need to be able to recreate the substrate lattice.
        # We assume substrate lattice to be constant.
        # Thus every grid point in substrate basis (A) is a valid dt vector
        span_range = np.arange(0, 2 * self.config.max_el + 1)
        span_range[(self.config.max_el + 1):] -= 2 * self.config.max_el + 1
        self.dt_As = np.transpose(np.meshgrid(span_range, span_range))
        self.dt_As[0, 0] = [0, 1]  # Hack to remove "[0, 0]" vector

    def solve(self) -> Tuple[List[float], Matrix2x2]:

        # Precalculate qualities of solutions here, since this problem is independent for
        # each layer; the final solution depends on a specific combination of thetas
        # but to calculate how much stretched the Br basis will be for each possible dt
        # vector will be we only need to know the Br (theta + XB)
        # Complexity: O(thetas_len * no. layers * max_el ** 2)
        memo = {}
        for i, (theta_, XB) in enumerate(zip(self.thetas, self.XBs)):
            for theta in theta_:
                qtys = self._moire(XB, theta)
                memo[float(theta), i] = qtys

        # We need to have this loop over all possible combinations since
        # ADt is one for the whole structure and must minimise strains in
        # all layers simulatneously
        # Complexity: O(thetas_len ** no. layers * max_el ** 2)
        # Could probably use a few tricks to get it down to stg like O(thetas_len ** no. layers)
        # or at least O(thetas_len ** no. layers * max_el)
        for theta_comb in itertools.product(*self.thetas):
            qtyss = [memo[float(theta), i] for i, theta in enumerate(theta_comb)]
            qtys_total = sum(qtyss)

            ADt = np.zeros((2, 2))
            # This loop is in practice O(1) since the number of vectors giving singular ADt
            # is O(max_el) and then, if for one such vector the qty != 0 then it's very
            # unprobable that the second one will happen to be parallel
            while all(ADt[:, 0] == 0) or all(ADt[:, 1] == 0):
                argmax_indices = np.unravel_index(qtys_total.argmax(), qtys_total.shape)
                qtys_total[argmax_indices] = -np.inf  # removes this vector from "search pool"
                vec = self.dt_As[argmax_indices]

                if all(ADt[:, 0] == 0):
                    ADt[:, 0] = vec
                else:
                    ADt[:, 1] = vec
                    if not self._is_ADt_acceptable(ADt, theta_comb):
                        # Back off from adding this vec
                        ADt[:, 1] = 0
                    # Note: we always use the best candidate vec and only consider
                    # the second one (ADt[:. 1]) suspicious because if it is in fact bad
                    # the it must be almost-parallel or parallel to the first one, so
                    # it doesn't matter much which one we pick to stay

            # Check if result for this angle combination is better than for the previous one
            # Here, we use the exact quality measure, which is sum of norms of strain tensors
            XBrs = [rotate(XB, theta) for (XB, theta) in zip(self.XBs, theta_comb)]
            sts = [self._calculate_strain_tensor(ADt, XBr) for XBr in XBrs]
            # TODO: move to update_opt_res

            self._update_opt_res(theta_comb, ADt, sts)

        return self.get_result()

    def _moire(self, XB: Matrix2x2, theta: float) -> np.ndarray:
        """
        Calculates qualities of each dt vector in dt_As,
        where the measure of quality is a heuristic of
        how much will the d vector be stretched for given dt
        vector; the higher the quality the better.

        Parameters
        ----------
        XB : Matrix2D
        theta : float

        Returns
        -------
        qtys : np.ndarray, shape (span, span)
            Quality of corresponding dt vectors
            Calculated as ||d - dt||_1 / |d|
        """
        # T = rotation_matrix(theta) in A basis @ AB = AX @ rotation_matrix(theta) @ XA @ AB
        # T_inv to solve equation [n1, n2] == T @ [m1, m2],
        # where dt_i == n1 a_1 + n2 a_2 == m1 b_1 + m2 b_2
        # this is a 2D matrix, so solving by inv is OK
        T_inv = inv(XB) @ np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ]) @ self.XA

        solution_vecs = matvecmul(T_inv, self.dt_As)  # dt_B
        qtys = -vecnorm((solution_vecs - np.round(solution_vecs)), 1) / np.sqrt(
            vecnorm(solution_vecs, 2))  # TODO: better measure
        # TOO self.__calc_strain_tensor_from_ADt(ADt, XBr)
        return qtys

    def _is_ADt_acceptable(self, ADt: Matrix2x2, thetas: Iterable[float]) -> bool:
        """
        Checks if given ADt can be a solution

        Parameters
        ----------
        ADt : Matrix2D
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
        XBrs = [rotate(XB, theta) for XB, theta in zip(self.XBs, thetas)]
        BrDts = [inv(XBr) @ self.XA @ ADt for XBr in XBrs]
        BtrDts = [np.round(BrDt) for BrDt in BrDts]
        zero_dets = [x[0][0] * x[1][1] == x[0][1] * x[1][0] for x in BtrDts]
        if any(zero_dets):
            ADt[:, 1] = 0
            return False

        return True
