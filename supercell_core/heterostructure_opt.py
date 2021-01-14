from typing import List, Iterable, Dict, Any
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

    TODO documentation
    """

    def __init__(self,
                 XA: Matrix2x2,
                 XBs: List[Matrix2x2],
                 thetas: List[np.ndarray],
                 config: OptSolverConfig):
        self.XA = XA
        self.XBs = XBs
        self.thetas = thetas  # List of arrays of floats, corresponding to XBs
        self.config = config
        if self.config.log:
            self.log = {}
            columns = [
                *["theta_{}".format(i) for i in range(len(thetas))],
                "max_strain",
                "supercell_size",
                "M_11", "M_12", "M_21", "M_22",
                *["N{}_{}{}".format(n+1, i+1, j+1) for n in range(len(XBs)) for i in range(2) for j in range(2)],
                "supercell_vectors_11", "supercell_vectors_12",
                "supercell_vectors_21", "supercell_vectors_22",
                *["strain_tensor_layer_{}_{}{}".format(k + 1, i + 1, j + 1)
                  for k in range(len(thetas))
                  for i in range(2) for j in range(2)]
            ]
            for column in columns:
                self.log[column] = []
            if pd is None:
                print("Pandas not installed! Returning a dictionary instead (log)") # Jezu jak źle

        # Prepare all possible dt vectors (in A basis)
        self.dt_As = np.array([])
        self.prepare_dt_As()

        # Dummy start value for classic O(n) find min algorithm, we want to find
        # values for which sum of `quality_fun`(strain tensor) is minimal
        # (thetas, min_qty, XDt, strain_tensors)
        self.res: Tuple[List[Angle], float, Matrix2x2, List[Matrix2x2]] = \
            ([], np.inf, np.identity(2), [])

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

    def _update_opt_res(self,
                        thetas: Tuple[Angle, ...],
                        ADt: np.ndarray,
                        sts: List[Matrix2x2],
                        ) -> None:
        """
        Checks if newly calculated result is better than the previous one
        (this usually means that the strain measure `qty` is smaller),
        and if so updates it accordingly

        Parameters
        ----------
        thetas : Collection[float]
        ADt : Matrix2x2
        sts : List[Matrix2x2]

        Returns
        -------
        None
        """
        min_qty = sum([matnorm(st, *self.config.ord) for st in sts])
        XDt = self.XA @ ADt

        # 0. Update log if necessary (TODO: tests, doc, make a warning if log=True and no pandas)
        new_res = (list(thetas), min_qty, XDt, sts)
        new_size = np.abs(np.linalg.det(XDt))
        if self.config.log:
            for i, theta in enumerate(thetas):
                self.log["theta_{}".format(i)].append(theta)
            self.log["max_strain"].append(min_qty)
            self.log["supercell_size"].append(new_size)
            for i in range(2):
                for j in range(2):
                    self.log["M_{}{}".format(i + 1, j + 1)].append(ADt[i, j])
                    self.log["supercell_vectors_{}{}".format(i + 1, j + 1)].append(XDt[i, j])
                    for k, st in enumerate(sts):
                        self.log["strain_tensor_layer_{}_{}{}".format(k + 1, i + 1, j + 1)].append(
                            sts[k][i, j]
                        )

        # 1. Check for smaller supercell quality function
        if min_qty - ABS_EPSILON > self.res[1]:
            return

        # Here it is most efficient to check whether we aren't at the beginning
        # (res[0] is None in that case)
        if (min_qty + ABS_EPSILON < self.res[1]) or (self.res[0] is None):
            self.res = new_res

        # 2. If qties are almost equal, choose smaller elementary cell
        old_size = np.abs(np.linalg.det(self.res[2]))
        if new_size < old_size - ABS_EPSILON:
            self.res = new_res
        elif old_size < new_size - ABS_EPSILON:
            return

        # 3. If even the sizes are equal, return the cell more 'square-y'
        # here it will mean the cell with smaller |max_lattice_vector_element|
        if np.max(np.abs(XDt)) < np.max(np.abs(self.res[2])):
            self.res = new_res

    def get_result(self) -> Tuple[List[float], Matrix2x2, Dict[str, Any]]:
        """
        TODO
        Returns:
        ADt, thetas
        """
        thetas, min_qty, XDt, strain_tensors = self.res
        ADt = inv(self.XA) @ XDt

        additional = {}
        if self.config.log:
            if pd is not None:
                self.log = pd.DataFrame(self.log)
            additional["log"] = self.log
        return thetas, ADt, additional

    @abstractmethod
    def solve(self) -> Tuple[List[float], Matrix2x2, Dict[str, Any]]:
        """
        This routine calculates optimal supercell lattice vectors, layers'
        rotation angles and their strain tensors. Here, optimal means
        values that result in L_{1, 1} strain tensor norm to be smallest
        :math:`L_{11}(\epsilon) = \max_{ij} |\epsilon_{ij}\` [1].
        You can change used norm by setting appropriate value of `ord`
        in object passed to `OptSolver.config`.

        Returns
        -------
        thetas: List[float]
            List of optimal theta values corresponding to the Heterostructure
            layers. Length is equal to the number of layers.
        ADt: Matrix2x2
            Best ADt matrix found
        """
        pass


class StrainOptimisator(OptSolver):
    """
    TODO: copy from supercell_core 0.0.6
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt_xs = matvecmul(self.XA, self.dt_As)
        span = self.dt_As.shape[0]
        self.ADts = np.empty((span, span, span, span, 2, 2))
        self.ADts[..., 0] = self.dt_As[np.newaxis, np.newaxis, ...]
        self.ADts[..., 1] = self.dt_As[..., np.newaxis, np.newaxis, :]

    def _get_strain_tensors_opt(self,
                                theta_comb: Tuple[float]) -> List[np.ndarray]:
        """
        Calculates strain tensor for `solve`.

        Parameters
        ----------
        theta_comb : Tuple[float]
            Length must correspond to self.XBs

        Returns
        -------
        List[np.ndarray with shape (span, span, span, span, 2, 2)]

        Notes
        -----
        Definiton of strain tensor here is the same as in documentation
        for `calc`
        """
        return [self._calculate_strain_tensor(self.ADts, rotate(XB, theta))
                for XB, theta in zip(self.XBs, theta_comb)]

    def _get_d_xs(self,
                  XB: np.ndarray,
                  theta: Angle) -> np.ndarray:
        """
        Returns an array of `d` vectors in Cartesian basis

        Parameters
        ----------
        XB : Matrix 2x2
        theta : float

        Returns
        -------
        np.ndarray, shape (..., 2)
        """
        XBr = rotate(XB, theta)
        BrA = inv(XBr) @ self.XA
        dt_Brs = matvecmul(BrA, self.dt_As)

        # Here we use the fact that the supercell must "stretch" lattice vectors
        # of constituent layers so that they superlattice vectors are linear
        # _integer_ combinations of any one layers' lattice vectors.
        dt_Btrs = np.round(dt_Brs)
        d_Brs = dt_Btrs
        return matvecmul(XBr, d_Brs)

    def solve(self) -> Tuple[List[float], Matrix2x2, Dict[str, Any]]:
        # embarrasingly parallel, but Python GIL makes this irrelevant
        for theta_comb in itertools.product(*self.thetas):
            strain_tensors = self._get_strain_tensors_opt(theta_comb)

            # qty – array which contains norms of the strain tensors
            qty = sum([matnorm(st, *self.config.ord) for st in strain_tensors])

            # if qty is NaN it means that we somehow ended up with linear
            # dependence; in the limit strain would go to infinity
            qty[np.isnan(qty)] = np.inf

            linalg_fix = True # sometimes we get a linearly dependent ADt anyway, TODO: why
            while linalg_fix:
                argmin_indices = np.unravel_index(qty.argmin(), qty.shape)
                min_sts = [st[argmin_indices] for st in strain_tensors]
                ADt = np.stack((self.dt_As[argmin_indices[0], argmin_indices[2]],
                                self.dt_As[argmin_indices[1], argmin_indices[3]]))
                if np.isclose(np.linalg.det(ADt), 0):
                    qty[argmin_indices] = np.inf
                else:
                    linalg_fix = False

            # let's check if the best values for this combination of theta vals
            # (theta_lay) are better than those we already have
            self._update_opt_res(theta_comb, ADt, min_sts)

        return self.get_result()


class MoireFinder(OptSolver):
    """
    TODO
    """

    def solve(self) -> Tuple[List[float], Matrix2x2, Dict[str, Any]]:
        # Precalculate qualities of solutions here, since this problem is independent for
        # each layer; the final solution depends on a specific combination of thetas
        # but to calculate how much stretched the Br basis will be for each possible dt
        # vector will be we only need to know the Br (theta + XB)
        # Complexity: O(thetas_len * no. layers * max_el ** 2)
        memo = {}
        for i, (theta_, XB) in enumerate(zip(self.thetas, self.XBs)):
            for theta in theta_:
                qtys = self._fast(XB, theta)
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

    def _fast(self, XB: Matrix2x2, theta: float) -> np.ndarray:
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
        # This can happen when the solution to the equation in `_fast` is bad and gives
        # e.g. values close to zero; this translates to impossible stretching of Br
        XBrs = [rotate(XB, theta) for XB, theta in zip(self.XBs, thetas)]
        BrDts = [inv(XBr) @ self.XA @ ADt for XBr in XBrs]
        BtrDts = [np.round(BrDt) for BrDt in BrDts]
        zero_dets = [x[0][0] * x[1][1] == x[0][1] * x[1][0] for x in BtrDts]
        if any(zero_dets):
            ADt[:, 1] = 0
            return False

        return True
