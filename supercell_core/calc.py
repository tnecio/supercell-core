"""
Small utilities to help with various numeric calculations,
especially with operations on arrays of matrices
"""

from typing import List
import numpy as np

ABS_EPSILON = 1e-7


def rotate(a: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotates vectors stored in 2D matrices columns in array `a` by angle `theta`
    counterclockwise

    Parameters
    ----------
    a : np.ndarray
    theta : float

    Returns
    -------
    np.ndarray
    """
    return np.array(((np.cos(theta), -np.sin(theta)),
                     (np.sin(theta), np.cos(theta)))) @ a


def matvecmul(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Acts with 2D matrix m on 2D vectors stored in array `v`

    Parameters
    ----------
    m : np.ndarray, shape (..., 2, 2)
    v : np.ndarray, shape (..., 2)

    Returns
    -------
    np.ndarray, shape = v.shape
    """
    res = np.empty(v.shape)
    res[..., 0] = m[..., 0, 0] * v[..., 0] + m[..., 0, 1] * v[..., 1]
    res[..., 1] = m[..., 1, 0] * v[..., 0] + m[..., 1, 1] * v[..., 1]
    return res


def matmul(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """
    Calculates matrix multiplication of 2D matrices stored in arrays `m1` and `m2`

    Parameters
    ----------
    m1 : np.ndarray, shape(..., 2, 2)
    m2 : np.ndarray, shape(..., 2, 2)

    Returns
    -------
    np.ndarray, shape(..., 2, 2)
    """
    return m1 @ m2


def inv(m: np.ndarray) -> np.ndarray:
    """
    Inverts 2x2 matrices. Why not use np.linalg.inv? To avoid LinAlgError when
    just one of the matrices is singular, and fill it with nans instead
    These are all 2x2 matrices so no harm in inverting them

    Parameters
    ----------
    m : np.ndarray, shape (..., 2, 2)
        Array of matrices

    Returns
    -------
    np.ndarray
    """
    det_m = np.linalg.det(m)
    res = np.moveaxis(
        (1.0 / det_m) * np.array(((m[..., 1, 1], -m[..., 0, 1]),
                                  (-m[..., 1, 0], m[..., 0, 0]))),
        (0, 1),
        (-2, -1))
    res[np.abs(det_m) < 1e-8] = np.nan
    return res


def matnorm(m: np.ndarray, p: float, q: float) -> np.ndarray:
    """
    Calculates L_{p, q} matrix norm for every 2D matrix in array `m`.
    This norm is defined as :math:`(\sum_j (\sum_i |m_{ij}|^p)^{q/p})^{1/q}` [1]

    Parameters
    ----------
    m : np.ndarray, shape (..., 2, 2)
    p : float >= 1
    q : float >= 1

    Returns
    -------
    np.ndarray

    Notes
    -----

    Useful norms for strain calculations:
    Frobenius norm: ord_p = ord_q = 2
    Max norm: ord_p = ord_q = np.inf

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Matrix_norm#L2,1_and_Lp,q_norms
    """
    return ((np.abs(m[..., 0, 0]) ** p + np.abs(m[..., 0, 1]) ** p) ** (q / p)
            + (np.abs(m[..., 1, 0]) ** p + np.abs(m[..., 1, 1]) ** p) ** (q / p)) \
           ** (1 / q)
