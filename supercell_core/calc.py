"""
Small utilities to help with various numeric calculations
"""

from typing import List
import numpy as np

def flatten_rect_array(M : np.ndarray) -> List[float]:
    """
    Converts a matrix (2D np.ndarray) into a flat list.

    Parameters
    ----------
    M : np.ndarray
        must be a rectangular 2D matrix

    Returns
    -------
    List[float]
    """
    return [M[i][j] for i in range(len(M)) for j in range(len(M[0]))]
