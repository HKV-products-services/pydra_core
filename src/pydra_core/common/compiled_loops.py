"""
Compiled loops for the hottest parts of the HBN calculation with model uncertainty.
"""

import numpy as np

from numba import njit


@njit(cache=True, nogil=True)
def sum_rows_per_class(row_class, indptr, indices, data, n_classes, n_columns):
    """
    Sum the rows of a sparse matrix per class, in ascending row order and starting from zero, as the product of a
    0/1 class matrix with the rows.

    Parameters
    ----------
    row_class : np.ndarray
        The class of each row.
    indptr : np.ndarray
        The row pointers of the csr matrix.
    indices : np.ndarray
        The column indices of the csr matrix.
    data : np.ndarray
        The nonzero values of the csr matrix.
    n_classes : int
        The number of classes.
    n_columns : int
        The number of columns of the csr matrix.

    Returns
    -------
    np.ndarray
        The sum of the rows per class, with the shape (n_classes, n_columns).
    """
    total = np.zeros((n_classes, n_columns))
    for i in range(row_class.shape[0]):
        k = row_class[i]
        for j in range(indptr[i], indptr[i + 1]):
            total[k, indices[j]] += data[j]
    return total
