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


@njit(cache=True, nogil=True)
def scalar_power(value, exponent):
    """
    Raise a value to a power as numpy's '**=' with a scalar exponent, which takes a shortcut for 0, 0.5, 1 and 2.

    Parameters
    ----------
    value : float
        The value.
    exponent : float
        The exponent.

    Returns
    -------
    float
        The value to the power of the exponent.
    """
    if exponent == 0.5:
        return np.sqrt(value)
    if exponent == 1.0:
        return value
    if exponent == 2.0:
        return value * value
    if exponent == 0.0:
        return 1.0
    return value**exponent


@njit(cache=True, nogil=True)
def exceedance_over_wave_shape(fp, xp, x, intidx, use_last, factor_second_last, factor_last):
    """
    Calculate 1 - the product over the blocks of the wave shape of the non-exceedance probability, for one slow
    stochastic and a block duration equal to that of the wind. The same operations per element and in the same order
    as ModelWaterSystem.process_wave_shape.

    Parameters
    ----------
    fp : np.ndarray
        The exceedance probability per level of the slow stochastic, with the shape (slow stochastic, rest).
    xp : np.ndarray
        The levels of the slow stochastic.
    x : np.ndarray
        The value of the slow stochastic per time step of each wave shape, with the shape (time step, wave shape).
    intidx : np.ndarray
        The index of the interval of xp that holds each value of x, with the same shape as x.
    use_last : bool
        Whether the last time step is used (the last block lasts at least half a block duration).
    factor_second_last : float
        The exponent of the second last block if the last time step is not used.
    factor_last : float
        The exponent of the last block if the last time step is used.

    Returns
    -------
    np.ndarray
        The exceedance probability over the wave shape, with the shape of fp.
    """
    n_wave, n_rest = fp.shape
    nt = x.shape[0]
    ondkanstot = np.ones((n_wave, n_rest))
    for it in range(nt if use_last else nt - 1):
        for i in range(n_wave):
            j = intidx[it, i]
            frac = (x[it, i] - xp[j]) / (xp[j + 1] - xp[j])
            for k in range(n_rest):
                ovkans = (1 - frac) * fp[j, k] + fp[j + 1, k] * frac
                if ovkans > 1.0:
                    ovkans = 1.0
                blokondkans = 1.0 - ovkans
                if it == 0:
                    blokondkans = np.sqrt(blokondkans)
                elif it == nt - 2:
                    if not use_last:
                        blokondkans = scalar_power(blokondkans, factor_second_last)
                elif it == nt - 1:
                    blokondkans = scalar_power(blokondkans, factor_last)
                ondkanstot[i, k] *= blokondkans
    return 1 - ondkanstot
