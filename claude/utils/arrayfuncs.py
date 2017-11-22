"""
Ancillary functions on numpy arrays

"""
import numpy as np


def unique_rows(ar: np.ndarray, return_counts: bool= False):
    """unique_rows
    Returns the unique rows of a multidimensional array.
    :param ar: array_like
        Input array.
    :param return_counts:
        If True, also return the number of times a unique row appears in `ar`.
    :return unique_x: ndarray
        The sorted unique rows.
    :return counts: ndarray, optional
        The number of times each unique row appears in `ar`.
    """
    if ar.ndim == 1:
        return np.unique(ar, return_counts=return_counts)
    else:
        dtype = np.dtype((np.void, ar.dtype.itemsize * ar.shape[1]))
        y = np.ascontiguousarray(ar).view(dtype)
        _, idx, counts = np.unique(y, return_index=True, return_counts=True)
        unique_x = ar[idx, :]
        if return_counts:
            return unique_x, counts
        else:
            return unique_x


def glue_xyz(x, y, z):
    """glue_xyz
    Returns an array with `x`, `y` and `z` as columns
    :param x: array_like
    :param y:
    :param z:
    :return xyz: ndarray
        Array with `x`, `y` and `z` as columns.
    """
    if y is not None and z is not None:
        xyz = np.column_stack((x, y, z))
    elif y is not None and z is None and x.shape[1] == 2:
        xyz = np.column_stack((x, y))
    else:
        xyz = x
    return xyz


def get_frequencies(x):
    values, counts = unique_rows(x, return_counts=True)
    if x.ndim == 1:
        frequencies = counts
    else:
        ndim = x.shape[1]
        symbols = [{v: j for j, v in enumerate(np.unique(x[:, i]))}
                   for i in range(ndim)]
        shape = [len(s) for s in symbols]
        frequencies = np.zeros(shape)
        for value, count in zip(values, counts):
            idx = tuple(symbols[i][v] for i, v in enumerate(value))
            frequencies[idx] = count
    frequencies = frequencies / frequencies.sum()
    return frequencies
