import numpy as np
from settings import *


def is_iterable(obj):
    """
    Check if an object is iterable.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bool
        True if the object is iterable, False otherwise.
    """
    return hasattr(obj, '__iter__')


class CustomBins:
    inf = 1e18
    bins = [(-np.inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14),
            (14, np.inf)]
    means = [
        11.450379, 35.070846, 59.206531, 83.382723, 107.487817, 131.579534, 155.643957, 179.660558,
        254.306624, 585.325890
    ]
    nbins = len(bins)
    # Precompute scaled bin boundaries
    scaled_bins = [(a * 24.0, b * 24.0) for a, b in bins]
    lower_bounds = [a * 24.0 for a, b in bins]

    @staticmethod
    def get_bin_custom(x, one_hot=False):
        index = np.digitize(x, CustomBins.lower_bounds) - 1
        if one_hot:
            if is_iterable(index):
                ret = np.zeros((x.size, CustomBins.nbins), dtype=np.int8)
                ret[np.arange(x.size), index] = 1
            else:
                ret = np.zeros((CustomBins.nbins,), dtype=np.int8)
                ret[index] = 1
            return ret.squeeze()
        return np.int8(index)


class LogBins:
    nbins = 10
    means = [
        0.611848, 2.587614, 6.977417, 16.465430, 37.053745, 81.816438, 182.303159, 393.334856,
        810.964040, 1715.702848
    ]

    def get_bin_log(x, nbins=10, one_hot=False):
        binid = int(np.log(x + 1) / 8.0 * nbins)
        if binid < 0:
            binid = 0
        if binid >= nbins:
            binid = nbins - 1

        if one_hot:
            ret = np.zeros((LogBins.nbins,), dtype=np.int8)
            ret[binid] = 1
            return ret
        return binid
