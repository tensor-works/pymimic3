import numpy as np
import bisect
from settings import *


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
        index = bisect.bisect_right(CustomBins.lower_bounds, x) - 1
        if one_hot:
            ret = np.zeros((CustomBins.nbins,), dtype=np.int8)
            ret[index] = 1
            return ret
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
