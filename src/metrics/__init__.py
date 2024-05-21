import numpy as np
import bisect
from settings import *


class CustomBins:
    inf = 1e18
    bins = LOS_BINS
    nbins = len(bins)
    means = [
        11.450379, 35.070846, 59.206531, 83.382723, 107.487817, 131.579534, 155.643957, 179.660558,
        254.306624, 585.325890
    ]

    # Precompute scaled bin boundaries
    scaled_bins = [(a * 24.0, b * 24.0) for a, b in bins]
    lower_bounds = [a * 24.0 for a, b in bins]

    @staticmethod
    def get_bin_custom(x, one_hot=False):
        index = bisect.bisect_right(CustomBins.lower_bounds, x) - 1
        if one_hot:
            ret = np.zeros((CustomBins.nbins,))
            ret[index] = 1
            return ret
        return index
