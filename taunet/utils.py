import numpy as np

def get_quantile_width(arr, cl=0.68):
    """
    """
    q1 = (1. - cl) / 2.
    q2 = 1. - q1
    y = np.quantile(arr, [q1, q2])
    width = (y[1] - y[0]) / 2.
    return width


def response_curve(res, var, bins):
    """
    """
    _bin_centers = []
    _bin_errors = []
    _means = []
    _resol = []
    for _bin in bins:
        a = res[(var > _bin[0]) & (var < _bin[1])]
        # mean = 
        # mean, std = norm.fit(a)
        # y = np.quantile(a, [q1, q2])
        # res_68 = (y[1] - y[0]) / 2.
        # print (_bin, len(a), a.mean(), mean, a.std(), std, (y[1] - y[0]) / 2., np.quantile(a-1, 0.95))
        _means += [np.mean(a)]
        _resol += [get_quantile_width(a)]
        _bin_centers += [_bin[0] + (_bin[1] - _bin[0]) / 2]
        _bin_errors += [(_bin[1] - _bin[0]) / 2]
    return np.array(_bin_centers), np.array(_bin_errors), np.array(_means), np.array(_resol)
