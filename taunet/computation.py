import numpy as np

def chi_squared(obs, exp):
    """
    Compute chi squared value of a given dataset
    """
    chi_squared = 0;
    for i in range(len(obs)):
        if exp[i] != 0:
            chi_squared += (obs[i] - exp[i]) ** 2 / exp[i]
    return chi_squared

def StandardScalar(x, mean, std):
    """
    Standard Scalar function for pre-processing data 
    """
    return (x - mean) / std

def SSNormalize(data, vars=range(0,18)):
    """
    Pre-process data using the standard scalar function. 
    Optionally select which variables to scale using the vars argument. 
    Takes a vector 
    """
    for i in vars:
        dat = data[:,i]
        mean = np.mean(dat)
        std = np.std(dat)
        data[:,i] = StandardScalar(dat, mean, std)
    return data

def getVarIndices(features):
    i = 0
    indices = []
    for _feat in features:
        if ('mu' not in _feat) and ('nVtxPU' not in _feat) \
            and ('PanTau_' not in _feat) and ('nTracks' not in _feat):
            indices += [i]
        i = i + 1
    return indices
