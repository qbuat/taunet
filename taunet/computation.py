import numpy as np
from numpy import empty

from taunet.fields import FEATURES

from . import log; log = log.getChild(__name__)

#%%-------------------------------------------------------------
# Implement chi^2 test
def chi_squared(obs, exp):
    """
    Compute chi squared of variable obs wrt exp (expectation)
    """
    chi_squared = 0;
    for i in range(len(obs)):
        if exp[i] != 0:
            chi_squared += (obs[i] - exp[i]) ** 2 / exp[i]
    return chi_squared

#%%-------------------------------------------------------------
# Normalization functions

def StandardScalar(x, mean, std):
    """
    Standard Scalar function for pre-processing data 
    """
    if std == 0:
        log.info("Standard deviation is zero! Returning nothing :(")
        return;
    else:
        return (x - mean) / std

def getSSNormalize(data, target, savepath='data/normFactors'):
    """
    Pre-process data using the standard scalar function. 
    """
    norms = []
    for i in range(len(data[1,:])):
        dat = data[:,i]
        mean = np.mean(dat)
        std = np.std(dat)
        norms.append([mean, std])
    mean = np.mean(target)
    std = np.std(target)
    norms.append([mean, std])
    np.save(file=savepath, arr=norms)
    log.info('Saving normalization to {}.npy'.format(savepath))
    return norms

def applySSNormalize(data, norms, vars=[]):
    """
    Use alread found means and stds to re-shape data
    Optionally choose which variables to normalize
    """
    if vars == []:
        vars = range(len(data[0,:]))
    for i in vars:
        data[:,i] = StandardScalar(data[:,i], norms[i][0], norms[i][1])
    return data

def applySSNormalizeTest(data, norms, vars=[]):
    """
    Apply norms to testing data
    """
    if vars == []:
        vars = range(len(data[:,0]))
    for i in vars:
        data[i,:] = StandardScalar(data[i,:], norms[i][0], norms[i][1])
    return data; 

def getVarIndices(features, vars=FEATURES):
    """
    Get indices of variable to apply normalization to
    """
    i = 0
    indices = []
    for _feat in features:
        if _feat in vars:
            indices += [i]
        i = i + 1
    return indices

# sketch function for condor things 
def select_norms(norms, vec):
    tempNorms = []
    for i in vec:
        tempNorms += [norms[i]]
    return tempNorms

# variables to normalize
VARNORM = [
    'TauJetsAuxDyn.mu', 
    'TauJetsAuxDyn.nVtxPU',
    'TauJetsAuxDyn.rho',
    'TauJetsAuxDyn.ClustersMeanCenterLambda',
    'TauJetsAuxDyn.ClustersMeanFirstEngDens',
    'TauJetsAuxDyn.ClustersMeanSecondLambda',
    #'TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.etaPanTauCellBased',
    'TauJetsAuxDyn.ptTauEnergyScale'
]
