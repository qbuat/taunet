import numpy as np

from taunet.fields import FEATURES

from . import log; log = log.getChild(__name__)

def chi_squared(obs, exp):
    """
    Compute chi squared of variable obs wrt exp (expectation)
    """
    chi_squared = 0;
    for i in range(len(obs)):
        if exp[i] != 0:
            chi_squared += (obs[i] - exp[i]) ** 2 / exp[i]
        else:
            log.info('Potential error: expected value was zero!')
    return chi_squared

def StandardScalar(x, mean, std):
    """
    Standard Scalar function for pre-processing data 
    """
    if std == 0:
        log.info("Standard deviation is zero! Returning nothing :(")
        return;
    else:
        return (x - mean) / std

def getSSNormalize(data, target):
    """
    Pre-process data using the standard scalar function. 
    Optionally select which variables to scale using the vars argument. 
    Takes a vector 
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
    np.save('data/normFactors', norms)
    return norms

def applySSNormalize(data, norms, vars=[]):
    """
    """
    if vars == []:
        vars = range(len(data[0,:]))
    for i in vars:
        data[:,i] = StandardScalar(data[:,i], norms[i][0], norms[i][1])
    return data; 

def applySSNormalizeTest(data, norms):
    """
    """
    for i in range(len(data)):
        data = StandardScalar(data, norms[i][0], norms[i][1])
    return data; 

# indices of variables to normalize
# Note: variable 12 doesn't really need to be normalized
# Update function getVarIndices to make this clearer... 
NormVarIndices = [0, 1, 2, 3, 4, 5, 9, 12, len(FEATURES)-1]

# variables to normalize
VARNORM = [
    'TauJetsAuxDyn.mu', 
    'TauJetsAuxDyn.nVtxPU',
    'TauJetsAuxDyn.rho',
    'TauJetsAuxDyn.ClustersMeanCenterLambda',
    'TauJetsAuxDyn.ClustersMeanFirstEngDens',
    'TauJetsAuxDyn.ClustersMeanSecondLambda',
    'TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.etaPanTauCellBased',
    'TauJetsAuxDyn.ptTauEnergyScale'
]

def getVarIndices(features, vars=FEATURES):
    i = 0
    indices = []
    for _feat in features:
        if _feat in vars:
            indices += [i]
        i = i + 1
    return indices

#-----------------------------------------------------------
# Loss function for MDN
# Function from https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e
import keras.backend as K


def gaussian_nll(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 
    
    """
    
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]
    
    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)