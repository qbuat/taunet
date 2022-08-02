import numpy as np
from numpy import empty

from taunet.fields import FEATURES

from . import log; log = log.getChild(__name__)

#%%-------------------------------------------------------------
# some simple computations

# compute chi^2
def chi_squared(obs, exp):
    """
    Compute chi squared of variable obs wrt exp (expectation)
    """
    chi_squared = 0;
    for i in range(len(obs)):
        if exp[i] != 0:
            chi_squared += (obs[i] - exp[i]) ** 2 / exp[i]
    return chi_squared

def logit2prob(logits):
    odds = np.exp(logits)
    probs = odds / (1 + odds)
    return probs

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
    'TauJetsAuxDyn.ptCombined',
    'TauJetsAuxDyn.etaPanTauCellBased',
    'TauJetsAuxDyn.ptTauEnergyScale'
]

#%%-------------------------------------------------------------
import tensorflow as tf

# MDN loss function
def tf_mdn_loss(y, model):
    return -model.log_prob(y)

# Gaussian mixture loss function
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
    K = tf.keras.backend
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]
    
    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)

#%% Find total number of entries in the dataset, for checking if things are working!
## Total number of entries in current dataset: 12606727 ~ 1.3e7
def find_num_entries(path, dataset, nfiles=-1):
    from taunet.database import file_list
    import uproot
    import awkward as ak
    import numpy as np
    numEntries = 0
    numEvents = []
    # extract number of entries from file
    files = file_list(path, dataset)
    for i_f, _file in enumerate(files):
        if nfiles > 0 and i_f > nfiles:
            break
        with uproot.open(_file) as up_file:
            tree = up_file['CollectionTree']
            print('File {} / {} -- entries = {}'.format(i_f, len(files), tree.num_entries))
            numEntries += tree.num_entries
            print('Current total = {}'.format(numEntries))
            # test some stuffs
            a = tree.arrays('EventInfoAux.eventNumber')
            f = a['EventInfoAux.eventNumber']
            numEvents += [f]
    np.save(file='data/numEvents', arr=np.stack(ak.flatten(numEvents).to_numpy()))
    return numEntries
