import numpy as np
from numpy import empty

from taunet.fields import FEATURES

from . import log; log = log.getChild(__name__)

#%%-------------------------------------------------------------
# some simple computations

# compute chi^2
def chi_squared(obs, exp):
    """Compute chi squared of variable obs wrt exp (expectation)"""

    chi_squared = 0;
    for i in range(len(obs)):
        if exp[i] != 0:
            chi_squared += (obs[i] - exp[i]) ** 2 / exp[i]
    return chi_squared


#%%-------------------------------------------------------------
# Normalization functions

def StandardScalar(x, mean, std):
    """Standard Scalar function for pre-processing data"""
    
    if std == 0:
        log.info("Standard deviation is zero! Returning nothing :(")
        return;
    else:
        return (x - mean) / std

def getSSNormalize(data, target, savepath='data/normFactors'):
    """Pre-process data using the standard scalar function.
    
    Parameters:
    ----------

    data : array of dimension (FEATURES, nEntries)

    target : array of length (nEntries)

    savepath : str
        optional path to save location of norms

    Returns:
    -------

    norms : list of tuples
        Mean and stddev of each variable collected in a tuple
        of shape (mean, stddev). List is in order of FEATURES. 
        Last element is of the target. 
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
    Use alread found means and stds to re-shape data. 
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
    """Get indices of variable to apply normalization to"""

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
    """Negative log-probability loss function for use with tensorflow."""

    return -model.log_prob(y)

#%% ---------------------------------------------------------
# Get global mean and stddev from mode
def get_global_params(regressor, arr, mode=0):
    """
    Extract global mean and stddev parameters from keras model 

    Parameters:
    ----------
    regressor : tf.keras.model
        model returned from keras
    arr : np.array
        array of events to be passed
    mode : int
        if 0 returns mean and stddev,
        if 1 returns only mean,
        if 2 returns only stddev.
    """

    dist = regressor(arr)
    logits = dist.tensor_distribution.mixture_distribution.logits.numpy()
    probs = tf.nn.softmax(logits[0:,]).numpy() # convert logits to probabilities
    means = dist.tensor_distribution.components_distribution.tensor_distribution.mean().numpy()
    globalmean = np.array(
        [probs[i][0]*means[i][0] + probs[i][1]*means[i][1] 
                    for i in range(len(means))]).flatten()
    if mode==0 or mode==2:
        stddevs = dist.tensor_distribution.components_distribution.tensor_distribution.stddev().numpy()
        globalstd = np.sqrt(np.array(
            [probs[i][0]*(stddevs[i][0]**2 + means[i][0]**2)
            + probs[i][1]*(stddevs[i][1]**2 + means[i][1]**2) 
            - globalmean[i]**2 for i in range(len(means))]).flatten())

    if mode==0:
        return globalmean, globalstd
    elif mode==1:
        return globalmean
    elif mode==2:
        return globalstd
    else:
        raise ValueError("Mode specified is out of range")

def cut_above_below(globalmean, globalstd):
    """
    Obtain vector of bools for events above or below
    abs(gloabalstd/globalmean). Above referes to greater 
    than 1, while below is less than one. 

    Returns:
    -------
    cutabove : array of bools
        abs(gloabalstd/globalmean) >= 1
    cutbelow : array of bools
        abs(gloabalstd/globalmean) < 1
    """
    
    cutabove = (abs(globalstd/globalmean) >= 1).flatten()
    cutbelow = (abs(globalstd/globalmean) < 1).flatten()
    return cutabove, cutbelow

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
