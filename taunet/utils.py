"""Utility funtions to be used in both fitting and plotting. 

Contains function for loss, chi^2 calculation, standardization, 
global mean and stddev calculation, and copying plots to cernbox. 

Authors : Miles Cochran-Branson, Quentin Buat
Date : Summer 2022
"""

import numpy as np
import os
import subprocess

import tensorflow as tf

from taunet.fields import FEATURES

from . import log; log = log.getChild(__name__)


#%%----------------------------------------------------------
# MDN loss function
def tf_mdn_loss(y, model):
    """Negative log-probability loss function for use with tensorflow."""

    return -model.log_prob(y)

#%%----------------------------------------------------------
# compute chi^2
def chi_squared(obs, exp):
    """Compute chi squared of variable obs wrt exp (expectation)
    
    Paramaters:
    ----------

    obs : vector{float}
        Vector of observastions

    exp : vector{float}
        Vector of expected data
    """

    chi_squared = 0;
    for i in range(len(obs)):
        if exp[i] != 0:
            chi_squared += (obs[i] - exp[i]) ** 2 / exp[i]
    return chi_squared

#% ---------------------------------------------------------
# Functions for applying standardarization

def StandardScalar(x, mean, std):
    """Standard Scalar function for pre-processing data
    
    Parameters:
    ----------

    x : vector 
        Data to be transformed

    mean : float
        Mean of data

    std : float
        Standard deviation of data

    Returns:
    -------

    (x - mean) / std
    """
    
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

    Parameters:
    ----------

    data : ntuple 
        All testing data for plotting

    norms : 
        Means and standard deviations found from `getSSNormalize`
    """

    if vars == []:
        vars = range(len(data[0,:]))
    for i in vars:
        data[:,i] = StandardScalar(data[:,i], norms[i][0], norms[i][1])
    return data

def applySSNormalizeTest(data, norms, vars=[]):
    """Apply norms to testing data. See `applySSNormalize` for more information."""

    if vars == []:
        vars = range(len(data[:,0]))
    for i in vars:
        data[i,:] = StandardScalar(data[i,:], norms[i][0], norms[i][1])
    return data; 

def getVarIndices(features, vars=FEATURES):
    """Get indices of variable to apply normalization to
    
    features : list of str
        All variable names used 

    vars : list of str
        Variables to be normalized
    """

    i = 0
    indices = []
    for _feat in features:
        if _feat in vars:
            indices += [i]
        i = i + 1
    return indices

#% ---------------------------------------------------------
# Functions for response on resolution curves 

def makeBins(bmin, bmax, nbins):
    """Make tuples to extract data between bins. 

    Parameters:
    ----------

    bmin : float
        Smallest value of bin

    bmax : float
        Largest value of bins

    nbins : int
        Number of desired bins

    Returns: 
    -------

    List of tuples containing bins
    """

    returnBins = []
    stepsize = (bmax - bmin) / nbins
    for i in range(nbins):
        returnBins.append((bmin + i*stepsize, bmin + (i+1)*stepsize))
    return returnBins

def get_quantile_width(arr, cl=0.68):
    """Get width of `arr` at `cl`%. Default is 68% CL"""

    q1 = (1. - cl) / 2.
    q2 = 1. - q1
    y = np.quantile(arr, [q1, q2])
    width = (y[1] - y[0]) / 2.
    return width

def response_curve(res, var, bins, cl=0.68):
    """Prepare data fot plotting the response and resolution curve

    Parameters:
    ----------

    res : vector
        Data to be prepared

    var : vector of floats
        Variable to be plotted against

    bins : int
        Number of bins to be used in computation

    cl=0.68 : float
        Confidence level to be used. Default is 68%

    Returns: 
    -------

    - bin centers : center value of each bin

    - bin errors : distance to nearest bin edge

    - means : mean of distribution within each bin

    - means statistical error : stat error of distribution within each bin

    - resolution : quantile width at cl% of distribution
    """

    _bin_centers = []
    _bin_errors = []
    _means = []
    _mean_stat_err = []
    _resol = []
    for _bin in bins:
        a = res[(var > _bin[0]) & (var < _bin[1])]
        if len(a) == 0:
            log.info('Bin was empty! Moving on to next bin')
            continue
        _means += [np.mean(a)]
        _mean_stat_err += [np.std(a, ddof=1) / np.sqrt(np.size(a))]
        _resol += [get_quantile_width(a, cl=cl)]
        _bin_centers += [_bin[0] + (_bin[1] - _bin[0]) / 2]
        _bin_errors += [(_bin[1] - _bin[0]) / 2]
    return np.array(_bin_centers), np.array(_bin_errors), np.array(_means), np.array(_mean_stat_err), np.array(_resol)

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
        if 2 returns only stddev, 
        if 3 returns probs, means, and stddevs for each component
            distribution. 
    """

    dist = regressor(arr)
    logits = dist.tensor_distribution.mixture_distribution.logits.numpy()
    probs = tf.nn.softmax(logits[0:,]).numpy() # convert logits to probabilities
    means = dist.tensor_distribution.components_distribution.tensor_distribution.mean().numpy()
    means = np.reshape(means, (len(means), len(means[0])))
    # compute global mean: \mu_g = \sum_{i=1}^k \pi_i \sigma_i
    globalmean = 0
    for i in range(len(means[0])):
        globalmean += np.multiply(probs[:,i], means[:,i])
    if mode==0 or mode==2 or mode==3:
        stddevs = dist.tensor_distribution.components_distribution.tensor_distribution.stddev().numpy()
        stddevs = np.reshape(stddevs, (len(stddevs), len(means[0])))
        # compute global variance: \sigma_g^2 = \sum_{i=1}^k \pi_1 (\sigma_i^2 + \mu_i^2) - \mu_g^2
        globalvartemp = 0
        for i in range(len(stddevs[0])):
            globalvartemp += np.multiply(probs[:,i], (stddevs[:,i]**2 + means[:,i]**2))
        globalstd = np.sqrt(globalvartemp - globalmean**2)

    if mode==0:
        return globalmean, globalstd
    elif mode==1:
        return globalmean
    elif mode==2:
        return globalstd
    elif mode==3:
        return probs[:,0], means[:,0], stddevs[:,0], probs[:,1], means[:,1], stddevs[:,1]
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

#% ---------------------------------------------------------
# Miscellaneous functions

def copy_plots_to_cernbox(fmt='pdf', location='taunet_plots'):
    """Copy plots of format `fmt` to given `location`"""

    _cernbox = os.path.join(
        '/eos/user/',
        os.getenv('USER')[0],
        os.getenv('USER'),
        location)
    if not os.path.exists(_cernbox):
        cmd = 'mkdir -p {}'.format(_cernbox)
        log.info(cmd)
        subprocess.run(cmd, shell=True)

    #! kinda a sketch way but should work...
    if location != 'taunet_plots':
        os.listdir(os.path.join(location, 'plots'))
        for _fig in os.listdir(os.path.join(location, 'plots')):
            if _fig.endswith(fmt):
                cmd = 'cp {} {}'.format(
                    os.path.join(location, 'plots', _fig),
                    _cernbox)
                log.info(cmd)
                subprocess.run(cmd, shell=True)
    else:
        for _fig in os.listdir('./plots/'):
            if _fig.endswith(fmt):
                cmd = 'cp {} {}'.format(
                    os.path.join('./plots', _fig),
                    _cernbox)
                log.info(cmd)
                subprocess.run(cmd, shell=True)

def find_num_entries(path, dataset, nfiles=-1):
    """Find total numbet of entries in dataset"""

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
