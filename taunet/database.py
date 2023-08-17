"""Pre-process data from learning and plotting

Authors: Miles Cochran-Branson and Quentin Buat
Date: Summer 2022
"""
import os
from unittest.mock import DEFAULT

from taunet.utils import StandardScalar, applySSNormalizeTest, getSSNormalize, applySSNormalizeTest, getVarIndices, applySSNormalize, get_global_params, cut_above_below
from taunet.fields import VARNORM
from . import log; log = log.getChild(__name__)

# get data
DEFAULT_PATH = '/eos/atlas/atlascerngroupdisk/perf-tau/MxAODs/R22/Run2repro/TES/'
PATH = os.getenv("TAUNET_PATH", DEFAULT_PATH)
DATASET = 'group.perf-tau.MC20d_StreamTES.425200.Pythia8EvtGen_A14NNPDF23LO_Gammatautau_MassWeight_v3_output.root'

def file_list(path, dataset):
    """Find files from simulated data. 

    Parameters:
    ----------

    path : str
        path to dataset
    
    dataset : str
        dataset name

    Returns:
    -------

    List of .root files where data is stored. 
    """

    log.info('Looking in folder {}'.format(path))
    if not os.path.exists(
            os.path.join(path, dataset)):
        raise ValueError('{} not found in {}'.format(dataset, path))
    log.info('Gathering files from dataset {}'.format(dataset))
    _files = []
    for _f in  os.listdir(
            os.path.join(
                path, dataset)):
        _files += [os.path.join(
            path, dataset, _f)]
    log.info('Found {} files'.format(len(_files)))
    return _files

def retrieve_arrays(tree, fields, cut=None, select_1p=False, select_3p=False):
    """Get arrays of data from tree of root files

    Parameters:
    ----------

    tree : selected tree from root file. 

    fields : variables used in analysis. 

    cut : str
        Cut to be applied to whole dataset

    select_1p : bool
        Select 1-prong events

    select_3p : bool
        Select 3-prong events

    Returns:
    -------

    numpy array of data from root files. 
    """

    if not 'TauJetsAuxDyn.nTracks' in fields:
        log.info("nTracks temporarrily added to variable list")
        fields = fields + ['TauJetsAuxDyn.nTracks']

    if select_1p and select_3p:
        raise ValueError

    arrays = tree.arrays(fields, cut=cut)
    arrays = arrays[ arrays['TauJetsAuxDyn.nTracks'] > 0 ]
    arrays = arrays[ arrays['TauJetsAuxDyn.nTracks'] < 6 ]
    if select_1p:
        arrays = arrays[ arrays['TauJetsAuxDyn.nTracks'] == 1 ]
    if select_3p: 
        arrays = arrays[ arrays['TauJetsAuxDyn.nTracks'] == 3 ]

    return arrays

# select only part of the data for debuging
def debug_mode(tree, features, select_1p = False, select_3p = False, cut = None, stepsize=200000):
    """Get arrays of data from tree of root files in chunks of `stepsize`

    Same function as `retrieve_arrays` but with optional parameter
    `stepsize` to select size of chunks taken from tree
    """

    feats_new = features + ['TauJetsAuxDyn.nTracks']
    log.info('Taking {} chunck with {} events from file'.format(1, stepsize))
    for arr in tree.iterate(feats_new, step_size=stepsize, cut=cut):
        # apply cuts
        arr = arr[ arr['TauJetsAuxDyn.nTracks'] > 0 ]
        arr = arr[ arr['TauJetsAuxDyn.nTracks'] < 6 ]
        if select_1p and select_3p:
            raise ValueError
        if select_1p:
            arr = arr[ arr['TauJetsAuxDyn.nTracks'] == 1 ]
        if select_3p: 
            arr = arr[ arr['TauJetsAuxDyn.nTracks'] == 3 ]

        return arr
      
def training_data(path, dataset, features, target, nfiles=-1, select_1p=False, select_3p=False, 
    use_cache=False, save_to_cache=False, tree_name='CollectionTree', no_normalize=False, no_norm_target=False, 
    normSavePath='data/normFactors', debug=False):
    """Optain properly-formatted training and validation data from given dataset

    Parameters:
    ----------

    path : str
        Path to directory where dataset is held

    dataset : str
        Directory where .root files from dataset are held

    features : list of str
        List of variables to pass to network

    target : str
        Regression target of network

    nfiles=-1 : int
        Number of files to use in obtaining training data

    debug=False : bool
        Use only a portion of the data from `dataset`

    select_1p=False : bool
        Select one prong events

    select_3p=False : bool
        Select three prong events

    use_cache=False : bool
        Use previously formatted data saved in `data` directory in .npy files

    add_to_cache=False : bool
        Save formatted data in .npy files

    tree_name='CollectionTree' : str
        Tree of .root file from which to get data

    no_normalize=False : bool
        Optionally apply standard scalar normalization to selected variables

    no_norm_target=False : bool
        Optionally apply standard scalar normalization to target variable

    normSavePath='data/normFactors' : str
        Path to where means and stddevs of each variable in `features` are stored. 
        Saved in .npy file format

    Returns:
    -------

    X_train : np.array
        Multi-dimensional array of training data. Columns contain training
        variables and each row represents one event (80% of total sample given)

    X_val : np.array
        Same as X_train (20% of total sample)

    y_train : np.array
        Vector of training data (80% of total sample)

    y_val : np.array
        Same as y_train (20% of total sample)
    """

    import numpy as np

    if use_cache:
        info.cache('Using cache')
        X_train = np.load('data/X_train.npy')
        X_val = np.load('data/X_val.npy')
        y_train = np.load('data/y_train.npy')
        y_val = np.load('data/y_val.npy')
        return X_train, X_val, y_train, y_val

    else:
        import uproot
        import awkward as ak
        _train  = []
        _target = []
        _files = file_list(path, dataset)

        # iterate through files in dataset
        for i_f, _file in enumerate(_files):
            # stop after certain limit
            if nfiles >= 0 and i_f > nfiles:
                break
            with uproot.open(_file) as up_file:
                tree = up_file[tree_name]
                log.info('file {} / {} -- entries = {}'.format(i_f, len(_files), tree.num_entries))
                # make some cuts of the data
                if debug:
                    a = debug_mode(
                        tree,
                        features + [target], 
                        cut = 'EventInfoAuxDyn.eventNumber%3 != 0',
                        select_1p=select_1p,
                        select_3p=select_3p)
                else:
                    a = retrieve_arrays(
                        tree,
                        features + [target], 
                        cut = 'EventInfoAuxDyn.eventNumber%3 != 0',
                        select_1p=select_1p,
                        select_3p=select_3p)
                a = a[ a['TauJetsAuxDyn.ptIntermediateAxisEM/TauJetsAuxDyn.ptIntermediateAxis'] < 15. ]
                if 'Combined' in target:
                    a = a[ a['TauJetsAuxDyn.ptPanTauCellBased/TauJetsAuxDyn.ptCombined'] < 25. ] 
                    a = a[ a['TauJetsAuxDyn.ptIntermediateAxis/TauJetsAuxDyn.ptCombined'] < 25. ] 
                if 'Combined' not in target:
                    a = a[ a['TauJetsAuxDyn.ptPanTauCellBased/TauJetsAuxDyn.ptTauEnergyScale'] < 25. ] 
                    a = a[ a['TauJetsAuxDyn.ptIntermediateAxis/TauJetsAuxDyn.ptTauEnergyScale'] < 25. ] 
                f = np.stack(
                    [ak.flatten(a[__feat]).to_numpy() for __feat in features])
                _train  += [f.T]
                _target += [ak.flatten(a[target]).to_numpy()]

        _train  = np.concatenate(_train, axis=0)
        _target = np.concatenate(_target)
        _target = _target.reshape(_target.shape[0], 1)
        log.info('Total training input = {}'.format(_train.shape))

        #! added for testing
        og_train = np.array(_train)
        old_target = np.array(_target)

        #normalize here!
        old_train = np.array(_train)
        if not no_normalize:
            log.info('Normalizing training data')
            norms = getSSNormalize(_train, _target, savepath=normSavePath)
            _train = applySSNormalize(_train, norms, 
                        vars=getVarIndices(features, VARNORM))
            log.info("Variables to be normalized: {}".format(
                list(features[i] for i in getVarIndices(features, VARNORM))))
            if not no_norm_target:
                log.info('Normalizing validation data')
                _target = StandardScalar(_target, norms[len(norms) - 1][0], norms[len(norms) - 1][1])
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            _train, _target, test_size=0.2, random_state=42)
        log.info('Total validation input {}'.format(len(X_val)))

        if save_to_cache:
            log.info('Saving to cache')
            np.save(file='data/X_train.npy', arr=X_train)
            np.save(file='data/X_val', arr=X_val)
            np.save(file='data/y_train', arr=y_train)
            np.save(file='data/y_val', arr=y_val)

    return X_train, X_val, y_train, y_val, old_train, _train

def testing_data(
        path, dataset, features, plotting_fields, regressor, nfiles=-1, select_1p=False, select_3p=False, 
        tree_name='CollectionTree', saveToCache=False, useCache=False, optional_path='', getAboveBelow=False, 
        getGMMcomponents=False, no_normalize=False, no_norm_target=False, debug=False, noCombined=False):
    """Optain properly-formatted training and validation data from given dataset

    Parameters:
    ----------

    path : str
        Path to directory where dataset is held

    dataset : str
        Directory where .root files from dataset are held

    plotting_fields : list of str
        List of variables to pass to network

    regressor : tf.keras.model
        Previously trained network

    optional_path='' : str
        Optional path to where regessor, norm factors, and history is saved. 
        Save plots in a subfolder called `plots` in this same location. 

    nfiles=-1 : int
        Number of files to use in obtaining training data

    debug=False : bool
        Use only a portion of the data from `dataset`

    select_1p=False : bool
        Select one prong events

    select_3p=False : bool
        Select three prong events

    use_cache=False : bool
        Use previously formatted data saved in `data` directory in .npy files

    add_to_cache=False : bool
        Save formatted data in .npy files

    tree_name='CollectionTree' : str
        Tree of .root file from which to get data

    no_normalize=False : bool
        Optionally apply standard scalar normalization to selected variables

    no_norm_target=False : bool
        Optionally apply standard scalar normalization to target variable

    noCombined=False : bool
        If trained with no combined variables, plot without these as well

    getAboveBelow=False : bool
        Get better and worse events based on |sigma/mean| < 1

    getGMMcomponents=False : bool
        Get pi, mean, stddev from components of gaussian mixture model

    Returns:
    -------

    Array of dimension (num plotting fields, num testing events). 
    - If getAboveBelow, returns array of all events, array of worse events, array of better events. 
    - If getGMMcompoenets returns array of all events, pi1, mu1, sigma1, pi2, mu2, sigma2. 
    """

    if getAboveBelow and getGMMcomponents:
        raise ValueError("Change code to make this possible dawg!")
    
    import numpy as np

    if useCache:
        log.info('Using cache')
        if getAboveBelow:
            return np.load('data/d.npy'), np.load('data/d_above.npy'), np.load('data/d_below.npy')
        elif getGMMcomponents:
            return np.load('data/d.npy'), np.load('data/GMMcomp.npy')
        else:
            return np.load('data/d.npy')

    import uproot
    import awkward as ak
    
    # from numpy.lib.recfunctions import append_fields
    
    _files = file_list(path, dataset)
    # build unique list of variables to retrieve
    _fields_to_lookup = list(set(features + plotting_fields))

    # load normalization from training if used
    if not no_normalize or not no_norm_target:
        if optional_path == '':
            norms = np.load('data/normFactors.npy')
        else:
            norms = np.load(os.path.join(optional_path, 'normFactors.npy'))
    
    _arrs = []
    if getAboveBelow:
        _arrs_above = []
        _arrs_below = []
    if getGMMcomponents:
        _output_arrs = []
    for i_f, _file in enumerate(_files):
        if nfiles > 0 and i_f > nfiles:
            break
        with uproot.open(_file) as up_file:
            tree = up_file[tree_name]
            log.info('file {} / {} -- entries = {}'.format(i_f, len(_files), tree.num_entries))
            if debug:
                a = debug_mode(
                    tree,
                    _fields_to_lookup, 
                    cut = 'EventInfoAuxDyn.eventNumber%3 == 0',
                    select_1p=select_1p,
                    select_3p=select_3p)
            else:
                a = retrieve_arrays(
                    tree,
                    _fields_to_lookup, 
                    cut = 'EventInfoAuxDyn.eventNumber%3 == 0',
                    select_1p=select_1p,
                    select_3p=select_3p)
            a = a[ a['TauJetsAuxDyn.ptIntermediateAxisEM/TauJetsAuxDyn.ptIntermediateAxis'] < 25. ]
            if not noCombined:
                a = a[ a['TauJetsAuxDyn.ptPanTauCellBased/TauJetsAuxDyn.ptCombined'] < 25. ] 
                a = a[ a['TauJetsAuxDyn.ptIntermediateAxis/TauJetsAuxDyn.ptCombined'] < 25. ]
            if noCombined:
                a = a[ a['TauJetsAuxDyn.ptPanTauCellBased/TauJetsAuxDyn.ptTauEnergyScale'] < 25. ] 
                a = a[ a['TauJetsAuxDyn.ptIntermediateAxis/TauJetsAuxDyn.ptTauEnergyScale'] < 25. ] 
            f = np.stack(
                [ak.flatten(a[__feat]).to_numpy() for __feat in features])
            # Optionally normalize data if done in the training
            if not no_normalize:
                f = applySSNormalizeTest(f, norms, vars=getVarIndices(features, VARNORM))
                log.info('Normalizing input data to regressor')
            regressed_target, regressed_target_sigma = get_global_params(regressor, f.T, mode=0)
            if getAboveBelow:
                cut1, cut2 = cut_above_below(regressed_target, regressed_target_sigma)
                f1 = f.T[cut1]
                f2 = f.T[cut2]
                regressed_target1 = get_global_params(regressor, f1, mode=1)
                regressed_target2 = get_global_params(regressor, f2, mode=1)
            if not no_norm_target:
                # If target was normalized, revert to original. 
                # Last element of variable "norms" contains mean (element 0) 
                # and std (element 1) of target. 
                log.info('Returning data to orginal format for plotting')
                regressed_target = norms[len(norms)-1][1] * regressed_target + norms[len(norms)-1][0]
                if getAboveBelow:
                    regressed_target1 = norms[len(norms)-1][1] * regressed_target1 + norms[len(norms)-1][0]
                    regressed_target2 = norms[len(norms)-1][1] * regressed_target2 + norms[len(norms)-1][0]
            regressed_target = regressed_target.reshape((regressed_target.shape[0], ))
            regressed_target_sigma = regressed_target_sigma.reshape((regressed_target_sigma.shape[0], ))
            _arr = np.stack([ak.flatten(a[_var]).to_numpy() for _var in plotting_fields], axis=1)
            if getAboveBelow:
                _arr_above = _arr[cut1]
                _arr_below = _arr[cut2]
            temp_len = len(_arr[0])
            _arr = np.insert(_arr, temp_len, regressed_target, axis=1)
            _arr = np.insert(_arr, temp_len+1, regressed_target_sigma, axis=1)
            _arr = np.core.records.fromarrays(
                _arr.transpose(), 
                names=[_var for _var in plotting_fields] + ['regressed_target'] + ['regressed_target_sigma'])
            _arrs += [_arr]
            if getAboveBelow:
                _arr_above = np.insert(_arr_above, temp_len, regressed_target1, axis=1)
                _arr_below = np.insert(_arr_below, temp_len, regressed_target2, axis=1)
                _arr_above = np.core.records.fromarrays(_arr_above.transpose(), 
                                names=[_var for _var in plotting_fields] + ['regressed_target'])
                _arr_below = np.core.records.fromarrays(_arr_below.transpose(), 
                                names=[_var for _var in plotting_fields] + ['regressed_target'])
                _arrs_above += [_arr_above]
                _arrs_below += [_arr_below]

            if getGMMcomponents:
                compos = get_global_params(regressor, f.T, mode=3)
                templist = []
                for i in range(len(compos)):
                    temp = np.array(compos[i])
                    templist += [temp.reshape((temp.shape[0], ))]
                _output_arr = np.array(np.stack([templist[i] for i in range(len(templist))], axis=1))
                names = list(np.array([['pi{}'.format(i+1), 'mu{}'.format(i+1), 'sig{}'.format(i+1)] for i in range(int(len(templist)/3))]).flatten())
                _output_arr = np.core.records.fromarrays(_output_arr.transpose(), names=names)
                _output_arrs += [_output_arr]

    log.info("Variables normalized: {}".format(
                list(features[i] for i in getVarIndices(features, VARNORM))))
    _arrs = np.concatenate(_arrs)
    if getAboveBelow:
        _arrs_above = np.concatenate(_arrs_above)
        _arrs_below = np.concatenate(_arrs_below)
    if getGMMcomponents:
        _output_arrs = np.concatenate(_output_arrs)
    log.info('Total testing input = {}'.format(_arrs.shape))
    if saveToCache:
        log.info('Saving data to cache')
        if getAboveBelow:
            np.save(file='data/d', arr=_arrs)
            np.save(file='data/d_above', arr=_arrs_above)
            np.save(file='data/d_below', arr=_arrs_below)
        elif getGMMcomponents:
            np.save(file='data/d', arr=_arrs)
            np.save(file='data/GMMcomp', arr=_output_arrs)
        else:
            np.save(file='data/d', arr=_arrs)
    
    # output based on user preference
    if getAboveBelow:
        return _arrs, _arrs_above, _arrs_below
    elif getGMMcomponents:
        return _arrs, _output_arrs
    else:
        return _arrs