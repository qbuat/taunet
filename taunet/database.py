"""
Database changes:
    - added ability to normalize input / output data
    - removed nTracks from vairiable list and cuts
"""
import os
from unittest.mock import DEFAULT

from taunet.computation import StandardScalar, applySSNormalizeTest, getSSNormalize, applySSNormalizeTest, getVarIndices, select_norms, VARNORM, applySSNormalize, get_global_params
from . import log; log = log.getChild(__name__)

# local location of data
if '/Users/miles_cb' in os.getcwd():
    DEFAULT_PATH = '/Users/miles_cb/cernbox/TES_dataset'
    PATH = os.getenv("TAUNET_PATH", DEFAULT_PATH)
    DATASET = 'group.perf-tau.MC20d_StreamTES.425200.Pythia8EvtGen_A14NNPDF23LO_Gammatautau_MassWeight_v3_output.root'
# lxplus location of data
else:
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
    normSavePath='data/normFactors', normIndices=range(9), debug=False):
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

    normIndices=range(9) : vector of ints
        Indices of variables to apply standard scalar normalization

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
        Same as y_train (20% or total sample)
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

        #normalize here!
        old_train = np.array(_train)
        varnom = select_norms(VARNORM, normIndices)
        if not no_normalize:
            log.info('Normalizing training data')
            norms = getSSNormalize(_train, _target, savepath=normSavePath)
            _train = applySSNormalize(_train, norms, 
                        vars=getVarIndices(features, varnom))
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
        path, dataset, features, plotting_fields, regressor, 
        nfiles=-1, select_1p=False, select_3p=False, tree_name='CollectionTree',
        saveToCache=False, useCache=False, optional_path='', 
        no_normalize=False, no_norm_target=False, normIndices=range(9), debug=False, noCombined=False):
    """
    """
    
    import numpy as np

    if useCache:
        log.info('Using cache')
        return np.load('data/testingData_temp.npy')

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
    varnom = select_norms(VARNORM, normIndices) # variables to normalize
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
            a = a[ a['TauJetsAuxDyn.ptIntermediateAxisEM/TauJetsAuxDyn.ptIntermediateAxis'] < 25. ] # may not be necessary; must justify this cut
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
                f = applySSNormalizeTest(f, norms, vars=getVarIndices(features, varnom))
                log.info('Normalizing input data to regressor')
            regressed_target, regressed_target_sigma = get_global_params(regressor, f.T, mode=0)
            if not no_norm_target:
                # If target was normalized, revert to original
                # Last element of variable "norms" contains mean (element 0) 
                # and std (element 1) of target. 
                regressed_target = norms[len(norms)-1][1] * regressed_target + norms[len(norms)-1][0]
                log.info('Returning data to orginal format for plotting')
            regressed_target = regressed_target.reshape((regressed_target.shape[0], ))
            regressed_target_sigma = regressed_target_sigma.reshape((regressed_target_sigma.shape[0], ))
            _arr = np.stack(
                [ak.flatten(a[_var]).to_numpy() for _var in plotting_fields] + [regressed_target] + [regressed_target_sigma], axis=1)
            _arr = np.core.records.fromarrays(
                _arr.transpose(), 
                names=[_var for _var in plotting_fields] + ['regressed_target'] + ['regressed_target_sigma'])
            _arrs += [_arr]

    log.info("Variables normalized: {}".format(
                list(features[i] for i in getVarIndices(features, VARNORM))))
    _arrs = np.concatenate(_arrs)
    log.info('Total testing input = {}'.format(_arrs.shape))
    if saveToCache:
        np.save('data/testingData_temp', _arrs)
        log.info('Saving data to cache')
    return _arrs