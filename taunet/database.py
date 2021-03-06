import os
from . import log; log = log.getChild(__name__)

DEFAULT_PATH = '/eos/atlas/atlascerngroupdisk/perf-tau/MxAODs/R22/Round3/TES/'
PATH = os.getenv("TAUNET_PATH", DEFAULT_PATH)
DATASET = 'group.perf-tau.Round3_FinalMVATES.425200.Pythia8EvtGen_A14NNPDF23LO_Gammatautau_MassWeight_v1_output.root'


def file_list(path, dataset):
    """
    """
    log.info('Looking in folder {}'.format(path))
    if not os.path.exists(
            os.path.join(path, dataset)):
        raise ValueError('{} not found in {}'.format(path, dataset))
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
    """
    """
    if not 'TauJetsAuxDyn.nTracks' in fields:
        raise ValueError

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
        
def training_data(path, dataset, features, target, nfiles=-1, select_1p=False, select_3p=False, use_cache=False, tree_name='CollectionTree'):
    """
    """
    if use_cache:
        pass

    else:
        import uproot
        import awkward as ak
        import numpy as np
        _train  = []
        _target = []
        _files = file_list(path, dataset)

        for i_f, _file in enumerate(_files):
            if nfiles > 0 and i_f > nfiles:
                break
            with uproot.open(_file) as up_file:
                tree = up_file[tree_name]
                log.info('file {} / {} -- entries = {}'.format(i_f, len(_files), tree.num_entries))
                a = retrieve_arrays(
                    tree,
                    features + [target], 
                    cut = 'EventInfoAux.eventNumber%3 != 0',
                    select_1p=select_1p,
                    select_3p=select_3p)
                a = a[ a['TauJetsAuxDyn.ptPanTauCellBased/TauJetsAuxDyn.ptCombined'] < 25. ]
                a = a[ a['TauJetsAuxDyn.ptIntermediateAxis/TauJetsAuxDyn.ptCombined'] < 25. ]
                f = np.stack(
                    [ak.flatten(a[__feat]).to_numpy() for __feat in features])
                _train  += [f.T]
                _target += [ak.flatten(a[target]).to_numpy()]

        _train  = np.concatenate(_train, axis=0)
        _target = np.concatenate(_target)
        _target = _target.reshape(_target.shape[0], 1)
        log.info('Total training input = {}'.format(_train.shape))

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            _train, _target, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

def testing_data(
        path, dataset, features, plotting_fields, regressor, 
        nfiles=-1, select_1p=False, select_3p=False, tree_name='CollectionTree'):
    """
    """
    import uproot
    import awkward as ak
    import numpy as np
    
    # from numpy.lib.recfunctions import append_fields
    
    _files = file_list(path, dataset)
    # build unique list of variables to retrieve
    _fields_to_lookup = list(set(features + plotting_fields))
    
    _arrs = []
    for i_f, _file in enumerate(_files):
        if nfiles > 0 and i_f > nfiles:
            break
        with uproot.open(_file) as up_file:
            tree = up_file[tree_name]
            log.info('file {} / {} -- entries = {}'.format(i_f, len(_files), tree.num_entries))
            a = retrieve_arrays(
                tree,
                _fields_to_lookup,
                cut = 'EventInfoAux.eventNumber%3 != 0',
                select_1p=select_1p,
                select_3p=select_3p)
            f = np.stack(
                [ak.flatten(a[__feat]).to_numpy() for __feat in features])
            regressed_target = regressor.predict(f.T)
            regressed_target = regressed_target.reshape((regressed_target.shape[0], ))
            _arr = np.stack(
                [ak.flatten(a[_var]).to_numpy() for _var in plotting_fields] + [regressed_target], axis=1)
            _arr = np.core.records.fromarrays(
                _arr.transpose(), 
                names=[_var for _var in plotting_fields] + ['regressed_target'])
            # append_fields(_arr, 'regressed_target', regressed_target, usemask=False)
            _arrs += [_arr]

    _arrs = np.concatenate(_arrs)
    log.info('Total testing input = {}'.format(_arrs.shape))
    return _arrs


