import numpy as np
import matplotlib.pyplot as plt

import subprocess
from genericpath import exists
import os

from taunet.database import PATH, DATASET
from taunet.fields import FEATURES, TRUTH_FIELDS, OTHER_TES
from taunet.computation import tf_mdn_loss, VARNORM

from taunet.parser import plot_parser
args = plot_parser.parse_args()

from taunet.database import file_list, retrieve_arrays, debug_mode, select_norms
from taunet.computation import applySSNormalizeTest, getVarIndices, get_global_params, cut_above_below

#%------------------------------------------------------------------
# Cut above and below sigma/mu

def get_cut_abovebelow_2gauss(regressor, arr):
    dist = regressor(arr)
    logits = dist.tensor_distribution.mixture_distribution.logits.numpy()
    means = dist.tensor_distribution.components_distribution.tensor_distribution.mean().numpy()
    stddevs = dist.tensor_distribution.components_distribution.tensor_distribution.stddev().numpy()
    probs = tf.nn.softmax(logits[0:,]).numpy() # convert logits to probabilities
    # get vector of global means
    globalmean = np.array(
        [probs[i][0]*means[i][0] + probs[i][1]*means[i][1] 
                        for i in range(len(means))]).flatten()
    # get vector of global stddevs
    globalstd = np.sqrt(np.array(
        [probs[i][0]*stddevs[i][0]**2 
        + probs[i][1]*stddevs[i][1]**2 
        + probs[i][0]*probs[i][1]*(means[i][0]-means[i][1])**2 
                        for i in range(len(means))]).flatten())
    cutabove = (abs(globalstd/globalmean) > 1).flatten()
    cutbelow = (abs(globalstd/globalmean) < 1).flatten()
    return cutabove, cutbelow
#%------------------------------------------------------------------
# Load data 

def testing_data(
    path, dataset, features, plotting_fields, regressor, 
    nfiles=-1, select_1p=False, select_3p=False, tree_name='CollectionTree',
    saveToCache=False, useCache=False, optional_path='', 
    no_normalize=False, no_norm_target=False, normIndices=range(9), debug=False):
    """
    """
    import numpy as np

    if useCache:
        print('Using cache')
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
    _arrs_above = []
    _arrs_below = []
    varnom = select_norms(VARNORM, normIndices) # variables to normalize
    for i_f, _file in enumerate(_files):
        if nfiles > 0 and i_f > nfiles:
            break
        with uproot.open(_file) as up_file:
            tree = up_file[tree_name]
            print('file {} / {} -- entries = {}'.format(i_f, len(_files), tree.num_entries))
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
            a = a[ a['TauJetsAuxDyn.ptPanTauCellBased/TauJetsAuxDyn.ptCombined'] < 25. ] 
            a = a[ a['TauJetsAuxDyn.ptIntermediateAxis/TauJetsAuxDyn.ptCombined'] < 25. ]
            f = np.stack(
                [ak.flatten(a[__feat]).to_numpy() for __feat in features])
            # print('Shape of f is {}'.format(np.shape(f)))
            # Optionally normalize data if done in the training
            if not no_normalize:
                f = applySSNormalizeTest(f, norms, vars=getVarIndices(features, varnom))
                print('Normalizing input data to regressor')
            regressed_target, stddev = get_global_params(regressor, f.T)
            cut1, cut2 = cut_above_below(regressed_target, stddev)
            f1 = f.T[cut1]
            f2 = f.T[cut2]
            regressed_target1 = regressor.predict(f1)
            regressed_target2 = regressor.predict(f2)
            if not no_norm_target:
                # If target was normalized, revert to original
                # Last element of variable "norms" contains mean (element 0) 
                # and std (element 1) of target. 
                regressed_target = norms[len(norms)-1][1] * regressed_target + norms[len(norms)-1][0]
                regressed_target1 = norms[len(norms)-1][1] * regressed_target1 + norms[len(norms)-1][0]
                regressed_target2 = norms[len(norms)-1][1] * regressed_target2 + norms[len(norms)-1][0]
                print('Returning data to orginal format for plotting')
            regressed_target = regressed_target.reshape((regressed_target.shape[0], ))
            regressed_target1 = regressed_target1.reshape((regressed_target1.shape[0], ))
            regressed_target2 = regressed_target2.reshape((regressed_target2.shape[0], ))
            _arr = np.stack([ak.flatten(a[_var]).to_numpy() for _var in plotting_fields], axis=1)
            _arr_above = _arr[cut1]
            _arr_below = _arr[cut2]
            temp_len = len(_arr[0])
            _arr = np.insert(_arr, temp_len, regressed_target, axis=1)
            _arr_above = np.insert(_arr_above, temp_len, regressed_target1, axis=1)
            _arr_below = np.insert(_arr_below, temp_len, regressed_target2, axis=1)
            _arr = np.core.records.fromarrays(_arr.transpose(), names=[_var for _var in plotting_fields] + ['regressed_target'])
            _arr_above = np.core.records.fromarrays(_arr_above.transpose(), names=[_var for _var in plotting_fields] + ['regressed_target'])
            _arr_below = np.core.records.fromarrays(_arr_below.transpose(), names=[_var for _var in plotting_fields] + ['regressed_target'])
            _arrs += [_arr]
            _arrs_above += [_arr_above]
            _arrs_below += [_arr_below]

    print("Variables normalized: {}".format(list(features[i] for i in getVarIndices(features, VARNORM))))
    _arrs = np.concatenate(_arrs)
    _arrs_above = np.concatenate(_arrs_above)
    _arrs_below = np.concatenate(_arrs_below)
    print('Total testing input = {}'.format(_arrs.shape))
    if saveToCache:
        np.save('data/testingData_temp', _arrs)
        print('Saving data to cache')
    return _arrs, _arrs_above, _arrs_below

KINEMATICS = ['TauJetsAuxDyn.mu', 'TauJetsAuxDyn.etaPanTauCellBased', 'TauJetsAuxDyn.phiPanTauCellBased', 'TauJetsAuxDyn.etaDetectorAxis', 'TauJetsAuxDyn.etaIntermediateAxis']

if not args.use_cache:
    import tensorflow as tf
    import tensorflow_probability as tfp
    path = 'launch_condor/fitpy_small2gaussnoreg_job0/gauss2_simple_mdn_noreg.h5'
    #path = 'cache/gauss2_simple_mdn.h5'
    regressor = tf.keras.models.load_model(path, 
                custom_objects={'MixtureNormal': tfp.layers.MixtureNormal, 'tf_mdn_loss': tf_mdn_loss})
    d, d_above, d_below = testing_data(
        PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES + KINEMATICS, regressor, nfiles=args.nfiles, debug=args.debug)

if args.add_to_cache:
    print('Saving data to cache')
    np.save(file='data/d', arr=d)
    np.save(file='data/d_above', arr=d_above)
    np.save(file='data/d_below', arr=d_below)

if args.use_cache:
    print('Getting data from cache')
    d = np.load('data/d.npy')
    d_above = np.load('data/d_above.npy')
    d_below = np.load('data/d_below.npy')

#%------------------------------------------------------------------
# Plotting functions

def plot_thang(d, d_above, d_below, save_loc, name):

    from taunet.utils import response_curve

    response_reg = d['regressed_target'] * d['TauJetsAuxDyn.ptCombined'] / d['TauJetsAuxDyn.truthPtVisDressed']
    response_reg_above = d_above['regressed_target'] * d_above['TauJetsAuxDyn.ptCombined'] / d_above['TauJetsAuxDyn.truthPtVisDressed']
    response_reg_below = d_below['regressed_target'] * d_below['TauJetsAuxDyn.ptCombined'] / d_below['TauJetsAuxDyn.truthPtVisDressed']
    response_ref = d['TauJetsAuxDyn.ptFinalCalib'] / d['TauJetsAuxDyn.truthPtVisDressed']
    response_comb = d['TauJetsAuxDyn.ptCombined'] / d['TauJetsAuxDyn.truthPtVisDressed']
    truth_pt = d['TauJetsAuxDyn.truthPtVisDressed'] / 1000.
    truth_pt_above = d_above['TauJetsAuxDyn.truthPtVisDressed'] / 1000.
    truth_pt_below = d_below['TauJetsAuxDyn.truthPtVisDressed'] / 1000.

    bins = [
        # (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
        (50, 60),
        (60, 70),
        (70, 80),
        (80, 90),
        (90, 100),
        (100, 150),
        (150, 200),
    ]

    bins_reg, bin_errors_reg, means_reg, errs_reg, resol_reg = response_curve(response_reg, truth_pt, bins)
    abins_reg, abin_errors_reg, ameans_reg, aerrs_reg, aresol_reg = response_curve(response_reg_above, truth_pt_above, bins)
    bbins_reg, bbin_errors_reg, bmeans_reg, berrs_reg, bresol_reg = response_curve(response_reg_below, truth_pt_below, bins)
    bins_ref, bin_errors_ref, means_ref, errs_ref, resol_ref = response_curve(response_ref, truth_pt, bins)
    bins_comb, bin_errors_comb, means_comb, errs_comb, resol_comb = response_curve(response_comb, truth_pt, bins)

    # fig = plt.figure(figsize=(5,5), dpi = 100)
    # plt.ticklabel_format(axis='y',style='sci', scilimits=(-3,3))
    # plt.errorbar(bins_comb, means_comb, errs_comb, bin_errors_comb, fmt='o', color='black', label='Combined')
    # plt.errorbar(bins_ref, means_ref, errs_ref, bin_errors_ref, fmt='o', color='red', label='Final')
    # plt.errorbar(bins_reg, means_reg, errs_reg, bin_errors_reg, fmt='o', color='purple', label='This work')
    # plt.grid(color='0.95')
    # plt.ylabel('Predicted $p_{T}(\\tau_{had-vis})$ / True $p_{T}(\\tau_{had-vis})$', loc = 'top')
    # plt.xlabel('True $p_{T}(\\tau_{had-vis})$ [GeV]', loc = 'right')
    # plt.legend()
    # plt.savefig(os.path.join(save_loc, 'response_vs_pt_{}.pdf'.format(name)))
    # plt.close(fig) 

    fig = plt.figure(figsize=(5,5), dpi = 100)
    plt.plot(bins_ref, 100 * resol_ref, color='red', label='Final')
    plt.plot(bins_ref, 100 * resol_comb, color='black', label='Combined')
    plt.plot(bins_ref, 100 * resol_reg, color='purple', label='This work')
    plt.plot(bins_ref, 100 * aresol_reg, '--', color = 'purple', label = '$|\\frac{\\sigma}{\\mu}| > 1$')
    plt.plot(bins_ref, 100 * bresol_reg, '-.', color = 'purple', label = '$|\\frac{\\sigma}{\\mu}| < 1$')
    plt.ylabel('$p_{T}(\\tau_{had-vis})$ resolution, 68% CL [%]', loc = 'top')
    plt.xlabel('True $p_{T}(\\tau_{had-vis})$ [GeV]', loc = 'right')
    plt.legend()
    plt.savefig(os.path.join(save_loc, 'resolution_vs_truth_{}.pdf'.format(name)))
    plt.close(fig)

def response_lineshape(testing_data, alt_data, plotSaveLoc, 
            plotSaveName='plots/tes_response_lineshape.pdf', txt=''):
    """
    """
    fig = plt.figure(figsize=(5,5), dpi = 300)
    plt.yscale('log')
    plt.hist(
        alt_data['regressed_target'] * alt_data['TauJetsAuxDyn.ptCombined'] / alt_data['TauJetsAuxDyn.truthPtVisDressed'],
        density=True,
        bins=200, 
        range=(0, 2), 
        histtype='step', 
        color='purple', 
        label='This work')
    plt.hist(
        testing_data['TauJetsAuxDyn.ptFinalCalib'] / testing_data['TauJetsAuxDyn.truthPtVisDressed'],
        density=True,
        bins=200, 
        range=(0, 2), 
        histtype='step', 
        color='red', 
        label='Final')
    plt.hist(
        testing_data['TauJetsAuxDyn.ptCombined'] / testing_data['TauJetsAuxDyn.truthPtVisDressed'],
        density=True,
        bins=200, 
        range=(0, 2), 
        histtype='step', 
        color='black', 
        label='Combined')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.plot([1.0, 1.0], [ymin, ymax], linestyle='dashed', color='grey')
    if txt!='':
        plt.text(1.5, 4e-1, txt, fontsize=15)
    plt.ylabel('Number of $\\tau_{had-vis}$ / $\\int$ Number of $\\tau_{had-vis}$', loc = 'top')
    plt.xlabel('Predicted $p_{T}(\\tau_{had-vis})$ / True $p_{T}(\\tau_{had-vis})$', loc = 'right')
    plt.legend()
    plt.savefig(os.path.join(plotSaveLoc, plotSaveName), bbox_inches='tight')
    plt.yscale('linear')
    plt.close(fig)

#plot distributions
plot_thang(d, d_above, d_below, 'debug_plots/plots', 'all')
response_lineshape(d, d_above, 'debug_plots/plots', 'response_lineshape_above.pdf', txt='$|\\frac{\\sigma}{\\mu}| > 1$')
response_lineshape(d, d_below, 'debug_plots/plots', 'response_lineshape_below.pdf', txt='$|\\frac{\\sigma}{\\mu}| < 1$')

# ------------------------------------------------------------
# explore the kinematics of these vars!

def pT_explore(dens=True):
    fig = plt.figure(figsize=(5,5), dpi = 100)
    plt.ticklabel_format(axis='y',style='sci', scilimits=(-3,3), useMathText=True)
    plt.hist(d['regressed_target'] * d['TauJetsAuxDyn.ptCombined'] / 1000., 
            bins=200, histtype='step', range=(0, 200), label='All Events', density=dens)
    plt.hist(d_above['regressed_target'] * d_above['TauJetsAuxDyn.ptCombined'] / 1000., 
            bins=200, histtype='step', range=(0, 200), label='$|\\frac{\\sigma}{\\mu}| > 1$', density=dens)
    plt.hist(d_below['regressed_target'] * d_below['TauJetsAuxDyn.ptCombined'] / 1000., 
            bins=200, histtype='step', range=(0, 200), label='$|\\frac{\\sigma}{\\mu}| < 1$', density=dens)
    plt.xlabel('$p_T(\\tau_{had-vis})$', loc='right')
    plt.ylabel('Number of $\\tau_{had-vis}$', loc='top')
    plt.legend()
    plt.savefig('debug_plots/plots/pT_explore.pdf', bbox_inches='tight')
    plt.close(fig)

def variable_explore(var, xtitle, varname, legloc=1, dens=True):
    fig = plt.figure(figsize=(5,5), dpi = 100)
    plt.ticklabel_format(axis='y',style='sci', scilimits=(-3,3), useMathText=True)
    plt.hist(d[var], 
            bins=200, histtype='step', label='All Events', density=dens)
    plt.hist(d_above[var], 
            bins=200, histtype='step', label='$|\\frac{\\sigma}{\\mu}| > 1$', density=dens)
    plt.hist(d_below[var], 
            bins=200, histtype='step', label='$|\\frac{\\sigma}{\\mu}| < 1$', density=dens)
    plt.legend(loc=legloc)
    plt.xlabel(xtitle, loc='right')
    plt.ylabel('Number of $\\tau_{had-vis}$ / $\\int$ Number of $\\tau_{had-vis}$', loc='top')
    plt.savefig('debug_plots/plots/{}_explore.pdf'.format(varname), bbox_inches='tight')
    plt.close(fig)

# plot them thangs
pT_explore(dens=False)
variable_explore('TauJetsAuxDyn.etaPanTauCellBased', '$\\eta (\\tau_{had-vis})$', 'eta', legloc=8)
variable_explore('TauJetsAuxDyn.phiPanTauCellBased', '$\\phi (\\tau_{had-vis})$', 'phi', legloc=8)
variable_explore('TauJetsAuxDyn.mu', '$\\mu (\\tau_{had-vis})$', 'mu')

# ------------------------------------------------------------
# Copy to cernbox!
from taunet.utils import copy_plots_to_cernbox
copy_plots_to_cernbox(location='debug_plots')