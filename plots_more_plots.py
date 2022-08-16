import numpy as np
import matplotlib.pyplot as plt

import subprocess
from genericpath import exists
import os

from taunet.database import PATH, DATASET, testing_data
from taunet.fields import FEATURES, TRUTH_FIELDS, OTHER_TES
from taunet.computation import tf_mdn_loss, VARNORM

from taunet.parser import plot_parser
args = plot_parser.parse_args()

#---------------------------------------------------------------
# Get data, etc. 

if not args.use_cache:
    import tensorflow as tf
    import tensorflow_probability as tfp
    path = 'launch_condor/fitpy_small2gaussnoreg_job0'
    #path = 'cache/gauss2_simple_mdn.h5'
    regressor = tf.keras.models.load_model(os.path.join(path, 'gauss2_simple_mdn_noreg.h5'), 
                custom_objects={'MixtureNormal': tfp.layers.MixtureNormal, 'tf_mdn_loss': tf_mdn_loss})
    d = testing_data(
        PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, nfiles=args.nfiles, 
        optional_path=path, select_1p=args.oneProng, select_3p=args.threeProngs, normIndices=list(map(int, args.normIDs)),
        no_normalize=args.no_normalize, no_norm_target=args.no_norm_target, debug=args.debug)

if args.add_to_cache:
    print('Saving data to cache')
    np.save(file='data/d', arr=d)

if args.use_cache:
    print('Getting data from cache')
    d = np.load('data/d.npy')

np.save(file='/eos/user/m/mcochran/TES_dataset/np_arrays_testing/testing_data_all', arr=d)

#---------------------------------------------------------------
# Make plots nstuff!
# from taunet.plotting import response_and_resol_vs_var
# from taunet.utils import copy_plots_to_cernbox

# response_and_resol_vs_var(d, 'perf_plots')
# copy_plots_to_cernbox(location='perf_plots')

# pltText = ['1p0n', '1p1n', '1pXn', '3p0n', '3pXn']
# pltVars = ['pt', 'eta', 'mu']
# for i in range(5):
#     dtemp = np.array(d[d['TauJetsAuxDyn.NNDecayMode'] == i])
#     np.save(file='/eos/user/m/mcochran/TES_dataset/np_arrays_testing/NNDecayMode{}'.format(i), arr=dtemp)
    # for j in range(len(pltVars)):
    #     response_and_resol_vs_var(dtemp, 'perf_plots/DecayMode{}'.format(i), xvar=pltVars[j], nbins=15)
    #     copy_plots_to_cernbox(location='perf_plots/DecayMode{}'.format(i))
