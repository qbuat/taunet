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

from taunet.database import file_list, retrieve_arrays, debug_mode, select_norms
from taunet.computation import applySSNormalizeTest, getVarIndices, get_global_params, cut_above_below

#---------------------------------------------------------------
# Get data, etc. 

if not args.use_cache:
    import tensorflow as tf
    import tensorflow_probability as tfp
    #path = 'launch_condor/fitpy_small2gaussnoreg_job0/gauss2_simple_mdn_noreg.h5'
    path = 'cache/gauss2_simple_mdn.h5'
    regressor = tf.keras.models.load_model(path, 
                custom_objects={'MixtureNormal': tfp.layers.MixtureNormal, 'tf_mdn_loss': tf_mdn_loss})
    d = testing_data(
        PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, nfiles=args.nfiles, debug=args.debug)

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

#---------------------------------------------------------------
# Make plots nstuff!
from taunet.plotting import response_and_resol_vs_var

pltVars = ['pt', 'eta', 'mu']
for i in range(5):
    dtemp = d[d['TauJetsAuxDyn.NNDecayMode'] == i]
    for j in range(len(pltVars)):
        response_and_resol_vs_var(dtemp, 'plots/resol_resp_DecayMode{}'.format(i), xvar=pltVars[j])
