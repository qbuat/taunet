import subprocess
import os
import tensorflow as tf

from taunet.database import PATH, DATASET, testing_data
from taunet.fields import FEATURES, TRUTH_FIELDS, OTHER_TES

from taunet.parser import plot_parser
args = plot_parser.parse_args()

# debug number of files 
n_files = 2

# make plots folder if not already present
path = args.path 
if not os.path.exists(os.path.join(path, 'plots')):
        cmd = 'mkdir -p {}'.format(os.path.join(path, 'plots'))
        subprocess.run(cmd, shell=True)

# load model if data not already stored in cache
if not args.use_cache:
    if args.model != 'simple_dnn.h5':
        regressor = tf.keras.models.load_model(os.path.join(args.model))
    else:
        regressor = tf.keras.models.load_model(os.path.join('cache', args.model))
else:
    regressor = ''

# load data 
d = testing_data(
        PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, nfiles=n_files,
        saveToCache=args.add_to_cache, useCache=args.use_cache)

from taunet.plotting import pt_lineshape
pt_lineshape(d, path)

if args.copy_to_cernbox:
    from taunet.utils import copy_plots_to_cernbox
    if path != '':
        copy_plots_to_cernbox(location=path)
    else:
        copy_plots_to_cernbox()