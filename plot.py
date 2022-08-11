"""
Authors: Quentin Buat and Miles Cochran-Branson
Date: 6/24/22

Create plots of performance of machine learning analysis in comparison
to standard methods currently in place. 

Optional command-line arguments:

    --debug : run with only three files
    --copy-to-cernbox : copy plots to cernbox
    --path : specify path where plots are saved. If used, training *.h5 file
             may also be located in this folder without needing to specify a location 
             as in --model below. 
    --model : specify path to training model
    --use-cache : use data from cache for plotting
"""
#from asyncio import subprocess
import subprocess
from genericpath import exists
import os

from taunet.database import PATH, DATASET, testing_data
from taunet.fields import FEATURES, FEATURES_NEW, TRUTH_FIELDS, OTHER_TES
if __name__ == '__main__':
    
    from taunet.parser import plot_parser
    # train_parser = argparse.ArgumentParser(parents=[common_parser])
    # train_parser.add_argument('--use-cache', action='store_true')
    args = plot_parser.parse_args()

    if args.newTarget:
        FEATURES = FEATURES_NEW
        target_normalize_var = 'TauJetsAuxDyn.ptTauEnergyScale'
    else: 
        target_normalize_var = 'TauJetsAuxDyn.ptCombined'

    if args.debug:
        n_files = 3
    else:
        n_files = args.nfiles

    path = args.path # path to where training is stored
    # make plot folder if it doesn't already exist
    if not os.path.exists(os.path.join(path, 'plots')):
        cmd = 'mkdir -p {}'.format(os.path.join(path, 'plots'))
        subprocess.run(cmd, shell=True)
    # load potentially required packages
    if not args.use_cache:
        import tensorflow as tf
        import tensorflow_probability as tfp
    # loads result of training to make plots 
    #! I did some weird things here... be careful when running with different 
    #! models, etc
    from taunet.computation import tf_mdn_loss
    if path != '' and not args.use_cache:
        regressor = tf.keras.models.load_model(os.path.join(path, args.model), custom_objects={'MixtureNormal': tfp.layers.MixtureNormal, 'tf_mdn_loss': tf_mdn_loss})
    elif not args.use_cache:
        regressor = tf.keras.models.load_model(os.path.join('cache', args.model), custom_objects={'MixtureNormal': tfp.layers.MixtureNormal, 'tf_mdn_loss': tf_mdn_loss})
    else:
        regressor = ''

    d = testing_data(
        PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, nfiles=n_files, 
        optional_path=path, select_1p=args.oneProng, select_3p=args.threeProngs, normIndices=list(map(int, args.normIDs)),
        no_normalize=args.no_normalize, no_norm_target=args.no_norm_target, debug=args.debug, noCombined=args.newTarget)

    from taunet.plotting import nn_history
    nn_history(os.path.join(path, 'history.p'), path)

    from taunet.plotting import pt_lineshape
    pt_lineshape(d, path, target_normalize_var=target_normalize_var)

    from taunet.plotting import response_lineshape
    response_lineshape(d, path)
    response_lineshape(d, path, plotSaveName='plots/tes_response_lineshape_zoomedin.pdf', 
        Range=(0.9, 1.1), scale='linear', lineat1=True, target_normalize_var=target_normalize_var)

    from taunet.plotting import target_lineshape
    target_lineshape(d, plotSaveLoc=path, target_normalize_var=target_normalize_var)
    target_lineshape(d, bins=100, range=(0.5, 1.5), basename='tes_target_lineshape_zoomedin', 
        logy=False, plotSaveLoc=path, target_normalize_var=target_normalize_var)

    from taunet.plotting import response_and_resol_vs_var
    response_and_resol_vs_var(d, path, target_normalize_var=target_normalize_var)
    response_and_resol_vs_var(d, path, xvar='eta', target_normalize_var=target_normalize_var)

    if args.copy_to_cernbox:
        from taunet.utils import copy_plots_to_cernbox
        if path != '':
            copy_plots_to_cernbox(location=path)
        else:
            copy_plots_to_cernbox()