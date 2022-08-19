"""
Authors: Quentin Buat and Miles Cochran-Branson
Date: Summer 2022

Create plots of performance of machine learning analysis in comparison
to standard methods currently in place for TES calibration at ATLAS. 
"""

import subprocess
from genericpath import exists
import os

from taunet.database import PATH, DATASET, testing_data
from taunet.fields import FEATURES, TRUTH_FIELDS, OTHER_TES, FEATURES_NEW
if __name__ == '__main__':
    
    from taunet.parser import plot_parser
    args = plot_parser.parse_args()

    if args.newTarget:
        FEATURES = FEATURES_NEW
        target_normalize_var = 'TauJetsAuxDyn.ptTauEnergyScale'
    else: 
        target_normalize_var = 'TauJetsAuxDyn.ptCombined'

    n_files = args.nfiles

    path = args.path # path to where training is stored
    # make plot folder for plots if it doesn't already exist
    if not os.path.exists(os.path.join(path, 'plots')):
        cmd = 'mkdir -p {}'.format(os.path.join(path, 'plots'))
        subprocess.run(cmd, shell=True)

    # if not using cache load these packages
    if not args.use_cache:
        import tensorflow as tf
        import tensorflow_probability as tfp
        from taunet.utils import tf_mdn_loss
    if path != '' and not args.use_cache:
        regressor = tf.keras.models.load_model(os.path.join(path, args.model), 
            custom_objects={'MixtureNormal': tfp.layers.MixtureNormal, 'tf_mdn_loss': tf_mdn_loss})
    elif not args.use_cache:
        regressor = tf.keras.models.load_model(os.path.join('cache', args.model), 
            custom_objects={'MixtureNormal': tfp.layers.MixtureNormal, 'tf_mdn_loss': tf_mdn_loss})
    else:
        regressor = ''

    if args.get_above_below:
        d, d_above, d_below = testing_data(PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, getAboveBelow=True,
            nfiles=n_files, optional_path=path, select_1p=args.oneProng, select_3p=args.threeProngs, no_normalize=args.no_normalize, 
            no_norm_target=args.no_norm_target, debug=args.debug, noCombined=args.newTarget)
    elif args.get_GMM_components:
        d, d_GMM_output = testing_data(PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, getGMMcomponents=True,
            nfiles=n_files, optional_path=path, select_1p=args.oneProng, select_3p=args.threeProngs, no_normalize=args.no_normalize, 
            no_norm_target=args.no_norm_target, debug=args.debug, noCombined=args.newTarget)
        import numpy as np
        np.save(file='/eos/user/m/mcochran/temp_data_files/d', arr=d)
        np.save(file='/eos/user/m/mcochran/temp_data_files/GMM_output', arr=d_GMM_output)
    else:
        d = testing_data(PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, 
            nfiles=n_files, optional_path=path, select_1p=args.oneProng, select_3p=args.threeProngs, no_normalize=args.no_normalize, 
            no_norm_target=args.no_norm_target, debug=args.debug, noCombined=args.newTarget)

    from taunet.plotting import nn_history
    nn_history(os.path.join(path, 'history.p'), path)

    from taunet.plotting import pt_lineshape
    pt_lineshape(d, path, target_normalize_var=target_normalize_var, nbins=100)

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
    response_and_resol_vs_var(d, path, xvar='eta', target_normalize_var=target_normalize_var, nbins=35)

    if args.get_above_below:
        from taunet.plotting import response_lineshape_above_below, response_above_below
        response_above_below(d, d_above, d_below, path, 'all')
        response_lineshape_above_below(d, d_above, path, 'plots/response_lineshape_above.pdf', txt='$|\\frac{\\sigma}{\\mu}| > 1$')
        response_lineshape_above_below(d, d_below, path, 'plots/response_lineshape_below.pdf', txt='$|\\frac{\\sigma}{\\mu}| < 1$')

    if args.get_GMM_components:
        from taunet.plotting import visualize_GMM_vars
        truth_pt = d['TauJetsAuxDyn.truthPtVisDressed']
        visualize_GMM_vars(truth_pt/1000, d_GMM_output['mu1'], path, ytitle='$\\mu_1$', plotSaveName='plots/mu1.pdf')
        visualize_GMM_vars(truth_pt/1000, d_GMM_output['mu2'], path, ytitle='$\\mu_2$', plotSaveName='plots/mu2.pdf')
        visualize_GMM_vars(truth_pt/1000, d_GMM_output['sig1'], path, ytitle='$\\sigma_1$', plotSaveName='plots/sig1.pdf')
        visualize_GMM_vars(truth_pt/1000, d_GMM_output['sig2'], path, ytitle='$\\sigma_2$', plotSaveName='plots/sig2.pdf')
        visualize_GMM_vars(truth_pt/1000, d_GMM_output['pi1'], path, ytitle='$\\pi_1$', plotSaveName='plots/pi1.pdf')
        visualize_GMM_vars(truth_pt/1000, d_GMM_output['pi2'], path, ytitle='$\\pi_1$', plotSaveName='plots/pi2.pdf')
        visualize_GMM_vars(truth_pt/1000, abs(d_GMM_output['mu1']-d_GMM_output['mu2']), path, 
            ytitle='$|\\mu_1 - \\mu_2|$', plotSaveName='plots/mu1_minus_mu2_vs_pt.pdf')
        visualize_GMM_vars(truth_pt/1000, abs(d_GMM_output['sig1']-d_GMM_output['sig2']), path, 
            ytitle='$|\\sigma_1 - \\sigma_2|$', plotSaveName='plots/sig1_minus_sig2_vs_pt.pdf')

    if args.copy_to_cernbox:
        from taunet.utils import copy_plots_to_cernbox
        if path != '':
            copy_plots_to_cernbox(location=path)
        else:
            copy_plots_to_cernbox()