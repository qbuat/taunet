import os
import tensorflow as tf

from taunet.database import PATH, DATASET, testing_data
from taunet.fields import FEATURES, TRUTH_FIELDS, OTHER_TES
if __name__ == '__main__':
    
    from taunet.parser import plot_parser
    # train_parser = argparse.ArgumentParser(parents=[common_parser])
    # train_parser.add_argument('--use-cache', action='store_true')
    args = plot_parser.parse_args()

    if args.debug:
        n_files = 2
    else:
        n_files = -1

    #? loads result of training to make plots?    
    regressor = tf.keras.models.load_model(os.path.join(
        'cache', args.model))

    d = testing_data(
        PATH, DATASET, FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, nfiles=n_files)


    from taunet.plotting import pt_lineshape
    pt_lineshape(d)

    from taunet.plotting import response_lineshape
    response_lineshape(d)

    from taunet.plotting import target_lineshape
    target_lineshape(d)
    target_lineshape(d, bins=100, range=(0.5, 1.5), basename='tes_target_lineshape_zoomedin', logy=False)

    from taunet.plotting import response_and_resol_vs_pt
    response_and_resol_vs_pt(d)

    if args.copy_to_cernbox:
        from taunet.utils import copy_plots_to_cernbox
        copy_plots_to_cernbox()