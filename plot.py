import tensorflow as tf

from taunet.database import PATH, DATASET, testing_data
from taunet.fields import DEFAULT_FEATURES, TRUTH_FIELDS, OTHER_TES
if __name__ == '__main__':
    
    from taunet.parser import train_parser
    # train_parser = argparse.ArgumentParser(parents=[common_parser])
    # train_parser.add_argument('--use-cache', action='store_true')
    args = train_parser.parse_args()

    if args.debug:
        n_files = 2
    else:
        n_files = -1


    regressor = tf.keras.models.load_model(args.model)

    d = testing_data(
        PATH, DATASET, DEFAULT_FEATURES, TRUTH_FIELDS + OTHER_TES, regressor, nfiles=n_files)


    
    from taunet.plotting import pt_lineshape
    pt_lineshape(d)

    from taunet.plotting import response_lineshape
    response_lineshape(d)

    from taunet.plotting import target_lineshape
    target_lineshape(d)


    from taunet.plotting import response_and_resol_vs_pt
    response_and_resol_vs_pt(d)
