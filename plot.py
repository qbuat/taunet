import tensorflow as tf

from taunet.database import PATH, DATASET, training_data, 
from taunet.fields import DEFAULT_FEATURES, TARGET_FIELD
if __name__ == '__main__':
    
    from taunet.parser import train_parser
    # train_parser = argparse.ArgumentParser(parents=[common_parser])
    # train_parser.add_argument('--use-cache', action='store_true')
    args = train_parser.parse_args()

    if args.debug:
        n_files = 2
    else:
        n_files = -1


    regressor = tf.keras.model.load_model(args.model)



    X_train, X_val, y_train, y_val = training_data(
        PATH, DATASET, DEFAULT_FEATURES, TARGET_FIELD, nfiles=n_files)



    
