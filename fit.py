"""
Authors: Quinten Buat and Miles Cochran-Branson
Date: 6/24/22

Top-level file to perform machine learning training in order to better
detect tau leptons at the ATLAS detector. 

Command-line options:
    --debug : run with only two files
    --rate : specify sample rate for learning; defualt is 0.001
    --batch-size : specify batch size for learning; default is 64
"""
import os
import pickle
import tensorflow as tf

from taunet.database import PATH, DATASET, training_data
from taunet.fields import FEATURES, TARGET_FIELD
if __name__ == '__main__':
    
    from taunet.parser import train_parser
    args = train_parser.parse_args()

    if args.debug:
        n_files = 1 #set limit on files for testing / debugging
    else:
        n_files = -1
    
    # get training data
    X_train, X_val, y_train, y_val, test1, test2 = training_data(
        PATH, DATASET, FEATURES, TARGET_FIELD, nfiles=n_files, 
        select_1p=args.oneProng, select_3p=args.threeProngs,
        no_normalize=args.no_normalize, no_norm_target=args.no_norm_target)

    # import model
    from taunet.models import keras_model_mdn
    from taunet.computation import tf_mdn_loss
    regressor = keras_model_mdn((len(FEATURES),))
    # create location to save training
    _model_file = os.path.join('cache', regressor.name+'.h5')
    try:
        rate = args.rate #default rate 1e-5
        batch_size = args.batch_size #default size 64
        # optimized as a stochastic gradient descent (i.e. Adam)
        adam = tf.keras.optimizers.get('Adam')
        #? why is this printed twice
        print (adam.learning_rate)
        adam.learning_rate = rate
        print (adam.learning_rate)
        _epochs = 10
        regressor.compile(
            loss=tf_mdn_loss, 
            optimizer=adam,
            metrics=['mse', 'mae'])
        history = regressor.fit(
            X_train, # input data
            y_train, # target data
            epochs=_epochs,
            batch_size=batch_size, #number of samples per gradient update
            shuffle=True,
            verbose=2, # reports on progress
            sample_weight=None,
            ## validation_split=0.1,
            validation_data=(X_val, y_val),
            callbacks=[
                # tf.keras.callbacks.EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                tf.keras.callbacks.ModelCheckpoint(
                    _model_file, 
                    monitor='val_loss',
                    verbose=True, 
                    save_best_only=True)])
        # save only best run
        regressor.save(_model_file) # save results of training
        #Now save history in a pickle file for future use
        pickle.dump(history.history, open("history.p", "wb"))
    # Allow to keyboard interupt to not go over all epochs
    except KeyboardInterrupt:
        print('Ended early...')