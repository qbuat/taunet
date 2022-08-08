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
import numpy as np

from taunet.database import PATH, DATASET, training_data
from taunet.fields import FEATURES, TARGET_FIELD, TARGET_FIELD_NEW
if __name__ == '__main__':
    
    from taunet.parser import train_parser
    args = train_parser.parse_args()

    if args.newTarget:
        TARGET_FIELD = TARGET_FIELD_NEW

    if args.debug:
        n_files = 3 #set limit on files for testing / debugging
    else:
        n_files = args.nfiles
    
    # get training data
    if args.use_cache:
        X_train = np.load('data/X_train.npy')
        X_val = np.load('data/X_val.npy')
        y_train = np.load('data/y_train.npy')
        y_val = np.load('data/y_val.npy')
    else:
        X_train, X_val, y_train, y_val = training_data(
            PATH, DATASET, FEATURES, TARGET_FIELD, nfiles=n_files, 
            select_1p=args.oneProng, select_3p=args.threeProngs, normIndices=list(map(int, args.normIDs)),
            no_normalize=args.no_normalize, no_norm_target=args.no_norm_target, debug=args.debug)

    if args.add_to_cache:
        np.save(file='data/X_train', arr=X_train)
        np.save(file='data/X_val', arr=X_val)
        np.save(file='data/y_train', arr=y_train)
        np.save(file='data/y_val', arr=y_val)

    # import model
    from taunet.models import keras_model_2gauss_mdn_small, keras_model_2gauss_mdn_small_noreg, keras_model_1gauss_mdn_small, keras_model_big_mdn
    from taunet.computation import tf_mdn_loss
    # choose model 
    if args.small_1gauss:
        regressor = keras_model_1gauss_mdn_small((len(FEATURES),))
    elif args.small_2gauss:
        regressor = keras_model_2gauss_mdn_small((len(FEATURES),))
    elif args.big_2gauss:
        regressor = keras_model_big_mdn((len(FEATURES),))
    else:
        regressor = keras_model_2gauss_mdn_small_noreg((len(FEATURES),))
    _model_file = os.path.join('cache', regressor.name+'.h5')
    try:
        rate = args.rate #default rate 1e-7
        batch_size = args.batch_size #default size 64
        # optimized as a stochastic gradient descent (i.e. Adam)
        adam = tf.keras.optimizers.get('Adam')
        #? why is this printed twice
        print (adam.learning_rate)
        adam.learning_rate = rate
        print (adam.learning_rate)
        _epochs = 300
        regressor.compile(
            loss=tf_mdn_loss, 
            optimizer=adam, 
            metrics=['mse', 'mae']) #metrics=['mse', 'mae']
        history = regressor.fit(
            X_train, # input data
            y_train, # target data
            epochs=_epochs,
            batch_size=batch_size, #number of samples per gradient update
            shuffle=True,
            verbose=2, # reports on progress
            #sample_weight=None,
            ## validation_split=0.1,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
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