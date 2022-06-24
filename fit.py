"""
Authors: Quinten Buat and Miles Cochran-Branson
Date: 6/24/22

Top-level file to perform machine learning analysis on 
"""
import os
import tensorflow as tf
import pickle

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
    X_train, X_val, y_train, y_val = training_data(
        PATH, DATASET, FEATURES, TARGET_FIELD, nfiles=n_files)

    # import model
    from taunet.models import keras_model_main
    regressor = keras_model_main(len(FEATURES))
    # create location to save training
    _model_file = os.path.join('cache', regressor.name+'.h5')
    try:
        rate = args.rate #default rate 0.001
        batch_size = args.batch_size #default size 64
        # optimized as a stochastic gradient descent (i.e. Adam)
        adam = tf.keras.optimizers.get('Adam')
        #? why is this printed twice
        print (adam.learning_rate)
        adam.learning_rate = rate
        print (adam.learning_rate)
        _epochs = 100
        regressor.compile(
            loss='mean_squared_error', 
            optimizer=adam, 
            metrics=['mse', 'mae'])
        history = regressor.fit(
            X_train, # input data
            y_train, # target data
            epochs=_epochs,
            batch_size=batch_size, #number of samples per gradient update
            shuffle=True,
            verbose=2, # reports on progress
            # sample_weight=sample_weights,
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