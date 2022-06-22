import os
import tensorflow as tf

from taunet.database import PATH, DATASET, training_data
from taunet.fields import FEATURES, TARGET_FIELD
if __name__ == '__main__':
    
    from taunet.parser import train_parser
    # train_parser = argparse.ArgumentParser(parents=[common_parser])
    # train_parser.add_argument('--use-cache', action='store_true')
    args = train_parser.parse_args()

    if args.debug:
        n_files = 1 #set limit on files for testing / debugging
    else:
        n_files = -1

    # get training data
    X_train, X_val, y_train, y_val = training_data(
        PATH, DATASET, FEATURES, TARGET_FIELD, nfiles=n_files)


    from taunet.models import keras_model_main
    regressor = keras_model_main(len(FEATURES))
    #? create location to save training
    _model_file = os.path.join('cache', regressor.name+'.h5')
    try:
        # TODO varry values of rate and batch_size
        rate = 0.001 #default rate 0.001
        batch_size = 64 #default size 32
        # print ("Rate = {}, batch_size = {}".format(rate, batch_size))
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
        #? Does this only save the best model? How would this work in a for loop for e.g.
        regressor.save(_model_file) # save results of training
        from taunet.plotting import nn_history
        for k in history.history.keys():
            if 'val' in k:
                continue
            nn_history(history, metric=k)
    except KeyboardInterrupt:
        print('Ended early...')



    
