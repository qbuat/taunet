import tensorflow as tf

from taunet.database import PATH, DATASET, training_data
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

    X_train, X_val, y_train, y_val = training_data(
        PATH, DATASET, DEFAULT_FEATURES, TARGET_FIELD, nfiles=n_files)

    from taunet.models import keras_model_main
    regressor = keras_model_main(len(DEFAULT_FEATURES))
    try:
        # rate = 0.001
        batch_size = 30
        adam = tf.keras.optimizers.get('Adam')
        # adam.learning_rate = rate
        _epochs = 100
        regressor.compile(
            loss='mean_squared_error', optimizer=adam, metrics=['mse', 'mae'])
        history = regressor.fit(
            X_train, y_train,
            epochs=_epochs,
            batch_size=batch_size,
            shuffle=True,
            # sample_weight=sample_weights,
            ## validation_split=0.1,
            validation_data=(X_val, y_val),
            callbacks=[
                # tf.keras.callbacks.EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                tf.keras.callbacks.ModelCheckpoint(
                    args.model, monitor='val_loss',
                    verbose=True, save_best_only=True)])
        regressor.save(args.model)
        from taunet.plotting import nn_history
        for k in history.history.keys():
            if 'val' in k:
                continue
            nn_history(history, metric=k)
    except KeyboardInterrupt:
        print('Ended early...')



    
