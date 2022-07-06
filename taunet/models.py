import tensorflow as tf
#import tensorflow_probability as tfp

def keras_model_main(n_variables, name='simple_dnn'):
    x_1 = tf.keras.Input(shape=n_variables)
    # create some densely-connected NN layers
    # relu = rectified linear unit activation, i.e. f(x) = max(x, 0)
    hidden_0 = tf.keras.layers.Dense(1024, activation='relu')(x_1)
    hidden_1 = tf.keras.layers.Dense(512, activation='relu')(hidden_0)
    hidden_2 = tf.keras.layers.Dense(256, activation='relu')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(128, activation='relu')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(64, activation='relu')(hidden_3)
    hidden_5 = tf.keras.layers.Dense(32, activation='relu')(hidden_4)
    hidden_6 = tf.keras.layers.Dense(16, activation='relu')(hidden_5)
    hidden_7 = tf.keras.layers.Dense(8, activation='relu')(hidden_6)
    output   = tf.keras.layers.Dense(1, activation='linear')(hidden_7)
    return tf.keras.Model(inputs=x_1, outputs=output, name=name)

def keras_model_terry(n_variables, name='more_simple_dnn'):
    x_1 = tf.keras.Input(shape=n_variables)
    """
    Potentially the NN network used in:
    https://indico.cern.ch/event/830584/contributions/3653901/attachments/1955169/3247362/20191203_first_RNN_TES.pdf
    """
    # create some densely-connected NN layers
    # relu = rectified linear unit activation, i.e. f(x) = max(x, 0)
    hidden_0 = tf.keras.layers.Dense(192, activation='relu')(x_1)
    hidden_1 = tf.keras.layers.Dense(192, activation='relu')(hidden_0)
    hidden_2 = tf.keras.layers.Dense(192, activation='relu')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(128, activation='relu')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(64, activation='relu')(hidden_3)
    output   = tf.keras.layers.Dense(1, activation='linear')(hidden_4)
    return tf.keras.Model(inputs=x_1, outputs=output, name=name)

def keras_model_mdn(n_variables, name='simple_mdn'):
    x_1 = tf.keras.Input(shape=n_variables)
    # create some densely-connected NN layers
    # relu = rectified linear unit activation, i.e. f(x) = max(x, 0)
    hidden_0 = tf.keras.layers.Dense(192, activation='relu')(x_1)
    hidden_1 = tf.keras.layers.Dense(192, activation='relu')(hidden_0)
    hidden_2 = tf.keras.layers.Dense(192, activation='relu')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(128, activation='relu')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(64, activation='relu')(hidden_3)
    output   = tfp.layers.MixtureNormal(1)(hidden_4)
    return tf.keras.Model(inputs=x_1, output=output, name=name)