import tensorflow as tf
import tensorflow_probability as tfp

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

#%%----------------------------------------------------------------
# MDN models

# smaller MDN model 
def keras_model_small_mdn(n_variables, name='simple_mdn'):
    x_1 = tf.keras.Input(shape=n_variables)
    hidden_0 = tf.keras.layers.Dense(192, activation='relu')(x_1)
    hidden_1 = tf.keras.layers.Dense(192, activation='relu')(hidden_0)
    hidden_2 = tf.keras.layers.Dense(192, activation='relu')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(128, activation='relu')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(64, activation='relu')(hidden_3)
    hidden_5 = tf.keras.layers.Dense(tfp.layers.MixtureNormal.params_size(1, [1]), activation=None)(hidden_4)
    output   = tfp.layers.MixtureNormal(1, [1])(hidden_5)
    return tf.keras.Model(inputs=x_1, outputs=output, name=name)

def keras_model_small_mdn_regular(n_variables, name='simple_mdn_regular'):
    x_1 = tf.keras.Input(shape=n_variables)
    hidden_0 = tf.keras.layers.Dense(192, activation='relu', kernel_regularizer='l2')(x_1)
    hidden_1 = tf.keras.layers.Dense(192, activation='relu', kernel_regularizer='l2')(hidden_0)
    hidden_2 = tf.keras.layers.Dense(192, activation='relu', kernel_regularizer='l2')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2')(hidden_3)
    hidden_5 = tf.keras.layers.Dense(tfp.layers.MixtureNormal.params_size(1, [1]), activation=None)(hidden_4)
    output   = tfp.layers.MixtureNormal(1, [1])(hidden_5)
    return tf.keras.Model(inputs=x_1, outputs=output, name=name)

# bigger MDN model
def keras_model_big_mdn(n_variables, name='less_simple_mdn'):
    x_1 = tf.keras.Input(shape=n_variables) 
    hidden_0 = tf.keras.layers.Dense(1024, activation='relu')(x_1)
    hidden_1 = tf.keras.layers.Dense(512, activation='relu')(hidden_0)
    hidden_2 = tf.keras.layers.Dense(256, activation='relu')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(128, activation='relu')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(64, activation='relu')(hidden_3)
    hidden_5 = tf.keras.layers.Dense(32, activation='relu')(hidden_4)
    hidden_6 = tf.keras.layers.Dense(16, activation='relu')(hidden_5)
    hidden_7 = tf.keras.layers.Dense(8, activation='relu')(hidden_6)
    hidden_8 = tf.keras.layers.Dense(tfp.layers.MixtureNormal.params_size(1, [1]), activation=None)(hidden_7)
    output   = tfp.layers.MixtureNormal(1, [1])(hidden_8)
    return tf.keras.Model(inputs=x_1, outputs=output, name=name)

def keras_model_big_mdn_regular(n_variables, name='less_simple_mdn_regular'):
    x_1 = tf.keras.Input(shape=n_variables) 
    hidden_0 = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer='l2')(x_1)
    hidden_1 = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer='l2')(hidden_0)
    hidden_2 = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l2')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2')(hidden_3)
    hidden_5 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2')(hidden_4)
    hidden_6 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer='l2')(hidden_5)
    hidden_7 = tf.keras.layers.Dense(8, activation='relu', kernel_regularizer='l2')(hidden_6)
    hidden_8 = tf.keras.layers.Dense(tfp.layers.MixtureNormal.params_size(1, [1]), activation=None)(hidden_7)
    output   = tfp.layers.MixtureNormal(1, [1])(hidden_8)
    return tf.keras.Model(inputs=x_1, outputs=output, name=name)

def keras_model_MultiGauss_mdn(n_variables, name='less_simple_mdn'):
    # compute some initial values for the model
    event_shape = [5]
    num_components = 10
    param_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)
    # Build the model
    x_1 = tf.keras.Input(shape=n_variables) 
    hidden_0 = tf.keras.layers.Dense(1024, activation='relu')(x_1)
    hidden_1 = tf.keras.layers.Dense(512, activation='relu')(hidden_0)
    hidden_2 = tf.keras.layers.Dense(256, activation='relu')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(128, activation='relu')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(64, activation='relu')(hidden_3)
    hidden_5 = tf.keras.layers.Dense(32, activation='relu')(hidden_4)
    hidden_6 = tf.keras.layers.Dense(16, activation='relu')(hidden_5)
    hidden_7 = tf.keras.layers.Dense(8, activation='relu')(hidden_6)
    hidden_8 = tf.keras.layers.Dense(param_size, activation=None)(hidden_7)
    output   = tfp.layers.MixtureNormal(num_components, event_shape)(hidden_8)
    return tf.keras.Model(inputs=x_1, outputs=output, name=name)

def keras_model_2gauss_mdn_small(n_variables, name='gauss2_simple_mdn'):
    event_shape = [1]
    num_components = 2
    param_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)
    x_1 = tf.keras.Input(shape=n_variables)
    hidden_0 = tf.keras.layers.Dense(192, activation='relu', kernel_regularizer='l2')(x_1)
    hidden_1 = tf.keras.layers.Dense(192, activation='relu', kernel_regularizer='l2')(hidden_0)
    hidden_2 = tf.keras.layers.Dense(192, activation='relu', kernel_regularizer='l2')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2')(hidden_3)
    hidden_5 = tf.keras.layers.Dense(param_size, activation=None)(hidden_4)
    output   = tfp.layers.MixtureNormal(num_components, event_shape)(hidden_5)
    return tf.keras.Model(inputs=x_1, outputs=output, name=name)

def keras_model_1gauss_mdn_small(n_variables, name='gauss_simple_mdn'):
    event_shape = [1]
    num_components = 1
    param_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)
    x_1 = tf.keras.Input(shape=n_variables)
    hidden_0 = tf.keras.layers.Dense(192, activation='relu')(x_1)
    hidden_1 = tf.keras.layers.Dense(192, activation='relu')(hidden_0)
    hidden_2 = tf.keras.layers.Dense(192, activation='relu')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(128, activation='relu')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(64, activation='relu')(hidden_3)
    hidden_5 = tf.keras.layers.Dense(param_size, activation=None)(hidden_4)
    output   = tfp.layers.MixtureNormal(num_components, event_shape)(hidden_5)
    return tf.keras.Model(inputs=x_1, outputs=output, name=name)

def keras_model_2gauss_mdn_small_noreg(n_variables, name='gauss2_simple_mdn_noreg'):
    event_shape = [1]
    num_components = 2
    param_size = tfp.layers.MixtureNormal.params_size(num_components, event_shape)
    x_1 = tf.keras.Input(shape=n_variables)
    hidden_0 = tf.keras.layers.Dense(192, activation='relu')(x_1)
    hidden_1 = tf.keras.layers.Dense(192, activation='relu')(hidden_0)
    hidden_2 = tf.keras.layers.Dense(192, activation='relu')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(128, activation='relu')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(64, activation='relu')(hidden_3)
    hidden_5 = tf.keras.layers.Dense(param_size, activation=None)(hidden_4)
    output   = tfp.layers.MixtureNormal(num_components, event_shape)(hidden_5)
    return tf.keras.Model(inputs=x_1, outputs=output, name=name)