import tensorflow as tf

def keras_model_main(n_variables):
    x_1 = tf.keras.Input(shape=n_variables)
    # hidden_0 = tf.keras.layers.Dense(1024, activation='relu')(x_1)
    hidden_1 = tf.keras.layers.Dense(128, activation='relu')(x_1)
    hidden_2 = tf.keras.layers.Dense(8, activation='relu')(hidden_1)
    output   = tf.keras.layers.Dense(1, activation='linear')(hidden_2)
    return tf.keras.Model(inputs=x_1, outputs=output)
