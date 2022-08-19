import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from taunet.computation import tf_mdn_loss

regressor = tf.keras.models.load_model('cache/gauss2_simple_mdn.h5', custom_objects={'MixtureNormal': tfp.layers.MixtureNormal, 'tf_mdn_loss': tf_mdn_loss})
temp = tf.constant([[7.0 for _ in range(20)]])
dist = regressor(temp)
weights = dist.tensor_distribution.mixture_distribution.logits.numpy()
means = dist.tensor_distribution.components_distribution.tensor_distribution.mean().numpy()
stddevs = dist.tensor_distribution.components_distribution.tensor_distribution.stddev().numpy()
dictionary = {'mean':means.flatten(), 'stddevs':stddevs.flatten()}
dictionary
regressor.get_weights()[-1]

def logit2prob(logits):
    return np.exp(logits) / (1 + np.exp(logits))

logit2prob(weights)