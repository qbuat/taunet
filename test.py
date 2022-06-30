import os
import pickle
import tensorflow as tf
import numpy as np

from taunet.database import PATH, DATASET, training_data
from taunet.fields import FEATURES, TARGET_FIELD

# get training data
X_train, X_val, y_train, y_val, _train, _target = training_data(
    PATH, DATASET, FEATURES, TARGET_FIELD, nfiles=1, normalize=True)

print(np.mean(_target), " ", np.std(_target))
for i in range(18):
    print(np.mean(_train[:,i]), " ", np.std(_train[:,i]))