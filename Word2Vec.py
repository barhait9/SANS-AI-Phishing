import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

tf.keras.preprocessing.sequence.skipgrams()
epochs = 50
learning_rate = 0.01
batch_size = 16
embedding_size = 300  #might change to 500 if it makes it better


