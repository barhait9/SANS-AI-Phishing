import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


body = "hello this will be a test of the email today semantic analysis of words for skipgram this is needed because why not type type"
context_window_size = 3
epochs = 50
learning_rate = 0.01
batch_size = 16
embedding_size = 300  #might change to 500 if it makes it better
tokens = body.split(" ")

# For every target word in tokens
for target in range(0,len(tokens)):
    # Create new window around target word using context_window_size
    window = []
    for context in range((-1*context_window_size),(context_window_size+1)):
        if target+context > len(tokens)-1 or target+context < 0:
            continue
        else:
            window.append(tokens[target+context])
    print("window array: ", window)

    # Now train Neural network using window array and target word





