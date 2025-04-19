import numpy as np
from EmailAI.Network import Network
from EmailAI.Activation import Activation


class EmailNetwork:
    """A specialized neural network for classifying email embeddings."""

    def __init__(self):
        """
        Initializes an email classification neural network.
        """
        self.network = Network(input_size=300, hidden_sizes=[256, 128], output_size=1)

    def classify(self, embedding):
        """
        Classifies a given email embedding.

        Args:
            embedding (numpy array): A 300-dimensional email embedding.

        Returns:
            int: 1 (Spam) or 0 (Not Spam).
        """
        output = self.network.forward(embedding,training=False)
        print(output)
        if output >= 0.5:
            return "Spam"
        else:
            return "Not Spam"

    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        """
        Trains the network.

        Args:
            X_train (numpy array): Training data.
            y_train (numpy array): Labels (0 for not spam, 1 for spam).
            epochs (int): Training iterations.
            learning_rate (float): Learning rate.
        """
        self.network.train(X_train, y_train, epochs, learning_rate)
    def save_weights(self,filename):
        self.network.save_weights(filename)
    def load_weights(self,filename):
        self.network.load_weights(filename)