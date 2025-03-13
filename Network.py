import numpy as np
from Layer import Layer
from Activation import Activation


class Network:
    """Represents a simple feedforward neural network with backpropagation."""

    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initializes a neural network with specified layer sizes.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (list of int): Number of neurons in each hidden layer.
            output_size (int): Number of output neurons.
        """
        self.layers = []
        prev_size = input_size

        for size in hidden_sizes:
            self.layers.append(Layer(prev_size, size, activation="relu"))
            prev_size = size

        self.layers.append(Layer(prev_size, output_size, activation="sigmoid"))  # Output layer

    def forward(self, inputs):
        """Performs forward propagation through the network."""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, y_true, y_pred, learning_rate):
        """
        Performs backpropagation and updates weights.

        Args:
            y_true (float): True label (0 or 1).
            y_pred (float): Predicted output.
            learning_rate (float): Learning rate.
        """
        loss_gradient = y_pred - y_true  # Derivative of loss w.r.t. output
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        """
        Trains the network using stochastic gradient descent.

        Args:
            X_train (numpy array): Training inputs.
            y_train (numpy array): Training labels (0 or 1).
            epochs (int): Number of training iterations.
            learning_rate (float): Learning rate.
        """
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X_train, y_train):
                y_pred = self.forward(x)
                total_loss += (y_pred - y) ** 2  # MSE Loss
                self.backward(y, y_pred, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(y_train)}")