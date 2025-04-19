import numpy as np


class Activation:
    """Contains activation functions and their derivatives."""

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of the sigmoid function."""
        return x * (1 - x)

    @staticmethod
    def relu(x):
        """ReLU activation function."""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """Derivative of the ReLU function."""
        return (x > 0).astype(float)


