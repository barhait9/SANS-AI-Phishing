import numpy as np
from Activation import Activation


class Neuron:
    """Represents a single neuron with weights, bias, and activation function."""

    def __init__(self, input_size):
        """
        Initializes a neuron with small random weights and a bias.

        Args:
            input_size (int): Number of inputs to the neuron.
        """
        self.weights = np.random.randn(input_size) * 0.01  # Small random values
        self.bias = np.random.randn() * 0.01  # Small bias value

    def forward(self, inputs):
        """
        Computes the weighted sum and applies bias.

        Args:
            inputs (numpy array): Input vector.

        Returns:
            float: The output before activation.
        """
        return np.dot(self.weights, inputs) + self.bias