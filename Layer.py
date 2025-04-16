import numpy as np
from Neuron import Neuron
from Activation import Activation


class Layer:
    """Represents a layer of neurons in the network."""

    def __init__(self, input_size, num_neurons, activation):
        """
        Initializes a layer with multiple neurons.

        Args:
            input_size (int): Number of inputs per neuron.
            num_neurons (int): Number of neurons in the layer.
            activation (str): Activation function name ('relu' or 'sigmoid').
        """
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]
        self.activation = activation
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        """
        Performs forward propagation for all neurons in the layer.

        Args:
            inputs (numpy array): Input vector.

        Returns:
            numpy array: Layer outputs after applying activation function.
        """
        self.inputs = inputs
        raw_outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])

        if self.activation == "relu":
            self.outputs = Activation.relu(raw_outputs)
        elif self.activation == "sigmoid":
            self.outputs = Activation.sigmoid(raw_outputs)
        else:
            raise ValueError("Unsupported activation function")

        return self.outputs

    def backward(self, d_output, learning_rate, momentum=0.9):
        """
        Performs backpropagation for the layer.

        Args:
            d_output (numpy array): Gradient of loss with respect to output.
            learning_rate (float): Learning rate for gradient descent.

        Returns:
            numpy array: Gradient with respect to inputs.
        """
        if self.activation == "relu":
            d_activation = Activation.relu_derivative(self.outputs) * d_output
        elif self.activation == "sigmoid":
            d_activation = Activation.sigmoid_derivative(self.outputs) * d_output
        else:
            raise ValueError("Unsupported activation function")

        d_inputs = np.zeros_like(self.inputs)

        for i, neuron in enumerate(self.neurons):
            '''
            neuron.weights -= learning_rate * d_activation[i] * self.inputs
            neuron.bias -= learning_rate * d_activation[i]
            d_inputs += d_activation[i] * neuron.weights
            '''

            d_weights = d_activation[i] * self.inputs
            d_bias = d_activation[i]

            neuron.update_weights(d_weights, d_bias, learning_rate, momentum)

            d_inputs += d_activation[i] * neuron.weights

        return d_inputs

