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

        for size in hidden_sizes:  #For loop to add hidden layers
            self.layers.append(Layer(prev_size, size, activation="relu",dropout_rate=0.5))
            prev_size = size

        self.layers.append(Layer(prev_size, output_size, activation="sigmoid"))  # Output layer

    def forward(self, inputs, training=True):
        """Performs forward propagation through the network."""
        for layer in self.layers:
            inputs = layer.forward(inputs,training=training)
        return inputs

    def backward(self, y_true, y_pred, learning_rate, momentum=0.9):
        """
        Performs backpropagation and updates weights.

        Args:
            y_true (float): True label (0 or 1).
            y_pred (float): Predicted output.
            learning_rate (float): Learning rate.
        """
        loss_gradient = y_pred - y_true  # Derivative of loss w.r.t. output
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate, momentum)

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
                y_pred = self.forward(x,training=True)
                y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
                loss = -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
                total_loss += loss

                self.backward(y, y_pred, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(y_train)}")

    def save_weights(self, filename):
        """
        Saves all weights and biases of the network to a text file.

        Args:
            self (Network): The network instance.
            filename (str): Path to save the weights.
        """
        with open(filename, 'w') as f:
            for layer in self.layers:
                for neuron in layer.neurons:
                    weights_str = ','.join(map(str, neuron.weights))
                    f.write(f"{weights_str}|{neuron.bias}\n")
        print(f"Weights saved to {filename}")

    def load_weights(self, filename):
        """
        Loads weights and biases from a text file into the network.

        Args:
            self (Network): The network instance to update.
            filename (str): Path to the saved weights.
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        i = 0  # Line index
        for layer in self.layers:
            for neuron in layer.neurons:
                weights_str, bias_str = lines[i].strip().split('|')
                neuron.weights = list(map(float, weights_str.split(',')))
                neuron.bias = float(bias_str)
                i += 1

        print(f"Weights loaded from {filename}")