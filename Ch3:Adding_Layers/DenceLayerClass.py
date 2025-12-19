import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense (fully-connected) layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random values
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        
        # Initialize biases to zero (one bias per neuron)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Forward pass: compute layer output
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

# Generate a non-linear dataset
X, y = spiral_data(samples=100, classes=3)

# Create a dense layer (2 input features → 3 neurons)
dense1 = Layer_Dense(2, 3)

# Perform a forward pass
dense1.forward(X)

# Print outputs of the first 5 samples
print(dense1.output[:5])
