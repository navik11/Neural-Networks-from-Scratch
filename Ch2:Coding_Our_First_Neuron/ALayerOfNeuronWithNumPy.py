# A layer of neurons using NumPy
import numpy as np

# Input features (one sample)
inputs = [1.0, 2.0, 3.0, 2.5]

# Weights for multiple neurons
# Each neuron has one weight per input feature
# → 4 input features ⇒ 4 weights per neuron
# Number of neurons = number of weight rows
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# Biases: one bias per neuron
# Number of biases = number of neurons
biases = [2.0, 3.0, 0.5]

# Compute layer output:
# matrix (weights) × vector (inputs) + biases
layer_output = np.dot(weights, inputs) + biases
print(layer_output)
