# A Layer of Neurons And Batch Data with NumPy
import numpy as np

# Batch of input samples (each row = one sample, each column = one feature)
inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# Weights for a layer of neurons
# Each row represents the weights of one neuron
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# Bias for each neuron
biases = [2.0, 3.0, 0.5]

# Compute layer outputs:
# 1. Transpose weights so neurons align with input features
# 2. Perform matrix multiplication (inputs × weights)
# 3. Add biases to each neuron output
layer_outputs = np.dot(inputs, np.array(weights).T) + biases

# Output: one row per input sample, one column per neuron
print(layer_outputs)
