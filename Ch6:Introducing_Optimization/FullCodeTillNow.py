import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss:
    
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        clipped_y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = clipped_y_pred[
                range(samples),
                y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(
                clipped_y_pred * y_true,
                axis=1
            )
            
        negative_log_likelihood = -np.log(correct_confidence)

        return negative_log_likelihood


from nnfs.datasets import vertical_data

X, y = vertical_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

lowest_loss = float("inf")

# Store best weights and biases
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

# --------------------------------------------------
# 1. Optimization loop (random hill-climbing)
# --------------------------------------------------

for iteration in range(10000):

    # ----------------------------------------------
    # 1.1 Randomly adjust weights and biases slightly
    # ----------------------------------------------
    # We do NOT reset weights completely.
    # Instead, we apply small random changes.
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases  += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases  += 0.05 * np.random.randn(1, 3)

    # ----------------------------------------------
    # 1.2 Forward pass through the network
    # ----------------------------------------------
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # ----------------------------------------------
    # 1.3 Compute loss
    # ----------------------------------------------
    loss = loss_function.calculate(activation2.output, y)

    # ----------------------------------------------
    # 1.4 Compute accuracy
    # ----------------------------------------------
    # Convert probabilities to predicted class indices
    predictions = np.argmax(activation2.output, axis=1)

    # Compare predictions with true labels
    accuracy = np.mean(predictions == y)

    # ----------------------------------------------
    # 1.5 Accept or reject the new weights
    # ----------------------------------------------
    if loss < lowest_loss:
        # If loss improved, keep the new weights
        print(
            f"New best found | Iteration: {iteration} | "
            f"Loss: {loss:.4f} | Accuracy: {accuracy:.3f}"
        )

        lowest_loss = loss

        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases  = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases  = dense2.biases.copy()

    else:
        # If loss got worse, revert to previous best weights
        dense1.weights = best_dense1_weights.copy()
        dense1.biases  = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases  = best_dense2_biases.copy()
