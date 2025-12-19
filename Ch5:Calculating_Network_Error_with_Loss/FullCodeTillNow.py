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


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)
print("loss:", loss)

# ---------------------------------------------------------
# Accuracy calculation for a real network output
# ---------------------------------------------------------

# Convert model output probabilities to predicted classes
predictions = np.argmax(activation2.output, axis=1)

# Convert labels if they are one-hot encoded
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

# Calculate accuracy the same way
accuracy = np.mean(predictions == y)

print("acc:", accuracy)
