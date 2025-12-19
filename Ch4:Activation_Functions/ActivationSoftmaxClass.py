import numpy as np

# Softmax activation function
# Converts raw neuron outputs (logits) into probabilities
class Activation_Softmax:
    def forward(self, inputs):
        # Step 1: Numerical stability
        # Subtract the maximum value in each sample to prevent
        # very large exponentials (overflow) during exp()
        # This does NOT change the final probability distribution
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)

        # Step 2: Exponentiate the shifted values
        # Ensures all values are non-negative and preserves relative ordering
        exp_values = np.exp(shifted_inputs)

        # Step 3: Normalize the exponentiated values
        # Divide each value by the sum of its row so that
        # probabilities for each sample sum to 1
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
