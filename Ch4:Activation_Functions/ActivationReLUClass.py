import numpy as np

# ReLU activation function
class Activation_ReLU:
    def forward(self, inputs):
        # Apply ReLU: max(0, x)
        self.output = np.maximum(0, inputs)