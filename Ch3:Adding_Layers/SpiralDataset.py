import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

# Initialize nnfs (sets random seed and default settings)
nnfs.init()

# Generate a non-linear spiral dataset
X, y = spiral_data(samples=500, classes=3)

# Plot the dataset
# X[:, 0] and X[:, 1] are the two feature dimensions
# Color points by class label
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()
