import numpy as np

# ---------------------------------------------------------
# Example: Accuracy calculation on known softmax outputs
# ---------------------------------------------------------

# Softmax probabilities for 3 samples and 3 classes
softmax_outputs = np.array([
    [0.7,  0.2,  0.1],
    [0.5,  0.1,  0.4],
    [0.02, 0.9,  0.08]
])

# Ground-truth class labels (sparse form)
class_targets = np.array([0, 1, 1])

# ---------------------------------------------------------
# Step 1: Convert probabilities to predicted class indices
# ---------------------------------------------------------
# np.argmax(..., axis=1) selects the index of the largest
# probability in each row (each sample).
#
# Example:
# [0.7, 0.2, 0.1]  -> class 0
# [0.5, 0.1, 0.4]  -> class 0
# [0.02, 0.9, 0.08] -> class 1
predictions = np.argmax(softmax_outputs, axis=1)

# ---------------------------------------------------------
# Step 2: Ensure targets are in sparse format
# ---------------------------------------------------------
# If targets are one-hot encoded (2D array),
# convert them to class indices.
#
# Example:
# [0, 1, 0] -> 1
if len(class_targets.shape) == 2:
    class_targets = np.argmax(class_targets, axis=1)

# ---------------------------------------------------------
# Step 3: Compare predictions with true labels
# ---------------------------------------------------------
# (predictions == class_targets) produces a boolean array:
# True  -> correct prediction
# False -> incorrect prediction
#
# Example:
# [True, False, True]
#
# np.mean converts True → 1 and False → 0,
# giving classification accuracy.
accuracy = np.mean(predictions == class_targets)

print("acc:", accuracy)