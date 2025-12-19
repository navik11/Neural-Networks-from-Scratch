import math
import numpy as np

# ============================================================
# PART 1: Categorical Cross-Entropy for a SINGLE SAMPLE
# ============================================================

# Output probabilities from the Softmax layer.
# These are the model's confidence scores for 3 classes.
# All values are in [0, 1] and they sum to 1.
softmax_output = [0.7, 0.1, 0.2]

# Ground truth in one-hot encoded format.
# Correct class is index 0, so:
# class 0 -> 1 (correct)
# class 1 -> 0
# class 2 -> 0
target_output = [1, 0, 0]

# ------------------------------------------------------------
# Full categorical cross-entropy loss formula:
#
#     L = - Σ ( y * log(y_hat) )
#
# y       = true label (one-hot encoded)
# y_hat   = predicted probability (softmax output)
#
# Only the correct class contributes to the loss because
# y = 0 for all incorrect classes.
# ------------------------------------------------------------

loss = -(
    math.log(softmax_output[0]) * target_output[0] +  # correct class
    math.log(softmax_output[1]) * target_output[1] +  # ignored (×0)
    math.log(softmax_output[2]) * target_output[2]    # ignored (×0)
)

print("Single-sample loss:", loss)

# ============================================================
# PART 2: Categorical Cross-Entropy for a BATCH (Sparse Labels)
# ============================================================

# Softmax outputs for a batch of 3 samples.
# Shape: (number_of_samples, number_of_classes)
softmax_outputs = np.array([
    [0.7,  0.1,  0.2],   # sample 1
    [0.1,  0.5,  0.4],   # sample 2
    [0.02, 0.9,  0.08]   # sample 3
])

# True class labels in sparse format
# Each value is the index of the correct class
class_targets = [0, 1, 1]

# ------------------------------------------------------------
# Extract the confidence score of the correct class
# for each sample using NumPy advanced indexing.
#
# range(len(softmax_outputs)) selects rows:
# [0, 1, 2]
#
# class_targets selects the correct column per row.
# ------------------------------------------------------------

correct_confidences = softmax_outputs[
    range(len(softmax_outputs)),
    class_targets
]

print("Correct class confidences:", correct_confidences)

# ------------------------------------------------------------
# Apply negative log to convert confidence into loss.
#
# High confidence → low loss
# Low confidence  → high loss
# ------------------------------------------------------------

negative_log_likelihoods = -np.log(correct_confidences)

# Average loss over the batch
average_loss = np.mean(negative_log_likelihoods)

print("Average batch loss (sparse):", average_loss)

# ============================================================
# PART 3: Handling ONE-HOT ENCODED TARGETS
# ============================================================

# One-hot encoded labels for the same batch
class_targets = np.array([
    [1, 0, 0],  # class 0
    [0, 1, 0],  # class 1
    [0, 1, 0]   # class 1
])

# ------------------------------------------------------------
# Check target format:
# - 1D array → sparse labels
# - 2D array → one-hot encoded labels
# ------------------------------------------------------------

if len(class_targets.shape) == 1:
    # Sparse label case
    correct_confidences = softmax_outputs[
        range(len(softmax_outputs)),
        class_targets
    ]

elif len(class_targets.shape) == 2:
    # One-hot case:
    # Multiply predictions by targets.
    # Only the correct class remains non-zero.
    # Sum along axis=1 to get one value per sample.
    correct_confidences = np.sum(
        softmax_outputs * class_targets,
        axis=1
    )

# Compute loss
negative_log_likelihoods = -np.log(correct_confidences)
average_loss = np.mean(negative_log_likelihoods)

print("Average batch loss (one-hot):", average_loss)

# ============================================================
# PART 4: Numerical Stability (Avoid log(0))
# ============================================================

# ------------------------------------------------------------
# log(0) → infinity
# log(1) → 0
#
# To avoid numerical issues:
# - Clip probabilities to a small range
# ------------------------------------------------------------

y_pred_clipped = np.clip(
    softmax_outputs,
    1e-7,          # minimum value
    1 - 1e-7       # maximum value
)

# ------------------------------------------------------------
# Compute loss safely using clipped predictions
# ------------------------------------------------------------

negative_log_likelihoods = -np.log(
    y_pred_clipped[
        range(len(y_pred_clipped)),
        [0, 1, 1]  # sparse targets
    ]
)

average_loss = np.mean(negative_log_likelihoods)

print("Average batch loss (clipped):", average_loss)
