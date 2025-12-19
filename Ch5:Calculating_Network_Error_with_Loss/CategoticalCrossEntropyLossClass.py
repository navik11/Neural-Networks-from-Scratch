import numpy as np

# ============================================================
# Base Loss class
# ------------------------------------------------------------
# This class defines a common interface for all loss functions.
# Any specific loss (like categorical cross-entropy) will
# inherit from this class.
# ============================================================
class Loss:
    
    def calculate(self, output, y):
        """
        Computes the average loss over a batch of samples.

        Parameters:
        output : predicted probabilities from the model (y_pred)
        y      : true labels (can be sparse or one-hot encoded)

        Returns:
        data_loss : scalar value representing mean loss
        """

        # Call the forward() method of the child loss class.
        # This returns the loss for EACH sample in the batch.
        sample_losses = self.forward(output, y)

        # Compute the mean loss over all samples
        data_loss = np.mean(sample_losses)
        
        return data_loss


# ============================================================
# Categorical Cross-Entropy Loss
# ------------------------------------------------------------
# Used for multi-class classification problems.
# Works with:
# 1) Sparse labels   -> [0, 1, 2]
# 2) One-hot labels  -> [[1,0,0], [0,1,0], ...]
# ============================================================
class Loss_CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """
        Computes categorical cross-entropy loss per sample.

        Parameters:
        y_pred : predicted probabilities (Softmax output)
        y_true : true labels (sparse or one-hot)

        Returns:
        negative_log_likelihoods : loss value for each sample
        """

        # Number of samples in the batch
        samples = len(y_pred)

        # ----------------------------------------------------
        # Numerical stability:
        # Prevent log(0) -> infinity
        # Prevent log(1) -> 0 with floating-point issues
        # ----------------------------------------------------
        clipped_y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # ----------------------------------------------------
        # Case 1: Sparse labels
        # Example: y_true = [0, 1, 2]
        # ----------------------------------------------------
        if len(y_true.shape) == 1:
            # Select the predicted probability of the correct class
            correct_confidence = clipped_y_pred[
                range(samples),
                y_true
            ]

        # ----------------------------------------------------
        # Case 2: One-hot encoded labels
        # Example:
        # y_true = [[1,0,0],
        #           [0,1,0],
        #           [0,0,1]]
        # ----------------------------------------------------
        elif len(y_true.shape) == 2:
            # Multiply predictions by one-hot labels
            # Sum keeps only the correct class probability
            correct_confidence = np.sum(
                clipped_y_pred * y_true,
                axis=1
            )

        # ----------------------------------------------------
        # Categorical Cross-Entropy formula:
        #
        #     L = -log(correct_class_probability)
        #
        # High confidence → small loss
        # Low confidence  → large loss
        # ----------------------------------------------------
        negative_log_likelihood = -np.log(correct_confidence)

        return negative_log_likelihood
