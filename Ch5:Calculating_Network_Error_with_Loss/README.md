*Source: [**Neural Networks from Scratch in Python**](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła*

# Chapter 5: Calculating Network Error with Loss

To **train a neural network**, we must measure how wrong its predictions are. This is done using a
**loss function** (also called a cost function). Loss quantifies the error between the model’s
predicted outputs and the true targets. Training aims to **minimize loss**, ideally reaching 0.

Accuracy alone is insufficient because it ignores **confidence**. For example, predictions
`[0.6, 0.2, 0.2]` and `[0.36, 0.32, 0.32]` both yield the same class via `argmax`, but the first is
clearly more confident. Loss functions capture this difference.


### Categorical Cross-Entropy Loss

For **classification models** with a **Softmax output**, the standard loss function is
**Categorical Cross-Entropy**. It compares two probability distributions:
- **Target distribution** \( $y$ \) (ground truth)
- **Predicted distribution** \( $\hat{y}$ \) (Softmax output)

#### General formula:

$L_i = -\sum_{j} y_{i,j} \log(\hat{y}_{i,j})$


Where:
- \( $i$ \) = sample index  
- \( $j$ \) = class index  
- \( $y_{i,j}$ \) = target probability  
- \( $\hat{y}_{i,j}$ \) = predicted probability  



### One-Hot Target Simplification

In most classification tasks, targets are **one-hot encoded** (only one class has value 1).
This simplifies the formula to:

$L_i = -\log(\hat{y}_{i,k})$

Where:
- \( $k$ \) is the index of the correct class  

So, the loss is simply the **negative log of the model’s confidence in the correct class**.

### Why Logarithms?

- If confidence = **1**, then loss = **0**
- Lower confidence → higher loss
- The log function heavily penalizes confident wrong predictions

$\log(1) = 0, \quad \log(0.1) \approx -2.3$

Loss grows rapidly as confidence approaches 0.

### Batch Loss Computation

For a batch of samples, we:
1. Extract predicted probabilities for the correct class
2. Apply negative log
3. Compute the **mean loss**

$\text{Loss}_{batch} = \frac{1}{N} \sum_{i=1}^{N} -\log(\hat{y}_{i,k})$


### Sparse vs One-Hot Targets

- **Sparse labels**: `[0, 1, 1]`
- **One-hot labels**:
```
[1, 0, 0]
[0, 1, 0]
[0, 1, 0]
```

Both are supported:
- Sparse → index directly into predictions
- One-hot → multiply and sum across classes


### Numerical Stability (log(0) Problem)

- \( $\log(0)$ \) is undefined → results in `inf`
- This breaks training (infinite loss)

### Solution: Clipping

$\hat{y}_{clipped} = \text{clip}(\hat{y}, 10^{-7}, 1 - 10^{-7})$

This:
- Prevents infinite loss
- Avoids negative loss values
- Keeps gradients stable

### Key Takeaways

- **Loss drives learning**, not accuracy
- **Categorical Cross-Entropy** is ideal for Softmax classifiers
- Loss penalizes low confidence in correct predictions
- **Clipping probabilities** is essential for numerical stability
- Average batch loss is used to track training progress

