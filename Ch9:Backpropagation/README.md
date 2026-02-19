*Source: [**Neural Networks from Scratch in Python**](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła*

# Chapter 9: Backpropagation

Backpropagation is the algorithm that allows a neural network to **learn**.  
It computes how much each weight and bias contributed to the final loss and uses this information to update them.

This chapter builds from:
- derivatives (Chapter 7)
- gradients and chain rule (Chapter 8)

### 1. Core Idea of Backpropagation

During the **forward pass**, data flows through the network:

$
x \rightarrow \text{Dense} \rightarrow \text{Activation} \rightarrow \dots \rightarrow \text{Loss}
$

During the **backward pass**, gradients flow in the reverse direction:

$
\text{Loss} \rightarrow \dots \rightarrow \text{Activation} \rightarrow \text{Dense} \rightarrow x
$

The goal is to compute:

$
\frac{\partial L}{\partial w}, \quad
\frac{\partial L}{\partial b}
$

These are the gradients used to update parameters.

### 2. Backpropagation on a Single Neuron

Forward pass for one neuron:

$z = x_0w_0 + x_1w_1 + x_2w_2 + b$

$y = \text{ReLU}(z) = \max(0, z)$

This is a chain of operations:

$y = \text{ReLU}(\text{sum}(\text{mul}(x, w), b))$

Backpropagation uses the **chain rule**:

$\frac{\partial y}{\partial w_0}=\frac{\partial y}{\partial z}\cdot\frac{\partial z}{\partial (x_0w_0)}\cdot\frac{\partial (x_0w_0)}{\partial w_0}$

Animation: [Backpropagation through neuron](https://nnfs.io/pro)

### 3. Derivatives of Each Operation

#### ReLU derivative

$\frac{d}{dz} \text{ReLU}(z) =
\begin{cases}
1 & z > 0 \\
0 & z \le 0
\end{cases}$

#### Sum operation

For:

$z = a + b + c + d$

$\frac{\partial z}{\partial a} = 1$

The derivative of a sum is always **1**.

#### Multiplication operation

For:

$
f(x, w) = xw
$

$
\frac{\partial f}{\partial x} = w
$

$
\frac{\partial f}{\partial w} = x
$

### 4. Final Gradients for a Single Neuron

After applying the chain rule:

Gradient with respect to inputs:

$
\frac{\partial y}{\partial x_i} = w_i \cdot d
$

Gradient with respect to weights:

$
\frac{\partial y}{\partial w_i} = x_i \cdot d
$

Gradient with respect to bias:

$
\frac{\partial y}{\partial b} = d
$

Where \( d \) is the gradient from the next layer.


### 5. Updating Weights

Weights are updated using gradient descent:

$
w = w - \eta \frac{\partial L}{\partial w}
$

$
b = b - \eta \frac{\partial L}{\partial b}
$

Where:
- \( $\eta$ \) = learning rate

This moves parameters in the direction that **reduces the loss**.

### 6. Backpropagation Through a Layer

For a full dense layer:

#### Gradient with respect to inputs

$
dinputs = dvalues \cdot W^T
$


#### Gradient with respect to weights

$
dweights = X^T \cdot dvalues
$


#### Gradient with respect to biases

$
dbiases = \sum dvalues
$

These formulas allow efficient computation using matrix multiplication.

### 7. ReLU Backward Pass

ReLU gradient:

$
dinputs =
\begin{cases}
dvalues & z > 0 \\
0 & z \le 0
\end{cases}
$

This simply:
- keeps gradients where input > 0
- zeroes gradients where input ≤ 0

### 8. Categorical Cross-Entropy Loss Derivative

Loss function:

$
L_i = -\sum y_{i,j} \log(\hat{y}_{i,j})
$

Derivative:


$\frac{\partial L}{\partial \hat{y}}=-\frac{y}{\hat{y}}$

This gives the gradient of the loss with respect to predictions.

### 9. Softmax Derivative

Softmax output:

$
S_j = \frac{e^{z_j}}{\sum e^{z_k}}
$

Its derivative forms a **Jacobian matrix**:

$
\frac{\partial S_j}{\partial z_k}=S_j(\delta_{jk} - S_k)
$

Where:
- \( $\delta_{jk}$ \) is the Kronecker delta.

### 10. Combined Softmax + Cross-Entropy Gradient

When Softmax and cross-entropy are combined, the gradient simplifies to:

$
\frac{\partial L}{\partial z}=\hat{y} - y
$

This is:
- much simpler
- much faster to compute

Animation: [Chain rule simplification](https://nnfs.io/com)

### 11. Why the Combined Version Is Used

Separate gradients:
- require Jacobian matrices
- involve loops
- slower computation

Combined gradient:
- simple subtraction
- no Jacobian
- about **7× faster**

### 12. Full Backpropagation Flow

For a two-layer network:

1. Forward pass  
2. Compute loss  
3. Backward pass:
   - Loss → Softmax
   - Softmax → Dense layer
   - Dense → ReLU
   - ReLU → previous Dense
4. Update weights

## Key Takeaways

- Backpropagation uses the chain rule to compute gradients.
- Each layer computes:
  - gradients for weights
  - gradients for biases
  - gradients for inputs
- Dense layer gradients use matrix multiplication.
- ReLU blocks gradients for negative inputs.
- Softmax + cross-entropy simplifies to: $\hat{y} - y$
- This combined gradient is faster and widely used in practice.
