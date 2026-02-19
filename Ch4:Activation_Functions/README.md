*Source: [**Neural Networks from Scratch in Python**](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła*

# Chapter 4: Activation Functions

Activation functions are applied to the outputs of neurons or layers to **modify their values**.
They are essential because **nonlinear activation functions** allow neural networks with multiple
hidden layers to model **nonlinear relationships**, which would not be possible using only linear
operations.

Different activation functions are suited for different purposes, and choosing the right one
depends on the task. Neural networks typically use **one type of activation function in hidden
layers** and a **different one in the output layer**. While hidden layers usually share the same
activation function, this is not a strict requirement.

Understanding how activation functions work is crucial for building effective neural networks
and selecting appropriate architectures for specific problems.

### The Step Activation Function

The **step activation function** was designed to mimic a biological neuron that either *fires* or
*does not fire*. It produces a binary output based on a threshold.

For a neuron, the computation is:

$z = \sum (inputs \cdot weights) + bias$

The step function is defined as:

$
f(z) =
\begin{cases}
1, & \text{if } z > 0 \\
0, & \text{otherwise}
\end{cases}
$

If the weighted sum plus bias is greater than zero, the neuron outputs `1`; otherwise, it outputs
`0`. Although this activation was used in early neural networks, it is **rarely used today** because
it is not differentiable, making learning with gradient-based methods difficult.


### The Linear Activation Function

The **linear activation function** outputs the input directly without modification. It represents
a straight-line relationship.

The function is defined as:

$f(x) = x$

or, in neuron form:

$output = \sum (inputs \cdot weights) + bias$

Linear activation functions are typically used in the **output layer of regression models**, where
the goal is to predict a continuous numerical value rather than a class. However, using linear
activations in hidden layers limits a network’s ability to learn complex, nonlinear patterns.

### The Sigmoid Activation Function

The **step function** provides very limited information to learning algorithms because its output
is strictly binary (0 or 1). During training, optimizers need to understand **how much** a weight
or bias affects the output, not just whether a neuron fired or not. Since the step function does not
indicate *how close* an input was to the threshold, it makes optimization difficult.

To address this, neural networks historically adopted the **Sigmoid activation function**, which
produces **smooth, continuous, and informative outputs**.

The sigmoid function is defined as:

$\sigma(x) = \frac{1}{1 + e^{-x}}$

Its output range is:

- \( 0 \) as \( $x \to -\infty$ \)
- \( 0.5 \) when \( $x = 0$ \)
- \( 1 \) as \( $x \to +\infty$ \)

Because sigmoid outputs values between **0 and 1**, it provides a **granular measure** of neuron
activation, preserving information about the input magnitude. This makes it more useful for
gradient-based optimization compared to the step function.

Sigmoid also introduces **nonlinearity**, which is essential for neural networks to learn complex
patterns. However, while sigmoid was historically used in **hidden layers**, it has largely been
replaced by **ReLU (Rectified Linear Unit)** due to training efficiency issues.

Today, the sigmoid function is most commonly used in the **output layer**, especially for
binary classification tasks, where outputs can be interpreted as probabilities.

### The Rectified Linear Unit (ReLU) Activation Function

The **Rectified Linear Unit (ReLU)** is one of the simplest and most widely used activation
functions in modern neural networks.

It is defined as:

$f(x) = \max(0, x)$

This means:
- If \( $x \le 0$ \), the output is **0**
- If \( $x > 0$ \), the output is **x**

ReLU can be seen as a **linear function clipped at zero** on the negative side. While it behaves
linearly for positive values, the cutoff at zero introduces **nonlinearity**, which is essential for
learning complex patterns.

ReLU is preferred over functions like sigmoid because:
- It is **computationally efficient** and fast
- It avoids expensive operations like exponentials
- It allows networks to train deeper models more effectively

Due to its simplicity, speed, and strong empirical performance, ReLU is the **default activation
function for hidden layers** in most neural networks today.

### Use of Activation Functions?

Activation functions are essential because they introduce **nonlinearity** into neural networks.
Without them, even deep networks with many layers behave like a single **linear model** and
cannot learn complex patterns.

Most real-world problems are **nonlinear**, and linear activations (`y = x`) limit a network to
learning only straight-line relationships. Nonlinear activation functions, such as **ReLU**, 
allow neurons to activate only in specific input ranges, creating localized **“areas of effect.”**

By stacking multiple layers with nonlinear activations, neural networks can combine these areas
of effect to approximate complex functions (like sine waves). Increasing the number of neurons
and layers further improves this capability.

In short, **nonlinear activation functions + multiple hidden layers** are what give neural
networks their power to model complex, real-world problems.

Read this article, [Why use activation functions?](https://nnfs.io/mvp/), for better understanding.

ReLU [implementation](./ActivationReLUClass.py).


### The Softmax Activation Function

The **Softmax activation function** is used in the **output layer of classification models** to
convert raw network outputs into a **probability distribution** over classes. Unlike ReLU, which
is unbounded and independent per neuron, Softmax produces **normalized, comparable outputs**
that sum to 1.

Softmax works by:
1. **Exponentiating** each output value to ensure non-negative numbers
2. **Normalizing** them by dividing by the sum of all exponentiated values

$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$

This results in **confidence scores** for each class, where the predicted class is the one with the
highest probability. These confidence values are important, as they indicate how certain the
model is about its prediction.

For numerical stability, Softmax subtracts the **maximum input value** before exponentiation.
This prevents extremely large numbers (overflow) without changing the final probabilities.

When applied to batches, Softmax normalizes outputs **sample-wise**, producing one probability
distribution per input sample. Softmax is typically paired with a Dense output layer and is the
standard choice for **multi-class classification**.

Softmax [implementation](./ActivationSoftmaxClass.py).