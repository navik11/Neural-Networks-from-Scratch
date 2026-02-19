*Source: [**Neural Networks from Scratch in Python**](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła*

# Chapter 8: Gradients, Partial Derivatives, and the Chain Rule

This chapter explains how neural networks learn by measuring how each parameter (weights and biases) affects the loss. Since neural networks depend on many variables at once, we move beyond simple derivatives to **partial derivatives**, **gradients**, and the **chain rule**, which together form the foundation of backpropagation.

### 1. Why Partial Derivatives Are Required

Neural network outputs depend on multiple variables simultaneously:
- inputs  
- weights  
- biases  

A normal derivative measures change with respect to only one variable.  
To measure the effect of *one variable while holding others constant*, we use **partial derivatives**.

Mathematically:

$
\frac{\partial f(x, y)}{\partial x}
$

This represents how much the output changes when only \($x$\) changes.


### 2. Partial Derivative of a Sum

For a function:

$
f(x, y) = x + y
$

Partial derivatives are:

$
\frac{\partial f}{\partial x} = 1, \quad
\frac{\partial f}{\partial y} = 1
$

**Rule:**  
The partial derivative of a sum is the sum of the partial derivatives, treating other variables as constants.

### 3. Partial Derivative of Multiplication

For multiplication:

$
f(x, y) = x \cdot y
$

Partial derivatives:

$
\frac{\partial f}{\partial x} = y, \quad
\frac{\partial f}{\partial y} = x
$

**Intuition:**  
Changing \(x\) by 1 changes the output by \(y\).

This operation is fundamental in neural networks because neurons compute:

$
z = w \cdot x + b
$

### 4. Partial Derivative of the Max Function (ReLU)

The ReLU activation function is:

$
\text{ReLU}(x) = \max(0, x)
$

Its derivative is:

$
\frac{d}{dx} \max(0, x) =
\begin{cases}
1, & x > 0 \\
0, & x \le 0
\end{cases}
$

This explains:
- why ReLU is efficient  
- why dead neurons occur (gradient becomes zero)

### 5. Gradient

For a function with multiple inputs:

$
f(x, y, z)
$

The **gradient** is a vector of partial derivatives:

$
\nabla f =
\begin{bmatrix}
\frac{\partial f}{\partial x} \\
\frac{\partial f}{\partial y} \\
\frac{\partial f}{\partial z}
\end{bmatrix}
$

Meaning:
- the gradient shows how loss changes in each direction  
- gradient descent moves in the opposite direction of the gradient  

Animation: [Gradient intuition](https://nnfs.io/rok)

### 6. The Chain Rule

Neural networks are a sequence (chain) of functions:

$
x \rightarrow f \rightarrow g \rightarrow h \rightarrow L
$

The chain rule states:

$
\frac{dL}{dx} = \frac{dL}{dh}\cdot\frac{dh}{dg}\cdot\frac{dg}{df}\cdot\frac{df}{dx}
$

This rule allows gradients to flow backward through the network.

Animation: [Chain rule visualization](https://nnfs.io/tom)

### 7. Chain Rule Example

Given:

$
h(x) = 3(2x^2)^5
$

Define:
- inner function: $g(x) = 2x^2$
- outer function: $f(y) = 3y^5$

Using the chain rule:

$\frac{dh}{dx}=\frac{df}{dy}\cdot\frac{dg}{dx}$

Final derivative:

$
\frac{dh}{dx} = 240x^9
$

This demonstrates how complex derivatives are built from simpler ones.