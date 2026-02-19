*Source: [**Neural Networks from Scratch in Python**](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła*

# Chapter 7: Derivatives

### Why Derivatives Are Needed
- Randomly searching for weights fails because the parameter space is **infinite**.
- Each weight and bias affects the **loss differently**, depending on:
  - Its current value
  - The input sample
  - Its position in the network
- To **intelligently reduce loss**, we must know **how much** each parameter influences the loss.
- This influence is measured using **derivatives**.


### Core Idea: Impact = Slope
- A **derivative** measures the **instantaneous rate of change** of a function.
- Geometrically, it is the **slope of the tangent line** at a point.
- In neural networks, this slope tells us:
  > “If I slightly change this weight, how does the loss change?”


### Linear vs Nonlinear Functions
#### Linear Function
- Example:  $f(x) = 2x$
- The slope is constant: $\frac{df}{dx} = 2$
- Impact of `x` on `y` is always the same.

#### Nonlinear Function
- Example:  $f(x) = 2x^2$
- The slope **depends on x**: $\frac{df}{dx} = 4x$
- Impact changes at different points → this is why neural networks can model complex behavior.

Animation: [Approximation of the parabolic function’s example tangents](https://nnfs.io/bro)

### Numerical Derivative
- Approximates the derivative using two very close points: $\frac{f(x + \Delta x) - f(x)}{\Delta x}$
- Pros:
  - Easy to understand
  - Works for any function
- Cons:
  - Very slow
  - Numerically unstable
  - Impractical for neural networks (millions of parameters)

>**Not usable for training neural networks**

Animation: [Why small delta improves derivative accuracy](https://nnfs.io/cat)

### Analytical Derivative
- Computes the **exact derivative** using math rules.
- Fast, precise, and scalable.
- Neural networks rely entirely on **analytical derivatives**.

### Key Derivative Rules Learned

#### Constants: 
$\frac{d}{dx}(c) = 0$

#### Linear Function: 
$\frac{d}{dx}(x) = 1$

$\frac{d}{dx}(mx) = m$


#### Power Rule
$\frac{d}{dx}(x^n) = n x^{n-1}$

Example:
$\frac{d}{dx}(3x^2) = 6x$

#### Sum Rule
$\frac{d}{dx}(f(x) + g(x)) = f'(x) + g'(x)$

(Same applies to subtraction)

### Why This Matters for Neural Networks
- Loss functions depend on **outputs**, not directly on weights.
- Weights affect outputs → outputs affect loss.
- Derivatives tell us:
  - Direction to move each weight
  - How big the update should be
- This leads to:
  - **Gradients**
  - **Gradient Descent**
  - **Backpropagation**

See all these animations to understand better:
- [Bias and slope intuition (tangent line shifting)](https://nnfs.io/but)
- [Derivative of a constant function](https://nnfs.io/cow)
- [Derivative of a linear function](https://nnfs.io/tob)
- [Derivative of another linear function (slope)](https://nnfs.io/pop)
- [Derivative of a quadratic function](https://nnfs.io/rok)
- [Derivative of a quadratic function with addition](https://nnfs.io/mob)
- [Analytical derivative example (multi-term function)](https://nnfs.io/tom)
- [Another analytical derivative example](https://nnfs.io/sun)


### Big Takeaway
> Derivatives quantify how each parameter affects the loss.  
> Without derivatives, learning in neural networks is impossible.

### What Comes Next
- **Partial derivatives** (functions with many variables)
- **Gradients**
- **Chain rule**
- **Backpropagation**

These will allow us to compute derivatives for **entire neural networks efficiently**.
