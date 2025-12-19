*Source: [**Neural Networks from Scratch in Python**](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła*

# Chapter 2: Coding Our First Neuron

### A Single Neuron

A neuron receives multiple inputs, each associated with a **weight**, and includes a single
additional parameter called a **bias**. Inputs can be raw data or outputs from neurons in a
previous layer. Weights and biases are **trainable parameters** that are adjusted during training
to make the model work effectively.

The neuron computes its output by:
- Multiplying each input by its corresponding weight
- Summing these values
- Adding the bias

Mathematically, a single neuron performs:
```
output = (input₁ × weight₁) + (input₂ × weight₂) + (input₃ × weight₃) + bias
```

This simple calculation forms the foundation of all neural networks. More complex networks
are built by combining many such neurons into layers and connecting them together.

Watch [Single neuron with 3 inputs example](https://nnfs.io/bkr) and [A single neuron with 4 inputs](https://nnfs.io/djp/) in action.


### A Layer of Neurons

A neural network layer is simply a collection of neurons. All neurons within a layer receive
the **same input**, either directly from the data or from the previous layer. However, each
neuron has its **own set of weights and bias**, allowing it to produce a unique output.  
The output of a layer is the collection of outputs from all its neurons, which then serves as
input to the next layer.

Ex: [3 neuron layer with 4 inputs](https://nnfs.io/mxo)

### Dot Product and Vector Addition

Neural networks rely heavily on **vector operations**, especially the **dot product**, to perform
efficient calculations. A vector can be understood simply as a one-dimensional array or Python
list.

#### Dot Product

The **dot product** multiplies corresponding elements of two vectors and sums the results.
Both vectors must have the same length, and the output is a **single scalar value**. This mirrors
the operation of multiplying inputs by weights in a neuron.

**Example (pure Python):**
```python
a = [1, 2, 3]
b = [2, 3, 4]

dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
print(dot_product)  # 20

import numpy as np
print(np.dot(a, b))  # 20
```

The dot product provides a concise and efficient way to compute neuron outputs.

**Vector addition** adds corresponding elements of two vectors. Both vectors must be the same size, and the result is another vector.

Visualize [Dot Product in Python](https://nnfs.io/xpo).

### A Single Neuron with NumPy

See implementation of [ASingleNeuronWithNumPy.py](./ASingleNeuronWithNumPy.py) and [visualization](​https://nnfs.io/blq).

### A Layer of Neurons with NumPy

When working with a layer containing multiple neurons, the weights are organized as a
**matrix** (a 2D array), where each row represents the weight vector of one neuron. The input
remains a single vector.

Using **NumPy**, the dot product between the weight matrix and the input vector is computed
in one operation. NumPy treats the matrix as a collection of vectors and performs the dot
product for each neuron automatically, producing a vector of outputs, one per neuron.

After the dot product, a **bias vector** (with one bias per neuron) is added element-wise to
the result. This reordering of operations, computing all dot products first and adding biases
afterward, simplifies the code and significantly improves performance.

This approach replaces manual, neuron-by-neuron calculations with clean, efficient vectorized
operations, which is essential for building scalable neural networks.

See implementation of [ALayerOfNeuronWithNumPy.py](./ALayerOfNeuronWithNumPy.py) and [visualization](https://nnfs.io/cyx/).

### A Batch of Data

Neural networks are typically trained using **batches of data** rather than a single sample.
A **sample** (also called an **observation** or **feature set**) is one collection of feature
values, such as multiple sensor readings taken at the same time.

Training with batches is preferred for two main reasons:
- **Efficiency**: batches allow parallel computation, making training faster.
- **Generalization**: updating weights using multiple samples helps the model learn patterns
  that fit the entire dataset instead of memorizing individual samples.

A batch is represented as a **2D structure (matrix)**, where each row is a sample and each
column corresponds to a feature. In Python, this is commonly stored as a list of lists or a
NumPy array.

When both the inputs and weights are matrices, the network computes outputs using a
**matrix product**, which performs dot products across all samples and all neurons at once.
This results in a matrix of outputs and enables scalable and efficient neural network training.

Watch these animations to get a better understanding:
 * [Example of what an array of a batch of samples looks like, compared to a single sample](https://nnfs.io/lqw/)
 * [How batches can help with fitment](https://nnfs.io/vyu/)

To operate in batches you need to understand *Matrix Product* and *Transposition of Matrix*.

### A Layer of Neurons & Batch of Data with NumPy

When inputs are provided as a **batch** (a matrix of samples) and weights represent a **layer of
neurons** (also a matrix), computing outputs requires a **matrix product** rather than a simple
vector dot product.

To perform this correctly, the **weights matrix must be transposed** so that its rows (neurons)
become columns. This alignment ensures that:
- each **input sample (row)** is dotted with
- each **neuron’s weight vector (column)**

The matrix product between the input matrix and the **transposed weights** produces a matrix
where:
- each row corresponds to one input sample
- each column corresponds to one neuron’s output

After computing this matrix of dot products, a **bias vector** (one bias per neuron) is added.
NumPy applies this addition row-wise, meaning each neuron’s bias is added to its output across
all samples.

This approach produces outputs that are **sample-oriented**, which is essential because neural
networks pass batches of samples from one layer to the next. This explains why deep learning
libraries always accept and return batches, even when working with a single input.

See implementation of [ALayerOfNeuronAndBatchDataWithNumPy.py](./ALayerOfNeuronAndBatchDataWithNumPy.py) and following animations.

* [Why we need to transpose weights](https://nnfs.io/crq/)
* [Matrix product with row and column vectors with a batch of inputs to the neural network](https://nnfs.io/gjw/)
* [Adding biases after the matrix product from a batch of inputs](https://nnfs.io/qty/)