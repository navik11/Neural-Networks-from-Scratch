*Source: [**Neural Networks from Scratch in Python**](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła*

# Chapter 3: Adding Layers

Neural networks become **deep** when they contain **two or more hidden layers**. A **hidden
layer** is any layer that lies between the input and output layers. Although their internal values
are not part of the final output, they play a critical role in learning and can be inspected for
debugging and improvement.

Each layer consists of a set of neurons, and the structure of a layer is defined by its **weights**
and **biases**. The number of neurons in a layer determines how many weight sets and biases it
has. For a layer to connect correctly to the next one, the number of weights per neuron must
match the number of neurons in the previous layer.

Neural networks process data sequentially through layers. The **output of one layer becomes
the input to the next**, allowing the network to build increasingly abstract representations of
the data. By stacking layers in this manner, neural networks gain the ability to model complex
patterns and relationships, which is the foundation of deep learning.

### Training Data

Instead of manually creating random inputs, neural networks are often trained using **non-linear
datasets**, which cannot be accurately represented by a straight line. **Linear data** can be
modeled easily with simple algorithms, while **non-linear data** requires more powerful models
such as neural networks.

To conveniently generate non-linear data for learning and experimentation, the **`nnfs`**
package is used. It provides utility functions that create complex datasets and ensures
**reproducibility** by fixing random seeds, default data types, and NumPy behavior.

One such utility generates **spiral-shaped datasets** with multiple classes. These datasets are
useful because they are visually separable but mathematically difficult for simple classifiers to
solve, making them ideal for demonstrating the strength of neural networks.

In these datasets:
- Each point represents a **sample**
- Coordinates represent **features**
- Each sample belongs to a **class**, encoded numerically for the model

Such data helps illustrate why deeper neural networks are needed to learn complex,
non-linear decision boundaries.

Run [SpiralDataset.py](./SpiralDataset.py) to visualize the spiral dataset.

### Dense Layer Class

To avoid hardcoding neural network calculations, a **Dense (fully connected) layer** is
implemented as a reusable class. Dense layers are the most common type of neural network
layer, where every neuron is connected to every input.

The `Layer_Dense` class consists of two main parts:
- **Initialization (`__init__`)**: sets up the trainable parameters
- **Forward pass (`forward`)**: computes the layer’s output

Weights are typically **initialized randomly**, while biases are commonly **initialized to zero**.
Random initialization helps break symmetry between neurons, and keeping initial values small
(preferably close to zero) stabilizes training. Biases are stored as a row vector so they can be
added easily during computation.

Random weights are generated using a **Gaussian distribution** centered at zero and scaled
down (e.g., multiplied by `0.01`) to prevent excessively large starting values. This helps ensure
that early training steps are effective and stable. Biases initialized to zero still allow neurons
to activate once learning begins.

The **forward pass** performs a dot product between inputs and weights, then adds the biases.
This operation computes the output of the layer and prepares it to be passed to the next layer.

Encapsulating this logic into a class makes neural networks **modular, readable, and scalable**,
allowing multiple layers to be stacked easily and reused throughout a model.

See [DenceLayerClass](./DenceLayerClass.py) implementation to get a better understanding.
