*Source: [**Neural Networks from Scratch in Python**](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła*

# Chapter 1: Introducing Neural Networks

Artificial neural networks are computer models inspired by the human brain. While they are not exact biological replicas, they mimic key ideas such as neurons, activations, and dense interconnections. A single neuron has limited capability, but when many neurons are connected in layers, their collective interactions enable the network to learn complex patterns and often outperform traditional machine learning methods.

Weights and biases act like adjustable controls that shape how a neural network fits data. During
training, an optimizer tunes thousands or even millions of these parameters. Although both
influence a neuron’s output, they play different roles: **weights**, which are multiplied with
inputs, control the strength and direction of the influence, while **biases** shift the output
independently. Together, they resemble the slope and intercept in the linear equation  
\( y = mx + b \).

Watch these animations from the book itself:

* [How weights and biases impact a single neuron](https://nnfs.io/bru)
* [Visualization of an example Dogs vs Cats neural network classifier](https://nnfs.io/qtb/)
* [The math behind an example forward pass through a neural network](https://nnfs.io/vkt/)

Neural networks contain thousands to millions of adjustable parameters, known as **weights**
and **biases**, making them extremely large mathematical functions. With neurons arranged in
interconnected layers, certain combinations of these parameters can produce correct outputs.
The main challenge in training is discovering those optimal parameter values.

The goal of training is to adjust weights and biases so the network performs well on **unseen
data**, not just the examples it was trained on. A key risk is **overfitting**, where the model
memorizes training data instead of learning meaningful patterns. To prevent this, data is split
into **training (in-sample)** and **validation (out-of-sample)** sets.

This ability to perform well on new data is called **generalization**. Neural networks achieve
this by repeatedly measuring error (**loss**) and gradually updating parameters to reduce it.
While commonly used for **classification**, neural networks are also capable of **regression,
clustering**, and many other tasks.
