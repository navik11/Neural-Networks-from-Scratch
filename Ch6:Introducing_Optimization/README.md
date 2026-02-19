*Source: [**Neural Networks from Scratch in Python**](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła*

# Chapter 6: Introducing Optimization

Once a neural network can **forward data** and **compute loss**, the next goal is to **reduce that loss** by adjusting **weights and biases**. This process is called **optimization**.

### Why Optimization Is Needed
- Loss measures how wrong the model is.
- Training means **changing parameters** so loss decreases.
- Randomly guessing good weights is inefficient and unreliable.

### Naive Approach: Random Search
- Randomly assign new weights and biases.
- Compute loss and keep the best-performing set.
- Result:
  - Loss decreases **very slowly**
  - Accuracy barely improves
  - Even millions of iterations are impractical

**Conclusion:** Pure random search does not scale.

### Slight Improvement: Random Adjustments
Instead of replacing parameters entirely:
- Start from current best weights
- Add **small random changes**
- Keep the update **only if loss decreases**
- Revert if loss increases

This is similar to a **hill-climbing** strategy.

**Outcome:**
- Loss decreases much faster
- Accuracy improves significantly on simple datasets
- Still unreliable for complex data (e.g., spiral dataset)

### Key Observations
- Small, incremental updates are better than full random resets
- Optimization can get **stuck in local minima**
- Data complexity strongly affects learning
- Random-based optimization is still not sufficient

### Core Takeaway
> Random guessing, even with small steps, is not a robust optimization strategy.  
> To reliably train neural networks, we need **gradient-based optimization methods** (covered next).
