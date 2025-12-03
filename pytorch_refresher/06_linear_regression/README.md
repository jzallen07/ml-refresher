# Lesson 06: Building & Training a Linear Regression Model

## Overview

Before moving to complex models like CNNs, we take a step back and build the simplest model possible to understand the essential building blocks: how to build ML models and train them in PyTorch. This lesson brings together everything we've learned about tensors, gradients, and operations into a complete training pipeline.

> **Source**: This lesson is based on the "Building and training a Linear Regression" section of [Mastering PyTorch: From Linear Regression to Computer Vision](https://www.iamtk.co/mastering-pytorch-from-linear-regression-to-com) by TK.

## Learning Objectives

By the end of this lesson, you will be able to:

1. Build models using `nn.Linear` and `nn.Module`
2. Understand the training loop: predict → loss → backprop → update
3. Configure loss functions and optimizers
4. Implement a complete training pipeline from scratch

## Key Concepts

### The Linear Regression Model

Linear regression models a relationship: `y = xw + b`

Where:
- `x` is the input
- `w` is the weight (learned)
- `b` is the bias (learned)
- `y` is the prediction

### Using nn.Linear

PyTorch provides `nn.Linear` for linear layers:

```python
model = nn.Linear(in_features=1, out_features=1, bias=True)
model(torch.tensor([[2.], [4.]]))
# if weight = -0.4443 and bias = -0.5045,
# the result will be tensor([[-1.3930], [-2.2815]])
```

### Custom Models with nn.Module

For flexibility, we can create custom models:

```python
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
```

### The Training Loop

As TK explains, training looks like: **predict → calculate loss → adjust parameters → repeat**

More specifically:
1. Zero the gradients in the optimizer
2. Make a prediction (forward pass)
3. Compute the loss
4. Compute the gradients (backward pass)
5. Update the parameters

```python
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(distances)
    loss = loss_function(outputs, times)
    loss.backward()
    optimizer.step()
```

### Why Zero Gradients?

From Lesson 05, we learned that gradients accumulate. The `optimizer.zero_grad()` call ensures we don't carry gradients from previous iterations, which would lead to incorrect parameter updates.

## Code Walkthrough

Run the lesson code to see these concepts in action:

```bash
python lesson.py
```

The code demonstrates:
1. Creating a simple `nn.Linear` model
2. Building a custom `LinearRegression` class
3. Defining loss functions and optimizers
4. Implementing the complete training loop
5. Training on the distance/time dataset from TK's article

## Practice Problems

After completing the main lesson, try these exercises:

1. **Different Dataset**: Create a height/weight dataset and train a linear regression to predict weight from height. Generate synthetic data if needed.

2. **Optimizer Comparison**: Train the same model twice - once with SGD and once with Adam. Plot both loss curves and compare convergence speed.

3. **Loss Visualization**: Modify the training loop to store loss values at each epoch. After training, use matplotlib to plot the loss curve.

## Key Takeaways

- `nn.Linear` provides a simple linear layer: `y = xW + b`
- `nn.Module` is the base class for all neural network models
- The `forward()` method defines how input flows through the model
- Training loop: zero_grad → forward → loss → backward → step
- Loss functions (like MSELoss) measure prediction error
- Optimizers (like SGD, Adam) update parameters based on gradients
- Always zero gradients before each training step

## Next Lesson

[Lesson 07: Data Management - Downloading & Custom Datasets](../07_data_management/README.md) - Learn how to manage real datasets for training.
