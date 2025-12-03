# Lesson 05: Tensor Operations - Derivatives & Gradients

## Overview

Derivatives in tensors are the building blocks of how neural networks learn through gradient descent. PyTorch's autograd system automatically computes gradients, enabling backpropagation without manual calculus. This lesson introduces the fundamental concepts that power all neural network training.

> **Source**: This lesson is based on the "Derivatives in tensor" section of [Mastering PyTorch: From Linear Regression to Computer Vision](https://www.iamtk.co/mastering-pytorch-from-linear-regression-to-com) by TK.

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand autograd and automatic differentiation
2. Create tensors that track gradients with `requires_grad=True`
3. Compute gradients using `backward()`
4. Access computed gradients via the `.grad` attribute
5. Use `torch.no_grad()` to disable gradient tracking

## Key Concepts

### Why Gradients Matter

Neural networks learn by:
1. Making predictions (forward pass)
2. Computing how wrong the predictions are (loss)
3. Computing how to adjust weights to reduce loss (gradients via backpropagation)
4. Updating weights in the direction that reduces loss (gradient descent)

Gradients tell us: "If I increase this weight slightly, how much does the loss change?"

### Creating Gradient-Tracking Tensors

```python
x = torch.tensor(2.0, requires_grad=True)
# tensor(2., requires_grad=True)
```

### Computing Gradients

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2  # y = x²

y.backward()  # Compute gradients

x.grad  # tensor(4.)
# The derivative of x² is 2x
# At x=2: 2*2 = 4
```

### The Computational Graph

When you perform operations on tensors with `requires_grad=True`, PyTorch builds a computational graph tracking all operations. Calling `backward()` traverses this graph in reverse to compute gradients.

```python
y = x ** 2
# y has grad_fn=<PowBackward0> - it knows how it was created
```

### Disabling Gradient Tracking

During inference (making predictions), we don't need gradients:

```python
with torch.no_grad():
    # Operations here won't track gradients
    prediction = model(input)
```

## Code Walkthrough

Run the lesson code to see these concepts in action:

```bash
python lesson.py
```

The code demonstrates:
1. Creating tensors with gradient tracking
2. Building computational graphs
3. Computing gradients with backward()
4. Multi-variable gradient computation
5. Gradient accumulation behavior
6. Using torch.no_grad()

## Practice Problems

After completing the main lesson, try these exercises:

1. **Multi-variable Gradients**: Create a function `z = x² + y³` with `x=2` and `y=3`. Compute the gradients with respect to both x and y. Verify: ∂z/∂x = 2x = 4, ∂z/∂y = 3y² = 27.

2. **Gradient Accumulation**: Call `backward()` twice without zeroing gradients between calls. Observe how gradients accumulate. Then use `x.grad.zero_()` and show the difference.

3. **no_grad Context**: Create a tensor with `requires_grad=True`, perform operations inside `torch.no_grad()`, and verify the result has `requires_grad=False`.

## Key Takeaways

- `requires_grad=True` tells PyTorch to track operations for gradient computation
- `backward()` computes gradients by traversing the computational graph
- `.grad` stores the computed gradient
- Gradients accumulate by default - zero them before each training step
- Use `torch.no_grad()` during inference to save memory and computation
- This automatic differentiation is what makes training neural networks tractable

## Next Lesson

[Lesson 06: Building & Training a Linear Regression Model](../06_linear_regression/README.md) - Apply these gradient concepts to train your first model.
