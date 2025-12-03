# Lesson 02: Tensor Operations - Reshaping

## Overview

Reshaping operations are essential in machine learning pipelines. Data often needs to be transformed to match model input requirements, fix shape mismatches, or prepare data for specific operations. In this lesson, we'll learn the core reshaping functions: `unsqueeze()`, `squeeze()`, and `view()`.

> **Source**: This lesson is based on the "Reshaping Operations" section of [Mastering PyTorch: From Linear Regression to Computer Vision](https://www.iamtk.co/mastering-pytorch-from-linear-regression-to-com) by TK.

## Learning Objectives

By the end of this lesson, you will be able to:

1. Add dimensions to tensors using `unsqueeze()`
2. Remove dimensions from tensors using `squeeze()`
3. Flexibly reshape tensors using `view()`
4. Use the `-1` placeholder for dynamic sizing

## Key Concepts

### Why Reshape?

Common scenarios requiring reshaping:
- Adding a batch dimension to a single sample before passing to a model
- Flattening feature maps from a CNN before a fully connected layer
- Fixing shape mismatch errors between model and input data
- Converting between channel-first (PyTorch) and channel-last (TensorFlow) formats

### `unsqueeze()` - Adding Dimensions

Adds a new dimension of size 1 at the specified index.

```python
tensor = torch.tensor(1)
batch = tensor.unsqueeze(0)

tensor.shape  # torch.Size([])
batch.shape   # torch.Size([1])
```

### `squeeze()` - Removing Dimensions

Removes dimensions of size 1.

```python
tensor = torch.tensor([1.0])
squeezed = tensor.squeeze(0)

tensor.shape        # torch.Size([1])
squeezed.shape      # torch.Size([])
```

### `view()` - Flexible Reshaping

Reshapes a tensor to the specified dimensions. The total number of elements must remain the same.

```python
tensor = torch.tensor([0, 1, 2, 3, 4])  # Shape: [5]
reshaped = tensor.view(5, 1)             # Shape: [5, 1]
```

### The `-1` Placeholder

Use `-1` to let PyTorch infer the size of one dimension:

```python
tensor = torch.tensor([0, 1, 2, 3, 4])
reshaped = tensor.view(-1, 1)  # PyTorch infers first dim as 5
```

## Code Walkthrough

Run the lesson code to see these concepts in action:

```bash
python lesson.py
```

The code demonstrates:
1. Adding dimensions with `unsqueeze()` at different positions
2. Removing dimensions with `squeeze()`
3. Reshaping with `view()` using explicit and inferred dimensions
4. Common real-world reshaping scenarios

## Practice Problems

After completing the main lesson, try these exercises:

1. **Image Format Conversion**: Create a tensor with shape `(N, H, W, C)` representing a batch of images in channel-last format. Reshape it to `(N, C, H, W)` (channel-first, PyTorch's preferred format) using `permute()` and `view()`.

2. **Batch Dimension**: Given a single image tensor of shape `(3, 224, 224)`, add a batch dimension to make it `(1, 3, 224, 224)` suitable for model input.

3. **Flatten and Restore**: Create a 3D tensor of shape `(2, 3, 4)`, flatten it completely, then reshape it back to the original dimensions.

## Key Takeaways

- `unsqueeze()` adds a dimension of size 1 at the specified index
- `squeeze()` removes dimensions of size 1
- `view()` reshapes tensors while preserving total element count
- Use `-1` to let PyTorch infer one dimension automatically
- These operations are essential for fixing shape mismatches in ML pipelines

## Next Lesson

[Lesson 03: Tensor Operations - Indexing & Slicing](../03_tensor_indexing_slicing/README.md) - Learn how to access and extract portions of tensors.
