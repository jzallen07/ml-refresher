# Lesson 01: Tensors - The Data Building Blocks

## Overview

A tensor is a fundamental piece of PyTorch and how it's used to build and train models. In this lesson, we'll learn how tensors hold data, how to extract information about them, and how to convert between tensors and other data structures like NumPy arrays and Pandas DataFrames.

> **Source**: This lesson is based on the "Tensors, the data building blocks" section of [Mastering PyTorch: From Linear Regression to Computer Vision](https://www.iamtk.co/mastering-pytorch-from-linear-regression-to-com) by TK.

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand what tensors are and their role in PyTorch
2. Create tensors from scalars, vectors, and matrices
3. Explore tensor properties: `dtype`, `ndimension()`, `size()`, `shape`, `item()`
4. Convert between NumPy arrays, Pandas DataFrames, and PyTorch tensors

## Key Concepts

### What is a Tensor?

A tensor is PyTorch's primary data structure for holding and manipulating data. Think of it as a generalization of arrays:
- A **scalar** is a 0-dimensional tensor (single value)
- A **vector** is a 1-dimensional tensor (list of values)
- A **matrix** is a 2-dimensional tensor (grid of values)
- Higher dimensions follow the same pattern

### Creating Tensors

```python
import torch

# Scalar (0D tensor)
value = torch.tensor(1)

# Vector (1D tensor)
vector = torch.tensor([1, 2, 3, 4, 5])

# Matrix (2D tensor)
matrix = torch.tensor([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]
])
```

### Tensor Properties

- `dtype` - The data type of elements (e.g., `torch.int64`, `torch.float32`)
- `ndimension()` - Number of dimensions (0 for scalar, 1 for vector, 2 for matrix)
- `size()` / `shape` - The shape of the tensor
- `item()` - Extract the Python value from a single-element tensor

### NumPy and Pandas Interoperability

PyTorch tensors can be easily converted to and from NumPy arrays and Pandas DataFrames:

```python
# NumPy to Tensor
tensor = torch.from_numpy(numpy_array)

# Tensor to NumPy
array = tensor.numpy()

# Pandas to Tensor
tensor = torch.from_numpy(df.values)

# Tensor to Pandas
df = pd.DataFrame(tensor)
```

## Code Walkthrough

Run the lesson code to see these concepts in action:

```bash
python lesson.py
```

The code demonstrates:
1. Creating tensors of different dimensions
2. Inspecting tensor properties
3. Extracting values from tensors
4. Converting between NumPy, Pandas, and PyTorch

## Practice Problems

After completing the main lesson, try these exercises:

1. **3D Tensor Exploration**: Create a 3D tensor (e.g., a batch of images with shape `[batch, height, width]`) and explore its properties using `ndimension()`, `size()`, and `shape`.

2. **Pandas Round-Trip**: Create a Pandas DataFrame with multiple columns, convert it to a tensor, perform a simple operation (like adding 1), and convert it back to a DataFrame.

3. **Memory Comparison**: Create tensors of the same shape but different dtypes (`torch.float64`, `torch.float32`, `torch.int8`) and compare their memory usage using `tensor.element_size() * tensor.numel()`.

## Key Takeaways

- Tensors are the fundamental data structure in PyTorch
- They can hold scalars, vectors, matrices, and higher-dimensional data
- Properties like `dtype`, `shape`, and `ndimension()` help you understand your data
- Seamless conversion to/from NumPy and Pandas enables integration with the broader Python ecosystem

## Next Lesson

[Lesson 02: Tensor Operations - Reshaping](../02_tensor_reshaping/README.md) - Learn how to reshape tensors using `unsqueeze()`, `squeeze()`, and `view()`.
