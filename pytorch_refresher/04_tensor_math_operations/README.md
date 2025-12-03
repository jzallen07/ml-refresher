# Lesson 04: Tensor Operations - Math & Boolean Operations

## Overview

Mathematical and statistical operations on tensors are fundamental to machine learning. From computing loss functions to normalizing data, these operations are used constantly in ML pipelines. This lesson covers statistical functions, element-wise operations, and linear algebra operations essential for neural networks.

> **Source**: This lesson is based on the "Math & Boolean Operations" section of [Mastering PyTorch: From Linear Regression to Computer Vision](https://www.iamtk.co/mastering-pytorch-from-linear-regression-to-com) by TK.

## Learning Objectives

By the end of this lesson, you will be able to:

1. Perform statistical operations: `min()`, `max()`, `mean()`, `std()`
2. Execute element-wise operations: addition, multiplication
3. Understand and use linear algebra operations: dot product, matrix multiplication
4. Apply these operations in ML contexts

## Key Concepts

### Statistical Operations

```python
# min and max
torch.tensor([1, 2, 3, 4, 5]).min()  # tensor(1)
torch.tensor([1, 2, 3, 4, 5]).max()  # tensor(5)

# Works on multi-dimensional tensors too
torch.tensor([
    [16, 32, 73, 94, 75],
    [31, 42, 23, 84, 35],
    [41, 52, 13, 74, 55]
]).min()  # tensor(13)

# mean and standard deviation
data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
data.mean()  # tensor(30.)
data.std()   # tensor(15.8114)
```

### Element-wise Operations

Operations applied to corresponding elements:

```python
# Addition
t1 = torch.tensor([1, 0])
t2 = torch.tensor([0, 1])
t1 + t2  # tensor([1, 1])

# Multiplication (element-wise, NOT matrix multiplication)
t1 = torch.tensor([1, 2])
t2 = torch.tensor([3, 4])
t1 * t2  # tensor([3, 8])
```

### Linear Algebra Operations

#### Dot Product
Multiplies corresponding elements and sums the result:

```python
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])
torch.dot(t1, t2)  # tensor(32)
# 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
```

#### Matrix Multiplication
Follows linear algebra rules - inner dimensions must match:

```python
matrix1 = torch.tensor([[0, 1, 1], [1, 0, 1]])  # Shape: (2, 3)
matrix2 = torch.tensor([[1, 1], [1, 0], [0, 1]])  # Shape: (3, 2)
torch.mm(matrix1, matrix2)  # Shape: (2, 2)
# tensor([[1, 1],
#         [1, 2]])
```

## Code Walkthrough

Run the lesson code to see these concepts in action:

```bash
python lesson.py
```

The code demonstrates:
1. Statistical functions on 1D and multi-dimensional tensors
2. Element-wise arithmetic operations
3. Dot product computation
4. Matrix multiplication with shape considerations

## Practice Problems

After completing the main lesson, try these exercises:

1. **Normalization**: Given a tensor, normalize it to have mean=0 and std=1 using the formula: `(tensor - tensor.mean()) / tensor.std()`.

2. **Cosine Similarity**: Implement cosine similarity between two vectors: `cos_sim = dot(a, b) / (norm(a) * norm(b))`. Use `torch.dot()` and `torch.norm()`.

3. **Matrix Multiplication Shapes**: Create matrices of shapes that cannot be multiplied (e.g., (2,3) and (2,3)). Catch the error and print a helpful message explaining why it failed.

## Key Takeaways

- `min()`, `max()`, `mean()`, `std()` work on entire tensors or along specific dimensions
- Element-wise operations (`+`, `*`) require matching shapes (or broadcasting)
- `torch.dot()` computes dot product for 1D tensors
- `torch.mm()` performs matrix multiplication - inner dimensions must match
- These operations are the building blocks of neural network computations

## Next Lesson

[Lesson 05: Tensor Operations - Derivatives & Gradients](../05_tensor_gradients/README.md) - Learn about automatic differentiation, the foundation of neural network training.
