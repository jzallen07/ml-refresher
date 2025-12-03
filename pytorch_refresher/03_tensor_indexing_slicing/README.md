# Lesson 03: Tensor Operations - Indexing & Slicing

## Overview

Indexing and slicing allow you to access specific elements or subsets of tensors. These operations work similarly to Python lists and NumPy arrays, making them intuitive for Python developers. Mastering these operations is essential for data manipulation in ML pipelines.

> **Source**: This lesson is based on the "Indexing & Slicing Operations" section of [Mastering PyTorch: From Linear Regression to Computer Vision](https://www.iamtk.co/mastering-pytorch-from-linear-regression-to-com) by TK.

## Learning Objectives

By the end of this lesson, you will be able to:

1. Access individual tensor elements using indexing
2. Extract tensor subsets using slicing
3. Combine row and column slicing operations
4. Use negative indexing to access elements from the end

## Key Concepts

### Basic Indexing

Access elements using square brackets, similar to Python lists:

```python
tensor = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# Two equivalent ways to get element at row 1, column 2
tensor[1, 2]     # tensor(7)
tensor[1][2]    # tensor(7)
```

### Negative Indexing

Use negative indices to count from the end:

```python
last_row = tensor[-1]  # tensor([9, 10, 11, 12])
```

### Slicing

Extract ranges using `start:stop` syntax:

```python
# Get columns 2 onwards from row 1
tensor[1][2:]  # tensor([7, 8])

# Get first 2 rows, columns 2 onwards
tensor[:2, 2:]
# tensor([[3, 4],
#         [7, 8]])
```

### Slicing Syntax

- `tensor[start:stop]` - Elements from `start` to `stop-1`
- `tensor[start:]` - Elements from `start` to end
- `tensor[:stop]` - Elements from beginning to `stop-1`
- `tensor[::step]` - Every `step`-th element

## Code Walkthrough

Run the lesson code to see these concepts in action:

```bash
python lesson.py
```

The code demonstrates:
1. Single element access with indexing
2. Row and column extraction
3. Range extraction with slicing
4. Combined row-column slicing
5. Step-based slicing

## Practice Problems

After completing the main lesson, try these exercises:

1. **Every Other Row**: Given a matrix, extract every other row (rows 0, 2, 4, ...) using step slicing.

2. **Diagonal Elements**: Extract the diagonal elements of a square matrix using a loop and indexing (bonus: find the one-liner using `torch.diag()`).

3. **Sliding Window**: Given a 1D tensor, implement a sliding window that extracts all consecutive pairs of elements (e.g., `[1,2,3,4]` → `[[1,2], [2,3], [3,4]]`).

## Key Takeaways

- Indexing in PyTorch works like Python lists: `tensor[row, col]` or `tensor[row][col]`
- Negative indices count from the end: `tensor[-1]` gets the last element
- Slicing extracts ranges: `tensor[start:stop]`
- Combine row and column slices: `tensor[:2, 2:]`
- These operations return views (not copies) when possible, sharing memory with the original tensor

## Next Lesson

[Lesson 04: Tensor Operations - Math & Boolean Operations](../04_tensor_math_operations/README.md) - Learn statistical and mathematical operations on tensors.
