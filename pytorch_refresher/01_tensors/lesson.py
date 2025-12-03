"""
================================================================================
PYTORCH TENSORS LESSON
================================================================================

This lesson covers the fundamentals of PyTorch tensors, including:
- Creating tensors (scalars, vectors, matrices)
- Tensor properties and attributes
- Extracting values from tensors
- Interoperability with NumPy and Pandas

Source: Based on TK's article on PyTorch tensors
https://www.freecodecamp.org/news/pytorch-101-tensors/

Author: Educational material for ML refresher
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import torch
import numpy as np
import pandas as pd

print("=" * 80)
print("PYTORCH TENSORS: A COMPREHENSIVE GUIDE")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: CREATING TENSORS
# ============================================================================

print("=" * 80)
print("SECTION 1: CREATING TENSORS")
print("=" * 80)
print()

# ----------------------------------------------------------------------------
# Scalar (0-dimensional tensor)
# ----------------------------------------------------------------------------
print("1.1 Creating a Scalar (0-dimensional tensor)")
print("-" * 80)

# A scalar is a single value wrapped in a tensor
value = torch.tensor(1)
print(f"Scalar tensor: {value}")
print(f"Type: {type(value)}")
print()

# Checking the data type of the tensor
# By default, integers are stored as torch.int64 (64-bit integer)
print(f"Data type (dtype): {torch.tensor(1).dtype}")
print()

# ----------------------------------------------------------------------------
# Vector (1-dimensional tensor)
# ----------------------------------------------------------------------------
print("1.2 Creating a Vector (1-dimensional tensor)")
print("-" * 80)

# A vector is a 1D array of values
vector = torch.tensor([1, 2, 3, 4, 5])
print(f"Vector tensor: {vector}")
print(f"Type: {type(vector)}")
print()

# ----------------------------------------------------------------------------
# Matrix (2-dimensional tensor)
# ----------------------------------------------------------------------------
print("1.3 Creating a Matrix (2-dimensional tensor)")
print("-" * 80)

# A matrix is a 2D array with rows and columns
# This creates a 5x3 matrix (5 rows, 3 columns)
matrix = torch.tensor([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7]
])
print(f"Matrix tensor:\n{matrix}")
print(f"Type: {type(matrix)}")
print()

# ============================================================================
# SECTION 2: TENSOR PROPERTIES
# ============================================================================

print("=" * 80)
print("SECTION 2: TENSOR PROPERTIES")
print("=" * 80)
print()

# ----------------------------------------------------------------------------
# 2.1 Data Type (dtype)
# ----------------------------------------------------------------------------
print("2.1 Data Type (dtype)")
print("-" * 80)

# The dtype property tells us what type of data is stored in the tensor
print(f"Scalar dtype: {value.dtype}")
print(f"Vector dtype: {vector.dtype}")
print(f"Matrix dtype: {matrix.dtype}")
print()

# Creating tensors with different data types
float_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"Float tensor: {float_tensor}")
print(f"Float tensor dtype: {float_tensor.dtype}")  # torch.float32
print()

# ----------------------------------------------------------------------------
# 2.2 Number of Dimensions (ndimension)
# ----------------------------------------------------------------------------
print("2.2 Number of Dimensions (ndimension)")
print("-" * 80)

# The ndimension() method returns the number of dimensions (axes) in a tensor
# Scalar = 0D, Vector = 1D, Matrix = 2D, etc.
print(f"Scalar dimensions: {value.ndimension()}")  # 0
print(f"Vector dimensions: {vector.ndimension()}")  # 1
print(f"Matrix dimensions: {matrix.ndimension()}")  # 2
print()

# Alternative: Use .ndim property (same as ndimension())
print(f"Scalar dimensions (using .ndim): {value.ndim}")
print(f"Vector dimensions (using .ndim): {vector.ndim}")
print(f"Matrix dimensions (using .ndim): {matrix.ndim}")
print()

# ----------------------------------------------------------------------------
# 2.3 Size (size method)
# ----------------------------------------------------------------------------
print("2.3 Size (size method)")
print("-" * 80)

# The size() method returns the dimensions of each axis
# It tells us how many elements are in each dimension
print(f"Scalar size: {value.size()}")  # torch.Size([]) - no dimensions
print(f"Vector size: {vector.size()}")  # torch.Size([5]) - 5 elements
print(f"Matrix size: {matrix.size()}")  # torch.Size([5, 3]) - 5 rows, 3 columns
print()

# You can also access specific dimension sizes
print(f"Matrix dimension 0 (rows): {matrix.size(0)}")  # 5 rows
print(f"Matrix dimension 1 (columns): {matrix.size(1)}")  # 3 columns
print()

# ----------------------------------------------------------------------------
# 2.4 Shape (shape property)
# ----------------------------------------------------------------------------
print("2.4 Shape (shape property)")
print("-" * 80)

# The shape property is identical to size() but is more commonly used
# It returns a torch.Size object (which is a tuple-like object)
print(f"Scalar shape: {value.shape}")  # torch.Size([])
print(f"Vector shape: {vector.shape}")  # torch.Size([5])
print(f"Matrix shape: {matrix.shape}")  # torch.Size([5, 3])
print()

# You can access shape dimensions using indexing
print(f"Matrix shape[0] (rows): {matrix.shape[0]}")  # 5
print(f"Matrix shape[1] (columns): {matrix.shape[1]}")  # 3
print()

# ============================================================================
# SECTION 3: EXTRACTING VALUES
# ============================================================================

print("=" * 80)
print("SECTION 3: EXTRACTING VALUES")
print("=" * 80)
print()

# ----------------------------------------------------------------------------
# 3.1 The item() Method
# ----------------------------------------------------------------------------
print("3.1 Using the item() method to extract Python values")
print("-" * 80)

# The item() method extracts a single value from a tensor and converts it
# to a standard Python number (int, float, etc.)
# This ONLY works for tensors with a single element

# Extracting from a scalar
print(f"Scalar tensor: {value}")
print(f"Extracted value: {value.item()}")  # 1
print(f"Type after extraction: {type(value.item())}")  # <class 'int'>
print()

# Extracting from a vector (must index first to get a single element)
print(f"Vector tensor: {vector}")
print(f"First element: {vector[0]}")  # tensor(1)
print(f"Extracted first element: {vector[0].item()}")  # 1
print(f"Type: {type(vector[0].item())}")  # <class 'int'>
print()

# Extracting from a matrix (must index to get a single element)
print(f"Matrix tensor:\n{matrix}")
print(f"Element at [0][0]: {matrix[0][0]}")  # tensor(1)
print(f"Extracted element: {matrix[0][0].item()}")  # 1
print(f"Type: {type(matrix[0][0].item())}")  # <class 'int'>
print()

# ----------------------------------------------------------------------------
# 3.2 Indexing and Slicing
# ----------------------------------------------------------------------------
print("3.2 Indexing and Slicing Tensors")
print("-" * 80)

# You can index and slice tensors just like NumPy arrays
print(f"Vector: {vector}")
print(f"First element: {vector[0]}")
print(f"Last element: {vector[-1]}")
print(f"First three elements: {vector[:3]}")
print()

print(f"Matrix:\n{matrix}")
print(f"First row: {matrix[0]}")
print(f"First column: {matrix[:, 0]}")
print(f"Element at row 2, column 1: {matrix[2, 1]}")
print()

# ============================================================================
# SECTION 4: NUMPY INTEROPERABILITY
# ============================================================================

print("=" * 80)
print("SECTION 4: NUMPY INTEROPERABILITY")
print("=" * 80)
print()

# PyTorch and NumPy work seamlessly together!
# This is important because many data science libraries use NumPy

# ----------------------------------------------------------------------------
# 4.1 NumPy Array to PyTorch Tensor
# ----------------------------------------------------------------------------
print("4.1 Converting NumPy Array to PyTorch Tensor")
print("-" * 80)

# Create a NumPy array
numpy_vector = np.array([1.0, 2.0, 3.0])
print(f"NumPy array: {numpy_vector}")
print(f"NumPy array type: {type(numpy_vector)}")
print(f"NumPy array dtype: {numpy_vector.dtype}")
print()

# Convert NumPy array to PyTorch tensor using torch.from_numpy()
tensor_from_numpy = torch.from_numpy(numpy_vector)
print(f"PyTorch tensor: {tensor_from_numpy}")
print(f"PyTorch tensor type: {type(tensor_from_numpy)}")
print(f"PyTorch tensor dtype: {tensor_from_numpy.dtype}")  # torch.float64
print()

# IMPORTANT: torch.from_numpy() creates a tensor that SHARES MEMORY with the
# original NumPy array. Changes to one will affect the other!
print("Memory sharing example:")
print(f"Original NumPy array: {numpy_vector}")
tensor_from_numpy[0] = 999
print(f"After changing tensor, NumPy array: {numpy_vector}")  # Also changed!
print()

# Reset for next example
numpy_vector[0] = 1.0

# ----------------------------------------------------------------------------
# 4.2 PyTorch Tensor to NumPy Array
# ----------------------------------------------------------------------------
print("4.2 Converting PyTorch Tensor to NumPy Array")
print("-" * 80)

# Create a PyTorch tensor
pytorch_tensor = torch.tensor([1., 2., 3.])
print(f"PyTorch tensor: {pytorch_tensor}")
print(f"PyTorch tensor type: {type(pytorch_tensor)}")
print(f"PyTorch tensor dtype: {pytorch_tensor.dtype}")  # torch.float32
print()

# Convert PyTorch tensor to NumPy array using .numpy()
numpy_from_tensor = pytorch_tensor.numpy()
print(f"NumPy array: {numpy_from_tensor}")
print(f"NumPy array type: {type(numpy_from_tensor)}")
print(f"NumPy array dtype: {numpy_from_tensor.dtype}")  # float32
print()

# Again, memory is shared!
print("Memory sharing example:")
print(f"Original tensor: {pytorch_tensor}")
numpy_from_tensor[0] = 888
print(f"After changing NumPy array, tensor: {pytorch_tensor}")  # Also changed!
print()

# ============================================================================
# SECTION 5: PANDAS INTEROPERABILITY
# ============================================================================

print("=" * 80)
print("SECTION 5: PANDAS INTEROPERABILITY")
print("=" * 80)
print()

# Pandas DataFrames and Series can also be converted to/from PyTorch tensors

# ----------------------------------------------------------------------------
# 5.1 Pandas DataFrame to PyTorch Tensor
# ----------------------------------------------------------------------------
print("5.1 Converting Pandas DataFrame to PyTorch Tensor")
print("-" * 80)

# Create a Pandas DataFrame
df = pd.DataFrame([1.0, 2.0, 3.0])
print(f"Pandas DataFrame:\n{df}")
print(f"DataFrame type: {type(df)}")
print()

# Convert DataFrame to tensor via NumPy (DataFrame -> NumPy -> Tensor)
# We use .values to get the underlying NumPy array
tensor_from_df = torch.from_numpy(df.values)
print(f"PyTorch tensor from DataFrame:\n{tensor_from_df}")
print(f"Tensor shape: {tensor_from_df.shape}")  # torch.Size([3, 1])
print(f"Tensor dtype: {tensor_from_df.dtype}")  # torch.float64
print()

# For a 1D Series
series = pd.Series([1.0, 2.0, 3.0])
print(f"Pandas Series: {series.values}")
tensor_from_series = torch.from_numpy(series.values)
print(f"Tensor from Series: {tensor_from_series}")
print(f"Tensor shape: {tensor_from_series.shape}")  # torch.Size([3])
print()

# ----------------------------------------------------------------------------
# 5.2 PyTorch Tensor to Pandas DataFrame
# ----------------------------------------------------------------------------
print("5.2 Converting PyTorch Tensor to Pandas DataFrame")
print("-" * 80)

# Create a PyTorch tensor
tensor_for_df = torch.tensor([1., 2., 3.])
print(f"PyTorch tensor: {tensor_for_df}")
print()

# Convert tensor to DataFrame
# The tensor is automatically converted to NumPy array internally
df_from_tensor = pd.DataFrame(tensor_for_df.numpy())
print(f"Pandas DataFrame from tensor:\n{df_from_tensor}")
print()

# For 2D tensors
matrix_for_df = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])
df_from_matrix = pd.DataFrame(matrix_for_df.numpy(), columns=['A', 'B', 'C'])
print(f"Pandas DataFrame from 2D tensor:\n{df_from_matrix}")
print()

# ============================================================================
# SECTION 6: PRACTICE PROBLEMS
# ============================================================================

print("=" * 80)
print("SECTION 6: PRACTICE PROBLEMS")
print("=" * 80)
print()

# Uncomment and complete the following exercises to practice what you've learned!

# ----------------------------------------------------------------------------
# Problem 1: Create and Analyze a Tensor
# ----------------------------------------------------------------------------
print("Problem 1: Create and Analyze a Tensor")
print("-" * 80)
print("Task: Create a 3x4 matrix with values from 1 to 12 (reshaped)")
print("Then print its shape, number of dimensions, and dtype")
print()

# Your code here:
# problem1_matrix = torch.tensor([...])
# print(f"Matrix:\n{problem1_matrix}")
# print(f"Shape: {problem1_matrix.shape}")
# print(f"Dimensions: {problem1_matrix.ndim}")
# print(f"Data type: {problem1_matrix.dtype}")

# Solution (commented out - try it yourself first!):
# problem1_matrix = torch.arange(1, 13).reshape(3, 4)
# print(f"Matrix:\n{problem1_matrix}")
# print(f"Shape: {problem1_matrix.shape}")
# print(f"Dimensions: {problem1_matrix.ndim}")
# print(f"Data type: {problem1_matrix.dtype}")
print()

# ----------------------------------------------------------------------------
# Problem 2: NumPy to PyTorch and Back
# ----------------------------------------------------------------------------
print("Problem 2: NumPy to PyTorch and Back")
print("-" * 80)
print("Task: Create a NumPy array with values [10, 20, 30, 40, 50]")
print("Convert it to a PyTorch tensor, multiply all values by 2,")
print("then convert back to NumPy and print the result")
print()

# Your code here:
# numpy_arr = np.array([...])
# tensor = torch.from_numpy(numpy_arr)
# tensor = tensor * 2
# result = tensor.numpy()
# print(f"Result: {result}")

# Solution (commented out):
# numpy_arr = np.array([10, 20, 30, 40, 50])
# tensor = torch.from_numpy(numpy_arr.astype(np.float32))
# tensor = tensor * 2
# result = tensor.numpy()
# print(f"Result: {result}")
print()

# ----------------------------------------------------------------------------
# Problem 3: Extract and Manipulate Values
# ----------------------------------------------------------------------------
print("Problem 3: Extract and Manipulate Values")
print("-" * 80)
print("Task: Create a 4x4 matrix with random integers from 0 to 9")
print("Extract the center 2x2 submatrix and calculate its sum using .item()")
print()

# Your code here:
# matrix = torch.randint(0, 10, (4, 4))
# print(f"Original matrix:\n{matrix}")
# center_submatrix = matrix[1:3, 1:3]
# print(f"Center 2x2:\n{center_submatrix}")
# total_sum = center_submatrix.sum().item()
# print(f"Sum: {total_sum}")

# Solution (commented out):
# torch.manual_seed(42)  # For reproducibility
# matrix = torch.randint(0, 10, (4, 4))
# print(f"Original matrix:\n{matrix}")
# center_submatrix = matrix[1:3, 1:3]
# print(f"Center 2x2:\n{center_submatrix}")
# total_sum = center_submatrix.sum().item()
# print(f"Sum of center elements: {total_sum}")
print()

# ============================================================================
# CONCLUSION
# ============================================================================

print("=" * 80)
print("LESSON COMPLETE!")
print("=" * 80)
print()
print("Key Takeaways:")
print("1. Tensors are the fundamental data structure in PyTorch")
print("2. Tensors can be 0D (scalar), 1D (vector), 2D (matrix), or higher")
print("3. Important properties: dtype, shape, ndim")
print("4. Use .item() to extract single values as Python numbers")
print("5. PyTorch works seamlessly with NumPy and Pandas")
print("6. Memory is shared between NumPy arrays and PyTorch tensors")
print()
print("Next steps: Practice the exercises above and explore tensor operations!")
print("=" * 80)
