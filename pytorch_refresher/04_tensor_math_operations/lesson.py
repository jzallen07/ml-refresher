"""
================================================================================
LESSON 4: PyTorch Tensor Math and Boolean Operations
================================================================================

This lesson covers mathematical operations on PyTorch tensors, including:
- Statistical operations (min, max, mean, std)
- Element-wise operations (addition, subtraction, multiplication, division)
- Dot products and matrix multiplication
- Dimension-wise operations

Source: Based on TK's article on PyTorch tensor operations
Author: Machine Learning Refresher Course
================================================================================
"""

import torch
import sys

print("=" * 80)
print("LESSON 4: PyTorch Tensor Math and Boolean Operations")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: Statistical Operations - min() and max()
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Statistical Operations - min() and max()")
print("=" * 80)
print()

# The min() function finds the smallest value in a tensor
# It works on both 1D and multi-dimensional tensors
print("1.1 Finding minimum values with min()")
print("-" * 40)

# Example 1: min() on a 1D tensor
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(f"1D Tensor: {tensor_1d}")
min_value = tensor_1d.min()
print(f"Minimum value: {min_value}")
print(f"Result type: {type(min_value)}")  # Returns a tensor, not a Python number
print()

# Example 2: min() on a 2D tensor (matrix)
# When applied to a multi-dimensional tensor, min() returns the global minimum
tensor_2d = torch.tensor([
    [16, 32, 73, 94, 75],
    [31, 42, 23, 84, 35],
    [41, 52, 13, 74, 55]
])
print(f"2D Tensor (3x5 matrix):")
print(tensor_2d)
min_value_2d = tensor_2d.min()
print(f"Global minimum value: {min_value_2d}")
print(f"Explanation: Scans all 15 elements and finds 13 as the smallest")
print()

# The max() function finds the largest value in a tensor
print("1.2 Finding maximum values with max()")
print("-" * 40)

# Example 1: max() on a 1D tensor
print(f"1D Tensor: {tensor_1d}")
max_value = tensor_1d.max()
print(f"Maximum value: {max_value}")
print()

# Example 2: max() on a 2D tensor
print(f"2D Tensor (3x5 matrix):")
print(tensor_2d)
max_value_2d = tensor_2d.max()
print(f"Global maximum value: {max_value_2d}")
print(f"Explanation: Scans all 15 elements and finds 94 as the largest")
print()

# ============================================================================
# SECTION 2: Statistical Operations - mean() and std()
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Statistical Operations - mean() and std()")
print("=" * 80)
print()

# mean() calculates the average of all elements
# std() calculates the standard deviation (measure of spread/variability)
# IMPORTANT: These functions require floating-point tensors!

print("2.1 Calculating mean (average)")
print("-" * 40)

data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
print(f"Data: {data}")
mean_value = data.mean()
print(f"Mean: {mean_value}")
print(f"Manual calculation: (10 + 20 + 30 + 40 + 50) / 5 = {(10 + 20 + 30 + 40 + 50) / 5}")
print()

print("2.2 Calculating standard deviation")
print("-" * 40)
print(f"Data: {data}")
std_value = data.std()
print(f"Standard deviation: {std_value}")
print("Standard deviation measures how spread out the values are from the mean")
print("A smaller std means values are clustered near the mean")
print("A larger std means values are more spread out")
print()

# Important note about data types
print("2.3 Important: mean() and std() require float tensors")
print("-" * 40)
try:
    integer_tensor = torch.tensor([1, 2, 3, 4, 5])
    print(f"Integer tensor: {integer_tensor}")
    print(f"Data type: {integer_tensor.dtype}")
    # This will raise an error!
    # mean_int = integer_tensor.mean()
    print("Attempting to call mean() on integer tensor will raise an error!")
    print("Solution: Convert to float first")
    float_tensor = integer_tensor.float()
    print(f"Converted to float: {float_tensor}")
    print(f"Mean of float tensor: {float_tensor.mean()}")
except Exception as e:
    print(f"Error: {e}")
print()

# ============================================================================
# SECTION 3: Element-wise Operations
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Element-wise Operations")
print("=" * 80)
print()

# Element-wise operations apply the operation to corresponding elements
# Tensors must have compatible shapes (same shape or broadcastable)

print("3.1 Element-wise Addition")
print("-" * 40)
t1 = torch.tensor([1, 0])
t2 = torch.tensor([0, 1])
print(f"Tensor 1: {t1}")
print(f"Tensor 2: {t2}")
result_add = t1 + t2
print(f"t1 + t2 = {result_add}")
print("Breakdown: [1+0, 0+1] = [1, 1]")
print()

print("3.2 Element-wise Subtraction")
print("-" * 40)
t3 = torch.tensor([5, 8, 3])
t4 = torch.tensor([2, 3, 1])
print(f"Tensor 3: {t3}")
print(f"Tensor 4: {t4}")
result_sub = t3 - t4
print(f"t3 - t4 = {result_sub}")
print("Breakdown: [5-2, 8-3, 3-1] = [3, 5, 2]")
print()

print("3.3 Element-wise Multiplication")
print("-" * 40)
t5 = torch.tensor([1, 2])
t6 = torch.tensor([3, 4])
print(f"Tensor 5: {t5}")
print(f"Tensor 6: {t6}")
result_mul = t5 * t6
print(f"t5 * t6 = {result_mul}")
print("Breakdown: [1*3, 2*4] = [3, 8]")
print("Note: This is NOT matrix multiplication, just element-by-element multiplication")
print()

print("3.4 Element-wise Division")
print("-" * 40)
t7 = torch.tensor([10.0, 20.0, 30.0])
t8 = torch.tensor([2.0, 4.0, 5.0])
print(f"Tensor 7: {t7}")
print(f"Tensor 8: {t8}")
result_div = t7 / t8
print(f"t7 / t8 = {result_div}")
print("Breakdown: [10/2, 20/4, 30/5] = [5.0, 5.0, 6.0]")
print()

print("3.5 Element-wise operations on 2D tensors")
print("-" * 40)
matrix_a = torch.tensor([[1, 2], [3, 4]])
matrix_b = torch.tensor([[5, 6], [7, 8]])
print(f"Matrix A:\n{matrix_a}")
print(f"Matrix B:\n{matrix_b}")
print(f"A + B:\n{matrix_a + matrix_b}")
print(f"A * B (element-wise):\n{matrix_a * matrix_b}")
print("Note: Each element is operated on independently")
print()

# ============================================================================
# SECTION 4: Dot Product
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Dot Product")
print("=" * 80)
print()

# The dot product is a fundamental operation in linear algebra
# For two vectors, it's the sum of the products of corresponding elements
# Result is a SCALAR (single number), not a tensor!

print("4.1 Basic Dot Product")
print("-" * 40)
vec1 = torch.tensor([1, 2, 3])
vec2 = torch.tensor([4, 5, 6])
print(f"Vector 1: {vec1}")
print(f"Vector 2: {vec2}")

dot_result = torch.dot(vec1, vec2)
print(f"Dot product: {dot_result}")
print()

print("Step-by-step calculation:")
print("-" * 40)
print(f"Step 1: Multiply corresponding elements")
print(f"  1 × 4 = {1 * 4}")
print(f"  2 × 5 = {2 * 5}")
print(f"  3 × 6 = {3 * 6}")
print()
print(f"Step 2: Sum all the products")
print(f"  {1 * 4} + {2 * 5} + {3 * 6} = {1 * 4 + 2 * 5 + 3 * 6}")
print()

print("4.2 Geometric Interpretation")
print("-" * 40)
print("The dot product measures how 'aligned' two vectors are:")
print("- Large positive value: vectors point in similar directions")
print("- Zero: vectors are perpendicular (orthogonal)")
print("- Large negative value: vectors point in opposite directions")
print()

# Example with perpendicular vectors
perpendicular_1 = torch.tensor([1.0, 0.0])
perpendicular_2 = torch.tensor([0.0, 1.0])
print(f"Perpendicular vectors: {perpendicular_1} and {perpendicular_2}")
print(f"Dot product: {torch.dot(perpendicular_1, perpendicular_2)}")
print("Result is 0 because vectors are perpendicular")
print()

print("4.3 Important Requirements")
print("-" * 40)
print("1. Both tensors must be 1-dimensional (vectors)")
print("2. Both tensors must have the SAME length")
print("3. Cannot use dot() for matrices - use mm() or matmul() instead")
print()

# ============================================================================
# SECTION 5: Matrix Multiplication
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Matrix Multiplication (torch.mm)")
print("=" * 80)
print()

# Matrix multiplication is fundamental to neural networks!
# Rule: For A × B, the number of columns in A must equal the number of rows in B
# Result shape: (A_rows × B_cols)

print("5.1 Basic Matrix Multiplication")
print("-" * 40)
matrix1 = torch.tensor([[0, 1, 1],
                        [1, 0, 1]])
matrix2 = torch.tensor([[1, 1],
                        [1, 0],
                        [0, 1]])

print(f"Matrix 1 (2×3):")
print(matrix1)
print(f"Shape: {matrix1.shape}")
print()

print(f"Matrix 2 (3×2):")
print(matrix2)
print(f"Shape: {matrix2.shape}")
print()

result_mm = torch.mm(matrix1, matrix2)
print(f"Result of matrix1 @ matrix2 (2×2):")
print(result_mm)
print(f"Shape: {result_mm.shape}")
print()

print("5.2 Step-by-step Matrix Multiplication Breakdown")
print("-" * 40)
print("Matrix multiplication: Each element in the result is a dot product")
print()

print("Calculating result[0, 0] (row 0 of matrix1, column 0 of matrix2):")
print(f"  Row 0 of matrix1: {matrix1[0].tolist()}")
print(f"  Column 0 of matrix2: {matrix2[:, 0].tolist()}")
print(f"  Dot product: 0×1 + 1×1 + 1×0 = {0*1 + 1*1 + 1*0}")
print()

print("Calculating result[0, 1] (row 0 of matrix1, column 1 of matrix2):")
print(f"  Row 0 of matrix1: {matrix1[0].tolist()}")
print(f"  Column 1 of matrix2: {matrix2[:, 1].tolist()}")
print(f"  Dot product: 0×1 + 1×0 + 1×1 = {0*1 + 1*0 + 1*1}")
print()

print("Calculating result[1, 0] (row 1 of matrix1, column 0 of matrix2):")
print(f"  Row 1 of matrix1: {matrix1[1].tolist()}")
print(f"  Column 0 of matrix2: {matrix2[:, 0].tolist()}")
print(f"  Dot product: 1×1 + 0×1 + 1×0 = {1*1 + 0*1 + 1*0}")
print()

print("Calculating result[1, 1] (row 1 of matrix1, column 1 of matrix2):")
print(f"  Row 1 of matrix1: {matrix1[1].tolist()}")
print(f"  Column 1 of matrix2: {matrix2[:, 1].tolist()}")
print(f"  Dot product: 1×1 + 0×0 + 1×1 = {1*1 + 0*0 + 1*1}")
print()

print("Final result:")
print(result_mm)
print()

print("5.3 Shape Compatibility Rules")
print("-" * 40)
print("For matrix multiplication A @ B:")
print("✓ A.shape = (m, n) and B.shape = (n, p) → Result.shape = (m, p)")
print("✗ If A's columns ≠ B's rows → Shape mismatch error!")
print()

# Demonstrate shape mismatch
print("Example of incompatible shapes:")
incompatible_1 = torch.tensor([[1, 2], [3, 4]])  # 2×2
incompatible_2 = torch.tensor([[5, 6, 7], [8, 9, 10], [11, 12, 13]])  # 3×3
print(f"Matrix 1 shape: {incompatible_1.shape} (2×2)")
print(f"Matrix 2 shape: {incompatible_2.shape} (3×3)")
print("Cannot multiply: 2 columns ≠ 3 rows")
try:
    torch.mm(incompatible_1, incompatible_2)
except RuntimeError as e:
    print(f"Error: {e}")
print()

print("5.4 Alternative: @ operator for matrix multiplication")
print("-" * 40)
print("PyTorch also supports the @ operator (Python 3.5+)")
result_at = matrix1 @ matrix2
print(f"matrix1 @ matrix2:")
print(result_at)
print("This is equivalent to torch.mm(matrix1, matrix2)")
print()

# ============================================================================
# SECTION 6: Dimension-wise Operations
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: Dimension-wise Operations (using dim parameter)")
print("=" * 80)
print()

# Many operations support a 'dim' parameter to operate along specific dimensions
# dim=0 operates along rows (down the columns)
# dim=1 operates along columns (across the rows)

print("6.1 Understanding Dimensions")
print("-" * 40)
sample_matrix = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
print(f"Sample 3×3 matrix:")
print(sample_matrix)
print(f"Shape: {sample_matrix.shape} (3 rows, 3 columns)")
print()
print("dim=0: operates down columns (along rows)")
print("dim=1: operates across rows (along columns)")
print()

print("6.2 Min and Max along dimensions")
print("-" * 40)

# Min along dim=0 (minimum in each column)
min_dim0 = sample_matrix.min(dim=0)
print(f"Minimum along dim=0 (min of each column):")
print(f"  Values: {min_dim0.values}")
print(f"  Indices: {min_dim0.indices}")
print(f"  Explanation: [10, 20, 30] are the smallest in each column")
print()

# Min along dim=1 (minimum in each row)
min_dim1 = sample_matrix.min(dim=1)
print(f"Minimum along dim=1 (min of each row):")
print(f"  Values: {min_dim1.values}")
print(f"  Indices: {min_dim1.indices}")
print(f"  Explanation: [10, 40, 70] are the smallest in each row")
print()

# Max along dim=0
max_dim0 = sample_matrix.max(dim=0)
print(f"Maximum along dim=0 (max of each column):")
print(f"  Values: {max_dim0.values}")
print(f"  Indices: {max_dim0.indices}")
print()

# Max along dim=1
max_dim1 = sample_matrix.max(dim=1)
print(f"Maximum along dim=1 (max of each row):")
print(f"  Values: {max_dim1.values}")
print(f"  Indices: {max_dim1.indices}")
print()

print("6.3 Sum and Mean along dimensions")
print("-" * 40)
data_matrix = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])
print(f"Data matrix:")
print(data_matrix)
print()

# Sum along dimensions
sum_dim0 = data_matrix.sum(dim=0)
sum_dim1 = data_matrix.sum(dim=1)
print(f"Sum along dim=0 (sum each column): {sum_dim0}")
print(f"  Column 0: 1+4+7 = 12")
print(f"  Column 1: 2+5+8 = 15")
print(f"  Column 2: 3+6+9 = 18")
print()

print(f"Sum along dim=1 (sum each row): {sum_dim1}")
print(f"  Row 0: 1+2+3 = 6")
print(f"  Row 1: 4+5+6 = 15")
print(f"  Row 2: 7+8+9 = 24")
print()

# Mean along dimensions
mean_dim0 = data_matrix.mean(dim=0)
mean_dim1 = data_matrix.mean(dim=1)
print(f"Mean along dim=0 (average of each column): {mean_dim0}")
print(f"Mean along dim=1 (average of each row): {mean_dim1}")
print()

print("6.4 Practical Use Case: Batch Processing")
print("-" * 40)
print("In neural networks, dim=0 often represents the batch dimension")
print("Example: Processing a batch of images")
print()

# Simulated batch of 4 samples, each with 3 features
batch_data = torch.tensor([
    [0.5, 0.8, 0.2],  # Sample 1
    [0.6, 0.7, 0.3],  # Sample 2
    [0.4, 0.9, 0.1],  # Sample 3
    [0.7, 0.6, 0.4],  # Sample 4
])
print(f"Batch of 4 samples (4×3):")
print(batch_data)
print()

# Calculate mean of each feature across the batch
feature_means = batch_data.mean(dim=0)
print(f"Mean of each feature across all samples (dim=0):")
print(f"  {feature_means}")
print("  This is useful for normalization and batch statistics")
print()

# Calculate mean of features for each sample
sample_means = batch_data.mean(dim=1)
print(f"Mean of features for each sample (dim=1):")
print(f"  {sample_means}")
print("  Each value is the average of that sample's features")
print()

# ============================================================================
# SECTION 7: Practice Problems
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: Practice Problems")
print("=" * 80)
print()

print("Now it's your turn! Try to solve these problems:")
print()

print("=" * 80)
print("PROBLEM 1: Statistical Analysis")
print("=" * 80)
print("""
Given the following tensor representing test scores:
scores = torch.tensor([78.5, 92.0, 85.5, 88.0, 95.5, 73.0, 89.5])

Tasks:
a) Find the minimum score
b) Find the maximum score
c) Calculate the mean (average) score
d) Calculate the standard deviation
e) Count how many scores are above the mean (Hint: use comparison operators)

Uncomment the code below and fill in the solutions:
""")

scores = torch.tensor([78.5, 92.0, 85.5, 88.0, 95.5, 73.0, 89.5])
print(f"Scores: {scores}")
print()

# # SOLUTION CODE (uncomment to test):
# print("Solutions:")
# print(f"a) Minimum score: {scores.min()}")
# print(f"b) Maximum score: {scores.max()}")
# print(f"c) Mean score: {scores.mean()}")
# print(f"d) Standard deviation: {scores.std()}")
# above_mean = scores > scores.mean()
# print(f"e) Scores above mean: {above_mean.sum()} students")
print()

print("=" * 80)
print("PROBLEM 2: Matrix Operations")
print("=" * 80)
print("""
Given two matrices:
A = torch.tensor([[2, 4], [6, 8]])
B = torch.tensor([[1, 3], [5, 7]])

Tasks:
a) Perform element-wise addition: A + B
b) Perform element-wise multiplication: A * B
c) Perform matrix multiplication: A @ B
d) Calculate the sum of each row in matrix A (use dim parameter)
e) Calculate the mean of each column in matrix B (use dim parameter)

Uncomment the code below and fill in the solutions:
""")

A = torch.tensor([[2, 4], [6, 8]])
B = torch.tensor([[1, 3], [5, 7]])
print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
print()

# # SOLUTION CODE (uncomment to test):
# print("Solutions:")
# print(f"a) A + B:\n{A + B}")
# print(f"b) A * B (element-wise):\n{A * B}")
# print(f"c) A @ B (matrix multiplication):\n{A @ B}")
# print(f"d) Sum of each row in A: {A.sum(dim=1)}")
# print(f"e) Mean of each column in B: {B.float().mean(dim=0)}")
print()

print("=" * 80)
print("PROBLEM 3: Neural Network Weight Initialization")
print("=" * 80)
print("""
Simulate a simple neural network layer calculation:
- Input vector: x = torch.tensor([1.0, 2.0, 3.0])
- Weight matrix: W = torch.tensor([[0.5, 0.2], [0.8, 0.3], [0.1, 0.9]])
- Bias vector: b = torch.tensor([0.1, 0.2])

Tasks:
a) Calculate the linear transformation: y = W.T @ x + b
   (Note: W.T means transpose of W)
b) Verify the output shape is (2,) - a 2-element vector
c) Apply a simple activation: clip all negative values to 0 (ReLU-like)
   Hint: Use torch.clamp(tensor, min=0)
d) Calculate the sum of the activated output

Uncomment the code below and fill in the solutions:
""")

x = torch.tensor([1.0, 2.0, 3.0])
W = torch.tensor([[0.5, 0.2], [0.8, 0.3], [0.1, 0.9]])
b = torch.tensor([0.1, 0.2])
print(f"Input x: {x}")
print(f"Weight matrix W:\n{W}")
print(f"Bias b: {b}")
print()

# # SOLUTION CODE (uncomment to test):
# print("Solutions:")
# y = W.T @ x + b
# print(f"a) Linear transformation y = W.T @ x + b: {y}")
# print(f"b) Output shape: {y.shape}")
# activated = torch.clamp(y, min=0)
# print(f"c) After activation (ReLU-like): {activated}")
# print(f"d) Sum of activated output: {activated.sum()}")
print()

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "=" * 80)
print("LESSON COMPLETE!")
print("=" * 80)
print()
print("Key Takeaways:")
print("1. Statistical operations (min, max, mean, std) work on entire tensors")
print("2. Element-wise operations apply operations to corresponding elements")
print("3. Dot product creates a scalar from two vectors")
print("4. Matrix multiplication follows strict shape rules: (m×n) @ (n×p) = (m×p)")
print("5. Dimension-wise operations (dim parameter) enable row/column operations")
print("6. These operations are fundamental to neural networks and deep learning")
print()
print("Next steps:")
print("- Complete the practice problems above")
print("- Experiment with different tensor shapes")
print("- Try combining multiple operations in sequence")
print("- Explore torch.matmul() for more flexible matrix operations")
print()
print("=" * 80)
