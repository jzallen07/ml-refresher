"""
PyTorch Tensor Indexing and Slicing - Comprehensive Lesson

This lesson covers all aspects of indexing and slicing PyTorch tensors,
from basic single-element access to advanced multi-dimensional operations.

Source: Based on TK's article on PyTorch tensor operations
Author: Learning ML Fundamentals
Date: 2025-12-02
"""

import torch

print("=" * 80)
print("PYTORCH TENSOR INDEXING AND SLICING LESSON")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: BASIC INDEXING (Single Element Access)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: BASIC INDEXING")
print("=" * 80)

# Create our example tensor - a 3x4 matrix
tensor = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print("\nOriginal Tensor:")
print(tensor)
print(f"Shape: {tensor.shape}")

# Single element access: tensor[row, column]
# Remember: Python uses 0-based indexing (first element is index 0)
print("\n--- Single Element Access ---")
second_row_third_column = tensor[1, 2]  # Row 1 (second row), Column 2 (third column)
print(f"tensor[1, 2] = {second_row_third_column}")
print(f"Type: {type(second_row_third_column)}")
print("Explanation: Row index 1 is the SECOND row, column index 2 is the THIRD column")

# Alternative syntax using double brackets
# This is equivalent but less efficient (indexes twice instead of once)
print("\n--- Alternative Indexing Syntax ---")
second_row_third_column_alt = tensor[1][2]
print(f"tensor[1][2] = {second_row_third_column_alt}")
print("Note: tensor[1, 2] is preferred over tensor[1][2] for better performance")

# Accessing different elements
print("\n--- More Examples ---")
first_element = tensor[0, 0]  # Top-left corner
print(f"tensor[0, 0] (first element) = {first_element}")

last_column_second_row = tensor[1, 3]  # Second row, last column
print(f"tensor[1, 3] (second row, last column) = {last_column_second_row}")

# ============================================================================
# SECTION 2: NEGATIVE INDEXING (Accessing from the End)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: NEGATIVE INDEXING")
print("=" * 80)

print("\nNegative indices count backwards from the end:")
print("  -1 = last element/row/column")
print("  -2 = second to last element/row/column")
print("  -3 = third to last element/row/column, etc.")

# Access the last row using negative indexing
print("\n--- Accessing Last Row ---")
last_row = tensor[-1]
print(f"tensor[-1] (last row) = {last_row}")

# Access second to last row
second_to_last_row = tensor[-2]
print(f"tensor[-2] (second to last row) = {second_to_last_row}")

# Access last element in last row
print("\n--- Negative Indexing for Both Dimensions ---")
last_element = tensor[-1, -1]
print(f"tensor[-1, -1] (last element) = {last_element}")

# Access last element in first row
last_in_first_row = tensor[0, -1]
print(f"tensor[0, -1] (last element of first row) = {last_in_first_row}")

# Access first element in last row
first_in_last_row = tensor[-1, 0]
print(f"tensor[-1, 0] (first element of last row) = {first_in_last_row}")

# ============================================================================
# SECTION 3: ROW AND COLUMN EXTRACTION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: ROW AND COLUMN EXTRACTION")
print("=" * 80)

print("\n--- Extracting Entire Rows ---")
# Extract a single row (returns a 1D tensor)
first_row = tensor[0]
print(f"tensor[0] (first row):\n{first_row}")
print(f"Shape: {first_row.shape}")

second_row = tensor[1]
print(f"\ntensor[1] (second row):\n{second_row}")

print("\n--- Extracting Entire Columns ---")
# To extract a column, use : for all rows, then specify column index
first_column = tensor[:, 0]  # ":" means "all rows"
print(f"tensor[:, 0] (first column):\n{first_column}")
print(f"Shape: {first_column.shape}")

third_column = tensor[:, 2]
print(f"\ntensor[:, 2] (third column):\n{third_column}")

last_column = tensor[:, -1]
print(f"\ntensor[:, -1] (last column using negative index):\n{last_column}")

# ============================================================================
# SECTION 4: SLICING BASICS (start:stop syntax)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: SLICING BASICS")
print("=" * 80)

print("\nSlicing Syntax: tensor[start:stop]")
print("  - 'start' is inclusive (included in result)")
print("  - 'stop' is exclusive (NOT included in result)")
print("  - If 'start' is omitted, it defaults to 0")
print("  - If 'stop' is omitted, it goes to the end")

# Slicing from the article: "Get the third column and further of the second row"
print("\n--- Row Slicing Example (from TK's article) ---")
second_row_from_third_column = tensor[1][2:]  # Row 1, columns from index 2 to end
print(f"tensor[1][2:] (second row, third column onwards):")
print(second_row_from_third_column)
print("Explanation: [2:] means 'start at index 2 (third column) and go to the end'")

print("\n--- More Row Slicing Examples ---")
# Get first three elements of first row
first_row_subset = tensor[0, :3]  # Columns 0, 1, 2 (stops before 3)
print(f"tensor[0, :3] (first row, first three columns):")
print(first_row_subset)

# Get middle two elements of second row
second_row_middle = tensor[1, 1:3]  # Columns 1, 2 (stops before 3)
print(f"\ntensor[1, 1:3] (second row, middle two columns):")
print(second_row_middle)

print("\n--- Column Slicing Examples ---")
# Get first two columns (all rows)
first_two_cols = tensor[:, :2]
print(f"tensor[:, :2] (all rows, first two columns):")
print(first_two_cols)

# Get last two columns
last_two_cols = tensor[:, -2:]
print(f"\ntensor[:, -2:] (all rows, last two columns):")
print(last_two_cols)

# ============================================================================
# SECTION 5: COMBINED ROW-COLUMN SLICING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: COMBINED ROW-COLUMN SLICING")
print("=" * 80)

print("\nCombined Syntax: tensor[row_start:row_stop, col_start:col_stop]")
print("Both dimensions can be sliced simultaneously!")

# Example from the article: "Get the third column and further for the first and second rows"
print("\n--- Example from TK's Article ---")
result = tensor[:2, 2:]
print(f"tensor[:2, 2:]:")
print(result)
print("Explanation:")
print("  - :2 means 'rows from start to index 2' (rows 0 and 1, i.e., first and second rows)")
print("  - 2: means 'columns from index 2 to end' (columns 2 and 3, i.e., third and fourth columns)")

print("\n--- More Combined Slicing Examples ---")

# Get top-left 2x2 submatrix
top_left = tensor[:2, :2]
print(f"tensor[:2, :2] (top-left 2x2 submatrix):")
print(top_left)

# Get bottom-right 2x2 submatrix
bottom_right = tensor[-2:, -2:]
print(f"\ntensor[-2:, -2:] (bottom-right 2x2 submatrix):")
print(bottom_right)

# Get center 2x2 submatrix
center = tensor[0:2, 1:3]
print(f"\ntensor[0:2, 1:3] (center portion):")
print(center)

# Get every row, skip first and last column (middle columns only)
middle_cols = tensor[:, 1:-1]
print(f"\ntensor[:, 1:-1] (all rows, middle columns only):")
print(middle_cols)

# Get first and last row, all columns
first_and_last_rows = tensor[::2, :]  # We'll explain :: syntax in the next section
print(f"\ntensor[::2, :] (first and last rows using step):")
print(first_and_last_rows)

# ============================================================================
# SECTION 6: ADVANCED SLICING (Step, 3D Tensors)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: ADVANCED SLICING")
print("=" * 80)

print("\nAdvanced Slicing Syntax: tensor[start:stop:step]")
print("  - 'step' determines the increment between indices")
print("  - Default step is 1")
print("  - Negative step reverses the order")

print("\n--- Using Step Parameter ---")
# Get every other column
every_other_col = tensor[:, ::2]  # Start:stop omitted = all, step=2
print(f"tensor[:, ::2] (all rows, every other column):")
print(every_other_col)
print("Explanation: ::2 means 'from start to end with step of 2'")

# Reverse all columns
reversed_cols = tensor[:, ::-1]
print(f"\ntensor[:, ::-1] (all rows, columns reversed):")
print(reversed_cols)
print("Explanation: ::-1 means 'from start to end with step of -1 (reverse)'")

# Reverse all rows
reversed_rows = tensor[::-1, :]
print(f"\ntensor[::-1, :] (rows reversed, all columns):")
print(reversed_rows)

# Get every other row and every other column
checkered = tensor[::2, ::2]
print(f"\ntensor[::2, ::2] (every other row and column):")
print(checkered)

print("\n--- Working with 3D Tensors ---")
# Create a 3D tensor (e.g., representing batches of images or video frames)
tensor_3d = torch.tensor([
    [[1, 2, 3],
     [4, 5, 6]],

    [[7, 8, 9],
     [10, 11, 12]],

    [[13, 14, 15],
     [16, 17, 18]]
])

print(f"\n3D Tensor (shape {tensor_3d.shape}):")
print(tensor_3d)
print("Think of this as: [batch_size=3, height=2, width=3]")

# Access first batch
print(f"\ntensor_3d[0] (first batch):")
print(tensor_3d[0])

# Access specific element in 3D tensor
element = tensor_3d[1, 0, 2]  # Batch 1, row 0, column 2
print(f"\ntensor_3d[1, 0, 2] (second batch, first row, third column) = {element}")

# Slice across all dimensions
print(f"\ntensor_3d[:, :, 0] (all batches, all rows, first column only):")
print(tensor_3d[:, :, 0])

# Get first two batches, all rows, last two columns
subset_3d = tensor_3d[:2, :, -2:]
print(f"\ntensor_3d[:2, :, -2:] (first two batches, all rows, last two columns):")
print(subset_3d)

print("\n--- Boolean Indexing (Bonus) ---")
# You can also use boolean masks for advanced indexing
mask = tensor > 6  # Create boolean mask
print(f"\nBoolean mask (tensor > 6):")
print(mask)

filtered = tensor[mask]
print(f"\ntensor[tensor > 6] (all elements greater than 6):")
print(filtered)
print("Note: This returns a 1D tensor of elements that match the condition")

# ============================================================================
# SECTION 7: PRACTICE PROBLEMS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: PRACTICE PROBLEMS")
print("=" * 80)

print("\nSolve these problems using the original tensor:")
print(tensor)
print()

# Problem 1
print("PROBLEM 1: Extract the middle 2x2 submatrix (values 6, 7, 10, 11)")
print("Your code: tensor[?, ?]")
print("\nSolution:")
problem1_solution = tensor[1:3, 1:3]
print(f"tensor[1:3, 1:3] =")
print(problem1_solution)
print("Explanation: Rows 1-2 (indices 1:3), Columns 1-2 (indices 1:3)")

# Problem 2
print("\n" + "-" * 80)
print("PROBLEM 2: Get every other element from the last row")
print("Your code: tensor[?, ?]")
print("\nSolution:")
problem2_solution = tensor[-1, ::2]
print(f"tensor[-1, ::2] =")
print(problem2_solution)
print("Explanation: Last row (-1), every other column (::2)")

# Problem 3
print("\n" + "-" * 80)
print("PROBLEM 3: Extract the four corner elements as a 2x2 tensor")
print("Hint: You'll need to use both positive and negative indices")
print("Your code: You might need to use torch.stack() or tensor operations")
print("\nSolution:")
# One approach: manually create new tensor with corner values
corners = torch.tensor([
    [tensor[0, 0], tensor[0, -1]],
    [tensor[-1, 0], tensor[-1, -1]]
])
print(f"Corner elements:")
print(corners)
print("\nAlternative solution using indexing:")
# Another approach using advanced indexing
top_corners = tensor[0, [0, -1]]
bottom_corners = tensor[-1, [0, -1]]
corners_alt = torch.stack([top_corners, bottom_corners])
print(f"Using torch.stack:")
print(corners_alt)
print("Explanation: Stack the first and last elements of first and last rows")

# ============================================================================
# SUMMARY AND KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)

summary = """
1. BASIC INDEXING:
   - tensor[row, col] for single elements (preferred)
   - tensor[row][col] also works but is less efficient

2. NEGATIVE INDEXING:
   - Use negative indices to count from the end
   - -1 is the last element, -2 is second to last, etc.

3. SLICING SYNTAX:
   - start:stop:step
   - start is inclusive, stop is exclusive
   - Omit start/stop to go from beginning/to end
   - step determines increment (default is 1)

4. COMBINED SLICING:
   - tensor[row_slice, col_slice] for 2D tensors
   - Each dimension can be independently sliced

5. ADVANCED TECHNIQUES:
   - Use :: for all elements with custom step
   - ::-1 reverses the order
   - Boolean indexing for conditional selection
   - Works with higher dimensional tensors (3D, 4D, etc.)

6. COMMON PATTERNS:
   - tensor[:, i]     → column i
   - tensor[i, :]     → row i
   - tensor[:, ::-1]  → reverse columns
   - tensor[::2, ::2] → every other row and column
"""

print(summary)

print("\n" + "=" * 80)
print("LESSON COMPLETE!")
print("=" * 80)
print("\nPractice these operations with your own tensors to master indexing and slicing!")
