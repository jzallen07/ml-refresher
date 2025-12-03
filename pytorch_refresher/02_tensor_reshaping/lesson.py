"""
================================================================================
PYTORCH TENSOR RESHAPING OPERATIONS
================================================================================

Lesson: Understanding tensor shape manipulation in PyTorch
Source: Based on TK's article on PyTorch tensor reshaping
Author: PyTorch Refresher Course

Topics Covered:
- unsqueeze(): Adding dimensions to tensors
- squeeze(): Removing dimensions from tensors
- view(): Flexible tensor reshaping
- Dynamic sizing with -1 placeholder
- Real-world ML applications

Why Reshaping Matters:
In deep learning, tensor shapes must match exactly for operations to work.
Reshaping is essential for:
- Adding batch dimensions for model input
- Flattening feature maps before fully connected layers
- Preparing data for different layer types (Conv2D vs Linear)
- Broadcasting operations between tensors

================================================================================
"""

import torch
import sys

# Print PyTorch version for reference
print("=" * 80)
print("PYTORCH TENSOR RESHAPING LESSON")
print("=" * 80)
print(f"PyTorch Version: {torch.__version__}")
print("=" * 80)
print()


# ============================================================================
# SECTION 1: UNSQUEEZE() - ADDING DIMENSIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: UNSQUEEZE() - ADDING DIMENSIONS")
print("=" * 80)
print()

print("What is unsqueeze()?")
print("-" * 80)
print("unsqueeze() adds a new dimension of size 1 at the specified position.")
print("This is crucial when you need to add a batch dimension or match tensor shapes.")
print()

# Example 1.1: Basic unsqueeze - scalar to 1D tensor
print("Example 1.1: Adding a dimension to a scalar")
print("-" * 80)
tensor = torch.tensor(1)
print(f"Original tensor: {tensor}")
print(f"Original shape: {tensor.shape}")
print(f"Number of dimensions: {tensor.ndim}")
print()

# Add dimension at position 0 (becomes the first dimension)
batch = tensor.unsqueeze(0)
print(f"After unsqueeze(0): {batch}")
print(f"New shape: {batch.shape}")
print(f"Number of dimensions: {batch.ndim}")
print("→ We added a dimension of size 1 at position 0")
print()

# Example 1.2: Unsqueeze a 1D tensor
print("Example 1.2: Adding dimensions to a 1D tensor")
print("-" * 80)
tensor_1d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Original 1D tensor: {tensor_1d}")
print(f"Original shape: {tensor_1d.shape}")  # torch.Size([5])
print()

# Unsqueeze at dimension 0 (add batch dimension)
unsqueezed_0 = tensor_1d.unsqueeze(0)
print(f"After unsqueeze(0): {unsqueezed_0}")
print(f"Shape: {unsqueezed_0.shape}")  # torch.Size([1, 5])
print("→ Added dimension at the front - useful for batch dimension")
print()

# Unsqueeze at dimension 1 (add feature dimension)
unsqueezed_1 = tensor_1d.unsqueeze(1)
print(f"After unsqueeze(1): {unsqueezed_1}")
print(f"Shape: {unsqueezed_1.shape}")  # torch.Size([5, 1])
print("→ Added dimension at the end - converts to column vector")
print()

# Example 1.3: Unsqueeze with negative indexing
print("Example 1.3: Using negative indices with unsqueeze")
print("-" * 80)
print("Negative indices count from the end: -1 is the last position")
tensor_1d = torch.tensor([10, 20, 30])
print(f"Original tensor: {tensor_1d}, shape: {tensor_1d.shape}")
print()

unsqueezed_neg = tensor_1d.unsqueeze(-1)
print(f"After unsqueeze(-1): {unsqueezed_neg}")
print(f"Shape: {unsqueezed_neg.shape}")
print("→ unsqueeze(-1) adds dimension at the last position")
print()

# Example 1.4: Real ML use case - adding batch dimension
print("Example 1.4: ML Use Case - Adding batch dimension for model input")
print("-" * 80)
# Imagine we have a single image flattened to 784 pixels (28x28 MNIST image)
single_image = torch.randn(784)
print(f"Single image (flattened): shape = {single_image.shape}")
print("Problem: PyTorch models expect input with batch dimension [batch_size, features]")
print()

# Add batch dimension
batched_image = single_image.unsqueeze(0)
print(f"After adding batch dimension: shape = {batched_image.shape}")
print("→ Now compatible with model.forward() which expects [batch_size, 784]")
print()


# ============================================================================
# SECTION 2: SQUEEZE() - REMOVING DIMENSIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: SQUEEZE() - REMOVING DIMENSIONS")
print("=" * 80)
print()

print("What is squeeze()?")
print("-" * 80)
print("squeeze() removes dimensions of size 1 from a tensor.")
print("Useful for cleaning up extra dimensions or preparing output.")
print()

# Example 2.1: Basic squeeze from TK's article
print("Example 2.1: Squeezing a single dimension")
print("-" * 80)
tensor = torch.tensor([1.0])
print(f"Original tensor: {tensor}")
print(f"Original shape: {tensor.shape}")  # torch.Size([1])
print()

squeezed_tensor = tensor.squeeze(0)
print(f"After squeeze(0): {squeezed_tensor}")
print(f"Squeezed shape: {squeezed_tensor.shape}")  # torch.Size([])
print("→ Removed dimension of size 1 at position 0, now it's a scalar")
print()

# Example 2.2: Squeeze without specifying dimension
print("Example 2.2: Squeezing all dimensions of size 1")
print("-" * 80)
tensor_with_ones = torch.tensor([[[1.0, 2.0, 3.0]]])
print(f"Original tensor shape: {tensor_with_ones.shape}")  # torch.Size([1, 1, 3])
print(f"Original tensor: {tensor_with_ones}")
print()

squeezed_all = tensor_with_ones.squeeze()
print(f"After squeeze(): {squeezed_all}")
print(f"Squeezed shape: {squeezed_all.shape}")  # torch.Size([3])
print("→ squeeze() without argument removes ALL dimensions of size 1")
print()

# Example 2.3: Squeeze specific dimension
print("Example 2.3: Squeezing only a specific dimension")
print("-" * 80)
tensor_multi = torch.randn(1, 5, 1, 3)
print(f"Original shape: {tensor_multi.shape}")  # torch.Size([1, 5, 1, 3])
print()

squeezed_dim0 = tensor_multi.squeeze(0)
print(f"After squeeze(0): shape = {squeezed_dim0.shape}")  # torch.Size([5, 1, 3])
print("→ Removed dimension at position 0")
print()

squeezed_dim2 = tensor_multi.squeeze(2)
print(f"After squeeze(2): shape = {squeezed_dim2.shape}")  # torch.Size([1, 5, 3])
print("→ Removed dimension at position 2")
print()

# Example 2.4: ML use case - removing batch dimension after inference
print("Example 2.4: ML Use Case - Removing batch dimension from single prediction")
print("-" * 80)
# Model output for a single sample (but still has batch dimension)
model_output = torch.tensor([[0.1, 0.7, 0.2]])
print(f"Model output (with batch dim): {model_output}")
print(f"Shape: {model_output.shape}")  # torch.Size([1, 3])
print()

prediction = model_output.squeeze(0)
print(f"After squeezing batch dimension: {prediction}")
print(f"Shape: {prediction.shape}")  # torch.Size([3])
print("→ Clean output without extra batch dimension")
print()


# ============================================================================
# SECTION 3: VIEW() - FLEXIBLE RESHAPING WITH EXPLICIT DIMENSIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: VIEW() - FLEXIBLE RESHAPING")
print("=" * 80)
print()

print("What is view()?")
print("-" * 80)
print("view() reshapes a tensor to a new shape without copying data.")
print("The total number of elements must remain the same.")
print("Key point: The tensor must be contiguous in memory for view() to work.")
print()

# Example 3.1: Basic view from TK's article
print("Example 3.1: Reshaping 1D to 2D tensor")
print("-" * 80)
tensor = torch.tensor([0, 1, 2, 3, 4])
print(f"Original tensor: {tensor}")
print(f"Original shape: {tensor.shape}")  # torch.Size([5])
print(f"Total elements: {tensor.numel()}")
print()

reshaped_tensor = tensor.view(5, 1)
print(f"After view(5, 1):")
print(reshaped_tensor)
print(f"New shape: {reshaped_tensor.shape}")  # torch.Size([5, 1])
print(f"Total elements: {reshaped_tensor.numel()}")
print("→ Reshaped from [5] to [5, 1] - same data, different shape")
print()

# Example 3.2: Multiple reshaping possibilities
print("Example 3.2: Different ways to reshape the same tensor")
print("-" * 80)
tensor = torch.arange(12)  # Creates tensor [0, 1, 2, ..., 11]
print(f"Original tensor: {tensor}")
print(f"Original shape: {tensor.shape}")  # torch.Size([12])
print()

# Reshape to 3x4 matrix
reshaped_3x4 = tensor.view(3, 4)
print("Reshaped to 3x4:")
print(reshaped_3x4)
print(f"Shape: {reshaped_3x4.shape}")
print()

# Reshape to 4x3 matrix
reshaped_4x3 = tensor.view(4, 3)
print("Reshaped to 4x3:")
print(reshaped_4x3)
print(f"Shape: {reshaped_4x3.shape}")
print()

# Reshape to 2x2x3 tensor (3D)
reshaped_3d = tensor.view(2, 2, 3)
print("Reshaped to 2x2x3:")
print(reshaped_3d)
print(f"Shape: {reshaped_3d.shape}")
print()

# Example 3.3: View constraint - total elements must match
print("Example 3.3: Understanding view() constraints")
print("-" * 80)
tensor = torch.arange(10)
print(f"Tensor with {tensor.numel()} elements: {tensor}")
print()

try:
    # This will fail because 10 elements can't be reshaped to 3x4 (12 elements)
    wrong_reshape = tensor.view(3, 4)
except RuntimeError as e:
    print(f"ERROR: {e}")
    print("→ Can't reshape 10 elements into 3x4 shape (which needs 12 elements)")
print()

# This works because 10 = 2 * 5
correct_reshape = tensor.view(2, 5)
print("Correct reshape to 2x5:")
print(correct_reshape)
print(f"Shape: {correct_reshape.shape}")
print()


# ============================================================================
# SECTION 4: USING -1 PLACEHOLDER FOR DYNAMIC SIZING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: USING -1 PLACEHOLDER FOR DYNAMIC SIZING")
print("=" * 80)
print()

print("What is the -1 placeholder?")
print("-" * 80)
print("-1 in view() means 'infer this dimension automatically'.")
print("PyTorch calculates the size based on the total number of elements")
print("and the other specified dimensions.")
print("You can only use -1 for ONE dimension at a time.")
print()

# Example 4.1: Basic -1 usage from TK's article
print("Example 4.1: Using -1 to infer dimension size")
print("-" * 80)
tensor = torch.tensor([0, 1, 2, 3, 4])
print(f"Original tensor: {tensor}")
print(f"Original shape: {tensor.shape}")  # torch.Size([5])
print()

reshaped_tensor = tensor.view(-1, 1)
print(f"After view(-1, 1):")
print(reshaped_tensor)
print(f"New shape: {reshaped_tensor.shape}")  # torch.Size([5, 1])
print("→ PyTorch inferred the first dimension should be 5")
print("   Calculation: total_elements / 1 = 5 / 1 = 5")
print()

# Example 4.2: -1 in different positions
print("Example 4.2: Using -1 at different positions")
print("-" * 80)
tensor = torch.arange(24)
print(f"Tensor with {tensor.numel()} elements")
print()

# Let PyTorch infer the first dimension
reshaped_1 = tensor.view(-1, 6)
print(f"view(-1, 6) → shape: {reshaped_1.shape}")
print(f"PyTorch calculated: 24 / 6 = 4 for first dimension")
print()

# Let PyTorch infer the last dimension
reshaped_2 = tensor.view(4, -1)
print(f"view(4, -1) → shape: {reshaped_2.shape}")
print(f"PyTorch calculated: 24 / 4 = 6 for last dimension")
print()

# Works with multiple dimensions
reshaped_3 = tensor.view(2, 3, -1)
print(f"view(2, 3, -1) → shape: {reshaped_3.shape}")
print(f"PyTorch calculated: 24 / (2 * 3) = 4 for last dimension")
print()

# Example 4.3: Using -1 to flatten
print("Example 4.3: Flattening a tensor with -1")
print("-" * 80)
tensor_3d = torch.randn(2, 3, 4)
print(f"Original 3D tensor shape: {tensor_3d.shape}")
print(f"Total elements: {tensor_3d.numel()}")
print()

flattened = tensor_3d.view(-1)
print(f"After view(-1): {flattened.shape}")
print("→ Flattened to 1D tensor with all elements")
print()

# Alternative: flatten to keep batch dimension
batch_flattened = tensor_3d.view(2, -1)
print(f"After view(2, -1): {batch_flattened.shape}")
print("→ Kept batch dimension (2), flattened the rest (3*4=12)")
print()

# Example 4.4: Why -1 is useful - works with variable sizes
print("Example 4.4: Dynamic reshaping with -1 (works for any batch size)")
print("-" * 80)
print("This is crucial when batch size varies during training/inference:")
print()

for batch_size in [1, 4, 8]:
    # Simulate different batch sizes of 28x28 images
    images = torch.randn(batch_size, 28, 28)
    print(f"Batch of {batch_size} images, shape: {images.shape}")

    # Flatten while preserving batch dimension
    flattened_images = images.view(batch_size, -1)
    print(f"  After view({batch_size}, -1): {flattened_images.shape}")

    # Even better: use -1 for batch dimension too!
    auto_flattened = images.view(-1, 28*28)
    print(f"  After view(-1, 784): {auto_flattened.shape}")
    print()


# ============================================================================
# SECTION 5: REAL-WORLD ML EXAMPLES
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: REAL-WORLD ML EXAMPLES")
print("=" * 80)
print()

# Example 5.1: Preparing images for CNN
print("Example 5.1: Adding Batch and Channel Dimensions for CNN")
print("-" * 80)
print("Scenario: Single grayscale image needs to be fed to a CNN")
print("CNN expects: [batch_size, channels, height, width]")
print()

# Single grayscale image (28x28 pixels)
single_image = torch.randn(28, 28)
print(f"Original image shape: {single_image.shape}")
print("Missing: batch dimension and channel dimension")
print()

# Add channel dimension first
image_with_channel = single_image.unsqueeze(0)
print(f"After unsqueeze(0) [add channel]: {image_with_channel.shape}")
print()

# Add batch dimension
image_ready = image_with_channel.unsqueeze(0)
print(f"After unsqueeze(0) [add batch]: {image_ready.shape}")
print("→ Now ready for CNN: [1, 1, 28, 28]")
print()

# Example 5.2: Flattening CNN output for fully connected layer
print("Example 5.2: Flattening Feature Maps for Fully Connected Layer")
print("-" * 80)
print("Scenario: CNN outputs feature maps, need to flatten for FC layer")
print()

# Simulated output from CNN layer: [batch=32, channels=64, height=7, width=7]
cnn_output = torch.randn(32, 64, 7, 7)
print(f"CNN output shape: {cnn_output.shape}")
print(f"Total features per sample: {64 * 7 * 7} = 3136")
print()

# Flatten while keeping batch dimension
flattened = cnn_output.view(32, -1)
print(f"After view(32, -1): {flattened.shape}")
print("→ Ready for fully connected layer: [32, 3136]")
print()

# Better: use -1 for batch size (works with any batch size)
flattened_dynamic = cnn_output.view(-1, 64*7*7)
print(f"After view(-1, 3136): {flattened_dynamic.shape}")
print("→ Works for any batch size!")
print()

# Example 5.3: Batch matrix operations
print("Example 5.3: Reshaping for Batch Matrix Multiplication")
print("-" * 80)
print("Scenario: Computing attention scores across batch")
print()

# Query, Key, Value in transformer (simplified)
batch_size = 4
seq_length = 10
hidden_dim = 64

queries = torch.randn(batch_size, seq_length, hidden_dim)
print(f"Queries shape: {queries.shape}")
print()

# Reshape for multi-head attention (num_heads=8)
num_heads = 8
head_dim = hidden_dim // num_heads

queries_reshaped = queries.view(batch_size, seq_length, num_heads, head_dim)
print(f"After view({batch_size}, {seq_length}, {num_heads}, {head_dim}):")
print(f"Shape: {queries_reshaped.shape}")
print("→ Split hidden_dim into (num_heads, head_dim)")
print()

# Transpose for batch matrix multiplication
queries_transposed = queries_reshaped.transpose(1, 2)
print(f"After transpose(1, 2): {queries_transposed.shape}")
print("→ Now shape is [batch, num_heads, seq_length, head_dim]")
print("   Ready for multi-head attention computation")
print()

# Example 5.4: Preparing time series data
print("Example 5.4: Reshaping Time Series Data for RNN")
print("-" * 80)
print("Scenario: Have flat time series, need [batch, sequence, features]")
print()

# Flat time series data: 100 time steps with 5 features each
flat_data = torch.randn(500)  # 100 * 5 = 500 values
print(f"Flat data shape: {flat_data.shape}")
print()

# Reshape to [sequence_length, features]
time_series = flat_data.view(100, 5)
print(f"After view(100, 5): {time_series.shape}")
print("→ [100 time steps, 5 features]")
print()

# Add batch dimension for RNN
batched_series = time_series.unsqueeze(0)
print(f"After unsqueeze(0): {batched_series.shape}")
print("→ [1 batch, 100 time steps, 5 features]")
print("   Ready for RNN/LSTM input")
print()


# ============================================================================
# SECTION 6: PRACTICE PROBLEMS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: PRACTICE PROBLEMS")
print("=" * 80)
print()

print("Test your understanding with these practice problems!")
print("Try to solve them before looking at the solutions below.")
print()

# Problem 1
print("PROBLEM 1: Image Batch Preparation")
print("-" * 80)
print("You have 16 RGB images of size 224x224 stored as a flat tensor.")
print("The tensor has shape [802816] (16 * 3 * 224 * 224 = 802816)")
print("Task: Reshape it to proper format [batch, channels, height, width]")
print()

# CREATE YOUR SOLUTION HERE:
# flat_images = torch.randn(802816)
# solution_1 = ???

# Provided solution:
flat_images = torch.randn(802816)
solution_1 = flat_images.view(16, 3, 224, 224)
print(f"Solution shape: {solution_1.shape}")
print("Expected: torch.Size([16, 3, 224, 224])")
print(f"Correct: {solution_1.shape == torch.Size([16, 3, 224, 224])}")
print()

# Problem 2
print("PROBLEM 2: Remove Unnecessary Dimensions")
print("-" * 80)
print("A model outputs predictions with shape [1, 1, 10] (1 batch, 1 sequence, 10 classes)")
print("Task: Remove all dimensions of size 1 to get just the class scores")
print()

# CREATE YOUR SOLUTION HERE:
# model_out = torch.randn(1, 1, 10)
# solution_2 = ???

# Provided solution:
model_out = torch.randn(1, 1, 10)
solution_2 = model_out.squeeze()
print(f"Original shape: {model_out.shape}")
print(f"Solution shape: {solution_2.shape}")
print("Expected: torch.Size([10])")
print(f"Correct: {solution_2.shape == torch.Size([10])}")
print()

# Problem 3
print("PROBLEM 3: Dynamic Batch Flattening")
print("-" * 80)
print("You have feature maps from a CNN with shape [B, 256, 8, 8] where B varies.")
print("Task: Flatten spatial dimensions while keeping batch and channel info.")
print("Target shape: [B, 256, 64] where 64 = 8*8")
print()

# CREATE YOUR SOLUTION HERE:
# feature_maps = torch.randn(B, 256, 8, 8)  # B can be any value
# solution_3 = ???

# Provided solution (testing with different batch sizes):
for B in [1, 4, 16, 32]:
    feature_maps = torch.randn(B, 256, 8, 8)
    solution_3 = feature_maps.view(B, 256, -1)
    # Alternative: feature_maps.view(-1, 256, 64)
    print(f"Batch size {B}: {feature_maps.shape} → {solution_3.shape}")
    assert solution_3.shape == torch.Size([B, 256, 64]), f"Wrong shape for batch size {B}"

print("Expected: Shape should be [B, 256, 64] for any batch size B")
print("All batch sizes correct! ✓")
print()


# ============================================================================
# SUMMARY AND KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: KEY TAKEAWAYS")
print("=" * 80)
print()

print("📚 UNSQUEEZE() - Adding Dimensions")
print("   • Adds dimension of size 1 at specified position")
print("   • Use case: Adding batch/channel dimensions")
print("   • Syntax: tensor.unsqueeze(dim)")
print()

print("📚 SQUEEZE() - Removing Dimensions")
print("   • Removes dimensions of size 1")
print("   • Use case: Cleaning up extra dimensions")
print("   • Syntax: tensor.squeeze(dim) or tensor.squeeze() for all")
print()

print("📚 VIEW() - Flexible Reshaping")
print("   • Reshapes tensor to new shape without copying data")
print("   • Total elements must stay the same")
print("   • Tensor must be contiguous in memory")
print("   • Syntax: tensor.view(shape)")
print()

print("📚 -1 PLACEHOLDER - Dynamic Sizing")
print("   • PyTorch infers the dimension size automatically")
print("   • Can only use -1 for ONE dimension")
print("   • Essential for dynamic batch sizes")
print("   • Syntax: tensor.view(-1, other_dims) or tensor.view(dims, -1)")
print()

print("🎯 BEST PRACTICES")
print("   1. Use unsqueeze(0) to add batch dimension for model input")
print("   2. Use view(-1, ...) for dynamic batch-size handling")
print("   3. Use squeeze() to clean up model outputs")
print("   4. Always verify shapes with .shape after reshaping")
print("   5. Use view(-1) to flatten tensors completely")
print()

print("=" * 80)
print("LESSON COMPLETE! Practice these operations to master tensor manipulation.")
print("=" * 80)
