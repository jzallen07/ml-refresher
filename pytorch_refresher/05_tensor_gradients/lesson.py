"""
================================================================================
LESSON 5: PyTorch Tensor Derivatives and Gradients
================================================================================

Source: Based on TK's article on PyTorch tensor gradients
https://www.freecodecamp.org/news/pytorch-tutorial-for-deep-learning/

This lesson covers:
- What gradients are and why they're essential for deep learning
- Creating tensors that track gradients (requires_grad=True)
- Computing gradients using backward()
- Understanding the computational graph
- Multi-variable gradient computation
- Gradient accumulation and zeroing
- Using torch.no_grad() for inference

Author: Machine Learning Refresher Course
Date: 2025-12-02
================================================================================
"""

import torch
import math

print("=" * 80)
print("PYTORCH TENSOR GRADIENTS AND DERIVATIVES")
print("=" * 80)
print()


# ============================================================================
# SECTION 1: INTRODUCTION TO GRADIENTS
# ============================================================================
print("=" * 80)
print("SECTION 1: WHY GRADIENTS MATTER IN MACHINE LEARNING")
print("=" * 80)
print()

print("Gradients are the foundation of how neural networks learn!")
print()
print("Key Concepts:")
print("  • A gradient is the derivative of a function with respect to its inputs")
print("  • In ML, we use gradients to optimize our model's parameters")
print("  • Gradients tell us HOW to adjust parameters to minimize loss")
print()
print("The Learning Process:")
print("  1. Forward pass: Make predictions using current parameters")
print("  2. Compute loss: Measure how wrong the predictions are")
print("  3. Backward pass: Compute gradients (how loss changes w.r.t. parameters)")
print("  4. Update parameters: Move in the direction that reduces loss")
print("  5. Repeat until the model converges")
print()
print("PyTorch's autograd system automatically computes these gradients!")
print("This is called 'automatic differentiation' or 'autodiff'")
print()


# ============================================================================
# SECTION 2: CREATING TENSORS WITH requires_grad=True
# ============================================================================
print("=" * 80)
print("SECTION 2: ENABLING GRADIENT TRACKING")
print("=" * 80)
print()

print("To compute gradients, we need to tell PyTorch which tensors to track.")
print("We do this by setting requires_grad=True when creating a tensor.")
print()

# Create a tensor WITHOUT gradient tracking (default behavior)
a = torch.tensor(3.0)
print(f"Tensor without gradient tracking: {a}")
print(f"  requires_grad: {a.requires_grad}")
print()

# Create a tensor WITH gradient tracking
b = torch.tensor(3.0, requires_grad=True)
print(f"Tensor with gradient tracking: {b}")
print(f"  requires_grad: {b.requires_grad}")
print()

print("Why do we need requires_grad=True?")
print("  • It tells PyTorch to build a computational graph")
print("  • The graph tracks all operations performed on this tensor")
print("  • This allows automatic gradient computation via backpropagation")
print()

# You can also enable gradient tracking on existing tensors
c = torch.tensor(5.0)
c.requires_grad_(True)  # Note: underscore means in-place operation
print(f"Enabled gradient tracking on existing tensor: {c}")
print(f"  requires_grad: {c.requires_grad}")
print()


# ============================================================================
# SECTION 3: COMPUTING GRADIENTS WITH backward() - TK'S CORE EXAMPLE
# ============================================================================
print("=" * 80)
print("SECTION 3: COMPUTING GRADIENTS - THE CORE EXAMPLE")
print("=" * 80)
print()

print("Let's work through TK's example step by step:")
print()

# TK's exact example from the article
print("Step 1: Create a tensor with requires_grad=True")
x = torch.tensor(2.0, requires_grad=True)
print(f"x = {x}")
print()

print("Step 2: Build a relationship - let's compute y = x²")
y = x ** 2
print(f"y = x² = {y}")
print(f"  Notice the grad_fn: {y.grad_fn}")
print("  This shows that y was created by a power operation (PowBackward0)")
print()

print("Mathematical Background:")
print("  • We have: y = x²")
print("  • The derivative is: dy/dx = 2x")
print("  • At x = 2.0, the derivative is: 2 × 2.0 = 4.0")
print()

print("Step 3: Compute the gradient using backward()")
y.backward()
print("Called y.backward() - this computes dy/dx")
print()

print("Step 4: Access the computed gradient")
print(f"x.grad = {x.grad}")
print()

print("What just happened?")
print("  • PyTorch computed dy/dx = 2x")
print("  • It evaluated this at x = 2.0")
print("  • Result: 2 × 2.0 = 4.0")
print("  • The gradient is stored in x.grad")
print()

print("This gradient tells us: if we increase x by a tiny amount,")
print("y will increase by approximately 4 times that amount.")
print()


# ============================================================================
# SECTION 4: UNDERSTANDING THE COMPUTATIONAL GRAPH
# ============================================================================
print("=" * 80)
print("SECTION 4: THE COMPUTATIONAL GRAPH AND grad_fn")
print("=" * 80)
print()

print("PyTorch builds a 'computational graph' to track operations.")
print("Each operation creates a node with a grad_fn (gradient function).")
print()

# Create a more complex computation graph
x1 = torch.tensor(3.0, requires_grad=True)
x2 = torch.tensor(4.0, requires_grad=True)

print(f"Input tensors:")
print(f"  x1 = {x1}")
print(f"  x2 = {x2}")
print()

# Build a computation: z = x1² + x2³
y1 = x1 ** 2
y2 = x2 ** 3
z = y1 + y2

print(f"Computations:")
print(f"  y1 = x1² = {y1}")
print(f"    grad_fn: {y1.grad_fn}")
print(f"  y2 = x2³ = {y2}")
print(f"    grad_fn: {y2.grad_fn}")
print(f"  z = y1 + y2 = {z}")
print(f"    grad_fn: {z.grad_fn}")
print()

print("The computational graph looks like this:")
print()
print("     x1 (3.0)          x2 (4.0)")
print("        |                 |")
print("    [** 2]            [** 3]")
print("        |                 |")
print("     y1 (9.0)          y2 (64.0)")
print("        |                 |")
print("        +─────────────────+")
print("                 |")
print("              z (73.0)")
print()

# Compute gradients
z.backward()

print(f"Computed gradients:")
print(f"  dz/dx1 = {x1.grad}  (derivative of x² at x=3 is 2×3 = 6)")
print(f"  dz/dx2 = {x2.grad}  (derivative of x³ at x=4 is 3×16 = 48)")
print()

print("Mathematical verification:")
print(f"  z = x1² + x2³")
print(f"  dz/dx1 = 2×x1 = 2×3 = 6.0 ✓")
print(f"  dz/dx2 = 3×x2² = 3×16 = 48.0 ✓")
print()


# ============================================================================
# SECTION 5: MULTI-VARIABLE GRADIENTS
# ============================================================================
print("=" * 80)
print("SECTION 5: GRADIENTS WITH MULTIPLE VARIABLES")
print("=" * 80)
print()

print("Real neural networks have millions of parameters.")
print("Let's see how PyTorch computes gradients for multiple variables.")
print()

# Example: Linear function z = 3a + 2b
a = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(7.0, requires_grad=True)

print(f"Variables:")
print(f"  a = {a.item()}")
print(f"  b = {b.item()}")
print()

z = 3 * a + 2 * b
print(f"Function: z = 3a + 2b")
print(f"  z = 3×{a.item()} + 2×{b.item()} = {z.item()}")
print()

z.backward()

print(f"Gradients:")
print(f"  dz/da = {a.grad.item()} (expected: 3, because coefficient of a is 3)")
print(f"  dz/db = {b.grad.item()} (expected: 2, because coefficient of b is 2)")
print()

print("Interpretation:")
print("  • If we increase 'a' by 1, z increases by 3")
print("  • If we increase 'b' by 1, z increases by 2")
print("  • This tells us 'a' has more impact on z than 'b'")
print()


# ============================================================================
# SECTION 6: GRADIENT ACCUMULATION AND ZEROING
# ============================================================================
print("=" * 80)
print("SECTION 6: GRADIENT ACCUMULATION (AND WHY WE ZERO GRADIENTS)")
print("=" * 80)
print()

print("⚠️  IMPORTANT: PyTorch ACCUMULATES gradients by default!")
print("This means calling backward() multiple times ADDS to existing gradients.")
print()

# Demonstrate gradient accumulation
x = torch.tensor(2.0, requires_grad=True)

print("First computation:")
y1 = x ** 2
y1.backward()
print(f"  y1 = x² = {y1.item()}")
print(f"  After first backward(): x.grad = {x.grad.item()}")
print()

print("Second computation (WITHOUT zeroing gradients):")
y2 = x ** 3
y2.backward()
print(f"  y2 = x³ = {y2.item()}")
print(f"  After second backward(): x.grad = {x.grad.item()}")
print()

print("Notice that x.grad = 4 + 12 = 16!")
print("  • First backward: dy1/dx = 2x = 4")
print("  • Second backward: dy2/dx = 3x² = 12")
print("  • Total accumulated: 4 + 12 = 16")
print()

print("This is usually NOT what we want in training!")
print("Solution: Zero the gradients before each backward pass.")
print()

# Demonstrate proper gradient zeroing
x = torch.tensor(2.0, requires_grad=True)

print("Proper approach:")
y1 = x ** 2
y1.backward()
print(f"  After first backward(): x.grad = {x.grad.item()}")
print()

print("  Zero the gradient:")
x.grad.zero_()  # Zero the gradient before next computation
print(f"  After zero_(): x.grad = {x.grad.item()}")
print()

y2 = x ** 3
y2.backward()
print(f"  After second backward(): x.grad = {x.grad.item()}")
print("  ✓ Now we have the correct gradient for y2!")
print()

print("In PyTorch training loops, you'll often see:")
print("  optimizer.zero_grad()  # Zero gradients")
print("  loss.backward()        # Compute gradients")
print("  optimizer.step()       # Update parameters")
print()


# ============================================================================
# SECTION 7: torch.no_grad() CONTEXT
# ============================================================================
print("=" * 80)
print("SECTION 7: DISABLING GRADIENT COMPUTATION WITH torch.no_grad()")
print("=" * 80)
print()

print("Sometimes we don't need gradients (e.g., during inference/testing).")
print("Using torch.no_grad() saves memory and speeds up computation.")
print()

x = torch.tensor(3.0, requires_grad=True)

print("WITH gradient tracking:")
y1 = x ** 2
print(f"  y1 = {y1}")
print(f"  y1.requires_grad = {y1.requires_grad}")
print(f"  y1.grad_fn = {y1.grad_fn}")
print()

print("WITHOUT gradient tracking (using torch.no_grad()):")
with torch.no_grad():
    y2 = x ** 2
    print(f"  y2 = {y2}")
    print(f"  y2.requires_grad = {y2.requires_grad}")
    print(f"  y2.grad_fn = {y2.grad_fn}")
print()

print("When to use torch.no_grad():")
print("  ✓ During model evaluation/testing")
print("  ✓ When making predictions on new data")
print("  ✓ When computing metrics that don't need gradients")
print("  ✗ During training (we need gradients to learn!)")
print()

# Demonstrate memory savings
x = torch.randn(1000, 1000, requires_grad=True)

print("Memory comparison (1000×1000 tensor):")
y_with_grad = x ** 2 + x ** 3
print(f"  With gradient tracking: grad_fn exists: {y_with_grad.grad_fn is not None}")

with torch.no_grad():
    y_without_grad = x ** 2 + x ** 3
    print(f"  Without gradient tracking: grad_fn exists: {y_without_grad.grad_fn is not None}")
print()

print("The version without gradients uses less memory and computes faster!")
print()


# ============================================================================
# SECTION 8: PRACTICE PROBLEMS
# ============================================================================
print("=" * 80)
print("SECTION 8: PRACTICE PROBLEMS")
print("=" * 80)
print()

print("Now it's your turn! Try solving these problems.")
print("Uncomment the code and fill in the blanks.")
print()

print("-" * 80)
print("PROBLEM 1: Basic Gradient Computation")
print("-" * 80)
print()
print("Given: f(x) = 3x² + 2x + 1")
print("Find: df/dx at x = 4.0")
print()
print("Expected: df/dx = 6x + 2 = 6(4) + 2 = 26")
print()

# SOLUTION:
x = torch.tensor(4.0, requires_grad=True)
f = 3 * x**2 + 2 * x + 1
f.backward()
print(f"Solution: df/dx = {x.grad.item()}")
print()

# Uncomment to practice:
# x = torch.tensor(4.0, requires_grad=True)
# f = # TODO: Implement f(x) = 3x² + 2x + 1
# # TODO: Compute the gradient
# print(f"Your answer: df/dx = {x.grad.item()}")
# print()


print("-" * 80)
print("PROBLEM 2: Multi-Variable Gradients")
print("-" * 80)
print()
print("Given: g(x, y) = x²y + y³")
print("Find: dg/dx and dg/dy at x = 2.0, y = 3.0")
print()
print("Expected:")
print("  dg/dx = 2xy = 2(2)(3) = 12")
print("  dg/dy = x² + 3y² = 4 + 27 = 31")
print()

# SOLUTION:
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
g = x**2 * y + y**3
g.backward()
print(f"Solution: dg/dx = {x.grad.item()}, dg/dy = {y.grad.item()}")
print()

# Uncomment to practice:
# x = torch.tensor(2.0, requires_grad=True)
# y = torch.tensor(3.0, requires_grad=True)
# g = # TODO: Implement g(x, y) = x²y + y³
# # TODO: Compute gradients
# print(f"Your answer: dg/dx = {x.grad.item()}, dg/dy = {y.grad.item()}")
# print()


print("-" * 80)
print("PROBLEM 3: Chain Rule in Action")
print("-" * 80)
print()
print("Given: h(x) = (2x + 3)²")
print("Find: dh/dx at x = 1.0")
print()
print("Expected:")
print("  Let u = 2x + 3, then h = u²")
print("  dh/dx = dh/du × du/dx = 2u × 2 = 4u = 4(2x + 3)")
print("  At x = 1: dh/dx = 4(2(1) + 3) = 4(5) = 20")
print()

# SOLUTION:
x = torch.tensor(1.0, requires_grad=True)
h = (2 * x + 3) ** 2
h.backward()
print(f"Solution: dh/dx = {x.grad.item()}")
print()

# Uncomment to practice:
# x = torch.tensor(1.0, requires_grad=True)
# h = # TODO: Implement h(x) = (2x + 3)²
# # TODO: Compute the gradient
# print(f"Your answer: dh/dx = {x.grad.item()}")
# print()


# ============================================================================
# SUMMARY AND KEY TAKEAWAYS
# ============================================================================
print("=" * 80)
print("LESSON SUMMARY: KEY TAKEAWAYS")
print("=" * 80)
print()

print("🎯 Key Concepts Covered:")
print()
print("1. GRADIENTS are the foundation of neural network learning")
print("   • They tell us how to adjust parameters to reduce loss")
print()
print("2. requires_grad=True enables gradient tracking")
print("   • PyTorch builds a computational graph of operations")
print()
print("3. .backward() computes gradients automatically")
print("   • Uses automatic differentiation (autograd)")
print("   • Gradients stored in .grad attribute")
print()
print("4. GRADIENT ACCUMULATION: PyTorch adds gradients by default")
print("   • Always zero gradients in training loops")
print("   • Use .zero_() or optimizer.zero_grad()")
print()
print("5. torch.no_grad() for inference")
print("   • Disables gradient tracking")
print("   • Saves memory and speeds up computation")
print()
print("6. The COMPUTATIONAL GRAPH tracks all operations")
print("   • Each operation has a grad_fn")
print("   • Enables backpropagation through complex models")
print()

print("=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print()
print("• Experiment with different mathematical functions")
print("• Try computing gradients of complex expressions")
print("• Practice interpreting what gradients mean in context")
print("• Move on to using gradients in actual neural network training!")
print()

print("=" * 80)
print("END OF LESSON 5: TENSOR GRADIENTS AND DERIVATIVES")
print("=" * 80)
