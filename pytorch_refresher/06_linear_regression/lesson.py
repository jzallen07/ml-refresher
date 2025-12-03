"""
================================================================================
LESSON 6: LINEAR REGRESSION IN PYTORCH
================================================================================

Source: Based on TK's article on Linear Regression with PyTorch
Author: Educational lesson based on TK's tutorial
Topic: Building and training linear regression models using PyTorch

Linear regression is one of the fundamental machine learning algorithms.
It models the relationship between input features and output targets using
a linear equation: y = xw + b

Where:
- x = input feature(s)
- w = weight(s) - the slope of the line
- b = bias - the y-intercept
- y = predicted output

In this lesson, we'll:
1. Understand the linear model mathematically
2. Use PyTorch's built-in nn.Linear module
3. Build a custom linear regression model class
4. Train the model on distance-time data
5. Understand the training loop: predict → loss → adjust → repeat
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 80)
print("PYTORCH LINEAR REGRESSION LESSON")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: INTRODUCTION - THE LINEAR MODEL
# ============================================================================
print("SECTION 1: Understanding the Linear Model")
print("-" * 80)

"""
The linear equation: y = xw + b

This is the foundation of linear regression. Let's break it down:
- 'x' is our input (independent variable)
- 'w' is the weight/slope (what we want to learn)
- 'b' is the bias/intercept (also what we want to learn)
- 'y' is our output (dependent variable)

Example: If we're predicting time based on distance:
- x = distance traveled
- y = time taken
- w = how much time increases per unit distance
- b = base time (starting point)
"""

print("Linear equation: y = xw + b")
print("We need to LEARN the values of 'w' (weight) and 'b' (bias)")
print("PyTorch will help us find these through gradient descent!\n")

# ============================================================================
# SECTION 2: USING nn.Linear DIRECTLY
# ============================================================================
print("\nSECTION 2: Using PyTorch's nn.Linear Module")
print("-" * 80)

"""
PyTorch provides nn.Linear, a ready-to-use linear transformation module.
It implements y = xw + b automatically!

Parameters:
- in_features: number of input features (dimensions of x)
- out_features: number of output features (dimensions of y)
- bias: whether to include the bias term b (default=True)
"""

# Create a simple linear model: 1 input feature → 1 output feature
model_simple = nn.Linear(in_features=1, out_features=1, bias=True)

print("Created nn.Linear model with:")
print(f"  - in_features=1 (one input)")
print(f"  - out_features=1 (one output)")
print(f"  - bias=True (includes bias term)\n")

# Check the initial random weights and bias
print("Initial model parameters (randomly initialized):")
print(f"  Weight: {model_simple.weight.item():.4f}")
print(f"  Bias: {model_simple.bias.item():.4f}\n")

# Test the model with some inputs
test_input = torch.tensor([[2.], [4.]])
print(f"Testing model with input: {test_input.T}")
predictions = model_simple(test_input)
print(f"Model predictions: {predictions.T}")
print(f"  (These are random because the model hasn't been trained yet!)\n")

"""
From TK's article example:
If weight = -0.4443 and bias = -0.5045
Then for inputs [2., 4.]:
  - For x=2: y = 2*(-0.4443) + (-0.5045) = -1.3930
  - For x=4: y = 4*(-0.4443) + (-0.5045) = -2.2815
Result: tensor([[-1.3930], [-2.2815]])
"""

# ============================================================================
# SECTION 3: BUILDING A CUSTOM MODEL WITH nn.Module
# ============================================================================
print("\nSECTION 3: Building a Custom Linear Regression Model")
print("-" * 80)

"""
While nn.Linear works great on its own, we often want to build custom models
using nn.Module. This is THE standard way to define models in PyTorch.

All PyTorch models should:
1. Inherit from nn.Module
2. Define layers in __init__
3. Implement the forward() method (defines the forward pass)

This pattern scales from simple linear regression to complex neural networks!
"""

class LinearRegression(nn.Module):
    """
    Custom Linear Regression Model

    This class wraps nn.Linear in a custom nn.Module class.
    While this is overkill for simple linear regression, it demonstrates
    the pattern you'll use for all PyTorch models.
    """

    def __init__(self, input_size, output_size):
        """
        Initialize the model.

        Args:
            input_size (int): Number of input features
            output_size (int): Number of output features
        """
        # Always call the parent class constructor first!
        super(LinearRegression, self).__init__()

        # Define our linear layer
        # This creates trainable parameters (weight and bias)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        Forward pass: define how data flows through the model.

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Model predictions
        """
        # Simply pass input through our linear layer
        return self.linear(x)

# Create an instance of our custom model
model = LinearRegression(input_size=1, output_size=1)

print("Created custom LinearRegression model!")
print(f"Model architecture:\n{model}\n")

print("Model parameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}, value={param.item():.4f}")
print()

# ============================================================================
# SECTION 4: PREPARING TRAINING DATA
# ============================================================================
print("\nSECTION 4: Preparing Training Data")
print("-" * 80)

"""
From TK's article, we have distance-time data:
- distances: how far traveled (input, x)
- times: how long it took (output, y)

This is a perfect problem for linear regression because time should
increase linearly with distance (assuming constant speed).

The model will learn: time = distance * w + b
Where:
  w ≈ the inverse of speed (time per unit distance)
  b ≈ base time or startup time
"""

# Training data from TK's article
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

print("Training data (distance → time):")
print("  Distance  |  Time")
print("  " + "-" * 20)
for d, t in zip(distances, times):
    print(f"    {d.item():.1f}     |  {t.item():.2f}")
print()

print("Data shapes:")
print(f"  distances: {distances.shape} (4 samples, 1 feature)")
print(f"  times: {times.shape} (4 samples, 1 target)")
print()

print("Goal: Learn the relationship between distance and time")
print("The model will adjust its weight and bias to fit this data!\n")

# ============================================================================
# SECTION 5: DEFINING LOSS FUNCTION AND OPTIMIZER
# ============================================================================
print("\nSECTION 5: Defining Loss Function and Optimizer")
print("-" * 80)

"""
From TK's article: "We need the model, a loss function, and an optimizer
to build the entire training process."

LOSS FUNCTION (How wrong are we?):
  - Measures the difference between predictions and actual values
  - MSE (Mean Squared Error) = average of (prediction - actual)²
  - Lower loss = better model

OPTIMIZER (How do we improve?):
  - Adjusts model parameters to reduce loss
  - SGD (Stochastic Gradient Descent) is a simple but effective optimizer
  - Learning rate (lr) controls how big the adjustment steps are
"""

# Define the loss function: Mean Squared Error
loss_function = nn.MSELoss()

print("Loss Function: MSELoss (Mean Squared Error)")
print("  Formula: MSE = (1/n) * Σ(predicted - actual)²")
print("  Goal: Minimize this value!\n")

# Define the optimizer: Stochastic Gradient Descent
# It will adjust model.parameters() (the weight and bias)
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Optimizer: SGD (Stochastic Gradient Descent)")
print(f"  Learning rate: 0.01")
print(f"  Parameters to optimize: {list(model.parameters())}")
print("  (These are the weight and bias that will be adjusted)\n")

"""
Learning rate (lr=0.01) is crucial:
  - Too high: the model might overshoot and never converge
  - Too low: training will be very slow
  - 0.01 is a common starting point for simple problems
"""

# ============================================================================
# SECTION 6: THE TRAINING LOOP
# ============================================================================
print("\nSECTION 6: The Training Loop")
print("-" * 80)

"""
From TK's article: "The training loop follows this pattern:
  predict → calculate loss → adjust parameters → predict → calculate loss → adjust parameters
The loop goes on until it finishes all epochs."

Each iteration through the loop:
  1. optimizer.zero_grad() - Clear previous gradients
  2. outputs = model(distances) - Forward pass: make predictions
  3. loss = loss_function(outputs, times) - Calculate loss
  4. loss.backward() - Backward pass: compute gradients
  5. optimizer.step() - Update parameters using gradients

An EPOCH is one complete pass through all training data.
We'll train for 500 epochs to give the model time to learn.
"""

print("Starting training for 500 epochs...")
print("Watch the loss decrease as the model learns!\n")

# Store initial parameters to see how they change
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()

print(f"Initial parameters:")
print(f"  Weight: {initial_weight:.4f}")
print(f"  Bias: {initial_bias:.4f}\n")

print("Training progress:")
print("  Epoch  |  Loss")
print("  " + "-" * 30)

# Training loop (from TK's article)
for epoch in range(500):
    # ---- STEP 1: Zero the gradients ----
    # Gradients accumulate by default in PyTorch, so we must clear them
    # each iteration to avoid using stale gradient information
    optimizer.zero_grad()

    # ---- STEP 2: Forward pass (predict) ----
    # Pass our input data (distances) through the model to get predictions
    # This calls model.forward(distances) automatically
    outputs = model(distances)

    # ---- STEP 3: Calculate loss ----
    # Compare our predictions (outputs) with actual values (times)
    # The loss tells us how wrong our predictions are
    loss = loss_function(outputs, times)

    # ---- STEP 4: Backward pass (compute gradients) ----
    # Calculate gradients of the loss with respect to model parameters
    # This is where PyTorch's autograd magic happens!
    # It computes ∂loss/∂weight and ∂loss/∂bias
    loss.backward()

    # ---- STEP 5: Update parameters (learn) ----
    # Adjust the model parameters using the gradients
    # New_param = Old_param - learning_rate * gradient
    optimizer.step()

    # Print progress every 50 epochs
    if epoch % 50 == 0:
        print(f"  {epoch:4d}   |  {loss.item():.6f}")

# Final loss after training
print("  " + "-" * 30)
print(f"  Final  |  {loss.item():.6f}\n")

# Check learned parameters
final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()

print(f"Learned parameters:")
print(f"  Weight: {final_weight:.4f} (was {initial_weight:.4f})")
print(f"  Bias: {final_bias:.4f} (was {initial_bias:.4f})\n")

print("The model has learned! Weight and bias now capture the distance-time relationship.\n")

"""
What just happened?
  - The loss decreased from ~100+ to ~0.0x
  - The weight and bias were adjusted to fit the training data
  - The model now "understands" the relationship between distance and time

Mathematical insight:
  - The weight tells us approximately how much time increases per distance unit
  - The bias captures any base time or offset
  - Together, they form: time = distance * weight + bias
"""

# ============================================================================
# SECTION 7: EVALUATING THE TRAINED MODEL
# ============================================================================
print("\nSECTION 7: Evaluating the Trained Model")
print("-" * 80)

"""
Now that our model is trained, let's test it!
We'll compare its predictions with the actual training data,
and also make predictions for new, unseen distances.
"""

# Put model in evaluation mode (not strictly necessary for simple models,
# but it's good practice for when you use dropout, batch norm, etc.)
model.eval()

# Disable gradient calculation for inference (saves memory and computation)
with torch.no_grad():
    # Test on training data
    print("Predictions on training data:")
    print("  Distance  |  Actual Time  |  Predicted Time  |  Error")
    print("  " + "-" * 60)

    train_predictions = model(distances)
    for d, actual, pred in zip(distances, times, train_predictions):
        error = abs(actual.item() - pred.item())
        print(f"    {d.item():.1f}     |     {actual.item():.2f}      |      {pred.item():.2f}       |  {error:.2f}")
    print()

    # Test on new data
    print("Predictions on NEW distances (not in training data):")
    print("  Distance  |  Predicted Time")
    print("  " + "-" * 30)

    new_distances = torch.tensor([[0.5], [1.5], [2.5], [5.0], [10.0]], dtype=torch.float32)
    new_predictions = model(new_distances)

    for d, pred in zip(new_distances, new_predictions):
        print(f"    {d.item():.1f}     |     {pred.item():.2f}")
    print()

print("The model has successfully learned the distance-time relationship!")
print("It can now make predictions for any distance, even ones it hasn't seen before.\n")

"""
Notice how the model:
  1. Fits the training data very closely (small errors)
  2. Can extrapolate to new distances (like 5.0 and 10.0)
  3. Follows a linear pattern (as expected from linear regression)

The learned equation is: time = distance * {:.4f} + {:.4f}
""".format(final_weight, final_bias)

# ============================================================================
# SECTION 8: PRACTICE PROBLEMS
# ============================================================================
print("\nSECTION 8: Practice Problems")
print("-" * 80)

"""
Now it's your turn! Try these exercises to reinforce your understanding:
"""

print("""
PROBLEM 1: Change the Learning Rate
-------------------------------------
Experiment with different learning rates:
  - Try lr=0.001 (10x smaller)
  - Try lr=0.1 (10x larger)

Questions:
  a) How does a smaller learning rate affect training speed?
  b) How does a larger learning rate affect loss convergence?
  c) Can you find an optimal learning rate for this problem?

Hint: Copy the training loop code and create a new optimizer with a different lr.


PROBLEM 2: Different Training Data
------------------------------------
Create your own training data for a different linear relationship:
  - Example: hours studied vs exam score
  - Example: square footage vs house price
  - Example: years of experience vs salary

Steps:
  1. Create new input and output tensors
  2. Create a new model instance
  3. Train it using the same training loop
  4. Evaluate the learned relationship

Hint: Use torch.tensor([[val1], [val2], ...], dtype=torch.float32)


PROBLEM 3: Multiple Inputs (Challenge!)
-----------------------------------------
Extend the model to handle multiple input features.
For example, predict time based on BOTH distance AND speed:
  time = distance * w1 + speed * w2 + b

Steps:
  1. Create a model with input_size=2 instead of 1
  2. Create training data with 2 features per sample:
     inputs = torch.tensor([[dist1, speed1], [dist2, speed2], ...])
  3. Train and evaluate

Hint: Change model = LinearRegression(input_size=2, output_size=1)


BONUS CHALLENGE: Visualize the Results
----------------------------------------
Install matplotlib and plot:
  - Training data points (scatter plot)
  - Model's learned line
  - Loss over epochs

This will help you visualize what the model learned!
""")

print("\n" + "=" * 80)
print("END OF LESSON")
print("=" * 80)
print("""
Key Takeaways:
  1. Linear regression models y = xw + b
  2. PyTorch provides nn.Linear and nn.Module for building models
  3. Training requires: model, loss function, optimizer
  4. The training loop: zero_grad → forward → loss → backward → step
  5. Gradients tell us how to adjust parameters to reduce loss
  6. Lower loss = better fit to the data

Next Steps:
  - Try the practice problems above
  - Experiment with different data and hyperparameters
  - Learn about other loss functions and optimizers
  - Move on to more complex models (multiple layers, non-linear activations)

Happy learning! 🚀
""")
