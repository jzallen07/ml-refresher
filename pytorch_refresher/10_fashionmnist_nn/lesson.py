"""
================================================================================
LESSON 10: BUILDING A NEURAL NETWORK FOR FASHIONMNIST
================================================================================

This lesson brings together everything we've learned to build a real image
classifier! We'll use the FashionMNIST dataset to classify clothing items.

FashionMNIST is a dataset of 70,000 grayscale images of clothing items:
- 60,000 training images
- 10,000 test images
- Each image is 28x28 pixels
- 10 different clothing categories

This is the CAPSTONE of our PyTorch fundamentals course!

Key concepts covered:
1. Loading real-world image datasets
2. Data preprocessing and normalization
3. Building a multi-layer neural network
4. Training loop with progress tracking
5. Evaluation and accuracy metrics
6. Visualization of results
7. Making predictions on new data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

print("=" * 80)
print("LESSON 10: BUILDING A NEURAL NETWORK FOR FASHIONMNIST")
print("=" * 80)
print(f"\nStarting lesson at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("\n" + "=" * 80)
print("STEP 1: UNDERSTANDING THE DATASET")
print("=" * 80)

print("""
FashionMNIST Dataset Overview:
------------------------------
This dataset was created by Zalando Research as a more challenging replacement
for the classic MNIST handwritten digits dataset. Instead of digits (0-9),
we have 10 categories of clothing items.

Why FashionMNIST?
- Same format as MNIST (28x28 grayscale images)
- More challenging - clothing items have more variation than digits
- Real-world application - image classification for e-commerce
- Perfect size for learning - not too big, not too simple

Dataset Statistics:
- Total images: 70,000
- Training set: 60,000 images
- Test set: 10,000 images
- Image size: 28 × 28 pixels
- Channels: 1 (grayscale)
- Classes: 10 clothing categories
- Each class has exactly 6,000 training images and 1,000 test images
""")

# Define the 10 class names
class_names = [
    'T-shirt/top',  # Class 0
    'Trouser',      # Class 1
    'Pullover',     # Class 2
    'Dress',        # Class 3
    'Coat',         # Class 4
    'Sandal',       # Class 5
    'Shirt',        # Class 6
    'Sneaker',      # Class 7
    'Bag',          # Class 8
    'Ankle boot'    # Class 9
]

print("The 10 clothing categories:")
print("-" * 40)
for idx, name in enumerate(class_names):
    print(f"  {idx}: {name}")

print("\n" + "=" * 80)
print("STEP 2: LOADING AND PREPROCESSING THE DATASET")
print("=" * 80)

print("""
Data Preprocessing Steps:
--------------------------
1. ToTensor(): Converts PIL Image or numpy array to PyTorch tensor
   - Changes shape from (H, W, C) to (C, H, W)
   - Scales pixel values from [0, 255] to [0.0, 1.0]

2. Normalize(): Standardizes the data
   - Formula: output = (input - mean) / std
   - For FashionMNIST: mean=0.5, std=0.5
   - This scales values from [0, 1] to [-1, 1]

Why normalize?
- Helps the neural network train faster
- Makes gradient descent more stable
- Prevents certain features from dominating due to scale
- Centers the data around zero (ideal for neural networks)
""")

# Define the transforms to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),           # Convert to tensor and scale to [0, 1]
    transforms.Normalize((0.5,),     # Mean for single channel (grayscale)
                        (0.5,))      # Std for single channel
])

print("Creating data transform pipeline:")
print("  1. ToTensor() - Convert images to tensors")
print("  2. Normalize(mean=0.5, std=0.5) - Standardize pixel values")

# Set the data directory
data_dir = '/Users/zack/dev/ml-refresher/data/fashionmnist'
print(f"\nData will be stored in: {data_dir}")

print("\nDownloading and loading training dataset...")
print("(This may take a moment on first run - dataset is ~30MB)")

# Load the training dataset
train_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=True,           # Load training data
    download=True,        # Download if not present
    transform=transform   # Apply our preprocessing
)

print("✓ Training dataset loaded!")

print("\nDownloading and loading test dataset...")

# Load the test dataset
test_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=False,          # Load test data
    download=True,
    transform=transform
)

print("✓ Test dataset loaded!")

# Print dataset information
print("\n" + "-" * 60)
print("Dataset Information:")
print("-" * 60)
print(f"Training samples: {len(train_dataset):,}")
print(f"Test samples: {len(test_dataset):,}")
print(f"Total samples: {len(train_dataset) + len(test_dataset):,}")
print(f"Number of classes: {len(class_names)}")

# Examine a single sample
sample_image, sample_label = train_dataset[0]
print(f"\nSample image tensor shape: {sample_image.shape}")
print(f"  - Channels: {sample_image.shape[0]} (grayscale)")
print(f"  - Height: {sample_image.shape[1]} pixels")
print(f"  - Width: {sample_image.shape[2]} pixels")
print(f"Sample label: {sample_label} ({class_names[sample_label]})")
print(f"Pixel value range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
print("  (Values are normalized to roughly [-1, 1])")

print("\n" + "=" * 80)
print("STEP 3: VISUALIZING SAMPLE IMAGES")
print("=" * 80)

print("""
Let's visualize some sample images to understand what we're working with.
This helps us verify the data loaded correctly and understand the task.
""")

# Create a figure to show sample images
fig, axes = plt.subplots(3, 5, figsize=(12, 7))
fig.suptitle('FashionMNIST Sample Images', fontsize=16, fontweight='bold')

print("\nDisplaying 15 random training samples...")

for idx, ax in enumerate(axes.flat):
    # Get a random sample
    random_idx = np.random.randint(len(train_dataset))
    image, label = train_dataset[random_idx]

    # Convert from (C, H, W) to (H, W) for display
    image = image.squeeze()  # Remove channel dimension

    # Denormalize for visualization: reverse the (x - 0.5) / 0.5 transform
    # If x_norm = (x - 0.5) / 0.5, then x = x_norm * 0.5 + 0.5
    image = image * 0.5 + 0.5  # Scale from [-1, 1] back to [0, 1]

    # Display the image
    ax.imshow(image, cmap='gray')
    ax.set_title(f'{class_names[label]}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
sample_path = os.path.join(data_dir, 'sample_images.png')
plt.savefig(sample_path, dpi=150, bbox_inches='tight')
print(f"✓ Sample images saved to: {sample_path}")
plt.close()

print("\n" + "=" * 80)
print("STEP 4: CREATING DATA LOADERS")
print("=" * 80)

print("""
DataLoaders wrap our datasets and provide:
------------------------------------------
1. Batching: Group samples into batches for efficient training
   - Instead of processing 1 image at a time, we process 64 at once
   - GPU parallelization works much better with batches

2. Shuffling: Randomize the order of samples
   - Prevents the model from learning the order of examples
   - Helps the model generalize better
   - Only shuffle training data (not test data)

3. Automatic iteration: Easy to loop through batches
   - Handles the complexity of batching automatically
   - Loads data in the background (can use multiple workers)

Batch Size Choice:
- Larger batches (128, 256): Faster training, more memory
- Smaller batches (32, 64): Slower training, less memory, sometimes better generalization
- We'll use 64 as a good middle ground
""")

batch_size = 64
print(f"Setting batch_size = {batch_size}")

# Create DataLoader for training data
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,        # Shuffle training data
    num_workers=0        # Use main process (set to 2-4 for faster loading)
)

# Create DataLoader for test data
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,       # Don't shuffle test data (order doesn't matter)
    num_workers=0
)

print("\n" + "-" * 60)
print("DataLoader Information:")
print("-" * 60)
print(f"Training batches: {len(train_loader)}")
print(f"  ({len(train_dataset)} samples ÷ {batch_size} batch_size = "
      f"{len(train_dataset) / batch_size:.1f} batches)")
print(f"Test batches: {len(test_loader)}")
print(f"  ({len(test_dataset)} samples ÷ {batch_size} batch_size = "
      f"{len(test_dataset) / batch_size:.1f} batches)")

# Get a batch to examine the shape
sample_batch_images, sample_batch_labels = next(iter(train_loader))
print(f"\nSample batch shapes:")
print(f"  Images: {sample_batch_images.shape}")
print(f"    - Batch size: {sample_batch_images.shape[0]} images")
print(f"    - Channels: {sample_batch_images.shape[1]} (grayscale)")
print(f"    - Height: {sample_batch_images.shape[2]} pixels")
print(f"    - Width: {sample_batch_images.shape[3]} pixels")
print(f"  Labels: {sample_batch_labels.shape}")
print(f"    - {sample_batch_labels.shape[0]} labels (one per image)")

print("\n" + "=" * 80)
print("STEP 5: BUILDING THE NEURAL NETWORK")
print("=" * 80)

print("""
Network Architecture:
---------------------
We're building a feedforward neural network (also called a Multi-Layer Perceptron).

Structure:
    Input (28×28 image)
        ↓ Flatten
    784 neurons
        ↓ Linear + ReLU
    512 neurons (Hidden Layer 1)
        ↓ Linear + ReLU
    256 neurons (Hidden Layer 2)
        ↓ Linear
    10 neurons (Output - one per class)

Why this architecture?
----------------------
1. Flatten: Neural networks expect 1D input vectors
   - Converts 28×28 image into 784-length vector

2. First Hidden Layer (784 → 512):
   - Learns basic patterns from raw pixels
   - 512 neurons provide enough capacity to learn

3. Second Hidden Layer (512 → 256):
   - Learns higher-level combinations of patterns
   - Gradual decrease helps the network learn hierarchy

4. Output Layer (256 → 10):
   - 10 outputs (one per class)
   - Raw scores called "logits"
   - Higher score = model more confident that class is correct

ReLU Activation Function:
--------------------------
ReLU(x) = max(0, x)
- Keeps positive values unchanged
- Sets negative values to zero
- Introduces non-linearity (crucial for learning complex patterns)
- Fast to compute and train
- Helps prevent vanishing gradient problem

Why not use activation on output?
- CrossEntropyLoss expects raw logits (unnormalized scores)
- It applies softmax internally for efficiency
""")


class FashionNN(nn.Module):
    """
    Neural Network for FashionMNIST Classification

    Architecture:
    - Input: 28x28 grayscale images (flattened to 784)
    - Hidden Layer 1: 512 neurons with ReLU activation
    - Hidden Layer 2: 256 neurons with ReLU activation
    - Output Layer: 10 neurons (one per class)

    Total Parameters: ~400,000 (we'll calculate this below)
    """

    def __init__(self):
        super(FashionNN, self).__init__()

        # Flatten layer - converts 2D image to 1D vector
        # Input: (batch_size, 1, 28, 28)
        # Output: (batch_size, 784)
        self.flatten = nn.Flatten()

        # First hidden layer - learn basic features
        # Input: 784 (28 * 28 pixels)
        # Output: 512 neurons
        # Parameters: (784 * 512) + 512 = 401,920
        self.fc1 = nn.Linear(784, 512)

        # Second hidden layer - learn combinations of features
        # Input: 512 neurons
        # Output: 256 neurons
        # Parameters: (512 * 256) + 256 = 131,328
        self.fc2 = nn.Linear(512, 256)

        # Output layer - map to 10 classes
        # Input: 256 neurons
        # Output: 10 neurons (one per class)
        # Parameters: (256 * 10) + 10 = 2,570
        self.fc3 = nn.Linear(256, 10)

        # ReLU activation function
        # We'll apply this after fc1 and fc2
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output tensor of shape (batch_size, 10) containing logits
        """
        # Print shapes for educational purposes (only for first call)
        if not hasattr(self, '_shapes_printed'):
            print("\n" + "-" * 60)
            print("Forward Pass - Tensor Shapes:")
            print("-" * 60)
            print(f"Input shape: {x.shape}")
            self._shapes_printed = True

        # Step 1: Flatten the image from 2D to 1D
        # (batch_size, 1, 28, 28) → (batch_size, 784)
        x = self.flatten(x)
        if hasattr(self, '_shapes_printed') and self._shapes_printed:
            print(f"After flatten: {x.shape}")

        # Step 2: First hidden layer with ReLU activation
        # (batch_size, 784) → (batch_size, 512)
        x = self.fc1(x)
        if hasattr(self, '_shapes_printed') and self._shapes_printed:
            print(f"After fc1 (linear): {x.shape}")
        x = self.relu(x)
        if hasattr(self, '_shapes_printed') and self._shapes_printed:
            print(f"After relu: {x.shape} (same shape, but negative values → 0)")

        # Step 3: Second hidden layer with ReLU activation
        # (batch_size, 512) → (batch_size, 256)
        x = self.fc2(x)
        if hasattr(self, '_shapes_printed') and self._shapes_printed:
            print(f"After fc2 (linear): {x.shape}")
        x = self.relu(x)
        if hasattr(self, '_shapes_printed') and self._shapes_printed:
            print(f"After relu: {x.shape}")

        # Step 4: Output layer (no activation - raw logits)
        # (batch_size, 256) → (batch_size, 10)
        x = self.fc3(x)
        if hasattr(self, '_shapes_printed') and self._shapes_printed:
            print(f"Output (logits): {x.shape}")
            print("-" * 60)
            self._shapes_printed = False  # Only print once

        return x


# Create an instance of our model
print("\nCreating the neural network...")
model = FashionNN()

print("✓ Model created successfully!")

# Print model architecture
print("\n" + "-" * 60)
print("Model Architecture:")
print("-" * 60)
print(model)

# Calculate and display the number of parameters
def count_parameters(model):
    """Count the total number of trainable parameters in the model"""
    total = 0
    details = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()
            total += param_count
            details.append((name, param_count, list(parameter.shape)))
    return total, details

total_params, param_details = count_parameters(model)

print("\n" + "-" * 60)
print("Model Parameters:")
print("-" * 60)
for name, count, shape in param_details:
    print(f"  {name:20s}: {count:>10,} parameters  {shape}")
print("-" * 60)
print(f"  {'TOTAL':20s}: {total_params:>10,} parameters")
print("-" * 60)

print(f"""
Parameter Breakdown:
- fc1 (784 → 512): {(784 * 512 + 512):,} parameters
  (784 weights per neuron × 512 neurons + 512 biases)

- fc2 (512 → 256): {(512 * 256 + 256):,} parameters
  (512 weights per neuron × 256 neurons + 256 biases)

- fc3 (256 → 10): {(256 * 10 + 10):,} parameters
  (256 weights per neuron × 10 neurons + 10 biases)

These {total_params:,} parameters will be learned during training!
""")

# Test the model with a sample batch to verify it works
print("\n" + "-" * 60)
print("Testing Model with Sample Batch:")
print("-" * 60)
model.eval()  # Set to evaluation mode
with torch.no_grad():
    sample_output = model(sample_batch_images)

print(f"\nSample output shape: {sample_output.shape}")
print(f"  - Batch size: {sample_output.shape[0]}")
print(f"  - Number of classes: {sample_output.shape[1]}")

print(f"\nFirst sample's output (logits for 10 classes):")
print(sample_output[0])
print("""
These are raw scores (logits) for each class.
Higher score = model thinks this class is more likely.
We'll use softmax to convert these to probabilities later.
""")

# Show what the predicted class would be
predicted_class = sample_output[0].argmax()
print(f"Predicted class: {predicted_class} ({class_names[predicted_class]})")
print(f"Actual class: {sample_batch_labels[0]} ({class_names[sample_batch_labels[0]]})")
print("(Prediction is random at this point - model hasn't been trained yet!)")

print("\n" + "=" * 80)
print("STEP 6: DEFINING LOSS FUNCTION AND OPTIMIZER")
print("=" * 80)

print("""
Loss Function: CrossEntropyLoss
--------------------------------
CrossEntropyLoss is the standard loss function for multi-class classification.

What it does:
1. Applies softmax to convert logits to probabilities:
   probability(class_i) = exp(logit_i) / sum(exp(all_logits))

2. Computes the negative log-likelihood:
   loss = -log(probability_of_correct_class)

Why it works:
- If model is confident and correct: probability ≈ 1, loss ≈ 0 (good!)
- If model is confident but wrong: probability ≈ 0, loss ≈ ∞ (bad!)
- Encourages model to be confident in the correct class

Example:
  True class: 2 (Pullover)
  Model output: [0.1, 0.2, 0.6, 0.05, 0.05] (after softmax)
  Loss = -log(0.6) ≈ 0.51

  If model improves to: [0.05, 0.05, 0.85, 0.025, 0.025]
  Loss = -log(0.85) ≈ 0.16 (lower is better!)
""")

# Define the loss function
criterion = nn.CrossEntropyLoss()
print("✓ Loss function created: CrossEntropyLoss")

print("""
Optimizer: Adam (Adaptive Moment Estimation)
--------------------------------------------
Adam is one of the most popular optimizers for deep learning.

What it does:
- Adapts the learning rate for each parameter individually
- Combines ideas from RMSprop and SGD with momentum
- Uses moving averages of gradients and squared gradients

Why Adam?
- Works well "out of the box" with default settings
- Robust to choice of learning rate
- Handles sparse gradients well
- Commonly used as a default optimizer

Key hyperparameters:
- learning_rate (lr): How big a step to take (we'll use 0.001)
  - Too high: Training unstable, might not converge
  - Too low: Training very slow
  - 0.001 is a good default starting point

Other popular optimizers:
- SGD: Simple, requires careful tuning
- RMSprop: Good for RNNs
- AdamW: Adam with better weight decay (great for transformers)
""")

learning_rate = 0.001
print(f"Setting learning rate = {learning_rate}")

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print("✓ Optimizer created: Adam")

print(f"\nOptimizer configuration:")
print(f"  - Algorithm: Adam")
print(f"  - Learning rate: {learning_rate}")
print(f"  - Parameters to optimize: {total_params:,}")
print(f"  - Default beta1: 0.9 (momentum for gradients)")
print(f"  - Default beta2: 0.999 (momentum for squared gradients)")

print("\n" + "=" * 80)
print("STEP 7: TRAINING THE NEURAL NETWORK")
print("=" * 80)

print("""
Training Loop Overview:
-----------------------
Training a neural network is an iterative process:

For each epoch (complete pass through training data):
    For each batch of images:
        1. Forward Pass: Feed images through network → get predictions
        2. Compute Loss: Compare predictions to true labels
        3. Backward Pass: Compute gradients (how to adjust each parameter)
        4. Update Parameters: Adjust weights to reduce loss
        5. Reset Gradients: Clear gradients for next iteration

Key Concepts:
-------------
- Epoch: One complete pass through all training data
  * We'll train for multiple epochs to learn progressively

- Batch: Group of samples processed together
  * More efficient than processing one at a time
  * Provides more stable gradient estimates

- Gradient: Direction and magnitude to adjust each parameter
  * Computed by backpropagation
  * Tells us how to change weights to reduce loss

- optimizer.zero_grad(): Reset gradients to zero
  * PyTorch accumulates gradients by default
  * Must clear before each backward pass

- loss.backward(): Compute gradients via backpropagation
  * Calculates ∂loss/∂weight for every parameter
  * Uses chain rule to propagate through layers

- optimizer.step(): Update parameters using gradients
  * weights = weights - learning_rate * gradient
  * Adam uses more sophisticated update rule

Training Progress:
------------------
We'll track:
- Loss per epoch (should decrease over time)
- Accuracy on test set after each epoch (should increase)
- Time per epoch
""")

# Training configuration
num_epochs = 8
print(f"Training configuration:")
print(f"  - Number of epochs: {num_epochs}")
print(f"  - Batch size: {batch_size}")
print(f"  - Batches per epoch: {len(train_loader)}")
print(f"  - Total training steps: {num_epochs * len(train_loader):,}")
print(f"  - Learning rate: {learning_rate}")

# Lists to store metrics for visualization
train_losses = []
test_accuracies = []

print("\n" + "=" * 80)
print("STARTING TRAINING...")
print("=" * 80)

# Start training timer
training_start_time = datetime.now()

for epoch in range(num_epochs):
    epoch_start_time = datetime.now()

    print(f"\n{'=' * 80}")
    print(f"EPOCH {epoch + 1}/{num_epochs}")
    print(f"{'=' * 80}")

    # -------------------------------------------------------------------------
    # TRAINING PHASE
    # -------------------------------------------------------------------------
    model.train()  # Set model to training mode
    running_loss = 0.0

    print("\nTraining...")
    print("-" * 60)

    for batch_idx, (images, labels) in enumerate(train_loader):
        # images shape: (batch_size, 1, 28, 28)
        # labels shape: (batch_size,)

        # STEP 1: Zero the gradients
        # --------------------------
        # Clear gradients from previous iteration
        # PyTorch accumulates gradients, so we must reset them
        optimizer.zero_grad()

        # STEP 2: Forward pass
        # --------------------
        # Feed images through the network to get predictions
        outputs = model(images)  # Shape: (batch_size, 10)
        # outputs contains raw logits for each class

        # STEP 3: Compute loss
        # --------------------
        # Compare predictions with true labels
        loss = criterion(outputs, labels)
        # CrossEntropyLoss automatically applies softmax and computes NLL

        # STEP 4: Backward pass
        # ---------------------
        # Compute gradients using backpropagation
        loss.backward()
        # This computes ∂loss/∂weight for every parameter in the model

        # STEP 5: Update parameters
        # --------------------------
        # Adjust weights using computed gradients
        optimizer.step()
        # Adam uses gradients to update: weight = weight - lr * gradient

        # Track statistics
        running_loss += loss.item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            progress = (batch_idx + 1) / len(train_loader) * 100
            print(f"  Batch [{batch_idx + 1:>4}/{len(train_loader)}] "
                  f"({progress:>5.1f}%)  |  Loss: {loss.item():.4f}  |  "
                  f"Avg Loss: {avg_loss:.4f}")

    # Calculate average loss for this epoch
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    print("-" * 60)
    print(f"✓ Training complete - Avg Loss: {epoch_loss:.4f}")

    # -------------------------------------------------------------------------
    # EVALUATION PHASE
    # -------------------------------------------------------------------------
    print("\nEvaluating on test set...")
    print("-" * 60)

    model.eval()  # Set model to evaluation mode
    # In eval mode, layers like dropout and batch norm behave differently

    correct = 0
    total = 0

    # Disable gradient computation for evaluation (saves memory and computation)
    with torch.no_grad():
        for images, labels in test_loader:
            # Forward pass
            outputs = model(images)  # Shape: (batch_size, 10)

            # Get predicted class (index of maximum logit)
            # outputs are logits, higher value = more confident
            _, predicted = torch.max(outputs.data, 1)
            # predicted shape: (batch_size,)

            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"✓ Evaluation complete")
    print(f"  Correct predictions: {correct:,} / {total:,}")
    print(f"  Accuracy: {accuracy:.2f}%")

    # Calculate epoch time
    epoch_time = (datetime.now() - epoch_start_time).total_seconds()
    print(f"\n⏱ Epoch completed in {epoch_time:.1f} seconds")

# Calculate total training time
total_training_time = (datetime.now() - training_start_time).total_seconds()

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"Total training time: {total_training_time:.1f} seconds "
      f"({total_training_time / 60:.1f} minutes)")
print(f"Average time per epoch: {total_training_time / num_epochs:.1f} seconds")
print(f"\nFinal Results:")
print(f"  - Final training loss: {train_losses[-1]:.4f}")
print(f"  - Final test accuracy: {test_accuracies[-1]:.2f}%")
print(f"  - Best test accuracy: {max(test_accuracies):.2f}% "
      f"(Epoch {test_accuracies.index(max(test_accuracies)) + 1})")

print("\n" + "=" * 80)
print("STEP 8: VISUALIZING TRAINING PROGRESS")
print("=" * 80)

print("""
Visualizing training metrics helps us understand:
- Is the model learning? (loss should decrease)
- Is the model improving? (accuracy should increase)
- Is the model overfitting? (training accuracy >> test accuracy)
- Has training converged? (metrics plateau)
""")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('FashionMNIST Training Progress', fontsize=16, fontweight='bold')

# Plot 1: Training Loss
ax1.plot(range(1, num_epochs + 1), train_losses,
         marker='o', linewidth=2, markersize=8, color='#e74c3c')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Average Loss', fontsize=12)
ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, num_epochs + 1))

# Annotate first and last points
ax1.annotate(f'{train_losses[0]:.3f}',
             xy=(1, train_losses[0]),
             xytext=(10, 10),
             textcoords='offset points',
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
ax1.annotate(f'{train_losses[-1]:.3f}',
             xy=(num_epochs, train_losses[-1]),
             xytext=(10, -10),
             textcoords='offset points',
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# Plot 2: Test Accuracy
ax2.plot(range(1, num_epochs + 1), test_accuracies,
         marker='s', linewidth=2, markersize=8, color='#3498db')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Test Accuracy Over Time', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(1, num_epochs + 1))
ax2.set_ylim([0, 100])

# Annotate best accuracy
best_acc_epoch = test_accuracies.index(max(test_accuracies)) + 1
ax2.annotate(f'Best: {max(test_accuracies):.2f}%',
             xy=(best_acc_epoch, max(test_accuracies)),
             xytext=(10, -15),
             textcoords='offset points',
             fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.tight_layout()
progress_path = os.path.join(data_dir, 'training_progress.png')
plt.savefig(progress_path, dpi=150, bbox_inches='tight')
print(f"✓ Training progress plots saved to: {progress_path}")
plt.close()

# Print summary statistics
print("\n" + "-" * 60)
print("Training Metrics Summary:")
print("-" * 60)
print("\nLoss per epoch:")
for i, loss in enumerate(train_losses, 1):
    print(f"  Epoch {i}: {loss:.4f}")

print("\nAccuracy per epoch:")
for i, acc in enumerate(test_accuracies, 1):
    marker = " ⭐" if acc == max(test_accuracies) else ""
    print(f"  Epoch {i}: {acc:.2f}%{marker}")

print("\n" + "=" * 80)
print("STEP 9: MAKING PREDICTIONS AND VISUALIZING RESULTS")
print("=" * 80)

print("""
Now let's see how our trained model performs on individual images!

We'll:
1. Get predictions on test images
2. Show the images with predicted vs actual labels
3. Highlight correct predictions (green) and incorrect predictions (red)
4. Display the model's confidence (probability) for each prediction
""")

# Set model to evaluation mode
model.eval()

# Get a batch of test images
test_images, test_labels = next(iter(test_loader))

print(f"\nGetting predictions for {len(test_images)} test images...")

# Get predictions
with torch.no_grad():
    outputs = model(test_images)
    # outputs shape: (batch_size, 10) - logits for each class

    # Convert logits to probabilities using softmax
    probabilities = F.softmax(outputs, dim=1)
    # probabilities shape: (batch_size, 10)
    # Each row sums to 1.0

    # Get predicted class and confidence
    confidences, predictions = torch.max(probabilities, 1)
    # predictions: predicted class index
    # confidences: probability of predicted class

print("✓ Predictions complete!")

# Show prediction details for first 5 images
print("\n" + "-" * 60)
print("Sample Predictions (first 5 images):")
print("-" * 60)
for i in range(5):
    pred_class = predictions[i].item()
    true_class = test_labels[i].item()
    confidence = confidences[i].item() * 100
    correct = "✓" if pred_class == true_class else "✗"

    print(f"\nImage {i + 1}: {correct}")
    print(f"  Predicted: {class_names[pred_class]} ({confidence:.1f}% confidence)")
    print(f"  Actual: {class_names[true_class]}")

    if pred_class == true_class:
        print(f"  Status: Correct! 🎉")
    else:
        print(f"  Status: Incorrect ❌")

# Create visualization
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Model Predictions on Test Set\n'
             'Green = Correct, Red = Incorrect',
             fontsize=16, fontweight='bold')

print("\n" + "-" * 60)
print("Creating visualization of predictions...")
print("-" * 60)

correct_count = 0
incorrect_count = 0

for idx, ax in enumerate(axes.flat):
    if idx >= len(test_images):
        ax.axis('off')
        continue

    # Get image and denormalize for display
    image = test_images[idx].squeeze()  # Remove channel dimension
    image = image * 0.5 + 0.5  # Denormalize from [-1, 1] to [0, 1]

    # Get prediction and true label
    pred_class = predictions[idx].item()
    true_class = test_labels[idx].item()
    confidence = confidences[idx].item()

    # Check if correct
    is_correct = (pred_class == true_class)
    if is_correct:
        correct_count += 1
    else:
        incorrect_count += 1

    # Display image
    ax.imshow(image, cmap='gray')

    # Set title color based on correctness
    title_color = 'green' if is_correct else 'red'
    title = f'Pred: {class_names[pred_class]}\n({confidence*100:.0f}%)'
    if not is_correct:
        title += f'\nTrue: {class_names[true_class]}'

    ax.set_title(title, fontsize=8, color=title_color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
predictions_path = os.path.join(data_dir, 'model_predictions.png')
plt.savefig(predictions_path, dpi=150, bbox_inches='tight')
print(f"✓ Predictions visualization saved to: {predictions_path}")
plt.close()

print(f"\nVisualization shows {correct_count} correct and {incorrect_count} incorrect "
      f"predictions")
print(f"Accuracy on this batch: {correct_count / (correct_count + incorrect_count) * 100:.1f}%")

print("\n" + "=" * 80)
print("STEP 10: ANALYZING MODEL PERFORMANCE")
print("=" * 80)

print("""
Let's compute detailed performance metrics to understand:
- Which classes the model predicts well
- Which classes are confused with each other
- Overall model performance
""")

# Evaluate on entire test set
print("\nEvaluating model on entire test set...")
model.eval()

all_predictions = []
all_labels = []
all_confidences = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_confidences = np.array(all_confidences)

print("✓ Evaluation complete!")

# Calculate per-class accuracy
print("\n" + "-" * 60)
print("Per-Class Performance:")
print("-" * 60)
print(f"{'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Avg Confidence'}")
print("-" * 60)

for class_idx in range(len(class_names)):
    # Find all instances of this class
    class_mask = (all_labels == class_idx)
    class_predictions = all_predictions[class_mask]
    class_true_labels = all_labels[class_mask]
    class_confidences = all_confidences[class_mask]

    # Calculate accuracy
    correct = (class_predictions == class_true_labels).sum()
    total = len(class_true_labels)
    accuracy = correct / total * 100 if total > 0 else 0
    avg_confidence = class_confidences.mean() * 100

    print(f"{class_names[class_idx]:<15} {correct:<10} {total:<10} "
          f"{accuracy:>6.2f}%    {avg_confidence:>6.1f}%")

# Overall statistics
overall_accuracy = (all_predictions == all_labels).sum() / len(all_labels) * 100
avg_confidence = all_confidences.mean() * 100

print("-" * 60)
print(f"{'OVERALL':<15} {(all_predictions == all_labels).sum():<10} "
      f"{len(all_labels):<10} {overall_accuracy:>6.2f}%    {avg_confidence:>6.1f}%")
print("-" * 60)

# Find most confident correct and incorrect predictions
correct_mask = all_predictions == all_labels
incorrect_mask = ~correct_mask

most_confident_correct_idx = np.argmax(all_confidences * correct_mask)
most_confident_incorrect_idx = np.argmax(all_confidences * incorrect_mask)

print("\n" + "-" * 60)
print("Interesting Predictions:")
print("-" * 60)
print(f"\nMost confident CORRECT prediction:")
print(f"  Predicted: {class_names[all_predictions[most_confident_correct_idx]]}")
print(f"  Actual: {class_names[all_labels[most_confident_correct_idx]]}")
print(f"  Confidence: {all_confidences[most_confident_correct_idx] * 100:.2f}%")

print(f"\nMost confident INCORRECT prediction:")
print(f"  Predicted: {class_names[all_predictions[most_confident_incorrect_idx]]}")
print(f"  Actual: {class_names[all_labels[most_confident_incorrect_idx]]}")
print(f"  Confidence: {all_confidences[most_confident_incorrect_idx] * 100:.2f}%")
print("  (This might indicate similar-looking items!)")

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"""
🎉 Congratulations! You've successfully trained a neural network! 🎉

Model Performance:
------------------
Final Test Accuracy: {overall_accuracy:.2f}%
Average Confidence: {avg_confidence:.1f}%
Training Time: {total_training_time / 60:.1f} minutes

What We Learned:
----------------
✓ How to load and preprocess image datasets
✓ How to create DataLoaders for efficient batching
✓ How to build a multi-layer neural network with PyTorch
✓ How to implement a training loop with forward and backward passes
✓ How to evaluate model performance on a test set
✓ How to visualize training progress and predictions
✓ How to analyze per-class performance

Model Architecture:
-------------------
- Input: 28×28 grayscale images (784 pixels)
- Hidden Layer 1: 512 neurons with ReLU
- Hidden Layer 2: 256 neurons with ReLU
- Output: 10 classes (clothing categories)
- Total Parameters: {total_params:,}

Training Configuration:
-----------------------
- Epochs: {num_epochs}
- Batch Size: {batch_size}
- Learning Rate: {learning_rate}
- Optimizer: Adam
- Loss Function: CrossEntropyLoss

Files Created:
--------------
1. {os.path.join(data_dir, 'sample_images.png')}
   - Sample images from the dataset

2. {os.path.join(data_dir, 'training_progress.png')}
   - Loss and accuracy curves over training

3. {os.path.join(data_dir, 'model_predictions.png')}
   - Visualization of model predictions

Next Steps:
-----------
- Try different architectures (more/fewer layers, different sizes)
- Experiment with learning rate and batch size
- Add dropout for regularization
- Try different optimizers (SGD, RMSprop)
- Implement early stopping
- Save and load the trained model
- Use convolutional layers (CNNs) for better image performance
""")

print("\n" + "=" * 80)
print("PRACTICE PROBLEMS")
print("=" * 80)

print("""
Test your understanding with these exercises:

1. EASY: Experiment with hyperparameters
   - Change the learning rate to 0.0001 and 0.01
   - Change the batch size to 32 and 128
   - How does this affect training speed and accuracy?

2. EASY: Add more epochs
   - Train for 15-20 epochs instead of 8
   - Does the accuracy keep improving or plateau?
   - Plot the learning curves to visualize this

3. MEDIUM: Modify the architecture
   - Add a third hidden layer with 128 neurons
   - Try different layer sizes (e.g., 1024 → 512 → 256 → 128 → 10)
   - Try a smaller network (256 → 128 → 10)
   - How does this affect number of parameters and performance?

4. MEDIUM: Add dropout for regularization
   - Add nn.Dropout(0.2) after each ReLU activation
   - This randomly sets 20% of neurons to zero during training
   - Does this improve or hurt performance?
   - Hint: Remember dropout behaves differently in train vs eval mode!

5. MEDIUM: Implement model saving and loading
   - Save the trained model: torch.save(model.state_dict(), 'model.pth')
   - Load it later: model.load_state_dict(torch.load('model.pth'))
   - Verify it produces the same predictions

6. HARD: Create a confusion matrix
   - Build a 10×10 matrix showing predicted vs actual classes
   - Which classes are most often confused with each other?
   - Visualize it as a heatmap using matplotlib
   - Hint: Use a nested loop over all class pairs

7. HARD: Implement learning rate scheduling
   - Start with high learning rate, decrease over time
   - Use torch.optim.lr_scheduler.StepLR or ReduceLROnPlateau
   - Does this improve final accuracy?

8. HARD: Try a different optimizer
   - Replace Adam with SGD with momentum:
     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   - How does training differ?
   - Which converges faster?

9. ADVANCED: Implement early stopping
   - Stop training if test accuracy doesn't improve for 3 epochs
   - Save the best model (highest test accuracy)
   - This prevents overfitting!

10. ADVANCED: Build a CNN instead
    - Replace the fully connected layers with convolutional layers
    - Use nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d
    - CNNs are much better for image data!
    - Can you beat the current accuracy?

Challenge:
----------
Can you get the test accuracy above 90%? Try combining multiple
improvements: better architecture, dropout, learning rate scheduling,
data augmentation, etc.

Remember:
---------
- Always print shapes when debugging
- Visualize your results to understand what's happening
- Experiment and have fun!
- Machine learning is about iteration and experimentation
""")

print("\n" + "=" * 80)
print(f"Lesson completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print("\n🚀 You're now ready to build real deep learning models with PyTorch!")
print("This is just the beginning - there's so much more to explore!\n")
