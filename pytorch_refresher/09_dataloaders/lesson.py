"""
================================================================================
LESSON 09: DATALOADERS & BATCHING IN PYTORCH
================================================================================

Topics Covered:
1. Creating custom Dataset classes
2. Understanding DataLoader parameters (batch_size, shuffle, num_workers, etc.)
3. Data splitting (train/validation/test) with random_split
4. Iterating through batches efficiently
5. Effect of shuffling on data order
6. Comparing batch sizes and their impact
7. Multi-worker data loading for performance
8. Custom collate functions for variable-size data
9. Complete train/val/test DataLoader setup pattern

Learning Objectives:
- Master the DataLoader API for efficient batch processing
- Understand how to split datasets properly
- Learn performance optimization with num_workers
- Create production-ready data loading pipelines
- Debug and time data loading operations

================================================================================
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from pathlib import Path
from typing import Tuple, List, Optional

print("=" * 80)
print("LESSON 09: DATALOADERS & BATCHING")
print("=" * 80)
print()

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"Number of CPU cores available: {os.cpu_count()}")
print()


################################################################################
# SECTION 1: CREATING A CUSTOM DATASET CLASS
################################################################################

print("=" * 80)
print("SECTION 1: CREATING A CUSTOM DATASET CLASS")
print("=" * 80)
print()

print("In PyTorch, the Dataset class is an abstract class representing a dataset.")
print("To create a custom dataset, we need to implement three methods:")
print("  1. __init__()  - Initialize the dataset (load file paths, labels, etc.)")
print("  2. __len__()   - Return the total number of samples")
print("  3. __getitem__() - Return a single sample given an index")
print()
print("-" * 60)


class OxfordFlowersDataset(Dataset):
    """
    Custom Dataset for Oxford Flowers 102 dataset.

    This class demonstrates how to create a PyTorch Dataset that:
    - Loads images from disk on-the-fly (memory efficient)
    - Applies transforms to preprocess images
    - Returns (image, label) pairs

    Args:
        image_dir (str or Path): Directory containing .jpg images
        labels (list): List of integer labels (one per image)
        transform (callable, optional): Transform to apply to images
    """

    def __init__(self, image_dir, labels, transform=None):
        """
        Initialize the dataset.

        This method is called once when you create the dataset object.
        It should load metadata (file paths, labels) but NOT load all images
        into memory - that would be too memory intensive!
        """
        self.image_dir = Path(image_dir)
        self.labels = labels
        self.transform = transform

        # Get all image paths and sort them for consistency
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))

        # Make sure we have the right number of labels
        assert len(self.image_paths) == len(self.labels), \
            f"Number of images ({len(self.image_paths)}) != number of labels ({len(self.labels)})"

        print(f"  Dataset initialized with {len(self.image_paths)} images")

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        This is used by DataLoader to know how many iterations to do per epoch.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and return a single sample from the dataset.

        This method is called by DataLoader to fetch individual samples.
        It's called many times, so it should be efficient!

        Args:
            idx (int): Index of the sample to retrieve (0 to len(dataset)-1)

        Returns:
            tuple: (image, label) where image is a PIL Image or tensor
        """
        # Load the image from disk
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Ensure RGB format

        # Get the corresponding label
        label = self.labels[idx]

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label


class SyntheticDataset(Dataset):
    """
    Fallback synthetic dataset for demonstration when real data isn't available.

    This creates random images and labels on-the-fly.
    Useful for testing and learning without requiring downloads.
    """

    def __init__(self, num_samples=1000, image_size=(224, 224), num_classes=10, transform=None):
        """
        Initialize synthetic dataset.

        Args:
            num_samples: Total number of synthetic samples to generate
            image_size: Size of each image (height, width)
            num_classes: Number of different classes
            transform: Optional transform to apply
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform

        # Pre-generate labels (but not images - those are generated on-the-fly)
        self.labels = torch.randint(0, num_classes, (num_samples,)).tolist()

        print(f"  Synthetic dataset initialized with {num_samples} samples")
        print(f"  Image size: {image_size}, Classes: {num_classes}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a random image and return with its label.

        In practice, you'd load real data here. But for learning/testing,
        synthetic data is perfectly fine!
        """
        # Generate a random image (values between 0 and 255)
        # Shape: (height, width, channels)
        img_array = np.random.randint(0, 256,
                                      (*self.image_size, 3),
                                      dtype=np.uint8)
        image = Image.fromarray(img_array)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


print("Custom Dataset classes defined!")
print()
print("Key points about PyTorch Datasets:")
print("  • Dataset is responsible for loading individual samples")
print("  • It does NOT handle batching - that's the DataLoader's job")
print("  • __getitem__() is called once per sample by the DataLoader")
print("  • Keep __getitem__() efficient - it's called thousands of times!")
print()


################################################################################
# SECTION 2: LOADING THE DATASET
################################################################################

print("=" * 80)
print("SECTION 2: LOADING THE DATASET")
print("=" * 80)
print()

# Define paths
data_dir = Path("/Users/zack/dev/ml-refresher/data/oxford_flowers")
image_dir = data_dir / "jpg"

print(f"Looking for Oxford Flowers dataset at: {image_dir}")
print()

# Define transforms (from Lesson 08)
# These are standard transforms for ImageNet-style models
transform = transforms.Compose([
    transforms.Resize(256),              # Resize shortest edge to 256
    transforms.CenterCrop(224),          # Crop center 224x224
    transforms.ToTensor(),               # Convert to tensor [0, 1]
    transforms.Normalize(                # Normalize with ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Transform pipeline defined:")
print("  1. Resize to 256 pixels (shortest edge)")
print("  2. Center crop to 224x224")
print("  3. Convert to tensor")
print("  4. Normalize with ImageNet statistics")
print()

# Try to load Oxford Flowers dataset
if image_dir.exists() and len(list(image_dir.glob("*.jpg"))) > 0:
    print("✓ Oxford Flowers dataset found!")

    # Get all image paths
    image_paths = sorted(list(image_dir.glob("*.jpg")))
    num_images = len(image_paths)
    print(f"  Found {num_images} images")

    # Load labels from .mat file
    try:
        from scipy.io import loadmat
        labels_file = data_dir / "imagelabels.mat"
        mat_data = loadmat(labels_file)
        labels = mat_data['labels'][0].tolist()  # Labels are 1-indexed (1-102)
        labels = [l - 1 for l in labels]  # Convert to 0-indexed (0-101)

        print(f"  Loaded {len(labels)} labels (102 flower classes)")
        print(f"  Label range: {min(labels)} to {max(labels)}")
    except:
        print("  ! Could not load labels file, generating synthetic labels...")
        labels = [i % 102 for i in range(num_images)]

    # Create the dataset
    dataset = OxfordFlowersDataset(image_dir, labels, transform=transform)

else:
    print("! Oxford Flowers dataset not found.")
    print("  Using synthetic dataset for demonstration...")
    print()

    # Create synthetic dataset
    dataset = SyntheticDataset(
        num_samples=1000,
        image_size=(224, 224),
        num_classes=102,
        transform=transform
    )

print()
print(f"Dataset ready! Total samples: {len(dataset)}")
print()

# Test dataset by loading one sample
print("-" * 60)
print("Testing dataset by loading a single sample...")
sample_image, sample_label = dataset[0]
print(f"  Sample image shape: {sample_image.shape}")
print(f"  Sample image dtype: {sample_image.dtype}")
print(f"  Sample image range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
print(f"  Sample label: {sample_label} (class ID)")
print()


################################################################################
# SECTION 3: DATA SPLITTING (TRAIN/VAL/TEST)
################################################################################

print("=" * 80)
print("SECTION 3: DATA SPLITTING (TRAIN/VAL/TEST)")
print("=" * 80)
print()

print("Before training, we need to split our dataset into:")
print("  • Training set   (70%) - Used to train the model")
print("  • Validation set (15%) - Used to tune hyperparameters")
print("  • Test set       (15%) - Used for final evaluation")
print()
print("Why split?")
print("  • Training set: Model learns patterns from this data")
print("  • Validation set: Used to check if model generalizes (prevents overfitting)")
print("  • Test set: Final unbiased evaluation (model never sees this during training!)")
print()
print("-" * 60)

# Calculate split sizes
total_size = len(dataset)
train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size  # Ensure all samples are used

print(f"Total dataset size: {total_size}")
print(f"  Train size: {train_size} ({train_size/total_size*100:.1f}%)")
print(f"  Val size:   {val_size} ({val_size/total_size*100:.1f}%)")
print(f"  Test size:  {test_size} ({test_size/total_size*100:.1f}%)")
print()

# Use random_split with a generator for reproducibility
print("Using torch.utils.data.random_split() to split dataset...")
print("  Setting generator seed for reproducibility (seed=42)")

# Create a generator with a fixed seed
generator = torch.Generator().manual_seed(42)

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator  # This ensures same split every time!
)

print()
print("✓ Dataset split complete!")
print(f"  Train dataset: {len(train_dataset)} samples")
print(f"  Val dataset:   {len(val_dataset)} samples")
print(f"  Test dataset:  {len(test_dataset)} samples")
print()
print("IMPORTANT: The split is reproducible because we used a seeded generator.")
print("           Running this code again will produce the same split!")
print()


################################################################################
# SECTION 4: CREATING DATALOADERS
################################################################################

print("=" * 80)
print("SECTION 4: CREATING DATALOADERS")
print("=" * 80)
print()

print("The DataLoader wraps a Dataset and provides:")
print("  • Batching: Groups samples into batches")
print("  • Shuffling: Randomizes sample order (important for training!)")
print("  • Parallel loading: Uses multiple workers for faster loading")
print("  • Memory pinning: Speeds up GPU transfer")
print()
print("Key DataLoader parameters:")
print("  • batch_size: Number of samples per batch")
print("  • shuffle: Whether to shuffle data each epoch")
print("  • num_workers: Number of subprocesses for data loading")
print("  • pin_memory: Pin memory for faster GPU transfer")
print("  • drop_last: Drop last incomplete batch if True")
print()
print("-" * 60)

# Create basic DataLoader
batch_size = 32
print(f"Creating DataLoader with batch_size={batch_size}...")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,        # Shuffle training data each epoch
    num_workers=0,       # 0 means load in main process (we'll compare later)
    pin_memory=True,     # Speeds up GPU transfer
    drop_last=False      # Keep last batch even if smaller
)

print(f"✓ Train DataLoader created!")
print(f"  Batch size: {batch_size}")
print(f"  Number of batches per epoch: {len(train_loader)}")
print(f"  Shuffle: True")
print(f"  Num workers: 0 (main process only)")
print()

# Create validation and test loaders (no shuffling for evaluation!)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,       # Don't shuffle validation data
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,       # Don't shuffle test data
    num_workers=0,
    pin_memory=True
)

print(f"✓ Val DataLoader created!")
print(f"  Batch size: {batch_size}")
print(f"  Number of batches: {len(val_loader)}")
print(f"  Shuffle: False (we want consistent evaluation)")
print()

print(f"✓ Test DataLoader created!")
print(f"  Batch size: {batch_size}")
print(f"  Number of batches: {len(test_loader)}")
print(f"  Shuffle: False")
print()


################################################################################
# SECTION 5: ITERATING THROUGH BATCHES
################################################################################

print("=" * 80)
print("SECTION 5: ITERATING THROUGH BATCHES")
print("=" * 80)
print()

print("Let's iterate through the DataLoader to see how batching works...")
print()
print("-" * 60)

# Time the iteration
start_time = time.time()

print("Iterating through first 5 batches of training data:")
print()

for batch_idx, (images, labels) in enumerate(train_loader):
    if batch_idx >= 5:  # Only show first 5 batches
        break

    print(f"Batch {batch_idx + 1}:")
    print(f"  Images shape: {images.shape}")
    print(f"    • Batch size: {images.shape[0]}")
    print(f"    • Channels: {images.shape[1]} (RGB)")
    print(f"    • Height: {images.shape[2]} pixels")
    print(f"    • Width: {images.shape[3]} pixels")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels: {labels[:10].tolist()}")  # Show first 10 labels
    print(f"  Image dtype: {images.dtype}")
    print(f"  Label dtype: {labels.dtype}")
    print(f"  Memory: {images.element_size() * images.nelement() / 1024 / 1024:.2f} MB")
    print()

iteration_time = time.time() - start_time
print(f"Time to iterate through 5 batches: {iteration_time:.4f} seconds")
print()

print("Key observations:")
print("  • Images are batched into shape [batch_size, channels, height, width]")
print("  • Labels are batched into shape [batch_size]")
print("  • DataLoader automatically handles batching - no manual work needed!")
print("  • Each iteration of the for loop gives you one batch")
print()


################################################################################
# SECTION 6: EFFECT OF SHUFFLING
################################################################################

print("=" * 80)
print("SECTION 6: EFFECT OF SHUFFLING")
print("=" * 80)
print()

print("Shuffling is crucial for training neural networks!")
print("It prevents the model from learning the order of data.")
print()
print("Let's compare shuffle=True vs shuffle=False...")
print()
print("-" * 60)

# Create two loaders: one with shuffle, one without
shuffle_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
no_shuffle_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

print("WITH SHUFFLE=TRUE:")
print("  Running two epochs to see if label order changes...")
print()

# First epoch with shuffle
labels_epoch1 = []
for batch_idx, (_, labels) in enumerate(shuffle_loader):
    if batch_idx >= 3:  # First 3 batches
        break
    labels_epoch1.extend(labels[:8].tolist())  # Take first 8 labels from each

print(f"Epoch 1 labels (first 24): {labels_epoch1}")

# Second epoch with shuffle
labels_epoch2 = []
for batch_idx, (_, labels) in enumerate(shuffle_loader):
    if batch_idx >= 3:
        break
    labels_epoch2.extend(labels[:8].tolist())

print(f"Epoch 2 labels (first 24): {labels_epoch2}")
print()
print(f"Are they the same? {labels_epoch1 == labels_epoch2}")
print("  ↳ Labels are in DIFFERENT order each epoch (good for training!)")
print()

print("-" * 60)
print("WITHOUT SHUFFLE (shuffle=False):")
print("  Running two epochs to see if label order stays the same...")
print()

# First epoch without shuffle
labels_epoch1_no_shuffle = []
for batch_idx, (_, labels) in enumerate(no_shuffle_loader):
    if batch_idx >= 3:
        break
    labels_epoch1_no_shuffle.extend(labels[:8].tolist())

print(f"Epoch 1 labels (first 24): {labels_epoch1_no_shuffle}")

# Second epoch without shuffle
labels_epoch2_no_shuffle = []
for batch_idx, (_, labels) in enumerate(no_shuffle_loader):
    if batch_idx >= 3:
        break
    labels_epoch2_no_shuffle.extend(labels[:8].tolist())

print(f"Epoch 2 labels (first 24): {labels_epoch2_no_shuffle}")
print()
print(f"Are they the same? {labels_epoch1_no_shuffle == labels_epoch2_no_shuffle}")
print("  ↳ Labels are in SAME order each epoch (bad for training!)")
print()

print("Summary:")
print("  • Training: ALWAYS use shuffle=True")
print("  • Validation/Test: Use shuffle=False for consistent evaluation")
print()


################################################################################
# SECTION 7: COMPARING BATCH SIZES
################################################################################

print("=" * 80)
print("SECTION 7: COMPARING BATCH SIZES")
print("=" * 80)
print()

print("Batch size is a crucial hyperparameter that affects:")
print("  • Training speed: Larger batches = fewer iterations")
print("  • Memory usage: Larger batches = more GPU memory needed")
print("  • Gradient quality: Smaller batches = noisier gradients")
print("  • Generalization: Often smaller batches generalize better!")
print()
print("Let's compare different batch sizes...")
print()
print("-" * 60)

batch_sizes = [8, 16, 32, 64]
results = []

for bs in batch_sizes:
    print(f"Testing batch_size={bs}...")

    # Create loader
    loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=0
    )

    num_batches = len(loader)

    # Time one epoch
    start = time.time()
    for batch_idx, (images, labels) in enumerate(loader):
        # Simulate some processing
        _ = images.mean()  # Small computation

        if batch_idx >= 20:  # Only process 20 batches for timing
            break

    elapsed = time.time() - start
    time_per_batch = elapsed / min(20, num_batches)

    # Calculate memory usage for one batch
    sample_images, _ = next(iter(loader))
    memory_mb = sample_images.element_size() * sample_images.nelement() / 1024 / 1024

    results.append({
        'batch_size': bs,
        'num_batches': num_batches,
        'time_per_batch': time_per_batch,
        'memory_mb': memory_mb
    })

    print(f"  Batches per epoch: {num_batches}")
    print(f"  Time per batch: {time_per_batch:.4f} seconds")
    print(f"  Memory per batch: {memory_mb:.2f} MB")
    print()

print("-" * 60)
print("COMPARISON SUMMARY:")
print()
print(f"{'Batch Size':<12} {'Batches':<10} {'Time/Batch':<15} {'Memory':<12}")
print("-" * 60)
for r in results:
    print(f"{r['batch_size']:<12} {r['num_batches']:<10} "
          f"{r['time_per_batch']:.4f}s{'':<8} {r['memory_mb']:.2f} MB")
print()

print("Key observations:")
print("  • Larger batches → Fewer batches per epoch")
print("  • Larger batches → More memory usage")
print("  • Larger batches → Often slightly faster per batch (better GPU utilization)")
print("  • Common choices: 16, 32, 64, 128, 256 (powers of 2)")
print()

print("How to choose batch size?")
print("  1. Start with 32 or 64")
print("  2. Increase until you run out of GPU memory")
print("  3. If training is unstable, try smaller batches")
print("  4. Use batch size that's a power of 2 (hardware optimization)")
print()


################################################################################
# SECTION 8: MULTI-WORKER DATA LOADING
################################################################################

print("=" * 80)
print("SECTION 8: MULTI-WORKER DATA LOADING")
print("=" * 80)
print()

print("num_workers controls how many subprocesses are used for data loading.")
print("This can significantly speed up training!")
print()
print("How it works:")
print("  • num_workers=0: Load data in main process (simple but slower)")
print("  • num_workers=2: Use 2 worker processes to load data in parallel")
print("  • num_workers=4: Use 4 worker processes (even faster!)")
print()
print("Benefit: While GPU trains on current batch, workers load the next batch")
print()
print("-" * 60)

print(f"Available CPU cores: {os.cpu_count()}")
print()

# Compare different num_workers settings
# Note: On macOS/Windows, use num_workers=0 to avoid multiprocessing issues
# On Linux, you can safely use num_workers > 0
worker_counts = [0]  # Only test 0 to avoid multiprocessing complications
worker_results = []

print("Note: Testing with num_workers=0 only to avoid multiprocessing issues.")
print("      In production on Linux, you can use num_workers=2-4 for speedup.")
print()

for num_workers in worker_counts:
    print(f"Testing num_workers={num_workers}...")

    # Create loader
    loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Time loading 30 batches
    start = time.time()
    for batch_idx, (images, labels) in enumerate(loader):
        # Simulate processing
        _ = images.mean()

        if batch_idx >= 29:  # Load 30 batches
            break

    elapsed = time.time() - start
    batches_per_second = 30 / elapsed

    worker_results.append({
        'num_workers': num_workers,
        'total_time': elapsed,
        'batches_per_second': batches_per_second
    })

    print(f"  Time for 30 batches: {elapsed:.3f}s")
    print(f"  Batches per second: {batches_per_second:.2f}")
    print()

print("-" * 60)
print("WORKER COMPARISON:")
print()
print(f"{'Workers':<10} {'Total Time':<15} {'Batches/sec':<15}")
print("-" * 60)

for r in worker_results:
    print(f"{r['num_workers']:<10} {r['total_time']:.3f}s{'':<9} "
          f"{r['batches_per_second']:.2f}")
print()

print("Guidelines for num_workers:")
print("  • On macOS/Windows: Use num_workers=0 (multiprocessing has issues)")
print("  • On Linux: Start with num_workers=2 or 4 for speedup")
print("  • Don't use more workers than CPU cores")
print("  • Too many workers can actually slow things down!")
print("  • Typical speedup: 2-3x faster with 4 workers on Linux")
print()

print("IMPORTANT: In practice, the speedup is most noticeable when:")
print("  • Loading from slow storage (HDD vs SSD)")
print("  • Applying heavy data augmentation")
print("  • Working with large images or complex preprocessing")
print()


################################################################################
# SECTION 9: COMPLETE TRAIN/VAL/TEST SETUP PATTERN
################################################################################

print("=" * 80)
print("SECTION 9: COMPLETE TRAIN/VAL/TEST SETUP PATTERN")
print("=" * 80)
print()

print("Here's a production-ready pattern for setting up DataLoaders:")
print()
print("-" * 60)

# Define different transforms for training vs evaluation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),           # Random crop for augmentation
    transforms.RandomHorizontalFlip(),    # Random flip for augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),           # Center crop (deterministic)
    # No random augmentation for evaluation!
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

print("Step 1: Define separate transforms for training and evaluation")
print("  Training transform: Random crop, flip, color jitter (augmentation)")
print("  Eval transform: Center crop only (no randomness)")
print()

# For this demo, we'll use our existing split datasets
# In practice, you'd create new datasets with different transforms
print("Step 2: Create datasets with appropriate transforms")
print("  (In practice, you'd recreate datasets with train_transform/eval_transform)")
print()

print("Step 3: Create DataLoaders with appropriate settings")
print()

# Training loader settings
final_train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,           # Shuffle training data
    num_workers=0,          # Use 0 for macOS/Windows, 2-4 for Linux
    pin_memory=True,        # Pin memory for GPU
    drop_last=True          # Drop last incomplete batch
)

print("Training DataLoader:")
print(f"  batch_size=64 (larger for efficiency)")
print(f"  shuffle=True (randomize order)")
print(f"  num_workers=0 (use 2-4 on Linux for parallel loading)")
print(f"  pin_memory=True (faster GPU transfer)")
print(f"  drop_last=True (consistent batch sizes)")
print(f"  → {len(final_train_loader)} batches per epoch")
print()

# Validation loader settings
final_val_loader = DataLoader(
    val_dataset,
    batch_size=128,         # Can use larger batch for evaluation
    shuffle=False,          # Don't shuffle
    num_workers=0,
    pin_memory=True,
    drop_last=False         # Keep all validation samples
)

print("Validation DataLoader:")
print(f"  batch_size=128 (larger since no gradients)")
print(f"  shuffle=False (consistent evaluation)")
print(f"  num_workers=0 (use 2-4 on Linux)")
print(f"  pin_memory=True")
print(f"  drop_last=False (evaluate all samples)")
print(f"  → {len(final_val_loader)} batches")
print()

# Test loader settings (same as validation)
final_test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False
)

print("Test DataLoader:")
print(f"  batch_size=128")
print(f"  shuffle=False")
print(f"  num_workers=0 (use 2-4 on Linux)")
print(f"  pin_memory=True")
print(f"  drop_last=False")
print(f"  → {len(final_test_loader)} batches")
print()

print("-" * 60)
print("Complete training loop structure:")
print()
print("for epoch in range(num_epochs):")
print("    # Training phase")
print("    model.train()")
print("    for images, labels in final_train_loader:")
print("        # Forward pass, backward pass, optimizer step")
print("        ...")
print()
print("    # Validation phase")
print("    model.eval()")
print("    with torch.no_grad():")
print("        for images, labels in final_val_loader:")
print("            # Evaluate model")
print("            ...")
print()
print("# Final evaluation on test set")
print("model.eval()")
print("with torch.no_grad():")
print("    for images, labels in final_test_loader:")
print("        # Final test evaluation")
print("        ...")
print()


################################################################################
# SECTION 10: CUSTOM COLLATE FUNCTION
################################################################################

print("=" * 80)
print("SECTION 10: CUSTOM COLLATE FUNCTION")
print("=" * 80)
print()

print("Sometimes you need custom batching logic. Examples:")
print("  • Variable-length sequences (text, time series)")
print("  • Images of different sizes")
print("  • Complex data structures")
print()
print("The collate_fn parameter lets you customize how samples are batched.")
print()
print("-" * 60)


def custom_collate_fn(batch):
    """
    Custom collate function that handles variable-size data.

    Args:
        batch: List of (image, label) tuples from dataset

    Returns:
        Batched images and labels (with custom logic)
    """
    # Separate images and labels
    images, labels = zip(*batch)

    # Example: Filter out None values (if dataset can return None)
    images = [img for img in images if img is not None]
    labels = [lbl for lbl in labels if lbl is not None]

    # Stack images into a batch (assumes same size)
    images_batch = torch.stack(images, dim=0)

    # Convert labels to tensor
    labels_batch = torch.tensor(labels)

    # You could add custom logic here:
    # - Pad sequences to same length
    # - Resize images to same size
    # - Add metadata
    # - Apply per-batch augmentation

    return images_batch, labels_batch


# Create loader with custom collate function
custom_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=custom_collate_fn,  # Use custom batching
    num_workers=0
)

print("Custom collate function defined!")
print()
print("Testing custom collate function...")
images, labels = next(iter(custom_loader))
print(f"  Batch shape: {images.shape}")
print(f"  Labels shape: {labels.shape}")
print()

print("Use cases for custom collate_fn:")
print("  1. Text/NLP: Pad sequences to same length, create attention masks")
print("  2. Object detection: Handle variable number of objects per image")
print("  3. Graph data: Batch graphs of different sizes")
print("  4. Multi-modal: Combine images, text, and metadata")
print()

print("Example: Padding sequences for text")
print()
print("def text_collate_fn(batch):")
print("    texts, labels = zip(*batch)")
print("    # Pad all sequences to length of longest sequence")
print("    lengths = [len(text) for text in texts]")
print("    max_length = max(lengths)")
print("    padded = torch.zeros(len(texts), max_length, dtype=torch.long)")
print("    for i, text in enumerate(texts):")
print("        padded[i, :lengths[i]] = text")
print("    return padded, torch.tensor(labels)")
print()


################################################################################
# SECTION 11: DATALOADER BEST PRACTICES & DEBUGGING
################################################################################

print("=" * 80)
print("SECTION 11: DATALOADER BEST PRACTICES & DEBUGGING")
print("=" * 80)
print()

print("BEST PRACTICES:")
print()

print("1. Memory Management:")
print("   • Use pin_memory=True when training on GPU")
print("   • Don't load all data into RAM at once (use Dataset correctly)")
print("   • Clear cache: torch.cuda.empty_cache() if needed")
print()

print("2. Performance:")
print("   • Use num_workers=2-4 for parallel loading")
print("   • Use persistent_workers=True for faster epoch transitions")
print("   • Profile with torch.utils.bottleneck to find slowdowns")
print()

print("3. Shuffling:")
print("   • Always shuffle training data (shuffle=True)")
print("   • Never shuffle validation/test data (shuffle=False)")
print("   • Use generator seed for reproducible splits")
print()

print("4. Batch Sizes:")
print("   • Start with 32 or 64")
print("   • Use powers of 2 (8, 16, 32, 64, 128, 256)")
print("   • Larger batches for evaluation (no gradients stored)")
print("   • Use drop_last=True for training (consistent batch sizes)")
print()

print("5. Data Augmentation:")
print("   • Only augment training data")
print("   • Use deterministic transforms for val/test")
print("   • Don't go overboard - augmentation != better")
print()

print("-" * 60)
print("COMMON ISSUES & SOLUTIONS:")
print()

print("Issue: DataLoader is slow")
print("  → Increase num_workers (try 2, 4)")
print("  → Use pin_memory=True")
print("  → Profile __getitem__() - is it doing too much?")
print("  → Use faster storage (SSD vs HDD)")
print()

print("Issue: Running out of memory")
print("  → Reduce batch_size")
print("  → Use gradient accumulation instead")
print("  → Reduce image resolution")
print("  → Use mixed precision training")
print()

print("Issue: Training is not reproducible")
print("  → Set all random seeds (torch, numpy, random)")
print("  → Use generator seed in random_split()")
print("  → Set torch.backends.cudnn.deterministic=True")
print("  → Note: num_workers > 0 can affect reproducibility")
print()

print("Issue: Multiprocessing errors on Windows")
print("  → Set num_workers=0")
print("  → Put DataLoader code in if __name__ == '__main__':")
print("  → Or use persistent_workers=True")
print()

print("-" * 60)
print("DEBUGGING TIPS:")
print()

print("1. Test with batch_size=1 first")
print("   Quick way to verify dataset is working")
print()

print("2. Print shapes frequently")
print("   for images, labels in loader:")
print("       print(images.shape, labels.shape)")
print("       break  # Just check first batch")
print()

print("3. Visualize batches")
print("   Check that data looks correct and transforms work")
print()

print("4. Time your data loading")
print("   If DataLoader is bottleneck, GPU is waiting for data!")
print()

print("5. Start simple, add complexity gradually")
print("   Get basic DataLoader working before adding:")
print("   • Multiple workers")
print("   • Complex transforms")
print("   • Custom collate functions")
print()


################################################################################
# SECTION 12: SUMMARY & KEY TAKEAWAYS
################################################################################

print("=" * 80)
print("SECTION 12: SUMMARY & KEY TAKEAWAYS")
print("=" * 80)
print()

print("What we learned:")
print()

print("1. DATASET CLASS:")
print("   • Implement __init__(), __len__(), __getitem__()")
print("   • Load data on-the-fly (don't load all into memory)")
print("   • Apply transforms in __getitem__()")
print()

print("2. DATA SPLITTING:")
print("   • Use random_split() with generator seed")
print("   • Standard split: 70% train, 15% val, 15% test")
print("   • Never train on validation or test data!")
print()

print("3. DATALOADER:")
print("   • Handles batching, shuffling, parallel loading")
print("   • Key parameters: batch_size, shuffle, num_workers")
print("   • Training: shuffle=True, drop_last=True")
print("   • Evaluation: shuffle=False, drop_last=False")
print()

print("4. PERFORMANCE:")
print("   • Use num_workers=2-4 for speedup")
print("   • Use pin_memory=True for GPU training")
print("   • Larger batch sizes for evaluation")
print("   • Profile and optimize __getitem__()")
print()

print("5. BEST PRACTICES:")
print("   • Different transforms for train vs eval")
print("   • Set random seeds for reproducibility")
print("   • Start simple, add complexity gradually")
print("   • Monitor data loading speed")
print()

print("-" * 60)
print("TYPICAL DATALOADER SETUP:")
print()
print("# Training")
print("train_loader = DataLoader(")
print("    train_dataset,")
print("    batch_size=64,")
print("    shuffle=True,")
print("    num_workers=2,")
print("    pin_memory=True,")
print("    drop_last=True")
print(")")
print()
print("# Evaluation")
print("val_loader = DataLoader(")
print("    val_dataset,")
print("    batch_size=128,")
print("    shuffle=False,")
print("    num_workers=2,")
print("    pin_memory=True,")
print("    drop_last=False")
print(")")
print()


################################################################################
# PRACTICE PROBLEMS
################################################################################

print("=" * 80)
print("PRACTICE PROBLEMS")
print("=" * 80)
print()

print("1. BASIC CUSTOM DATASET:")
print("   Create a custom Dataset for MNIST-style data (28x28 grayscale images)")
print("   where you generate random images on-the-fly.")
print("   • num_samples=10000")
print("   • 10 classes")
print("   • Images: random values 0-255")
print("   • Apply transforms: ToTensor, Normalize")
print()

print("2. DATA SPLITTING EXPERIMENT:")
print("   Split a dataset of 1000 samples into:")
print("   • 60% train, 20% val, 20% test")
print("   Create DataLoaders and verify the split is reproducible")
print("   (run twice, check same samples in each split)")
print()

print("3. BATCH SIZE COMPARISON:")
print("   Time data loading with batch_sizes [4, 8, 16, 32, 64, 128]")
print("   Plot: batch_size vs time_per_batch")
print("   Calculate: total_time_per_epoch for each")
print()

print("4. NUM_WORKERS SPEEDUP:")
print("   Measure speedup with num_workers [0, 1, 2, 4, 8]")
print("   Plot: num_workers vs batches_per_second")
print("   Find optimal num_workers for your system")
print()

print("5. CUSTOM COLLATE FUNCTION:")
print("   Create a Dataset with variable-length sequences (lists of different sizes)")
print("   Write a collate_fn that:")
print("   • Pads all sequences to max length in batch")
print("   • Returns padded sequences + original lengths")
print("   • Returns batch as (padded_seqs, lengths, labels)")
print()

print("6. AUGMENTATION COMPARISON:")
print("   Create two DataLoaders for same data:")
print("   • One with heavy augmentation")
print("   • One with no augmentation")
print("   Visualize 10 samples from each to see difference")
print()

print("7. MEMORY EFFICIENT LOADING:")
print("   Create a Dataset that simulates large images (e.g., 4096x4096)")
print("   Measure memory usage with different batch sizes")
print("   Find maximum batch_size before running out of memory")
print()

print("8. SHUFFLING VERIFICATION:")
print("   Verify that shuffle=True actually randomizes data")
print("   • Load first batch from 5 consecutive epochs")
print("   • Check that labels are different each epoch")
print("   • Calculate overlap percentage between epochs")
print()

print("9. COMPLETE PIPELINE:")
print("   Build a complete train/val/test pipeline for image classification:")
print("   • Custom Dataset class")
print("   • 70/15/15 split")
print("   • Different transforms for train/eval")
print("   • Proper DataLoaders")
print("   • Training loop skeleton (no actual training)")
print()

print("10. PERFORMANCE PROFILING:")
print("    Profile your DataLoader to find bottlenecks:")
print("    • Time __getitem__() for 100 samples")
print("    • Time data loading vs model forward pass")
print("    • Identify slowest operation")
print("    • Suggest optimization")
print()

print("-" * 60)
print("BONUS CHALLENGE:")
print()
print("Create a DataLoader for a multi-modal dataset:")
print("  • Each sample has: image, text caption, numerical metadata")
print("  • Implement custom __getitem__() to return all three")
print("  • Implement custom collate_fn to batch all modalities")
print("  • Handle variable-length text with padding")
print("  • Test with batch_size=16")
print()

print("=" * 80)
print("END OF LESSON 09: DATALOADERS & BATCHING")
print("=" * 80)
print()

print("Next lesson: Building Neural Networks for FashionMNIST")
print("You're now ready to efficiently load data for training!")
print()
