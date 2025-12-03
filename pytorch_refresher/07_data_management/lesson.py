"""
================================================================================
Lesson 7: Data Management - Downloading Datasets and Building Custom PyTorch Datasets
================================================================================

Source: Based on TK's article on building custom PyTorch datasets
Topic: Learn how to download datasets, extract archives, and create custom Dataset classes

Learning Objectives:
1. Download datasets programmatically with progress bars
2. Extract compressed archives (tar.gz files)
3. Build custom PyTorch Dataset classes
4. Load and visualize dataset samples
5. Understand the Dataset interface (__init__, __len__, __getitem__)

Dataset: Oxford Flowers 102
- 102 flower categories
- 8,189 images
- Source: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
================================================================================
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("LESSON 7: Data Management - Custom PyTorch Datasets")
print("=" * 80)
print()

# ================================================================================
# IMPORTS
# ================================================================================
print("Importing required libraries...")

import requests
from tqdm import tqdm
import tarfile
import scipy.io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

print("✓ All libraries imported successfully")
print()

# ================================================================================
# SECTION 1: SETTING UP DATA DIRECTORY
# ================================================================================
print("-" * 80)
print("SECTION 1: Setting Up Data Directory")
print("-" * 80)

# Define the shared data directory (consistent across all lessons)
DATA_DIR = "/Users/zack/dev/ml-refresher/data/oxford_flowers"
print(f"Data directory: {DATA_DIR}")

# Create the directory if it doesn't exist
# exist_ok=True means no error if directory already exists (idempotent operation)
os.makedirs(DATA_DIR, exist_ok=True)
print(f"✓ Data directory created/verified: {DATA_DIR}")
print()

# ================================================================================
# SECTION 2: DOWNLOADING THE IMAGES
# ================================================================================
print("-" * 80)
print("SECTION 2: Downloading the Images Dataset")
print("-" * 80)

# URL for the Oxford Flowers 102 dataset (images)
image_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
tgz_path = os.path.join(DATA_DIR, "102flowers.tgz")

print(f"Image dataset URL: {image_url}")
print(f"Download destination: {tgz_path}")

# Check if the file already exists to avoid re-downloading
if os.path.exists(tgz_path):
    print(f"✓ Image archive already exists, skipping download")
else:
    print("Downloading image archive...")
    try:
        # Use stream=True to download in chunks (memory efficient for large files)
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise exception for bad status codes

        # Get total file size from HTTP headers (for progress bar)
        total_size = int(response.headers.get("content-length", 0))
        print(f"Total size: {total_size / (1024*1024):.2f} MB")

        # Download with progress bar using tqdm
        # iter_content() downloads in chunks (1024 bytes = 1 KB)
        with open(tgz_path, "wb") as file:
            with tqdm(
                total=total_size // 1024,  # Total chunks
                unit="KB",
                desc="Downloading images",
                ncols=80
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    pbar.update(1)

        print(f"✓ Image archive downloaded successfully")

    except requests.RequestException as e:
        print(f"✗ Error downloading image archive: {e}")
        sys.exit(1)

print()

# ================================================================================
# SECTION 3: EXTRACTING THE ARCHIVE
# ================================================================================
print("-" * 80)
print("SECTION 3: Extracting the Image Archive")
print("-" * 80)

# Path where images will be extracted (inside DATA_DIR/jpg/)
jpg_dir = os.path.join(DATA_DIR, "jpg")

# Check if already extracted
if os.path.exists(jpg_dir) and len(os.listdir(jpg_dir)) > 0:
    print(f"✓ Images already extracted to: {jpg_dir}")
    print(f"  Found {len(os.listdir(jpg_dir))} files")
else:
    print(f"Extracting archive to: {DATA_DIR}")
    try:
        # Open and extract tar.gz archive
        # 'r:gz' means read mode with gzip compression
        with tarfile.open(tgz_path, "r:gz") as tar:
            # extractall() extracts all files to the specified directory
            tar.extractall(DATA_DIR)

        print(f"✓ Archive extracted successfully")
        if os.path.exists(jpg_dir):
            print(f"  Extracted {len(os.listdir(jpg_dir))} image files")

    except tarfile.TarError as e:
        print(f"✗ Error extracting archive: {e}")
        sys.exit(1)

print()

# ================================================================================
# SECTION 4: DOWNLOADING THE LABELS
# ================================================================================
print("-" * 80)
print("SECTION 4: Downloading the Labels")
print("-" * 80)

# URL for the labels (MATLAB format)
labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
labels_path = os.path.join(DATA_DIR, "imagelabels.mat")

print(f"Labels URL: {labels_url}")
print(f"Download destination: {labels_path}")

# Check if labels already exist
if os.path.exists(labels_path):
    print(f"✓ Labels file already exists, skipping download")
else:
    print("Downloading labels file...")
    try:
        # Download labels file (smaller, so we can download all at once)
        response = requests.get(labels_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        print(f"Total size: {total_size / 1024:.2f} KB")

        # Download with progress bar
        with open(labels_path, "wb") as file:
            with tqdm(
                total=total_size // 1024,
                unit="KB",
                desc="Downloading labels",
                ncols=80
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    pbar.update(1)

        print(f"✓ Labels file downloaded successfully")

    except requests.RequestException as e:
        print(f"✗ Error downloading labels: {e}")
        sys.exit(1)

print()

# ================================================================================
# SECTION 5: BUILDING THE CUSTOM DATASET CLASS
# ================================================================================
print("-" * 80)
print("SECTION 5: Building the Custom Dataset Class")
print("-" * 80)

print("""
PyTorch Dataset Interface:
- Must inherit from torch.utils.data.Dataset
- Must implement three methods:
  1. __init__(): Initialize dataset, load metadata
  2. __len__(): Return total number of samples
  3. __getitem__(idx): Return the sample at index idx
""")

class OxfordFlowersDataset(Dataset):
    """
    Custom PyTorch Dataset for Oxford Flowers 102.

    The Oxford Flowers 102 dataset contains images of 102 flower categories.
    Images are numbered from image_00001.jpg to image_08189.jpg.
    Labels are stored in a MATLAB .mat file.

    Args:
        root_dir (str): Root directory containing the dataset
        transform (callable, optional): Optional transform to apply to images
    """

    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset by loading metadata and labels.

        This method runs once when you create the dataset object.
        It should load any metadata (file paths, labels) but NOT load images
        (images are loaded on-demand in __getitem__).
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "jpg")
        self.transform = transform

        # Load labels from MATLAB file using scipy
        labels_path = os.path.join(root_dir, "imagelabels.mat")
        labels_mat = scipy.io.loadmat(labels_path)

        # Labels are stored in the "labels" key as a 2D array
        # Shape: (1, 8189) - we take [0] to get 1D array
        # Labels are 1-indexed (1-102), so we subtract 1 for 0-indexing (0-101)
        self.labels = labels_mat["labels"][0] - 1

        print(f"  Initialized OxfordFlowersDataset")
        print(f"  - Root directory: {root_dir}")
        print(f"  - Image directory: {self.img_dir}")
        print(f"  - Number of samples: {len(self.labels)}")
        print(f"  - Label range: {self.labels.min()} to {self.labels.max()}")

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        This is used by DataLoader to know how many samples exist.
        Called when you use len(dataset).
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset at the given index.

        This method is called every time you access dataset[idx].
        It should load the image from disk and return it with its label.

        Args:
            idx (int): Index of the sample to load (0 to len-1)

        Returns:
            tuple: (image, label) where image is a PIL Image and label is an int
        """
        # Construct image filename (images are named image_00001.jpg to image_08189.jpg)
        # :05d means zero-pad to 5 digits (e.g., 1 becomes 00001)
        image_name = f"image_{idx + 1:05d}.jpg"
        image_path = os.path.join(self.img_dir, image_name)

        # Load image using PIL (Pillow)
        # PIL images are in (width, height, channels) format
        image = Image.open(image_path)

        # Get the corresponding label
        label = self.labels[idx]

        # Apply transforms if provided (e.g., resize, normalize, to tensor)
        if self.transform:
            image = self.transform(image)

        return image, label

print("✓ OxfordFlowersDataset class defined")
print()

# ================================================================================
# SECTION 6: USING THE DATASET
# ================================================================================
print("-" * 80)
print("SECTION 6: Using the Dataset")
print("-" * 80)

# Create an instance of our custom dataset
print("Creating dataset instance...")
dataset = OxfordFlowersDataset(DATA_DIR)
print()

# Check the length of the dataset
print(f"Dataset length: {len(dataset)}")
print(f"(This tells us how many samples are in the dataset)")
print()

# Access individual samples
print("Accessing dataset samples:")
print("-" * 40)

# Get the first sample (index 0)
img, label = dataset[0]
print(f"Sample 0:")
print(f"  - Image type: {type(img)}")
print(f"  - Image mode: {img.mode}")  # RGB, L (grayscale), etc.
print(f"  - Image size: {img.size}")  # (width, height)
print(f"  - Label: {label} (class {label + 1} in 1-indexed format)")
print()

# Get a few more samples to see variety
for idx in [10, 100, 1000]:
    img, label = dataset[idx]
    print(f"Sample {idx}:")
    print(f"  - Image size: {img.size}")
    print(f"  - Label: {label}")

print()

# Show label distribution (how many images per class)
print("Label distribution:")
print("-" * 40)
unique_labels, counts = np.unique(dataset.labels, return_counts=True)
print(f"Number of classes: {len(unique_labels)}")
print(f"Images per class (min/max/mean): {counts.min()}/{counts.max()}/{counts.mean():.1f}")
print(f"(Note: This dataset is roughly balanced across classes)")
print()

# ================================================================================
# SECTION 7: VISUALIZING IMAGES
# ================================================================================
print("-" * 80)
print("SECTION 7: Visualizing Images")
print("-" * 80)

print("Creating visualization of sample images...")

# Create a figure with subplots to show multiple images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Oxford Flowers 102 - Sample Images", fontsize=16, fontweight='bold')

# Flatten axes array for easier iteration
axes = axes.flatten()

# Sample some random indices
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(len(dataset), size=6, replace=False)

# Display each sample
for idx, ax in enumerate(axes):
    # Get image and label
    img, label = dataset[sample_indices[idx]]

    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(img)

    # Display image
    ax.imshow(img_array)
    ax.set_title(f"Class: {label} | Size: {img.size[0]}x{img.size[1]}", fontsize=10)
    ax.axis("off")  # Hide axis ticks and labels

plt.tight_layout()

# Save the figure instead of showing it (since this is a script)
viz_path = os.path.join(DATA_DIR, "sample_visualization.png")
plt.savefig(viz_path, dpi=100, bbox_inches='tight')
print(f"✓ Visualization saved to: {viz_path}")
print(f"  (Use plt.show() instead of savefig() to display interactively)")
print()

# Optional: Display a single large image
print("Displaying a single sample image:")
img, label = dataset[0]
print(f"  - Sample index: 0")
print(f"  - Label (class): {label}")
print(f"  - Image dimensions: {img.size[0]}x{img.size[1]}")
print(f"  - Image mode: {img.mode}")

# Create single image visualization
plt.figure(figsize=(8, 6))
plt.title(f"Oxford Flowers Sample - Class {label}", fontsize=14, fontweight='bold')
plt.imshow(np.array(img))
plt.axis("off")

single_viz_path = os.path.join(DATA_DIR, "single_sample.png")
plt.savefig(single_viz_path, dpi=100, bbox_inches='tight')
print(f"✓ Single sample visualization saved to: {single_viz_path}")
print()

# ================================================================================
# SECTION 8: PRACTICE PROBLEMS
# ================================================================================
print("-" * 80)
print("SECTION 8: Practice Problems")
print("-" * 80)

print("""
Practice these exercises to deepen your understanding:

PROBLEM 1: Add Train/Test Split
-------------------------------
Modify the OxfordFlowersDataset class to support train/test/validation splits.
The dataset provides setid.mat with predefined splits.

Hints:
- Download setid.mat from the same URL base
- Load it with scipy.io.loadmat()
- Add a 'split' parameter to __init__ ('train', 'val', or 'test')
- Filter self.labels to only include samples in the chosen split
- Update __getitem__ to use the correct indices

Example usage:
    train_dataset = OxfordFlowersDataset(DATA_DIR, split='train')
    test_dataset = OxfordFlowersDataset(DATA_DIR, split='test')


PROBLEM 2: Add Transforms Support
----------------------------------
Enhance the dataset to work with torchvision transforms.

Hints:
- Import torchvision.transforms as transforms
- Create a transform pipeline:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
- Pass transform to the dataset: dataset = OxfordFlowersDataset(DATA_DIR, transform)
- The transform is already supported in __getitem__!

Test it:
    img, label = dataset[0]
    print(f"Transformed image shape: {img.shape}")  # Should be torch.Size([3, 224, 224])


PROBLEM 3: Create a DataLoader
-------------------------------
Use PyTorch's DataLoader to batch and shuffle the dataset.

Hints:
- Import: from torch.utils.data import DataLoader
- Create DataLoader with batching and shuffling:
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster transfer to GPU
    )
- Iterate through batches:
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        if batch_idx == 0:  # Just show first batch
            break

Expected output:
    Batch 0: images shape torch.Size([32, 3, 224, 224]), labels shape torch.Size([32])


BONUS CHALLENGE: Custom Collate Function
-----------------------------------------
Images in this dataset have different sizes. When batching, this causes issues.
Write a custom collate_fn that handles variable-sized images.

Hints:
- Define: def custom_collate(batch): ...
- Option 1: Resize all images to the same size
- Option 2: Pad images to the largest size in the batch
- Option 3: Return a list instead of a tensor for images
- Pass to DataLoader: DataLoader(dataset, collate_fn=custom_collate)
""")

print()
print("=" * 80)
print("LESSON COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print("✓ Downloaded Oxford Flowers 102 dataset")
print("✓ Extracted images and loaded labels")
print("✓ Built custom PyTorch Dataset class")
print("✓ Accessed and visualized dataset samples")
print("✓ Understood the Dataset interface (__init__, __len__, __getitem__)")
print()
print("Next Steps:")
print("1. Complete the practice problems above")
print("2. Try using torch.utils.data.DataLoader with this dataset")
print("3. Experiment with different transforms (resize, normalize, augmentation)")
print("4. Build a model to classify these flowers!")
print()
print(f"Data location: {DATA_DIR}")
print(f"Visualizations saved in: {DATA_DIR}")
print("=" * 80)
