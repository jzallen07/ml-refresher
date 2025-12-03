"""
================================================================================
LESSON 08: DATA TRANSFORMS IN PYTORCH
================================================================================

Topics Covered:
1. Understanding torchvision.transforms
2. Basic transforms (Resize, Crop, ToTensor, Normalize)
3. Data augmentation transforms (Flip, Rotate, ColorJitter, etc.)
4. Composing transforms together
5. Training vs Validation transforms
6. Debugging transforms step-by-step

Learning Objectives:
- Understand how transforms preprocess images for neural networks
- Learn the difference between deterministic preprocessing and augmentation
- Master the art of composing transform pipelines
- Debug and visualize what transforms do to your data

================================================================================
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

print("=" * 80)
print("LESSON 08: DATA TRANSFORMS")
print("=" * 80)
print()

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision available: {True}")
print()


################################################################################
# SECTION 1: LOADING A SAMPLE IMAGE
################################################################################

print("=" * 80)
print("SECTION 1: LOADING A SAMPLE IMAGE")
print("=" * 80)
print()

# Define paths
data_dir = Path("/Users/zack/dev/ml-refresher/data/oxford_flowers")
image_dir = data_dir / "jpg"
output_dir = data_dir

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Looking for images in: {image_dir}")
print()

# Try to load an image from the Oxford Flowers dataset
if image_dir.exists() and list(image_dir.glob("*.jpg")):
    # Get a sample image (use the first one)
    sample_image_path = list(image_dir.glob("*.jpg"))[0]
    print(f"✓ Found Oxford Flowers dataset!")
    print(f"  Loading sample image: {sample_image_path.name}")

    # Load the image using PIL
    original_image = Image.open(sample_image_path)

else:
    print("! Oxford Flowers dataset not found. Creating synthetic image...")

    # Create a synthetic colorful image (RGB gradient)
    # This creates a nice gradient from red to blue
    width, height = 500, 500
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Create a radial gradient pattern with colors
    for y in range(height):
        for x in range(width):
            # Calculate distance from center
            center_x, center_y = width // 2, height // 2
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)

            # Create RGB values based on position
            img_array[y, x, 0] = int(255 * (x / width))  # Red increases left to right
            img_array[y, x, 1] = int(255 * (y / height))  # Green increases top to bottom
            img_array[y, x, 2] = int(255 * (1 - dist / max_dist))  # Blue decreases from center

    original_image = Image.fromarray(img_array)
    print("  Created synthetic 500x500 RGB image with gradient pattern")

print()
print(f"Image format: {original_image.format or 'N/A'}")
print(f"Image mode: {original_image.mode}")  # Should be RGB
print(f"Image size: {original_image.size}")  # (width, height)
print()

print("-" * 60)
print("KEY CONCEPT: PIL Images vs Tensors")
print("-" * 60)
print("""
PIL (Python Imaging Library) Images:
  - Size format: (width, height)
  - Pixel access: image[x, y]
  - Value range: 0-255 (uint8)
  - Color order: RGB
  - Used by: Human viewing, file I/O

PyTorch Tensors:
  - Shape format: (Channels, Height, Width) = (C, H, W)
  - Indexing: tensor[c, h, w]
  - Value range: 0.0-1.0 (float32)
  - Color order: RGB
  - Used by: Neural networks, GPU computation

Transforms bridge these two worlds!
""")
print()


################################################################################
# SECTION 2: BASIC TRANSFORMS - RESIZE
################################################################################

print("=" * 80)
print("SECTION 2: BASIC TRANSFORMS - RESIZE")
print("=" * 80)
print()

print("-" * 60)
print("Transform: transforms.Resize()")
print("-" * 60)
print("""
Purpose: Resize images to a specific size
Why it's needed:
  - Neural networks expect fixed input sizes
  - Batch processing requires uniform dimensions
  - Control computational cost

Usage:
  - Resize(size): if size is int, resize shorter edge to that size
  - Resize((height, width)): resize to exact dimensions
""")
print()

# Create resize transform
resize_transform = transforms.Resize((224, 224))  # Standard ImageNet size

print("Applying: transforms.Resize((224, 224))")
print(f"Original size: {original_image.size} (width × height)")

resized_image = resize_transform(original_image)

print(f"Resized size:  {resized_image.size} (width × height)")
print()

# Visualize the resize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(original_image)
axes[0].set_title(f"Original\nSize: {original_image.size[0]}×{original_image.size[1]}")
axes[0].axis('off')

axes[1].imshow(resized_image)
axes[1].set_title(f"Resized\nSize: {resized_image.size[0]}×{resized_image.size[1]}")
axes[1].axis('off')

plt.suptitle("Transform: Resize", fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_01_resize.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()

print("💡 TIP: Different resize options:")
print("   - Resize(256): Resize shortest edge to 256, keep aspect ratio")
print("   - Resize((224, 224)): Exact size, may distort aspect ratio")
print("   - Resize((224, 224), antialias=True): Better quality (newer PyTorch)")
print()


################################################################################
# SECTION 3: BASIC TRANSFORMS - CENTER CROP
################################################################################

print("=" * 80)
print("SECTION 3: BASIC TRANSFORMS - CENTER CROP")
print("=" * 80)
print()

print("-" * 60)
print("Transform: transforms.CenterCrop()")
print("-" * 60)
print("""
Purpose: Crop the center portion of an image
Why it's useful:
  - Focus on the central subject
  - Remove borders/edges
  - Often combined with Resize

Common pattern:
  1. Resize to slightly larger (e.g., 256)
  2. CenterCrop to target size (e.g., 224)
  This gives better quality than direct resize!
""")
print()

# First resize to larger, then center crop
resize_256 = transforms.Resize((256, 256))
center_crop = transforms.CenterCrop((224, 224))

print("Two-step process:")
print("  Step 1: Resize to (256, 256)")
image_256 = resize_256(original_image)
print(f"    → Size: {image_256.size}")

print("  Step 2: CenterCrop to (224, 224)")
cropped_image = center_crop(image_256)
print(f"    → Size: {cropped_image.size}")
print()

# Visualize the crop
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_image)
axes[0].set_title(f"Original\n{original_image.size[0]}×{original_image.size[1]}")
axes[0].axis('off')

axes[1].imshow(image_256)
axes[1].set_title(f"Resized\n{image_256.size[0]}×{image_256.size[1]}")
axes[1].axis('off')

axes[2].imshow(cropped_image)
axes[2].set_title(f"Center Cropped\n{cropped_image.size[0]}×{cropped_image.size[1]}")
axes[2].axis('off')

plt.suptitle("Transform: Resize → Center Crop", fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_02_center_crop.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()


################################################################################
# SECTION 4: BASIC TRANSFORMS - TO TENSOR
################################################################################

print("=" * 80)
print("SECTION 4: BASIC TRANSFORMS - TO TENSOR")
print("=" * 80)
print()

print("-" * 60)
print("Transform: transforms.ToTensor()")
print("-" * 60)
print("""
Purpose: Convert PIL Image to PyTorch Tensor
This is THE MOST IMPORTANT transform!

What it does:
  1. Converts PIL Image (H, W, C) to Tensor (C, H, W)
  2. Converts uint8 [0, 255] to float32 [0.0, 1.0]
  3. Prepares data for neural network input

The conversion formula: tensor = pil_image / 255.0
""")
print()

# Use our resized image for this demo
to_tensor = transforms.ToTensor()

print("Before ToTensor():")
print(f"  Type: {type(resized_image)}")
print(f"  Mode: {resized_image.mode}")
print(f"  Size: {resized_image.size} (W, H)")
print(f"  Pixel value at (100, 100): {resized_image.getpixel((100, 100))}")
print(f"    → This is a tuple of (R, G, B) values in range [0, 255]")
print()

print("Applying: transforms.ToTensor()")
tensor_image = to_tensor(resized_image)

print()
print("After ToTensor():")
print(f"  Type: {type(tensor_image)}")
print(f"  Shape: {tensor_image.shape} (C, H, W)")
print(f"  Dtype: {tensor_image.dtype}")
print(f"  Device: {tensor_image.device}")
print(f"  Min value: {tensor_image.min().item():.4f}")
print(f"  Max value: {tensor_image.max().item():.4f}")
print(f"  Mean value: {tensor_image.mean().item():.4f}")
print()

print("Value at position [channel=0, height=100, width=100]:")
print(f"  {tensor_image[0, 100, 100].item():.4f}")
print()

print("Verifying the conversion:")
# Get the original PIL pixel value
pil_pixel = resized_image.getpixel((100, 100))
print(f"  PIL pixel at (100, 100): R={pil_pixel[0]}, G={pil_pixel[1]}, B={pil_pixel[2]}")

# Calculate expected tensor values
expected_r = pil_pixel[0] / 255.0
expected_g = pil_pixel[1] / 255.0
expected_b = pil_pixel[2] / 255.0

print(f"  Expected tensor values: R={expected_r:.4f}, G={expected_g:.4f}, B={expected_b:.4f}")

# Get actual tensor values
actual_r = tensor_image[0, 100, 100].item()
actual_g = tensor_image[1, 100, 100].item()
actual_b = tensor_image[2, 100, 100].item()

print(f"  Actual tensor values:   R={actual_r:.4f}, G={actual_g:.4f}, B={actual_b:.4f}")
print(f"  ✓ Match: {np.allclose([expected_r, expected_g, expected_b], [actual_r, actual_g, actual_b])}")
print()

# Visualize the tensor channels
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Original image
axes[0, 0].imshow(resized_image)
axes[0, 0].set_title("Original Image (RGB)")
axes[0, 0].axis('off')

# Red channel
axes[0, 1].imshow(tensor_image[0].numpy(), cmap='Reds')
axes[0, 1].set_title(f"Red Channel\nRange: [{tensor_image[0].min():.3f}, {tensor_image[0].max():.3f}]")
axes[0, 1].axis('off')

# Green channel
axes[1, 0].imshow(tensor_image[1].numpy(), cmap='Greens')
axes[1, 0].set_title(f"Green Channel\nRange: [{tensor_image[1].min():.3f}, {tensor_image[1].max():.3f}]")
axes[1, 0].axis('off')

# Blue channel
axes[1, 1].imshow(tensor_image[2].numpy(), cmap='Blues')
axes[1, 1].set_title(f"Blue Channel\nRange: [{tensor_image[2].min():.3f}, {tensor_image[2].max():.3f}]")
axes[1, 1].axis('off')

plt.suptitle("Transform: ToTensor() - Channel Visualization", fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_03_to_tensor.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()


################################################################################
# SECTION 5: BASIC TRANSFORMS - NORMALIZE
################################################################################

print("=" * 80)
print("SECTION 5: BASIC TRANSFORMS - NORMALIZE")
print("=" * 80)
print()

print("-" * 60)
print("Transform: transforms.Normalize()")
print("-" * 60)
print("""
Purpose: Normalize tensor with mean and standard deviation
Why it's crucial:
  - Centers data around zero (easier to train)
  - Scales data to similar ranges (faster convergence)
  - Uses dataset statistics (ImageNet is common)

Formula: output[channel] = (input[channel] - mean[channel]) / std[channel]

Common ImageNet normalization:
  mean = [0.485, 0.456, 0.406]  # RGB means
  std  = [0.229, 0.224, 0.225]  # RGB standard deviations

After normalization:
  - Values typically in range [-2, 2]
  - Mean ≈ 0, Std ≈ 1 for each channel
""")
print()

# ImageNet normalization statistics
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

print(f"Before Normalize:")
print(f"  Shape: {tensor_image.shape}")
print(f"  Per-channel statistics:")
for i, color in enumerate(['Red', 'Green', 'Blue']):
    channel = tensor_image[i]
    print(f"    {color:5s}: mean={channel.mean():.4f}, std={channel.std():.4f}, "
          f"range=[{channel.min():.4f}, {channel.max():.4f}]")
print()

print("Applying: transforms.Normalize(mean=ImageNet_mean, std=ImageNet_std)")
normalized_tensor = normalize(tensor_image.clone())  # Clone to keep original

print()
print(f"After Normalize:")
print(f"  Shape: {normalized_tensor.shape}")
print(f"  Per-channel statistics:")
for i, color in enumerate(['Red', 'Green', 'Blue']):
    channel = normalized_tensor[i]
    print(f"    {color:5s}: mean={channel.mean():.4f}, std={channel.std():.4f}, "
          f"range=[{channel.min():.4f}, {channel.max():.4f}]")
print()

print("Understanding the math:")
print("  For each channel c and pixel (h, w):")
print("    normalized[c,h,w] = (tensor[c,h,w] - mean[c]) / std[c]")
print()
print("  Example for Red channel at pixel (100, 100):")
original_val = tensor_image[0, 100, 100].item()
normalized_val = normalized_tensor[0, 100, 100].item()
expected_val = (original_val - imagenet_mean[0]) / imagenet_std[0]

print(f"    Original value:  {original_val:.4f}")
print(f"    Mean (Red):      {imagenet_mean[0]:.4f}")
print(f"    Std (Red):       {imagenet_std[0]:.4f}")
print(f"    Calculation:     ({original_val:.4f} - {imagenet_mean[0]:.4f}) / {imagenet_std[0]:.4f}")
print(f"    Expected result: {expected_val:.4f}")
print(f"    Actual result:   {normalized_val:.4f}")
print(f"    ✓ Match: {np.isclose(expected_val, normalized_val)}")
print()

# Visualize the normalization effect
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original tensor channels
for i, (color, cmap) in enumerate(zip(['Red', 'Green', 'Blue'], ['Reds', 'Greens', 'Blues'])):
    axes[0, i].imshow(tensor_image[i].numpy(), cmap=cmap, vmin=0, vmax=1)
    axes[0, i].set_title(f"Before: {color} Channel\nRange: [0, 1]")
    axes[0, i].axis('off')

# Normalized tensor channels (need to rescale for visualization)
for i, (color, cmap) in enumerate(zip(['Red', 'Green', 'Blue'], ['Reds', 'Greens', 'Blues'])):
    axes[1, i].imshow(normalized_tensor[i].numpy(), cmap=cmap)
    axes[1, i].set_title(f"After: {color} Channel\n"
                         f"Range: [{normalized_tensor[i].min():.2f}, {normalized_tensor[i].max():.2f}]")
    axes[1, i].axis('off')

plt.suptitle("Transform: Normalize() - Before and After", fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_04_normalize.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()

print("💡 IMPORTANT: When to use ImageNet normalization?")
print("   - Using pre-trained models (ResNet, VGG, etc.): YES")
print("   - Training from scratch on ImageNet: YES")
print("   - Training from scratch on other data: Compute your own statistics!")
print()


################################################################################
# SECTION 6: DATA AUGMENTATION - RANDOM HORIZONTAL FLIP
################################################################################

print("=" * 80)
print("SECTION 6: DATA AUGMENTATION - RANDOM HORIZONTAL FLIP")
print("=" * 80)
print()

print("-" * 60)
print("Transform: transforms.RandomHorizontalFlip()")
print("-" * 60)
print("""
Purpose: Randomly flip images horizontally (left ↔ right)
Why it's useful:
  - Doubles effective dataset size
  - Object orientation shouldn't matter (for most tasks)
  - Cheap and effective augmentation

Parameters:
  - p: Probability of flipping (default=0.5)

⚠️  IMPORTANT: This is RANDOM! Each call may give different results.
""")
print()

random_flip = transforms.RandomHorizontalFlip(p=0.5)

print("Applying RandomHorizontalFlip 5 times to the same image:")
print("(Notice: Some flipped, some not - it's random!)")
print()

# Apply multiple times to show randomness
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
axes[0, 0].imshow(resized_image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Apply 5 times
for idx in range(5):
    row = (idx + 1) // 3
    col = (idx + 1) % 3

    flipped = random_flip(resized_image)
    axes[row, col].imshow(flipped)

    # Check if actually flipped by comparing with original
    is_flipped = not np.array_equal(np.array(flipped), np.array(resized_image))
    axes[row, col].set_title(f"Attempt {idx+1}: {'FLIPPED' if is_flipped else 'NOT flipped'}")
    axes[row, col].axis('off')

plt.suptitle("Transform: RandomHorizontalFlip(p=0.5)", fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_05_random_flip.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()

print("💡 When NOT to use horizontal flip:")
print("   - Text recognition (would flip letters!)")
print("   - Medical images with side labels (L/R matters)")
print("   - Asymmetric objects where orientation matters")
print()


################################################################################
# SECTION 7: DATA AUGMENTATION - RANDOM ROTATION
################################################################################

print("=" * 80)
print("SECTION 7: DATA AUGMENTATION - RANDOM ROTATION")
print("=" * 80)
print()

print("-" * 60)
print("Transform: transforms.RandomRotation()")
print("-" * 60)
print("""
Purpose: Randomly rotate images by a random angle
Why it's useful:
  - Object orientation varies in real world
  - Makes model rotation-invariant
  - Good for objects that appear at different angles

Parameters:
  - degrees: Range of degrees (int or tuple)
    - degrees=30 means rotate between -30 and +30 degrees
    - degrees=(0, 180) means rotate between 0 and 180 degrees
""")
print()

random_rotation = transforms.RandomRotation(degrees=30)

print("Applying RandomRotation(degrees=30) 5 times:")
print("(Will rotate between -30° and +30° randomly)")
print()

# Set seed for reproducibility in this demo
torch.manual_seed(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
axes[0, 0].imshow(resized_image)
axes[0, 0].set_title("Original Image\n(No rotation)")
axes[0, 0].axis('off')

# Apply 5 times
for idx in range(5):
    row = (idx + 1) // 3
    col = (idx + 1) % 3

    rotated = random_rotation(resized_image)
    axes[row, col].imshow(rotated)
    axes[row, col].set_title(f"Random Rotation #{idx+1}")
    axes[row, col].axis('off')

plt.suptitle("Transform: RandomRotation(degrees=30)", fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_06_random_rotation.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()

print("💡 TIP: Fill parameter")
print("   - RandomRotation(degrees=30, fill=255) for white background")
print("   - RandomRotation(degrees=30, fill=0) for black background")
print("   - Default fills with black (0)")
print()


################################################################################
# SECTION 8: DATA AUGMENTATION - RANDOM RESIZED CROP
################################################################################

print("=" * 80)
print("SECTION 8: DATA AUGMENTATION - RANDOM RESIZED CROP")
print("=" * 80)
print()

print("-" * 60)
print("Transform: transforms.RandomResizedCrop()")
print("-" * 60)
print("""
Purpose: Crop a random portion, then resize to target size
Why it's powerful:
  - Creates scale variation (zoom in/out effect)
  - Creates positional variation
  - One of THE MOST EFFECTIVE augmentations!

Parameters:
  - size: Target size after crop and resize
  - scale: Range of crop size (proportion of original)
    - scale=(0.08, 1.0) means crop 8% to 100% of original
  - ratio: Aspect ratio range of crop
    - ratio=(3/4, 4/3) allows some distortion
""")
print()

random_crop = transforms.RandomResizedCrop(
    size=(224, 224),
    scale=(0.5, 1.0),  # Crop 50% to 100% of image
    ratio=(0.9, 1.1)   # Nearly square crops
)

print("Applying RandomResizedCrop 6 times:")
print("  size=(224, 224)")
print("  scale=(0.5, 1.0)  # Crop 50%-100% of image")
print("  ratio=(0.9, 1.1)  # Aspect ratio variation")
print()

torch.manual_seed(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx in range(6):
    row = idx // 3
    col = idx % 3

    cropped = random_crop(original_image)
    axes[row, col].imshow(cropped)
    axes[row, col].set_title(f"Random Crop #{idx+1}\nDifferent position & scale")
    axes[row, col].axis('off')

plt.suptitle("Transform: RandomResizedCrop() - Different Crops Each Time",
             fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_07_random_resized_crop.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()

print("💡 This transform is why neural networks learn to focus on parts of objects!")
print("   The network sees the same object at different scales and positions.")
print()


################################################################################
# SECTION 9: DATA AUGMENTATION - RANDOM AFFINE
################################################################################

print("=" * 80)
print("SECTION 9: DATA AUGMENTATION - RANDOM AFFINE")
print("=" * 80)
print()

print("-" * 60)
print("Transform: transforms.RandomAffine()")
print("-" * 60)
print("""
Purpose: Apply random affine transformations
Affine transformations include:
  - Rotation
  - Translation (shifting)
  - Scale (zoom)
  - Shear (slanting)

Parameters:
  - degrees: Rotation range
  - translate: Translation as fraction of image (e.g., 0.1 = 10%)
  - scale: Scale factor range
  - shear: Shear angle range

This is like a "swiss army knife" of geometric augmentations!
""")
print()

random_affine = transforms.RandomAffine(
    degrees=15,                    # Rotate ±15 degrees
    translate=(0.1, 0.1),          # Translate up to 10% in each direction
    scale=(0.9, 1.1),              # Scale 90%-110%
    shear=10                        # Shear ±10 degrees
)

print("Applying RandomAffine with multiple transformations:")
print("  degrees=15        # Rotation ±15°")
print("  translate=(0.1, 0.1)  # Shift up to 10%")
print("  scale=(0.9, 1.1)      # Zoom 90%-110%")
print("  shear=10              # Shear ±10°")
print()

torch.manual_seed(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original
axes[0, 0].imshow(resized_image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Apply 5 times
for idx in range(5):
    row = (idx + 1) // 3
    col = (idx + 1) % 3

    transformed = random_affine(resized_image)
    axes[row, col].imshow(transformed)
    axes[row, col].set_title(f"Affine Transform #{idx+1}\n(rotate+translate+scale+shear)")
    axes[row, col].axis('off')

plt.suptitle("Transform: RandomAffine() - Combined Geometric Transforms",
             fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_08_random_affine.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()


################################################################################
# SECTION 10: DATA AUGMENTATION - COLOR JITTER
################################################################################

print("=" * 80)
print("SECTION 10: DATA AUGMENTATION - COLOR JITTER")
print("=" * 80)
print()

print("-" * 60)
print("Transform: transforms.ColorJitter()")
print("-" * 60)
print("""
Purpose: Randomly change brightness, contrast, saturation, and hue
Why it's important:
  - Images have different lighting conditions
  - Camera settings vary
  - Makes model robust to color variations

Parameters (all values between 0 and 1):
  - brightness: How much to jitter brightness
    - 0.2 means brightness between 80% and 120%
  - contrast: How much to jitter contrast
  - saturation: How much to jitter saturation (color intensity)
  - hue: How much to jitter hue (color shift)
    - 0.1 means hue shift ±10% (±36° on color wheel)
""")
print()

color_jitter = transforms.ColorJitter(
    brightness=0.3,  # ±30% brightness
    contrast=0.3,    # ±30% contrast
    saturation=0.3,  # ±30% saturation
    hue=0.1          # ±10% hue (±36° on 360° color wheel)
)

print("Applying ColorJitter 6 times:")
print("  brightness=0.3  # ±30%")
print("  contrast=0.3    # ±30%")
print("  saturation=0.3  # ±30%")
print("  hue=0.1         # ±36° on color wheel")
print()

torch.manual_seed(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx in range(6):
    row = idx // 3
    col = idx % 3

    jittered = color_jitter(resized_image)
    axes[row, col].imshow(jittered)

    if idx == 0:
        axes[row, col].set_title("Original Image")
    else:
        axes[row, col].set_title(f"Color Jitter #{idx}\n(random brightness/contrast/sat/hue)")
    axes[row, col].axis('off')

# Use original image for first slot
axes[0, 0].imshow(resized_image)
axes[0, 0].set_title("Original Image\n(No jitter)")

plt.suptitle("Transform: ColorJitter() - Color/Brightness Variations",
             fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_09_color_jitter.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()

print("💡 ColorJitter makes your model robust to:")
print("   - Different lighting conditions (indoor/outdoor)")
print("   - Camera settings and quality")
print("   - Day/night, shadows, etc.")
print()


################################################################################
# SECTION 11: COMPOSING TRANSFORMS
################################################################################

print("=" * 80)
print("SECTION 11: COMPOSING TRANSFORMS WITH transforms.Compose()")
print("=" * 80)
print()

print("-" * 60)
print("Transform: transforms.Compose()")
print("-" * 60)
print("""
Purpose: Chain multiple transforms together in a pipeline
This is how you build a complete preprocessing pipeline!

Transforms are applied IN ORDER:
  1. First transform gets the original image
  2. Second transform gets output of first
  3. And so on...

ORDER MATTERS! Always:
  1. Do PIL operations first (Resize, Crop, ColorJitter, etc.)
  2. ToTensor() comes after all PIL operations
  3. Normalize() comes after ToTensor()
""")
print()

print("=" * 60)
print("Example 1: Basic Preprocessing Pipeline")
print("=" * 60)
print()

basic_transform = transforms.Compose([
    transforms.Resize((256, 256)),           # 1. Resize
    transforms.CenterCrop((224, 224)),        # 2. Crop
    transforms.ToTensor(),                    # 3. PIL → Tensor
    transforms.Normalize(                     # 4. Normalize
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Pipeline steps:")
print("  1. Resize((256, 256))")
print("  2. CenterCrop((224, 224))")
print("  3. ToTensor()")
print("  4. Normalize(mean=ImageNet, std=ImageNet)")
print()

print("Applying composed transform...")
processed = basic_transform(original_image)

print()
print(f"Result:")
print(f"  Type: {type(processed)}")
print(f"  Shape: {processed.shape}")
print(f"  Dtype: {processed.dtype}")
print(f"  Range: [{processed.min():.3f}, {processed.max():.3f}]")
print(f"  Mean: {processed.mean():.3f}")
print()

print("=" * 60)
print("Example 2: Training Transform with Augmentation")
print("=" * 60)
print()

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),    # Random crop & scale
    transforms.RandomHorizontalFlip(p=0.5),      # 50% chance flip
    transforms.ColorJitter(                       # Color variations
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomRotation(degrees=15),        # Small rotations
    transforms.ToTensor(),                        # PIL → Tensor
    transforms.Normalize(                         # Normalize
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Training pipeline (with augmentation):")
print("  1. RandomResizedCrop((224, 224))")
print("  2. RandomHorizontalFlip(p=0.5)")
print("  3. ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)")
print("  4. RandomRotation(degrees=15)")
print("  5. ToTensor()")
print("  6. Normalize(mean=ImageNet, std=ImageNet)")
print()

# Apply the same transform 6 times to show variation
print("Applying training transform 6 times to the same image:")
print("(Notice: Each result is different due to random augmentations!)")
print()

torch.manual_seed(42)

results = []
for i in range(6):
    augmented = train_transform(original_image)
    results.append(augmented)

print(f"✓ Generated 6 different augmented versions")
print()

# Visualize the augmented results
# Need to denormalize for visualization
def denormalize(tensor, mean, std):
    """Denormalize a tensor image."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, augmented in enumerate(results):
    row = idx // 3
    col = idx % 3

    # Denormalize for visualization
    img_denorm = denormalize(
        augmented,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Clamp to [0, 1] range
    img_denorm = torch.clamp(img_denorm, 0, 1)

    # Convert to numpy and transpose to (H, W, C)
    img_np = img_denorm.permute(1, 2, 0).numpy()

    axes[row, col].imshow(img_np)
    axes[row, col].set_title(f"Augmentation #{idx+1}\n(Different each time!)")
    axes[row, col].axis('off')

plt.suptitle("Training Transform: Same Image → 6 Different Augmentations",
             fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_10_composed_training.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()


################################################################################
# SECTION 12: TRAINING VS VALIDATION TRANSFORMS
################################################################################

print("=" * 80)
print("SECTION 12: TRAINING VS VALIDATION TRANSFORMS")
print("=" * 80)
print()

print("-" * 60)
print("KEY CONCEPT: Different Transforms for Training and Validation")
print("-" * 60)
print("""
TRAINING transforms:
  - Use random augmentations
  - Create variations of the data
  - Help model generalize
  - Each epoch sees different variations!

VALIDATION/TEST transforms:
  - Use deterministic preprocessing only
  - NO random augmentations
  - Consistent results for evaluation
  - Fair comparison across models

Why this matters:
  - Training: Want model to learn from varied data
  - Validation: Want consistent evaluation metrics
  - Using augmentation on validation would give inconsistent results!
""")
print()

print("=" * 60)
print("TRAINING TRANSFORM")
print("=" * 60)

train_transform = transforms.Compose([
    # Augmentations (random)
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=10),
    # Preprocessing (deterministic)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("""
train_transform = transforms.Compose([
    # === AUGMENTATION (Random) ===
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=10),

    # === PREPROCESSING (Deterministic) ===
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
""")

print("=" * 60)
print("VALIDATION TRANSFORM")
print("=" * 60)

val_transform = transforms.Compose([
    # NO augmentations! Only preprocessing
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("""
val_transform = transforms.Compose([
    # === NO AUGMENTATION - Only preprocessing ===
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
""")

print("=" * 60)
print("Comparison: Same Image, Different Transforms")
print("=" * 60)
print()

torch.manual_seed(42)

# Apply training transform 3 times
train_results = []
for i in range(3):
    train_results.append(train_transform(original_image))

# Apply validation transform 3 times
val_results = []
for i in range(3):
    val_results.append(val_transform(original_image))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Training results (different each time)
for idx in range(3):
    img_denorm = denormalize(
        train_results[idx],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_denorm = torch.clamp(img_denorm, 0, 1)
    img_np = img_denorm.permute(1, 2, 0).numpy()

    axes[0, idx].imshow(img_np)
    axes[0, idx].set_title(f"Training Transform #{idx+1}\n(Different!)")
    axes[0, idx].axis('off')

# Validation results (same each time)
for idx in range(3):
    img_denorm = denormalize(
        val_results[idx],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_denorm = torch.clamp(img_denorm, 0, 1)
    img_np = img_denorm.permute(1, 2, 0).numpy()

    axes[1, idx].imshow(img_np)
    axes[1, idx].set_title(f"Validation Transform #{idx+1}\n(Identical!)")
    axes[1, idx].axis('off')

plt.suptitle("Training (Random) vs Validation (Deterministic) Transforms",
             fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_11_train_vs_val.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()

print("💡 KEY TAKEAWAY:")
print("   Training: Apply transform each epoch → different augmentations")
print("   Validation: Apply once → consistent evaluation")
print()


################################################################################
# SECTION 13: DEBUGGING TRANSFORMS
################################################################################

print("=" * 80)
print("SECTION 13: DEBUGGING TRANSFORMS STEP-BY-STEP")
print("=" * 80)
print()

print("-" * 60)
print("How to Debug Your Transform Pipeline")
print("-" * 60)
print("""
When transforms don't work as expected:
  1. Apply each transform individually
  2. Print shape, dtype, and value ranges after each step
  3. Visualize intermediate results
  4. Check for common mistakes

Common mistakes:
  - Normalize before ToTensor (wrong order!)
  - Wrong mean/std values
  - Forgetting to convert back to PIL for visualization
  - Using training transforms on validation data
""")
print()

print("=" * 60)
print("Debug Example: Step-by-Step Pipeline")
print("=" * 60)
print()

# Start with original image
print("STEP 0: Original Image")
print(f"  Type: {type(original_image)}")
print(f"  Mode: {original_image.mode}")
print(f"  Size: {original_image.size}")
print()

# Step 1: Resize
resize_t = transforms.Resize((256, 256))
img_step1 = resize_t(original_image)
print("STEP 1: After Resize((256, 256))")
print(f"  Type: {type(img_step1)}")
print(f"  Size: {img_step1.size}")
print(f"  ✓ Still PIL Image")
print()

# Step 2: Center Crop
crop_t = transforms.CenterCrop((224, 224))
img_step2 = crop_t(img_step1)
print("STEP 2: After CenterCrop((224, 224))")
print(f"  Type: {type(img_step2)}")
print(f"  Size: {img_step2.size}")
print(f"  ✓ Still PIL Image")
print()

# Step 3: ToTensor
to_tensor_t = transforms.ToTensor()
img_step3 = to_tensor_t(img_step2)
print("STEP 3: After ToTensor()")
print(f"  Type: {type(img_step3)}")
print(f"  Shape: {img_step3.shape}")
print(f"  Dtype: {img_step3.dtype}")
print(f"  Range: [{img_step3.min():.4f}, {img_step3.max():.4f}]")
print(f"  Mean: {img_step3.mean():.4f}")
print(f"  ✓ Now a Tensor! Shape is (C, H, W)")
print()

# Step 4: Normalize
normalize_t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_step4 = normalize_t(img_step3)
print("STEP 4: After Normalize()")
print(f"  Type: {type(img_step4)}")
print(f"  Shape: {img_step4.shape}")
print(f"  Dtype: {img_step4.dtype}")
print(f"  Range: [{img_step4.min():.4f}, {img_step4.max():.4f}]")
print(f"  Mean: {img_step4.mean():.4f}")
print(f"  ✓ Values now centered around 0")
print()

print("=" * 60)
print("Visualizing Each Step")
print("=" * 60)
print()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Step 0: Original
axes[0, 0].imshow(original_image)
axes[0, 0].set_title(f"Step 0: Original\nPIL Image {original_image.size}")
axes[0, 0].axis('off')

# Step 1: Resized
axes[0, 1].imshow(img_step1)
axes[0, 1].set_title(f"Step 1: Resized\nPIL Image {img_step1.size}")
axes[0, 1].axis('off')

# Step 2: Cropped
axes[0, 2].imshow(img_step2)
axes[0, 2].set_title(f"Step 2: Cropped\nPIL Image {img_step2.size}")
axes[0, 2].axis('off')

# Step 3: ToTensor
img_step3_np = img_step3.permute(1, 2, 0).numpy()
axes[1, 0].imshow(img_step3_np)
axes[1, 0].set_title(f"Step 3: ToTensor\nTensor {img_step3.shape}\nRange: [0, 1]")
axes[1, 0].axis('off')

# Step 4: Normalized (need to denormalize for visualization)
img_step4_denorm = denormalize(img_step4, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_step4_denorm = torch.clamp(img_step4_denorm, 0, 1)
img_step4_np = img_step4_denorm.permute(1, 2, 0).numpy()
axes[1, 1].imshow(img_step4_np)
axes[1, 1].set_title(f"Step 4: Normalized\nTensor {img_step4.shape}\nRange: ≈[-2, 2]")
axes[1, 1].axis('off')

# Hide last subplot
axes[1, 2].axis('off')

plt.suptitle("Debugging Transforms: Step-by-Step Visualization",
             fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = output_dir / "transform_12_debugging.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved visualization: {output_path}")
print()

print("=" * 60)
print("Common Issues and Solutions")
print("=" * 60)
print("""
Issue 1: "Normalize() expects a tensor"
  ❌ WRONG order:
     Compose([Normalize(), ToTensor()])
  ✓ CORRECT order:
     Compose([ToTensor(), Normalize()])

Issue 2: "Images look weird after normalization"
  → This is normal! Normalized images have negative values
  → Always denormalize before visualizing

Issue 3: "Getting different results on validation set"
  → Check: Are you using random transforms on val data?
  → Solution: Use deterministic transforms for validation

Issue 4: "Model isn't learning"
  → Check: Are you normalizing with correct mean/std?
  → Check: Is ToTensor() converting to [0, 1] range?
  → Use debugging to verify each step!
""")
print()


################################################################################
# SECTION 14: PRACTICAL TIPS AND BEST PRACTICES
################################################################################

print("=" * 80)
print("SECTION 14: PRACTICAL TIPS AND BEST PRACTICES")
print("=" * 80)
print()

print("-" * 60)
print("Best Practices for Data Transforms")
print("-" * 60)
print()

print("1. ALWAYS follow this order:")
print("""
   ✓ CORRECT:
     transforms.Compose([
         # 1. PIL transformations first
         transforms.Resize(...),
         transforms.RandomCrop(...),
         transforms.ColorJitter(...),

         # 2. ToTensor (PIL → Tensor)
         transforms.ToTensor(),

         # 3. Normalize (Tensor → Normalized Tensor)
         transforms.Normalize(...)
     ])
""")
print()

print("2. Standard preprocessing recipe:")
print("""
   For ImageNet-pretrained models:

   Training:
     - RandomResizedCrop(224)
     - RandomHorizontalFlip(p=0.5)
     - ToTensor()
     - Normalize(mean=ImageNet, std=ImageNet)

   Validation:
     - Resize(256)
     - CenterCrop(224)
     - ToTensor()
     - Normalize(mean=ImageNet, std=ImageNet)
""")
print()

print("3. Computing your own normalization statistics:")
print("""
   If training from scratch on your own dataset:

   from torch.utils.data import DataLoader

   # Load data with ToTensor() only
   dataset = YourDataset(transform=transforms.ToTensor())
   loader = DataLoader(dataset, batch_size=32)

   # Compute mean and std
   mean = 0.0
   std = 0.0
   total = 0

   for images, _ in loader:
       batch_samples = images.size(0)
       images = images.view(batch_samples, images.size(1), -1)
       mean += images.mean(2).sum(0)
       std += images.std(2).sum(0)
       total += batch_samples

   mean /= total
   std /= total

   print(f"Mean: {mean}")
   print(f"Std: {std}")
""")
print()

print("4. When to use which augmentation:")
print("""
   Natural images (objects, animals, scenes):
     ✓ RandomResizedCrop
     ✓ RandomHorizontalFlip
     ✓ ColorJitter
     ✓ RandomRotation (small angles)

   Medical images:
     ✓ RandomRotation
     ✓ RandomAffine
     ✗ ColorJitter (may change diagnostic features!)
     ✗ HorizontalFlip (if L/R matters)

   Text/Documents:
     ✗ HorizontalFlip (would flip text!)
     ✗ RandomRotation (unless small angles)
     ✓ ColorJitter (for different paper/lighting)
""")
print()

print("5. How much augmentation is too much?")
print("""
   Start conservative:
     - brightness=0.1, contrast=0.1, saturation=0.1
     - rotation=5-10 degrees
     - RandomResizedCrop scale=(0.8, 1.0)

   Gradually increase if:
     - Training loss is much lower than validation loss
     - Model overfits quickly

   Decrease if:
     - Training loss stays high
     - Model can't learn
     - Augmentations destroy important features
""")
print()


################################################################################
# SECTION 15: SUMMARY
################################################################################

print("=" * 80)
print("SECTION 15: SUMMARY")
print("=" * 80)
print()

print("🎯 What we learned:")
print()

print("1. Basic Transforms:")
print("   • Resize() - Make images uniform size")
print("   • CenterCrop() - Crop center portion")
print("   • ToTensor() - PIL Image → PyTorch Tensor")
print("   • Normalize() - Standardize pixel values")
print()

print("2. Data Augmentation:")
print("   • RandomHorizontalFlip() - Mirror images")
print("   • RandomRotation() - Rotate by random angles")
print("   • RandomResizedCrop() - Random zoom and position")
print("   • RandomAffine() - Combined geometric transforms")
print("   • ColorJitter() - Vary colors and brightness")
print()

print("3. Key Concepts:")
print("   • Compose() - Chain transforms together")
print("   • Training transforms - Use augmentation")
print("   • Validation transforms - Deterministic only")
print("   • Always: PIL ops → ToTensor() → Normalize()")
print()

print("4. Debugging:")
print("   • Apply transforms step-by-step")
print("   • Print shapes and value ranges")
print("   • Visualize intermediate results")
print("   • Check transform order!")
print()

print("📊 Files created:")
for i in range(1, 13):
    filename = f"transform_{i:02d}_*.png"
    files = list(output_dir.glob(f"transform_{i:02d}_*.png"))
    if files:
        print(f"   ✓ {files[0].name}")
print()


################################################################################
# PRACTICE PROBLEMS
################################################################################

print("=" * 80)
print("PRACTICE PROBLEMS")
print("=" * 80)
print()

print("""
Practice Problem 1: Create Weak Augmentation
────────────────────────────────────────────
Create a transform pipeline with "weak" augmentation suitable for fine-tuning:
  - RandomResizedCrop(224, scale=(0.9, 1.0))
  - RandomHorizontalFlip(p=0.3)
  - ColorJitter(brightness=0.05, contrast=0.05)
  - ToTensor()
  - Normalize with ImageNet stats

Apply it to 5 images and visualize. Are the augmentations subtle?


Practice Problem 2: Medical Image Augmentation
───────────────────────────────────────────────
Create augmentation for medical images where:
  - Horizontal flip is NOT allowed (L/R matters)
  - Rotation up to ±20 degrees is allowed
  - Color changes should be minimal
  - Slight translation/shearing is allowed

Design an appropriate transform pipeline.


Practice Problem 3: Compute Dataset Statistics
───────────────────────────────────────────────
Write code to compute mean and std for the Oxford Flowers dataset:
  1. Load all images with ToTensor() only
  2. Compute per-channel mean and std
  3. Create a Normalize() transform with these values
  4. Compare with ImageNet statistics

Hint: See Best Practices section for the code template!


Practice Problem 4: Debug a Broken Pipeline
────────────────────────────────────────────
This pipeline is BROKEN. Find and fix the issues:

broken_transform = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomResizedCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])

Hint: Apply each transform individually and see where it fails!


Practice Problem 5: Create Test-Time Augmentation
──────────────────────────────────────────────────
Test-Time Augmentation (TTA) applies multiple transforms at inference:
  1. Original image (center crop)
  2. Flipped version
  3. 4 corner crops + center crop
  4. Average predictions

Create 5 different deterministic crops/flips of the same image.
(Useful for better test accuracy!)


Practice Problem 6: Visualize Transform Effects
────────────────────────────────────────────────
Create a comprehensive visualization showing:
  - Original image
  - All 5 augmentation types we learned (side-by-side)
  - Before/after comparison for each

Save as one large figure with appropriate titles.


Practice Problem 7: Strong Augmentation
────────────────────────────────────────
Create "strong" augmentation for robust training:
  - RandomResizedCrop(224, scale=(0.5, 1.0))
  - RandomHorizontalFlip(p=0.5)
  - RandomRotation(degrees=30)
  - ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
  - RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2))

Apply to an image. Is it still recognizable? That's the balance!


Practice Problem 8: Transform Probability
──────────────────────────────────────────
Apply RandomHorizontalFlip(p=0.5) to the same image 100 times.
  - Count how many times it flipped
  - Is it close to 50%?
  - Plot a histogram of "flipped" vs "not flipped"

This demonstrates the randomness in augmentation!
""")

print()
print("=" * 80)
print("END OF LESSON 08: DATA TRANSFORMS")
print("=" * 80)
print()

print("🎉 Congratulations! You've mastered PyTorch data transforms!")
print()
print("Next steps:")
print("  • Try the practice problems")
print("  • Experiment with different augmentation strengths")
print("  • Apply transforms to your own datasets")
print("  • Move on to Lesson 09: Data Splitting and DataLoaders")
print()
print(f"All visualizations saved to: {output_dir}")
print()
