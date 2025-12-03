# Lesson 08: Data Transforms

## Overview

This lesson covers torchvision transforms and data augmentation in PyTorch. You'll learn how to preprocess images for neural networks and apply augmentation techniques to improve model generalization.

> **Source**: This lesson is based on the "Data Management: datasets, data splitting, and dataloaders" section of [Mastering PyTorch: From Linear Regression to Computer Vision](https://www.iamtk.co/mastering-pytorch-from-linear-regression-to-com) by TK.

## Learning Objectives

By the end of this lesson, you will be able to:

1. Understand how transforms preprocess images for neural networks
2. Apply basic transforms: `Resize`, `CenterCrop`, `ToTensor`, `Normalize`
3. Implement data augmentation: `RandomHorizontalFlip`, `RandomRotation`, `RandomResizedCrop`, `RandomAffine`, `ColorJitter`
4. Use `transforms.Compose()` to build preprocessing pipelines
5. Distinguish between training and validation transforms
6. Debug transforms step-by-step

## Key Concepts

### Why Transforms?

Neural networks expect:
- **Consistent input sizes** - All images must be the same dimensions
- **Tensor format** - PyTorch works with tensors, not PIL images
- **Normalized values** - Pixel values scaled appropriately for training

Data augmentation creates variations of training images to:
- Increase effective dataset size
- Reduce overfitting
- Make models more robust to real-world variations

### The Transform Pipeline

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),              # Resize shortest edge to 256
    transforms.CenterCrop(224),          # Crop center 224x224
    transforms.ToTensor(),               # Convert PIL to tensor [0, 1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],      # ImageNet mean
        std=[0.229, 0.224, 0.225]        # ImageNet std
    )
])
```

### Structural Transforms

#### Resize
```python
transforms.Resize(256)        # Resize shortest edge to 256
transforms.Resize((224, 224)) # Resize to exact dimensions
```

#### CenterCrop
```python
transforms.CenterCrop(224)    # Crop 224x224 from center
```

#### ToTensor
```python
transforms.ToTensor()
# Converts PIL Image (H, W, C) with values [0, 255]
# to Tensor (C, H, W) with values [0.0, 1.0]
```

### Normalization

ImageNet-trained models expect specific normalization:

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # RGB means
    std=[0.229, 0.224, 0.225]    # RGB standard deviations
)
```

This shifts and scales each channel so values are roughly in [-2, 2] range, which helps with training stability.

### Data Augmentation Transforms

Augmentation transforms are applied **only during training** to artificially increase dataset diversity:

```python
# Random horizontal flip (50% chance)
transforms.RandomHorizontalFlip(p=0.5)

# Random rotation up to 15 degrees
transforms.RandomRotation(15)

# Randomly crop and resize - very common for training
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))

# Random affine transformations
transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
```

### Training vs Validation Transforms

```python
# Training: includes augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Validation/Test: deterministic, no augmentation
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

### Integrating with Dataset

From Lesson 07, add transform support:

```python
class OxfordFlowersDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        # ... rest of init

    def __getitem__(self, idx):
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label
```

## Running the Lesson

Run the lesson code to see these concepts in action:

```bash
# With poetry (recommended)
poetry run python pytorch_refresher/08_data_transforms/lesson.py

# Or activate poetry shell first
poetry shell
python pytorch_refresher/08_data_transforms/lesson.py
```

The code demonstrates:
1. Individual transform effects with visualizations (12 saved images)
2. Basic transforms: Resize, CenterCrop, ToTensor, Normalize
3. Augmentation transforms: RandomHorizontalFlip, RandomRotation, RandomResizedCrop, RandomAffine, ColorJitter
4. Building Compose pipelines
5. Training vs validation transform patterns
6. Step-by-step debugging of transform pipelines

**Note**: This lesson uses images from Lesson 07's Oxford Flowers dataset.

## Generated Visualizations

The lesson creates 12 visualization files showing:
1. **transform_01_resize.png** - Resize effects
2. **transform_02_center_crop.png** - Center crop demonstration
3. **transform_03_to_tensor.png** - ToTensor channel separation
4. **transform_04_normalize.png** - Normalization before/after
5. **transform_05_random_flip.png** - Random horizontal flip variations
6. **transform_06_random_rotation.png** - Random rotation examples
7. **transform_07_random_resized_crop.png** - Random resized crop diversity
8. **transform_08_random_affine.png** - Random affine transformations
9. **transform_09_color_jitter.png** - Color jitter effects
10. **transform_10_composed_training.png** - Composed training transforms
11. **transform_11_train_vs_val.png** - Training vs validation comparison
12. **transform_12_debugging.png** - Step-by-step debugging

All saved to: `/Users/zack/dev/ml-refresher/data/oxford_flowers/`

## Practice Problems

The lesson includes 8 comprehensive practice problems:

1. **Create Weak Augmentation**: Build a transform pipeline with subtle augmentation suitable for fine-tuning pre-trained models.

2. **Medical Image Augmentation**: Design augmentation for medical images where horizontal flip is not allowed and color changes should be minimal.

3. **Compute Dataset Statistics**: Calculate mean and std for the Oxford Flowers dataset and compare with ImageNet statistics.

4. **Debug a Broken Pipeline**: Find and fix issues in a broken transform pipeline with wrong ordering.

5. **Create Test-Time Augmentation**: Implement TTA with multiple deterministic crops/flips of the same image.

6. **Visualize Transform Effects**: Create a comprehensive visualization showing all augmentation types side-by-side.

7. **Strong Augmentation**: Apply aggressive augmentation and determine if images are still recognizable.

8. **Transform Probability**: Analyze randomness in augmentation by applying transforms 100 times and plotting results.

## Key Takeaways

- **Transform order matters**: Always do PIL operations → `ToTensor()` → `Normalize()`
- **Training needs augmentation**: Random transforms help models generalize
- **Validation needs consistency**: No random transforms for fair evaluation
- `ToTensor()` converts PIL images (H,W,C) to tensors (C,H,W) and scales [0,255] → [0,1]
- `Normalize()` centers data using dataset or ImageNet statistics
- **ImageNet normalization**: Use standard values (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for pre-trained models
- **Custom normalization**: Compute your own statistics when training from scratch
- Use augmentation (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, etc.) only during training
- Validation transforms should be deterministic (`Resize` + `CenterCrop`)
- Always debug transforms step-by-step and visualize intermediate results

## What You'll Learn

1. **How transforms work**: Understanding the pipeline from PIL Image to normalized tensor
2. **Data augmentation**: Techniques to artificially expand your dataset
3. **Best practices**: When to use each transform and how to compose them
4. **Training vs Validation**: Why you need different transforms for each
5. **Debugging**: How to diagnose transform issues step-by-step

## Next Steps

After completing this lesson:
1. Try all the practice problems
2. Experiment with different augmentation strengths
3. Apply transforms to your own datasets
4. Move on to Lesson 09: Data Splitting and DataLoaders

## Dependencies

- torch
- torchvision
- PIL (Pillow)
- matplotlib
- numpy

All included in the project's poetry dependencies.
