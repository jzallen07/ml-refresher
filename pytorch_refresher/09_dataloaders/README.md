# Lesson 09: Data Management - DataLoaders & Batching

## Overview

Individual samples are inefficient for training - we need batches. DataLoaders handle batching, shuffling, and parallel data loading. This lesson covers the DataLoader class, which is the bridge between your Dataset and the training loop.

> **Source**: This lesson is based on the "Data Management: datasets, data splitting, and dataloaders" section of [Mastering PyTorch: From Linear Regression to Computer Vision](https://www.iamtk.co/mastering-pytorch-from-linear-regression-to-com) by TK.

## Learning Objectives

By the end of this lesson, you will be able to:

1. Create DataLoaders from Datasets with proper configuration
2. Understand and configure batch size, shuffling, and num_workers
3. Split data into train/validation/test sets using `random_split`
4. Iterate through batches in a training loop
5. Use `pin_memory` for faster GPU transfer
6. Handle edge cases with `drop_last` and custom `collate_fn`

## Key Concepts

### Why DataLoaders?

Training neural networks requires:
- **Batching**: Process multiple samples together for efficiency
- **Shuffling**: Randomize order each epoch to prevent learning order
- **Parallel loading**: Load data in background while GPU trains
- **Memory efficiency**: Don't load entire dataset at once

### Creating a DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,         # Samples per batch
    shuffle=True,          # Randomize order each epoch
    num_workers=4,         # Parallel data loading processes
    pin_memory=True        # Faster GPU transfer
)
```

### Key Parameters

#### batch_size
```python
# Larger = faster training, more memory, less noisy gradients
DataLoader(dataset, batch_size=32)   # Common for images
DataLoader(dataset, batch_size=64)   # If memory allows
DataLoader(dataset, batch_size=128)  # Large GPU
```

#### shuffle
```python
# Training: shuffle=True (prevent memorizing order)
train_loader = DataLoader(train_data, shuffle=True)

# Validation/Test: shuffle=False (reproducible results)
val_loader = DataLoader(val_data, shuffle=False)
```

#### num_workers
```python
# 0 = load in main process (debugging)
# 2-4 = typical for CPU
# 4-8 = typical for fast SSD
DataLoader(dataset, num_workers=4)
```

#### pin_memory
```python
# True for GPU training - faster CPU to GPU transfer
DataLoader(dataset, pin_memory=True)
```

### Data Splitting with random_split

```python
from torch.utils.data import random_split

# Split 8000 samples into train/val/test
train_size = int(0.7 * len(dataset))  # 70%
val_size = int(0.15 * len(dataset))   # 15%
test_size = len(dataset) - train_size - val_size  # 15%

train_data, val_data, test_data = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # Reproducible
)
```

### Iterating Through Batches

```python
for batch_idx, (images, labels) in enumerate(dataloader):
    # images: tensor of shape (batch_size, C, H, W)
    # labels: tensor of shape (batch_size,)

    predictions = model(images)
    loss = criterion(predictions, labels)
    # ... training step
```

### The Training Loop Pattern

```python
for epoch in range(num_epochs):
    # Training
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            # ... compute metrics
```

### Handling Edge Cases

#### drop_last
```python
# If dataset size not divisible by batch_size,
# last batch will be smaller. drop_last=True discards it.
DataLoader(dataset, batch_size=32, drop_last=True)
```

#### collate_fn
```python
# Custom batching logic for variable-size data
def custom_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)  # Stack into batch
    labels = torch.tensor(labels)
    return images, labels

DataLoader(dataset, collate_fn=custom_collate)
```

## Code Walkthrough

Run the lesson code to see these concepts in action:

```bash
python lesson.py
```

The code demonstrates:
1. Creating basic DataLoaders
2. Data splitting with random_split
3. Iterating through batches
4. Effect of different batch sizes
5. Shuffling visualization
6. num_workers performance comparison
7. Complete train/val/test setup

**Note**: This lesson uses the Oxford Flowers dataset with transforms from previous lessons.

## Practice Problems

After completing the main lesson, try these exercises:

1. **Batch Size Experiment**: Create DataLoaders with batch sizes 8, 32, 64, 128. Measure iteration time and memory usage for one epoch.

2. **Stratified Split**: The basic `random_split` doesn't preserve class distribution. Implement stratified splitting using `sklearn.model_selection.train_test_split` with the `stratify` parameter.

3. **Custom Sampler**: Create a `WeightedRandomSampler` to handle class imbalance - oversample minority classes.

## Key Takeaways

- DataLoaders batch, shuffle, and parallelize data loading
- `shuffle=True` for training, `shuffle=False` for validation/test
- `num_workers > 0` enables parallel data loading (faster)
- `pin_memory=True` speeds up GPU transfer
- Use `random_split` for train/val/test splits
- Batch size affects training speed, memory, and gradient quality
- Larger batches = faster epochs but may need learning rate adjustment

## Next Lesson

[Lesson 10: Building a Neural Network for FashionMNIST](../10_fashionmnist_nn/README.md) - Put it all together to train a real image classifier.
