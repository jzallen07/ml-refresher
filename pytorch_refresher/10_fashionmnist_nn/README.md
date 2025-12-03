# Lesson 10: Building a Neural Network for FashionMNIST

## Overview

This lesson brings everything together. We'll build a complete neural network to classify FashionMNIST images, combining all concepts from previous lessons: tensors, gradients, datasets, transforms, and dataloaders. This is your first real image classification model.

> **Source**: This lesson is based on the "Building a simple neural network for the FashionMNIST dataset" section of [Mastering PyTorch: From Linear Regression to Computer Vision](https://www.iamtk.co/mastering-pytorch-from-linear-regression-to-com) by TK.

## Learning Objectives

By the end of this lesson, you will be able to:

1. Load and preprocess the FashionMNIST dataset
2. Build a feedforward neural network with multiple layers
3. Choose appropriate activation functions (ReLU, Softmax)
4. Implement a complete training loop with validation
5. Track and visualize training metrics
6. Evaluate model performance with accuracy

## Key Concepts

### The FashionMNIST Dataset

FashionMNIST is a dataset of 70,000 grayscale images (28x28 pixels) of clothing items:
- 60,000 training images
- 10,000 test images
- 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

As TK notes: "FashionMNIST has 60,000 training samples and 10,000 test samples. Each sample is a 28x28 grayscale image, and each image is categorized into 10 classes."

### Loading with torchvision

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
```

### Building the Neural Network

```python
class FashionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)
```

### Why ReLU?

ReLU (Rectified Linear Unit) is the most common activation function:
- `ReLU(x) = max(0, x)`
- Introduces non-linearity (allows learning complex patterns)
- Computationally efficient
- Helps avoid vanishing gradients

### The Complete Training Loop

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
        correct = 0
        total = 0
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
```

### Loss Functions

For classification, we use Cross-Entropy Loss:
```python
criterion = nn.CrossEntropyLoss()
```

Cross-entropy combines LogSoftmax and NLLLoss - it measures the difference between predicted probability distribution and true labels.

### Understanding the Output

The model outputs 10 logits (raw scores), one per class:
```python
outputs = model(image)  # Shape: [batch_size, 10]
predicted = torch.argmax(outputs, dim=1)  # Get class with highest score
```

## Code Walkthrough

Run the lesson code to see these concepts in action:

```bash
python lesson.py
```

The code demonstrates:
1. Loading and visualizing FashionMNIST
2. Building a multi-layer neural network
3. Training for multiple epochs
4. Tracking loss and accuracy
5. Evaluating on test set
6. Visualizing predictions vs actual labels

**Note**: This lesson downloads ~30MB of FashionMNIST data.

## Practice Problems

After completing the main lesson, try these exercises:

1. **Deeper Network**: Add more layers (512 → 256 → 128 → 64 → 10). Does performance improve? Does training take longer?

2. **Dropout Regularization**: Add `nn.Dropout(0.2)` after each ReLU. Train again and compare validation accuracy. Does it reduce overfitting?

3. **Learning Rate Experiment**: Try learning rates [0.1, 0.01, 0.001, 0.0001]. Plot training curves for each. Which converges fastest? Which gives best final accuracy?

## Key Takeaways

- FashionMNIST is a standard benchmark for image classification
- `nn.Sequential` chains layers together for simple architectures
- `nn.Flatten` converts images (1, 28, 28) to vectors (784,)
- ReLU activation introduces non-linearity
- CrossEntropyLoss handles multi-class classification
- `model.train()` / `model.eval()` toggle training vs evaluation mode
- `torch.no_grad()` disables gradient computation during evaluation
- Track both loss and accuracy to monitor training progress

## Next Lesson

This completes the core PyTorch fundamentals from TK's article. The extended curriculum continues with:

[Lesson 11: Word Embeddings & nn.Embedding](../11_embeddings/README.md) - Learn how to represent words as dense vectors for NLP.
