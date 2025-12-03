# PyTorch Refresher - Product Requirements Document

## Overview

This document outlines the lesson structure for the PyTorch Refresher module. The core curriculum (Lessons 01-10) is based on the article "Mastering PyTorch: From Linear Regression to Computer Vision" by TK. Lessons 11-18 extend the curriculum toward understanding and implementing multi-head attention and transformer architectures.

## Source Material

- **Primary Reference (Lessons 01-10)**: "Mastering PyTorch: From Linear Regression to Computer Vision" (docs/Mastering PyTorch_ From Linear Regression to Computer Vision.pdf)
- **Author**: TK
- **Extended Curriculum (Lessons 11-18)**: Supplementary material building toward transformer architectures

---

## Lesson Structure

Each lesson will be a subdirectory under `pytorch_refresher/` containing:
- `README.md` - Concept explanation, learning objectives, and code walkthrough
- `lesson.py` - Highly commented, executable Python code with print statements demonstrating concepts and outputs

### Lesson File Format

Each `lesson.py` will follow this structure:
- Extensive inline comments explaining each concept
- Print statements showing intermediate outputs and results
- Executable as a standalone script: `python lesson.py`
- Up to 3 practice problems at the end of each lesson

---

## Core Curriculum (Lessons 01-10)

### Lesson 01: Tensors - The Data Building Blocks
**Directory**: `pytorch_refresher/01_tensors/`

**Learning Objectives**:
- Understand what tensors are and their role in PyTorch
- Create tensors from scalars, vectors, and matrices
- Explore tensor properties: `dtype`, `ndimension()`, `size()`, `shape`, `item()`
- Convert between NumPy arrays, Pandas DataFrames, and PyTorch tensors

**Code Examples from Article**:
- Creating scalar, vector, and matrix tensors
- Checking dimensions and shapes
- Extracting values with `item()`
- NumPy/Pandas interoperability (`from_numpy`, `numpy()`)

**Practice Problems**:
1. Create a 3D tensor and explore its properties
2. Convert a pandas DataFrame to tensor and back
3. Create tensors of different dtypes and compare memory usage

---

### Lesson 02: Tensor Operations - Reshaping
**Directory**: `pytorch_refresher/02_tensor_reshaping/`

**Learning Objectives**:
- Master reshaping operations for data manipulation
- Understand when and why reshaping is needed in ML pipelines

**Code Examples from Article**:
- `unsqueeze()` - adding dimensions
- `squeeze()` - removing dimensions
- `view()` - flexible reshaping with row/column parameters
- Using `-1` placeholder for dynamic sizing

**Practice Problems**:
1. Reshape a batch of images from (N, H, W, C) to (N, C, H, W)
2. Use unsqueeze to add a batch dimension to a single sample
3. Flatten a multi-dimensional tensor and reshape it back

---

### Lesson 03: Tensor Operations - Indexing & Slicing
**Directory**: `pytorch_refresher/03_tensor_indexing_slicing/`

**Learning Objectives**:
- Access tensor elements using indexing
- Extract tensor subsets using slicing
- Combine row and column slicing operations

**Code Examples from Article**:
- Single element indexing: `tensor[1, 2]`, `tensor[1][2]`
- Negative indexing: `tensor[-1]`
- Slicing operations: `tensor[1][2:]`, `tensor[:2, 2:]`

**Practice Problems**:
1. Extract every other row from a matrix
2. Get the diagonal elements of a square matrix
3. Implement a sliding window extraction using slicing

---

### Lesson 04: Tensor Operations - Math & Boolean Operations
**Directory**: `pytorch_refresher/04_tensor_math_operations/`

**Learning Objectives**:
- Perform statistical operations on tensors
- Execute mathematical operations essential for ML
- Understand matrix operations for neural networks

**Code Examples from Article**:
- Statistical: `min()`, `max()`, `mean()`, `std()`
- Element-wise: addition, multiplication
- Linear algebra: `torch.dot()` (dot product), `torch.mm()` (matrix multiplication)

**Practice Problems**:
1. Normalize a tensor to have mean=0 and std=1
2. Implement a simple cosine similarity function using tensor ops
3. Verify matrix multiplication dimensions with mismatched shapes

---

### Lesson 05: Tensor Operations - Derivatives & Gradients
**Directory**: `pytorch_refresher/05_tensor_gradients/`

**Learning Objectives**:
- Understand autograd and automatic differentiation
- Use `requires_grad` for gradient tracking
- Compute gradients with `backward()`
- Access gradients via `.grad` attribute

**Code Examples from Article**:
- Creating tensors with `requires_grad=True`
- Forward computation: `y = x ** 2`
- Backpropagation: `y.backward()`
- Gradient access: `x.grad`

**Practice Problems**:
1. Compute gradients for a multi-variable function
2. Demonstrate gradient accumulation and when to zero gradients
3. Use `torch.no_grad()` context and explain when it's needed

---

### Lesson 06: Building & Training a Linear Regression Model
**Directory**: `pytorch_refresher/06_linear_regression/`

**Learning Objectives**:
- Build models using `nn.Linear` and `nn.Module`
- Understand the training loop: predict → loss → backprop → update
- Configure loss functions and optimizers
- Implement a complete training pipeline

**Code Examples from Article**:
- Simple `nn.Linear` model
- Custom `LinearRegression` class with `nn.Module`
- Training loop with MSELoss and SGD optimizer
- Distance/time regression example

**Practice Problems**:
1. Train linear regression on a different dataset (e.g., height/weight)
2. Compare SGD vs Adam optimizer convergence
3. Plot the loss curve over epochs

---

### Lesson 07: Data Management - Downloading & Custom Datasets
**Directory**: `pytorch_refresher/07_data_management/`

**Learning Objectives**:
- Download and extract datasets programmatically
- Build custom `Dataset` classes
- Implement `__init__`, `__len__`, `__getitem__` methods
- Handle image data with lazy loading

**Code Examples from Article**:
- Downloading Oxford Flowers dataset with `requests`
- Extracting `.tgz` files with `tarfile`
- `OxfordFlowersDataset` custom class implementation
- Loading and displaying images with PIL/matplotlib

**Practice Problems**:
1. Add a transform parameter to the custom dataset
2. Implement `__repr__` for better dataset inspection
3. Create a subset dataset that only includes certain classes

---

### Lesson 08: Data Management - Transforms & Augmentation
**Directory**: `pytorch_refresher/08_data_transforms/`

**Learning Objectives**:
- Apply preprocessing transforms for standardization
- Implement data augmentation for training diversity
- Compose multiple transforms into pipelines

**Code Examples from Article**:
- Preprocessing: `Resize`, `CenterCrop`, `ToTensor`, `Normalize`
- Augmentation: `RandomHorizontalFlip`, `RandomRotation`, `RandomResizedCrop`, `RandomAffine`
- Using `transforms.Compose()`
- Step-by-step transform debugging

**Practice Problems**:
1. Visualize the same image with different augmentations
2. Create separate transform pipelines for train vs. validation
3. Implement a custom transform class

---

### Lesson 09: Data Management - Splitting & DataLoaders
**Directory**: `pytorch_refresher/09_data_splitting_loaders/`

**Learning Objectives**:
- Split datasets into train/validation/test sets
- Create DataLoaders for batching and shuffling
- Understand when to shuffle data

**Code Examples from Article**:
- Using `random_split()` for 70/15/15 split
- Creating `DataLoader` with batch sizes
- Shuffling considerations for train vs. validation/test
- Iterating through DataLoader batches

**Practice Problems**:
1. Implement stratified splitting to maintain class balance
2. Experiment with different batch sizes and observe memory usage
3. Use `num_workers` for parallel data loading

---

### Lesson 10: Building & Training a Neural Network on FashionMNIST
**Directory**: `pytorch_refresher/10_fashionmnist_nn/`

**Learning Objectives**:
- Download and prepare the FashionMNIST dataset
- Build a multi-layer neural network classifier
- Implement training and evaluation functions
- Track accuracy metrics during training

**Code Examples from Article**:
- FashionMNIST dataset loading with `datasets.FashionMNIST`
- `FashionMNISTModel` with `Flatten`, `Linear`, `ReLU`, `LogSoftmax`
- `NLLLoss` and `Adam` optimizer setup
- `train()` and `test()` function implementations
- Accuracy calculation

**Practice Problems**:
1. Add a third hidden layer and compare performance
2. Implement early stopping based on validation loss
3. Visualize misclassified examples

---

## Extended Curriculum: Path to Multi-Head Attention (Lessons 11-18)

### Lesson 11: Sequence Data & Embeddings
**Directory**: `pytorch_refresher/11_embeddings/`

**Learning Objectives**:
- Understand why sequences need different treatment than fixed-size inputs
- Use `nn.Embedding` for learnable lookup tables
- Handle variable-length sequences with padding and packing
- Recognize positional information challenges in sequences

**Key Concepts**:
- Word/token embeddings vs one-hot encoding
- `nn.Embedding(num_embeddings, embedding_dim)`
- Padding sequences to uniform length
- `pack_padded_sequence` / `pad_packed_sequence`

**Practice Problems**:
1. Create an embedding layer and visualize embedding vectors
2. Implement a simple vocabulary-to-embedding pipeline
3. Compare memory usage of one-hot vs embeddings for large vocabularies

---

### Lesson 12: Recurrent Neural Networks (RNNs & LSTMs)
**Directory**: `pytorch_refresher/12_rnns_lstms/`

**Learning Objectives**:
- Understand sequential processing and hidden states
- Build RNN and LSTM models in PyTorch
- Recognize limitations: vanishing gradients, sequential bottleneck
- Motivate the need for attention mechanisms

**Key Concepts**:
- `nn.RNN`, `nn.LSTM`, `nn.GRU`
- Hidden state propagation
- Bidirectional RNNs
- Why sequential processing limits parallelization

**Practice Problems**:
1. Build a character-level RNN that predicts the next character
2. Compare training speed of RNN vs LSTM on the same task
3. Visualize hidden states across a sequence

---

### Lesson 13: Normalization Techniques - Batch Norm vs Layer Norm
**Directory**: `pytorch_refresher/13_normalization/`

**Learning Objectives**:
- Understand why normalization accelerates training
- Implement and compare BatchNorm and LayerNorm
- Recognize which axes each method normalizes over
- Explain why transformers use LayerNorm (not BatchNorm)

**Key Concepts**:
- Internal covariate shift and training instability
- `nn.BatchNorm1d/2d` - normalizes across batch dimension
- `nn.LayerNorm` - normalizes across feature dimension
- Batch dimension dependency problem:
  - BatchNorm needs running statistics (train vs eval mode)
  - Breaks down with batch_size=1 or variable sequence lengths
  - Sequence models: each position has different statistics
- LayerNorm advantages for transformers:
  - No batch dependency
  - Works identically at train and inference
  - Each token normalized independently

**Code Examples**:
```python
# BatchNorm: normalize across batch (N), per feature
# Input shape: (N, C) or (N, C, H, W)
# Computes mean/var over N for each C
bn = nn.BatchNorm1d(num_features=512)

# LayerNorm: normalize across features, per sample
# Input shape: (N, seq_len, d_model)
# Computes mean/var over d_model for each (N, seq_len) position
ln = nn.LayerNorm(normalized_shape=512)
```

**Practice Problems**:
1. Visualize what dimensions BatchNorm vs LayerNorm normalize over
2. Show BatchNorm's train/eval mode difference; show LayerNorm is identical
3. Demonstrate BatchNorm instability with batch_size=1

---

### Lesson 14: Attention Mechanism (Bahdanau/Additive)
**Directory**: `pytorch_refresher/14_attention_basics/`

**Learning Objectives**:
- Understand attention as a weighted sum over inputs
- Implement basic additive attention from scratch
- Visualize attention weights
- See how attention solves the "bottleneck" problem in seq2seq

**Key Concepts**:
- Query, Key, Value intuition (informal introduction)
- Alignment scores and softmax normalization
- Context vector computation
- Attention weight visualization
- Historical context: Bahdanau et al. (2014) with RNNs

**Practice Problems**:
1. Implement additive attention from scratch
2. Visualize attention weights on a simple sequence task
3. Compare seq2seq performance with and without attention

---

### Lesson 15: Scaled Dot-Product Attention
**Directory**: `pytorch_refresher/15_scaled_dot_product_attention/`

**Learning Objectives**:
- Implement scaled dot-product attention from scratch
- Understand the scaling factor (√d_k) and why it matters
- Apply attention masks for causal/padding scenarios
- Compare computational efficiency to additive attention

**Key Concepts**:
- `Attention(Q, K, V) = softmax(QK^T / √d_k) V`
- Why scale? Dot products grow with dimension, pushing softmax to extremes
- Matrix multiplication for parallel computation (no sequential dependency)
- Causal masks (for autoregressive models)
- Padding masks (for variable-length sequences)

**Code Examples**:
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V), attention_weights
```

**Practice Problems**:
1. Implement scaled dot-product attention from scratch
2. Create and apply a causal mask for autoregressive attention
3. Compare attention output with and without scaling

---

### Lesson 16: Multi-Head Attention
**Directory**: `pytorch_refresher/16_multi_head_attention/`

**Learning Objectives**:
- Implement multi-head attention from scratch
- Understand why multiple heads help (different representation subspaces)
- Project Q, K, V with learned linear layers
- Concatenate and project head outputs

**Key Concepts**:
- Single attention head captures one type of relationship
- Multiple heads capture different relationship types in parallel
- Head splitting: reshape (batch, seq, d_model) → (batch, heads, seq, d_k)
- `nn.Linear` for Q, K, V projections (W_q, W_k, W_v)
- Output projection after concatenation (W_o)
- Comparison with `nn.MultiheadAttention`

**Code Examples**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        # Project and reshape to (batch, heads, seq, d_k)
        # Apply scaled dot-product attention per head
        # Concatenate and project output
        ...
```

**Practice Problems**:
1. Implement multi-head attention from scratch
2. Visualize attention patterns from different heads
3. Compare your implementation output with `nn.MultiheadAttention`

---

### Lesson 17: Transformer Building Blocks
**Directory**: `pytorch_refresher/17_transformer_blocks/`

**Learning Objectives**:
- Implement a full transformer encoder block
- Understand residual connections and layer normalization placement
- Build position-wise feed-forward networks
- Stack multiple blocks

**Key Concepts**:
- Encoder block structure:
  1. Multi-head self-attention
  2. Add & Norm (residual + LayerNorm)
  3. Feed-forward network (expand then contract)
  4. Add & Norm
- Pre-norm vs post-norm architectures
- Feed-forward expansion: d_model → d_ff → d_model (typically d_ff = 4 * d_model)
- Residual connections for gradient flow
- Stacking N identical blocks

**Code Examples**:
```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
```

**Practice Problems**:
1. Implement a full transformer encoder block
2. Compare pre-norm vs post-norm placement
3. Stack multiple blocks and train on a simple classification task

---

### Lesson 18: Positional Encoding
**Directory**: `pytorch_refresher/18_positional_encoding/`

**Learning Objectives**:
- Understand why position information is needed (attention is permutation-invariant)
- Implement sinusoidal positional encoding
- Compare with learned positional embeddings
- Overview of relative position encodings

**Key Concepts**:
- Attention treats input as a set, not a sequence
- Sinusoidal formula:
  - `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
  - `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
- Why sinusoidal? Can extrapolate to longer sequences than seen in training
- Adding vs concatenating position information
- Learned positional embeddings (BERT-style)
- Relative position encodings (brief overview: RoPE, ALiBi)

**Code Examples**:
```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

**Practice Problems**:
1. Implement sinusoidal positional encoding from scratch
2. Visualize the positional encoding patterns
3. Compare model performance with sinusoidal vs learned positions

---

## Appendix Lessons

### Appendix A: Building & Training a CNN for Image Classification
**Directory**: `pytorch_refresher/appendix_a_cnn/`

**Note**: This lesson covers CNNs from the original article. It's placed in the appendix as it follows a computer vision track orthogonal to the attention/transformer path.

**Learning Objectives**:
- Build a Convolutional Neural Network from scratch
- Understand Conv2d, MaxPool2d, and fully connected layers
- Apply dropout for regularization
- Train a binary image classifier

**Code Examples from Article**:
- Downloading horse-or-human dataset
- Using `ImageFolder` for dataset creation
- `HorsesHumansCNN` model architecture:
  - 3 Conv2d layers with ReLU and MaxPool2d
  - Fully connected layers with Dropout
  - Sigmoid output for binary classification
- BCELoss and Adam optimizer
- Complete training loop with validation

**Practice Problems**:
1. Add batch normalization layers and compare training
2. Implement learning rate scheduling
3. Save and load model checkpoints

---

### Appendix B: GPU Training & CUDA Memory Management
**Directory**: `pytorch_refresher/appendix_b_gpu_cuda/`

**Note**: This lesson covers GPU/CUDA topics. It's placed in the appendix as it's a logistical/operational concern applicable across all model types.

**Learning Objectives**:
- Move models and data to GPU devices
- Understand CUDA out-of-memory (OOM) errors
- Resolve memory issues through hyperparameter adjustment
- Implement gradient accumulation for large effective batch sizes

**Topics Covered**:
- Device detection: `torch.cuda.is_available()`, `torch.backends.mps.is_available()`
- Moving tensors/models: `.to(device)`, `.cuda()`, `.cpu()`
- Common CUDA OOM causes and solutions
- Batch size reduction strategies
- Gradient accumulation implementation
- `torch.cuda.empty_cache()` usage
- Memory profiling with `torch.cuda.memory_allocated()`
- Mixed precision training basics with `torch.amp`

**Code Examples**:
```python
# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gradient accumulation pattern
accumulation_steps = 4
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs.to(device))
    loss = criterion(outputs, labels.to(device))
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Practice Problems**:
1. Benchmark training speed on CPU vs GPU (if available)
2. Implement gradient accumulation to simulate a larger batch size
3. Profile memory usage before and after `empty_cache()`

---

## Directory Structure

```
pytorch_refresher/
├── __init__.py
│
├── # Core Curriculum (Article-based)
├── 01_tensors/
├── 02_tensor_reshaping/
├── 03_tensor_indexing_slicing/
├── 04_tensor_math_operations/
├── 05_tensor_gradients/
├── 06_linear_regression/
├── 07_data_management/
├── 08_data_transforms/
├── 09_data_splitting_loaders/
├── 10_fashionmnist_nn/
│
├── # Extended Curriculum (Path to Transformers)
├── 11_embeddings/
├── 12_rnns_lstms/
├── 13_normalization/
├── 14_attention_basics/
├── 15_scaled_dot_product_attention/
├── 16_multi_head_attention/
├── 17_transformer_blocks/
├── 18_positional_encoding/
│
├── # Appendix (Orthogonal Topics)
├── appendix_a_cnn/
└── appendix_b_gpu_cuda/
```

---

## Data Directory Structure

All datasets will be downloaded to a shared `data/` directory at the project root:

```
data/
├── fashion_mnist/          # FashionMNIST dataset (Lesson 10)
├── oxford_flowers/         # Oxford Flowers 102 dataset (Lesson 07)
│   ├── jpg/
│   └── imagelabels.mat
├── horse_or_human/        # Horse or Human dataset (Appendix A)
│   ├── training/
│   └── validation/
└── text/                  # Text datasets for sequence lessons (11-18)
```

---

## Dependencies

The following packages will be required (to be added to `pyproject.toml`):

```
torch
torchvision
numpy
pandas
scipy
matplotlib
Pillow
tqdm
requests
```

---

## Implementation Notes

1. **Code Attribution**: Lessons 01-10 are sourced from the referenced article. Lessons 11-18 and appendix lessons are supplementary material.

2. **Progressive Complexity**: Lessons build upon each other. The main track (01-18) provides a path from tensors to transformers.

3. **Runnable Code**: Each `lesson.py` is executable as a standalone script with extensive comments and print statements showing outputs.

4. **Practice Problems**: Each lesson includes up to 3 practice problems for hands-on reinforcement.

5. **CPU-First Approach**: All main lessons use CPU for simplicity. Appendix B covers GPU/CUDA topics.

6. **Shared Data Directory**: All datasets download to `data/` at project root for reuse across lessons.

7. **README Structure**: Each README should include:
   - Learning objectives
   - Concept explanation
   - Code walkthrough with explanations
   - Key takeaways
   - Practice problem descriptions
   - Link to next lesson

---

## Milestone Progression

| Phase | Lessons | Focus |
|-------|---------|-------|
| **Foundation** | 01-05 | Tensor fundamentals and operations |
| **First Model** | 06 | Linear regression training pipeline |
| **Data Handling** | 07-09 | Dataset management and preprocessing |
| **Neural Networks** | 10 | Multi-layer perceptron classifier |
| **Sequences** | 11-12 | Embeddings and RNNs |
| **Normalization** | 13 | BatchNorm vs LayerNorm |
| **Attention** | 14-16 | Additive → Scaled Dot-Product → Multi-Head |
| **Transformers** | 17-18 | Encoder blocks and positional encoding |

---

## Success Criteria

- Each lesson can be completed independently (after prerequisites)
- Code runs without errors on Python 3.12 with specified dependencies
- Concepts align with the source article's explanations (Lessons 01-10)
- Progressive difficulty leads from tensors to implementing multi-head attention
- Practice problems reinforce learning with hands-on exercises
- By Lesson 16, learner can implement multi-head attention from scratch
- By Lesson 17, learner can build a complete transformer encoder block
