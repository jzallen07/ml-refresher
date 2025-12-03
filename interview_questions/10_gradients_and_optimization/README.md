# Gradients & Optimization

## Interview Questions Covered
- **Q26**: How are gradients computed for embeddings in LLMs?
- **Q27**: What is the Jacobian matrix's role in transformer backpropagation?
- **Q48**: What is a hyperparameter, and why is it important?

---

## Q26: Gradients for Embeddings

### How Embedding Gradients Work

Embedding lookup is a sparse operation—only the looked-up rows receive gradients.

```python
# Forward pass
embedding_table = nn.Embedding(vocab_size=10000, embedding_dim=768)
token_ids = [5, 23, 891]  # Input tokens
vectors = embedding_table(token_ids)  # Shape: [3, 768]

# Backward pass
# Only rows 5, 23, 891 get gradients
# Other 9,997 rows have zero gradient
```

### The Math

```
∂L/∂E = ∂L/∂logits * ∂logits/∂E
```

Where:
- E is the embedding matrix
- The gradient is sparse (most rows zero)
- Only tokens that appeared in the batch get updated

### Sparse Updates

```python
# Conceptually:
for token_id in batch_token_ids:
    embedding_table.weight[token_id] -= lr * gradient[token_id]

# PyTorch handles this efficiently with sparse gradients
```

### Implications

1. **Rare words**: Embeddings for rare tokens update slowly
2. **Efficient computation**: Don't need to compute full matrix gradient
3. **Optimization**: Specialized optimizers (SparseAdam) can help

---

## Q27: Jacobian Matrix in Backpropagation

### Definition

The **Jacobian** is a matrix of all first-order partial derivatives:

```
For f: R^n → R^m

J = [∂f_1/∂x_1  ...  ∂f_1/∂x_n]
    [    ...    ...      ...   ]
    [∂f_m/∂x_1  ...  ∂f_m/∂x_n]
```

### Role in Transformers

For vector-to-vector operations in transformers, we need Jacobians:

```python
# Attention output: R^(seq_len × d) → R^(seq_len × d)
# Jacobian captures how each output element depends on each input

# Softmax: R^n → R^n
# Jacobian of softmax(x)_i with respect to x_j:
#   = softmax(x)_i * (δ_ij - softmax(x)_j)
```

### Vector-Jacobian Products (VJP)

PyTorch doesn't compute full Jacobians—too expensive!

Instead, it computes **vector-Jacobian products**:
```
v^T @ J  (not J itself)
```

This is what `.backward()` computes, propagating gradients efficiently.

### Practical Understanding

```python
# For layer y = f(x):
# Forward: compute y from x
# Backward: given ∂L/∂y, compute ∂L/∂x = (∂L/∂y)^T @ J_f
```

---

## Q48: Hyperparameters

### Definition

**Hyperparameters** are configuration values set BEFORE training, not learned from data.

### Key Hyperparameters in LLMs

| Category | Hyperparameter | Typical Values |
|----------|---------------|----------------|
| **Architecture** | num_layers | 12-96 |
| | hidden_size | 768-12288 |
| | num_heads | 12-96 |
| | vocab_size | 32K-100K |
| **Training** | learning_rate | 1e-4 to 6e-4 |
| | batch_size | 256-2048 |
| | warmup_steps | 1000-10000 |
| | weight_decay | 0.01-0.1 |
| **Regularization** | dropout | 0.0-0.1 |
| | gradient_clip | 1.0 |

### Learning Rate: The Most Important

```
Too high: Training diverges, loss explodes
Too low: Training too slow, may get stuck
Just right: Fast convergence to good minimum
```

### Learning Rate Schedules

```python
# Warmup + Cosine Decay (common for LLMs)
if step < warmup_steps:
    lr = base_lr * step / warmup_steps
else:
    lr = base_lr * cos_decay(step)
```

### Batch Size Tradeoffs

| Larger Batch | Smaller Batch |
|--------------|---------------|
| More stable gradients | Noisier gradients |
| Higher throughput | Lower memory |
| May need higher LR | Implicit regularization |
| Better parallelization | |

### Hyperparameter Tuning Methods

1. **Grid Search**: Try all combinations (expensive)
2. **Random Search**: Sample randomly (often better than grid)
3. **Bayesian Optimization**: Model the objective function
4. **Population-Based Training**: Evolutionary approach

---

## Code Demo

See `gradient_flow_demo.py` for:
- Embedding gradient visualization
- Gradient flow through layers
- Learning rate experiments
- Hyperparameter sensitivity analysis

```bash
python interview_questions/10_gradients_and_optimization/gradient_flow_demo.py
```
