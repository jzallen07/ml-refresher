# Regularization

## Interview Questions Covered
- **Q18**: What is the role of dropout in training LLMs?

---

## Q18: Dropout in Training LLMs

### Definition

**Dropout** is a regularization technique that randomly "drops" (sets to zero) a fraction of neurons during training, preventing co-adaptation and reducing overfitting.

### How It Works

```python
# During training (p = 0.1 dropout rate)
# Randomly zero out 10% of activations
# Scale remaining by 1/(1-p) to maintain expected value

x = [1.0, 2.0, 3.0, 4.0, 5.0]
mask = [1, 0, 1, 1, 1]  # Random, 10% are 0
output = x * mask / (1 - p)
# output ≈ [1.11, 0, 3.33, 4.44, 5.55]

# During inference
# No dropout, use all neurons
output = x  # [1.0, 2.0, 3.0, 4.0, 5.0]
```

### Why Dropout Works

1. **Prevents co-adaptation**: Neurons can't rely on specific other neurons
2. **Implicit ensemble**: Training ~2^n different networks (where n = neurons)
3. **Noise injection**: Acts as data augmentation in activation space
4. **Weight regularization**: Similar effect to L2 regularization

### Dropout in Transformers

Modern LLMs apply dropout in several places:

```python
class TransformerBlock:
    def __init__(self, d_model, dropout=0.1):
        # Attention dropout
        self.attention_dropout = nn.Dropout(dropout)

        # Residual dropout (after attention, after FFN)
        self.residual_dropout = nn.Dropout(dropout)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)
```

### Dropout Locations

| Location | What It Does |
|----------|--------------|
| **Embedding dropout** | Regularizes input representations |
| **Attention dropout** | Drops attention weights (prevents attending to same positions) |
| **Residual dropout** | Applied before residual connection |
| **FFN dropout** | After activation in feed-forward network |

### Typical Dropout Rates

| Model Size | Dropout Rate |
|------------|--------------|
| Small (< 1B) | 0.1 - 0.3 |
| Medium (1-10B) | 0.0 - 0.1 |
| Large (> 10B) | 0.0 |

**Key insight**: Large models often use NO dropout because:
- Sufficient parameters to not overfit
- Other regularization (weight decay, data augmentation)
- Dropout hurts training efficiency

### Dropout Variants

#### 1. Standard Dropout
```python
dropout = nn.Dropout(p=0.1)
```

#### 2. DropConnect
Drop weights instead of activations:
```python
# W = W * mask during forward
```

#### 3. Spatial Dropout (for CNNs)
Drop entire feature maps.

#### 4. DropPath (Stochastic Depth)
Drop entire layers/residual blocks:
```python
def forward(self, x):
    if self.training and random.random() < self.drop_prob:
        return x  # Skip this layer entirely
    return x + self.layer(x)
```

### Attention Dropout Details

```python
# In scaled dot-product attention
attn_weights = softmax(Q @ K.T / sqrt(d_k))
attn_weights = dropout(attn_weights)  # Drop some attention connections
output = attn_weights @ V
```

This prevents the model from always attending to the same positions.

### Layer-wise Dropout

Different dropout rates per layer (often increasing):
```python
dropout_rates = [0.0, 0.05, 0.1, 0.1, 0.15, 0.2]  # Increasing by depth
```

Intuition: Early layers learn general features, later layers more task-specific.

---

## Other Regularization Techniques in LLMs

### Weight Decay (L2 Regularization)

```python
optimizer = AdamW(model.parameters(), weight_decay=0.01)
```

- Adds `λ * ||w||²` to loss
- Encourages smaller weights
- AdamW applies it correctly (decoupled from Adam updates)

### Label Smoothing

```python
# Instead of one-hot [0, 0, 1, 0, 0]
# Use smoothed [0.025, 0.025, 0.9, 0.025, 0.025]
loss = CrossEntropyLoss(label_smoothing=0.1)
```

Prevents overconfident predictions.

### Data Augmentation

- **Token masking**: MLM-style masking during fine-tuning
- **Back-translation**: Translate to another language and back
- **Paraphrasing**: Generate variations of training examples

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Not strictly regularization, but prevents exploding gradients.

---

## Interview Tips

1. **Know dropout locations**: Attention, residual, embedding
2. **Scaling insight**: Large models often disable dropout
3. **Training vs inference**: Dropout only during training
4. **Attention dropout**: Prevents attending to same positions
5. **Alternative regularization**: Weight decay, label smoothing

---

## Code Demo

See `regularization_demo.py` for:
- Dropout behavior visualization
- Comparison of dropout rates
- Attention dropout implementation
- Training with vs without dropout

```bash
poetry run python interview_questions/15_regularization/regularization_demo.py
```
