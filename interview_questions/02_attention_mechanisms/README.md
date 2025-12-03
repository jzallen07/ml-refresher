# Attention Mechanisms

## Interview Questions Covered
- **Q2**: How does the attention mechanism function in transformer models?
- **Q22**: What is multi-head attention, and how does it enhance LLMs?
- **Q23**: How is the softmax function applied in attention mechanisms?
- **Q24**: How does the dot product contribute to self-attention?
- **Q32**: How are attention scores calculated in transformers?

---

## Q2: How does the attention mechanism function in transformer models?

### Answer

The **attention mechanism** allows the model to focus on relevant parts of the input when processing each token. It computes weighted combinations of all positions based on their relevance.

### The Core Idea

For each token, attention answers: "Which other tokens should I pay attention to?"

Example: "The cat sat on the mat because **it** was tired"
- When processing "it", attention should focus heavily on "cat"

### Key Components

1. **Query (Q)**: "What am I looking for?"
2. **Key (K)**: "What do I contain?"
3. **Value (V)**: "What information do I provide?"

### The Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- `QK^T`: Computes similarity between queries and keys
- `√d_k`: Scaling factor to prevent large dot products
- `softmax`: Normalizes scores to probabilities
- `× V`: Weighted sum of values

---

## Q22: What is multi-head attention?

### Answer

**Multi-head attention** runs multiple attention operations in parallel, each focusing on different aspects of the input.

### Why Multiple Heads?

Single attention can only focus on one type of relationship. Multiple heads can simultaneously capture:
- **Head 1**: Syntactic relationships (subject-verb)
- **Head 2**: Semantic relationships (synonyms)
- **Head 3**: Positional patterns (adjacent words)
- **Head 4**: Long-range dependencies

### Implementation

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(Q × W_Q^i, K × W_K^i, V × W_V^i)
```

### Typical Configuration

- GPT-3: 96 heads, d_model=12288, d_head=128
- BERT-base: 12 heads, d_model=768, d_head=64

---

## Q23: How is softmax applied in attention?

### Answer

Softmax converts raw attention scores into a **probability distribution** that sums to 1.

### The Formula

```
softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
```

### Purpose in Attention

1. **Normalization**: Ensures attention weights sum to 1
2. **Differentiability**: Enables gradient-based learning
3. **Sparsity encouragement**: Large scores dominate after softmax

### Example

```python
scores = [2.0, 1.0, 0.1]
# After softmax: [0.659, 0.242, 0.099]
# The highest score (2.0) gets ~66% of the attention
```

---

## Q24: How does the dot product contribute to self-attention?

### Answer

The **dot product** between Query and Key vectors measures **similarity**:

```
Score = (Q · K) / √d_k
```

### Why Dot Product?

1. **Efficiency**: Matrix multiplication is highly optimized on GPUs
2. **Semantic similarity**: Similar vectors have higher dot products
3. **Learnable**: Q and K projections are learned parameters

### The Scaling Factor (√d_k)

Without scaling, dot products grow large with dimension size, pushing softmax into regions with tiny gradients. Dividing by √d_k keeps values in a reasonable range.

```python
# Without scaling (d_k = 512):
# dot products might be ~20-30, softmax becomes nearly one-hot

# With scaling:
# dot products are ~1-2, softmax remains smooth
```

### Complexity

- **Time**: O(n² × d) for sequence length n
- **Space**: O(n²) for attention matrix
- This quadratic cost limits context length in standard transformers

---

## Q32: How are attention scores calculated?

### Answer

Complete attention calculation:

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]

    # Step 1: Compute similarity scores
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, heads, seq, seq]

    # Step 2: Scale
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask (for causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Step 4: Softmax to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)

    # Step 5: Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

### Causal Masking (for GPT-style models)

In autoregressive models, each position can only attend to previous positions:

```
Mask = [[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]
```

---

## Interview Tips

1. **Draw the diagram**: Be ready to sketch Q, K, V flow
2. **Know the complexity**: O(n²) is the key limitation
3. **Explain intuitively**: "Attention lets the model decide what's relevant"
4. **Mention alternatives**: Flash Attention, sparse attention for efficiency

---

## Code Demo

See `attention_demo.py` for:
- Scaled dot-product attention from scratch
- Multi-head attention implementation
- Visualization of attention patterns

```bash
python interview_questions/02_attention_mechanisms/attention_demo.py
```
