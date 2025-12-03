# Embeddings

## Interview Questions Covered
- **Q10**: What are embeddings, and how are they initialized in LLMs?

---

## Q10: What are embeddings?

### Definition

**Embeddings** are dense vector representations of discrete tokens in a continuous vector space. They map tokens (words, subwords) to fixed-size vectors where similar tokens have similar representations.

### From Tokens to Vectors

```
Vocabulary: {"cat": 0, "dog": 1, "the": 2, ...}

Token ID 0 ("cat") → [0.23, -0.45, 0.67, ..., 0.12]  # d-dimensional vector
Token ID 1 ("dog") → [0.25, -0.42, 0.71, ..., 0.09]  # Similar to "cat"
Token ID 2 ("the") → [-0.89, 0.12, -0.34, ..., 0.56] # Different (function word)
```

### Why Not One-Hot Encoding?

| One-Hot | Embeddings |
|---------|------------|
| Sparse (mostly zeros) | Dense (all values meaningful) |
| Size = vocabulary (50K+) | Size = embedding dim (768) |
| No similarity info | Similar words → similar vectors |
| Fixed | Learned during training |

### The Embedding Layer

```python
import torch.nn as nn

# Create embedding layer
vocab_size = 50000
embedding_dim = 768
embeddings = nn.Embedding(vocab_size, embedding_dim)

# Look up embeddings for token IDs
token_ids = torch.tensor([0, 1, 2])  # "cat", "dog", "the"
vectors = embeddings(token_ids)      # Shape: [3, 768]
```

### Initialization Methods

| Method | Description | Used By |
|--------|-------------|---------|
| **Random** | Normal(0, 0.02) or Uniform | Most transformers |
| **Xavier/Glorot** | Scaled by layer size | Common default |
| **Pretrained** | Load GloVe, Word2Vec | Fine-tuning scenarios |
| **Tied** | Share input/output embeddings | GPT-2, T5 |

### Typical Initialization

```python
# Standard transformer initialization
nn.init.normal_(embeddings.weight, mean=0.0, std=0.02)

# Or Xavier
nn.init.xavier_uniform_(embeddings.weight)
```

### What Embeddings Capture

After training, embeddings encode:

1. **Semantic similarity**: "king" close to "queen"
2. **Syntactic patterns**: Verbs cluster together
3. **Analogies**: king - man + woman ≈ queen
4. **Polysemy**: Same word in different contexts (via attention)

### Embedding Dimensions in Practice

| Model | Embedding Dim | Vocab Size |
|-------|---------------|------------|
| BERT-base | 768 | 30,522 |
| BERT-large | 1,024 | 30,522 |
| GPT-2 | 768-1,600 | 50,257 |
| GPT-3 | 12,288 | 50,257 |
| LLaMA 2 7B | 4,096 | 32,000 |

### Static vs Contextual Embeddings

**Static (Word2Vec, GloVe)**:
- One vector per word, regardless of context
- "bank" has same embedding in "river bank" and "bank account"

**Contextual (BERT, GPT)**:
- Different vector based on surrounding context
- The embedding layer produces static vectors, but attention creates context-dependent representations

```
"The bank by the river" → bank = [water-related vector]
"I went to the bank"    → bank = [finance-related vector]
```

### Weight Tying

Many models share weights between input embeddings and output projection:

```python
# Input: token_id → embedding vector
# Output: hidden state → logits over vocabulary

# Weight tying: Use same matrix for both
model.output_projection.weight = model.embeddings.weight
```

**Benefits**:
- Fewer parameters
- Better generalization
- Embeddings trained from both directions

---

## Interview Tips

1. **Know the dimensions**: BERT is 768, GPT-3 is 12,288
2. **Explain the lookup**: It's just a matrix multiplication with one-hot
3. **Contextual vs static**: Transformers create contextual representations
4. **Weight tying**: Common optimization in decoder models

---

## Code Demo

See `embeddings_demo.py` for:
- Creating and using nn.Embedding
- Visualizing embedding space (PCA/t-SNE)
- Computing cosine similarity
- Weight tying demonstration

```bash
python interview_questions/07_embeddings/embeddings_demo.py
```
