# Transformer Architecture

## Interview Questions Covered
- **Q17**: How do transformers improve on traditional Seq2Seq models?
- **Q21**: What are positional encodings, and why are they used?
- **Q43**: How do transformers address the vanishing gradient problem?
- **Q46**: How do encoders and decoders differ in transformers?

---

## Q17: How do transformers improve on traditional Seq2Seq models?

### Answer

Traditional Seq2Seq models (RNN/LSTM-based) have fundamental limitations that transformers overcome:

### RNN Limitations → Transformer Solutions

| RNN Problem | Transformer Solution |
|-------------|---------------------|
| **Sequential processing** (slow) | **Parallel processing** via self-attention |
| **Vanishing gradients** over long sequences | **Direct connections** to all positions |
| **Fixed bottleneck** (single context vector) | **Attention over all encoder states** |
| **Limited long-range dependencies** | **Global attention** captures any distance |

### Key Transformer Advantages

1. **Parallelization**: All positions processed simultaneously during training
2. **Constant path length**: Any two positions are 1 attention step apart
3. **Scalability**: Scales better with compute (more layers, more heads)

### Architecture Comparison

```
RNN Seq2Seq:
Input → [RNN → RNN → RNN] → context → [RNN → RNN → RNN] → Output
        Sequential, O(n) depth         Sequential

Transformer:
Input → [Self-Attention + FFN] × N → [Cross-Attention + FFN] × N → Output
        Parallel, O(1) depth           Parallel
```

---

## Q21: What are positional encodings?

### Answer

**Positional encodings** inject sequence order information into transformer inputs, since self-attention is inherently **order-agnostic**.

### The Problem

Self-attention treats input as a set, not a sequence:
- "The cat sat on the mat" and "mat the on sat cat The" produce identical attention patterns without positional information

### Solution: Add Position Information

```python
output = embedding(token) + positional_encoding(position)
```

### Sinusoidal Positional Encoding (Original Transformer)

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Why Sinusoidal?

1. **Unique encoding**: Each position gets a distinct vector
2. **Relative positions**: PE(pos+k) can be represented as a linear function of PE(pos)
3. **Extrapolation**: Can handle sequences longer than training data

### Alternative: Learned Positional Embeddings

- GPT, BERT use learned position embeddings
- Simpler but limited to maximum training length

### Modern Approaches

- **RoPE** (Rotary Position Embedding): Used in LLaMA, encodes relative positions
- **ALiBi**: Adds position bias to attention scores
- **Relative positional encoding**: T5, Transformer-XL

---

## Q43: How do transformers address vanishing gradients?

### Answer

Transformers use three key mechanisms:

### 1. Self-Attention (No Sequential Dependency)

```
RNN: gradient must flow through O(n) steps
Transformer: direct attention connection, O(1) steps
```

### 2. Residual Connections

```python
# Every sub-layer has a skip connection
output = LayerNorm(x + SubLayer(x))
```

Benefits:
- Gradients flow directly through skip connections
- Enables training of very deep networks (100+ layers)

### 3. Layer Normalization

```python
# Normalizes activations, stabilizes training
LayerNorm(x) = γ * (x - μ) / σ + β
```

Benefits:
- Prevents activation explosion/vanishing
- Enables higher learning rates

### Why RNNs Suffer

```
RNN gradient: ∏(∂h_t/∂h_{t-1}) over many steps
- If < 1: vanishes exponentially
- If > 1: explodes exponentially

Transformer gradient: direct path through attention + residual
- No multiplicative chain over sequence length
```

---

## Q46: How do encoders and decoders differ?

### Answer

### Encoder (BERT-style)
- **Bidirectional**: Sees all tokens simultaneously
- **Self-attention**: Each token attends to all others
- **Use case**: Understanding tasks (classification, NER, QA)

```
Input: "The cat [MASK] on the mat"
       ↓ ↓ ↓ ↓ ↓ ↓ ↓ (all attend to all)
Output: Contextualized representations
```

### Decoder (GPT-style)
- **Unidirectional**: Only sees previous tokens (causal masking)
- **Masked self-attention**: Prevents "peeking" at future
- **Use case**: Generation tasks (text completion, chat)

```
Input: "The cat sat"
       ↓   ↓   ↓
       The→cat→sat (each only attends to previous)
Output: Next token prediction
```

### Encoder-Decoder (T5, BART)
- Encoder processes input bidirectionally
- Decoder generates output autoregressively
- Cross-attention connects decoder to encoder
- **Use case**: Seq2Seq tasks (translation, summarization)

```
Encoder: "Translate: Hello" → [h1, h2]
                                ↓ (cross-attention)
Decoder: "Bonjour" (generated token by token)
```

### Summary Table

| Architecture | Attention Type | Training Objective | Example Models |
|-------------|----------------|-------------------|----------------|
| Encoder-only | Bidirectional | MLM, NSP | BERT, RoBERTa |
| Decoder-only | Causal | Next token | GPT, LLaMA |
| Encoder-Decoder | Both | Seq2Seq | T5, BART |

---

## Code Demo

See `positional_encoding.py` for:
- Sinusoidal positional encoding implementation
- Visualization of encoding patterns
- Comparison with learned embeddings

```bash
python interview_questions/03_transformer_architecture/positional_encoding.py
```
