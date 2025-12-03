# Model Architectures

## Interview Questions Covered
- **Q20**: How does the architecture of GPT differ from BERT?
- **Q33**: What are the key differences between GPT and BERT architectures?
- **Q34**: Explain the role of positional encoding in the transformer architecture.
- **Q37**: What are the main components of a transformer model?
- **Q47**: What are the key components of the LLaMA architecture?
- **Q49**: What is the Mixture of Experts (MoE) model, and how does it improve efficiency?

---

## Q20 & Q33: GPT vs BERT Architectures

### BERT (Bidirectional Encoder Representations from Transformers)

**Architecture**: Encoder-only

```
[CLS] The cat sat [MASK] the mat [SEP]
  ↓    ↓   ↓   ↓    ↓    ↓   ↓   ↓
  ════════════════════════════════
         Bidirectional Attention
  ════════════════════════════════
  ↓    ↓   ↓   ↓    ↓    ↓   ↓   ↓
Output representations (all positions)
```

**Key Features**:
- Bidirectional attention (sees all tokens)
- Trained with MLM (predict [MASK]) and NSP
- Good for understanding tasks

### GPT (Generative Pre-trained Transformer)

**Architecture**: Decoder-only

```
The    cat    sat    on    the    →
 ↓      ↓      ↓      ↓      ↓
 ════════════════════════════════
       Causal (Left-to-Right) Attention
 ════════════════════════════════
 ↓      ↓      ↓      ↓      ↓
cat    sat    on    the    mat
```

**Key Features**:
- Unidirectional (causal) attention
- Trained with next token prediction
- Good for generation tasks

### Direct Comparison

| Aspect | BERT | GPT |
|--------|------|-----|
| **Architecture** | Encoder-only | Decoder-only |
| **Attention** | Bidirectional | Causal (unidirectional) |
| **Training** | MLM + NSP | Next token prediction |
| **Use case** | Classification, NER, QA | Generation, chat |
| **Context** | Sees all tokens | Only sees past tokens |
| **Output** | Embeddings | Token probabilities |

### Size Comparison

| Model | Parameters | Layers | Hidden | Heads |
|-------|------------|--------|--------|-------|
| BERT-base | 110M | 12 | 768 | 12 |
| BERT-large | 340M | 24 | 1024 | 16 |
| GPT-2 | 1.5B | 48 | 1600 | 25 |
| GPT-3 | 175B | 96 | 12288 | 96 |
| GPT-4 | ~1.8T* | ~120* | ? | ? |

*Estimated/rumored

---

## Q34: Positional Encoding

### The Problem

Attention is permutation-invariant:
```
Attention("cat sat mat") = Attention("mat cat sat")
```

We need to inject position information.

### Sinusoidal Encoding (Original Transformer)

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties**:
- Deterministic (no learning)
- Can extrapolate to longer sequences
- Encodes relative positions via linear combinations

### Learned Positional Embeddings (GPT, BERT)

```python
self.position_embeddings = nn.Embedding(max_seq_len, d_model)

def forward(self, x):
    positions = torch.arange(seq_len)
    return x + self.position_embeddings(positions)
```

**Properties**:
- Learned during training
- Limited to max_seq_len
- Often works better than sinusoidal

### Rotary Position Embeddings (RoPE) - LLaMA, GPT-NeoX

```python
# Rotate query and key vectors based on position
def apply_rope(x, positions):
    # Complex rotation in 2D subspaces
    cos = torch.cos(positions * freqs)
    sin = torch.sin(positions * freqs)
    return x * cos + rotate(x) * sin
```

**Properties**:
- Applied to Q and K, not values
- Encodes relative positions naturally
- Better length extrapolation

### ALiBi (Attention with Linear Biases) - BLOOM

```python
# Add position-based bias to attention scores
attention_scores = Q @ K.T / sqrt(d_k)
attention_scores += alibi_bias  # Linear decay based on distance
```

**Properties**:
- No positional embeddings in input
- Bias attention scores directly
- Excellent extrapolation

---

## Q37: Main Transformer Components

### Complete Architecture

```
Input Tokens
     ↓
[Token Embedding] + [Position Embedding]
     ↓
┌─────────────────────────────────┐
│  Multi-Head Self-Attention      │
│  ↓                              │
│  Add & LayerNorm (residual)     │
│  ↓                              │
│  Feed-Forward Network           │
│  ↓                              │
│  Add & LayerNorm (residual)     │
└─────────────────────────────────┘
     ↓ (repeat N times)
[Output Projection]
     ↓
Logits over vocabulary
```

### Component Details

#### 1. Token Embeddings
```python
embed = nn.Embedding(vocab_size, d_model)
# vocab_size: ~50K-100K
# d_model: 768-12288
```

#### 2. Multi-Head Attention
```python
# Split into h heads
Q, K, V = linear_q(x), linear_k(x), linear_v(x)
# Q, K, V: (batch, seq, d_model)

# Reshape to (batch, heads, seq, d_head)
# Apply attention per head
attention = softmax(Q @ K.T / sqrt(d_head)) @ V

# Concatenate heads
output = linear_out(concat(all_heads))
```

#### 3. Feed-Forward Network
```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        self.linear1 = nn.Linear(d_model, d_ff)  # Expand
        self.linear2 = nn.Linear(d_ff, d_model)  # Contract

    def forward(self, x):
        return self.linear2(gelu(self.linear1(x)))
# d_ff typically 4 * d_model
```

#### 4. Layer Normalization
```python
# Normalize across feature dimension
x = (x - mean) / std * gamma + beta
```

#### 5. Residual Connections
```python
x = x + attention(x)  # Skip connection
x = x + ffn(x)        # Skip connection
```

---

## Q47: LLaMA Architecture

### Key Components

LLaMA introduced several architectural improvements:

#### 1. Pre-normalization (RMSNorm)

```python
# Standard: Post-LayerNorm
x = LayerNorm(x + Attention(x))

# LLaMA: Pre-RMSNorm
x = x + Attention(RMSNorm(x))
```

RMSNorm (simpler than LayerNorm):
```python
def rmsnorm(x):
    return x / sqrt(mean(x²)) * scale
```

#### 2. SwiGLU Activation

```python
# Standard FFN: GELU
FFN(x) = Linear2(GELU(Linear1(x)))

# LLaMA: SwiGLU (gated)
FFN(x) = Linear2(Swish(Linear1(x)) * Linear3(x))
```

SwiGLU adds a gating mechanism that improves quality.

#### 3. Rotary Position Embeddings (RoPE)

```python
# Instead of adding positional embeddings to input
# Rotate Q and K vectors based on position
Q_rotated = apply_rotary_emb(Q, positions)
K_rotated = apply_rotary_emb(K, positions)
attention = Q_rotated @ K_rotated.T
```

### LLaMA Model Sizes

| Model | Params | Layers | Hidden | Heads | Context |
|-------|--------|--------|--------|-------|---------|
| LLaMA-7B | 6.7B | 32 | 4096 | 32 | 2048 |
| LLaMA-13B | 13B | 40 | 5120 | 40 | 2048 |
| LLaMA-33B | 33B | 60 | 6656 | 52 | 2048 |
| LLaMA-65B | 65B | 80 | 8192 | 64 | 2048 |
| LLaMA-2-70B | 70B | 80 | 8192 | 64 | 4096 |

### Why LLaMA Matters

1. **Open weights**: Available for research
2. **Efficient**: Good quality at smaller sizes
3. **Foundational**: Basis for Alpaca, Vicuna, etc.
4. **Proven architecture**: RoPE + SwiGLU now standard

---

## Q49: Mixture of Experts (MoE)

### Core Idea

Instead of one large FFN, use multiple "expert" FFNs and route tokens to different experts.

```
Input token
     ↓
[Router/Gate] → Which expert(s)?
     ↓
┌─────┬─────┬─────┬─────┐
│Exp1 │Exp2 │Exp3 │Exp4 │  (Only 1-2 activated)
└─────┴─────┴─────┴─────┘
     ↓
Weighted sum of expert outputs
```

### How Routing Works

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        self.experts = nn.ModuleList([FFN() for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # Get routing probabilities
        gate_logits = self.gate(x)  # (batch, seq, num_experts)
        gate_probs = softmax(gate_logits)

        # Select top-k experts
        top_k_probs, top_k_indices = topk(gate_probs, k=2)

        # Route to selected experts and combine
        output = sum(prob * expert(x) for prob, expert in selected)
        return output
```

### Efficiency Benefits

| Aspect | Dense Model | MoE Model |
|--------|-------------|-----------|
| **Total params** | 175B | 1.2T |
| **Active params** | 175B | 175B |
| **FLOPs per token** | High | Same as smaller dense |
| **Quality** | Good | Better (more capacity) |

### Famous MoE Models

- **GShard** (Google): 600B params, 2048 experts
- **Switch Transformer** (Google): 1.6T params
- **Mixtral** (Mistral): 8x7B = 46.7B total, 12.9B active
- **GPT-4**: Rumored 8x220B MoE

### Challenges

1. **Load balancing**: Some experts get all tokens
   - Solution: Auxiliary loss to balance routing

2. **Communication**: Experts on different devices
   - Solution: Expert parallelism

3. **Training instability**: Routing can collapse
   - Solution: Noise in routing, capacity limits

### Load Balancing Loss

```python
# Encourage uniform expert usage
importance = sum(gate_probs, dim=tokens)
load_balance_loss = num_experts * sum(importance²)
total_loss = task_loss + 0.01 * load_balance_loss
```

---

## Interview Tips

1. **GPT vs BERT**: Decoder-only vs encoder-only, causal vs bidirectional
2. **Positional encoding**: Sinusoidal, learned, RoPE, ALiBi
3. **Transformer components**: Attention + FFN + LayerNorm + Residual
4. **LLaMA innovations**: RoPE, SwiGLU, RMSNorm
5. **MoE**: More params, same compute via sparse routing

---

## No Code Demo

This is primarily a conceptual/architectural topic. Related code is in:
- `02_attention_mechanisms/` - Multi-head attention implementation
- `03_transformer_architecture/` - Positional encoding implementation
