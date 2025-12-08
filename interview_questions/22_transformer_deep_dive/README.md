# Transformer Architecture Deep Dive

## The Real Interview Question

The most common question in ML and LLM interviews isn't "Have you used Transformers?" — it's **"Do you actually understand how a Transformer works at a core level?"**

Using a Transformer is easy. Designing systems around one requires real depth.

## Core Concept

At its heart, the Transformer is built on a simple but powerful idea: **attention replaces recurrence and convolution**. Every token learns how much to focus on every other token — in parallel — enabling both speed and long-range dependency modeling.

### The Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- **Q** (Query): What am I looking for?
- **K** (Key): What do I contain?
- **V** (Value): What information do I provide?
- **√d_k**: Scaling factor to prevent vanishing gradients in softmax

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      ENCODER                             │
│  ┌─────────────────┐      ┌─────────────────┐           │
│  │   Multi-Head    │ ──── │   Add & Norm    │           │
│  │   Attention     │      │   (Residual)    │           │
│  └─────────────────┘      └─────────────────┘           │
│           │                        │                     │
│           ▼                        ▼                     │
│  ┌─────────────────┐      ┌─────────────────┐           │
│  │  Feed Forward   │ ──── │   Add & Norm    │ ──────────┼──┐
│  │    Network      │      │   (Residual)    │           │  │
│  └─────────────────┘      └─────────────────┘           │  │
└─────────────────────────────────────────────────────────┘  │
                                                              │
┌─────────────────────────────────────────────────────────┐  │
│                      DECODER                             │  │
│  ┌─────────────────┐      ┌─────────────────┐           │  │
│  │   Masked        │ ──── │   Add & Norm    │           │  │
│  │   Multi-Head    │      │   (Residual)    │           │  │
│  └─────────────────┘      └─────────────────┘           │  │
│           │                        │                     │  │
│           ▼                        ▼     Encoder Output  │  │
│  ┌─────────────────┐      ┌─────────────────┐◄──────────┼──┘
│  │  Cross-Attention│ ──── │   Add & Norm    │           │
│  │  (Enc-Dec Attn) │      │   (Residual)    │           │
│  └─────────────────┘      └─────────────────┘           │
│           │                        │                     │
│           ▼                        ▼                     │
│  ┌─────────────────┐      ┌─────────────────┐           │
│  │  Feed Forward   │ ──── │   Add & Norm    │           │
│  │    Network      │      │   (Residual)    │           │
│  └─────────────────┘      └─────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

## Key Concepts to Master

### 1. Self-Attention Complexity: O(n²)

Self-attention computes pairwise relationships between all tokens, resulting in O(n²) time and memory complexity with respect to sequence length.

**Why it matters:**
- A 512-token sequence: 262,144 attention computations
- A 4096-token sequence: 16,777,216 attention computations
- This is why context windows have limits

### 2. Positional Encodings

Transformers are **permutation-invariant** by design — they have no inherent notion of order. Positional encodings inject sequence order information.

**Original sinusoidal encoding:**
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**Modern alternatives:**
- Learned positional embeddings (BERT, GPT)
- Rotary Position Embeddings (RoPE) — used in LLaMA, Mistral
- ALiBi (Attention with Linear Biases)

### 3. Residual Connections & Layer Normalization

**Residual connections** enable gradient flow through deep networks:
```
output = LayerNorm(x + Sublayer(x))
```

**Why they matter:**
- Allow training of very deep networks (100+ layers)
- Preserve information from earlier layers
- Stabilize training dynamics

### 4. Multi-Head Attention

Multiple attention heads learn **different types of relationships**:
- Head 1 might focus on syntactic dependencies
- Head 2 might capture semantic similarity
- Head 3 might track coreference

```python
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

**It's about representation diversity, not just parallelism.**

### 5. Encoder-Decoder Attention (Cross-Attention)

In sequence-to-sequence tasks (translation, summarization):
- **Query** comes from the decoder (what am I generating?)
- **Key/Value** come from the encoder (what was the input?)

This enables the decoder to "look at" the encoded input while generating.

## The Depth Question: Why Do Transformers Still Fail?

> "If attention is all you need, why do Transformers still fail?"

The answer lives in:

| Limitation | Description |
|------------|-------------|
| **Context Limits** | Fixed context window (n tokens) — can't process arbitrarily long sequences |
| **Compute Cost** | O(n²) scaling makes very long contexts expensive |
| **Data Bias** | Models inherit and amplify biases from training data |
| **Alignment** | Raw prediction ≠ helpful/safe behavior — requires RLHF, DPO, etc. |

## Interview Questions

### Conceptual Understanding
1. **Explain the attention mechanism in your own words.** What problem does it solve that RNNs couldn't?
2. **Why do we scale by √d_k in the attention formula?** What happens if we don't?
3. **What's the difference between self-attention and cross-attention?**
4. **Why are positional encodings necessary?** What would happen without them?

### Architecture Details
5. **Walk through the encoder-decoder architecture.** What happens at each step?
6. **Why multi-head attention instead of single-head with larger dimensions?**
7. **What role do residual connections play?** Could we train deep Transformers without them?
8. **Pre-norm vs post-norm LayerNorm — what's the difference and why does it matter?**

### Complexity & Scaling
9. **Explain the O(n²) complexity of self-attention.** Why is this a problem?
10. **What approaches exist to reduce attention complexity?** (Sparse attention, linear attention, etc.)
11. **How do modern LLMs handle long contexts despite O(n²)?**

### Practical Understanding
12. **If attention is all you need, why do Transformers still fail?**
13. **What's the difference between encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5)?**
14. **Why do decoder-only models dominate modern LLMs?**

## Sample Answer: "Explain Self-Attention"

> Self-attention allows each position in a sequence to attend to all other positions, computing relevance scores that determine how much each token should influence the representation of every other token.
>
> For each token, we compute three vectors: Query (what I'm looking for), Key (what I contain), and Value (what information I provide). The attention score between tokens is the dot product of Query and Key, scaled and softmaxed to create a probability distribution. This distribution weights the Values to produce the output.
>
> Unlike RNNs that process sequentially and struggle with long-range dependencies, attention computes all relationships in parallel with direct connections between any two positions. The trade-off is O(n²) complexity — every token attends to every other token.

## Code Demo

Run the interactive demonstration:
```bash
uv run python interview_questions/22_transformer_deep_dive/transformer_demo.py
```

The demo visualizes:
- Attention weight computation and heatmaps
- Multi-head attention diversity
- Positional encoding patterns
- O(n²) complexity scaling

## Key Takeaways

1. **Attention replaces recurrence** — parallel computation with direct long-range connections
2. **O(n²) is the fundamental trade-off** — power vs. scalability
3. **Positional encodings inject order** into a permutation-invariant architecture
4. **Residuals + LayerNorm enable depth** — stable training of 100+ layer models
5. **Multi-head = representation diversity** — different heads learn different patterns
6. **Understanding limitations is as important as understanding capabilities**

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)
