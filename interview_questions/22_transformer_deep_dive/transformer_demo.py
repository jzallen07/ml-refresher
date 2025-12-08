# %% [markdown]
# ---
# title: Transformer Deep Dive Demo
# ---

# %% [markdown]
# # Transformer Architecture Deep Dive
#
# This demo visualizes the core concepts of the Transformer architecture
# that are commonly tested in ML/LLM interviews.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

# %%
print("=" * 70)
print("TRANSFORMER ARCHITECTURE DEEP DIVE")
print("=" * 70)

# %% [markdown]
# ## 1. Scaled Dot-Product Attention
#
# The core attention formula:
# $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

# %%
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        query: (batch, seq_len, d_k)
        key: (batch, seq_len, d_k)
        value: (batch, seq_len, d_v)
        mask: optional mask for decoder self-attention

    Returns:
        attention_output: (batch, seq_len, d_v)
        attention_weights: (batch, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # Step 1: Compute attention scores (QK^T)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k) to prevent vanishing gradients
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask if provided (for decoder)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Step 5: Weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# Demo: Simple attention computation
print("\n1. SCALED DOT-PRODUCT ATTENTION")
print("-" * 50)

# Create sample Q, K, V
batch_size, seq_len, d_k = 1, 4, 8
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Query shape: {Q.shape}")
print(f"Key shape: {K.shape}")
print(f"Value shape: {V.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\nAttention weights (each row sums to 1):")
print(weights[0].numpy().round(3))
print(f"Row sums: {weights[0].sum(dim=-1).numpy()}")

# %% [markdown]
# ## 2. Why Scale by √d_k?
#
# Without scaling, large d_k causes dot products to have large magnitude,
# pushing softmax into regions with extremely small gradients.

# %%
print("\n2. WHY SCALE BY √d_k?")
print("-" * 50)

# Compare scaled vs unscaled attention
d_k_large = 512
Q_large = torch.randn(1, 4, d_k_large)
K_large = torch.randn(1, 4, d_k_large)

# Unscaled scores
unscaled_scores = torch.matmul(Q_large, K_large.transpose(-2, -1))
# Scaled scores
scaled_scores = unscaled_scores / math.sqrt(d_k_large)

print(f"d_k = {d_k_large}")
print(f"Unscaled score magnitude: {unscaled_scores.abs().mean():.2f}")
print(f"Scaled score magnitude: {scaled_scores.abs().mean():.2f}")
print(f"\nUnscaled softmax (saturated - bad gradients):")
print(F.softmax(unscaled_scores[0], dim=-1).numpy().round(4))
print(f"\nScaled softmax (distributed - good gradients):")
print(F.softmax(scaled_scores[0], dim=-1).numpy().round(4))

# %% [markdown]
# ## 3. O(n²) Complexity Visualization
#
# Self-attention computes all pairwise relationships, leading to
# quadratic scaling with sequence length.

# %%
print("\n3. O(n²) COMPLEXITY")
print("-" * 50)

seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
computations = [n * n for n in seq_lengths]

print("Sequence Length | Attention Computations | Memory (approx)")
print("-" * 60)
for n, c in zip(seq_lengths, computations):
    # Assuming float32 (4 bytes) for attention matrix
    memory_mb = (c * 4) / (1024 * 1024)
    print(f"{n:>14} | {c:>22,} | {memory_mb:>10.2f} MB")

print("\nThis is why context windows have limits!")

# %% [markdown]
# ## 4. Positional Encodings
#
# Transformers are permutation-invariant, so positional encodings
# inject sequence order information.

# %%
def sinusoidal_positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encodings.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

    return pe


print("\n4. POSITIONAL ENCODINGS")
print("-" * 50)

# Generate positional encodings
seq_len, d_model = 100, 64
pe = sinusoidal_positional_encoding(seq_len, d_model)

print(f"Positional encoding shape: {pe.shape}")
print(f"Position 0, first 8 dims: {pe[0, :8].numpy().round(3)}")
print(f"Position 1, first 8 dims: {pe[1, :8].numpy().round(3)}")

# Show that relative positions can be computed
print(f"\nRelative position property:")
print("PE(pos+k) can be expressed as a linear function of PE(pos)")

# %% [markdown]
# ## 5. Multi-Head Attention
#
# Multiple heads learn different types of relationships:
# syntactic, semantic, coreference, etc.

# %%
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention as described in 'Attention Is All You Need'."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention for all heads in parallel
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)

        return output, attn_weights


print("\n5. MULTI-HEAD ATTENTION")
print("-" * 50)

# Create multi-head attention
d_model, num_heads = 64, 8
mha = MultiHeadAttention(d_model, num_heads)

# Sample input
x = torch.randn(1, 10, d_model)  # batch=1, seq_len=10, d_model=64
output, attn_weights = mha(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")
print(f"  - {num_heads} heads, each attending over {attn_weights.shape[-1]} positions")

# Show head diversity
print(f"\nHead diversity (attention entropy per head):")
for h in range(num_heads):
    head_weights = attn_weights[0, h]
    entropy = -(head_weights * torch.log(head_weights + 1e-9)).sum(dim=-1).mean()
    print(f"  Head {h}: entropy = {entropy:.3f}")

# %% [markdown]
# ## 6. Residual Connections & Layer Norm
#
# These enable training of very deep networks (100+ layers).

# %%
class TransformerBlock(nn.Module):
    """A single Transformer encoder block with residuals and layer norm."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))  # Residual + LayerNorm

        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))  # Residual + LayerNorm

        return x


print("\n6. RESIDUAL CONNECTIONS & LAYER NORM")
print("-" * 50)

# Create a transformer block
block = TransformerBlock(d_model=64, num_heads=8, d_ff=256)

# Forward pass
x = torch.randn(1, 10, 64)
output = block(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Show gradient flow with residuals
print(f"\nWhy residuals matter:")
print("  - Without residuals: gradients vanish in deep networks")
print("  - With residuals: gradient = 1 + other_terms (always flows)")

# %% [markdown]
# ## 7. Encoder-Decoder (Cross) Attention
#
# In seq2seq tasks, the decoder attends to encoder outputs.

# %%
print("\n7. ENCODER-DECODER ATTENTION")
print("-" * 50)

# Simulate encoder-decoder attention
encoder_output = torch.randn(1, 20, 64)  # Source sequence (e.g., French)
decoder_input = torch.randn(1, 15, 64)   # Target sequence (e.g., English)

cross_attention = MultiHeadAttention(d_model=64, num_heads=8)

# Q from decoder, K/V from encoder
output, attn_weights = cross_attention(
    query=decoder_input,    # "What am I generating?"
    key=encoder_output,     # "What was the input?"
    value=encoder_output
)

print(f"Encoder output shape: {encoder_output.shape} (source sequence)")
print(f"Decoder input shape: {decoder_input.shape} (target sequence)")
print(f"Cross-attention output: {output.shape}")
print(f"Attention weights: {attn_weights.shape}")
print(f"  - Each decoder position attends to all encoder positions")

# %% [markdown]
# ## 8. Why Transformers Still Fail
#
# Understanding limitations is as important as understanding capabilities.

# %%
print("\n8. WHY TRANSFORMERS STILL FAIL")
print("-" * 50)

limitations = [
    ("Context Limits", "Fixed window size (n tokens)", "Can't process arbitrarily long documents"),
    ("Compute Cost", "O(n²) attention complexity", "Long contexts are expensive"),
    ("Data Bias", "Trained on internet data", "Inherits and amplifies biases"),
    ("Alignment", "Predicts next token", "Prediction ≠ helpful/safe behavior"),
    ("Hallucination", "No grounding in truth", "Confidently generates false information"),
    ("Reasoning", "Pattern matching", "Struggles with novel logical problems"),
]

print(f"{'Limitation':<20} {'Description':<35} {'Impact'}")
print("-" * 90)
for name, desc, impact in limitations:
    print(f"{name:<20} {desc:<35} {impact}")

# %% [markdown]
# ## Visualization: Attention Patterns

# %%
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

import os
os.makedirs("data/interview_viz", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Attention heatmap
ax = axes[0, 0]
tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
seq_len = len(tokens)
Q = torch.randn(1, seq_len, 32)
K = torch.randn(1, seq_len, 32)
V = torch.randn(1, seq_len, 32)
_, attn = scaled_dot_product_attention(Q, K, V)
im = ax.imshow(attn[0].numpy(), cmap='Blues')
ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xticklabels(tokens, rotation=45)
ax.set_yticklabels(tokens)
ax.set_xlabel("Key (attending to)")
ax.set_ylabel("Query (from)")
ax.set_title("Self-Attention Weights")
plt.colorbar(im, ax=ax)

# 2. Positional encoding patterns
ax = axes[0, 1]
pe = sinusoidal_positional_encoding(50, 64)
im = ax.imshow(pe.numpy().T, cmap='RdBu', aspect='auto')
ax.set_xlabel("Position")
ax.set_ylabel("Dimension")
ax.set_title("Sinusoidal Positional Encodings")
plt.colorbar(im, ax=ax)

# 3. O(n²) complexity
ax = axes[1, 0]
seq_lengths = np.array([64, 128, 256, 512, 1024, 2048, 4096])
computations = seq_lengths ** 2
ax.plot(seq_lengths, computations / 1e6, 'b-o', linewidth=2)
ax.fill_between(seq_lengths, 0, computations / 1e6, alpha=0.3)
ax.set_xlabel("Sequence Length")
ax.set_ylabel("Attention Computations (millions)")
ax.set_title("O(n²) Complexity Scaling")
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 4. Multi-head attention diversity
ax = axes[1, 1]
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(1, 8, 64)
_, attn_weights = mha(x, x, x)
# Show different patterns per head
head_data = []
for h in range(8):
    head_data.append(attn_weights[0, h].detach().numpy().flatten())
ax.boxplot(head_data, labels=[f"H{i}" for i in range(8)])
ax.set_xlabel("Attention Head")
ax.set_ylabel("Attention Weight Distribution")
ax.set_title("Multi-Head Attention Diversity")

plt.tight_layout()
plt.savefig("data/interview_viz/transformer_deep_dive.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: data/interview_viz/transformer_deep_dive.png")

# %% [markdown]
# ## Summary

# %%
print("\n" + "=" * 70)
print("KEY INTERVIEW TAKEAWAYS")
print("=" * 70)

takeaways = """
1. ATTENTION MECHANISM
   - Replaces recurrence with parallel pairwise attention
   - Formula: softmax(QK^T / √d_k) × V
   - Scale factor prevents vanishing gradients

2. O(n²) COMPLEXITY
   - Every token attends to every other token
   - Limits context window size
   - Various optimizations exist (sparse, linear attention)

3. POSITIONAL ENCODINGS
   - Transformers are permutation-invariant
   - Must inject order information externally
   - Options: sinusoidal, learned, RoPE, ALiBi

4. RESIDUALS + LAYER NORM
   - Enable training very deep networks
   - Gradient flows through skip connections
   - Pre-norm vs post-norm affects stability

5. MULTI-HEAD ATTENTION
   - Different heads learn different patterns
   - Diversity > parallelism
   - Typically 8-128 heads in modern models

6. LIMITATIONS MATTER
   - Context limits, compute cost
   - Data bias, alignment, hallucination
   - Understanding failures shows depth
"""
print(takeaways)

print("=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
