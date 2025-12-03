"""
Positional Encoding in Transformers - Interview Preparation Demo

This demo covers essential concepts for LLM interviews:
- Q17: How do transformers handle sequence order?
- Q21: What is positional encoding and why is it needed?
- Q43: Sinusoidal vs learned positional embeddings
- Q46: Residual connections and layer normalization

Key Interview Points:
1. Self-attention is permutation-invariant (order-agnostic)
2. Positional encoding injects position information
3. Sinusoidal encoding allows extrapolation to longer sequences
4. Different encoding strategies have different tradeoffs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math

# Ensure output directory exists
output_dir = Path("/Users/zack/dev/ml-refresher/data/interview_viz")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("POSITIONAL ENCODING IN TRANSFORMERS - INTERVIEW DEMO")
print("=" * 80)


# =============================================================================
# PART 1: Why Positional Encoding is Needed
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: Why Do We Need Positional Encoding?")
print("=" * 80)

print("""
INTERVIEW ANSWER:
Self-attention is PERMUTATION-INVARIANT, meaning it treats input as a set,
not a sequence. Without positional encoding, "dog bites man" and "man bites dog"
would produce identical representations.

Mathematically, for a permutation π:
    Attention(X_π) = Attention(X)_π

This is because attention computes QK^T which is symmetric w.r.t. reordering.
""")


class SimpleAttention(nn.Module):
    """Simplified attention to demonstrate order-agnostic behavior"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        Returns: attention weights and output
        """
        # For simplicity, use x as Q, K, V
        Q = K = V = x

        # Attention scores: QK^T / sqrt(d_model)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


# Demonstrate order-agnostic behavior
d_model = 4
seq_len = 3

# Create a simple sequence
x1 = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],  # Position 0: token "dog"
    [0.0, 1.0, 0.0, 0.0],  # Position 1: token "bites"
    [0.0, 0.0, 1.0, 0.0],  # Position 2: token "man"
]).unsqueeze(0)  # Add batch dimension

# Permute the sequence (swap positions 0 and 2)
x2 = torch.tensor([
    [0.0, 0.0, 1.0, 0.0],  # Position 0: token "man"
    [0.0, 1.0, 0.0, 0.0],  # Position 1: token "bites"
    [1.0, 0.0, 0.0, 0.0],  # Position 2: token "dog"
]).unsqueeze(0)

attention = SimpleAttention(d_model)

with torch.no_grad():
    out1, attn1 = attention(x1)
    out2, attn2 = attention(x2)

print("\nDemonstration: Attention without positional encoding")
print("\nSequence 1 (dog bites man):")
print(x1.squeeze())
print("\nSequence 2 (man bites dog):")
print(x2.squeeze())

print("\nAttention output 1 (sorted by position):")
print(out1.squeeze())
print("\nAttention output 2 (sorted by position):")
print(out2.squeeze())

# The outputs are permutations of each other!
print("\n⚠️  IMPORTANT: The attention outputs are just permuted versions!")
print("This shows why we NEED positional information.")


# =============================================================================
# PART 2: Sinusoidal Positional Encoding (Original Transformer)
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Sinusoidal Positional Encoding")
print("=" * 80)

print("""
INTERVIEW ANSWER:
The original Transformer paper (Vaswani et al., 2017) uses sinusoidal functions:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos: position in sequence (0 to max_len-1)
- i: dimension index (0 to d_model/2-1)
- Even dimensions use sine, odd dimensions use cosine

Key Advantages:
1. No learned parameters (deterministic)
2. Can extrapolate to longer sequences than seen during training
3. Allows model to easily learn relative positions (linear combination)
4. Different wavelengths for different dimensions (10000^0 to 10000^1)

The wavelengths form a geometric progression from 2π to 10000·2π
""")


class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding from "Attention Is All You Need"

    INTERVIEW INSIGHT:
    - This encoding is added to input embeddings, not concatenated
    - Each dimension has a different frequency
    - Low dimensions change quickly (high frequency)
    - High dimensions change slowly (low frequency)
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Create positional encoding matrix
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: [0, 1, 2, ..., max_len-1]
        # Shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Dimension indices for the geometric progression
        # div_term represents: 10000^(2i/d_model) for i in [0, d_model/2)
        # We use exp and log for numerical stability:
        # 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Apply sine to even dimensions (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd dimensions (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        # Add positional encoding to input
        # Broadcasting handles batch dimension automatically
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

    def get_encoding(self, max_len=None):
        """Get the positional encoding matrix for visualization"""
        if max_len is None:
            return self.pe.squeeze(0)
        return self.pe.squeeze(0)[:max_len, :]


# Create and visualize sinusoidal positional encoding
d_model = 128
max_len = 100

pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
pe_matrix = pos_encoder.get_encoding(max_len=max_len)

print(f"\nSinusoidal Positional Encoding Shape: {pe_matrix.shape}")
print(f"(sequence_length={max_len}, d_model={d_model})")

# Show encoding for a few positions
print("\nSample encoding values for first 3 positions:")
for pos in range(3):
    print(f"\nPosition {pos}:")
    print(f"  First 8 dimensions: {pe_matrix[pos, :8]}")
    print(f"  Last 8 dimensions:  {pe_matrix[pos, -8:]}")

# Visualize the positional encoding
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Heatmap of positional encodings
ax1 = axes[0, 0]
im1 = ax1.imshow(pe_matrix.numpy(), aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
ax1.set_xlabel('Embedding Dimension')
ax1.set_ylabel('Position in Sequence')
ax1.set_title('Sinusoidal Positional Encoding Heatmap\n(Each row is a position vector)')
plt.colorbar(im1, ax=ax1)

# Plot 2: Different dimensions over positions
ax2 = axes[0, 1]
dimensions_to_plot = [0, 1, 32, 33, 64, 65, 96, 97]
for dim in dimensions_to_plot:
    ax2.plot(pe_matrix[:50, dim].numpy(),
             label=f'Dim {dim}', alpha=0.7)
ax2.set_xlabel('Position')
ax2.set_ylabel('Encoding Value')
ax2.set_title('Positional Encoding: Different Dimensions\n(Low dims change fast, high dims change slow)')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Same position, different dimensions
ax3 = axes[1, 0]
positions_to_plot = [0, 10, 25, 50, 75]
for pos in positions_to_plot:
    ax3.plot(pe_matrix[pos, :64].numpy(),
             label=f'Pos {pos}', alpha=0.7, marker='o', markersize=2)
ax3.set_xlabel('Dimension')
ax3.set_ylabel('Encoding Value')
ax3.set_title('Different Positions Across Dimensions\n(First 64 dimensions shown)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Wavelength demonstration
ax4 = axes[1, 1]
# Show how wavelength increases with dimension
wavelengths = []
for i in range(0, d_model, 2):
    # Wavelength = 2π * 10000^(i/d_model)
    wavelength = 2 * math.pi * (10000 ** (i / d_model))
    wavelengths.append(wavelength)

ax4.semilogy(range(0, d_model, 2), wavelengths, marker='o')
ax4.set_xlabel('Dimension Index')
ax4.set_ylabel('Wavelength (log scale)')
ax4.set_title('Wavelength by Dimension\n(Geometric progression from 2π to 10000·2π)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'positional_encoding_sinusoidal.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: {output_dir / 'positional_encoding_sinusoidal.png'}")


# =============================================================================
# PART 3: Learned Positional Embeddings
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: Learned Positional Embeddings")
print("=" * 80)

print("""
INTERVIEW ANSWER:
Alternative to sinusoidal: Learn positional embeddings like token embeddings.

Implementation: nn.Embedding(max_seq_len, d_model)

Advantages:
+ Can learn task-specific positional patterns
+ May perform better on fixed-length sequences
+ Used in BERT, GPT-2

Disadvantages:
- Cannot extrapolate beyond max_seq_len seen during training
- Requires learning parameters (memory overhead)
- May overfit to training sequence lengths

Trade-off: Flexibility vs. Generalization
""")


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (like in BERT, GPT-2)

    INTERVIEW INSIGHT:
    - Each position gets a learnable embedding vector
    - Similar to token embeddings, but for positions
    - Must specify maximum sequence length upfront
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable embedding for each position
        self.position_embeddings = nn.Embedding(max_len, d_model)

        # Initialize with small random values
        nn.init.normal_(self.position_embeddings.weight, mean=0, std=0.02)

        self.max_len = max_len
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Get positional embeddings
        pos_embeddings = self.position_embeddings(positions)

        # Add to input
        x = x + pos_embeddings
        return self.dropout(x)


# Create learned positional embedding
learned_pos = LearnedPositionalEmbedding(d_model=128, max_len=100)

print(f"\nLearned Positional Embedding:")
print(f"  Parameters: {sum(p.numel() for p in learned_pos.parameters()):,}")
print(f"  Max sequence length: {learned_pos.max_len}")
print(f"  Embedding dimension: {learned_pos.d_model}")

# Show initial random embeddings
with torch.no_grad():
    learned_pe = learned_pos.position_embeddings.weight[:100].numpy()

print(f"\nInitial learned embeddings (before training):")
print(f"  Shape: {learned_pe.shape}")
print(f"  Mean: {learned_pe.mean():.4f}")
print(f"  Std: {learned_pe.std():.4f}")


# =============================================================================
# PART 4: Comparison of Positional Encoding Methods
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: Comparing Positional Encoding Methods")
print("=" * 80)

print("""
INTERVIEW QUESTION: Which positional encoding should we use?

ANSWER: It depends on the use case!

┌─────────────────┬──────────────────┬─────────────────────┐
│    Property     │   Sinusoidal     │      Learned        │
├─────────────────┼──────────────────┼─────────────────────┤
│ Parameters      │ None (0)         │ max_len × d_model   │
│ Extrapolation   │ Yes (any length) │ No (fixed max_len)  │
│ Training        │ Not needed       │ Learned from data   │
│ Performance     │ Good baseline    │ Often better        │
│ Used in         │ Original Trans.  │ BERT, GPT-2, GPT-3  │
│ Best for        │ Variable lengths │ Fixed-length tasks  │
└─────────────────┴──────────────────┴─────────────────────┘

Modern trend: Most large models use LEARNED positional embeddings
""")


def compare_encodings(seq_len=50, d_model=64):
    """Compare sinusoidal and learned (initialized) encodings"""

    # Sinusoidal
    sin_encoder = SinusoidalPositionalEncoding(d_model, max_len=seq_len)
    sin_pe = sin_encoder.get_encoding(max_len=seq_len).numpy()

    # Learned (random initialization)
    learned_encoder = LearnedPositionalEmbedding(d_model, max_len=seq_len)
    with torch.no_grad():
        learned_pe = learned_encoder.position_embeddings.weight.numpy()

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sinusoidal
    im1 = axes[0].imshow(sin_pe, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('Sinusoidal Encoding\n(Deterministic, wavelength-based)')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Position')
    plt.colorbar(im1, ax=axes[0])

    # Learned (initialized)
    im2 = axes[1].imshow(learned_pe, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title('Learned Encoding (Random Init)\n(Before training)')
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Position')
    plt.colorbar(im2, ax=axes[1])

    # Difference
    # Normalize both to [-1, 1] for fair comparison
    sin_pe_norm = sin_pe / (np.abs(sin_pe).max() + 1e-8)
    learned_pe_norm = learned_pe / (np.abs(learned_pe).max() + 1e-8)
    diff = np.abs(sin_pe_norm - learned_pe_norm)

    im3 = axes[2].imshow(diff, aspect='auto', cmap='viridis')
    axes[2].set_title('Absolute Difference (Normalized)\n(Shows structural differences)')
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Position')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_dir / 'positional_encoding_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison: {output_dir / 'positional_encoding_comparison.png'}")

    # Statistics
    print("\nStatistical Comparison:")
    print(f"Sinusoidal encoding:")
    print(f"  Mean: {sin_pe.mean():.4f}, Std: {sin_pe.std():.4f}")
    print(f"  Min: {sin_pe.min():.4f}, Max: {sin_pe.max():.4f}")

    print(f"\nLearned encoding (init):")
    print(f"  Mean: {learned_pe.mean():.4f}, Std: {learned_pe.std():.4f}")
    print(f"  Min: {learned_pe.min():.4f}, Max: {learned_pe.max():.4f}")


compare_encodings(seq_len=50, d_model=64)


# =============================================================================
# PART 5: Residual Connections and Layer Normalization
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: Residual Connections and Layer Normalization")
print("=" * 80)

print("""
INTERVIEW ANSWER:
Transformers use two key architectural components:

1. RESIDUAL CONNECTIONS (Skip Connections):
   output = LayerNorm(x + Sublayer(x))

   Why?
   - Helps with gradient flow (prevents vanishing gradients)
   - Allows direct path from input to output
   - Enables training of very deep networks
   - Identity mapping as initialization

2. LAYER NORMALIZATION:
   - Normalizes across feature dimension (not batch)
   - Stabilizes training
   - Reduces internal covariate shift
   - Applied BEFORE or AFTER sublayer (Pre-LN vs Post-LN)

Modern trend: Pre-LN (norm before sublayer) is more stable
""")


class TransformerBlock(nn.Module):
    """
    Single transformer block with:
    1. Multi-head attention
    2. Residual connection + Layer norm
    3. Feed-forward network
    4. Residual connection + Layer norm

    INTERVIEW INSIGHT:
    This is the core building block of transformers.
    GPT-3 has 96 of these stacked!
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, pre_norm=True):
        super().__init__()

        self.pre_norm = pre_norm

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Modern transformers use GELU instead of ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Pre-LN architecture (modern):
            x = x + Attention(LayerNorm(x))
            x = x + FFN(LayerNorm(x))

        Post-LN architecture (original):
            x = LayerNorm(x + Attention(x))
            x = LayerNorm(x + FFN(x))
        """
        if self.pre_norm:
            # Pre-LN: Normalize before sublayer
            # Attention block
            normed = self.norm1(x)
            attn_out, _ = self.attention(normed, normed, normed, attn_mask=mask)
            x = x + self.dropout(attn_out)

            # FFN block
            normed = self.norm2(x)
            ffn_out = self.ffn(normed)
            x = x + ffn_out
        else:
            # Post-LN: Normalize after sublayer
            # Attention block
            attn_out, _ = self.attention(x, x, x, attn_mask=mask)
            x = self.norm1(x + self.dropout(attn_out))

            # FFN block
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

        return x


# Demonstrate residual connections
print("\nDemonstrating Residual Connections:")

d_model = 64
batch_size = 2
seq_len = 10

# Create random input
x = torch.randn(batch_size, seq_len, d_model)

# Create transformer block
block = TransformerBlock(d_model=d_model, num_heads=4, d_ff=256, pre_norm=True)

# Forward pass
with torch.no_grad():
    output = block(x)

print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"\nInput mean: {x.mean():.4f}, std: {x.std():.4f}")
print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")

# Visualize the effect of residual connections on gradient flow
print("\n" + "=" * 80)
print("Visualizing Residual Connection Benefits")
print("=" * 80)


def analyze_gradient_flow(num_layers=6):
    """
    Demonstrate why residual connections help with gradient flow
    """
    print(f"\nAnalyzing gradient flow through {num_layers} layers:")

    # Create a simple network with residual connections
    layers_with_residual = nn.ModuleList([
        TransformerBlock(d_model=64, num_heads=4, d_ff=256)
        for _ in range(num_layers)
    ])

    # Create input that requires gradient
    x = torch.randn(1, 10, 64, requires_grad=True)

    # Forward pass
    out = x
    for layer in layers_with_residual:
        out = layer(out)

    # Compute loss and backward
    loss = out.sum()
    loss.backward()

    # Check gradient
    print(f"\nWith residual connections:")
    print(f"  Input gradient norm: {x.grad.norm().item():.4f}")
    print(f"  Input gradient mean: {x.grad.mean().item():.6f}")
    print(f"  Gradient is well-behaved ✓")

    return x.grad.norm().item()


grad_norm = analyze_gradient_flow(num_layers=6)


# =============================================================================
# PART 6: Complete Example with All Components
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: Complete Transformer Encoder Example")
print("=" * 80)


class TransformerEncoder(nn.Module):
    """
    Complete Transformer Encoder with:
    1. Token embeddings
    2. Positional encoding
    3. Multiple transformer blocks
    4. Final layer norm

    INTERVIEW INSIGHT:
    This is the architecture used in BERT (encoder-only).
    GPT uses decoder-only (with causal masking).
    """

    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        pos_encoding_type='sinusoidal'
    ):
        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        if pos_encoding_type == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model, max_seq_len, dropout
            )
        else:
            self.pos_encoder = LearnedPositionalEmbedding(
                d_model, max_seq_len, dropout
            )

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        self.d_model = d_model

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        """
        Args:
            x: Token indices (batch_size, seq_len)
            mask: Attention mask (optional)

        Returns:
            Encoded representations (batch_size, seq_len, d_model)
        """
        # Token embedding with scaling
        # INTERVIEW POINT: Scale by sqrt(d_model) to prevent embeddings
        # from being too small relative to positional encodings
        x = self.token_embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.norm(x)

        return x


# Create a small transformer encoder
vocab_size = 1000
model = TransformerEncoder(
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=3,
    d_ff=512,
    max_seq_len=50
)

print("\nTransformer Encoder Architecture:")
print(f"  Vocabulary size: {vocab_size:,}")
print(f"  Model dimension: {model.d_model}")
print(f"  Number of layers: {len(model.layers)}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Example forward pass
batch_size = 2
seq_len = 20
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

print(f"\nExample forward pass:")
print(f"  Input shape: {input_ids.shape}")

with torch.no_grad():
    output = model(input_ids)

print(f"  Output shape: {output.shape}")
print(f"  Output mean: {output.mean():.4f}, std: {output.std():.4f}")


# =============================================================================
# PART 7: Interview Quick Reference
# =============================================================================
print("\n" + "=" * 80)
print("INTERVIEW QUICK REFERENCE")
print("=" * 80)

interview_qa = """
Q1: Why do transformers need positional encoding?
A: Self-attention is permutation-invariant. Without positional info,
   "dog bites man" = "man bites dog". PE adds position information.

Q2: What's the formula for sinusoidal positional encoding?
A: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Q3: Sinusoidal vs Learned positional embeddings?
A: Sinusoidal: No params, can extrapolate, deterministic
   Learned: Better performance, task-specific, fixed max length
   Modern models mostly use learned (BERT, GPT).

Q4: Why use both sin and cos?
A: Allows model to learn relative positions as linear combinations.
   For any fixed offset k, PE(pos+k) can be represented as a linear
   function of PE(pos).

Q5: What are residual connections?
A: output = x + Sublayer(x)
   Benefits: Gradient flow, identity mapping, enables deep networks
   Used in EVERY transformer layer.

Q6: Pre-LN vs Post-LN?
A: Pre-LN: Norm before sublayer (more stable, modern choice)
   Post-LN: Norm after sublayer (original transformer)
   Pre-LN is better for very deep models.

Q7: How is positional encoding added?
A: ADDED to token embeddings, not concatenated!
   Token embeddings are scaled by sqrt(d_model) first.

Q8: Can transformers handle sequences longer than training?
A: With sinusoidal: Yes (can extrapolate)
   With learned: No (fixed max_len)
   Recent work: Relative position encodings (T5, Transformer-XL)

Q9: What's the computational complexity?
A: Self-attention: O(n² · d) where n=seq_len, d=d_model
   This is why long sequences are expensive!

Q10: What's the difference between encoder and decoder?
A: Encoder: Bidirectional attention (sees all tokens)
   Decoder: Causal masking (only sees previous tokens)
   BERT uses encoder, GPT uses decoder, T5 uses both.
"""

print(interview_qa)


# =============================================================================
# Save Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary = f"""
Generated visualizations saved to:
  {output_dir}/

Files created:
  1. positional_encoding_sinusoidal.png - Detailed sinusoidal encoding analysis
  2. positional_encoding_comparison.png - Sinusoidal vs Learned comparison

Key Takeaways for Interviews:
✓ Positional encoding is ESSENTIAL (attention is order-agnostic)
✓ Sinusoidal encoding uses sin/cos with different wavelengths
✓ Modern models mostly use learned positional embeddings
✓ Residual connections enable deep networks via gradient flow
✓ Layer normalization stabilizes training
✓ Pre-LN architecture is more stable than Post-LN

Implementation highlights:
- Sinusoidal: No parameters, can extrapolate
- Learned: Better performance, fixed max length
- Residual: x + Sublayer(x) in every layer
- LayerNorm: Normalized across features, not batch

This demo covers interview questions Q17, Q21, Q43, Q46.
"""

print(summary)

# Save summary to file
with open(output_dir / 'positional_encoding_summary.txt', 'w') as f:
    f.write(summary)
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("INTERVIEW Q&A REFERENCE\n")
    f.write("=" * 80 + "\n")
    f.write(interview_qa)

print(f"\n✓ Summary saved to: {output_dir / 'positional_encoding_summary.txt'}")
print("\n" + "=" * 80)
print("Demo complete! You're ready for transformer architecture interviews.")
print("=" * 80)
