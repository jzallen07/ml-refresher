"""
Comprehensive Attention Mechanisms Demo for LLM Interview Preparation
======================================================================

This file demonstrates key attention mechanism concepts for interview questions:
- Q2: Explain attention mechanisms
- Q22: How does multi-head attention work?
- Q23: Why is scaled dot-product attention important?
- Q24: What is causal masking?
- Q32: What are Query, Key, and Value in attention?

Author: Educational Demo
Purpose: LLM Interview Preparation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional


# =============================================================================
# SECTION 1: SCALED DOT-PRODUCT ATTENTION FROM SCRATCH
# =============================================================================

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements scaled dot-product attention from the "Attention is All You Need" paper.

    The attention mechanism allows the model to focus on different parts of the input
    when producing each element of the output.

    Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len_q, d_k)
        key: Key tensor of shape (batch, num_heads, seq_len_k, d_k)
        value: Value tensor of shape (batch, num_heads, seq_len_v, d_v)
        mask: Optional mask tensor to prevent attention to certain positions
        dropout: Optional dropout layer
        verbose: If True, print shape information

    Returns:
        output: Weighted sum of values, shape (batch, num_heads, seq_len_q, d_v)
        attention_weights: Attention weights, shape (batch, num_heads, seq_len_q, seq_len_k)

    Key Concepts (Interview Question 32):
    -------------------------------------
    - Query (Q): "What am I looking for?" - Represents the current position
    - Key (K): "What do I contain?" - Represents all positions that could be attended to
    - Value (V): "What information do I have?" - The actual content to retrieve

    The attention score measures how much the query "matches" each key.
    High scores mean the query finds that key relevant.
    """
    if verbose:
        print("\n" + "="*80)
        print("SCALED DOT-PRODUCT ATTENTION - STEP BY STEP")
        print("="*80)
        print(f"\nInput Shapes:")
        print(f"  Query (Q): {query.shape}  # (batch, heads, seq_len_q, d_k)")
        print(f"  Key (K):   {key.shape}  # (batch, heads, seq_len_k, d_k)")
        print(f"  Value (V): {value.shape}  # (batch, heads, seq_len_v, d_v)")

    # Get the dimension of the key (d_k) for scaling
    d_k = query.size(-1)

    if verbose:
        print(f"\nScaling Factor: sqrt(d_k) = sqrt({d_k}) = {np.sqrt(d_k):.4f}")
        print("  Why scale? To prevent dot products from growing too large,")
        print("  which would push softmax into regions with tiny gradients.")

    # Step 1: Compute attention scores (QK^T)
    # This measures the similarity between queries and keys
    scores = torch.matmul(query, key.transpose(-2, -1))

    if verbose:
        print(f"\nStep 1 - Attention Scores (QK^T):")
        print(f"  Shape: {scores.shape}  # (batch, heads, seq_len_q, seq_len_k)")
        print(f"  Formula: scores[i,j] = dot_product(query[i], key[j])")
        print(f"  Before scaling - Min: {scores.min():.4f}, Max: {scores.max():.4f}")

    # Step 2: Scale by sqrt(d_k) (Interview Question 23)
    # This is crucial! Without scaling, softmax can saturate for large d_k
    scores = scores / np.sqrt(d_k)

    if verbose:
        print(f"\nStep 2 - Scaled Scores (QK^T / sqrt(d_k)):")
        print(f"  After scaling - Min: {scores.min():.4f}, Max: {scores.max():.4f}")
        print(f"  Scaling prevents vanishing gradients in softmax")

    # Step 3: Apply mask if provided (Interview Question 24)
    # Masking is used for:
    # - Causal/autoregressive attention (can't see future tokens)
    # - Padding (ignore padded positions)
    if mask is not None:
        if verbose:
            print(f"\nStep 3 - Applying Mask:")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Setting masked positions to -inf (will become 0 after softmax)")
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Apply softmax to get attention weights
    # Softmax ensures weights sum to 1 and are non-negative
    attention_weights = F.softmax(scores, dim=-1)

    if verbose:
        print(f"\nStep 4 - Attention Weights (softmax over keys):")
        print(f"  Shape: {attention_weights.shape}")
        print(f"  Sum along last dim (should be ~1.0): {attention_weights.sum(dim=-1)[0, 0, 0]:.6f}")
        print(f"  Min: {attention_weights.min():.6f}, Max: {attention_weights.max():.6f}")
        print(f"  Each row represents how much attention a query pays to all keys")

    # Step 5: Apply dropout (during training)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Step 6: Weighted sum of values
    # This produces the final output by combining values according to attention weights
    output = torch.matmul(attention_weights, value)

    if verbose:
        print(f"\nStep 5 - Output (Attention_weights @ Value):")
        print(f"  Shape: {output.shape}  # (batch, heads, seq_len_q, d_v)")
        print(f"  Each output position is a weighted combination of all value vectors")
        print("="*80 + "\n")

    return output, attention_weights


# =============================================================================
# SECTION 2: MULTI-HEAD ATTENTION IMPLEMENTATION
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism (Interview Question 22).

    Instead of performing a single attention function, multi-head attention
    projects Q, K, V into multiple subspaces (heads) and performs attention
    in parallel, then concatenates the results.

    Benefits:
    1. Allows model to attend to information from different representation subspaces
    2. Each head can learn different attention patterns (e.g., syntax vs semantics)
    3. More expressive than single-head attention
    4. Prevents the model from focusing too much on a single aspect

    Formula:
        MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model (embedding dimension)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        # Instead of separate projections per head, we use one large projection
        # and split it into heads (more efficient)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor of shape (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()

        # Reshape: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back into a single tensor.

        Args:
            x: Tensor of shape (batch, num_heads, seq_len, d_k)

        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()

        # Transpose: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        x = x.transpose(1, 2).contiguous()

        # Reshape: (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional mask tensor
            verbose: If True, print detailed information

        Returns:
            output: Output tensor (batch, seq_len, d_model)
            attention_weights: Attention weights (batch, num_heads, seq_len, seq_len)
        """
        if verbose:
            print("\n" + "="*80)
            print("MULTI-HEAD ATTENTION - FORWARD PASS")
            print("="*80)
            print(f"\nConfiguration:")
            print(f"  Model dimension (d_model): {self.d_model}")
            print(f"  Number of heads: {self.num_heads}")
            print(f"  Dimension per head (d_k): {self.d_k}")

        batch_size = query.size(0)

        if verbose:
            print(f"\nInput shapes:")
            print(f"  Query: {query.shape}")
            print(f"  Key:   {key.shape}")
            print(f"  Value: {value.shape}")

        # Step 1: Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        if verbose:
            print(f"\nAfter linear projections:")
            print(f"  Q: {Q.shape}")
            print(f"  K: {K.shape}")
            print(f"  V: {V.shape}")

        # Step 2: Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        if verbose:
            print(f"\nAfter splitting into {self.num_heads} heads:")
            print(f"  Q: {Q.shape}  # (batch, num_heads, seq_len, d_k)")
            print(f"  K: {K.shape}")
            print(f"  V: {V.shape}")
            print(f"  Each head operates on a {self.d_k}-dimensional subspace")

        # Step 3: Scaled dot-product attention for all heads in parallel
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout, verbose=False
        )

        if verbose:
            print(f"\nAfter attention:")
            print(f"  Output: {attn_output.shape}")
            print(f"  Attention weights: {attention_weights.shape}")

        # Step 4: Concatenate heads
        attn_output = self.combine_heads(attn_output)

        if verbose:
            print(f"\nAfter combining heads:")
            print(f"  Output: {attn_output.shape}  # Back to (batch, seq_len, d_model)")

        # Step 5: Final linear projection
        output = self.W_o(attn_output)

        if verbose:
            print(f"\nFinal output after linear projection:")
            print(f"  Output: {output.shape}")
            print("="*80 + "\n")

        return output, attention_weights


# =============================================================================
# SECTION 3: CAUSAL MASKING FOR AUTOREGRESSIVE ATTENTION
# =============================================================================

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (lower triangular) mask for autoregressive attention.

    In autoregressive models (like GPT), each position can only attend to
    previous positions and itself, not future positions.

    Args:
        seq_len: Sequence length
        device: Device to create the mask on

    Returns:
        Causal mask of shape (seq_len, seq_len)
        1 indicates "can attend", 0 indicates "cannot attend"

    Example for seq_len=4:
        [[1, 0, 0, 0],   # Position 0 can only see position 0
         [1, 1, 0, 0],   # Position 1 can see positions 0,1
         [1, 1, 1, 0],   # Position 2 can see positions 0,1,2
         [1, 1, 1, 1]]   # Position 3 can see all positions
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def visualize_causal_mask(seq_len: int = 8, save_path: Optional[str] = None):
    """
    Visualize the causal mask structure.

    Args:
        seq_len: Sequence length
        save_path: Path to save the visualization
    """
    mask = create_causal_mask(seq_len).numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(mask, annot=True, fmt='.0f', cmap='Blues', cbar=True,
                xticklabels=range(seq_len), yticklabels=range(seq_len))
    plt.xlabel('Key Position (what we can attend to)')
    plt.ylabel('Query Position (current token)')
    plt.title('Causal Mask for Autoregressive Attention\n'
              '(1 = can attend, 0 = cannot attend)')

    # Add explanation text
    plt.text(seq_len/2, -1.5,
             'Each row shows what a token can attend to.\n'
             'Token at position i can only attend to positions 0 to i (past and present).',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Causal mask visualization saved to: {save_path}")
    plt.close()


# =============================================================================
# SECTION 4: ATTENTION VISUALIZATION
# =============================================================================

def visualize_attention_weights(
    attention_weights: torch.Tensor,
    tokens: list,
    head_idx: int = 0,
    save_path: Optional[str] = None,
    title: str = "Attention Weights"
):
    """
    Visualize attention weights as a heatmap.

    Args:
        attention_weights: Attention weights (batch, num_heads, seq_len, seq_len)
                          or (num_heads, seq_len, seq_len) or (seq_len, seq_len)
        tokens: List of token strings for labels
        head_idx: Which attention head to visualize
        save_path: Path to save the visualization
        title: Title for the plot
    """
    # Extract weights for the specified head
    if attention_weights.dim() == 4:  # (batch, num_heads, seq_len, seq_len)
        weights = attention_weights[0, head_idx].detach().cpu().numpy()
    elif attention_weights.dim() == 3:  # (num_heads, seq_len, seq_len)
        weights = attention_weights[head_idx].detach().cpu().numpy()
    else:  # (seq_len, seq_len)
        weights = attention_weights.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(weights, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=tokens, yticklabels=tokens, cbar=True)
    plt.xlabel('Key (attending to)', fontsize=12)
    plt.ylabel('Query (current token)', fontsize=12)
    plt.title(f'{title} (Head {head_idx})', fontsize=14, fontweight='bold')

    # Add explanation
    plt.text(len(tokens)/2, -1.5,
             'Each cell (i,j) shows how much token i attends to token j.\n'
             'Higher values (darker red) = more attention.',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention weights visualization saved to: {save_path}")
    plt.close()


def visualize_all_heads(
    attention_weights: torch.Tensor,
    tokens: list,
    save_path: Optional[str] = None
):
    """
    Visualize attention patterns for all heads in a grid.

    Args:
        attention_weights: Attention weights (batch, num_heads, seq_len, seq_len)
        tokens: List of token strings for labels
        save_path: Path to save the visualization
    """
    if attention_weights.dim() == 4:
        weights = attention_weights[0].detach().cpu().numpy()
    else:
        weights = attention_weights.detach().cpu().numpy()

    num_heads = weights.shape[0]
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        sns.heatmap(weights[head_idx], annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=tokens, yticklabels=tokens, ax=ax, cbar=True)
        ax.set_title(f'Head {head_idx}', fontweight='bold')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')

    # Hide empty subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Multi-Head Attention Patterns\n'
                 'Different heads learn different attention patterns',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multi-head visualization saved to: {save_path}")
    plt.close()


# =============================================================================
# SECTION 5: SOFTMAX NORMALIZATION DEMONSTRATION
# =============================================================================

def demonstrate_softmax_effect():
    """
    Demonstrate the effect of softmax normalization on attention scores.

    This shows why softmax is important for attention:
    1. Converts unbounded scores to probabilities (0 to 1)
    2. Ensures scores sum to 1 (valid probability distribution)
    3. Amplifies differences (high scores get higher, low scores get lower)
    """
    print("\n" + "="*80)
    print("SOFTMAX NORMALIZATION EFFECT ON ATTENTION SCORES")
    print("="*80)

    # Example attention scores (before softmax)
    scores = torch.tensor([2.0, 1.0, 0.5, -0.5, -1.0])

    print("\nOriginal Attention Scores (before softmax):")
    print(f"  Scores: {scores.tolist()}")
    print(f"  Sum: {scores.sum().item():.4f} (not normalized)")
    print(f"  Min: {scores.min().item():.4f}, Max: {scores.max().item():.4f}")

    # Apply softmax
    attention_probs = F.softmax(scores, dim=0)

    print("\nAfter Softmax Normalization:")
    print(f"  Probabilities: {[f'{p:.4f}' for p in attention_probs.tolist()]}")
    print(f"  Sum: {attention_probs.sum().item():.6f} (should be 1.0)")
    print(f"  Min: {attention_probs.min().item():.6f}, Max: {attention_probs.max().item():.6f}")

    # Show how different temperatures affect attention
    print("\n" + "-"*80)
    print("EFFECT OF SCALING (Temperature) ON SOFTMAX")
    print("-"*80)

    temperatures = [0.1, 0.5, 1.0, 2.0, 10.0]

    for temp in temperatures:
        scaled_scores = scores / temp
        probs = F.softmax(scaled_scores, dim=0)

        print(f"\nTemperature = {temp}:")
        print(f"  Scaled scores: {[f'{s:.2f}' for s in scaled_scores.tolist()]}")
        print(f"  Probabilities: {[f'{p:.4f}' for p in probs.tolist()]}")
        print(f"  Entropy: {-(probs * torch.log(probs + 1e-10)).sum().item():.4f}")
        print(f"  Effect: {'Sharper (peaked)' if temp < 1.0 else 'Smoother (uniform)' if temp > 1.0 else 'Standard'}")

    print("\nKey Insight:")
    print("  - Lower temperature (< 1) → Sharper attention (focus on top items)")
    print("  - Higher temperature (> 1) → Smoother attention (more uniform)")
    print("  - sqrt(d_k) scaling in attention acts as temperature control")
    print("="*80 + "\n")


# =============================================================================
# SECTION 6: EXAMPLE DEMONSTRATIONS
# =============================================================================

def example_1_basic_attention():
    """
    Example 1: Basic scaled dot-product attention with a simple sentence.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC ATTENTION MECHANISM")
    print("="*80)

    # Simple example with 5 tokens
    batch_size = 1
    num_heads = 1
    seq_len = 5
    d_k = 64

    print(f"\nScenario: Processing a simple sentence with {seq_len} tokens")
    print(f"Model dimension per head: {d_k}")

    # Create random Q, K, V tensors (in practice, these come from embeddings)
    torch.manual_seed(42)
    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_k)

    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V, verbose=True)

    # Visualize
    tokens = ['The', 'cat', 'sat', 'on', 'mat']
    viz_path = Path('/Users/zack/dev/ml-refresher/data/interview_viz')
    viz_path.mkdir(parents=True, exist_ok=True)

    visualize_attention_weights(
        attention_weights,
        tokens,
        save_path=str(viz_path / 'attention_basic.png'),
        title='Basic Attention Pattern'
    )


def example_2_multi_head_attention():
    """
    Example 2: Multi-head attention with multiple heads learning different patterns.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: MULTI-HEAD ATTENTION")
    print("="*80)

    # Configuration
    batch_size = 1
    seq_len = 6
    d_model = 512
    num_heads = 8

    print(f"\nScenario: Multi-head attention with {num_heads} heads")
    print(f"Model dimension: {d_model}")
    print(f"Dimension per head: {d_model // num_heads}")

    # Create model
    mha = MultiHeadAttention(d_model, num_heads)

    # Create input (in practice, these are embeddings)
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass (self-attention: Q=K=V)
    output, attention_weights = mha(x, x, x, verbose=True)

    # Visualize
    tokens = ['I', 'love', 'machine', 'learning', 'models', '.']
    viz_path = Path('/Users/zack/dev/ml-refresher/data/interview_viz')

    # Visualize individual head
    visualize_attention_weights(
        attention_weights,
        tokens,
        head_idx=0,
        save_path=str(viz_path / 'attention_multihead_head0.png'),
        title='Multi-Head Attention'
    )

    # Visualize all heads
    visualize_all_heads(
        attention_weights,
        tokens,
        save_path=str(viz_path / 'attention_all_heads.png')
    )


def example_3_causal_attention():
    """
    Example 3: Causal attention with masking (like in GPT).
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: CAUSAL (AUTOREGRESSIVE) ATTENTION")
    print("="*80)

    seq_len = 7
    batch_size = 1
    num_heads = 4
    d_model = 256

    print(f"\nScenario: Autoregressive language model (like GPT)")
    print(f"Sequence length: {seq_len}")
    print(f"Key property: Each token can only attend to past and current tokens")
    print(f"Purpose: Ensures the model can't 'cheat' by looking at future tokens")

    # Create causal mask
    causal_mask = create_causal_mask(seq_len)
    print(f"\nCausal mask shape: {causal_mask.shape}")
    print("Causal mask (1=attend, 0=mask):")
    print(causal_mask.numpy().astype(int))

    # Reshape mask for multi-head attention
    # Shape: (1, 1, seq_len, seq_len) for broadcasting
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # Create model and input
    torch.manual_seed(42)
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass with causal mask
    output, attention_weights = mha(x, x, x, mask=causal_mask, verbose=False)

    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Visualize
    tokens = ['Once', 'upon', 'a', 'time', 'there', 'was', '...']
    viz_path = Path('/Users/zack/dev/ml-refresher/data/interview_viz')

    # Visualize causal mask
    visualize_causal_mask(
        seq_len,
        save_path=str(viz_path / 'causal_mask.png')
    )

    # Visualize attention with mask
    visualize_attention_weights(
        attention_weights,
        tokens,
        head_idx=0,
        save_path=str(viz_path / 'attention_causal.png'),
        title='Causal Attention Pattern'
    )

    # Show how masking affects attention
    print("\nEffect of Causal Masking:")
    print("Notice in the visualization:")
    print("  - Lower triangular pattern (tokens can only see past)")
    print("  - No attention to future positions (zeros above diagonal)")
    print("  - Essential for autoregressive generation")


def example_4_attention_patterns():
    """
    Example 4: Demonstrate different attention patterns that emerge.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: DIFFERENT ATTENTION PATTERNS")
    print("="*80)

    print("\nIn real transformers, different heads learn different patterns:")
    print("  - Syntactic patterns (e.g., attending to related words)")
    print("  - Positional patterns (e.g., attending to adjacent tokens)")
    print("  - Semantic patterns (e.g., attending to similar meanings)")

    seq_len = 8
    tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']

    # Create different attention patterns manually for illustration

    # Pattern 1: Uniform attention (attends equally to all tokens)
    uniform = torch.ones(seq_len, seq_len) / seq_len

    # Pattern 2: Local attention (attends to nearby tokens)
    local = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(max(0, i-2), min(seq_len, i+3)):
            local[i, j] = 1.0
    local = local / local.sum(dim=1, keepdim=True)

    # Pattern 3: Self attention (each token attends mostly to itself)
    self_attn = torch.eye(seq_len) * 0.7
    self_attn = self_attn + (1 - torch.eye(seq_len)) * 0.3 / (seq_len - 1)

    # Pattern 4: Focused attention (attends to specific important tokens)
    focused = torch.zeros(seq_len, seq_len)
    focused[:, 3] = 0.6  # All tokens focus on position 3 (fox)
    focused[:, 4] = 0.3  # And position 4 (jumps)
    for i in range(seq_len):
        focused[i, i] += 0.1  # Small self-attention

    patterns = [
        (uniform, "Uniform Attention"),
        (local, "Local/Positional Attention"),
        (self_attn, "Self Attention"),
        (focused, "Focused Attention")
    ]

    viz_path = Path('/Users/zack/dev/ml-refresher/data/interview_viz')

    for idx, (pattern, name) in enumerate(patterns):
        print(f"\n{name}:")
        print(f"  Pattern shape: {pattern.shape}")
        print(f"  Row sums (should be 1.0): {pattern.sum(dim=1)[0]:.6f}")

        visualize_attention_weights(
            pattern,
            tokens,
            save_path=str(viz_path / f'attention_pattern_{idx}.png'),
            title=name
        )


# =============================================================================
# SECTION 7: INTERVIEW KEY POINTS SUMMARY
# =============================================================================

def print_interview_summary():
    """
    Print a summary of key points for interview questions.
    """
    print("\n" + "="*80)
    print("KEY INTERVIEW POINTS SUMMARY")
    print("="*80)

    summary = """
Q2: EXPLAIN ATTENTION MECHANISMS
---------------------------------
✓ Attention allows models to focus on different parts of input when producing output
✓ Core idea: weighted combination of values based on query-key similarity
✓ Formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
✓ Benefits: handles variable-length sequences, captures long-range dependencies

Q22: HOW DOES MULTI-HEAD ATTENTION WORK?
----------------------------------------
✓ Performs attention multiple times in parallel with different learned projections
✓ Each head learns to attend to different aspects of the input
✓ Formula: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
✓ Benefits: richer representations, different attention patterns, more expressive

Q23: WHY IS SCALED DOT-PRODUCT ATTENTION IMPORTANT?
---------------------------------------------------
✓ Scaling by sqrt(d_k) prevents dot products from growing too large
✓ Without scaling, softmax enters regions with tiny gradients (vanishing gradients)
✓ Larger d_k → larger dot products → more peaked softmax → worse gradients
✓ Scaling keeps gradients healthy and training stable

Q24: WHAT IS CAUSAL MASKING?
-----------------------------
✓ Prevents attention to future positions in autoregressive models
✓ Implemented by setting future positions to -inf before softmax
✓ Creates lower triangular attention pattern
✓ Essential for models like GPT that generate text left-to-right
✓ Ensures model doesn't "cheat" during training

Q32: WHAT ARE QUERY, KEY, VALUE?
---------------------------------
✓ Query (Q): "What am I looking for?" - represents current position
✓ Key (K): "What do I contain?" - represents positions that could be attended to
✓ Value (V): "What information do I have?" - actual content to retrieve
✓ Attention score = similarity between query and key
✓ Output = weighted sum of values based on attention scores

COMMON FOLLOW-UP QUESTIONS:
---------------------------
• Why use softmax? → Ensures weights are non-negative and sum to 1
• Self-attention vs cross-attention? → Self: Q=K=V, Cross: K,V from different source
• Computational complexity? → O(n²d) where n=seq_len, d=model_dim
• How to reduce complexity? → Sparse attention, linear attention, etc.
"""

    print(summary)
    print("="*80 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run all demonstrations.
    """
    print("\n" + "="*80)
    print("ATTENTION MECHANISMS - COMPREHENSIVE DEMO FOR LLM INTERVIEWS")
    print("="*80)
    print("\nThis demo covers:")
    print("  1. Scaled dot-product attention from scratch")
    print("  2. Multi-head attention implementation")
    print("  3. Causal masking for autoregressive models")
    print("  4. Attention visualization and patterns")
    print("  5. Softmax normalization effects")
    print("\nVisualizations will be saved to: /Users/zack/dev/ml-refresher/data/interview_viz/")
    print("="*80)

    # Run all examples
    example_1_basic_attention()
    example_2_multi_head_attention()
    example_3_causal_attention()
    example_4_attention_patterns()

    # Demonstrate softmax effect
    demonstrate_softmax_effect()

    # Print interview summary
    print_interview_summary()

    print("\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nNext steps for interview prep:")
    print("  1. Review the visualizations in data/interview_viz/")
    print("  2. Understand each step in the scaled_dot_product_attention function")
    print("  3. Be able to explain Q, K, V intuitively")
    print("  4. Practice explaining why scaling by sqrt(d_k) matters")
    print("  5. Understand when and why to use causal masking")
    print("\nGood luck with your interviews! 🚀")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
