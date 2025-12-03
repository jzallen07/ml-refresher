"""
Gradient Flow and Optimization Demo for LLM Interviews
=======================================================

This demo covers key concepts for interview questions:
- Q26: How do embeddings handle gradient flow with sparse updates?
- Q27: How do gradients flow through transformer layers?
- Q48: Hyperparameter sensitivity in LLM training

Key Concepts Demonstrated:
1. Embedding gradient sparsity and updates
2. Gradient flow through neural networks
3. Learning rate effects on convergence
4. Hyperparameter sensitivity
5. Gradient clipping for stability

Author: Educational Demo
Date: 2025-12-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
VIZ_DIR = Path("/Users/zack/dev/ml-refresher/data/interview_viz")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GRADIENT FLOW AND OPTIMIZATION DEMO FOR LLM INTERVIEWS")
print("=" * 80)
print()


# ============================================================================
# PART 1: EMBEDDING GRADIENT SPARSITY (Q26)
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: EMBEDDING GRADIENT VISUALIZATION - SPARSE UPDATES")
print("=" * 80)
print("""
Interview Talking Points:
- Embeddings only update for tokens present in the batch
- This creates sparse gradient updates (most embedding rows get zero gradient)
- Rare tokens update less frequently than common tokens
- This is memory-efficient: only active embeddings need gradient computation
""")

def demonstrate_embedding_gradients():
    """
    Show how embedding gradients are sparse - only active tokens get updates.

    Key Interview Point: In a batch, only the embeddings corresponding to
    tokens actually present receive gradient updates. This is why embedding
    layers are memory-efficient during training.
    """
    print("\n--- Setting up Embedding Layer ---")

    # Create embedding layer
    vocab_size = 1000
    embed_dim = 128
    embedding = nn.Embedding(vocab_size, embed_dim)

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Embedding Dimension: {embed_dim}")
    print(f"Total Parameters: {vocab_size * embed_dim:,}")

    # Sample batch with only a few unique tokens
    batch_size = 8
    seq_length = 10

    # Use only tokens 5, 10, 15, 20 (4 unique tokens out of 1000)
    input_tokens = torch.tensor([
        [5, 10, 15, 20, 5, 10, 15, 20, 5, 10],
        [10, 15, 20, 5, 10, 15, 20, 5, 10, 15],
        [15, 20, 5, 10, 15, 20, 5, 10, 15, 20],
        [20, 5, 10, 15, 20, 5, 10, 15, 20, 5],
        [5, 10, 15, 20, 5, 10, 15, 20, 5, 10],
        [10, 15, 20, 5, 10, 15, 20, 5, 10, 15],
        [15, 20, 5, 10, 15, 20, 5, 10, 15, 20],
        [20, 5, 10, 15, 20, 5, 10, 15, 20, 5],
    ])

    unique_tokens = torch.unique(input_tokens)
    print(f"\nBatch shape: {input_tokens.shape}")
    print(f"Unique tokens in batch: {unique_tokens.tolist()}")
    print(f"Token frequency in batch:")
    for token in unique_tokens:
        count = (input_tokens == token).sum().item()
        print(f"  Token {token}: {count} occurrences")

    # Forward pass
    embedded = embedding(input_tokens)

    # Create a simple loss (mean of embeddings)
    loss = embedded.mean()

    # Backward pass to compute gradients
    loss.backward()

    # Analyze gradient sparsity
    print("\n--- Analyzing Gradient Sparsity ---")
    grad = embedding.weight.grad

    # Count how many embedding rows have non-zero gradients
    non_zero_rows = (grad.abs().sum(dim=1) > 0).sum().item()
    zero_rows = vocab_size - non_zero_rows

    print(f"Embeddings with gradients: {non_zero_rows}/{vocab_size}")
    print(f"Embeddings with zero gradients: {zero_rows}/{vocab_size}")
    print(f"Sparsity: {100 * zero_rows / vocab_size:.2f}%")

    print("\nGradient norms for active tokens:")
    for token in unique_tokens:
        grad_norm = grad[token].norm().item()
        print(f"  Token {token}: gradient norm = {grad_norm:.6f}")

    print("\nGradient norms for inactive tokens (should be zero):")
    inactive_tokens = [0, 1, 2, 100, 500]
    for token in inactive_tokens:
        grad_norm = grad[token].norm().item()
        print(f"  Token {token}: gradient norm = {grad_norm:.6f}")

    # Visualize gradient sparsity
    print("\n--- Creating Visualization ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Gradient norm per embedding row
    ax = axes[0, 0]
    grad_norms = grad.norm(dim=1).detach().numpy()
    ax.plot(grad_norms, alpha=0.7, linewidth=0.5)
    ax.scatter(unique_tokens.numpy(), grad_norms[unique_tokens.numpy()],
               color='red', s=100, zorder=5, label='Active tokens')
    ax.set_xlabel('Embedding Index')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Sparsity: Only Active Tokens Updated')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Gradient distribution (log scale)
    ax = axes[0, 1]
    non_zero_grads = grad_norms[grad_norms > 0]
    ax.hist(non_zero_grads, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Gradient Norm')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Non-Zero Gradients (n={len(non_zero_grads)})')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    # Plot 3: Heatmap of gradients for active tokens
    ax = axes[1, 0]
    active_grads = grad[unique_tokens].detach().numpy()
    im = ax.imshow(active_grads, aspect='auto', cmap='RdBu_r',
                   vmin=-active_grads.std()*3, vmax=active_grads.std()*3)
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Active Token ID')
    ax.set_yticks(range(len(unique_tokens)))
    ax.set_yticklabels([f'Token {t}' for t in unique_tokens.tolist()])
    ax.set_title('Gradient Values for Active Tokens')
    plt.colorbar(im, ax=ax, label='Gradient Value')

    # Plot 4: Sparsity statistics
    ax = axes[1, 1]
    categories = ['Active\nEmbeddings', 'Zero-Gradient\nEmbeddings']
    values = [non_zero_rows, zero_rows]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Embedding Gradient Sparsity')

    # Add percentage labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({100*val/vocab_size:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'embedding_gradient_sparsity.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VIZ_DIR / 'embedding_gradient_sparsity.png'}")
    plt.close()

    return grad_norms, unique_tokens

grad_norms, active_tokens = demonstrate_embedding_gradients()


# ============================================================================
# PART 2: GRADIENT FLOW THROUGH NETWORK LAYERS (Q27)
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: GRADIENT FLOW THROUGH NEURAL NETWORK")
print("=" * 80)
print("""
Interview Talking Points:
- Gradients can vanish (shrink) or explode (grow) through layers
- Deep networks need careful initialization and normalization
- Monitor gradient magnitudes at each layer during training
- Residual connections help maintain gradient flow
""")

class SimpleTransformerBlock(nn.Module):
    """
    Simplified transformer block for gradient flow analysis.

    Interview Point: Transformers use residual connections and layer norm
    to maintain healthy gradient flow through many layers.
    """
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x

def demonstrate_gradient_flow():
    """
    Track gradient magnitudes as they flow backward through layers.

    Key Interview Point: In well-designed networks (like transformers),
    gradients should have similar magnitudes across layers. Large variations
    indicate potential training problems.
    """
    print("\n--- Building Multi-Layer Network ---")

    # Create a simple deep network
    d_model = 64
    num_layers = 6
    batch_size = 4
    seq_length = 8

    # Build stacked transformer blocks
    layers = [SimpleTransformerBlock(d_model) for _ in range(num_layers)]
    model = nn.Sequential(*layers)

    print(f"Model depth: {num_layers} transformer blocks")
    print(f"Hidden dimension: {d_model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create input
    x = torch.randn(batch_size, seq_length, d_model)
    print(f"Input shape: {x.shape}")

    # Forward pass
    print("\n--- Forward Pass ---")
    activations = [x]
    current = x

    for i, layer in enumerate(layers):
        current = layer(current)
        activations.append(current)
        print(f"Layer {i+1} output: mean={current.mean():.4f}, std={current.std():.4f}")

    # Compute loss
    loss = current.mean()
    print(f"\nLoss: {loss.item():.6f}")

    # Backward pass
    print("\n--- Backward Pass - Gradient Flow ---")
    loss.backward()

    # Collect gradient statistics for each layer
    gradient_stats = []

    for i, layer in enumerate(layers):
        layer_grads = []
        for name, param in layer.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                layer_grads.append(grad_norm)
                print(f"Layer {i+1} - {name}: norm={grad_norm:.6f}, "
                      f"mean={grad_mean:.6e}, std={grad_std:.6e}")

        avg_grad_norm = np.mean(layer_grads)
        gradient_stats.append({
            'layer': i + 1,
            'avg_norm': avg_grad_norm,
            'min_norm': min(layer_grads),
            'max_norm': max(layer_grads)
        })

    print("\n--- Gradient Flow Summary ---")
    print(f"{'Layer':<10} {'Avg Grad Norm':<15} {'Min':<15} {'Max':<15}")
    print("-" * 60)
    for stat in gradient_stats:
        print(f"{stat['layer']:<10} {stat['avg_norm']:<15.6f} "
              f"{stat['min_norm']:<15.6f} {stat['max_norm']:<15.6f}")

    # Visualize gradient flow
    print("\n--- Creating Gradient Flow Visualization ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Gradient norms by layer
    ax = axes[0, 0]
    layers_idx = [s['layer'] for s in gradient_stats]
    avg_norms = [s['avg_norm'] for s in gradient_stats]
    min_norms = [s['min_norm'] for s in gradient_stats]
    max_norms = [s['max_norm'] for s in gradient_stats]

    ax.plot(layers_idx, avg_norms, 'o-', linewidth=2, markersize=8, label='Average')
    ax.fill_between(layers_idx, min_norms, max_norms, alpha=0.3, label='Min-Max Range')
    ax.set_xlabel('Layer Number (1=earliest, 6=latest)')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Flow Through Layers')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(layers_idx)

    # Plot 2: Gradient norm ratios (detect vanishing/exploding)
    ax = axes[0, 1]
    if len(avg_norms) > 1:
        ratios = [avg_norms[i] / avg_norms[i+1] for i in range(len(avg_norms)-1)]
        ax.plot(range(1, len(ratios)+1), ratios, 'o-', linewidth=2, markersize=8, color='orange')
        ax.axhline(y=1.0, color='red', linestyle='--', label='Ratio = 1 (ideal)')
        ax.fill_between(range(1, len(ratios)+1), 0.5, 2.0, alpha=0.2, color='green',
                        label='Healthy range (0.5-2.0)')
        ax.set_xlabel('Layer Transition')
        ax.set_ylabel('Gradient Norm Ratio (layer_i / layer_i+1)')
        ax.set_title('Gradient Stability Across Layers')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_yscale('log')

    # Plot 3: Activation statistics through layers
    ax = axes[1, 0]
    activation_means = [a.mean().item() for a in activations]
    activation_stds = [a.std().item() for a in activations]

    x_pos = range(len(activations))
    ax.plot(x_pos, activation_means, 'o-', label='Mean', linewidth=2, markersize=8)
    ax.plot(x_pos, activation_stds, 's-', label='Std Dev', linewidth=2, markersize=8)
    ax.set_xlabel('Layer (0=input)')
    ax.set_ylabel('Value')
    ax.set_title('Activation Statistics Through Network')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(x_pos)

    # Plot 4: Gradient distribution (all parameters)
    ax = axes[1, 1]
    all_grads = []
    for layer in layers:
        for param in layer.parameters():
            if param.grad is not None:
                all_grads.extend(param.grad.flatten().detach().numpy())

    all_grads = np.array(all_grads)
    ax.hist(all_grads, bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Gradient Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Gradient Distribution (n={len(all_grads):,})')
    ax.set_yscale('log')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'gradient_flow_through_layers.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VIZ_DIR / 'gradient_flow_through_layers.png'}")
    plt.close()

    return gradient_stats

gradient_stats = demonstrate_gradient_flow()


# ============================================================================
# PART 3: LEARNING RATE EXPERIMENTS (Q48)
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: LEARNING RATE SENSITIVITY ANALYSIS")
print("=" * 80)
print("""
Interview Talking Points:
- Learning rate is the most important hyperparameter
- Too high: training unstable, loss diverges
- Too low: training too slow, may get stuck in local minima
- LLMs typically use learning rate warmup and decay schedules
- Common starting point: 3e-4 to 1e-3 for Adam optimizer
""")

def train_with_learning_rate(lr: float, num_steps: int = 100) -> Tuple[List[float], List[float]]:
    """
    Train a simple model with a specific learning rate and track metrics.

    Interview Point: Demonstrates how learning rate affects convergence
    speed and stability.
    """
    # Simple 2-layer network for a toy task
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Simple regression task: learn to predict sum of inputs
    losses = []
    grad_norms = []

    for step in range(num_steps):
        # Generate random data
        x = torch.randn(32, 10)
        y = x.sum(dim=1, keepdim=True)

        # Forward pass
        pred = model(x)
        loss = F.mse_loss(pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Track gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(total_norm)

    return losses, grad_norms

def demonstrate_learning_rate_effects():
    """
    Compare training with different learning rates.

    Key Interview Point: Learning rate selection significantly impacts
    training dynamics. LLMs use sophisticated schedules (warmup, cosine decay)
    to balance fast convergence and stability.
    """
    print("\n--- Testing Multiple Learning Rates ---")

    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    results = {}

    for lr in learning_rates:
        print(f"\nTraining with LR={lr:.0e}")
        losses, grad_norms = train_with_learning_rate(lr, num_steps=100)

        final_loss = losses[-1]
        min_loss = min(losses)
        avg_grad = np.mean(grad_norms)

        print(f"  Final loss: {final_loss:.6f}")
        print(f"  Min loss: {min_loss:.6f}")
        print(f"  Avg gradient norm: {avg_grad:.6f}")

        # Check for divergence
        if np.isnan(final_loss) or final_loss > 100:
            print(f"  ⚠️  DIVERGED - Learning rate too high!")
        elif final_loss < 0.01:
            print(f"  ✓ CONVERGED - Good learning rate")
        else:
            print(f"  ⚠️  SLOW - Learning rate might be too low")

        results[lr] = {
            'losses': losses,
            'grad_norms': grad_norms,
            'final_loss': final_loss,
            'min_loss': min_loss
        }

    # Visualize learning rate comparison
    print("\n--- Creating Learning Rate Comparison ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curves for all learning rates
    ax = axes[0, 0]
    for lr, data in results.items():
        losses = data['losses']
        # Clip extreme values for visualization
        losses_clipped = np.clip(losses, 0, 10)
        ax.plot(losses_clipped, label=f'LR={lr:.0e}', linewidth=2, alpha=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss (clipped at 10)')
    ax.set_title('Loss Curves: Learning Rate Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Final loss vs learning rate
    ax = axes[0, 1]
    lrs = list(results.keys())
    final_losses = [results[lr]['final_loss'] for lr in lrs]
    final_losses_clipped = np.clip(final_losses, 1e-6, 100)

    ax.plot(lrs, final_losses_clipped, 'o-', linewidth=2, markersize=10, color='purple')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Final Loss')
    ax.set_title('Final Loss vs Learning Rate')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    # Mark the optimal LR
    best_lr = min(results.keys(), key=lambda lr: results[lr]['final_loss'])
    best_loss = results[best_lr]['final_loss']
    ax.scatter([best_lr], [best_loss], color='red', s=200, zorder=5,
               marker='*', label=f'Best: {best_lr:.0e}')
    ax.legend()

    # Plot 3: Gradient norms
    ax = axes[1, 0]
    for lr, data in results.items():
        grad_norms = data['grad_norms']
        ax.plot(grad_norms, label=f'LR={lr:.0e}', linewidth=2, alpha=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms During Training')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # Plot 4: Learning rate recommendations
    ax = axes[1, 1]
    ax.axis('off')

    recommendations = """
Learning Rate Guidelines for LLMs:

1. TYPICAL RANGES:
   • Small models: 1e-3 to 1e-4
   • Large models: 1e-4 to 3e-5
   • Fine-tuning: 1e-5 to 1e-6

2. WARMUP STRATEGY:
   • Start with small LR (e.g., 1e-6)
   • Linearly increase to max LR
   • Typical warmup: 2-10% of total steps

3. DECAY SCHEDULE:
   • Cosine decay (most common)
   • Linear decay
   • Step decay

4. WARNING SIGNS:
   • Loss spikes: LR too high
   • No improvement: LR too low
   • NaN/Inf: Definitely too high!

5. ADAPTIVE OPTIMIZERS:
   • Adam: Most common for LLMs
   • AdamW: Adam with weight decay
   • Learning rate still crucial!
"""

    ax.text(0.05, 0.95, recommendations, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VIZ_DIR / 'learning_rate_comparison.png'}")
    plt.close()

    return results

lr_results = demonstrate_learning_rate_effects()


# ============================================================================
# PART 4: HYPERPARAMETER SENSITIVITY ANALYSIS (Q48)
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: HYPERPARAMETER SENSITIVITY ANALYSIS")
print("=" * 80)
print("""
Interview Talking Points:
- Different hyperparameters have different sensitivity levels
- Most sensitive: learning rate, batch size, model size
- Moderately sensitive: warmup steps, weight decay, dropout
- Less sensitive: optimizer choice (within Adam family)
- Always validate hyperparameters on a smaller dataset first
""")

def train_with_hyperparameters(
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 0.01,
    dropout: float = 0.1,
    num_steps: int = 50
) -> Dict[str, List[float]]:
    """
    Train with specific hyperparameters and return training metrics.

    Interview Point: Shows how to systematically test hyperparameter sensitivity.
    """
    # Model with dropout
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []

    for step in range(num_steps):
        # Generate batch
        x = torch.randn(batch_size, 10)
        y = x.sum(dim=1, keepdim=True)

        # Training step
        pred = model(x)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return {'losses': losses, 'final_loss': losses[-1]}

def demonstrate_hyperparameter_sensitivity():
    """
    Systematically vary hyperparameters to show their impact.

    Key Interview Point: Understanding which hyperparameters matter most
    helps prioritize tuning efforts and debug training issues.
    """
    print("\n--- Testing Hyperparameter Sensitivity ---")

    # Baseline
    baseline_config = {
        'lr': 1e-3,
        'batch_size': 32,
        'weight_decay': 0.01,
        'dropout': 0.1
    }

    print("\nBaseline configuration:")
    for k, v in baseline_config.items():
        print(f"  {k}: {v}")

    # Test variations of each hyperparameter
    experiments = {
        'learning_rate': {
            'param': 'lr',
            'values': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
            'results': []
        },
        'batch_size': {
            'param': 'batch_size',
            'values': [8, 16, 32, 64, 128],
            'results': []
        },
        'weight_decay': {
            'param': 'weight_decay',
            'values': [0.0, 0.001, 0.01, 0.1, 0.5],
            'results': []
        },
        'dropout': {
            'param': 'dropout',
            'values': [0.0, 0.05, 0.1, 0.2, 0.3],
            'results': []
        }
    }

    # Run experiments
    for exp_name, exp_config in experiments.items():
        print(f"\n--- Testing {exp_name} sensitivity ---")
        param_name = exp_config['param']

        for value in exp_config['values']:
            config = baseline_config.copy()
            config[param_name] = value

            print(f"  {param_name}={value}...", end=" ")
            result = train_with_hyperparameters(**config, num_steps=50)
            exp_config['results'].append(result['final_loss'])
            print(f"final_loss={result['final_loss']:.6f}")

    # Visualize sensitivity
    print("\n--- Creating Hyperparameter Sensitivity Visualization ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    exp_names = list(experiments.keys())

    for idx, (exp_name, ax) in enumerate(zip(exp_names, axes.flat)):
        exp_config = experiments[exp_name]
        values = exp_config['values']
        results = exp_config['results']

        # Plot
        ax.plot(range(len(values)), results, 'o-', linewidth=2, markersize=10)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([str(v) for v in values], rotation=45)
        ax.set_xlabel(exp_name.replace('_', ' ').title())
        ax.set_ylabel('Final Loss')
        ax.set_title(f'Sensitivity to {exp_name.replace("_", " ").title()}')
        ax.grid(alpha=0.3)

        # Mark baseline if present
        param_name = exp_config['param']
        baseline_val = baseline_config[param_name]
        if baseline_val in values:
            baseline_idx = values.index(baseline_val)
            ax.scatter([baseline_idx], [results[baseline_idx]],
                      color='red', s=200, zorder=5, marker='*',
                      label='Baseline')
            ax.legend()

        # Calculate sensitivity score (coefficient of variation)
        cv = np.std(results) / np.mean(results) * 100
        ax.text(0.02, 0.98, f'CV: {cv:.1f}%', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VIZ_DIR / 'hyperparameter_sensitivity.png'}")
    plt.close()

    # Print sensitivity ranking
    print("\n--- Hyperparameter Sensitivity Ranking ---")
    sensitivities = {}
    for exp_name, exp_config in experiments.items():
        results = exp_config['results']
        cv = np.std(results) / np.mean(results) * 100
        sensitivities[exp_name] = cv

    ranked = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    print("\nRanked by sensitivity (coefficient of variation):")
    for rank, (name, cv) in enumerate(ranked, 1):
        sensitivity_level = "HIGH" if cv > 50 else "MEDIUM" if cv > 20 else "LOW"
        print(f"{rank}. {name:20s}: {cv:6.2f}% - {sensitivity_level}")

    return experiments

hyperparam_results = demonstrate_hyperparameter_sensitivity()


# ============================================================================
# PART 5: GRADIENT CLIPPING DEMONSTRATION (Q48)
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: GRADIENT CLIPPING FOR TRAINING STABILITY")
print("=" * 80)
print("""
Interview Talking Points:
- Gradient clipping prevents exploding gradients
- Two methods: clip by norm (most common) or clip by value
- Typical threshold: 1.0 for clip by norm
- Essential for training RNNs and deep transformers
- Doesn't solve vanishing gradients (need better architecture for that)
""")

def train_with_gradient_clipping(
    clip_value: float = None,
    num_steps: int = 100
) -> Tuple[List[float], List[float], List[float]]:
    """
    Train a model with gradient clipping and track metrics.

    Interview Point: Gradient clipping is crucial for stable training,
    especially with large models or variable-length sequences.
    """
    # Deeper model more prone to gradient issues
    model = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # Higher LR to induce instability

    losses = []
    grad_norms_before = []
    grad_norms_after = []

    for step in range(num_steps):
        # Generate data with occasional large values (simulating difficult examples)
        x = torch.randn(32, 10)
        if step % 10 == 0:  # Occasional spike
            x = x * 5
        y = x.sum(dim=1, keepdim=True)

        # Forward pass
        pred = model(x)
        loss = F.mse_loss(pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm before clipping
        total_norm_before = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_before += p.grad.norm().item() ** 2
        total_norm_before = total_norm_before ** 0.5
        grad_norms_before.append(total_norm_before)

        # Apply gradient clipping if specified
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # Compute gradient norm after clipping
        total_norm_after = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.norm().item() ** 2
        total_norm_after = total_norm_after ** 0.5
        grad_norms_after.append(total_norm_after)

        optimizer.step()

        losses.append(loss.item())

    return losses, grad_norms_before, grad_norms_after

def demonstrate_gradient_clipping():
    """
    Compare training with and without gradient clipping.

    Key Interview Point: Gradient clipping is a simple but effective technique
    to prevent training instability from exploding gradients.
    """
    print("\n--- Testing Gradient Clipping ---")

    clip_values = [None, 0.5, 1.0, 5.0]
    results = {}

    for clip_val in clip_values:
        label = "No clipping" if clip_val is None else f"Clip={clip_val}"
        print(f"\nTraining with {label}")

        losses, grad_before, grad_after = train_with_gradient_clipping(
            clip_value=clip_val, num_steps=100
        )

        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Max gradient norm (before clip): {max(grad_before):.6f}")
        print(f"  Max gradient norm (after clip): {max(grad_after):.6f}")
        print(f"  Avg gradient norm (before clip): {np.mean(grad_before):.6f}")
        print(f"  Avg gradient norm (after clip): {np.mean(grad_after):.6f}")

        # Count clipping events
        if clip_val is not None:
            clip_events = sum(1 for g in grad_before if g > clip_val)
            print(f"  Gradient clipping events: {clip_events}/{len(grad_before)}")

        results[label] = {
            'losses': losses,
            'grad_before': grad_before,
            'grad_after': grad_after,
            'clip_value': clip_val
        }

    # Visualize gradient clipping effects
    print("\n--- Creating Gradient Clipping Visualization ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curves
    ax = axes[0, 0]
    for label, data in results.items():
        losses = np.clip(data['losses'], 0, 50)  # Clip for visualization
        ax.plot(losses, label=label, linewidth=2, alpha=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss (clipped at 50)')
    ax.set_title('Training Loss: Effect of Gradient Clipping')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Gradient norms before clipping
    ax = axes[0, 1]
    for label, data in results.items():
        grad_norms = data['grad_before']
        ax.plot(grad_norms, label=label, linewidth=2, alpha=0.8)

        # Show clipping threshold
        if data['clip_value'] is not None:
            ax.axhline(y=data['clip_value'], linestyle='--', alpha=0.5,
                      label=f"{label} threshold")

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm (before clipping)')
    ax.set_title('Gradient Magnitudes Before Clipping')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Before vs After clipping for one example
    ax = axes[1, 0]
    clip_example = "Clip=1.0"
    if clip_example in results:
        data = results[clip_example]
        steps = range(len(data['grad_before']))

        ax.plot(steps, data['grad_before'], label='Before Clipping',
                linewidth=2, alpha=0.8, color='red')
        ax.plot(steps, data['grad_after'], label='After Clipping',
                linewidth=2, alpha=0.8, color='green')
        ax.axhline(y=data['clip_value'], linestyle='--', color='black',
                  label=f"Threshold={data['clip_value']}")

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(f'Gradient Clipping Effect (Threshold={data["clip_value"]})')
        ax.legend()
        ax.grid(alpha=0.3)

    # Plot 4: Guidelines and best practices
    ax = axes[1, 1]
    ax.axis('off')

    guidelines = """
Gradient Clipping Best Practices:

1. WHEN TO USE:
   ✓ Training RNNs/LSTMs
   ✓ Very deep networks
   ✓ Variable sequence lengths
   ✓ When you see loss spikes

2. TYPICAL THRESHOLDS:
   • Transformers: 1.0 - 5.0
   • RNNs: 0.5 - 1.0
   • Very deep CNNs: 1.0 - 10.0

3. TWO METHODS:
   a) Clip by norm (most common):
      torch.nn.utils.clip_grad_norm_(
          model.parameters(), max_norm=1.0)

   b) Clip by value:
      torch.nn.utils.clip_grad_value_(
          model.parameters(), clip_value=0.5)

4. MONITORING:
   • Log gradient norms
   • Track clipping frequency
   • Adjust threshold if clipping
     happens too often (>50%)

5. LIMITATIONS:
   ✗ Doesn't fix vanishing gradients
   ✗ Can slow convergence if too aggressive
   ✓ Simple and effective for stability
"""

    ax.text(0.05, 0.95, guidelines, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'gradient_clipping_demo.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VIZ_DIR / 'gradient_clipping_demo.png'}")
    plt.close()

    return results

clipping_results = demonstrate_gradient_clipping()


# ============================================================================
# SUMMARY AND KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: KEY INTERVIEW POINTS")
print("=" * 80)

summary = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                  GRADIENT FLOW & OPTIMIZATION - KEY POINTS                ║
╚═══════════════════════════════════════════════════════════════════════════╝

1. EMBEDDING GRADIENTS (Q26):
   • Only embeddings for tokens in the batch receive gradient updates
   • Creates sparse gradient updates (typically 99%+ sparsity)
   • Memory efficient - don't need to compute/store all embedding gradients
   • Rare tokens update less frequently than common tokens

   Interview Answer: "Embedding layers only update parameters for tokens
   present in the current batch, making gradient updates very sparse. This
   is memory efficient and scales well to large vocabularies."

2. GRADIENT FLOW THROUGH LAYERS (Q27):
   • Monitor gradient magnitudes at each layer
   • Vanishing gradients: magnitudes shrink through layers (bad)
   • Exploding gradients: magnitudes grow through layers (bad)
   • Healthy: similar magnitudes across layers (good)
   • Solutions: residual connections, layer norm, careful initialization

   Interview Answer: "In transformers, residual connections and layer
   normalization help maintain stable gradient flow through many layers.
   We can verify this by checking that gradient norms are similar across
   layers during training."

3. LEARNING RATE SELECTION (Q48):
   • Most critical hyperparameter for training
   • Too high: unstable training, divergence
   • Too low: slow convergence, local minima
   • LLMs typically use: warmup + cosine decay
   • Starting points: 1e-4 to 3e-5 for large models

   Interview Answer: "Learning rate is the most sensitive hyperparameter.
   LLMs typically use a warmup phase (2-10% of training) to stabilize
   early training, followed by cosine decay to fine-tune convergence."

4. HYPERPARAMETER SENSITIVITY (Q48):
   • High sensitivity: learning rate, batch size
   • Medium sensitivity: warmup steps, weight decay, dropout
   • Lower sensitivity: optimizer choice (within Adam variants)
   • Always validate on smaller scale before full training

   Interview Answer: "Learning rate and batch size are most sensitive.
   I'd tune those first on a small dataset, then adjust regularization
   (weight decay, dropout) as needed. Optimizer choice matters less -
   AdamW is a safe default for LLMs."

5. GRADIENT CLIPPING (Q48):
   • Prevents exploding gradients
   • Essential for RNNs and deep transformers
   • Typical threshold: 1.0 for clip_grad_norm_
   • Monitor: track gradient norms and clipping frequency
   • Doesn't solve vanishing gradients (need architecture changes)

   Interview Answer: "Gradient clipping caps gradient norms to prevent
   training instability. It's standard practice in LLM training, typically
   with a max_norm of 1.0. We monitor clipping frequency - if it happens
   too often, we may need to adjust the learning rate or threshold."

╔═══════════════════════════════════════════════════════════════════════════╗
║                         PRACTICAL RECOMMENDATIONS                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

For LLM Training:
1. Start with AdamW optimizer (lr=3e-4, weight_decay=0.01)
2. Use warmup for 2-10% of total steps
3. Apply cosine decay schedule
4. Enable gradient clipping (max_norm=1.0)
5. Monitor: loss, gradient norms, learning rate
6. Log everything - helps debug training issues

For Debugging Training Issues:
• Loss spikes → reduce LR or increase gradient clipping
• Slow convergence → increase LR or check data
• NaN/Inf → definitely reduce LR, check for numerical instability
• Vanishing gradients → improve architecture (add residual connections)
• Exploding gradients → gradient clipping, reduce LR

For Interview Success:
✓ Know the tradeoffs (not just "best practices")
✓ Explain WHY techniques work, not just HOW
✓ Be ready to discuss debugging strategies
✓ Connect concepts to real training scenarios
"""

print(summary)

print("\n" + "=" * 80)
print("VISUALIZATION FILES CREATED")
print("=" * 80)
print(f"\nAll visualizations saved to: {VIZ_DIR}")
print("\nGenerated files:")
print("  1. embedding_gradient_sparsity.png")
print("  2. gradient_flow_through_layers.png")
print("  3. learning_rate_comparison.png")
print("  4. hyperparameter_sensitivity.png")
print("  5. gradient_clipping_demo.png")

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print("\nYou now have comprehensive visualizations and explanations for:")
print("  • Embedding gradient sparsity")
print("  • Gradient flow through networks")
print("  • Learning rate effects")
print("  • Hyperparameter sensitivity")
print("  • Gradient clipping")
print("\nReview the visualizations and practice explaining these concepts!")
print("Good luck with your LLM interviews! 🚀")
