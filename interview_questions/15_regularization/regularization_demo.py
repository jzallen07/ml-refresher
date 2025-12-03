"""
COMPREHENSIVE DROPOUT DEMONSTRATION FOR LLM INTERVIEWS
======================================================

Interview Question Q18: "What is the role of dropout in training LLMs?"

KEY INTERVIEW POINTS:
--------------------
1. Dropout is a regularization technique that prevents overfitting
2. During training: randomly zeros out activations with probability p
3. During inference: all neurons are active (no dropout)
4. Scaling: outputs are scaled by 1/(1-p) during training to maintain expected values
5. In transformers: applied to attention scores, FFN outputs, and embeddings
6. Typical rates: 0.1-0.3 for LLMs (lower than CNNs due to already high capacity)

This demo provides:
- Visualization of dropout behavior
- Training comparison with/without dropout
- Attention dropout implementation
- Effect on different layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
OUTPUT_DIR = "/Users/zack/dev/ml-refresher/data/interview_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("DROPOUT DEMONSTRATION FOR LLM TRAINING")
print("="*80)
print()

# ============================================================================
# PART 1: DROPOUT BEHAVIOR VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("PART 1: DROPOUT BEHAVIOR - TRAINING VS INFERENCE")
print("="*80)
print("""
INTERVIEW INSIGHT:
Dropout randomly zeros out neurons during training but is disabled during inference.
This creates an ensemble effect where each mini-batch trains a different sub-network.
""")

def visualize_dropout_behavior():
    """
    Demonstrates how dropout masks activations differently between training and inference.

    Interview Key Point: Dropout creates stochasticity during training but
    deterministic behavior during inference.
    """
    # Create a sample activation tensor (batch_size=1, features=100)
    activations = torch.randn(1, 100) * 2 + 5  # Mean ~5, std ~2

    # Different dropout rates to compare
    dropout_rates = [0.0, 0.1, 0.3, 0.5]

    fig, axes = plt.subplots(len(dropout_rates) + 1, 3, figsize=(15, 12))
    fig.suptitle('Dropout Behavior: Training vs Inference', fontsize=16, fontweight='bold')

    # Original activations
    axes[0, 0].bar(range(100), activations[0].numpy(), color='blue', alpha=0.7)
    axes[0, 0].set_title('Original Activations', fontweight='bold')
    axes[0, 0].set_ylabel('Activation Value')
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].set_ylim(-2, 12)
    axes[0, 0].text(50, 10, f'Mean: {activations.mean():.2f}\nStd: {activations.std():.2f}',
                    ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide the other two plots in first row
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')

    for idx, p in enumerate(dropout_rates, start=1):
        dropout_layer = nn.Dropout(p=p)

        # TRAINING MODE - Apply dropout (stochastic)
        dropout_layer.train()
        train_output = dropout_layer(activations.clone())

        # Show which neurons were dropped
        mask = (train_output != 0).float()[0]

        # INFERENCE MODE - No dropout (deterministic)
        dropout_layer.eval()
        eval_output = dropout_layer(activations.clone())

        # Plot training mode
        colors = ['red' if m == 0 else 'green' for m in mask]
        axes[idx, 0].bar(range(100), train_output[0].numpy(), color=colors, alpha=0.7)
        axes[idx, 0].set_title(f'Training Mode (p={p})', fontweight='bold')
        axes[idx, 0].set_ylabel('Activation Value')
        axes[idx, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[idx, 0].set_ylim(-2, 12)

        # Calculate statistics
        num_dropped = (mask == 0).sum().item()
        active_mean = train_output[0][mask.bool()].mean() if mask.sum() > 0 else 0

        axes[idx, 0].text(50, 10,
                         f'Dropped: {num_dropped}/100 ({num_dropped}%)\n'
                         f'Active Mean: {active_mean:.2f}\n'
                         f'Scale Factor: {1/(1-p):.2f}',
                         ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot inference mode
        axes[idx, 1].bar(range(100), eval_output[0].numpy(), color='blue', alpha=0.7)
        axes[idx, 1].set_title(f'Inference Mode (p={p})', fontweight='bold')
        axes[idx, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[idx, 1].set_ylim(-2, 12)
        axes[idx, 1].text(50, 10,
                         f'Mean: {eval_output.mean():.2f}\n'
                         f'All neurons active',
                         ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Plot dropout mask
        axes[idx, 2].bar(range(100), mask.numpy(), color='black', alpha=0.7)
        axes[idx, 2].set_title(f'Dropout Mask (p={p})', fontweight='bold')
        axes[idx, 2].set_ylim(-0.1, 1.1)
        axes[idx, 2].set_yticks([0, 1])
        axes[idx, 2].set_yticklabels(['Dropped', 'Active'])
        axes[idx, 2].text(50, 0.5, f'{int(mask.sum())}/100 active',
                         ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # Set x-labels only on bottom row
    for ax in axes[-1, :]:
        ax.set_xlabel('Neuron Index')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_dropout_behavior.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 08_dropout_behavior.png")
    print()
    print("INTERVIEW EXPLANATION:")
    print("- Red bars: Neurons that were dropped (set to 0)")
    print("- Green bars: Active neurons (scaled up by 1/(1-p))")
    print("- Inference mode: All neurons active, no randomness")
    print("- Higher dropout rate = more neurons dropped = stronger regularization")
    print()

visualize_dropout_behavior()


# ============================================================================
# PART 2: TRAINING COMPARISON - WITH VS WITHOUT DROPOUT
# ============================================================================
print("\n" + "="*80)
print("PART 2: TRAINING COMPARISON - EFFECT ON OVERFITTING")
print("="*80)
print("""
INTERVIEW INSIGHT:
Dropout reduces overfitting by preventing co-adaptation of neurons.
Networks learn more robust features that don't rely on specific neurons.
""")

class SimpleClassifier(nn.Module):
    """
    A simple 3-layer network for demonstrating dropout effects.

    Interview Note: This architecture mimics a simplified transformer FFN block.
    """
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=2, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        # First layer + activation + dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # Second layer + activation + dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # Output layer (no dropout after final layer)
        x = self.fc3(x)
        return x

def generate_synthetic_data(n_samples=500, n_features=20, noise_level=0.5):
    """
    Generate synthetic binary classification data.
    Deliberately make it small to encourage overfitting without dropout.
    """
    # Generate features
    X = torch.randn(n_samples, n_features)

    # Create a complex decision boundary (non-linear)
    # This mimics the complexity of language modeling tasks
    weights = torch.randn(n_features)
    logits = X @ weights + torch.sin(X[:, 0]) * 2 + X[:, 1] ** 2

    # Add noise to make it harder
    logits += torch.randn(n_samples) * noise_level

    # Binary labels
    y = (logits > logits.median()).long()

    return X, y

def train_model(model, train_X, train_y, val_X, val_y, epochs=100):
    """
    Train a model and track training/validation metrics.

    Interview Note: We track both losses to demonstrate overfitting behavior.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()

        train_output = model(train_X)
        train_loss = criterion(train_output, train_y)
        train_loss.backward()
        optimizer.step()

        # Calculate training accuracy
        train_pred = train_output.argmax(dim=1)
        train_acc = (train_pred == train_y).float().mean()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_output = model(val_X)
            val_loss = criterion(val_output, val_y)
            val_pred = val_output.argmax(dim=1)
            val_acc = (val_pred == val_y).float().mean()

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_accs.append(train_acc.item())
        val_accs.append(val_acc.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs

# Generate data
print("Generating synthetic classification data...")
train_X, train_y = generate_synthetic_data(n_samples=200, noise_level=0.3)
val_X, val_y = generate_synthetic_data(n_samples=100, noise_level=0.3)
print(f"Training samples: {len(train_X)}, Validation samples: {len(val_X)}")
print()

# Train models with different dropout rates
dropout_rates = [0.0, 0.1, 0.3, 0.5]
results = {}

for rate in dropout_rates:
    print(f"\n{'='*60}")
    print(f"Training model with dropout rate = {rate}")
    print(f"{'='*60}")

    model = SimpleClassifier(dropout_rate=rate)
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_X, train_y, val_X, val_y, epochs=100
    )

    results[rate] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'final_gap': train_accs[-1] - val_accs[-1]  # Overfitting indicator
    }

    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {train_accs[-1]:.4f}")
    print(f"  Val Accuracy:   {val_accs[-1]:.4f}")
    print(f"  Accuracy Gap:   {results[rate]['final_gap']:.4f} (lower = less overfitting)")

# Visualize training comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Dropout Effect on Training Dynamics', fontsize=16, fontweight='bold')

colors = ['red', 'orange', 'green', 'blue']
labels = [f'Dropout={rate}' for rate in dropout_rates]

# Plot 1: Training Loss
ax = axes[0, 0]
for idx, rate in enumerate(dropout_rates):
    ax.plot(results[rate]['train_losses'], color=colors[idx], label=labels[idx], linewidth=2)
ax.set_title('Training Loss', fontweight='bold', fontsize=12)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Validation Loss
ax = axes[0, 1]
for idx, rate in enumerate(dropout_rates):
    ax.plot(results[rate]['val_losses'], color=colors[idx], label=labels[idx], linewidth=2)
ax.set_title('Validation Loss', fontweight='bold', fontsize=12)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Training Accuracy
ax = axes[1, 0]
for idx, rate in enumerate(dropout_rates):
    ax.plot(results[rate]['train_accs'], color=colors[idx], label=labels[idx], linewidth=2)
ax.set_title('Training Accuracy', fontweight='bold', fontsize=12)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Validation Accuracy
ax = axes[1, 1]
for idx, rate in enumerate(dropout_rates):
    ax.plot(results[rate]['val_accs'], color=colors[idx], label=labels[idx], linewidth=2)
ax.set_title('Validation Accuracy', fontweight='bold', fontsize=12)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_dropout_training_comparison.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: 09_dropout_training_comparison.png")

# Print overfitting analysis
print("\n" + "="*80)
print("OVERFITTING ANALYSIS (Train-Val Accuracy Gap)")
print("="*80)
for rate in dropout_rates:
    gap = results[rate]['final_gap']
    print(f"Dropout {rate}: {gap:+.4f} {'⬆️ MORE OVERFITTING' if gap > 0.05 else '✓ LESS OVERFITTING'}")
print()
print("INTERVIEW KEY POINT:")
print("Lower accuracy gap indicates better generalization.")
print("Dropout reduces overfitting but may slightly lower training accuracy.")
print()


# ============================================================================
# PART 3: ATTENTION DROPOUT IMPLEMENTATION
# ============================================================================
print("\n" + "="*80)
print("PART 3: ATTENTION DROPOUT IN TRANSFORMERS")
print("="*80)
print("""
INTERVIEW INSIGHT:
In transformers, dropout is applied to:
1. Attention weights (after softmax)
2. Attention output (after value multiplication)
3. Feed-forward network outputs
4. Embeddings (input and positional)

This prevents the model from relying too heavily on specific attention patterns.
""")

class AttentionWithDropout(nn.Module):
    """
    Simplified multi-head attention with dropout.

    Interview Note: This shows where dropout is typically applied in transformer blocks.
    """
    def __init__(self, d_model=64, n_heads=4, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Dropout layers
        self.attn_dropout = nn.Dropout(p=dropout_rate)  # Applied to attention weights
        self.output_dropout = nn.Dropout(p=dropout_rate)  # Applied to output

    def forward(self, x, return_attention=False):
        """
        x: (batch_size, seq_len, d_model)

        Returns:
        - output: (batch_size, seq_len, d_model)
        - attention_weights: (batch_size, n_heads, seq_len, seq_len) if return_attention=True
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head attention
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # scores: (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # *** CRITICAL: Dropout on attention weights ***
        # This is where dropout is applied in attention mechanism
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # Apply attention to values
        # output: (batch_size, n_heads, seq_len, d_k)
        output = torch.matmul(attn_weights_dropped, V)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = self.W_o(output)

        # *** CRITICAL: Dropout on output ***
        output = self.output_dropout(output)

        if return_attention:
            return output, attn_weights, attn_weights_dropped
        return output

def visualize_attention_dropout():
    """
    Visualize how dropout affects attention patterns.

    Interview Key Point: Dropout in attention prevents over-reliance on specific tokens.
    """
    # Create sample input (batch_size=1, seq_len=10, d_model=64)
    seq_len = 10
    x = torch.randn(1, seq_len, 64)

    # Create attention modules with different dropout rates
    dropout_rates = [0.0, 0.3, 0.5]

    fig, axes = plt.subplots(len(dropout_rates), 4, figsize=(20, 12))
    fig.suptitle('Attention Dropout Visualization', fontsize=16, fontweight='bold')

    for idx, dropout_rate in enumerate(dropout_rates):
        attn_module = AttentionWithDropout(d_model=64, n_heads=4, dropout_rate=dropout_rate)
        attn_module.train()  # Enable dropout

        # Get attention weights (both original and dropped)
        _, attn_orig, attn_dropped = attn_module(x, return_attention=True)

        # attn_orig: (1, 4, 10, 10) - 4 heads
        # Let's visualize head 0
        head_idx = 0
        attn_orig_head = attn_orig[0, head_idx].detach().numpy()
        attn_dropped_head = attn_dropped[0, head_idx].detach().numpy()

        # Calculate dropout mask
        dropout_mask = (attn_dropped_head != 0).astype(float)

        # Plot original attention
        im1 = axes[idx, 0].imshow(attn_orig_head, cmap='viridis', aspect='auto', vmin=0, vmax=0.3)
        axes[idx, 0].set_title(f'Original Attention (dropout={dropout_rate})', fontweight='bold')
        axes[idx, 0].set_xlabel('Key Position')
        axes[idx, 0].set_ylabel('Query Position')
        plt.colorbar(im1, ax=axes[idx, 0], fraction=0.046, pad=0.04)

        # Plot attention after dropout
        im2 = axes[idx, 1].imshow(attn_dropped_head, cmap='viridis', aspect='auto', vmin=0, vmax=0.3)
        axes[idx, 1].set_title(f'After Dropout (dropout={dropout_rate})', fontweight='bold')
        axes[idx, 1].set_xlabel('Key Position')
        axes[idx, 1].set_ylabel('Query Position')
        plt.colorbar(im2, ax=axes[idx, 1], fraction=0.046, pad=0.04)

        # Plot dropout mask
        im3 = axes[idx, 2].imshow(dropout_mask, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[idx, 2].set_title(f'Dropout Mask (dropout={dropout_rate})', fontweight='bold')
        axes[idx, 2].set_xlabel('Key Position')
        axes[idx, 2].set_ylabel('Query Position')
        plt.colorbar(im3, ax=axes[idx, 2], fraction=0.046, pad=0.04)

        # Plot difference (what was removed)
        difference = attn_orig_head - attn_dropped_head
        im4 = axes[idx, 3].imshow(difference, cmap='Reds', aspect='auto', vmin=0, vmax=0.3)
        axes[idx, 3].set_title(f'Dropped Values (dropout={dropout_rate})', fontweight='bold')
        axes[idx, 3].set_xlabel('Key Position')
        axes[idx, 3].set_ylabel('Query Position')
        plt.colorbar(im4, ax=axes[idx, 3], fraction=0.046, pad=0.04)

        # Print statistics
        num_zeros = (dropout_mask == 0).sum()
        total = dropout_mask.size
        print(f"Dropout {dropout_rate}: {num_zeros}/{total} attention scores dropped "
              f"({100*num_zeros/total:.1f}%)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_attention_dropout.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: 10_attention_dropout.png")
    print()
    print("INTERVIEW EXPLANATION:")
    print("- Original Attention: The attention pattern before dropout")
    print("- After Dropout: Some attention scores are zeroed out")
    print("- Dropout Mask: Shows which attention scores were kept (green) vs dropped (red)")
    print("- Dropped Values: Visualizes what information was removed")
    print()
    print("WHY THIS MATTERS:")
    print("- Prevents model from always attending to the same tokens")
    print("- Forces learning of diverse attention patterns")
    print("- Creates ensemble effect across different dropout masks")
    print()

visualize_attention_dropout()


# ============================================================================
# PART 4: LAYER-WISE DROPOUT DEMONSTRATION
# ============================================================================
print("\n" + "="*80)
print("PART 4: LAYER-WISE DROPOUT ANALYSIS")
print("="*80)
print("""
INTERVIEW INSIGHT:
Different layers can use different dropout rates. In LLMs:
- Lower layers (closer to input): Often use higher dropout
- Middle layers: Moderate dropout
- Upper layers (closer to output): Sometimes lower or no dropout
- This is task-dependent and often found through hyperparameter tuning
""")

class LayerwiseDropoutNetwork(nn.Module):
    """
    Network with different dropout rates at different layers.

    Interview Note: Mimics how transformers might use varying dropout across layers.
    """
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=2):
        super().__init__()

        # Layer 1 - High dropout (near input, more noise)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.5)

        # Layer 2 - Medium dropout
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.3)

        # Layer 3 - Low dropout
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(p=0.1)

        # Output layer - No dropout
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_intermediates=False):
        intermediates = {}

        # Layer 1
        h1 = F.relu(self.fc1(x))
        h1_dropped = self.dropout1(h1)
        if return_intermediates:
            intermediates['layer1'] = (h1, h1_dropped)

        # Layer 2
        h2 = F.relu(self.fc2(h1_dropped))
        h2_dropped = self.dropout2(h2)
        if return_intermediates:
            intermediates['layer2'] = (h2, h2_dropped)

        # Layer 3
        h3 = F.relu(self.fc3(h2_dropped))
        h3_dropped = self.dropout3(h3)
        if return_intermediates:
            intermediates['layer3'] = (h3, h3_dropped)

        # Output
        output = self.fc_out(h3_dropped)

        if return_intermediates:
            return output, intermediates
        return output

def analyze_layerwise_dropout():
    """Analyze how dropout affects activations at different layers."""
    model = LayerwiseDropoutNetwork()
    model.train()  # Enable dropout

    # Create sample input
    x = torch.randn(1, 20)

    # Forward pass with intermediate values
    output, intermediates = model(x, return_intermediates=True)

    print("Layer-wise Dropout Analysis:")
    print("-" * 60)

    layer_names = ['layer1', 'layer2', 'layer3']
    dropout_rates = [0.5, 0.3, 0.1]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Layer-wise Dropout Effects', fontsize=16, fontweight='bold')

    for idx, (layer_name, dropout_rate) in enumerate(zip(layer_names, dropout_rates)):
        h_before, h_after = intermediates[layer_name]

        # Convert to numpy
        before = h_before[0].detach().numpy()
        after = h_after[0].detach().numpy()

        # Calculate statistics
        num_neurons = len(before)
        num_dropped = (after == 0).sum()
        percent_dropped = 100 * num_dropped / num_neurons

        print(f"\n{layer_name.upper()} (dropout={dropout_rate}):")
        print(f"  Neurons dropped: {num_dropped}/{num_neurons} ({percent_dropped:.1f}%)")
        print(f"  Mean before: {before.mean():.4f}, Std before: {before.std():.4f}")
        print(f"  Mean after:  {after.mean():.4f}, Std after:  {after.std():.4f}")

        # Plot 1: Activations before dropout
        axes[idx, 0].bar(range(num_neurons), before, color='blue', alpha=0.7)
        axes[idx, 0].set_title(f'{layer_name}: Before Dropout (p={dropout_rate})', fontweight='bold')
        axes[idx, 0].set_ylabel('Activation')
        axes[idx, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Plot 2: Activations after dropout
        colors = ['red' if a == 0 else 'green' for a in after]
        axes[idx, 1].bar(range(num_neurons), after, color=colors, alpha=0.7)
        axes[idx, 1].set_title(f'{layer_name}: After Dropout (p={dropout_rate})', fontweight='bold')
        axes[idx, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Plot 3: Dropout mask
        mask = (after != 0).astype(float)
        axes[idx, 2].bar(range(num_neurons), mask, color='black', alpha=0.7)
        axes[idx, 2].set_title(f'{layer_name}: Dropout Mask', fontweight='bold')
        axes[idx, 2].set_ylim(-0.1, 1.1)
        axes[idx, 2].set_yticks([0, 1])
        axes[idx, 2].set_yticklabels(['Dropped', 'Active'])

    # Set x-labels
    for ax in axes[-1, :]:
        ax.set_xlabel('Neuron Index')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_layerwise_dropout.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: 11_layerwise_dropout.png")
    print()

analyze_layerwise_dropout()


# ============================================================================
# PART 5: DROPOUT SCALING DEMONSTRATION
# ============================================================================
print("\n" + "="*80)
print("PART 5: DROPOUT SCALING BEHAVIOR")
print("="*80)
print("""
INTERVIEW INSIGHT:
During training, PyTorch automatically scales activations by 1/(1-p) to maintain
expected values. This is called "inverted dropout" and ensures that the scale
of activations is consistent between training and inference.

Mathematical intuition:
- If dropout rate is 0.5, half the neurons are dropped on average
- Remaining neurons are scaled by 1/(1-0.5) = 2
- Expected value of output stays the same as without dropout
""")

def demonstrate_dropout_scaling():
    """Show how dropout scaling maintains expected values."""

    # Create a tensor with known mean
    x = torch.ones(1000) * 10.0  # All values are 10.0
    print(f"Original tensor mean: {x.mean():.4f}")
    print()

    dropout_rates = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    n_trials = 100

    results = []

    for p in dropout_rates:
        dropout = nn.Dropout(p=p)
        dropout.train()  # Enable dropout

        means = []
        num_zeros_list = []

        # Run multiple trials to get average behavior
        for _ in range(n_trials):
            output = dropout(x.clone())
            means.append(output.mean().item())
            num_zeros_list.append((output == 0).sum().item())

        avg_mean = np.mean(means)
        avg_zeros = np.mean(num_zeros_list)
        theoretical_zeros = p * len(x)
        scale_factor = 1 / (1 - p) if p < 1.0 else float('inf')

        results.append({
            'dropout_rate': p,
            'avg_mean': avg_mean,
            'avg_zeros': avg_zeros,
            'theoretical_zeros': theoretical_zeros,
            'scale_factor': scale_factor
        })

        print(f"Dropout rate: {p:.1f}")
        print(f"  Average output mean: {avg_mean:.4f} (should be ~10.0)")
        print(f"  Average zeros: {avg_zeros:.1f} / {len(x)} ({100*avg_zeros/len(x):.1f}%)")
        print(f"  Theoretical zeros: {theoretical_zeros:.1f} ({100*p:.1f}%)")
        print(f"  Scale factor: {scale_factor:.4f}")
        print()

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Dropout Scaling Maintains Expected Values', fontsize=14, fontweight='bold')

    # Plot 1: Mean values across dropout rates
    dropout_vals = [r['dropout_rate'] for r in results]
    means = [r['avg_mean'] for r in results]

    axes[0].plot(dropout_vals, means, 'o-', linewidth=2, markersize=8, color='blue', label='Actual Mean')
    axes[0].axhline(y=10.0, color='red', linestyle='--', linewidth=2, label='Expected Mean (10.0)')
    axes[0].set_xlabel('Dropout Rate', fontweight='bold')
    axes[0].set_ylabel('Output Mean', fontweight='bold')
    axes[0].set_title('Mean Preservation with Dropout Scaling', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([9, 11])

    # Plot 2: Scale factors
    scale_factors = [r['scale_factor'] for r in results if r['scale_factor'] != float('inf')]
    dropout_vals_finite = [r['dropout_rate'] for r in results if r['scale_factor'] != float('inf')]

    axes[1].plot(dropout_vals_finite, scale_factors, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Dropout Rate', fontweight='bold')
    axes[1].set_ylabel('Scale Factor (1 / (1-p))', fontweight='bold')
    axes[1].set_title('Dropout Scale Factor vs Dropout Rate', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Add annotations
    for i, (p, sf) in enumerate(zip(dropout_vals_finite, scale_factors)):
        if i % 2 == 0:  # Annotate every other point to avoid crowding
            axes[1].annotate(f'{sf:.2f}', (p, sf), textcoords="offset points",
                           xytext=(0,10), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/12_dropout_scaling.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 12_dropout_scaling.png")
    print()
    print("INTERVIEW KEY POINT:")
    print("Despite dropping neurons, the expected value remains constant due to scaling.")
    print("This is why dropout doesn't require changing learning rates or other hyperparameters.")
    print()

demonstrate_dropout_scaling()


# ============================================================================
# FINAL SUMMARY FOR INTERVIEWS
# ============================================================================
print("\n" + "="*80)
print("INTERVIEW SUMMARY: KEY POINTS ABOUT DROPOUT IN LLMS")
print("="*80)
print("""
1. WHAT IS DROPOUT?
   - Regularization technique that randomly zeros out neurons during training
   - Rate 'p' determines probability of dropping a neuron
   - Disabled during inference (all neurons active)

2. WHY USE DROPOUT IN LLMS?
   - Prevents overfitting by reducing co-adaptation of neurons
   - Creates ensemble effect (each batch trains a different sub-network)
   - Forces network to learn robust, distributed representations
   - Particularly important with large models and limited data

3. WHERE IS DROPOUT APPLIED IN TRANSFORMERS?
   - Attention weights (after softmax)
   - Attention output projections
   - Feed-forward network outputs
   - Embedding layers (both token and positional)
   - Some architectures use layer-specific dropout rates

4. TYPICAL DROPOUT RATES FOR LLMS:
   - 0.1 - 0.3 is common (lower than CNNs)
   - Lower rates because transformers already have high capacity
   - Too high dropout can hurt performance
   - Often different rates for different components

5. TECHNICAL DETAILS:
   - Inverted dropout: scales by 1/(1-p) during training
   - Maintains expected values between train/inference
   - Implemented efficiently in modern frameworks
   - No additional computation during inference

6. TRADE-OFFS:
   ✓ Reduces overfitting
   ✓ Improves generalization
   ✓ Adds regularization without changing architecture
   ✗ Slower training (need more epochs)
   ✗ Can hurt performance if rate is too high
   ✗ Increases training variance

7. ALTERNATIVES AND COMPLEMENTS:
   - Layer normalization (used together with dropout)
   - Weight decay / L2 regularization
   - Data augmentation
   - Early stopping
   - DropConnect (drops connections instead of neurons)

8. MODERN TRENDS:
   - Some recent LLMs use less dropout or none at all
   - Pre-training often uses less dropout than fine-tuning
   - Task-dependent: more dropout for smaller datasets
   - Architecture-dependent: some designs are naturally regularized

Files generated:
- 08_dropout_behavior.png: Training vs inference dropout behavior
- 09_dropout_training_comparison.png: Effect on overfitting
- 10_attention_dropout.png: Attention mechanism dropout visualization
- 11_layerwise_dropout.png: Different dropout rates per layer
- 12_dropout_scaling.png: How scaling maintains expected values
""")

print("\n" + "="*80)
print("Demo completed successfully!")
print(f"All visualizations saved to: {OUTPUT_DIR}")
print("="*80)
