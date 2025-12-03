"""
Loss Functions and Math for LLM Interviews - Educational Demo

This comprehensive demo covers key mathematical concepts for LLM interviews:
- Q25: Cross-Entropy Loss
- Q29: Perplexity
- Q30: KL Divergence
- Q31: ReLU and activation functions
- Bonus: Chain Rule and Backpropagation

Each section implements concepts from scratch, then compares with PyTorch,
and includes visualizations for deep understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import math

# Setup visualization directory
VIZ_DIR = Path("/Users/zack/dev/ml-refresher/data/interview_viz")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("LOSS FUNCTIONS AND MATH FOR LLM INTERVIEWS - COMPREHENSIVE DEMO")
print("="*80)


# ============================================================================
# SECTION 1: CROSS-ENTROPY LOSS (Q25)
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: CROSS-ENTROPY LOSS")
print("="*80)

print("""
INTERVIEW CONTEXT:
Cross-entropy is THE fundamental loss function for language models.
It measures the difference between predicted probability distribution and
the true distribution (one-hot encoded target).

Mathematical Formula:
    L = -∑ y_true * log(y_pred)

For a single token prediction:
    L = -log(p_target)

where p_target is the predicted probability of the correct token.
""")


def cross_entropy_from_scratch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Implement cross-entropy loss from scratch.

    Args:
        logits: Raw model outputs (batch_size, num_classes)
        targets: True class indices (batch_size,)

    Returns:
        Average cross-entropy loss

    Steps:
        1. Convert logits to probabilities using softmax
        2. Extract probability of correct class
        3. Take negative log
        4. Average over batch
    """
    print("\nCross-Entropy Calculation Steps:")
    print(f"1. Input logits shape: {logits.shape}")
    print(f"   Logits (raw scores):\n{logits}")

    # Step 1: Apply softmax to convert logits to probabilities
    # Softmax: exp(x_i) / sum(exp(x_j))
    # We subtract max for numerical stability
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

    print(f"\n2. After softmax (probabilities):\n{probs}")
    print(f"   Probabilities sum to 1: {torch.allclose(probs.sum(dim=1), torch.ones(probs.shape[0]))}")

    # Step 2: Extract probability of correct class for each sample
    batch_size = logits.shape[0]
    target_probs = probs[range(batch_size), targets]

    print(f"\n3. Target classes: {targets}")
    print(f"   Probabilities of correct classes: {target_probs}")

    # Step 3: Take negative log
    neg_log_probs = -torch.log(target_probs)

    print(f"\n4. Negative log probabilities: {neg_log_probs}")

    # Step 4: Average over batch
    loss = torch.mean(neg_log_probs)

    print(f"\n5. Average loss: {loss.item():.4f}")

    return loss


# Example: Predicting next token in language modeling
print("\n" + "-"*80)
print("Example: Language Model Token Prediction")
print("-"*80)

# Vocabulary: ["the", "cat", "sat", "mat"]
vocab = ["the", "cat", "sat", "mat"]
vocab_size = len(vocab)

# Model predicts distribution over vocabulary for 3 positions
# Batch size = 3 (3 different positions)
logits = torch.tensor([
    [2.0, 1.0, 0.5, 0.2],  # Position 1: model predicts "the" strongly
    [0.3, 3.0, 0.4, 0.1],  # Position 2: model predicts "cat" strongly
    [0.2, 0.1, 2.5, 1.0],  # Position 3: model predicts "sat" strongly
])

# True next tokens: ["the", "cat", "sat"] -> indices [0, 1, 2]
targets = torch.tensor([0, 1, 2])

print(f"\nVocabulary: {vocab}")
print(f"Predictions for 3 positions in sequence")

# Calculate from scratch
print("\n" + "="*40)
print("FROM SCRATCH IMPLEMENTATION")
print("="*40)
loss_scratch = cross_entropy_from_scratch(logits, targets)

# Compare with PyTorch
print("\n" + "="*40)
print("PYTORCH IMPLEMENTATION")
print("="*40)
loss_pytorch = F.cross_entropy(logits, targets)
print(f"\nPyTorch cross_entropy: {loss_pytorch.item():.4f}")

print(f"\nDifference: {abs(loss_scratch - loss_pytorch).item():.10f}")
print("✓ Implementations match!" if torch.allclose(loss_scratch, loss_pytorch) else "✗ Mismatch!")


# Visualize cross-entropy behavior
print("\n" + "-"*80)
print("Visualizing Cross-Entropy Loss Behavior")
print("-"*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Loss vs predicted probability
probs_range = np.linspace(0.01, 1.0, 100)
ce_loss = -np.log(probs_range)

axes[0].plot(probs_range, ce_loss, linewidth=2, color='blue')
axes[0].set_xlabel('Predicted Probability of Correct Class', fontsize=12)
axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
axes[0].set_title('Cross-Entropy Loss vs Predicted Probability', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(x=1.0, color='r', linestyle='--', alpha=0.3, label='Perfect prediction')

# Add annotations
axes[0].annotate('High penalty for\nlow confidence', xy=(0.1, -np.log(0.1)),
                xytext=(0.3, 3), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
axes[0].annotate('Low penalty for\nhigh confidence', xy=(0.9, -np.log(0.9)),
                xytext=(0.7, 1), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
axes[0].legend()

# Right plot: Example predictions
example_probs = [0.9, 0.7, 0.5, 0.3, 0.1]
example_losses = [-np.log(p) for p in example_probs]
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(example_probs)))

bars = axes[1].bar(range(len(example_probs)), example_losses, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Predicted Probability', fontsize=12)
axes[1].set_ylabel('Cross-Entropy Loss', fontsize=12)
axes[1].set_title('Loss for Different Confidence Levels', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(len(example_probs)))
axes[1].set_xticklabels([f'{p:.1f}' for p in example_probs])
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, loss, prob) in enumerate(zip(bars, example_losses, example_probs)):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(VIZ_DIR / "01_cross_entropy_behavior.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {VIZ_DIR / '01_cross_entropy_behavior.png'}")
plt.close()


# ============================================================================
# SECTION 2: PERPLEXITY (Q29)
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: PERPLEXITY")
print("="*80)

print("""
INTERVIEW CONTEXT:
Perplexity is the primary evaluation metric for language models.
It measures how "surprised" the model is by the test data.

Mathematical Definition:
    Perplexity = exp(average cross-entropy loss)
    PPL = exp(L) where L = -(1/N) * ∑ log P(w_i)

Interpretation:
- Lower perplexity = better model
- PPL of K means model is as "confused" as if it had to choose uniformly
  from K possibilities at each step
- PPL = 1 means perfect prediction (100% confidence on correct tokens)
- PPL = vocab_size means random guessing
""")


def calculate_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate perplexity from logits and targets.

    Returns:
        (perplexity, cross_entropy_loss)
    """
    # Calculate cross-entropy loss
    ce_loss = F.cross_entropy(logits, targets)

    # Perplexity is exp of the loss
    perplexity = torch.exp(ce_loss)

    return perplexity.item(), ce_loss.item()


print("\n" + "-"*80)
print("Example: Comparing Model Quality with Perplexity")
print("-"*80)

# Scenario: Two models predicting the sentence "the cat sat"
vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "jumped"]
vocab_size = len(vocab)
targets = torch.tensor([0, 1, 2])  # "the cat sat"

print(f"Vocabulary size: {vocab_size}")
print(f"Target sequence: {[vocab[i] for i in targets]}")

# Model A: Good model (confident and correct)
logits_good = torch.tensor([
    [3.0, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1],  # Strongly predicts "the"
    [0.2, 3.5, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1],  # Strongly predicts "cat"
    [0.1, 0.2, 3.2, 0.3, 0.2, 0.1, 0.1, 0.1],  # Strongly predicts "sat"
])

# Model B: Mediocre model (less confident)
logits_mediocre = torch.tensor([
    [1.5, 1.0, 1.0, 0.8, 0.5, 0.5, 0.4, 0.3],  # Weakly predicts "the"
    [0.8, 1.8, 1.0, 0.7, 0.6, 0.5, 0.4, 0.3],  # Weakly predicts "cat"
    [0.7, 0.9, 1.6, 1.0, 0.8, 0.6, 0.5, 0.4],  # Weakly predicts "sat"
])

# Model C: Bad model (nearly random)
logits_bad = torch.tensor([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Nearly uniform
    [1.1, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0],  # Nearly uniform
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Nearly uniform
])

models = [
    ("Good Model", logits_good),
    ("Mediocre Model", logits_mediocre),
    ("Bad Model (Random)", logits_bad),
]

results = []
for name, logits in models:
    ppl, ce = calculate_perplexity(logits, targets)
    results.append((name, ppl, ce))

    print(f"\n{name}:")
    print(f"  Cross-Entropy Loss: {ce:.4f}")
    print(f"  Perplexity: {ppl:.4f}")

    # Show probability distribution for first prediction
    probs = F.softmax(logits[0], dim=0)
    print(f"  Probability of correct token 'the': {probs[0]:.4f}")
    print(f"  Top-3 predicted tokens: {[(vocab[i], probs[i].item()) for i in torch.topk(probs, 3).indices]}")

print(f"\n{'='*40}")
print("INTERPRETATION:")
print(f"{'='*40}")
print(f"Random baseline perplexity ≈ {vocab_size:.1f}")
print(f"\nGood model: Low perplexity ({results[0][1]:.2f}) = confident, accurate predictions")
print(f"Mediocre model: Medium perplexity ({results[1][1]:.2f}) = less confident")
print(f"Bad model: High perplexity ({results[2][1]:.2f}) = nearly random guessing")


# Visualize perplexity comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Perplexity comparison
names = [r[0] for r in results]
ppls = [r[1] for r in results]
colors = ['green', 'orange', 'red']

bars = axes[0].bar(range(len(names)), ppls, color=colors, edgecolor='black', linewidth=2, alpha=0.7)
axes[0].set_ylabel('Perplexity', fontsize=12)
axes[0].set_title('Perplexity Comparison Across Models', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(len(names)))
axes[0].set_xticklabels(names, rotation=15, ha='right')
axes[0].axhline(y=vocab_size, color='purple', linestyle='--', linewidth=2,
                label=f'Random baseline ({vocab_size})')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, ppl in zip(bars, ppls):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Right plot: Perplexity vs loss relationship
ce_range = np.linspace(0, 3, 100)
ppl_range = np.exp(ce_range)

axes[1].plot(ce_range, ppl_range, linewidth=2, color='blue')
axes[1].set_xlabel('Cross-Entropy Loss', fontsize=12)
axes[1].set_ylabel('Perplexity', fontsize=12)
axes[1].set_title('Perplexity = exp(Cross-Entropy Loss)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Mark our models on the curve
for name, ppl, ce in results:
    color = {'Good Model': 'green', 'Mediocre Model': 'orange', 'Bad Model (Random)': 'red'}[name]
    axes[1].plot(ce, ppl, 'o', markersize=12, color=color, label=name, markeredgecolor='black', markeredgewidth=2)

axes[1].legend()

plt.tight_layout()
plt.savefig(VIZ_DIR / "02_perplexity_comparison.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: {VIZ_DIR / '02_perplexity_comparison.png'}")
plt.close()


# ============================================================================
# SECTION 3: KL DIVERGENCE (Q30)
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: KL DIVERGENCE")
print("="*80)

print("""
INTERVIEW CONTEXT:
KL Divergence measures how one probability distribution differs from another.
Critical for:
- Knowledge distillation (student mimics teacher)
- Variational inference in VAEs
- Policy optimization in RL (RLHF for LLMs)

Mathematical Definition:
    KL(P || Q) = ∑ P(x) * log(P(x) / Q(x))

Properties:
- Always non-negative: KL(P || Q) ≥ 0
- Zero only when P = Q (distributions are identical)
- NOT symmetric: KL(P || Q) ≠ KL(Q || P)
- NOT a distance metric (doesn't satisfy triangle inequality)

Cross-entropy connection:
    H(P, Q) = H(P) + KL(P || Q)
    where H(P) is entropy of P, H(P, Q) is cross-entropy
""")


def kl_divergence_from_scratch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Calculate KL divergence from scratch: KL(P || Q).

    Args:
        p: True distribution (batch_size, num_classes)
        q: Approximate distribution (batch_size, num_classes)

    Returns:
        KL divergence
    """
    print("\nKL Divergence Calculation Steps:")
    print(f"Distribution P (true):\n{p}")
    print(f"Distribution Q (approximate):\n{q}")

    # KL(P || Q) = sum(P * log(P/Q))
    # = sum(P * (log(P) - log(Q)))

    # Add small epsilon for numerical stability
    eps = 1e-10
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)

    # Calculate log ratio
    log_ratio = torch.log(p_safe) - torch.log(q_safe)
    print(f"\nlog(P/Q):\n{log_ratio}")

    # Weight by P and sum
    kl = torch.sum(p * log_ratio, dim=1)
    print(f"\nKL divergence per sample: {kl}")

    # Average over batch
    kl_mean = torch.mean(kl)
    print(f"Average KL divergence: {kl_mean.item():.6f}")

    return kl_mean


print("\n" + "-"*80)
print("Example: Knowledge Distillation (Student Learning from Teacher)")
print("-"*80)

# Teacher model (large, confident)
teacher_logits = torch.tensor([
    [4.0, 1.0, 0.5, 0.2],  # Very confident about class 0
    [0.3, 3.5, 0.8, 0.4],  # Very confident about class 1
])
teacher_probs = F.softmax(teacher_logits, dim=1)

# Student model (smaller, learning)
student_logits_good = torch.tensor([
    [3.5, 1.2, 0.6, 0.3],  # Close to teacher
    [0.4, 3.2, 0.9, 0.5],  # Close to teacher
])
student_probs_good = F.softmax(student_logits_good, dim=1)

student_logits_bad = torch.tensor([
    [2.0, 2.0, 1.0, 1.0],  # Far from teacher (more uniform)
    [1.0, 2.0, 1.5, 1.2],  # Far from teacher
])
student_probs_bad = F.softmax(student_logits_bad, dim=1)

print("Teacher's confident predictions:")
print(f"Sample 1: {teacher_probs[0]}")
print(f"Sample 2: {teacher_probs[1]}")

print("\n" + "="*40)
print("Good Student (close to teacher)")
print("="*40)
kl_good_scratch = kl_divergence_from_scratch(teacher_probs, student_probs_good)

print("\n" + "="*40)
print("Bad Student (far from teacher)")
print("="*40)
kl_bad_scratch = kl_divergence_from_scratch(teacher_probs, student_probs_bad)

print("\n" + "="*40)
print("PyTorch Implementation")
print("="*40)
kl_good_pytorch = F.kl_div(student_probs_good.log(), teacher_probs, reduction='batchmean')
kl_bad_pytorch = F.kl_div(student_probs_bad.log(), teacher_probs, reduction='batchmean')

print(f"\nGood student KL: {kl_good_pytorch.item():.6f}")
print(f"Bad student KL: {kl_bad_pytorch.item():.6f}")
print(f"\nDifference (good): {abs(kl_good_scratch - kl_good_pytorch).item():.10f}")
print(f"Difference (bad): {abs(kl_bad_scratch - kl_bad_pytorch).item():.10f}")

print("\n" + "="*40)
print("INTERPRETATION:")
print("="*40)
print(f"Lower KL divergence ({kl_good_pytorch.item():.4f}) = student closely mimics teacher")
print(f"Higher KL divergence ({kl_bad_pytorch.item():.4f}) = student predictions differ significantly")


# Demonstrate asymmetry of KL divergence
print("\n" + "-"*80)
print("Demonstrating KL Divergence Asymmetry")
print("-"*80)

p1 = torch.tensor([[0.7, 0.2, 0.1]])
p2 = torch.tensor([[0.3, 0.5, 0.2]])

kl_p1_p2 = F.kl_div(p2.log(), p1, reduction='batchmean')
kl_p2_p1 = F.kl_div(p1.log(), p2, reduction='batchmean')

print(f"Distribution P1: {p1[0]}")
print(f"Distribution P2: {p2[0]}")
print(f"\nKL(P1 || P2) = {kl_p1_p2.item():.6f}")
print(f"KL(P2 || P1) = {kl_p2_p1.item():.6f}")
print(f"\nThese are NOT equal! KL divergence is asymmetric.")


# Visualize KL divergence
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Teacher vs Students probability distributions
x = np.arange(4)
width = 0.25

ax = axes[0, 0]
ax.bar(x - width, teacher_probs[0].numpy(), width, label='Teacher', color='blue', alpha=0.7, edgecolor='black')
ax.bar(x, student_probs_good[0].numpy(), width, label='Good Student', color='green', alpha=0.7, edgecolor='black')
ax.bar(x + width, student_probs_bad[0].numpy(), width, label='Bad Student', color='red', alpha=0.7, edgecolor='black')
ax.set_xlabel('Class', fontsize=11)
ax.set_ylabel('Probability', fontsize=11)
ax.set_title('Knowledge Distillation: Distribution Comparison (Sample 1)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: KL divergence comparison
ax = axes[0, 1]
kl_values = [kl_good_pytorch.item(), kl_bad_pytorch.item()]
colors = ['green', 'red']
bars = ax.bar(['Good Student', 'Bad Student'], kl_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('KL Divergence from Teacher', fontsize=11)
ax.set_title('KL Divergence: Measuring Student-Teacher Alignment', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for bar, kl in zip(bars, kl_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{kl:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 3: KL divergence heatmap
ax = axes[1, 0]
# Create a grid of distributions
n_points = 20
probs_grid = []
for i in range(n_points):
    for j in range(n_points - i):
        k = n_points - i - j
        if k >= 0:
            # Three-class distribution
            p1 = i / n_points
            p2 = j / n_points
            p3 = k / n_points
            if abs(p1 + p2 + p3 - 1.0) < 0.01:  # Valid probability distribution
                probs_grid.append([p1, p2, p3])

probs_grid = torch.tensor(probs_grid)
reference = torch.tensor([[0.5, 0.3, 0.2]])  # Reference distribution

# Calculate KL for each point
kl_grid = []
for p in probs_grid:
    kl = F.kl_div(p.unsqueeze(0).log(), reference, reduction='batchmean')
    kl_grid.append(kl.item())

# Plot scatter
scatter = ax.scatter(probs_grid[:, 0], probs_grid[:, 1], c=kl_grid,
                    cmap='RdYlGn_r', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.plot(reference[0, 0], reference[0, 1], 'b*', markersize=20,
        label='Reference', markeredgecolor='black', markeredgewidth=2)
ax.set_xlabel('P(class 0)', fontsize=11)
ax.set_ylabel('P(class 1)', fontsize=11)
ax.set_title('KL Divergence Landscape (3-class case)', fontsize=12, fontweight='bold')
ax.legend()
plt.colorbar(scatter, ax=ax, label='KL Divergence')
ax.grid(True, alpha=0.3)

# Plot 4: Asymmetry demonstration
ax = axes[1, 1]
# Generate pairs of distributions
dist_pairs = []
kl_forward = []
kl_reverse = []

for i in range(20):
    alpha = i / 19  # 0 to 1
    p = torch.tensor([[0.8, 0.15, 0.05]])
    q = torch.tensor([[0.8 * (1-alpha) + 0.3 * alpha,
                      0.15 * (1-alpha) + 0.5 * alpha,
                      0.05 * (1-alpha) + 0.2 * alpha]])

    dist_pairs.append(alpha)
    kl_forward.append(F.kl_div(q.log(), p, reduction='batchmean').item())
    kl_reverse.append(F.kl_div(p.log(), q, reduction='batchmean').item())

ax.plot(dist_pairs, kl_forward, 'o-', label='KL(P || Q)', linewidth=2, markersize=6, color='blue')
ax.plot(dist_pairs, kl_reverse, 's-', label='KL(Q || P)', linewidth=2, markersize=6, color='red')
ax.set_xlabel('Distribution Difference (α)', fontsize=11)
ax.set_ylabel('KL Divergence', fontsize=11)
ax.set_title('KL Divergence Asymmetry', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / "03_kl_divergence_analysis.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: {VIZ_DIR / '03_kl_divergence_analysis.png'}")
plt.close()


# ============================================================================
# SECTION 4: ACTIVATION FUNCTIONS - ReLU AND VARIANTS (Q31)
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: ACTIVATION FUNCTIONS - ReLU AND VARIANTS")
print("="*80)

print("""
INTERVIEW CONTEXT:
Activation functions introduce non-linearity, enabling neural networks
to learn complex patterns. Without them, deep networks would collapse
to a single linear transformation.

Key Activation Functions for LLMs:

1. ReLU (Rectified Linear Unit):
   f(x) = max(0, x)
   f'(x) = 1 if x > 0, else 0

   Pros: Simple, fast, helps with vanishing gradients
   Cons: "Dying ReLU" problem (neurons stuck at 0)

2. Leaky ReLU:
   f(x) = max(αx, x) where α = 0.01 typically
   f'(x) = 1 if x > 0, else α

   Pros: Fixes dying ReLU, allows small negative values
   Cons: Extra hyperparameter

3. GELU (Gaussian Error Linear Unit):
   f(x) = x * Φ(x) where Φ is CDF of standard normal
   Approximation: f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

   Pros: Smooth, probabilistic interpretation, used in BERT, GPT
   Cons: More computationally expensive

4. SwiGLU (Swish-Gated Linear Unit):
   Used in modern LLMs (LLaMA, PaLM)
   Combines Swish activation with gating
""")


def relu_from_scratch(x: torch.Tensor) -> torch.Tensor:
    """ReLU: max(0, x)"""
    return torch.maximum(torch.zeros_like(x), x)


def relu_derivative(x: torch.Tensor) -> torch.Tensor:
    """ReLU derivative: 1 if x > 0, else 0"""
    return (x > 0).float()


def leaky_relu_from_scratch(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """Leaky ReLU: max(αx, x)"""
    return torch.maximum(alpha * x, x)


def leaky_relu_derivative(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """Leaky ReLU derivative: 1 if x > 0, else α"""
    return torch.where(x > 0, torch.ones_like(x), torch.full_like(x, alpha))


def gelu_from_scratch(x: torch.Tensor) -> torch.Tensor:
    """
    GELU approximation used in practice.
    f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))


def gelu_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    GELU derivative (approximate).
    This is a simplified version for demonstration.
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x**3)
    tanh_out = torch.tanh(tanh_arg)

    # Derivative of tanh argument
    dtanh_arg = sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)

    # Chain rule
    cdf_approx = 0.5 * (1.0 + tanh_out)
    pdf_approx = 0.5 * x * (1 - tanh_out**2) * dtanh_arg

    return cdf_approx + pdf_approx


print("\n" + "-"*80)
print("Comparing Activation Functions")
print("-"*80)

# Test values
x_test = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

print("Input values:", x_test.numpy())
print("\nReLU outputs:", relu_from_scratch(x_test).numpy())
print("ReLU (PyTorch):", F.relu(x_test).numpy())
print("Match:", torch.allclose(relu_from_scratch(x_test), F.relu(x_test)))

print("\nLeaky ReLU outputs:", leaky_relu_from_scratch(x_test).numpy())
print("Leaky ReLU (PyTorch):", F.leaky_relu(x_test, 0.01).numpy())
print("Match:", torch.allclose(leaky_relu_from_scratch(x_test), F.leaky_relu(x_test, 0.01)))

print("\nGELU outputs:", gelu_from_scratch(x_test).numpy())
print("GELU (PyTorch):", F.gelu(x_test).numpy())
print("Match (approximate):", torch.allclose(gelu_from_scratch(x_test), F.gelu(x_test), atol=1e-3))


# Visualize activation functions and their derivatives
print("\n" + "-"*80)
print("Visualizing Activation Functions and Derivatives")
print("-"*80)

x_range = torch.linspace(-3, 3, 200)

# Calculate activations
relu_out = relu_from_scratch(x_range)
leaky_relu_out = leaky_relu_from_scratch(x_range)
gelu_out = gelu_from_scratch(x_range)

# Calculate derivatives
relu_grad = relu_derivative(x_range)
leaky_relu_grad = leaky_relu_derivative(x_range)
gelu_grad = gelu_derivative(x_range)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: ReLU
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x_range, relu_out, linewidth=2.5, color='blue', label='ReLU')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('f(x)', fontsize=11)
ax1.set_title('ReLU: max(0, x)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x_range, relu_grad, linewidth=2.5, color='red', label="ReLU'")
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel("f'(x)", fontsize=11)
ax2.set_title("ReLU Derivative", fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim([-0.2, 1.5])

ax3 = fig.add_subplot(gs[0, 2])
ax3.text(0.5, 0.7, 'ReLU Properties:', ha='center', fontsize=12, fontweight='bold', transform=ax3.transAxes)
ax3.text(0.5, 0.55, '✓ Simple and fast', ha='center', fontsize=10, transform=ax3.transAxes)
ax3.text(0.5, 0.45, '✓ No vanishing gradient (x > 0)', ha='center', fontsize=10, transform=ax3.transAxes)
ax3.text(0.5, 0.35, '✗ Dying ReLU problem', ha='center', fontsize=10, transform=ax3.transAxes, color='red')
ax3.text(0.5, 0.25, '✗ Not differentiable at 0', ha='center', fontsize=10, transform=ax3.transAxes, color='red')
ax3.text(0.5, 0.1, 'Used in: Earlier CNNs, some RNNs', ha='center', fontsize=9, style='italic', transform=ax3.transAxes)
ax3.axis('off')

# Row 2: Leaky ReLU
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(x_range, leaky_relu_out, linewidth=2.5, color='green', label='Leaky ReLU (α=0.01)')
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax4.set_xlabel('x', fontsize=11)
ax4.set_ylabel('f(x)', fontsize=11)
ax4.set_title('Leaky ReLU: max(αx, x)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(x_range, leaky_relu_grad, linewidth=2.5, color='darkgreen', label="Leaky ReLU'")
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax5.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax5.set_xlabel('x', fontsize=11)
ax5.set_ylabel("f'(x)", fontsize=11)
ax5.set_title("Leaky ReLU Derivative", fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()
ax5.set_ylim([-0.2, 1.5])

ax6 = fig.add_subplot(gs[1, 2])
ax6.text(0.5, 0.7, 'Leaky ReLU Properties:', ha='center', fontsize=12, fontweight='bold', transform=ax6.transAxes)
ax6.text(0.5, 0.55, '✓ Fixes dying ReLU', ha='center', fontsize=10, transform=ax6.transAxes)
ax6.text(0.5, 0.45, '✓ Allows negative values', ha='center', fontsize=10, transform=ax6.transAxes)
ax6.text(0.5, 0.35, '✓ Better gradient flow', ha='center', fontsize=10, transform=ax6.transAxes)
ax6.text(0.5, 0.25, '~ Extra hyperparameter α', ha='center', fontsize=10, transform=ax6.transAxes, color='orange')
ax6.text(0.5, 0.1, 'Used in: Various architectures', ha='center', fontsize=9, style='italic', transform=ax6.transAxes)
ax6.axis('off')

# Row 3: GELU
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(x_range, gelu_out, linewidth=2.5, color='purple', label='GELU')
ax7.plot(x_range, x_range, '--', linewidth=1.5, color='gray', alpha=0.5, label='Identity')
ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax7.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax7.set_xlabel('x', fontsize=11)
ax7.set_ylabel('f(x)', fontsize=11)
ax7.set_title('GELU: x·Φ(x)', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend()

ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(x_range, gelu_grad, linewidth=2.5, color='darkviolet', label="GELU'")
ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax8.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax8.set_xlabel('x', fontsize=11)
ax8.set_ylabel("f'(x)", fontsize=11)
ax8.set_title("GELU Derivative", fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.legend()
ax8.set_ylim([-0.2, 1.5])

ax9 = fig.add_subplot(gs[2, 2])
ax9.text(0.5, 0.7, 'GELU Properties:', ha='center', fontsize=12, fontweight='bold', transform=ax9.transAxes)
ax9.text(0.5, 0.55, '✓ Smooth (differentiable)', ha='center', fontsize=10, transform=ax9.transAxes)
ax9.text(0.5, 0.45, '✓ Probabilistic interpretation', ha='center', fontsize=10, transform=ax9.transAxes)
ax9.text(0.5, 0.35, '✓ Better empirical performance', ha='center', fontsize=10, transform=ax9.transAxes)
ax9.text(0.5, 0.25, '~ More expensive to compute', ha='center', fontsize=10, transform=ax9.transAxes, color='orange')
ax9.text(0.5, 0.1, 'Used in: BERT, GPT, most modern LLMs', ha='center', fontsize=9, style='italic', transform=ax9.transAxes)
ax9.axis('off')

plt.savefig(VIZ_DIR / "04_activation_functions_comprehensive.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {VIZ_DIR / '04_activation_functions_comprehensive.png'}")
plt.close()

# Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# All activations together
axes[0].plot(x_range, relu_out, linewidth=2, label='ReLU', color='blue')
axes[0].plot(x_range, leaky_relu_out, linewidth=2, label='Leaky ReLU', color='green')
axes[0].plot(x_range, gelu_out, linewidth=2, label='GELU', color='purple')
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('f(x)', fontsize=12)
axes[0].set_title('Activation Functions Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# All derivatives together
axes[1].plot(x_range, relu_grad, linewidth=2, label="ReLU'", color='blue')
axes[1].plot(x_range, leaky_relu_grad, linewidth=2, label="Leaky ReLU'", color='green')
axes[1].plot(x_range, gelu_grad, linewidth=2, label="GELU'", color='purple')
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_xlabel('x', fontsize=12)
axes[1].set_ylabel("f'(x)", fontsize=12)
axes[1].set_title('Derivatives Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([-0.2, 1.5])

plt.tight_layout()
plt.savefig(VIZ_DIR / "05_activation_comparison.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {VIZ_DIR / '05_activation_comparison.png'}")
plt.close()


# ============================================================================
# SECTION 5: CHAIN RULE AND BACKPROPAGATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: CHAIN RULE AND BACKPROPAGATION")
print("="*80)

print("""
INTERVIEW CONTEXT:
The chain rule is the mathematical foundation of backpropagation,
which enables training of deep neural networks.

Chain Rule:
    If z = f(g(x)), then dz/dx = (dz/dg) * (dg/dx)

In neural networks:
    - Forward pass: compute outputs layer by layer
    - Backward pass: compute gradients using chain rule

Example: Two-layer network
    x -> Linear -> ReLU -> Linear -> Loss

Gradient flow:
    dL/dW1 = (dL/dz2) * (dz2/dz1) * (dz1/dW1)

This is why activation function derivatives matter!
""")


class SimpleNetwork:
    """
    Manually implemented 2-layer network for educational purposes.

    Architecture:
        Input (2) -> Linear (3) -> ReLU -> Linear (1) -> Output
    """

    def __init__(self):
        # Initialize weights with small random values
        self.W1 = torch.randn(2, 3) * 0.1  # Input to hidden
        self.b1 = torch.zeros(3)
        self.W2 = torch.randn(3, 1) * 0.1  # Hidden to output
        self.b2 = torch.zeros(1)

        # Storage for intermediate values (needed for backward pass)
        self.x = None
        self.z1 = None  # Before activation
        self.a1 = None  # After activation
        self.z2 = None  # Final output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with detailed logging."""
        print("\n" + "="*40)
        print("FORWARD PASS")
        print("="*40)

        self.x = x
        print(f"Input x: {x}")
        print(f"  shape: {x.shape}")

        # Layer 1: Linear transformation
        self.z1 = x @ self.W1 + self.b1
        print(f"\nLayer 1 (before activation) z1 = x @ W1 + b1")
        print(f"  z1: {self.z1}")
        print(f"  shape: {self.z1.shape}")

        # ReLU activation
        self.a1 = relu_from_scratch(self.z1)
        print(f"\nReLU activation: a1 = max(0, z1)")
        print(f"  a1: {self.a1}")
        print(f"  (Note: negative values zeroed out)")

        # Layer 2: Linear transformation
        self.z2 = self.a1 @ self.W2 + self.b2
        print(f"\nLayer 2 (output) z2 = a1 @ W2 + b2")
        print(f"  z2: {self.z2}")
        print(f"  shape: {self.z2.shape}")

        return self.z2

    def backward(self, target: torch.Tensor) -> dict:
        """
        Backward pass using chain rule.

        Loss: L = 0.5 * (z2 - target)^2 (MSE)
        """
        print("\n" + "="*40)
        print("BACKWARD PASS (Chain Rule)")
        print("="*40)

        # Compute loss
        loss = 0.5 * (self.z2 - target) ** 2
        print(f"Loss (MSE): {loss.item():.6f}")
        print(f"  L = 0.5 * (z2 - target)^2")
        print(f"  L = 0.5 * ({self.z2.item():.4f} - {target.item():.4f})^2")

        # Gradient of loss w.r.t. output
        # dL/dz2 = z2 - target
        dL_dz2 = self.z2 - target
        print(f"\n1. Gradient of loss w.r.t. output:")
        print(f"   dL/dz2 = z2 - target = {dL_dz2.item():.6f}")

        # Gradient w.r.t. W2 and b2
        # z2 = a1 @ W2 + b2
        # dL/dW2 = a1^T @ dL/dz2
        # dL/db2 = dL/dz2
        dL_dW2 = self.a1.T @ dL_dz2
        dL_db2 = dL_dz2.sum(dim=0)

        print(f"\n2. Gradient w.r.t. Layer 2 parameters:")
        print(f"   dL/dW2 = a1^T @ dL/dz2")
        print(f"   dL/dW2:\n{dL_dW2}")
        print(f"   dL/db2: {dL_db2}")

        # Gradient w.r.t. a1 (chain rule!)
        # dL/da1 = dL/dz2 @ W2^T
        dL_da1 = dL_dz2 @ self.W2.T
        print(f"\n3. Gradient w.r.t. hidden activations (chain rule):")
        print(f"   dL/da1 = dL/dz2 @ W2^T")
        print(f"   dL/da1: {dL_da1}")

        # Gradient w.r.t. z1 (through ReLU)
        # ReLU gradient: 1 if z1 > 0, else 0
        # dL/dz1 = dL/da1 * d(ReLU)/dz1
        relu_grad = relu_derivative(self.z1)
        dL_dz1 = dL_da1 * relu_grad

        print(f"\n4. Gradient through ReLU (chain rule):")
        print(f"   ReLU gradient: {relu_grad}")
        print(f"   dL/dz1 = dL/da1 * d(ReLU)/dz1")
        print(f"   dL/dz1: {dL_dz1}")
        print(f"   (Note: zero gradient where ReLU was inactive)")

        # Gradient w.r.t. W1 and b1
        # z1 = x @ W1 + b1
        # dL/dW1 = x^T @ dL/dz1
        # dL/db1 = dL/dz1
        dL_dW1 = self.x.T @ dL_dz1
        dL_db1 = dL_dz1.sum(dim=0)

        print(f"\n5. Gradient w.r.t. Layer 1 parameters:")
        print(f"   dL/dW1 = x^T @ dL/dz1")
        print(f"   dL/dW1:\n{dL_dW1}")
        print(f"   dL/db1: {dL_db1}")

        print(f"\n" + "="*40)
        print("CHAIN RULE IN ACTION:")
        print("="*40)
        print("dL/dW1 = dL/dz2 * dz2/da1 * da1/dz1 * dz1/dW1")
        print("       = (z2-target) * W2 * ReLU'(z1) * x")
        print("\nEach gradient flows backward through the network,")
        print("multiplied by local derivatives at each layer!")

        return {
            'dL_dW1': dL_dW1,
            'dL_db1': dL_db1,
            'dL_dW2': dL_dW2,
            'dL_db2': dL_db2,
            'loss': loss.item()
        }


print("\n" + "-"*80)
print("Manual Backpropagation Example")
print("-"*80)

# Create simple network
net = SimpleNetwork()

# Input and target
x = torch.tensor([[1.0, 2.0]])
target = torch.tensor([[3.0]])

print(f"Input: {x}")
print(f"Target: {target}")

# Forward pass
output = net.forward(x)

print(f"\nPrediction: {output.item():.4f}")
print(f"Target: {target.item():.4f}")

# Backward pass
gradients = net.backward(target)


# Compare with PyTorch autograd
print("\n" + "-"*80)
print("Verification with PyTorch Autograd")
print("-"*80)

# Create PyTorch network with same weights
W1_torch = torch.tensor(net.W1.numpy(), requires_grad=True)
b1_torch = torch.tensor(net.b1.numpy(), requires_grad=True)
W2_torch = torch.tensor(net.W2.numpy(), requires_grad=True)
b2_torch = torch.tensor(net.b2.numpy(), requires_grad=True)

# Forward pass
z1_torch = x @ W1_torch + b1_torch
a1_torch = F.relu(z1_torch)
z2_torch = a1_torch @ W2_torch + b2_torch

# Loss
loss_torch = 0.5 * (z2_torch - target) ** 2

# Backward pass (PyTorch autograd)
loss_torch.backward()

print("Comparing manual gradients with PyTorch autograd:")
print(f"\ndL/dW1 difference: {torch.abs(gradients['dL_dW1'] - W1_torch.grad).max().item():.10f}")
print(f"dL/db1 difference: {torch.abs(gradients['dL_db1'] - b1_torch.grad).max().item():.10f}")
print(f"dL/dW2 difference: {torch.abs(gradients['dL_dW2'] - W2_torch.grad).max().item():.10f}")
print(f"dL/db2 difference: {torch.abs(gradients['dL_db2'] - b2_torch.grad).max().item():.10f}")

print("\n✓ Manual backpropagation matches PyTorch autograd!")


# Visualize gradient flow
print("\n" + "-"*80)
print("Visualizing Gradient Flow")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Network architecture with gradient magnitudes
ax = axes[0, 0]
ax.text(0.5, 0.9, 'Gradient Flow Through Network', ha='center',
        fontsize=14, fontweight='bold', transform=ax.transAxes)

# Draw network layers
layer_positions = [0.1, 0.4, 0.7]
layer_names = ['Input\n(2)', 'Hidden\n(ReLU)\n(3)', 'Output\n(1)']

for i, (pos, name) in enumerate(zip(layer_positions, layer_names)):
    circle = plt.Circle((pos, 0.5), 0.08, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(pos, 0.5, name, ha='center', va='center', fontsize=9, fontweight='bold')

# Draw connections with gradient magnitudes
grad_W1_mag = torch.abs(gradients['dL_dW1']).mean().item()
grad_W2_mag = torch.abs(gradients['dL_dW2']).mean().item()

# W1 connection
arrow1 = plt.Arrow(0.18, 0.5, 0.14, 0, width=0.1, color='red', alpha=0.7)
ax.add_patch(arrow1)
ax.text(0.25, 0.65, f'∇W1\n{grad_W1_mag:.4f}', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# W2 connection
arrow2 = plt.Arrow(0.48, 0.5, 0.14, 0, width=0.1, color='red', alpha=0.7)
ax.add_patch(arrow2)
ax.text(0.55, 0.65, f'∇W2\n{grad_W2_mag:.4f}', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Loss
ax.text(0.9, 0.5, 'Loss', ha='center', va='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7))
arrow3 = plt.Arrow(0.78, 0.5, 0.07, 0, width=0.08, color='blue', alpha=0.7)
ax.add_patch(arrow3)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Plot 2: Gradient magnitudes by layer
ax = axes[0, 1]
layers = ['W1', 'b1', 'W2', 'b2']
grad_mags = [
    torch.abs(gradients['dL_dW1']).mean().item(),
    torch.abs(gradients['dL_db1']).mean().item(),
    torch.abs(gradients['dL_dW2']).mean().item(),
    torch.abs(gradients['dL_db2']).mean().item(),
]

colors = ['blue', 'lightblue', 'red', 'lightcoral']
bars = ax.bar(layers, grad_mags, color=colors, edgecolor='black', linewidth=2, alpha=0.7)
ax.set_ylabel('Average Gradient Magnitude', fontsize=11)
ax.set_title('Gradient Magnitudes by Parameter', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for bar, mag in zip(bars, grad_mags):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mag:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: ReLU impact on gradients
ax = axes[1, 0]
z1_vals = net.z1[0].numpy()
dL_dz1_vals = gradients['dL_dW1'].sum(dim=0).numpy()

x_pos = np.arange(len(z1_vals))
width = 0.35

bars1 = ax.bar(x_pos - width/2, z1_vals, width, label='Pre-activation (z1)',
               color='green', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x_pos + width/2, dL_dz1_vals, width, label='Gradient (dL/dz1)',
               color='orange', alpha=0.7, edgecolor='black')

ax.set_xlabel('Hidden Unit', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('ReLU Effect: Negative Pre-activations → Zero Gradients', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Unit {i}' for i in range(len(z1_vals))])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

# Plot 4: Chain rule visualization
ax = axes[1, 1]
ax.text(0.5, 0.9, 'Chain Rule in Backpropagation', ha='center',
        fontsize=13, fontweight='bold', transform=ax.transAxes)

chain_text = """
Forward Pass:
  x → z₁ = x·W₁ + b₁
    → a₁ = ReLU(z₁)
      → z₂ = a₁·W₂ + b₂
        → L = ½(z₂ - y)²

Backward Pass (Chain Rule):
  dL/dW₁ = dL/dz₂ · dz₂/da₁ · da₁/dz₁ · dz₁/dW₁
         = (z₂-y) · W₂ · ReLU'(z₁) · x

  Each term is a local gradient!

  This is the power of backpropagation:
  - Compute local gradients
  - Chain them together
  - Efficient O(n) complexity
"""

ax.text(0.05, 0.75, chain_text, ha='left', va='top', fontsize=9,
        family='monospace', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.axis('off')

plt.tight_layout()
plt.savefig(VIZ_DIR / "06_chain_rule_backpropagation.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {VIZ_DIR / '06_chain_rule_backpropagation.png'}")
plt.close()


# ============================================================================
# SECTION 6: LOSS LANDSCAPES
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: LOSS LANDSCAPES")
print("="*80)

print("""
INTERVIEW CONTEXT:
Understanding loss landscapes helps explain:
- Why optimization is challenging
- The role of learning rate
- Local minima vs global minima
- Why initialization matters

We'll visualize a simple 2D loss landscape for a toy problem.
""")


def create_loss_landscape(w1_range, w2_range, X, y):
    """
    Create a loss landscape for a simple linear regression problem.

    Model: y = w1 * x1 + w2 * x2
    Loss: MSE
    """
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Loss = np.zeros_like(W1)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w = torch.tensor([[W1[i, j]], [W2[i, j]]], dtype=torch.float32)
            pred = X @ w
            loss = torch.mean((pred - y) ** 2)
            Loss[i, j] = loss.item()

    return W1, W2, Loss


# Create toy dataset
np.random.seed(42)
n_samples = 20
X_np = np.random.randn(n_samples, 2)
true_w = np.array([[2.0], [3.0]])
y_np = X_np @ true_w + np.random.randn(n_samples, 1) * 0.5

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

print(f"Dataset: {n_samples} samples")
print(f"True weights: w1={true_w[0, 0]:.2f}, w2={true_w[1, 0]:.2f}")

# Create loss landscape
w_range = np.linspace(-1, 5, 100)
W1, W2, Loss = create_loss_landscape(w_range, w_range, X, y)

print(f"\nLoss landscape computed over {len(w_range)}x{len(w_range)} grid")

# Perform gradient descent and record trajectory
w_init = torch.tensor([[0.0], [0.0]], requires_grad=True)
learning_rate = 0.1
n_steps = 50

trajectory = [w_init.detach().numpy().copy()]
losses = []

for step in range(n_steps):
    pred = X @ w_init
    loss = torch.mean((pred - y) ** 2)
    losses.append(loss.item())

    loss.backward()

    with torch.no_grad():
        w_init -= learning_rate * w_init.grad
        trajectory.append(w_init.numpy().copy())
        w_init.grad.zero_()

trajectory = np.array(trajectory)

print(f"\nGradient descent: {n_steps} steps")
print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Final weights: w1={trajectory[-1][0, 0]:.2f}, w2={trajectory[-1][1, 0]:.2f}")


# Visualize loss landscape
fig = plt.figure(figsize=(16, 5))

# Plot 1: 2D contour plot with trajectory
ax1 = fig.add_subplot(131)
contour = ax1.contour(W1, W2, Loss, levels=20, cmap='viridis', alpha=0.6)
ax1.clabel(contour, inline=True, fontsize=8)
contourf = ax1.contourf(W1, W2, Loss, levels=20, cmap='viridis', alpha=0.3)

# Plot trajectory
ax1.plot(trajectory[:, 0, 0], trajectory[:, 1, 0], 'r.-', linewidth=2,
         markersize=8, label='Gradient Descent Path')
ax1.plot(trajectory[0, 0, 0], trajectory[0, 1, 0], 'go', markersize=12,
         label='Start', markeredgecolor='black', markeredgewidth=2)
ax1.plot(trajectory[-1, 0, 0], trajectory[-1, 1, 0], 'r*', markersize=18,
         label='End', markeredgecolor='black', markeredgewidth=2)
ax1.plot(true_w[0, 0], true_w[1, 0], 'b*', markersize=18,
         label='True Optimum', markeredgecolor='black', markeredgewidth=2)

ax1.set_xlabel('w₁', fontsize=12)
ax1.set_ylabel('w₂', fontsize=12)
ax1.set_title('Loss Landscape with Gradient Descent Trajectory', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.colorbar(contourf, ax=ax1, label='Loss')

# Plot 2: 3D surface plot
ax2 = fig.add_subplot(132, projection='3d')
surf = ax2.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.6, edgecolor='none')
ax2.plot(trajectory[:, 0, 0], trajectory[:, 1, 0],
         [losses[min(i, len(losses)-1)] for i in range(len(trajectory))],
         'r.-', linewidth=2, markersize=6, label='GD Path')
ax2.set_xlabel('w₁', fontsize=11)
ax2.set_ylabel('w₂', fontsize=11)
ax2.set_zlabel('Loss', fontsize=11)
ax2.set_title('3D Loss Surface', fontsize=13, fontweight='bold')
plt.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)

# Plot 3: Loss over iterations
ax3 = fig.add_subplot(133)
ax3.plot(losses, linewidth=2, color='blue', marker='o', markersize=4)
ax3.set_xlabel('Iteration', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Loss Convergence During Training', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Add annotations
ax3.annotate('Fast initial decrease', xy=(5, losses[5]), xytext=(15, losses[5]*3),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10)
ax3.annotate('Slower convergence', xy=(40, losses[40]), xytext=(25, losses[40]*0.3),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=10)

plt.tight_layout()
plt.savefig(VIZ_DIR / "07_loss_landscape.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {VIZ_DIR / '07_loss_landscape.png'}")
plt.close()


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE SUMMARY FOR INTERVIEW PREPARATION")
print("="*80)

summary = """
KEY TAKEAWAYS FOR LLM INTERVIEWS:

1. CROSS-ENTROPY LOSS:
   - Foundation of language model training
   - Formula: L = -log(p_target)
   - Penalizes confident wrong predictions heavily
   - Used with softmax for multi-class classification

2. PERPLEXITY:
   - Primary LLM evaluation metric
   - PPL = exp(average cross-entropy)
   - Lower is better (less "surprised" by test data)
   - Interpretable: PPL=K means like choosing from K options

3. KL DIVERGENCE:
   - Measures distribution difference
   - KL(P || Q) = sum(P * log(P/Q))
   - Used in knowledge distillation, VAEs, RLHF
   - NOT symmetric! KL(P||Q) ≠ KL(Q||P)

4. ACTIVATION FUNCTIONS:
   - ReLU: Simple, fast, but can "die"
   - Leaky ReLU: Fixes dying ReLU problem
   - GELU: Smooth, used in modern LLMs (BERT, GPT)
   - Derivatives crucial for backpropagation!

5. CHAIN RULE & BACKPROPAGATION:
   - Foundation of neural network training
   - Efficiently computes gradients layer by layer
   - dL/dW1 = dL/dz2 * dz2/da1 * da1/dz1 * dz1/dW1
   - Each layer computes local gradient

6. LOSS LANDSCAPES:
   - Visualizes optimization challenges
   - Shows why learning rate matters
   - Explains local vs global minima
   - Convex in simple cases, complex in deep networks

INTERVIEW TIPS:
- Be able to derive cross-entropy from first principles
- Explain why perplexity is better than raw loss
- Describe when to use KL divergence vs cross-entropy
- Know which activation functions are used in modern LLMs
- Walk through backpropagation for a simple network
- Discuss optimization challenges (local minima, saddle points)
"""

print(summary)

print("\n" + "="*80)
print("ALL VISUALIZATIONS SAVED TO:")
print(f"  {VIZ_DIR}")
print("="*80)

# List all saved files
saved_files = sorted(VIZ_DIR.glob("*.png"))
print("\nGenerated visualizations:")
for i, file in enumerate(saved_files, 1):
    print(f"  {i}. {file.name}")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nYou now have a comprehensive understanding of loss functions")
print("and mathematical foundations critical for LLM interviews.")
print("\nReview the visualizations and run this script multiple times")
print("to reinforce your understanding. Good luck with your interviews!")
