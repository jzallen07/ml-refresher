"""
Comprehensive Embeddings Demo for LLM Interviews
=================================================

This demo covers all essential embedding concepts for technical interviews:
1. nn.Embedding creation and usage
2. Equivalence between embedding lookup and one-hot matrix multiplication
3. Different initialization methods
4. Cosine similarity computation
5. Embedding space visualization
6. Weight tying (input/output embeddings)
7. How embeddings are learned through backpropagation

Author: Interview Prep
Date: 2025-12-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory
OUTPUT_DIR = Path("/Users/zack/dev/ml-refresher/data/interview_viz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EMBEDDINGS DEMO FOR LLM INTERVIEWS")
print("="*80)


# ============================================================================
# SECTION 1: Basic Embedding Creation and Usage
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: Basic Embedding Creation and Usage")
print("="*80)

# Create a small vocabulary for demonstration
vocab = ["<PAD>", "<UNK>", "hello", "world", "machine", "learning", "neural", "network"]
vocab_size = len(vocab)
embedding_dim = 4  # Small dimension for easy visualization

# Word to index mapping
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

print(f"\nVocabulary: {vocab}")
print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")

# Create embedding layer
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

print(f"\nEmbedding layer created:")
print(f"  - Weight matrix shape: {embedding_layer.weight.shape}")
print(f"  - Total parameters: {embedding_layer.weight.numel()}")

# Look up embeddings for some words
word_indices = torch.LongTensor([word2idx["hello"], word2idx["world"], word2idx["learning"]])
word_embeddings = embedding_layer(word_indices)

print(f"\nEmbedding lookup example:")
print(f"  - Input indices: {word_indices.tolist()} -> {[idx2word[i] for i in word_indices.tolist()]}")
print(f"  - Output shape: {word_embeddings.shape}")
print(f"  - Embedding for 'hello':\n{word_embeddings[0]}")


# ============================================================================
# SECTION 2: Embedding Lookup = One-Hot @ Weight Matrix
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: Embedding Lookup ≡ One-Hot @ Weight Matrix")
print("="*80)

# Demonstrate that embedding lookup is equivalent to one-hot encoding @ weight matrix
word_idx = word2idx["machine"]
print(f"\nDemonstrating equivalence for word: '{idx2word[word_idx]}' (index {word_idx})")

# Method 1: Direct embedding lookup
embedding_direct = embedding_layer(torch.LongTensor([word_idx]))
print(f"\nMethod 1 - Direct embedding lookup:")
print(f"  Shape: {embedding_direct.shape}")
print(f"  Values: {embedding_direct.squeeze()}")

# Method 2: One-hot encoding @ weight matrix
one_hot = F.one_hot(torch.LongTensor([word_idx]), num_classes=vocab_size).float()
embedding_onehot = one_hot @ embedding_layer.weight

print(f"\nMethod 2 - One-hot @ Weight matrix:")
print(f"  One-hot shape: {one_hot.shape}")
print(f"  One-hot vector: {one_hot.squeeze().tolist()}")
print(f"  Weight matrix shape: {embedding_layer.weight.shape}")
print(f"  Result shape: {embedding_onehot.shape}")
print(f"  Values: {embedding_onehot.squeeze()}")

# Verify they are identical
are_equal = torch.allclose(embedding_direct, embedding_onehot, atol=1e-6)
print(f"\nAre the two methods equivalent? {are_equal}")
print(f"Max difference: {torch.max(torch.abs(embedding_direct - embedding_onehot)).item():.2e}")

# Explain why embedding lookup is more efficient
print("\n" + "-"*80)
print("EFFICIENCY COMPARISON:")
print("-"*80)
print("Why use embedding lookup instead of one-hot multiplication?")
print(f"  1. Memory: One-hot requires {vocab_size} floats, embedding lookup uses 1 integer")
print(f"  2. Computation: One-hot does {vocab_size} × {embedding_dim} = {vocab_size * embedding_dim} multiplications")
print(f"     Embedding lookup does 0 multiplications (direct indexing)")
print(f"  3. For vocab_size=50k, embedding_dim=768: One-hot needs 38.4M multiplications vs 0!")


# ============================================================================
# SECTION 3: Different Initialization Methods
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: Different Initialization Methods")
print("="*80)

# Default initialization (uniform)
emb_default = nn.Embedding(vocab_size, embedding_dim)
print(f"\n1. Default Initialization (Uniform):")
print(f"   Distribution: U(-sqrt(k), sqrt(k)) where k = 1/embedding_dim")
print(f"   Weight matrix stats:")
print(f"     Mean: {emb_default.weight.mean().item():.4f}")
print(f"     Std:  {emb_default.weight.std().item():.4f}")
print(f"     Min:  {emb_default.weight.min().item():.4f}")
print(f"     Max:  {emb_default.weight.max().item():.4f}")

# Xavier/Glorot initialization
emb_xavier = nn.Embedding(vocab_size, embedding_dim)
nn.init.xavier_uniform_(emb_xavier.weight)
print(f"\n2. Xavier/Glorot Uniform Initialization:")
print(f"   Distribution: U(-a, a) where a = sqrt(6/(fan_in + fan_out))")
print(f"   Weight matrix stats:")
print(f"     Mean: {emb_xavier.weight.mean().item():.4f}")
print(f"     Std:  {emb_xavier.weight.std().item():.4f}")
print(f"     Min:  {emb_xavier.weight.min().item():.4f}")
print(f"     Max:  {emb_xavier.weight.max().item():.4f}")

# Normal initialization
emb_normal = nn.Embedding(vocab_size, embedding_dim)
nn.init.normal_(emb_normal.weight, mean=0.0, std=0.02)
print(f"\n3. Normal Initialization (BERT-style):")
print(f"   Distribution: N(0, 0.02²)")
print(f"   Weight matrix stats:")
print(f"     Mean: {emb_normal.weight.mean().item():.4f}")
print(f"     Std:  {emb_normal.weight.std().item():.4f}")
print(f"     Min:  {emb_normal.weight.min().item():.4f}")
print(f"     Max:  {emb_normal.weight.max().item():.4f}")

# Zero initialization (for special tokens like padding)
emb_zero = nn.Embedding(vocab_size, embedding_dim)
nn.init.zeros_(emb_zero.weight)
emb_zero.weight.data[1:] = emb_normal.weight.data[1:]  # Keep only first row as zeros
print(f"\n4. Zero Initialization (for special tokens):")
print(f"   First row (PAD token): {emb_zero.weight[0].tolist()}")
print(f"   Second row (UNK token): {emb_zero.weight[1][:4].tolist()}...")

# Visualize initialization distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Embedding Initialization Methods", fontsize=16, fontweight='bold')

init_methods = [
    (emb_default, "Default (Uniform)", axes[0, 0]),
    (emb_xavier, "Xavier Uniform", axes[0, 1]),
    (emb_normal, "Normal (BERT-style)", axes[1, 0]),
    (emb_zero, "Mixed (Zero + Normal)", axes[1, 1])
]

for emb, title, ax in init_methods:
    weights_flat = emb.weight.detach().numpy().flatten()
    ax.hist(weights_flat, bins=50, alpha=0.7, edgecolor='black')
    ax.set_title(f"{title}\nMean: {weights_flat.mean():.3f}, Std: {weights_flat.std():.3f}")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "embedding_initialization_methods.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved initialization visualization to {OUTPUT_DIR / 'embedding_initialization_methods.png'}")
plt.close()


# ============================================================================
# SECTION 4: Cosine Similarity Between Embeddings
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: Cosine Similarity Between Embeddings")
print("="*80)

# Create embeddings with higher dimension for better demonstration
embedding_dim_large = 16
emb_large = nn.Embedding(vocab_size, embedding_dim_large)
nn.init.normal_(emb_large.weight, mean=0.0, std=0.02)

# Manually adjust some embeddings to create semantic relationships
with torch.no_grad():
    # Make "neural" and "network" similar
    base_vec = torch.randn(embedding_dim_large) * 0.02
    emb_large.weight[word2idx["neural"]] = base_vec + torch.randn(embedding_dim_large) * 0.005
    emb_large.weight[word2idx["network"]] = base_vec + torch.randn(embedding_dim_large) * 0.005

    # Make "machine" and "learning" similar
    base_vec2 = torch.randn(embedding_dim_large) * 0.02
    emb_large.weight[word2idx["machine"]] = base_vec2 + torch.randn(embedding_dim_large) * 0.005
    emb_large.weight[word2idx["learning"]] = base_vec2 + torch.randn(embedding_dim_large) * 0.005

print(f"\nComputing cosine similarities between word embeddings:")
print(f"Embedding dimension: {embedding_dim_large}")

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

# Compute pairwise similarities
word_pairs = [
    ("neural", "network"),
    ("machine", "learning"),
    ("neural", "machine"),
    ("hello", "world"),
    ("hello", "learning"),
]

print(f"\n{'Word 1':<12} {'Word 2':<12} {'Cosine Similarity':<20} {'Interpretation'}")
print("-" * 70)

for word1, word2 in word_pairs:
    emb1 = emb_large.weight[word2idx[word1]]
    emb2 = emb_large.weight[word2idx[word2]]
    sim = cosine_similarity(emb1, emb2)

    if sim > 0.7:
        interpretation = "Very similar"
    elif sim > 0.3:
        interpretation = "Somewhat similar"
    elif sim > 0:
        interpretation = "Slightly similar"
    else:
        interpretation = "Different"

    print(f"{word1:<12} {word2:<12} {sim:>18.4f}  {interpretation}")

# Compute full similarity matrix
similarity_matrix = torch.zeros(vocab_size, vocab_size)
for i in range(vocab_size):
    for j in range(vocab_size):
        similarity_matrix[i, j] = cosine_similarity(
            emb_large.weight[i],
            emb_large.weight[j]
        )

# Visualize similarity matrix
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(similarity_matrix.numpy(), cmap='RdYlBu', vmin=-1, vmax=1)
ax.set_xticks(range(vocab_size))
ax.set_yticks(range(vocab_size))
ax.set_xticklabels(vocab, rotation=45, ha='right')
ax.set_yticklabels(vocab)
ax.set_title("Cosine Similarity Matrix of Word Embeddings", fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Cosine Similarity", rotation=270, labelpad=20)

# Add text annotations
for i in range(vocab_size):
    for j in range(vocab_size):
        text = ax.text(j, i, f"{similarity_matrix[i, j].item():.2f}",
                      ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "embedding_similarity_matrix.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved similarity matrix to {OUTPUT_DIR / 'embedding_similarity_matrix.png'}")
plt.close()


# ============================================================================
# SECTION 5: 2D Visualization of Embedding Space
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: 2D Visualization of Embedding Space")
print("="*80)

# Method 1: PCA projection
print("\nMethod 1: PCA Projection")
embeddings_np = emb_large.weight.detach().numpy()
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_np)

print(f"  Original dimension: {embedding_dim_large}")
print(f"  Projected dimension: 2")
print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"  Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Method 2: Manual projection (take first 2 dimensions)
print("\nMethod 2: Manual Projection (first 2 dimensions)")
embeddings_manual = embeddings_np[:, :2]
print(f"  Simply taking first 2 dimensions of {embedding_dim_large}D space")

# Create side-by-side visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# PCA visualization
ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=200, c='steelblue', alpha=0.6, edgecolors='black')
for i, word in enumerate(vocab):
    ax1.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=11, fontweight='bold', ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')

ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
ax1.set_title("PCA Projection of Embedding Space", fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

# Manual projection visualization
ax2.scatter(embeddings_manual[:, 0], embeddings_manual[:, 1], s=200, c='coral', alpha=0.6, edgecolors='black')
for i, word in enumerate(vocab):
    ax2.annotate(word, (embeddings_manual[i, 0], embeddings_manual[i, 1]),
                fontsize=11, fontweight='bold', ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')

ax2.set_xlabel("Dimension 1", fontsize=12)
ax2.set_ylabel("Dimension 2", fontsize=12)
ax2.set_title("Manual Projection (First 2 Dimensions)", fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "embedding_space_visualization.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved embedding space visualization to {OUTPUT_DIR / 'embedding_space_visualization.png'}")
plt.close()

# Additional 3D visualization with trajectory showing semantic relationships
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Use PCA for 3D
pca_3d = PCA(n_components=3)
embeddings_3d = pca_3d.fit_transform(embeddings_np)

# Plot points
scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2],
                     s=200, c=range(vocab_size), cmap='viridis', alpha=0.6, edgecolors='black')

# Add labels
for i, word in enumerate(vocab):
    ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2],
            word, fontsize=10, fontweight='bold')

# Draw lines between related words
related_pairs = [
    ("neural", "network", 'red'),
    ("machine", "learning", 'blue'),
]

for word1, word2, color in related_pairs:
    idx1, idx2 = word2idx[word1], word2idx[word2]
    ax.plot([embeddings_3d[idx1, 0], embeddings_3d[idx2, 0]],
            [embeddings_3d[idx1, 1], embeddings_3d[idx2, 1]],
            [embeddings_3d[idx1, 2], embeddings_3d[idx2, 2]],
            color=color, linewidth=2, linestyle='--', alpha=0.7, label=f"{word1}-{word2}")

ax.set_xlabel(f"PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})", fontsize=11)
ax.set_ylabel(f"PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})", fontsize=11)
ax.set_zlabel(f"PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})", fontsize=11)
ax.set_title("3D PCA Projection of Embedding Space", fontsize=14, fontweight='bold', pad=20)
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "embedding_space_3d.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved 3D embedding space to {OUTPUT_DIR / 'embedding_space_3d.png'}")
plt.close()


# ============================================================================
# SECTION 6: Weight Tying (Input/Output Embeddings)
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: Weight Tying (Input/Output Embeddings)")
print("="*80)

class SimpleLanguageModel(nn.Module):
    """Simple language model demonstrating weight tying."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, tie_weights=True):
        super().__init__()
        self.tie_weights = tie_weights

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Simple hidden layer
        self.hidden = nn.Linear(embedding_dim, hidden_dim)

        # Output projection back to embedding dimension
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

        # Output layer (vocabulary projection)
        if tie_weights:
            # Share weights between input embeddings and output layer
            self.output = nn.Linear(embedding_dim, vocab_size, bias=False)
            self.output.weight = self.embedding.weight  # Weight tying!
        else:
            # Separate weights
            self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        hidden = F.relu(self.hidden(emb))  # (batch_size, seq_len, hidden_dim)
        output_emb = self.output_proj(hidden)  # (batch_size, seq_len, embedding_dim)
        logits = self.output(output_emb)  # (batch_size, seq_len, vocab_size)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Create models with and without weight tying
vocab_size_demo = 1000
embedding_dim_demo = 256
hidden_dim_demo = 512

model_tied = SimpleLanguageModel(vocab_size_demo, embedding_dim_demo, hidden_dim_demo, tie_weights=True)
model_separate = SimpleLanguageModel(vocab_size_demo, embedding_dim_demo, hidden_dim_demo, tie_weights=False)

print("\nWeight Tying Comparison:")
print("-" * 70)
print(f"Model with TIED weights:")
print(f"  Total parameters: {model_tied.count_parameters():,}")
print(f"  Input embedding is SAME as output layer weights")

print(f"\nModel with SEPARATE weights:")
print(f"  Total parameters: {model_separate.count_parameters():,}")
print(f"  Input embedding is DIFFERENT from output layer weights")

params_saved = model_separate.count_parameters() - model_tied.count_parameters()
reduction_pct = (params_saved / model_separate.count_parameters()) * 100

print(f"\nParameter reduction: {params_saved:,} ({reduction_pct:.1f}%)")
print(f"Saved parameters = vocab_size × embedding_dim = {vocab_size_demo} × {embedding_dim_demo} = {vocab_size_demo * embedding_dim_demo:,}")

# Verify weight tying
with torch.no_grad():
    test_input = torch.LongTensor([[0, 1, 2]])

    # Check if weights are actually shared
    emb_weight_id = id(model_tied.embedding.weight)
    out_weight_id = id(model_tied.output.weight)

    print(f"\nVerifying weight sharing:")
    print(f"  Input embedding weight tensor ID: {emb_weight_id}")
    print(f"  Output layer weight tensor ID: {out_weight_id}")
    print(f"  Are they the same object? {emb_weight_id == out_weight_id}")

    # Modify embedding and verify output changes
    original_weight = model_tied.embedding.weight[0].clone()
    model_tied.embedding.weight[0] = torch.ones_like(model_tied.embedding.weight[0])

    print(f"\nAfter modifying embedding weight[0]:")
    print(f"  Embedding weight[0]: {model_tied.embedding.weight[0][:5]}...")
    print(f"  Output weight[0]: {model_tied.output.weight[0][:5]}...")
    print(f"  Are they identical? {torch.allclose(model_tied.embedding.weight[0], model_tied.output.weight[0])}")

    # Restore
    model_tied.embedding.weight[0] = original_weight

# Visualize architecture
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Model with tied weights
ax1.text(0.5, 0.9, "Input Tokens", ha='center', va='center', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))
ax1.arrow(0.5, 0.85, 0, -0.1, head_width=0.05, head_length=0.03, fc='black', ec='black')

ax1.text(0.5, 0.7, f"Embedding\n({vocab_size_demo} × {embedding_dim_demo})", ha='center', va='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))
ax1.arrow(0.5, 0.65, 0, -0.1, head_width=0.05, head_length=0.03, fc='black', ec='black')

ax1.text(0.5, 0.5, f"Hidden Layer\n({embedding_dim_demo} → {hidden_dim_demo})", ha='center', va='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))
ax1.arrow(0.5, 0.45, 0, -0.1, head_width=0.05, head_length=0.03, fc='black', ec='black')

ax1.text(0.5, 0.3, f"Output Projection\n({hidden_dim_demo} → {embedding_dim_demo})", ha='center', va='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))
ax1.arrow(0.5, 0.25, 0, -0.1, head_width=0.05, head_length=0.03, fc='black', ec='black')

ax1.text(0.5, 0.1, f"Output Layer (TIED)\n({embedding_dim_demo} × {vocab_size_demo})", ha='center', va='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='black', linewidth=2))

# Draw tie connection
ax1.annotate('', xy=(0.7, 0.7), xytext=(0.7, 0.1),
            arrowprops=dict(arrowstyle='<->', color='red', lw=3, linestyle='--'))
ax1.text(0.85, 0.4, 'SHARED\nWEIGHTS', ha='center', va='center', fontsize=10,
         fontweight='bold', color='red', rotation=90)

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.set_title("Weight Tying (Fewer Parameters)", fontsize=13, fontweight='bold')

# Model with separate weights
ax2.text(0.5, 0.9, "Input Tokens", ha='center', va='center', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))
ax2.arrow(0.5, 0.85, 0, -0.1, head_width=0.05, head_length=0.03, fc='black', ec='black')

ax2.text(0.5, 0.7, f"Embedding\n({vocab_size_demo} × {embedding_dim_demo})", ha='center', va='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))
ax2.arrow(0.5, 0.65, 0, -0.1, head_width=0.05, head_length=0.03, fc='black', ec='black')

ax2.text(0.5, 0.5, f"Hidden Layer\n({embedding_dim_demo} → {hidden_dim_demo})", ha='center', va='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))
ax2.arrow(0.5, 0.45, 0, -0.1, head_width=0.05, head_length=0.03, fc='black', ec='black')

ax2.text(0.5, 0.3, f"Output Projection\n({hidden_dim_demo} → {embedding_dim_demo})", ha='center', va='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))
ax2.arrow(0.5, 0.25, 0, -0.1, head_width=0.05, head_length=0.03, fc='black', ec='black')

ax2.text(0.5, 0.1, f"Output Layer (SEPARATE)\n({embedding_dim_demo} × {vocab_size_demo})", ha='center', va='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='plum', edgecolor='black', linewidth=2))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.set_title("Separate Weights (More Parameters)", fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "weight_tying_architecture.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved weight tying visualization to {OUTPUT_DIR / 'weight_tying_architecture.png'}")
plt.close()


# ============================================================================
# SECTION 7: How Embeddings Are Learned (Gradient Updates)
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: How Embeddings Are Learned Through Backpropagation")
print("="*80)

# Create a simple embedding layer
vocab_size_train = 5
embedding_dim_train = 3
emb_train = nn.Embedding(vocab_size_train, embedding_dim_train)

# Initialize with small values for visibility
with torch.no_grad():
    emb_train.weight.fill_(0.1)

print(f"\nSimple training example:")
print(f"  Vocabulary size: {vocab_size_train}")
print(f"  Embedding dimension: {embedding_dim_train}")

print(f"\nInitial embedding weights:")
print(emb_train.weight.data)

# Create a simple task: predict next word
# Sequence: [0, 1] -> target: 2
input_seq = torch.LongTensor([0, 1])
target = torch.LongTensor([2])

# Simple linear layer for prediction
linear = nn.Linear(embedding_dim_train, vocab_size_train)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(emb_train.parameters()) + list(linear.parameters()), lr=0.5)

print(f"\nTraining setup:")
print(f"  Input sequence: {input_seq.tolist()}")
print(f"  Target: {target.tolist()}")
print(f"  Loss function: CrossEntropyLoss")
print(f"  Optimizer: SGD with lr=0.5")

# Store history for visualization
history = {
    'weights': [emb_train.weight.data.clone()],
    'losses': [],
    'gradients': []
}

# Training loop
num_epochs = 10
print(f"\n{'Epoch':<8} {'Loss':<12} {'Gradient Norm':<20} {'Weight Change'}")
print("-" * 70)

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    embeddings = emb_train(input_seq)  # (2, 3)
    # Simple aggregation: mean pooling
    pooled = embeddings.mean(dim=0, keepdim=True)  # (1, 3)
    logits = linear(pooled)  # (1, vocab_size)
    loss = criterion(logits, target)

    # Backward pass
    loss.backward()

    # Store gradient before update
    grad_norm = emb_train.weight.grad.norm().item()
    history['gradients'].append(grad_norm)
    history['losses'].append(loss.item())

    # Calculate weight change
    old_weight = emb_train.weight.data.clone()

    # Update
    optimizer.step()

    # Track changes
    weight_change = (emb_train.weight.data - old_weight).norm().item()
    history['weights'].append(emb_train.weight.data.clone())

    print(f"{epoch:<8} {loss.item():<12.6f} {grad_norm:<20.6f} {weight_change:.6f}")

print(f"\nFinal embedding weights:")
print(emb_train.weight.data)

print(f"\nKey observations:")
print(f"  1. Only embeddings for tokens in input_seq [0, 1] receive gradients")
print(f"  2. Embedding for token 2 (target) is NOT updated directly")
print(f"  3. Embeddings change to minimize prediction error")

# Show which embeddings changed
initial_weights = history['weights'][0]
final_weights = history['weights'][-1]
weight_changes = (final_weights - initial_weights).abs()

print(f"\nPer-token weight changes (L1 norm):")
for i in range(vocab_size_train):
    change = weight_changes[i].sum().item()
    status = "✓ UPDATED" if change > 0.01 else "○ unchanged"
    print(f"  Token {i}: {change:.6f}  {status}")

# Visualize learning dynamics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curve
ax = axes[0, 0]
ax.plot(history['losses'], marker='o', linewidth=2, markersize=6)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_title("Training Loss Over Time", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Gradient norm
ax = axes[0, 1]
ax.plot(history['gradients'], marker='s', linewidth=2, markersize=6, color='orange')
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Gradient Norm", fontsize=11)
ax.set_title("Gradient Magnitude Over Time", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Weight evolution for each token
ax = axes[1, 0]
for token_id in range(vocab_size_train):
    # Track first dimension of embedding for simplicity
    weight_trajectory = [w[token_id, 0].item() for w in history['weights']]
    linestyle = '-' if token_id in input_seq else '--'
    alpha = 1.0 if token_id in input_seq else 0.4
    label = f"Token {token_id}" + (" (in input)" if token_id in input_seq else " (not in input)")
    ax.plot(weight_trajectory, marker='o', linewidth=2, linestyle=linestyle,
            alpha=alpha, label=label, markersize=4)

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Weight Value (dim 0)", fontsize=11)
ax.set_title("Embedding Weight Evolution (First Dimension)", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Final weight heatmap
ax = axes[1, 1]
im = ax.imshow(final_weights.numpy(), cmap='coolwarm', aspect='auto')
ax.set_xlabel("Embedding Dimension", fontsize=11)
ax.set_ylabel("Token ID", fontsize=11)
ax.set_title("Final Embedding Weights", fontsize=12, fontweight='bold')
ax.set_xticks(range(embedding_dim_train))
ax.set_yticks(range(vocab_size_train))

# Annotate which tokens were in input
for token_id in range(vocab_size_train):
    label = "← in input" if token_id in input_seq else ""
    if label:
        ax.text(embedding_dim_train + 0.1, token_id, label,
                fontsize=9, va='center', fontweight='bold', color='red')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Weight Value", rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "embedding_learning_dynamics.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved learning dynamics to {OUTPUT_DIR / 'embedding_learning_dynamics.png'}")
plt.close()

# Detailed gradient flow explanation
print("\n" + "="*80)
print("DETAILED GRADIENT FLOW EXPLANATION")
print("="*80)

print("""
During backpropagation:

1. Forward Pass:
   - Input tokens [0, 1] → Embedding lookup → Get embeddings E[0] and E[1]
   - Mean pooling → Average embeddings
   - Linear layer → Logits
   - CrossEntropy loss with target token 2

2. Backward Pass (Gradient Flow):

   ∂Loss/∂Logits → ∂Logits/∂Linear_weights → ∂Linear_output/∂Pooled
   → ∂Pooled/∂Embeddings → ∂Embeddings/∂Embedding_weights

   Key insight: Only E[0] and E[1] receive gradients!
   - E[0].grad = ∂Loss/∂E[0] ≠ 0  (was in input)
   - E[1].grad = ∂Loss/∂E[1] ≠ 0  (was in input)
   - E[2].grad = 0  (target, but not in input)
   - E[3].grad = 0  (not used)
   - E[4].grad = 0  (not used)

3. Weight Update:
   - E[i] ← E[i] - learning_rate × ∂Loss/∂E[i]
   - Only E[0] and E[1] are updated
   - This is why embedding layers are SPARSE updates

4. Why This Matters:
   - Embeddings learn from the contexts they appear in
   - Rare words update less frequently (fewer gradient updates)
   - Common words get more gradient updates
   - This is a form of implicit regularization
""")


# ============================================================================
# BONUS: Common Interview Questions
# ============================================================================
print("\n" + "="*80)
print("COMMON INTERVIEW QUESTIONS & ANSWERS")
print("="*80)

qa_pairs = [
    ("Q1: Why use embeddings instead of one-hot encoding?",
     """A: Three main reasons:
     1. EFFICIENCY: One-hot is sparse and requires vocab_size × embedding_dim multiplications.
        Embedding lookup is just indexing (O(1) operation).
     2. MEMORY: One-hot stores vocab_size floats per token, embedding stores 1 integer.
     3. LEARNING: Embeddings can capture semantic relationships. One-hot vectors are
        orthogonal and don't capture any relationships."""),

    ("Q2: What is weight tying and why use it?",
     """A: Weight tying shares the embedding matrix between input embeddings and output
     projection layer. Benefits:
     1. Reduces parameters by vocab_size × embedding_dim
     2. For large vocab (50k) and embedding dim (768): saves 38M parameters!
     3. Acts as regularization - forces consistency between input and output spaces
     4. Used in GPT, BERT, and most modern transformers"""),

    ("Q3: How are embeddings initialized?",
     """A: Common methods:
     1. Random Uniform: U(-1/√d, 1/√d) - PyTorch default
     2. Xavier/Glorot: U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
     3. Normal: N(0, 0.02²) - Used in BERT
     4. Zero for special tokens (like padding)
     Choice affects training stability and convergence speed."""),

    ("Q4: Why do only some embeddings update during training?",
     """A: Sparse gradients! Only embeddings that appear in the current batch receive
     gradients through backpropagation. This means:
     1. Rare words update slowly (fewer gradient updates)
     2. Common words update frequently
     3. This is actually beneficial - rare words need fewer updates
     4. Can lead to cold-start problem for very rare words"""),

    ("Q5: What's the relationship between embedding dimension and model performance?",
     """A: Trade-offs:
     - Higher dimension: More expressiveness, but more parameters and slower training
     - Lower dimension: Faster, but may not capture complex relationships
     - Typical values: 128-256 (small models), 512-1024 (medium), 1024-4096 (large)
     - Rule of thumb: embedding_dim ≈ ⁴√vocabulary_size (but varies widely)"""),

    ("Q6: Can you explain the mathematical equivalence: Embedding[i] = OneHot[i] @ W?",
     f"""A: Mathematically identical but computationally different:

     OneHot approach:
       v = [0,0,1,0,0,...,0]  (vocab_size elements, mostly zeros)
       result = v @ W         (vocab_size × embedding_dim multiplication)

     Embedding approach:
       result = W[i,:]        (direct indexing, no multiplication)

     Same result, but embedding is O(1) vs O(vocab_size × embedding_dim)
     For vocab=50k, embedding_dim=768: That's 38.4M operations saved per lookup!"""),
]

for i, (question, answer) in enumerate(qa_pairs, 1):
    print(f"\n{question}")
    print(answer)
    if i < len(qa_pairs):
        print("\n" + "-"*80)


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY - KEY TAKEAWAYS FOR INTERVIEWS")
print("="*80)

summary = """
✓ Embeddings convert discrete tokens to continuous vectors
✓ Embedding lookup = One-hot @ Weight matrix (but way more efficient)
✓ Only embeddings of tokens in the batch receive gradient updates (sparse updates)
✓ Weight tying reduces parameters dramatically (vocab_size × embedding_dim savings)
✓ Common initialization: Uniform, Xavier, Normal (BERT uses N(0, 0.02²))
✓ Cosine similarity measures semantic similarity between embeddings
✓ Higher dimension = more expressiveness but more parameters
✓ Embeddings learn semantic relationships through backpropagation
✓ Rare words update less frequently than common words
✓ Embedding dimension is a key hyperparameter affecting model capacity

Files saved to: {}/
  ✓ embedding_initialization_methods.png
  ✓ embedding_similarity_matrix.png
  ✓ embedding_space_visualization.png
  ✓ embedding_space_3d.png
  ✓ weight_tying_architecture.png
  ✓ embedding_learning_dynamics.png
""".format(OUTPUT_DIR)

print(summary)

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
