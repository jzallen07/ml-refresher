"""
Comprehensive PCA and Dimensionality Reduction Demo

Interview Question Q28: "How does PCA relate to feature extraction in machine learning?"

This demo covers:
1. PCA from scratch implementation
2. Visualizations of principal components
3. Embedding dimensionality reduction
4. Comparison with t-SNE and UMAP
5. Practical denoising example

Author: Educational Demo
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_swiss_roll
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
    print("✓ UMAP is available")
except ImportError:
    UMAP_AVAILABLE = False
    print("✗ UMAP not available (install with: pip install umap-learn)")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Output directory for visualizations
VIZ_DIR = "/Users/zack/dev/ml-refresher/data/interview_viz"

print("=" * 80)
print("PCA AND DIMENSIONALITY REDUCTION DEMO")
print("=" * 80)

# ============================================================================
# PART 1: PCA FROM SCRATCH
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: PCA FROM SCRATCH IMPLEMENTATION")
print("=" * 80)

print("""
INTERVIEW CONTEXT: What is PCA?
---------------------------------
PCA (Principal Component Analysis) is an unsupervised dimensionality reduction
technique that:

1. FINDS DIRECTIONS OF MAXIMUM VARIANCE
   - These directions are called "principal components"
   - First PC: direction of highest variance
   - Second PC: direction of second highest variance (orthogonal to first)
   - And so on...

2. PROJECTS DATA onto these components
   - Reduces dimensionality while preserving most information
   - Removes correlations between features
   - Can be used for:
     * Visualization (reduce to 2D or 3D)
     * Denoising (remove low-variance components)
     * Feature extraction (use PCs as new features)
     * Compression (keep only top k components)

3. MATHEMATICAL FOUNDATION
   - Linear transformation based on eigendecomposition
   - Covariance matrix contains information about variance and correlations
   - Eigenvectors = directions (principal components)
   - Eigenvalues = magnitude of variance in each direction
""")

# Create sample 2D data for visualization
np.random.seed(42)
n_samples = 300

# Generate correlated 2D data
mean = [0, 0]
cov = [[3, 2.5],
       [2.5, 3]]  # High correlation
X_2d = np.random.multivariate_normal(mean, cov, n_samples)

print("\nOriginal Data Shape:", X_2d.shape)
print("First 5 samples:")
print(X_2d[:5])


class PCAFromScratch:
    """
    PCA implementation from scratch to understand the mathematics.

    Steps:
    1. Center the data (subtract mean)
    2. Compute covariance matrix
    3. Perform eigendecomposition
    4. Sort eigenvectors by eigenvalues (descending)
    5. Select top k eigenvectors
    6. Project data onto these eigenvectors
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """Fit PCA on data X"""
        print("\n--- PCA From Scratch: Step-by-step ---")

        # Step 1: Center the data
        print("\nStep 1: Center the data (subtract mean)")
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        print(f"  Original mean: {self.mean_}")
        print(f"  Centered mean: {np.mean(X_centered, axis=0)} (should be ~0)")

        # Step 2: Compute covariance matrix
        print("\nStep 2: Compute covariance matrix")
        # Cov = (X^T @ X) / (n - 1)
        n_samples = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        print(f"  Covariance matrix shape: {cov_matrix.shape}")
        print(f"  Covariance matrix:")
        print(f"  {cov_matrix}")

        # Step 3: Eigendecomposition
        print("\nStep 3: Eigendecomposition of covariance matrix")
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        print(f"  Eigenvalues (variance in each direction): {eigenvalues}")
        print(f"  Eigenvectors (principal component directions):")
        print(f"  {eigenvectors}")

        # Step 4: Sort by eigenvalues (descending)
        print("\nStep 4: Sort eigenvectors by eigenvalues (descending)")
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 5: Select top k components
        print(f"\nStep 5: Select top {self.n_components} components")
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]

        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        print(f"  Selected components shape: {self.components_.shape}")
        print(f"  Explained variance: {self.explained_variance_}")
        print(f"  Explained variance ratio: {self.explained_variance_ratio_}")
        print(f"  Total explained: {np.sum(self.explained_variance_ratio_):.2%}")

        return self

    def transform(self, X):
        """Project data onto principal components"""
        X_centered = X - self.mean_
        # Project: X_projected = X_centered @ components^T
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Reconstruct data from principal components"""
        return X_transformed @ self.components_ + self.mean_


# Fit our PCA implementation
pca_scratch = PCAFromScratch(n_components=2)
X_pca_scratch = pca_scratch.fit_transform(X_2d)

# Compare with sklearn's PCA
print("\n" + "-" * 80)
print("COMPARISON WITH SKLEARN'S PCA")
print("-" * 80)

pca_sklearn = PCA(n_components=2)
X_pca_sklearn = pca_sklearn.fit_transform(X_2d)

print("\nOur implementation:")
print(f"  Components:\n{pca_scratch.components_}")
print(f"  Explained variance ratio: {pca_scratch.explained_variance_ratio_}")

print("\nSklearn's implementation:")
print(f"  Components:\n{pca_sklearn.components_}")
print(f"  Explained variance ratio: {pca_sklearn.explained_variance_ratio_}")

print("\nDifference in transformed data (should be close to 0):")
print(f"  Max absolute difference: {np.max(np.abs(X_pca_scratch - X_pca_sklearn)):.10f}")

# ============================================================================
# PART 2: VISUALIZATION OF PCA
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: VISUALIZING PRINCIPAL COMPONENTS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Original data with principal components
ax = axes[0, 0]
ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5, c='steelblue', s=30)

# Draw principal components as arrows
mean_point = pca_scratch.mean_
for i, (component, variance) in enumerate(zip(pca_scratch.components_,
                                               pca_scratch.explained_variance_)):
    # Scale arrow by explained variance
    arrow_scale = 3 * np.sqrt(variance)
    ax.arrow(mean_point[0], mean_point[1],
             component[0] * arrow_scale, component[1] * arrow_scale,
             head_width=0.3, head_length=0.3, fc=f'C{i}', ec=f'C{i}',
             linewidth=3, alpha=0.8,
             label=f'PC{i+1} ({pca_scratch.explained_variance_ratio_[i]:.1%})')

ax.scatter(mean_point[0], mean_point[1], c='red', s=200, marker='X',
           label='Mean', zorder=5, edgecolors='black', linewidths=2)
ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('Original Data with Principal Components', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Plot 2: Data projected onto PC1 only
ax = axes[0, 1]
pca_1d = PCA(n_components=1)
X_pca_1d = pca_1d.fit_transform(X_2d)
X_reconstructed_1d = pca_1d.inverse_transform(X_pca_1d)

ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.3, c='lightgray', s=20, label='Original')
ax.scatter(X_reconstructed_1d[:, 0], X_reconstructed_1d[:, 1],
           alpha=0.6, c=X_pca_1d.ravel(), cmap='viridis', s=30,
           label='Projected to PC1')

# Draw projection lines
for i in range(0, n_samples, 10):  # Show every 10th point
    ax.plot([X_2d[i, 0], X_reconstructed_1d[i, 0]],
            [X_2d[i, 1], X_reconstructed_1d[i, 1]],
            'k-', alpha=0.1, linewidth=0.5)

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title(f'Projection to 1D (PC1 explains {pca_1d.explained_variance_ratio_[0]:.1%})',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Transformed data in PC space
ax = axes[1, 0]
scatter = ax.scatter(X_pca_scratch[:, 0], X_pca_scratch[:, 1],
                    alpha=0.5, c='coral', s=30)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel(f'PC1 ({pca_scratch.explained_variance_ratio_[0]:.1%} variance)',
              fontsize=12)
ax.set_ylabel(f'PC2 ({pca_scratch.explained_variance_ratio_[1]:.1%} variance)',
              fontsize=12)
ax.set_title('Data in Principal Component Space', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Plot 4: 3D example
ax = axes[1, 1]

# Create 3D data
mean_3d = [0, 0, 0]
cov_3d = [[3, 2.5, 1],
          [2.5, 3, 1.5],
          [1, 1.5, 1]]
X_3d = np.random.multivariate_normal(mean_3d, cov_3d, 200)

pca_3d = PCA(n_components=3)
pca_3d.fit(X_3d)

# Show explained variance
components = np.arange(1, 4)
ax.bar(components - 0.2, pca_3d.explained_variance_ratio_, 0.4,
       label='Individual', color='steelblue', alpha=0.8)
ax.bar(components + 0.2, np.cumsum(pca_3d.explained_variance_ratio_), 0.4,
       label='Cumulative', color='coral', alpha=0.8)
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Explained Variance Ratio', fontsize=12)
ax.set_title('Explained Variance by Component (3D→3D)', fontsize=14, fontweight='bold')
ax.set_xticks(components)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/14_pca_projection.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: {VIZ_DIR}/14_pca_projection.png")

# ============================================================================
# PART 3: EXPLAINED VARIANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: EXPLAINED VARIANCE ANALYSIS (SCREE PLOT)")
print("=" * 80)

print("""
INTERVIEW CONTEXT: Choosing Number of Components
-------------------------------------------------
How many components should we keep?

1. SCREE PLOT: Plot explained variance vs component number
   - Look for "elbow" where curve flattens
   - Shows diminishing returns

2. CUMULATIVE VARIANCE: Keep components until reaching threshold
   - Common thresholds: 80%, 90%, 95%, 99%
   - Trade-off: information vs dimensionality

3. DOMAIN KNOWLEDGE: Consider your use case
   - Visualization: usually 2-3 components
   - Feature extraction: depends on downstream task
   - Denoising: remove low-variance (noisy) components
""")

# Create high-dimensional data
n_samples = 500
n_features = 50

# Create data with decreasing variance structure
X_high_dim = np.random.randn(n_samples, n_features)
# Add structure: first few features have high variance
for i in range(n_features):
    variance_scale = np.exp(-i / 10)  # Exponential decay
    X_high_dim[:, i] *= variance_scale

# Fit PCA with all components
pca_full = PCA(n_components=n_features)
pca_full.fit(X_high_dim)

print(f"\nData shape: {X_high_dim.shape}")
print(f"Number of components: {n_features}")
print(f"\nExplained variance ratio (first 10 components):")
for i in range(10):
    print(f"  PC{i+1}: {pca_full.explained_variance_ratio_[i]:.4f} "
          f"(cumulative: {np.sum(pca_full.explained_variance_ratio_[:i+1]):.4f})")

# Create scree plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Scree plot (individual variance)
ax = axes[0, 0]
ax.plot(range(1, n_features + 1), pca_full.explained_variance_ratio_,
        'bo-', linewidth=2, markersize=4)
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Explained Variance Ratio', fontsize=12)
ax.set_title('Scree Plot: Individual Explained Variance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, n_features + 1)

# Highlight elbow region
ax.axvspan(0, 10, alpha=0.1, color='green', label='High variance')
ax.axvspan(10, 25, alpha=0.1, color='yellow', label='Medium variance')
ax.axvspan(25, n_features, alpha=0.1, color='red', label='Low variance (noise)')
ax.legend(fontsize=9)

# Plot 2: Cumulative explained variance
ax = axes[0, 1]
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
ax.plot(range(1, n_features + 1), cumulative_variance,
        'ro-', linewidth=2, markersize=4)

# Add threshold lines
thresholds = [0.80, 0.90, 0.95, 0.99]
colors = ['green', 'blue', 'orange', 'red']
for thresh, color in zip(thresholds, colors):
    n_components_needed = np.argmax(cumulative_variance >= thresh) + 1
    ax.axhline(y=thresh, color=color, linestyle='--', alpha=0.5,
               label=f'{thresh:.0%}: {n_components_needed} components')
    ax.plot(n_components_needed, thresh, 'o', color=color, markersize=8)

ax.set_xlabel('Number of Components', fontsize=12)
ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
ax.set_xlim(0, n_features + 1)
ax.set_ylim(0, 1.05)

# Plot 3: Log scale (better for seeing tail)
ax = axes[1, 0]
ax.semilogy(range(1, n_features + 1), pca_full.explained_variance_ratio_,
            'go-', linewidth=2, markersize=4)
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Explained Variance Ratio (log scale)', fontsize=12)
ax.set_title('Scree Plot (Log Scale) - Shows Tail Better', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0, n_features + 1)

# Plot 4: Comparison of dimensions
ax = axes[1, 1]
n_components_list = [5, 10, 15, 20, 30, 40, 50]
variance_explained = [np.sum(pca_full.explained_variance_ratio_[:n])
                      for n in n_components_list]
compression_ratio = [n_features / n for n in n_components_list]

ax2 = ax.twinx()
bars = ax.bar(range(len(n_components_list)), variance_explained,
              alpha=0.7, color='steelblue', label='Variance Explained')
line = ax2.plot(range(len(n_components_list)), compression_ratio,
                'ro-', linewidth=2, markersize=8, label='Compression Ratio')

ax.set_xlabel('Number of Components Kept', fontsize=12)
ax.set_ylabel('Variance Explained', fontsize=12, color='steelblue')
ax2.set_ylabel('Compression Ratio', fontsize=12, color='red')
ax.set_title('Variance vs Compression Trade-off', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(n_components_list)))
ax.set_xticklabels(n_components_list)
ax.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='red')
ax.grid(True, alpha=0.3, axis='y')

# Add text annotations
for i, (n, var, ratio) in enumerate(zip(n_components_list, variance_explained,
                                         compression_ratio)):
    ax.text(i, var + 0.02, f'{var:.1%}', ha='center', fontsize=8,
            fontweight='bold')

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/13_pca_explained_variance.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: {VIZ_DIR}/13_pca_explained_variance.png")

# ============================================================================
# PART 4: EMBEDDING VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: EMBEDDING DIMENSIONALITY REDUCTION")
print("=" * 80)

print("""
INTERVIEW CONTEXT: PCA for Embeddings
--------------------------------------
In NLP and ML, we often work with high-dimensional embeddings:
- Word embeddings (Word2Vec, GloVe): 50-300 dimensions
- Sentence embeddings (BERT, Sentence-BERT): 768-1024 dimensions
- Image embeddings (ResNet, ViT): 512-2048 dimensions

VISUALIZATION CHALLENGE:
- Can't plot 768-dimensional space!
- Need to reduce to 2D or 3D for visualization

PCA FOR EMBEDDINGS:
- Fast and scalable (linear transformation)
- Preserves global structure (distances)
- Good for overview of data distribution
- BUT: May not capture non-linear patterns

WHEN TO USE PCA vs t-SNE vs UMAP:
- PCA: Quick exploration, large datasets, preserving distances
- t-SNE: Finding clusters, local structure, final visualization
- UMAP: Balance between PCA and t-SNE, faster than t-SNE
""")

# Simulate high-dimensional embeddings
np.random.seed(42)
n_samples = 600
embedding_dim = 128  # Simulating smaller BERT-like embeddings

# Create synthetic embeddings with cluster structure
n_clusters = 5
embeddings_list = []
labels_list = []

for cluster_id in range(n_clusters):
    # Random center for each cluster
    center = np.random.randn(embedding_dim) * 3
    # Generate points around center
    cluster_samples = n_samples // n_clusters
    cluster_embeddings = center + np.random.randn(cluster_samples, embedding_dim) * 0.5

    embeddings_list.append(cluster_embeddings)
    labels_list.extend([cluster_id] * cluster_samples)

embeddings = np.vstack(embeddings_list)
labels = np.array(labels_list)

print(f"\nSimulated embeddings shape: {embeddings.shape}")
print(f"Number of clusters: {n_clusters}")
print(f"Samples per cluster: {n_samples // n_clusters}")

# Standardize embeddings (important for PCA)
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

print("\nOriginal embedding statistics:")
print(f"  Mean: {np.mean(embeddings, axis=0)[:5]}... (first 5 dims)")
print(f"  Std: {np.std(embeddings, axis=0)[:5]}... (first 5 dims)")

print("\nScaled embedding statistics:")
print(f"  Mean: {np.mean(embeddings_scaled, axis=0)[:5]}... (should be ~0)")
print(f"  Std: {np.std(embeddings_scaled, axis=0)[:5]}... (should be ~1)")

# Apply PCA
print("\nApplying PCA to reduce from 128D to 2D...")
pca_embeddings = PCA(n_components=2, random_state=42)
embeddings_pca = pca_embeddings.fit_transform(embeddings_scaled)

print(f"Explained variance ratio: {pca_embeddings.explained_variance_ratio_}")
print(f"Total variance explained: {np.sum(pca_embeddings.explained_variance_ratio_):.2%}")

# ============================================================================
# PART 5: COMPARISON WITH OTHER METHODS
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: COMPARING DIMENSIONALITY REDUCTION METHODS")
print("=" * 80)

print("""
METHOD COMPARISON:
------------------
1. PCA (Principal Component Analysis)
   - Type: Linear, global structure preservation
   - Speed: Very fast (O(min(n*d^2, d*n^2)))
   - Best for: Quick exploration, large datasets, linear patterns
   - Deterministic: Yes (same result every time)

2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
   - Type: Non-linear, local structure preservation
   - Speed: Slow (O(n^2))
   - Best for: Finding clusters, final visualizations
   - Deterministic: No (random initialization)
   - Note: Distances between clusters not meaningful

3. UMAP (Uniform Manifold Approximation and Projection)
   - Type: Non-linear, balances local and global structure
   - Speed: Fast (O(n log n))
   - Best for: Large datasets, preserving both local and global structure
   - Deterministic: No (but more stable than t-SNE)
""")

# Apply t-SNE
print("\nApplying t-SNE (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
embeddings_tsne = tsne.fit_transform(embeddings_scaled)
print("✓ t-SNE completed")

# Apply UMAP if available
if UMAP_AVAILABLE:
    print("\nApplying UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    embeddings_umap = reducer.fit_transform(embeddings_scaled)
    print("✓ UMAP completed")
else:
    embeddings_umap = None
    print("\n✗ UMAP not available - skipping")

# Create comparison plot
n_methods = 3 if UMAP_AVAILABLE else 2
fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
if n_methods == 2:
    axes = [axes[0], axes[1], None]

# Plot PCA
ax = axes[0]
scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                    c=labels, cmap='tab10', s=30, alpha=0.7)
ax.set_xlabel(f'PC1 ({pca_embeddings.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({pca_embeddings.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax.set_title('PCA\n(Linear, Global Structure)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot t-SNE
ax = axes[1]
scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                    c=labels, cmap='tab10', s=30, alpha=0.7)
ax.set_xlabel('t-SNE 1', fontsize=11)
ax.set_ylabel('t-SNE 2', fontsize=11)
ax.set_title('t-SNE\n(Non-linear, Local Structure)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot UMAP if available
if UMAP_AVAILABLE and axes[2] is not None:
    ax = axes[2]
    scatter = ax.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
                        c=labels, cmap='tab10', s=30, alpha=0.7)
    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)
    ax.set_title('UMAP\n(Non-linear, Balanced)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Add colorbar
if n_methods == 3:
    plt.colorbar(scatter, ax=axes, label='Cluster ID', fraction=0.02, pad=0.04)
else:
    plt.colorbar(scatter, ax=axes[:2], label='Cluster ID', fraction=0.02, pad=0.04)

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/15_dimensionality_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: {VIZ_DIR}/15_dimensionality_comparison.png")

# ============================================================================
# PART 6: PRACTICAL EXAMPLE - DENOISING
# ============================================================================
print("\n" + "=" * 80)
print("PART 6: PRACTICAL APPLICATION - DENOISING WITH PCA")
print("=" * 80)

print("""
INTERVIEW CONTEXT: PCA for Denoising
-------------------------------------
PCA can be used for denoising by:

1. ASSUMPTION: Signal has high variance, noise has low variance
   - First few PCs capture signal (high variance)
   - Last PCs capture noise (low variance)

2. PROCESS:
   - Apply PCA to noisy data
   - Keep only top k components (signal)
   - Discard remaining components (noise)
   - Reconstruct data from top k components

3. APPLICATIONS:
   - Image denoising
   - Signal processing
   - Feature preprocessing
   - Data compression

4. TRADE-OFF:
   - More components = less noise removal but more detail preserved
   - Fewer components = more noise removal but may lose signal
""")

# Create clean signal (2D spiral)
n_points = 300
t = np.linspace(0, 4 * np.pi, n_points)
clean_signal = np.column_stack([
    t * np.cos(t),
    t * np.sin(t)
])

# Add noise
noise_level = 0.5
noise = np.random.randn(*clean_signal.shape) * noise_level
noisy_signal = clean_signal + noise

print(f"\nSignal shape: {clean_signal.shape}")
print(f"Noise level: {noise_level}")
print(f"Signal-to-Noise Ratio (SNR): {10 * np.log10(np.var(clean_signal) / np.var(noise)):.2f} dB")

# Apply PCA with different numbers of components
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot clean signal
ax = axes[0, 0]
ax.plot(clean_signal[:, 0], clean_signal[:, 1], 'b-', linewidth=2, alpha=0.7)
ax.scatter(clean_signal[:, 0], clean_signal[:, 1], c=range(n_points),
          cmap='viridis', s=20, alpha=0.6)
ax.set_xlabel('X', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title('Clean Signal (Ground Truth)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Plot noisy signal
ax = axes[0, 1]
ax.plot(noisy_signal[:, 0], noisy_signal[:, 1], 'r-', linewidth=1, alpha=0.3)
ax.scatter(noisy_signal[:, 0], noisy_signal[:, 1], c=range(n_points),
          cmap='viridis', s=20, alpha=0.6)
ax.set_xlabel('X', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title('Noisy Signal', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Denoise with PCA (1 component)
ax = axes[0, 2]
pca_denoise_1 = PCA(n_components=1)
signal_transformed_1 = pca_denoise_1.fit_transform(noisy_signal)
signal_denoised_1 = pca_denoise_1.inverse_transform(signal_transformed_1)

ax.plot(signal_denoised_1[:, 0], signal_denoised_1[:, 1], 'g-', linewidth=2, alpha=0.7)
ax.scatter(signal_denoised_1[:, 0], signal_denoised_1[:, 1], c=range(n_points),
          cmap='viridis', s=20, alpha=0.6)
mse_1 = np.mean((clean_signal - signal_denoised_1) ** 2)
ax.set_xlabel('X', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title(f'Denoised (1 PC)\nMSE: {mse_1:.3f}', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Comparison with original (overlay)
ax = axes[1, 0]
ax.plot(clean_signal[:, 0], clean_signal[:, 1], 'b-', linewidth=2,
       alpha=0.5, label='Clean')
ax.plot(noisy_signal[:, 0], noisy_signal[:, 1], 'r-', linewidth=1,
       alpha=0.3, label='Noisy')
ax.plot(signal_denoised_1[:, 0], signal_denoised_1[:, 1], 'g-', linewidth=2,
       alpha=0.7, label='Denoised (1 PC)')
ax.set_xlabel('X', fontsize=11)
ax.set_ylabel('Y', fontsize=11)
ax.set_title('Overlay Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Error analysis
ax = axes[1, 1]
error_noisy = np.linalg.norm(clean_signal - noisy_signal, axis=1)
error_denoised = np.linalg.norm(clean_signal - signal_denoised_1, axis=1)

ax.plot(range(n_points), error_noisy, 'r-', linewidth=1.5, alpha=0.7,
       label=f'Noisy (mean: {np.mean(error_noisy):.3f})')
ax.plot(range(n_points), error_denoised, 'g-', linewidth=1.5, alpha=0.7,
       label=f'Denoised (mean: {np.mean(error_denoised):.3f})')
ax.set_xlabel('Point Index', fontsize=11)
ax.set_ylabel('Error (L2 distance)', fontsize=11)
ax.set_title('Point-wise Reconstruction Error', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Explained variance
ax = axes[1, 2]
pca_full_denoise = PCA(n_components=2)
pca_full_denoise.fit(noisy_signal)

components = [1, 2]
explained_var = pca_full_denoise.explained_variance_ratio_
ax.bar(components, explained_var, color=['steelblue', 'lightcoral'], alpha=0.8)
ax.set_xlabel('Principal Component', fontsize=11)
ax.set_ylabel('Explained Variance Ratio', fontsize=11)
ax.set_title('Variance in Noisy Signal', fontsize=12, fontweight='bold')
ax.set_xticks(components)
ax.set_xticklabels(['PC1\n(Signal)', 'PC2\n(Noise)'])
ax.grid(True, alpha=0.3, axis='y')

# Add text annotations
for i, var in enumerate(explained_var):
    ax.text(components[i], var + 0.02, f'{var:.1%}',
           ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/16_pca_denoising.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: {VIZ_DIR}/16_pca_denoising.png")

# Quantitative comparison
print("\nDenoising Results:")
print(f"  Noisy signal MSE: {np.mean((clean_signal - noisy_signal) ** 2):.4f}")
print(f"  Denoised signal MSE: {mse_1:.4f}")
improvement = (1 - mse_1 / np.mean((clean_signal - noisy_signal) ** 2)) * 100
print(f"  Improvement: {improvement:.1f}%")

# ============================================================================
# PART 7: ADVANCED CONCEPTS
# ============================================================================
print("\n" + "=" * 80)
print("PART 7: ADVANCED PCA CONCEPTS")
print("=" * 80)

print("""
ADVANCED TOPICS FOR INTERVIEWS:
--------------------------------

1. KERNEL PCA
   - Extends PCA to non-linear relationships
   - Uses kernel trick (similar to SVM)
   - Can capture complex patterns
   - More computationally expensive

2. INCREMENTAL PCA
   - Processes data in mini-batches
   - Useful for large datasets that don't fit in memory
   - sklearn: IncrementalPCA

3. SPARSE PCA
   - Produces sparse principal components
   - Easier to interpret (many zeros)
   - Trade-off: sparsity vs variance explained

4. PROBABILISTIC PCA
   - Probabilistic interpretation of PCA
   - Can handle missing data
   - Provides uncertainty estimates

5. PCA vs AUTOENCODERS
   - PCA: Linear compression (1 layer)
   - Autoencoder: Non-linear compression (deep network)
   - Autoencoder more flexible but requires more data/computation

6. WHITENING
   - PCA + scaling to unit variance
   - Makes components uncorrelated AND unit variance
   - Useful preprocessing for neural networks

7. RELATIONSHIP TO SVD
   - PCA via eigendecomposition of covariance matrix
   - Can also compute via SVD of data matrix
   - SVD more numerically stable
   - sklearn uses randomized SVD for efficiency
""")

# Demonstrate SVD relationship
print("\nDemonstrating PCA-SVD Relationship:")
print("-" * 50)

X_demo = np.random.randn(100, 10)
X_centered = X_demo - np.mean(X_demo, axis=0)

# Method 1: PCA via eigendecomposition (what we did earlier)
cov_matrix = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = eigenvalues.argsort()[::-1]
pca_components_eigen = eigenvectors[:, idx].T

# Method 2: PCA via SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
pca_components_svd = Vt

# Method 3: sklearn (uses randomized SVD)
pca_sklearn_demo = PCA(n_components=10)
pca_sklearn_demo.fit(X_centered)

print(f"Components from eigendecomposition (first 3, first 5 values):")
print(pca_components_eigen[:3, :5])
print(f"\nComponents from SVD (first 3, first 5 values):")
print(pca_components_svd[:3, :5])
print(f"\nComponents from sklearn (first 3, first 5 values):")
print(pca_sklearn_demo.components_[:3, :5])

print(f"\nAll methods produce same components (up to sign):")
print(f"Max difference (eigen vs SVD): {np.max(np.abs(np.abs(pca_components_eigen) - np.abs(pca_components_svd))):.10f}")

# ============================================================================
# SUMMARY AND INTERVIEW TIPS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: KEY INTERVIEW POINTS")
print("=" * 80)

print("""
ESSENTIAL CONCEPTS TO REMEMBER:
--------------------------------

1. WHAT IS PCA?
   ✓ Unsupervised dimensionality reduction
   ✓ Finds directions of maximum variance
   ✓ Linear transformation (orthogonal projection)
   ✓ Based on eigendecomposition of covariance matrix

2. WHY USE PCA?
   ✓ Dimensionality reduction (compress data)
   ✓ Visualization (reduce to 2D/3D)
   ✓ Feature extraction (new features are PCs)
   ✓ Denoising (remove low-variance components)
   ✓ Remove multicollinearity (decorrelate features)

3. HOW DOES IT WORK?
   ✓ Center data (subtract mean)
   ✓ Compute covariance matrix
   ✓ Find eigenvectors/eigenvalues
   ✓ Sort by eigenvalues (descending)
   ✓ Project data onto top k eigenvectors

4. CHOOSING NUMBER OF COMPONENTS
   ✓ Scree plot (look for elbow)
   ✓ Cumulative variance threshold (e.g., 95%)
   ✓ Domain knowledge/task requirements
   ✓ Cross-validation for downstream tasks

5. LIMITATIONS
   ✓ Only captures linear relationships
   ✓ Assumes high variance = important
   ✓ Sensitive to scaling (always standardize!)
   ✓ Interpretability: PCs are combinations of original features

6. ALTERNATIVES
   ✓ t-SNE: Better for visualization, captures non-linear patterns
   ✓ UMAP: Faster than t-SNE, preserves global structure
   ✓ Autoencoders: Non-linear, learnable compression
   ✓ Feature selection: Keep subset of original features

7. PRACTICAL TIPS
   ✓ Always standardize features before PCA
   ✓ Check explained variance ratio
   ✓ Use PCA for exploration, not always for final model
   ✓ Consider computational cost vs accuracy trade-off

8. COMMON INTERVIEW QUESTIONS
   Q: "What's the difference between PCA and LDA?"
   A: PCA is unsupervised (maximizes variance), LDA is supervised
      (maximizes class separation)

   Q: "Why do we center the data in PCA?"
   A: PCA finds directions of maximum variance from origin. Centering
      ensures we measure variance around the data's actual mean.

   Q: "Can PCA handle missing values?"
   A: Standard PCA cannot. Need to impute first, or use probabilistic
      PCA / matrix completion methods.

   Q: "Is PCA sensitive to outliers?"
   A: Yes! Outliers can dominate variance. Consider robust PCA or
      outlier removal.

9. CODE INTERVIEW TIPS
   ✓ Know how to implement PCA from scratch (covariance + eigen)
   ✓ Understand relationship between PCA and SVD
   ✓ Can explain each step mathematically
   ✓ Know sklearn API: fit(), transform(), fit_transform()
   ✓ Can interpret explained_variance_ratio_

10. MATHEMATICAL CONCEPTS
    ✓ Covariance matrix: measures feature correlations
    ✓ Eigenvectors: directions of principal components
    ✓ Eigenvalues: variance along each component
    ✓ Orthogonality: PCs are uncorrelated
    ✓ Linear algebra: PCA is matrix factorization
""")

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print("\nGenerated visualizations:")
print(f"  1. {VIZ_DIR}/13_pca_explained_variance.png")
print(f"  2. {VIZ_DIR}/14_pca_projection.png")
print(f"  3. {VIZ_DIR}/15_dimensionality_comparison.png")
print(f"  4. {VIZ_DIR}/16_pca_denoising.png")

print("\nNext steps for learning:")
print("  • Implement PCA from scratch on different datasets")
print("  • Try kernel PCA for non-linear patterns")
print("  • Compare PCA with autoencoders")
print("  • Apply PCA in a real project")
print("  • Study SVD and its relationship to PCA")

print("\n" + "=" * 80)
