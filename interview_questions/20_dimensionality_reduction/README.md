# Dimensionality Reduction

## Interview Questions Covered
- **Q28**: How does PCA relate to feature extraction in machine learning?

---

## Q28: PCA and Feature Extraction

### Definition

**Principal Component Analysis (PCA)** finds orthogonal directions of maximum variance in data, projecting high-dimensional data to lower dimensions while preserving the most important information.

### The Math

Given data matrix X (n_samples × n_features):

```
1. Center the data: X_centered = X - mean(X)
2. Compute covariance: C = X_centered.T @ X_centered / (n-1)
3. Eigendecomposition: C = V Λ V.T
4. Project: X_reduced = X_centered @ V[:, :k]  # Keep top k components
```

### Intuition

```
Original 2D data:          After PCA:
    *  *                   PC1 →  ************
  *  **  *                       (captures most variance)
 * ** ** *                  PC2 ↑
  *  **  *                       (captures remaining)
    *  *

If PC1 captures 95% of variance, we can drop PC2!
```

### Why PCA for Feature Extraction?

1. **Dimensionality reduction**: 1000 features → 50 components
2. **Noise removal**: Small components often capture noise
3. **Decorrelation**: Principal components are uncorrelated
4. **Visualization**: Project to 2D/3D for plotting

### PCA in NLP/LLMs

#### Analyzing Embeddings

```python
from sklearn.decomposition import PCA

# Word/sentence embeddings (e.g., 768 dimensions)
embeddings = model.encode(sentences)  # (1000, 768)

# Reduce for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)  # (1000, 2)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
```

#### Understanding Embedding Spaces

```python
# How many dimensions are "useful"?
pca = PCA(n_components=100)
pca.fit(embeddings)

# Plot explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
# Often find: 50 components explain 90% of variance!
```

#### Embedding Compression

```python
# Reduce embedding size for efficiency
pca = PCA(n_components=256)  # 768 → 256
compressed_embeddings = pca.fit_transform(embeddings)
# 3x smaller, often <5% performance loss
```

### When to Use PCA

| Use Case | Benefit |
|----------|---------|
| **Visualization** | See high-dim data in 2D/3D |
| **Preprocessing** | Reduce features before training |
| **Compression** | Smaller embeddings |
| **Denoising** | Remove low-variance components |
| **Understanding** | Analyze what dimensions capture |

### Limitations

1. **Linear only**: Can't capture nonlinear structure
2. **Variance ≠ importance**: Important info might be in small components
3. **Interpretability**: Principal components are linear combinations
4. **Sensitive to scaling**: Need to standardize features first

### Nonlinear Alternatives

| Method | Description | Use Case |
|--------|-------------|----------|
| **t-SNE** | Preserves local structure | Visualization |
| **UMAP** | Fast, preserves global+local | Visualization, clustering |
| **Autoencoders** | Neural network compression | Learning representations |
| **Kernel PCA** | PCA in kernel space | Nonlinear structure |

### PCA vs t-SNE vs UMAP

```
PCA:   Fast, linear, preserves global structure
t-SNE: Slow, local structure, great for visualization
UMAP:  Fast, global+local, good for clustering

For embeddings visualization:
  Small data (<1000): t-SNE
  Large data (>10000): UMAP
  Quick check: PCA
```

### Code Example

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample embedding data
embeddings = np.random.randn(1000, 768)  # Like BERT embeddings

# Fit PCA
pca = PCA(n_components=50)
reduced = pca.fit_transform(embeddings)

# Check explained variance
print(f"50 components explain: {pca.explained_variance_ratio_.sum():.1%}")

# Visualize with 2 components
pca_2d = PCA(n_components=2)
vis = pca_2d.fit_transform(embeddings)
plt.scatter(vis[:, 0], vis[:, 1], alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Embeddings in 2D (PCA)')
```

### PCA for Understanding Models

```python
# Analyze attention patterns
attention_matrices = []  # Collect from model
attention_flat = attention_matrices.reshape(-1, num_heads * seq_len * seq_len)

pca = PCA(n_components=10)
pca.fit(attention_flat)

# What patterns dominate?
for i, component in enumerate(pca.components_[:3]):
    plt.figure()
    plt.imshow(component.reshape(num_heads, seq_len, seq_len).mean(0))
    plt.title(f'PC{i+1}: {pca.explained_variance_ratio_[i]:.1%} variance')
```

### Practical Tips

1. **Always standardize first**: `StandardScaler().fit_transform(X)`
2. **Check explained variance**: Plot cumulative to choose k
3. **Whitening option**: `PCA(whiten=True)` for unit variance components
4. **Incremental PCA**: For data too large for memory
5. **Randomized PCA**: Faster for high dimensions

```python
from sklearn.decomposition import PCA, IncrementalPCA

# For large datasets
ipca = IncrementalPCA(n_components=50)
for batch in data_loader:
    ipca.partial_fit(batch)

# For high dimensions
pca = PCA(n_components=50, svd_solver='randomized')
```

---

## Interview Tips

1. **Know the math**: Covariance matrix → eigendecomposition → projection
2. **Variance explained**: How to choose number of components
3. **Linear limitation**: Know when to use t-SNE/UMAP instead
4. **NLP applications**: Embedding visualization and compression
5. **Preprocessing**: Often improves downstream models

---

## Code Demo

See `pca_demo.py` for a comprehensive educational demo covering:
- **PCA from scratch implementation** with step-by-step math explanations
- **High-dimensional embedding visualization** (128D → 2D)
- **Explained variance analysis** with scree plots
- **Method comparison**: PCA vs t-SNE vs UMAP
- **Practical denoising application** showing signal/noise separation
- **Advanced concepts**: SVD relationship, kernel PCA, whitening

### Running the Demo

```bash
poetry run python interview_questions/20_dimensionality_reduction/pca_demo.py
```

### Generated Visualizations

The demo creates four high-quality visualizations in `/data/interview_viz/`:

1. **`13_pca_explained_variance.png`** (532 KB)
   - Scree plot showing variance decay
   - Cumulative variance with threshold markers (80%, 90%, 95%, 99%)
   - Log-scale view for detailed tail analysis
   - Variance vs compression trade-off

2. **`14_pca_projection.png`** (862 KB)
   - Original 2D data with principal component arrows
   - Projection to 1D (PC1) showing information loss
   - Data in principal component space
   - 3D explained variance example

3. **`15_dimensionality_comparison.png`** (279 KB)
   - **PCA**: Linear transformation, preserves global structure
   - **t-SNE**: Non-linear, emphasizes clusters and local structure
   - Side-by-side comparison on 128D → 2D embedding reduction

4. **`16_pca_denoising.png`** (1.1 MB)
   - Clean signal (spiral pattern)
   - Noisy signal with Gaussian noise
   - Denoised result using top principal component
   - Overlay comparison and point-wise error analysis
   - Variance decomposition showing signal vs noise components

### Demo Highlights

The demo includes extensive educational content:

```
================================================================================
PART 1: PCA FROM SCRATCH IMPLEMENTATION
- Step-by-step eigendecomposition
- Comparison with sklearn's PCA
- Numerical verification

PART 2: VISUALIZING PRINCIPAL COMPONENTS
- Geometric interpretation
- Projection demonstrations
- Component space visualization

PART 3: EXPLAINED VARIANCE ANALYSIS
- Scree plots (linear and log scale)
- Cumulative variance curves
- Choosing number of components

PART 4: EMBEDDING DIMENSIONALITY REDUCTION
- Simulated 128D embeddings (like BERT)
- 5-cluster structure
- Standardization importance

PART 5: COMPARING METHODS
- PCA: Fast, linear, global
- t-SNE: Slow, non-linear, local
- UMAP: Fast, balanced (if available)

PART 6: PRACTICAL DENOISING
- Signal/noise separation
- Reconstruction quality
- Quantitative metrics

PART 7: ADVANCED CONCEPTS
- SVD relationship
- Kernel PCA
- Incremental PCA
- Whitening
================================================================================
```
