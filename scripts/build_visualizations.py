#!/usr/bin/env python3
"""Build the pre-generated visualization library.

Generates PNGs into data/viz_library/{topic_slug}/ and writes manifest.json.
Run: ``python -m scripts.build_visualizations``
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "viz_library"
MANIFEST_PATH = OUT / "manifest.json"

STYLE = "dark_background"
DPI = 150

manifest: dict[str, list[dict]] = {}


def _save(topic: str, name: str, fig: plt.Figure, title: str, description: str, tags: list[str]) -> None:
    topic_dir = OUT / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    path = topic_dir / f"{name}.png"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    manifest.setdefault(topic, []).append({
        "name": name,
        "file": f"{topic}/{name}.png",
        "title": title,
        "description": description,
        "tags": tags,
    })
    print(f"  ✓ {topic}/{name}.png")


# ─── Attention Mechanisms ─────────────────────────────────────────────────────

def build_attention_basic_heatmap():
    import torch
    torch.manual_seed(42)
    seq_len, d_k = 5, 64
    Q = torch.randn(1, 1, seq_len, d_k)
    K = torch.randn(1, 1, seq_len, d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)[0, 0].detach().numpy()
    tokens = ["The", "cat", "sat", "on", "mat"]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(weights, annot=True, fmt=".3f", cmap="YlOrRd",
                    xticklabels=tokens, yticklabels=tokens, ax=ax)
        ax.set_xlabel("Key (attending to)")
        ax.set_ylabel("Query (current token)")
        ax.set_title("Basic Attention Weights", fontweight="bold")
        fig.tight_layout()
    _save("attention_mechanisms", "attention_basic_heatmap", fig,
          "Basic Attention Heatmap",
          "Shows how each token attends to every other token using scaled dot-product attention.",
          ["attention", "heatmap", "self-attention"])


def build_attention_all_heads():
    import torch
    torch.manual_seed(42)
    seq_len, d_model, num_heads = 6, 512, 4
    d_k = d_model // num_heads
    Q = torch.randn(1, num_heads, seq_len, d_k)
    K = torch.randn(1, num_heads, seq_len, d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)[0].detach().numpy()
    tokens = ["I", "love", "machine", "learning", "models", "."]

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for h in range(num_heads):
            sns.heatmap(weights[h], annot=True, fmt=".2f", cmap="YlOrRd",
                        xticklabels=tokens, yticklabels=tokens, ax=axes[h], cbar=False)
            axes[h].set_title(f"Head {h}", fontweight="bold")
            axes[h].set_xlabel("Key")
            if h == 0:
                axes[h].set_ylabel("Query")
        fig.suptitle("Multi-Head Attention Patterns", fontweight="bold", y=1.02)
        fig.tight_layout()
    _save("attention_mechanisms", "attention_all_heads", fig,
          "Multi-Head Attention Patterns",
          "Each attention head learns to focus on different aspects of the input.",
          ["multi-head", "attention", "heads"])


def build_causal_mask():
    import torch
    seq_len = 8
    mask = torch.tril(torch.ones(seq_len, seq_len)).numpy()

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(mask, annot=True, fmt=".0f", cmap="Blues",
                    xticklabels=range(seq_len), yticklabels=range(seq_len), ax=ax)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_title("Causal Mask for Autoregressive Attention", fontweight="bold")
        fig.tight_layout()
    _save("attention_mechanisms", "causal_mask", fig,
          "Causal Attention Mask",
          "Lower triangular mask ensures each token can only attend to past and current positions.",
          ["causal", "mask", "autoregressive"])


def build_attention_patterns():
    seq_len = 8
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]

    uniform = np.ones((seq_len, seq_len)) / seq_len
    local = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(max(0, i - 2), min(seq_len, i + 3)):
            local[i, j] = 1.0
    local = local / local.sum(axis=1, keepdims=True)
    self_attn = np.eye(seq_len) * 0.7 + (1 - np.eye(seq_len)) * 0.3 / (seq_len - 1)
    focused = np.zeros((seq_len, seq_len))
    focused[:, 3] = 0.6
    focused[:, 4] = 0.3
    for i in range(seq_len):
        focused[i, i] += 0.1

    patterns = [
        (uniform, "Uniform"), (local, "Local/Positional"),
        (self_attn, "Self-Focused"), (focused, "Content-Focused"),
    ]

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, (pat, name) in zip(axes, patterns):
            sns.heatmap(pat, annot=False, cmap="YlOrRd",
                        xticklabels=tokens, yticklabels=tokens, ax=ax, cbar=False)
            ax.set_title(name, fontweight="bold", fontsize=10)
            ax.tick_params(labelsize=7)
        fig.suptitle("Common Attention Patterns", fontweight="bold", y=1.02)
        fig.tight_layout()
    _save("attention_mechanisms", "attention_patterns", fig,
          "Common Attention Patterns",
          "Four archetypal attention patterns: uniform, local, self-focused, and content-focused.",
          ["patterns", "attention", "comparison"])


# ─── Loss Functions and Math ─────────────────────────────────────────────────

def build_cross_entropy_curve():
    p = np.linspace(0.01, 0.99, 200)
    loss = -np.log(p)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(p, loss, linewidth=2, color="cyan")
        ax.set_xlabel("Predicted probability of correct class")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Cross-Entropy Loss Curve", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        fig.tight_layout()
    _save("loss_functions_and_math", "cross_entropy_curve", fig,
          "Cross-Entropy Loss Curve",
          "Shows how cross-entropy loss increases sharply as predicted probability of the correct class drops.",
          ["loss", "cross-entropy", "classification"])


def build_activation_functions():
    x = np.linspace(-5, 5, 500)
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    swish = x * sigmoid

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        for y, label in [(relu, "ReLU"), (sigmoid, "Sigmoid"), (tanh, "Tanh"),
                         (gelu, "GELU"), (swish, "Swish")]:
            ax.plot(x, y, linewidth=2, label=label)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title("Activation Functions", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
        fig.tight_layout()
    _save("loss_functions_and_math", "activation_functions", fig,
          "Activation Functions Comparison",
          "ReLU, Sigmoid, Tanh, GELU, and Swish — the most common activation functions in deep learning.",
          ["activation", "relu", "gelu", "sigmoid"])


def build_kl_divergence():
    x = np.linspace(-5, 5, 500)
    p = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)
    sigmas = [0.5, 1.0, 1.5, 2.0]

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax0, ax1 = axes
        ax0.plot(x, p, linewidth=2, label="P (target)", color="cyan")
        kl_values = []
        for s in sigmas:
            q = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / s) ** 2)
            ax0.plot(x, q, linewidth=1.5, linestyle="--", label=f"Q (σ={s})")
            kl = np.sum(np.where(p > 1e-10, p * np.log(p / np.clip(q, 1e-10, None)), 0)) * (x[1] - x[0])
            kl_values.append(kl)
        ax0.set_title("Distributions", fontweight="bold")
        ax0.legend(fontsize=8)
        ax0.grid(True, alpha=0.3)

        ax1.bar([f"σ={s}" for s in sigmas], kl_values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(sigmas))))
        ax1.set_title("KL(P || Q)", fontweight="bold")
        ax1.set_ylabel("KL Divergence")
        ax1.grid(True, alpha=0.3, axis="y")
        fig.suptitle("KL Divergence", fontweight="bold")
        fig.tight_layout()
    _save("loss_functions_and_math", "kl_divergence", fig,
          "KL Divergence",
          "KL divergence measures how one probability distribution differs from a reference distribution.",
          ["kl-divergence", "distribution", "loss"])


def build_loss_landscape():
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2  # Rosenbrock
    Z = np.log1p(Z)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        cs = ax.contourf(X, Y, Z, levels=30, cmap="magma")
        fig.colorbar(cs, ax=ax, label="log(1 + loss)")
        # SGD path simulation
        path_x = [2.5]
        path_y = [2.5]
        lr = 0.001
        for _ in range(200):
            gx = -2 * (1 - path_x[-1]) + 200 * (path_y[-1] - path_x[-1] ** 2) * (-2 * path_x[-1])
            gy = 200 * (path_y[-1] - path_x[-1] ** 2)
            path_x.append(path_x[-1] - lr * gx)
            path_y.append(path_y[-1] - lr * gy)
        ax.plot(path_x, path_y, "w.-", markersize=2, linewidth=0.8, alpha=0.7)
        ax.plot(1, 1, "r*", markersize=12, label="Minimum")
        ax.set_xlabel("θ₁")
        ax.set_ylabel("θ₂")
        ax.set_title("Loss Landscape (Rosenbrock)", fontweight="bold")
        ax.legend()
        fig.tight_layout()
    _save("loss_functions_and_math", "loss_landscape", fig,
          "Loss Landscape",
          "A contour plot of the Rosenbrock function with a gradient descent path, illustrating optimization challenges.",
          ["loss-landscape", "optimization", "gradient-descent"])


# ─── Gradients and Optimization ──────────────────────────────────────────────

def build_embedding_gradient_sparsity():
    import torch
    torch.manual_seed(42)
    vocab_size, embed_dim = 100, 32
    emb = torch.nn.Embedding(vocab_size, embed_dim)
    tokens = torch.randint(0, vocab_size, (8, 16))
    out = emb(tokens)
    loss = out.sum()
    loss.backward()
    grads = emb.weight.grad.abs().sum(dim=1).numpy()

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(range(vocab_size), grads, color=np.where(grads > 0, "cyan", "gray"), width=1.0)
        axes[0].set_xlabel("Token ID")
        axes[0].set_ylabel("Gradient Magnitude")
        axes[0].set_title("Gradient by Embedding Row", fontweight="bold")

        sns.heatmap(emb.weight.grad.abs().numpy()[:20, :20], cmap="viridis", ax=axes[1], cbar=True)
        axes[1].set_title("Gradient Heatmap (first 20)", fontweight="bold")
        axes[1].set_xlabel("Embedding Dim")
        axes[1].set_ylabel("Token ID")
        fig.suptitle("Embedding Gradient Sparsity", fontweight="bold")
        fig.tight_layout()
    _save("gradients_and_optimization", "embedding_gradient_sparsity", fig,
          "Embedding Gradient Sparsity",
          "Only tokens present in the batch receive gradient updates — most embedding rows stay at zero.",
          ["embeddings", "gradients", "sparsity"])


def build_gradient_flow_layers():
    import torch
    torch.manual_seed(42)
    layers = []
    x = torch.randn(32, 128, requires_grad=True)
    for i in range(8):
        w = torch.randn(128, 128) * 0.1
        x = torch.relu(x @ w)
        layers.append(x)
        x = x.clone().detach().requires_grad_(True)

    # Simulate gradient norms
    np.random.seed(42)
    grad_norms = np.exp(-np.linspace(0, 3, 8)) + np.random.normal(0, 0.02, 8)
    grad_norms = np.abs(grad_norms)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, 8))
        ax.bar(range(8), grad_norms, color=colors)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Flow Through Layers", fontweight="bold")
        ax.set_xticks(range(8))
        ax.set_xticklabels([f"Layer {i}" for i in range(8)], rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
    _save("gradients_and_optimization", "gradient_flow_layers", fig,
          "Gradient Flow Through Layers",
          "Gradient norms typically decrease in deeper layers, showing the vanishing gradient problem.",
          ["gradients", "vanishing", "layers"])


def build_learning_rate_comparison():
    np.random.seed(42)
    steps = np.arange(100)
    lrs = {"1e-2 (too high)": 1e-2, "1e-3 (good)": 1e-3, "1e-5 (too low)": 1e-5}

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, lr in lrs.items():
            loss = 5.0
            losses = []
            for _ in steps:
                grad = np.random.normal(0, 1) + (loss - 0.5) * 0.5
                loss -= lr * grad
                if lr > 5e-3:
                    loss += np.random.normal(0, 0.3)
                loss = max(loss, 0.1)
                losses.append(loss)
            ax.plot(steps, losses, linewidth=2, label=label)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Learning Rate Comparison", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    _save("gradients_and_optimization", "learning_rate_comparison", fig,
          "Learning Rate Comparison",
          "Too high: loss oscillates. Too low: convergence is slow. A good LR balances speed and stability.",
          ["learning-rate", "optimization", "training"])


def build_hyperparameter_sensitivity():
    np.random.seed(42)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # LR sensitivity
        lrs = np.logspace(-5, -1, 20)
        final_loss = 5 * np.exp(-500 * lrs) + 0.5 + 2 * (lrs > 5e-3)
        final_loss += np.random.normal(0, 0.1, len(lrs))
        axes[0, 0].semilogx(lrs, final_loss, "o-", color="cyan")
        axes[0, 0].set_title("Learning Rate", fontweight="bold")
        axes[0, 0].set_ylabel("Final Loss")
        axes[0, 0].grid(True, alpha=0.3)

        # Batch size
        bs = [8, 16, 32, 64, 128, 256]
        final_loss_bs = [1.2, 0.9, 0.7, 0.65, 0.63, 0.62]
        axes[0, 1].plot(bs, final_loss_bs, "o-", color="lime")
        axes[0, 1].set_title("Batch Size", fontweight="bold")
        axes[0, 1].set_ylabel("Final Loss")
        axes[0, 1].grid(True, alpha=0.3)

        # Weight decay
        wd = [0, 1e-4, 1e-3, 1e-2, 1e-1]
        val_loss = [1.0, 0.8, 0.7, 0.75, 1.1]
        axes[1, 0].plot(range(len(wd)), val_loss, "o-", color="orange")
        axes[1, 0].set_xticks(range(len(wd)))
        axes[1, 0].set_xticklabels([str(w) for w in wd])
        axes[1, 0].set_title("Weight Decay", fontweight="bold")
        axes[1, 0].set_ylabel("Val Loss")
        axes[1, 0].grid(True, alpha=0.3)

        # Dropout
        dp = [0.0, 0.1, 0.2, 0.3, 0.5]
        val_loss_dp = [0.9, 0.75, 0.7, 0.72, 0.85]
        axes[1, 1].plot(dp, val_loss_dp, "o-", color="magenta")
        axes[1, 1].set_title("Dropout Rate", fontweight="bold")
        axes[1, 1].set_ylabel("Val Loss")
        axes[1, 1].grid(True, alpha=0.3)

        fig.suptitle("Hyperparameter Sensitivity", fontweight="bold")
        fig.tight_layout()
    _save("gradients_and_optimization", "hyperparameter_sensitivity", fig,
          "Hyperparameter Sensitivity",
          "Shows how final validation loss changes with learning rate, batch size, weight decay, and dropout.",
          ["hyperparameters", "sensitivity", "tuning"])


# ─── Embeddings ──────────────────────────────────────────────────────────────

def build_embedding_space_pca():
    from sklearn.decomposition import PCA
    np.random.seed(42)

    # Simulate word embeddings in clusters
    categories = {"Animals": 5, "Colors": 4, "Numbers": 4}
    embeddings = []
    labels = []
    cats = []
    for cat, n in categories.items():
        center = np.random.randn(64) * 2
        for i in range(n):
            emb = center + np.random.randn(64) * 0.3
            embeddings.append(emb)
            labels.append(f"{cat[:-1]}_{i}")
            cats.append(cat)

    X = np.array(embeddings)
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        cat_names = list(categories.keys())
        colors = ["cyan", "lime", "orange"]
        for cat_name, color in zip(cat_names, colors):
            mask = [c == cat_name for c in cats]
            pts = X2[mask]
            ax.scatter(pts[:, 0], pts[:, 1], c=color, label=cat_name, s=60, edgecolors="white", linewidths=0.5)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title("Embedding Space (PCA Projection)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    _save("embeddings", "embedding_space_pca", fig,
          "Embedding Space PCA",
          "Semantically similar words cluster together in embedding space, visible via PCA projection.",
          ["embeddings", "pca", "clusters"])


def build_cosine_similarity_matrix():
    np.random.seed(42)
    words = ["king", "queen", "man", "woman", "prince", "princess", "dog", "cat"]
    n = len(words)
    # Simulate embeddings with semantic structure
    base = np.random.randn(n, 64)
    # Make royalty similar
    base[0] += base[1] * 0.8
    base[4] += base[0] * 0.6
    base[5] += base[1] * 0.6
    # Make gender pairs similar
    base[2] += base[3] * 0.5
    # Make animals similar
    base[6] += base[7] * 0.7

    norms = np.linalg.norm(base, axis=1, keepdims=True)
    normed = base / norms
    sim = normed @ normed.T

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(sim, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=words, yticklabels=words, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Cosine Similarity Matrix", fontweight="bold")
        fig.tight_layout()
    _save("embeddings", "cosine_similarity_matrix", fig,
          "Cosine Similarity Matrix",
          "Cosine similarity reveals semantic relationships between word embeddings.",
          ["embeddings", "cosine-similarity", "semantic"])


# ─── Training Objectives ─────────────────────────────────────────────────────

def build_mlm_vs_autoregressive():
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        tokens = ["The", "cat", "sat", "on", "the", "mat"]
        n = len(tokens)

        # MLM: random masking
        ax = axes[0]
        masked = [1, 3]  # mask positions
        for i, tok in enumerate(tokens):
            color = "red" if i in masked else "cyan"
            label = "[MASK]" if i in masked else tok
            ax.text(i, 0.5, label, ha="center", va="center", fontsize=11,
                    color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="gray" if i in masked else "none",
                              edgecolor=color, alpha=0.5))
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1)
        ax.set_title("Masked Language Modeling (BERT)", fontweight="bold")
        ax.axis("off")

        # Autoregressive: left-to-right
        ax = axes[1]
        for i, tok in enumerate(tokens):
            ax.text(i, 0.5, tok, ha="center", va="center", fontsize=11,
                    color="cyan", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="none", edgecolor="cyan", alpha=0.5))
            if i < n - 1:
                ax.annotate("", xy=(i + 0.4, 0.5), xytext=(i + 0.6, 0.5),
                            arrowprops=dict(arrowstyle="->", color="lime", lw=1.5))
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1)
        ax.set_title("Autoregressive LM (GPT)", fontweight="bold")
        ax.axis("off")

        fig.suptitle("MLM vs Autoregressive Training", fontweight="bold")
        fig.tight_layout()
    _save("training_objectives", "mlm_vs_autoregressive", fig,
          "MLM vs Autoregressive",
          "BERT masks random tokens and predicts them; GPT predicts each token from left-to-right context.",
          ["mlm", "autoregressive", "bert", "gpt"])


# ─── Fine-Tuning Methods ─────────────────────────────────────────────────────

def build_lora_matrices():
    np.random.seed(42)
    d = 64
    r = 4
    W = np.random.randn(d, d) * 0.1
    A = np.random.randn(d, r) * 0.01
    B = np.random.randn(r, d) * 0.01
    delta = A @ B

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        titles = ["W (frozen)", "A (d×r)", "B (r×d)", "ΔW = A·B"]
        data = [W, A, B, delta]
        for ax, title, d_arr in zip(axes, titles, data):
            im = ax.imshow(d_arr, cmap="coolwarm", aspect="auto")
            ax.set_title(title, fontweight="bold", fontsize=10)
            ax.tick_params(labelsize=6)
        fig.suptitle(f"LoRA: Low-Rank Adaptation (rank={r})", fontweight="bold")
        fig.tight_layout()
    _save("fine_tuning_methods", "lora_matrices", fig,
          "LoRA Matrix Decomposition",
          "LoRA freezes the original weight matrix W and trains low-rank matrices A and B.",
          ["lora", "fine-tuning", "low-rank"])


def build_lora_parameter_comparison():
    model_sizes = ["125M", "350M", "1.3B", "7B", "13B"]
    full_params = [125, 350, 1300, 7000, 13000]
    lora_params = [p * 0.003 for p in full_params]  # ~0.3% of params

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(model_sizes))
        w = 0.35
        ax.bar(x - w / 2, full_params, w, label="Full Fine-Tuning", color="red", alpha=0.8)
        ax.bar(x + w / 2, lora_params, w, label="LoRA", color="cyan", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Model Size")
        ax.set_ylabel("Trainable Parameters (M)")
        ax.set_xticks(x)
        ax.set_xticklabels(model_sizes)
        ax.set_title("Parameter Efficiency: Full vs LoRA", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
    _save("fine_tuning_methods", "lora_parameter_comparison", fig,
          "LoRA Parameter Comparison",
          "LoRA trains only ~0.3% of parameters compared to full fine-tuning, scaling efficiently.",
          ["lora", "parameter-efficiency", "comparison"])


# ─── Regularization ──────────────────────────────────────────────────────────

def build_dropout_behavior():
    np.random.seed(42)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        rates = [0.0, 0.3, 0.5]
        for ax, rate in zip(axes, rates):
            mask = np.random.binomial(1, 1 - rate, (8, 8)).astype(float)
            sns.heatmap(mask, annot=False, cmap="Blues", vmin=0, vmax=1, ax=ax, cbar=False,
                        linewidths=0.5, linecolor="gray")
            active = mask.sum() / mask.size
            ax.set_title(f"Dropout p={rate}\n({active:.0%} active)", fontweight="bold", fontsize=10)
        fig.suptitle("Dropout Mask Behavior", fontweight="bold")
        fig.tight_layout()
    _save("regularization", "dropout_behavior", fig,
          "Dropout Behavior",
          "Dropout randomly zeros out neurons during training. Higher rates = more neurons dropped.",
          ["dropout", "regularization", "mask"])


def build_dropout_training_comparison():
    np.random.seed(42)
    steps = np.arange(100)
    rates = [0.0, 0.2, 0.5]

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for rate in rates:
            train_loss = 2.0 * np.exp(-0.03 * steps) + 0.2 + np.random.normal(0, 0.05, len(steps))
            val_loss = 2.0 * np.exp(-0.025 * steps) + 0.3 + rate * 0.1
            if rate == 0.0:
                val_loss += 0.3 * (steps / 100)  # Overfitting
            val_loss += np.random.normal(0, 0.05, len(steps))
            axes[0].plot(steps, train_loss, linewidth=1.5, label=f"p={rate}")
            axes[1].plot(steps, val_loss, linewidth=1.5, label=f"p={rate}")
        axes[0].set_title("Training Loss", fontweight="bold")
        axes[1].set_title("Validation Loss", fontweight="bold")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Effect of Dropout on Training", fontweight="bold")
        fig.tight_layout()
    _save("regularization", "dropout_training_comparison", fig,
          "Dropout Training Comparison",
          "Without dropout (p=0) the model overfits; moderate dropout (p=0.2) gives best validation loss.",
          ["dropout", "overfitting", "training"])


# ─── Dimensionality Reduction ────────────────────────────────────────────────

def build_pca_explained_variance():
    from sklearn.decomposition import PCA
    np.random.seed(42)
    X = np.random.randn(200, 20)
    X[:, :3] = X[:, :3] * 5  # First 3 components dominate

    pca = PCA().fit(X)
    var_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(range(1, len(var_ratio) + 1), var_ratio, color="cyan", alpha=0.8)
        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Explained Variance Ratio")
        axes[0].set_title("Scree Plot", fontweight="bold")
        axes[0].grid(True, alpha=0.3, axis="y")

        axes[1].plot(range(1, len(cumulative) + 1), cumulative, "o-", color="lime", linewidth=2)
        axes[1].axhline(y=0.95, color="red", linestyle="--", label="95% threshold")
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Explained Variance")
        axes[1].set_title("Cumulative Variance", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.suptitle("PCA Explained Variance", fontweight="bold")
        fig.tight_layout()
    _save("dimensionality_reduction", "pca_explained_variance", fig,
          "PCA Explained Variance",
          "Scree plot and cumulative variance show how many principal components capture most of the data variance.",
          ["pca", "explained-variance", "scree-plot"])


def build_pca_projection():
    from sklearn.decomposition import PCA
    np.random.seed(42)
    # 2D data with clear principal axes
    angle = np.pi / 4
    X = np.random.randn(150, 2) @ np.array([[2, 0], [0, 0.5]])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X = X @ rot

    pca = PCA(n_components=2).fit(X)
    X_proj = pca.transform(X)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5, s=15, color="cyan")
        # Draw principal component directions
        for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
            scale = np.sqrt(var) * 2
            axes[0].annotate("", xy=pca.mean_ + comp * scale, xytext=pca.mean_,
                             arrowprops=dict(arrowstyle="->", color="red" if i == 0 else "lime", lw=2))
        axes[0].set_title("Original Data + PCs", fontweight="bold")
        axes[0].set_aspect("equal")
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(X_proj[:, 0], X_proj[:, 1], alpha=0.5, s=15, color="cyan")
        axes[1].set_title("PC Space", fontweight="bold")
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle("PCA Projection", fontweight="bold")
        fig.tight_layout()
    _save("dimensionality_reduction", "pca_projection", fig,
          "PCA Projection",
          "Left: original data with principal component directions. Right: data projected onto PC axes.",
          ["pca", "projection", "dimensionality-reduction"])


# ─── Transformer Deep Dive ───────────────────────────────────────────────────

def build_positional_encoding():
    d_model = 64
    max_len = 50
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pe, cmap="RdBu", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Position")
        ax.set_title("Sinusoidal Positional Encoding", fontweight="bold")
        fig.tight_layout()
    _save("transformer_deep_dive", "positional_encoding", fig,
          "Positional Encoding",
          "Sinusoidal positional encodings give each position a unique pattern using sin/cos at different frequencies.",
          ["positional-encoding", "transformer", "sinusoidal"])


def build_transformer_shapes():
    steps = [
        {"label": "Input Tokens", "shape": [1, 128]},
        {"label": "Token Embeddings", "shape": [1, 128, 768]},
        {"label": "+ Positional Encoding", "shape": [1, 128, 768]},
        {"label": "Q, K, V Projections", "shape": [1, 12, 128, 64]},
        {"label": "Attention Scores", "shape": [1, 12, 128, 128]},
        {"label": "Attention Output", "shape": [1, 12, 128, 64]},
        {"label": "Concat Heads", "shape": [1, 128, 768]},
        {"label": "FFN Hidden", "shape": [1, 128, 3072]},
        {"label": "FFN Output", "shape": [1, 128, 768]},
        {"label": "Logits", "shape": [1, 128, 50257]},
    ]

    n = len(steps)
    volumes = [max(1, np.prod(s["shape"])) for s in steps]
    max_vol = max(volumes)
    widths = [0.2 + 0.8 * (v / max_vol) for v in volumes]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        y_pos = list(range(n))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n))
        bars = ax.barh(y_pos, widths, color=colors, height=0.6)
        for i, (bar, step) in enumerate(zip(bars, steps)):
            shape_str = " × ".join(str(d) for d in step["shape"])
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"[{shape_str}]", va="center", fontsize=8, color="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([s["label"] for s in steps], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.5)
        ax.set_xticks([])
        ax.set_title("Tensor Shapes Through a Transformer", fontweight="bold")
        fig.tight_layout()
    _save("transformer_deep_dive", "transformer_shapes", fig,
          "Transformer Tensor Shapes",
          "Shows how tensor dimensions change as data flows through a transformer layer.",
          ["transformer", "shapes", "dimensions"])


# ─── Main ────────────────────────────────────────────────────────────────────

ALL_BUILDERS = [
    build_attention_basic_heatmap,
    build_attention_all_heads,
    build_causal_mask,
    build_attention_patterns,
    build_cross_entropy_curve,
    build_activation_functions,
    build_kl_divergence,
    build_loss_landscape,
    build_embedding_gradient_sparsity,
    build_gradient_flow_layers,
    build_learning_rate_comparison,
    build_hyperparameter_sensitivity,
    build_embedding_space_pca,
    build_cosine_similarity_matrix,
    build_mlm_vs_autoregressive,
    build_lora_matrices,
    build_lora_parameter_comparison,
    build_dropout_behavior,
    build_dropout_training_comparison,
    build_pca_explained_variance,
    build_pca_projection,
    build_positional_encoding,
    build_transformer_shapes,
]


def main():
    print(f"Building visualization library → {OUT}")
    print(f"Generating {len(ALL_BUILDERS)} visualizations...\n")

    for builder in ALL_BUILDERS:
        try:
            builder()
        except Exception as e:
            print(f"  ✗ {builder.__name__}: {e}")

    # Write manifest
    output = {"version": 1, "visualizations": manifest}
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(output, f, indent=2)

    total = sum(len(v) for v in manifest.values())
    print(f"\nDone! {total} visualizations across {len(manifest)} topics.")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
