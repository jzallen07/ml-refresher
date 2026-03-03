"""Visualization rendering engine.

Pure functions that produce matplotlib PNGs and return file paths.
All renders use the Agg backend and dark_background style.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_STYLE = "dark_background"
_DPI = 150
_FIGSIZE = (6, 4)  # 600x400 at 150 DPI

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "viz_library"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def _save_fig(fig: plt.Figure) -> str:
    """Save figure to a temp PNG and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return tmp.name


def render_heatmap(
    data: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colormap: str = "YlOrRd",
    annotate: bool = True,
) -> str:
    """Render a heatmap and return the path to the PNG."""
    arr = np.array(data, dtype=float)
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        im = ax.imshow(arr, cmap=colormap, aspect="auto")
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)

        if annotate and arr.size <= 100:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    color = "white" if val < (arr.max() + arr.min()) / 2 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=color, fontsize=7)

        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
    return _save_fig(fig)


def _eval_expression(expression: str, x: np.ndarray) -> np.ndarray:
    """Evaluate a math expression in a restricted namespace.

    Only ``np``, ``math``, and ``x`` are available — builtins are disabled so
    this cannot access the filesystem, import modules, or call dangerous
    functions.  This is intentionally restricted to plotting math expressions
    provided by the agent (e.g. ``"np.maximum(0, x)"``).
    """
    safe_ns = {"np": np, "math": math, "x": x, "__builtins__": {}}
    return eval(expression, safe_ns)  # noqa: S307


def render_function_plot(
    functions: list[dict],
    x_range: tuple[float, float] = (-5.0, 5.0),
    title: str = "Function Plot",
    x_label: str = "x",
    y_label: str = "y",
) -> str:
    """Plot mathematical functions and return the path to the PNG.

    Each function dict has ``expression`` (e.g. ``"np.maximum(0, x)"``) and
    ``label``.  Expressions are evaluated in a restricted namespace containing
    only ``np``, ``math``, and ``x``.
    """
    x = np.linspace(x_range[0], x_range[1], 500)

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE)

        for fn in functions:
            expr = fn["expression"]
            label = fn.get("label", expr)
            y = _eval_expression(expr, x)
            ax.plot(x, y, label=label, linewidth=2)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return _save_fig(fig)


def render_tensor_shapes(
    steps: list[dict],
    title: str = "Tensor Shape Flow",
) -> str:
    """Visualise tensor dimensions flowing through layers.

    Each step dict has ``label`` (str) and ``shape`` (list[int]).
    """
    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=(_FIGSIZE[0] + 2, _FIGSIZE[1]))

        n = len(steps)
        y_positions = list(range(n))
        shapes = [s["shape"] for s in steps]
        volumes = [max(1, np.prod(s)) for s in shapes]
        max_vol = max(volumes)

        # Normalise bar widths
        widths = [0.2 + 0.8 * (v / max_vol) for v in volumes]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, n))

        bars = ax.barh(y_positions, widths, color=colors, height=0.6)

        for i, (bar, step) in enumerate(zip(bars, steps)):
            shape_str = " x ".join(str(d) for d in step["shape"])
            ax.text(
                bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"[{shape_str}]",
                va="center", fontsize=9, color="white",
            )

        labels = [s["label"] for s in steps]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.4)
        ax.set_xticks([])
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
    return _save_fig(fig)


# ── Pre-generated visualization library ──────────────────────────────────────

def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"version": 1, "visualizations": {}}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def get_visualization(topic: str, name: str) -> dict | None:
    """Look up a pre-generated visualization by topic and name."""
    manifest = _load_manifest()
    vizs = manifest.get("visualizations", {}).get(topic, [])
    for v in vizs:
        if v["name"] == name:
            path = DATA_DIR / v["file"]
            if path.exists():
                return {
                    "type": "visualization",
                    "image_path": str(path),
                    "title": v.get("title", name),
                    "description": v.get("description", ""),
                }
            return None
    return None


def list_visualizations(topic: str | None = None) -> list[dict]:
    """Return a catalogue of available pre-generated visualizations."""
    manifest = _load_manifest()
    all_vizs = manifest.get("visualizations", {})
    results: list[dict] = []
    for t, items in all_vizs.items():
        if topic and t != topic:
            continue
        for v in items:
            results.append({"topic": t, "name": v["name"], "title": v.get("title", ""), "description": v.get("description", "")})
    return results
