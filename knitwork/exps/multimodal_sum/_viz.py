"""MDS visualization helpers — attention heatmap + sum confusion matrix."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from knitwork.exps.sdq._viz import fig_to_numpy, log_figure  # noqa: F401 (re-exported)


def plot_attn_heatmap(attn_weights: list, n_layers: int, n_columns: int, signal_columns, step: int):
    """attn_weights: per-layer [n_columns, n_columns] arrays (src -> dst)."""
    mats = [w for w in attn_weights if w is not None]
    if not mats:
        return None
    fig, axes = plt.subplots(1, len(mats), figsize=(3.2 * len(mats), 3), squeeze=False)
    axes = axes[0]
    for li, (ax, mat) in enumerate(zip(axes, mats)):
        mat = np.asarray(mat)
        im = ax.imshow(mat, vmin=0, vmax=max(mat.max(), 1e-6), cmap='Blues', aspect='auto')
        ax.set_title(f'Layer {li}')
        ax.set_xlabel('dst col')
        ax.set_ylabel('src col')
        ticks = range(n_columns)
        labels = [f'{c}{"*" if c in signal_columns else ""}' for c in ticks]
        ax.set_xticks(list(ticks)); ax.set_xticklabels(labels)
        ax.set_yticks(list(ticks)); ax.set_yticklabels(labels)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'Column attention (src->dst), * = signal col | step={step:,}', fontsize=10)
    plt.tight_layout()
    return fig


def plot_sum_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, step: int):
    if len(y_true) == 0:
        return None
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(max(5, n_classes * 0.35), max(4, n_classes * 0.35)))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set_xlabel('predicted sum')
    ax.set_ylabel('true sum')
    ax.set_title(f'Sum confusion (row-normalized) | step={step:,}')
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    return fig


def estimate_sum_location_r2(state, sum_target: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """
    Where does the network linearly represent the running sum?
    Fits a closed-form ridge regression per (layer, column) hidden state ->
    ground-truth running sum, returns an [n_layers, n_columns] grid of R^2.
    """
    h = state.detach().float().cpu().numpy()
    n_layers, n_cols, batch, hidden = h.shape
    y = sum_target.astype(np.float64)
    y_c = y - y.mean()
    var_y = float((y_c ** 2).sum())

    r2 = np.zeros((n_layers, n_cols))
    if var_y < 1e-8:
        return r2
    for li in range(n_layers):
        for ci in range(n_cols):
            X = h[li, ci].astype(np.float64)
            X_c = X - X.mean(axis=0, keepdims=True)
            XtX = X_c.T @ X_c
            XtX.flat[::hidden + 1] += ridge
            w = np.linalg.solve(XtX, X_c.T @ y_c)
            resid = y_c - X_c @ w
            r2[li, ci] = 1.0 - float((resid ** 2).sum()) / var_y
    return r2


def plot_sum_probe_r2(r2: np.ndarray, signal_columns, step: int):
    n_layers, n_columns = r2.shape
    fig, ax = plt.subplots(figsize=(max(4, n_columns * 0.9), max(3, n_layers * 0.8)))
    im = ax.imshow(r2, vmin=0, vmax=1, cmap='magma', aspect='auto')
    ax.set_xlabel('column')
    ax.set_ylabel('layer')
    ax.set_xticks(range(n_columns))
    ax.set_xticklabels([f'{c}{"*" if c in signal_columns else ""}' for c in range(n_columns)])
    ax.set_yticks(range(n_layers))
    for li in range(n_layers):
        for ci in range(n_columns):
            ax.text(ci, li, f'{r2[li, ci]:.2f}', ha='center', va='center',
                    color='white' if r2[li, ci] < 0.6 else 'black', fontsize=8)
    plt.colorbar(im, ax=ax, label='R² (linear probe -> running sum)')
    ax.set_title(f'Where is the sum represented? (* = signal col) | step={step:,}')
    plt.tight_layout()
    return fig
