from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from knitwork.visualization.cka import linear_cka


class ColumnProbe:
    """Context manager that records per-column activations from grid_step_postmsg.

    Usage:
        with ColumnProbe(model) as probe:
            for tokens, h in ...:
                y, h = model(tokens, h)
        acts = probe.get_tensor()  # [T, layers, cols, batch, H]
    """

    def __init__(self, model):
        self.model = model
        self._steps: list[torch.Tensor] = []
        self._orig = None

    def __enter__(self):
        orig = self.model.grid_step_postmsg
        steps = self._steps

        def _hooked(x, *, h, return_attn=True):
            h_n, extras = orig(x, h=h, return_attn=return_attn)
            steps.append(h_n.detach().cpu().float())
            return h_n, extras

        self._orig = orig
        self.model.grid_step_postmsg = _hooked
        return self

    def __exit__(self, *_):
        if self._orig is not None:
            self.model.grid_step_postmsg = self._orig
            self._orig = None

    def get_tensor(self) -> torch.Tensor:
        """Returns [T, layers, cols, batch, H]."""
        return torch.stack(self._steps, dim=0)

    def clear(self):
        self._steps.clear()


def mean_activation_norm(acts: torch.Tensor) -> torch.Tensor:
    """Mean L2 norm of activations per column, averaged over T, layers, batch.

    acts: [T, layers, cols, batch, H]
    returns: [cols]
    """
    # norm over H dim, then mean over T, layers, batch
    norms = acts.norm(dim=-1)  # [T, layers, cols, batch]
    return norms.mean(dim=(0, 1, 3))  # [cols]


def col_token_correlation(
    acts: torch.Tensor,
    tokens: torch.Tensor,
    vocab_size: int,
    layer: int = -1,
) -> torch.Tensor:
    """Mean activation norm per column per input token.

    acts:   [T, layers, cols, batch, H]
    tokens: [T, batch] — token ids aligned with acts steps
    returns: [cols, vocab_size] — normalized to [0, 1] per column
    """
    T, n_layers, n_cols, batch, H = acts.shape
    if layer < 0:
        layer = n_layers + layer

    result = torch.zeros(n_cols, vocab_size)
    count = torch.zeros(vocab_size)

    col_acts = acts[:, layer, :, :, :]  # [T, cols, batch, H]
    norms = col_acts.norm(dim=-1)        # [T, cols, batch]

    for t in range(T):
        for b in range(batch):
            tok = int(tokens[t, b].item())
            if tok < 0 or tok >= vocab_size:
                continue
            result[:, tok] += norms[t, :, b]
            count[tok] += 1

    nonzero = count > 0
    result[:, nonzero] /= count[nonzero].unsqueeze(0)
    # normalize per column to [0, 1]
    col_max = result.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    return result / col_max


def col_cka_matrix(acts: torch.Tensor, layer: int = -1) -> np.ndarray:
    """CKA similarity matrix between columns.

    acts:    [T, layers, cols, batch, H]
    returns: [cols, cols] in [0, 1]; low off-diagonal = high specialization
    """
    T, n_layers, n_cols, batch, H = acts.shape
    if layer < 0:
        layer = n_layers + layer

    col_acts = acts[:, layer, :, :, :]  # [T, cols, batch, H]
    # flatten T and batch into samples
    states = []
    for c in range(n_cols):
        x = col_acts[:, c, :, :].reshape(-1, H).numpy()  # [T*batch, H]
        states.append(x)

    mat = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            n = min(len(states[i]), len(states[j]))
            mat[i, j] = linear_cka(states[i][:n], states[j][:n])
    return mat


def top_activating_contexts(
    acts: torch.Tensor,
    tokens: torch.Tensor,
    k: int = 10,
    n: int = 2,
    layer: int = -1,
) -> dict[int, list[tuple[tuple, float]]]:
    """Top-k n-gram contexts that maximally activate each column.

    acts:   [T, layers, cols, batch, H]
    tokens: [T, batch]
    returns: {col_idx: [(ngram_tuple, mean_norm), ...]} sorted descending
    """
    T, n_layers, n_cols, batch, H = acts.shape
    if layer < 0:
        layer = n_layers + layer

    col_acts = acts[:, layer, :, :, :]   # [T, cols, batch, H]
    norms = col_acts.norm(dim=-1)         # [T, cols, batch]

    # accumulate norm per (col, ngram)
    ngram_norms: list[dict] = [defaultdict(list) for _ in range(n_cols)]

    for t in range(n - 1, T):
        for b in range(batch):
            ngram = tuple(int(tokens[t - i, b].item()) for i in range(n - 1, -1, -1))
            for c in range(n_cols):
                ngram_norms[c][ngram].append(float(norms[t, c, b]))

    result = {}
    for c in range(n_cols):
        scored = [(ng, float(np.mean(vs))) for ng, vs in ngram_norms[c].items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        result[c] = scored[:k]
    return result


def plot_activation_norms(norms: torch.Tensor) -> plt.Figure:
    """Bar chart of mean activation norm per column.

    norms: [cols]
    """
    n_cols = len(norms)
    fig, ax = plt.subplots(figsize=(max(4, n_cols), 3))
    ax.bar(range(n_cols), norms.numpy())
    ax.set_xlabel("Column")
    ax.set_ylabel("Mean activation norm")
    ax.set_title("Per-column activation norms")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"C{c}" for c in range(n_cols)])
    fig.tight_layout()
    return fig


def plot_token_correlation(
    corr: torch.Tensor,
    id2token: Optional[dict] = None,
    top_tokens: int = 30,
) -> plt.Figure:
    """Heatmap of per-column token correlation.

    corr: [cols, vocab_size]
    """
    n_cols, vocab_size = corr.shape
    # select top_tokens by max activation across columns
    col_max = corr.max(dim=0).values
    top_idx = col_max.argsort(descending=True)[:top_tokens].numpy()

    data = corr[:, top_idx].numpy()
    labels = [id2token.get(int(i), str(i)) if id2token else str(i) for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(8, top_tokens // 2), max(3, n_cols)))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(n_cols))
    ax.set_yticklabels([f"C{c}" for c in range(n_cols)])
    ax.set_xticks(range(top_tokens))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_title("Per-column token correlation (normalized)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    return fig


def plot_cka_matrix(mat: np.ndarray, layer: int = 0) -> plt.Figure:
    n_cols = mat.shape[0]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="equal")
    ax.set_title(f"Column CKA — layer {layer}")
    ax.set_xlabel("Column j")
    ax.set_ylabel("Column i")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_cols))
    col_labels = [f"C{c}" for c in range(n_cols)]
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(col_labels)
    for i in range(n_cols):
        for j in range(n_cols):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
