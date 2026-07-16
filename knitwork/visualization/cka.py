from __future__ import annotations
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from knitwork.common.torch import to_numpy
from knitwork.common.tracking import EmaTracker


def _centering(K: np.ndarray) -> np.ndarray:
    """Центрирование ядерной матрицы."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Линейная CKA между двумя матрицами [n_samples, d].
    Возвращает скаляр в [0, 1].
    """
    K = X @ X.T
    L = Y @ Y.T
    Kc = _centering(K)
    Lc = _centering(L)
    hsic_kl = np.sum(Kc * Lc)
    hsic_kk = np.sum(Kc * Kc)
    hsic_ll = np.sum(Lc * Lc)
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return float(hsic_kl / denom)


class CKAVisualizer:
    """
    n_layers, n_columns : int
    buffer_size : int
    """

    def __init__(self, n_layers: int, n_columns: int, buffer_size: int = 200):
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.buffer_size = buffer_size
        # [layer][col] → deque of np.ndarray [batch, hidden]
        self._buffers: list[list[deque]] = [
            [deque(maxlen=buffer_size) for _ in range(n_columns)]
            for _ in range(n_layers)
        ]

    def update(self, h: torch.Tensor):
        """
 
        h : Tensor  shape=(n_layers, n_cols, batch, hidden_size)
        """
        h_np = h.detach().float().cpu().numpy()
        for layer_idx in range(self.n_layers):
            for col_idx in range(self.n_columns):
                self._buffers[layer_idx][col_idx].append(
                    h_np[layer_idx, col_idx]   # [batch, hidden]
                )

    def compute_cka_matrices(self) -> list[np.ndarray]:
        """
        [n_cols × n_cols] for layers.
        """
        cka_matrices = []
        for layer_idx in range(self.n_layers):
            mat = np.zeros((self.n_columns, self.n_columns))
            # собираем накопленные состояния
            states = []
            for col_idx in range(self.n_columns):
                buf = self._buffers[layer_idx][col_idx]
                if len(buf) == 0:
                    states.append(None)
                else:
                    states.append(np.concatenate(list(buf), axis=0))

            for i in range(self.n_columns):
                for j in range(self.n_columns):
                    if states[i] is None or states[j] is None:
                        mat[i, j] = float("nan")
                    else:
                        n = min(len(states[i]), len(states[j]))
                        mat[i, j] = linear_cka(states[i][:n], states[j][:n])

            cka_matrices.append(mat)
        return cka_matrices

    def build_figure(self) -> plt.Figure:
        cka_matrices = self.compute_cka_matrices()
        n_layers = self.n_layers
        fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4),
                                 squeeze=False)
        axes = axes[0]
        col_labels = [f"C{j}" for j in range(self.n_columns)]

        for layer_idx, (ax, mat) in enumerate(zip(axes, cka_matrices)):
            im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdYlGn_r",
                           aspect="equal")
            ax.set_title(f"Layer {layer_idx} CKA", fontsize=11)
            ax.set_xlabel("Column j")
            ax.set_ylabel("Column i")
            ax.set_xticks(range(self.n_columns))
            ax.set_yticks(range(self.n_columns))
            ax.set_xticklabels(col_labels)
            ax.set_yticklabels(col_labels)
            for i in range(self.n_columns):
                for j in range(self.n_columns):
                    ax.text(j, i, f"{mat[i,j]:.2f}",
                            ha="center", va="center", fontsize=8,
                            color="black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("CKA between column hidden states", fontsize=13, y=1.02)
        fig.tight_layout()
        return fig

    def log(self, logger, step: int):
        import io
        from PIL import Image as PILImage
        try:
            from aim import Image as AimImage
            _wrap = AimImage
        except ImportError:
            _wrap = lambda x: x  # Logger.track() accepts PIL directly

        cka_matrices = self.compute_cka_matrices()
        col_labels = [f"C{j}" for j in range(self.n_columns)]

        for layer_idx, mat in enumerate(cka_matrices):
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(mat, vmin=0.0, vmax=1.0,
                        cmap="RdYlGn_r", aspect="equal")
            ax.set_title(f"CKA Layer {layer_idx}")
            ax.set_xlabel("Column j")
            ax.set_ylabel("Column i")
            ax.set_xticks(range(self.n_columns))
            ax.set_yticks(range(self.n_columns))
            ax.set_xticklabels(col_labels)
            ax.set_yticklabels(col_labels)
            for i in range(self.n_columns):
                for j in range(self.n_columns):
                    ax.text(j, i, f"{mat[i,j]:.2f}",
                            ha="center", va="center", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='png', dpi=120, bbox_inches='tight')
            io_buf.seek(0)
            logger.track(
                _wrap(PILImage.open(io_buf)),
                name=f"cka_layer_{layer_idx}",
                step=step,
            )
            plt.close(fig)
            io_buf.close()


class CKAVisualizerNew:
    # stable fig ids allow the object reuse to improve performance and memory consumption
    _fig_id = "CKA Layer {layer_idx}"

    def __init__(self, n_layers: int, n_columns: int, *, lr):
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.tracker = EmaTracker(lr)

    def update(self, h: torch.Tensor):
        # (n_layers, n_cols, batch, hidden_size)
        h = to_numpy(h)
        cka = {}
        for layer_idx in range(self.n_layers):
            # [cols, batch, hidden]
            layer_h = h[layer_idx]
            layer_cka = np.zeros((self.n_columns, self.n_columns))
            for i in range(self.n_columns):
                for j in range(self.n_columns):
                    layer_cka[i, j] = linear_cka(layer_h[i], layer_h[j])
            cka[layer_idx] = layer_cka
        self.tracker.put(cka)

    def get_figures(self):
        col_labels = [f"C{j}" for j in range(self.n_columns)]
        figures = {}
        
        for layer_idx, cka in self.tracker.get().items():
            cka = to_numpy(cka, copy=False)
            fig_id = self._fig_id.format(layer_idx=layer_idx)
            fig = plt.figure(num=fig_id, clear=True)
            ax = fig.subplots()

            im = ax.imshow(cka, vmin=0.0, vmax=1.0, cmap="RdYlGn_r", aspect="equal")

            ax.set_title(f"CKA Layer {layer_idx}")
            ax.set_xlabel("Column j")
            ax.set_ylabel("Column i")
            ax.set_xticks(range(self.n_columns))
            ax.set_yticks(range(self.n_columns))
            ax.set_xticklabels(col_labels)
            ax.set_yticklabels(col_labels)
            for i in range(self.n_columns):
                for j in range(self.n_columns):
                    ax.text(
                        j, i, f"{cka[i,j]:.2f}",
                        ha="center", va="center", fontsize=9,
                    )

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            figures[f'cka_layer_{layer_idx}'] = fig

        return figures
