
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional


class AttnFlowVisualizer:
    """
    n_layers : int
    n_columns : int
    buffer_size : int

    """

    def __init__(self, n_layers: int, n_columns: int, buffer_size: int = 500):
        self.n_layers = n_layers
        self.n_columns = n_columns
        self._buffers: list[deque] = [
            deque(maxlen=buffer_size) for _ in range(n_layers)
        ]

    def update(self, attn_weights: list):
        """
        attn_weights : list[Tensor | None]
            n_layers.  shape=(n_cols, n_cols)
        """
        for layer_idx, w in enumerate(attn_weights):
            if w is None:
                continue
            # w: (n_cols, n_cols)
            w_np = w.detach().float().cpu().numpy()
            if w_np.ndim == 3:
                w_np = w_np.mean(0)
            self._buffers[layer_idx].append(w_np)


    def build_figure(self) -> plt.Figure:
        n_layers = self.n_layers
        fig, axes = plt.subplots(
            1, n_layers,
            figsize=(4 * n_layers, 4),
            squeeze=False,
        )
        axes = axes[0]

        col_labels = [f"C{j}" for j in range(self.n_columns)]

        for layer_idx in range(n_layers):
            buf = self._buffers[layer_idx]
            if len(buf) == 0:
                axes[layer_idx].set_title(f"Layer {layer_idx} — no data")
                continue

            mean_w = np.stack(buf).mean(axis=0)  # [n_cols, n_cols]

            ax = axes[layer_idx]
            im = ax.imshow(mean_w, vmin=0.0, vmax=1.0, cmap="viridis",
                           aspect="equal")
            ax.set_title(f"Layer {layer_idx}", fontsize=11)
            ax.set_xlabel("Key column (source)")
            ax.set_ylabel("Query column (receiver)")
            ax.set_xticks(range(self.n_columns))
            ax.set_yticks(range(self.n_columns))
            ax.set_xticklabels(col_labels)
            ax.set_yticklabels(col_labels)

            for i in range(self.n_columns):
                for j in range(self.n_columns):
                    ax.text(j, i, f"{mean_w[i, j]:.2f}",
                            ha="center", va="center",
                            color="white" if mean_w[i, j] < 0.5 else "black",
                            fontsize=8)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("Attention Weight Flow Matrix", fontsize=13, y=1.02)
        fig.tight_layout()
        return fig

    def log(self, logger, step: int):
        import io
        from PIL import Image as PILImage
        from aim import Image as AimImage

        for layer_idx in range(self.n_layers):
            buf = self._buffers[layer_idx]
            if len(buf) == 0:
                continue

            mean_w = np.stack(buf).mean(axis=0)  # [n_cols, n_cols]
            col_labels = [f"C{j}" for j in range(self.n_columns)]

            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(mean_w, vmin=0.0, vmax=1.0,
                        cmap="viridis", aspect="equal")
            ax.set_title(f"Attn Flow Layer {layer_idx}")
            ax.set_xlabel("Key col (source)")
            ax.set_ylabel("Query col (receiver)")
            ax.set_xticks(range(self.n_columns))
            ax.set_yticks(range(self.n_columns))
            ax.set_xticklabels(col_labels)
            ax.set_yticklabels(col_labels)
            for i in range(self.n_columns):
                for j in range(self.n_columns):
                    ax.text(j, i, f"{mean_w[i,j]:.2f}",
                            ha="center", va="center", fontsize=9,
                            color="white" if mean_w[i,j] < 0.5 else "black")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='png', dpi=120, bbox_inches='tight')
            io_buf.seek(0)
            logger.track(
                AimImage(PILImage.open(io_buf)),
                name=f"attn_flow_layer_{layer_idx}",
                step=step,
            )
            plt.close(fig)
            io_buf.close()