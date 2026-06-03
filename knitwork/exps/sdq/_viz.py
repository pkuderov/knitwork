"""SDQ visualization helpers — matplotlib figures + AIM image logging."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from knitwork.visualization.attn_flow import AttnFlowVisualizer
from knitwork.visualization.cka import CKAVisualizer


#AIM image logging

def fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    plt.close(fig)
    return arr[..., :3]


def log_figure(logger, fig: plt.Figure, name: str, step: int) -> None:
    if fig is None:
        return
    try:
        arr = fig_to_numpy(fig)
        try:
            import aim
            logger.track(aim.Image(arr), name=name, step=step)
            return
        except (ImportError, AttributeError):
            pass
        if hasattr(logger, 'track'):
            logger.track(arr, name=name, step=step)
    except Exception as e:
        print(f'[VIS] error logging {name}: {e}')


# Plot functions
def plot_gate_saturation(gate_buffer: list[list[float]], n_layers: int, step: int) -> plt.Figure:
    fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 3), sharey=True)
    if n_layers == 1:
        axes = [axes]
    arr = np.array(gate_buffer) if gate_buffer else np.zeros((1, n_layers))
    for li, ax in enumerate(axes):
        if li < arr.shape[1]:
            ax.hist(arr[:, li], bins=20, range=(0, 1), color='steelblue', edgecolor='white')
        ax.set_title(f'Layer {li}')
        ax.set_xlabel('Gate value')
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color='red', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3)
        if li == 0:
            ax.set_ylabel('Count')
    fig.suptitle(f'Gate Saturation | step={step:,}', fontsize=11)
    plt.tight_layout()
    return fig


def plot_col_similarity_heatmap(
    col_sim: dict[str, float], n_layers: int, n_columns: int, step: int,
) -> plt.Figure:
    pairs = [(ci, cj) for ci in range(n_columns) for cj in range(ci + 1, n_columns)]
    pair_labels = [f"C{ci}-C{cj}" for ci, cj in pairs]
    data = np.zeros((n_layers, len(pairs)))
    for li in range(n_layers):
        for pi, (ci, cj) in enumerate(pairs):
            data[li, pi] = col_sim.get(f"col_sim/L{li}_C{ci}_C{cj}", 0.0)
    fig, ax = plt.subplots(figsize=(max(6, len(pairs) * 1.2), max(4, n_layers * 0.8)))
    im = ax.imshow(data, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_labels, rotation=45, ha='right')
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {li}" for li in range(n_layers)])
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    ax.set_title(f'Column Cosine Similarity | step={step:,}')
    plt.tight_layout()
    return fig


def plot_beta_dynamics(
    beta_history: list[dict[str, float]], step_history: list[int],
    n_layers: int, n_columns: int,
) -> plt.Figure:
    fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 3), sharey=True)
    if n_layers == 1:
        axes = [axes]
    cmap = plt.get_cmap('cool', n_columns)
    for li, ax in enumerate(axes):
        for ci in range(n_columns):
            vals = [h.get(f"hgrn/beta/L{li}_C{ci}", float('nan')) for h in beta_history]
            if any(not np.isnan(v) for v in vals):
                ax.plot(step_history, vals, color=cmap(ci), label=f"C{ci}", linewidth=1.5)
        ax.set_title(f'Layer {li}')
        ax.set_xlabel('Step')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if li == 0:
            ax.set_ylabel('β')
            ax.legend(fontsize=7, loc='upper left')
    fig.suptitle('HGRN β Dynamics', fontsize=12)
    plt.tight_layout()
    return fig


def plot_diversity_components(
    div_history: dict[str, list[float]], step_history: list[int],
) -> plt.Figure:
    keys = ['cosine', 'covariance', 'variance', 'gate_entropy', 'total']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for key, col in zip(keys[:-1], colors[:-1]):
        vals = div_history.get(f'div/{key}', [])
        if vals:
            ax1.plot(step_history[:len(vals)], vals, label=key, color=col, linewidth=1.5)
    ax1.set_title('Diversity Loss Components')
    ax1.set_xlabel('Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    total_vals = div_history.get('div/total', [])
    if total_vals:
        ax2.plot(step_history[:len(total_vals)], total_vals, color=colors[-1], linewidth=2)
    ax2.set_title('Total Diversity Loss')
    ax2.set_xlabel('Step')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_hidden_norm_heatmap(
    norm_history: list[dict[str, float]], n_layers: int, n_columns: int, step: int,
) -> plt.Figure:
    if not norm_history:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig
    data = np.zeros((n_layers, n_columns))
    for li in range(n_layers):
        for ci in range(n_columns):
            vals = [h.get(f"hidden_norm/L{li}_C{ci}", float('nan')) for h in norm_history]
            valid = [v for v in vals if not np.isnan(v)]
            data[li, ci] = np.mean(valid) if valid else 0.0
    fig, ax = plt.subplots(figsize=(max(4, n_columns * 0.9), max(3, n_layers * 0.7)))
    im = ax.imshow(data, cmap='viridis', aspect='auto')
    ax.set_xticks(range(n_columns))
    ax.set_xticklabels([f"Col {ci}" for ci in range(n_columns)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {li}" for li in range(n_layers)])
    plt.colorbar(im, ax=ax, label='Mean L2 Norm')
    ax.set_title(f'Hidden Norm | step={step:,}')
    plt.tight_layout()
    return fig


def plot_grad_norm_per_layer(
    grad_norm_history: list[dict[str, float]], step_history: list[int], n_layers: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap('plasma', n_layers)
    for li in range(n_layers):
        key = f"grad_norm/layer_{li}"
        vals = [h.get(key, float('nan')) for h in grad_norm_history]
        valid = [(s, v) for s, v in zip(step_history, vals) if not np.isnan(v)]
        if valid:
            ss, vs = zip(*valid)
            ax.plot(ss, vs, color=cmap(li), label=f"L{li}", linewidth=1.5)
    ax.set_title('Gradient Norm per Layer')
    ax.set_xlabel('Step')
    ax.set_ylabel('||∇||')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    try:
        ax.set_yscale('log')
    except Exception:
        pass
    plt.tight_layout()
    return fig


# Compute helpers

def compute_grad_norm_per_layer(model, n_layers: int) -> dict[str, float]:
    result = {}
    if not hasattr(model, 'cells'):
        return result
    for li in range(n_layers):
        norms = []
        try:
            for cell in model.cells[li]:
                for p in cell.parameters():
                    if p.grad is not None:
                        norms.append(p.grad.detach().norm().item())
        except (IndexError, TypeError):
            pass
        if norms:
            result[f"grad_norm/layer_{li}"] = float(np.mean(norms))
    return result


def compute_hidden_norms(h: torch.Tensor, n_layers: int, n_columns: int) -> dict[str, float]:
    result = {}
    with torch.no_grad():
        for li in range(n_layers):
            for ci in range(n_columns):
                result[f"hidden_norm/L{li}_C{ci}"] = h[li, ci].float().norm(dim=-1).mean().item()
    return result


#VizManager

class VizManager:
    """Collects visualization data and flushes to AIM at configured intervals."""

    def __init__(self, n_layers: int, n_columns: int, vis_interval: int = 10_000_000):
        self.n_layers    = n_layers
        self.n_columns   = n_columns
        self.interval    = vis_interval
        self.next_step   = vis_interval

        self.attn_vis = AttnFlowVisualizer(n_layers=n_layers, n_columns=n_columns, buffer_size=100)
        self.cka_vis  = CKAVisualizer(n_layers=n_layers, n_columns=n_columns, buffer_size=50)

        self.gate_buffer:        list[list[float]]  = []
        self.col_sim_buffer:     list[dict]          = []
        self.hidden_norm_buffer: list[dict]          = []
        self.beta_buffer:        list[dict]          = []
        self.beta_step_buffer:   list[int]           = []
        self.div_history:        dict[str, list]     = defaultdict(list)
        self.div_step_history:   list[int]           = []
        self.grad_norm_buffer:   list[dict]          = []
        self.grad_step_buffer:   list[int]           = []

    def should_capture(self, step: int) -> bool:
        return step >= self.next_step - self.n_layers  # one step before flush

    def update(
        self, step: int, extras: dict, rnn_state,
        *, has_hgrn: bool, has_fusion: bool, rnn,
    ) -> None:
        if extras.get('attn_weights'):
            self.attn_vis.update(extras['attn_weights'])
        h_for_cka = rnn_state[0] if isinstance(rnn_state, tuple) else rnn_state
        self.cka_vis.update(h_for_cka)

        gate_probs = [
            torch.sigmoid(g).detach().float().mean().item()
            for g in extras.get('gate_logits', extras.get('gates', []))
            if g is not None and isinstance(g, torch.Tensor)
        ]
        if gate_probs:
            self.gate_buffer.append(gate_probs)

        if has_fusion and isinstance(rnn_state, torch.Tensor):
            if hasattr(rnn, 'get_column_cosine_similarities'):
                self.col_sim_buffer.append(rnn.get_column_cosine_similarities(rnn_state))

        if isinstance(rnn_state, torch.Tensor) and rnn_state.ndim == 4:
            self.hidden_norm_buffer.append(
                compute_hidden_norms(rnn_state, self.n_layers, self.n_columns)
            )

        if has_hgrn and hasattr(rnn, 'get_hgrn_betas'):
            self.beta_buffer.append(rnn.get_hgrn_betas())
            self.beta_step_buffer.append(step)

    def update_div(self, step: int, div_mean: dict) -> None:
        for key, val in div_mean.items():
            self.div_history[f'div/{key}'].append(float(val) if not isinstance(val, float) else val)
        self.div_step_history.append(step)

    def update_grad_norms(self, step: int, rnn) -> dict[str, float]:
        grad_norms = compute_grad_norm_per_layer(rnn, self.n_layers)
        self.grad_norm_buffer.append(grad_norms)
        self.grad_step_buffer.append(step)
        return grad_norms

    def flush(self, logger, step: int, *, has_hgrn: bool, has_reservoir: bool, reservoir_sr_info: dict) -> None:
        try:
            self.attn_vis.log(logger, step=step)
        except Exception as e:
            print(f'[VIS] AttnFlow: {e}')
        try:
            self.cka_vis.log(logger, step=step)
        except Exception as e:
            print(f'[VIS] CKA: {e}')

        if self.gate_buffer:
            try:
                fig = plot_gate_saturation(self.gate_buffer, self.n_layers, step)
                log_figure(logger, fig, 'vis/gate_saturation', step)
                arr = np.array(self.gate_buffer)
                for li in range(min(arr.shape[1], self.n_layers)):
                    logger.track(float(arr[:, li].mean()), name=f"attn_gate/L{li}", step=step)
            except Exception as e:
                print(f'[VIS] gate sat: {e}')
            self.gate_buffer.clear()

        if self.col_sim_buffer:
            try:
                avg = defaultdict(list)
                for d in self.col_sim_buffer:
                    for k, v in d.items():
                        avg[k].append(v)
                avg_scalar = {k: float(np.mean(vs)) for k, vs in avg.items()}
                fig = plot_col_similarity_heatmap(avg_scalar, self.n_layers, self.n_columns, step)
                log_figure(logger, fig, 'vis/col_similarity_heatmap', step)
                for k, v in avg_scalar.items():
                    logger.track(v, name=k, step=step)
            except Exception as e:
                print(f'[VIS] col sim: {e}')
            self.col_sim_buffer.clear()

        if self.hidden_norm_buffer:
            try:
                fig = plot_hidden_norm_heatmap(self.hidden_norm_buffer, self.n_layers, self.n_columns, step)
                log_figure(logger, fig, 'vis/hidden_norm_heatmap', step)
            except Exception as e:
                print(f'[VIS] hidden norm: {e}')
            self.hidden_norm_buffer.clear()

        if has_hgrn and self.beta_buffer and len(self.beta_buffer) > 1:
            try:
                fig = plot_beta_dynamics(self.beta_buffer, self.beta_step_buffer, self.n_layers, self.n_columns)
                log_figure(logger, fig, 'vis/beta_dynamics', step)
                for k, v in self.beta_buffer[-1].items():
                    logger.track(v, name=k, step=step)
            except Exception as e:
                print(f'[VIS] beta: {e}')

        if self.div_step_history:
            try:
                fig = plot_diversity_components(self.div_history, self.div_step_history)
                log_figure(logger, fig, 'vis/diversity_loss_curves', step)
            except Exception as e:
                print(f'[VIS] diversity: {e}')

        if len(self.grad_norm_buffer) > 1:
            try:
                fig = plot_grad_norm_per_layer(self.grad_norm_buffer, self.grad_step_buffer, self.n_layers)
                log_figure(logger, fig, 'vis/grad_norm_per_layer', step)
            except Exception as e:
                print(f'[VIS] grad norm: {e}')

        if has_reservoir:
            for k, v in reservoir_sr_info.items():
                try:
                    logger.track(v, name=k, step=step)
                except Exception:
                    pass

        self.next_step += self.interval
