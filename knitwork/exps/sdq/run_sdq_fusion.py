# run_sdq3.py v2
from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
from datetime import datetime
from collections import defaultdict
from typing import Optional
from typing import Any
from knitwork.common.config import extracted
from knitwork.common.curriculum import CurriculumScheduler
from knitwork.common.dynamic_param import DynamicParameter
from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging import create_logger
from knitwork.common.scheduler import Scheduler
from knitwork.common.tracker import Tracker
from knitwork.common.utils import (
    CE_ignore_index, FpsCounter, flatten_dict,
    format_readable_num, get_device, get_dtype,
    to_numpy, to_torch,
)
from knitwork.gens.sdq import StoreDistractQueryGenerator
from knitwork.visualization.attn_flow import AttnFlowVisualizer
from knitwork.visualization.cka import CKAVisualizer

VIS_INTERVAL = 10_000_000

class NanHookManager:
    def __init__(self):
        self.hooks = []
        self.first_nan: dict | None = None   # первый пойманный NaN

    def register(self, model: nn.Module):
        for name, module in model.named_modules():
            h = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(h)

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            if self.first_nan is not None:
                return   # уже нашли — не спамим
            # Проверяем все тензоры выхода
            outs = [out] if isinstance(out, torch.Tensor) else (
                list(out) if isinstance(out, (tuple, list)) else []
            )
            for i, t in enumerate(outs):
                if not isinstance(t, torch.Tensor):
                    continue
                if not torch.isfinite(t).all():
                    self.first_nan = {
                        'module': name,
                        'type':   type(module).__name__,
                        'out_idx': i,
                        'shape':  tuple(t.shape),
                        'n_nan':  int((~torch.isfinite(t)).sum().item()),
                        'min':    float(t[torch.isfinite(t)].min()) if torch.isfinite(t).any() else float('nan'),
                        'max':    float(t[torch.isfinite(t)].max()) if torch.isfinite(t).any() else float('nan'),
                    }
                    # Проверяем входы этого модуля
                    bad_inputs = []
                    inps = [inp] if isinstance(inp, torch.Tensor) else (
                        list(inp) if isinstance(inp, (tuple, list)) else []
                    )
                    for j, ti in enumerate(inps):
                        if isinstance(ti, torch.Tensor) and not torch.isfinite(ti).all():
                            bad_inputs.append(j)
                    self.first_nan['bad_inputs'] = bad_inputs
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ПРОВЕРКА ПАРАМЕТРОВ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════

def check_params(model: nn.Module) -> list[str]:
    """Ищем NaN/Inf в весах модели."""
    bad = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            n_bad = int((~torch.isfinite(p)).sum().item())
            bad.append(f"  PARAM NaN/Inf: {name} | shape={tuple(p.shape)} | n_bad={n_bad}")
    return bad


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ПРОВЕРКА ГРАДИЕНТОВ — подробно по каждому параметру
# ═══════════════════════════════════════════════════════════════════════════════

def check_grads(model: nn.Module) -> list[str]:
    bad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if not torch.isfinite(g).all():
            n_bad = int((~torch.isfinite(g)).sum().item())
            gmax  = float(g[torch.isfinite(g)].abs().max()) if torch.isfinite(g).any() else float('nan')
            bad.append(
                f"  GRAD NaN/Inf: {name:<60} "
                f"shape={str(tuple(p.shape)):<20} "
                f"n_bad={n_bad:<6} "
                f"max_finite={gmax:.4f}"
            )
    return bad


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ПОШАГОВАЯ ДИАГНОСТИКА ЧЕРЕЗ ВЕСЬ GRID STEP
# ═══════════════════════════════════════════════════════════════════════════════

def debug_grid_step(rnn, x: torch.Tensor, h: torch.Tensor):
    """
    Повторяет логику _grid_step вручную с проверкой на каждом шаге.
    """
    print("\n── debug_grid_step ──────────────────────────────────────────")
    n_t = rnn.n_trainable_cols
    n_r = rnn.n_reservoir_cols

    def check(tag: str, t: torch.Tensor):
        ok = torch.isfinite(t).all()
        mn = float(t.min()) if ok else float('nan')
        mx = float(t.max()) if ok else float('nan')
        print(f"  {'✓' if ok else '✗ NaN/Inf':<12} {tag:<55} "
              f"shape={str(tuple(t.shape)):<25} min={mn:+.4f} max={mx:+.4f}")
        return ok

    check("input x", x)
    check("input h", h)

    # Embedding / input проекции
    x_cols = torch.stack([proj(x) for proj in rnn.col_input_projs], dim=1)
    check("x_cols (after col_input_projs)", x_cols)

    for li in range(rnn.n_layers):
        print(f"\n  ── Layer {li} ──")
        hl = h[li]
        check(f"  h[{li}] (input to layer)", hl)

        if li == 0:
            x_t_batch = x_cols
        else:
            x_t_batch = hl.permute(1, 0, 2)
        check(f"  x_t_batch L{li}", x_t_batch)

        h_t_in = hl[:n_t].permute(1, 0, 2)
        x_t_in = x_t_batch[:, :n_t, :]
        check(f"  h_t_in  L{li}", h_t_in)
        check(f"  x_t_in  L{li}", x_t_in)

        # Trainable: пошагово
        cell = rnn.trainable_cells[li]
        x_t = x_t_in.permute(1, 2, 0)
        h_t = h_t_in.permute(1, 2, 0)

        def gx(W, b, tag):
            out = torch.bmm(W, x_t).permute(0, 2, 1)
            if b is not None:
                out = out + b.unsqueeze(1)
            check(f"  gx({tag}) L{li}", out)
            return out

        def gh(U, h_src, tag):
            out = torch.bmm(U, h_src).permute(0, 2, 1)
            check(f"  gh({tag}) L{li}", out)
            return out

        o_raw = gx(cell.W_o, cell.b_o, "W_o") + gh(cell.U_o, h_t, "U_o")
        check(f"  o_raw L{li}", o_raw)
        o_t = torch.sigmoid(o_raw)
        check(f"  o_t   L{li}", o_t)

        h_p = h_t_in.permute(1, 0, 2)
        oh = (o_t * h_p).permute(0, 2, 1)
        check(f"  o_t*h L{li}", oh)

        c_raw = gx(cell.W_c, cell.b_c, "W_c") + gh(cell.U_c, oh, "U_c")
        check(f"  c_raw L{li}", c_raw)

        if cell.ln_c is not None:
            c_normed = torch.stack([cell.ln_c[i](c_raw[i]) for i in range(cell.n_cols)], dim=0)
            check(f"  c_normed (after LN) L{li}", c_normed)
        else:
            c_normed = c_raw

        c_t = torch.tanh(c_normed)
        check(f"  c_t   L{li}", c_t)

        f_raw = gx(cell.W_f, cell.b_f, "W_f") + gh(cell.U_f, h_t, "U_f")
        check(f"  f_raw L{li}", f_raw)
        betas = cell.betas.view(cell.n_cols, 1, 1)
        lam_t = torch.sigmoid(f_raw) * (1.0 - betas) + betas
        check(f"  lam_t L{li}", lam_t)

        h_t_new_raw = lam_t * h_p + (1.0 - lam_t) * c_t
        check(f"  h_t_new L{li}", h_t_new_raw)
        h_t_new = h_t_new_raw.permute(1, 0, 2)

        if n_r > 0:
            h_r_in = hl[n_t:].permute(1, 0, 2)
            x_r_in = x_t_batch[:, n_t:, :]
            check(f"  h_r_in L{li}", h_r_in)
            check(f"  x_r_in L{li}", x_r_in)
            h_r_new = rnn._batched_reservoir_forward(li, x_r_in, h_r_in)
            check(f"  h_r_new L{li}", h_r_new)

            if rnn.cross_attns is not None:
                h_t_new = rnn.cross_attns[li](h_t_new, h_r_new)
                check(f"  h_t_new after CrossAttn L{li}", h_t_new)

            h_all = torch.cat([h_t_new, h_r_new], dim=1)
        else:
            h_all = h_t_new

        h_all_seq = h_all.permute(1, 0, 2)
        check(f"  h_all_seq L{li}", h_all_seq)

        msg, _ = rnn.attn[li](h_all_seq, return_weights=False)
        check(f"  msg (after MHA) L{li}", msg)

        msg_t   = msg[:n_t]
        h_t_seq = h_t_new.permute(1, 0, 2)
        gate_in = torch.cat([h_t_seq, msg_t], dim=-1)
        check(f"  gate_in L{li}", gate_in)
        gate_logit = rnn.attn_gates[li](gate_in)
        check(f"  gate_logit L{li}", gate_logit)
        g = torch.sigmoid(gate_logit)
        h_t_merged = (1.0 - g) * h_t_seq + g * msg_t
        check(f"  h_t_merged L{li}", h_t_merged)

    print("── end debug_grid_step ──────────────────────────────────────\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ГЛАВНАЯ ФУНКЦИЯ ДИАГНОСТИКИ — вставить в main() сразу после создания rnn
# ═══════════════════════════════════════════════════════════════════════════════

def run_nan_diagnostics(rnn, gen, device, loss_fn):
    print("\n" + "═" * 70)
    print("NaN DIAGNOSTICS START")
    print("═" * 70)

    # 5.1 Проверяем веса модели ДО обучения
    print("\n[1] Параметры модели:")
    bad_params = check_params(rnn)
    if bad_params:
        for s in bad_params:
            print(s)
    else:
        print("  ✓ Все параметры конечны")

    # 5.2 Синтетический входной батч
    bsz = 4
    # Случайные токены в диапазоне vocab
    fake_tokens = torch.randint(0, gen.n_tokens, (bsz, 1), device=device)
    fake_targets = torch.randint(0, gen.V, (bsz,), device=device)
    h = rnn.init_state(bsz)

    print(f"\n[2] Тестовый батч: tokens={tuple(fake_tokens.shape)} targets={tuple(fake_targets.shape)}")
    print(f"    dtype={h.dtype} device={device}")

    # 5.3 Forward с хуками
    print("\n[3] Forward pass с хуками на NaN:")
    hook_mgr = NanHookManager()
    hook_mgr.register(rnn)
    try:
        with torch.autograd.set_detect_anomaly(True):
            y, h_new, extras = rnn(fake_tokens, h, return_attn=True)
    except Exception as e:
        print(f"  ✗ ИСКЛЮЧЕНИЕ в forward: {e}")
        hook_mgr.remove()
        return
    hook_mgr.remove()

    if hook_mgr.first_nan:
        print(f"  ✗ Первый NaN/Inf в модуле:")
        for k, v in hook_mgr.first_nan.items():
            print(f"      {k}: {v}")
    else:
        print(f"  ✓ Forward pass чист")

    print(f"\n[4] Выход модели y:")
    ok_y = torch.isfinite(y).all()
    print(f"  {'✓' if ok_y else '✗ NaN/Inf'} shape={tuple(y.shape)} "
          f"min={float(y.min()):.4f} max={float(y.max()):.4f}")

    # 5.4 Loss
    print(f"\n[5] Loss:")
    try:
        loss = loss_fn(y, fake_targets)
        ok_loss = torch.isfinite(loss)
        print(f"  {'✓' if ok_loss else '✗ NaN/Inf'} loss={float(loss):.6f}")
    except Exception as e:
        print(f"  ✗ ИСКЛЮЧЕНИЕ в loss: {e}")
        return

    # 5.5 Diversity loss
    if hasattr(rnn, 'compute_diversity_loss') and extras:
        print(f"\n[6] Diversity loss:")
        try:
            div = rnn.compute_diversity_loss(extras)
            for k, v in div.items():
                ok = torch.isfinite(v)
                print(f"  {'✓' if ok else '✗ NaN/Inf'} {k}: {float(v):.6f}")
            total_loss = loss + div.get('total', torch.tensor(0.0))
        except Exception as e:
            print(f"  ✗ ИСКЛЮЧЕНИЕ в diversity_loss: {e}")
            total_loss = loss
    else:
        total_loss = loss

    # 5.6 Backward
    print(f"\n[7] Backward pass:")
    rnn.zero_grad()
    try:
        total_loss.backward()
        print(f"  ✓ backward завершён без исключений")
    except Exception as e:
        print(f"  ✗ ИСКЛЮЧЕНИЕ в backward: {e}")
        return

    # 5.7 Градиенты — подробно
    print(f"\n[8] Градиенты (только плохие):")
    bad_grads = check_grads(rnn)
    if bad_grads:
        for s in bad_grads:
            print(s)
    else:
        print("  ✓ Все градиенты конечны")

    # 5.8 Топ-10 наибольших градиентов (даже если finite)
    print(f"\n[9] Топ-10 параметров по норме градиента:")
    grad_norms = []
    for name, p in rnn.named_parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            grad_norms.append((name, float(p.grad.norm().item())))
    grad_norms.sort(key=lambda x: x[1], reverse=True)
    for name, norm in grad_norms[:10]:
        print(f"  {norm:10.4f}  {name}")

    # 5.9 Пошаговая диагностика grid step
    print(f"\n[10] Пошаговая диагностика _grid_step:")
    with torch.no_grad():
        x_emb = rnn.embedding(fake_tokens.view(-1))
        debug_grid_step(rnn, x_emb, rnn.init_state(bsz))

    print("\n" + "═" * 70)
    print("NaN DIAGNOSTICS END")
    print("═" * 70 + "\n")



# ═══════════════════════════════════════════════════════════════════════════════
# AIM-совместимое логирование изображений
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    plt.close(fig)
    return arr[..., :3]


def _log_figure(logger, fig: plt.Figure, name: str, step: int) -> None:
    """
    FIX v2: правильное логирование изображений в AIM.

    AIM не имеет метода track_image. Правильный способ:
    - создать aim.Image из numpy array
    - передать в logger.track(aim.Image(...), name=name, step=step)
    """
    if fig is None:
        return
    try:
        arr = _fig_to_numpy(fig)  # закрывает fig внутри

        # Попытка 1: AIM logger (основной случай)
        try:
            import aim
            aim_image = aim.Image(arr)
            logger.track(aim_image, name=name, step=step)
            return
        except (ImportError, AttributeError):
            pass

        # Попытка 2: AIM через experiment напрямую
        try:
            if hasattr(logger, '_run') and logger._run is not None:
                import aim
                aim_image = aim.Image(arr)
                logger._run.track(aim_image, name=name, step=step)
                return
        except (AttributeError, Exception):
            pass

        # Попытка 3: wandb
        try:
            import wandb
            if hasattr(logger, 'log'):
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(arr)
                logger.log({name: wandb.Image(pil_img), 'global_step': step})
                return
        except (ImportError, AttributeError):
            pass

        # Попытка 4: track как numpy (некоторые кастомные логгеры)
        if hasattr(logger, 'track'):
            try:
                logger.track(arr, name=name, step=step)
            except Exception:
                pass

    except Exception as e:
        print(f'[VIS] Ошибка логирования {name}: {e}')


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_col_similarity_heatmap(
    col_sim: dict[str, float],
    n_layers: int,
    n_columns: int,
    step: int,
) -> plt.Figure:
    n_pairs = n_columns * (n_columns - 1) // 2
    pairs = [(ci, cj) for ci in range(n_columns) for cj in range(ci + 1, n_columns)]
    pair_labels = [f"C{ci}-C{cj}" for ci, cj in pairs]
    data = np.zeros((n_layers, n_pairs))

    for li in range(n_layers):
        for pi, (ci, cj) in enumerate(pairs):
            key = f"col_sim/L{li}_C{ci}_C{cj}"
            data[li, pi] = col_sim.get(key, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, n_pairs * 1.2), max(4, n_layers * 0.8)))
    im = ax.imshow(data, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(n_pairs))
    ax.set_xticklabels(pair_labels, rotation=45, ha='right')
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {li}" for li in range(n_layers)])
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    ax.set_title(f'Column Cosine Similarity Heatmap | step={step:,}')
    plt.tight_layout()
    return fig


def plot_beta_dynamics(
    beta_history: list[dict[str, float]],
    step_history: list[int],
    n_layers: int,
    n_columns: int,
) -> plt.Figure:
    fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 3), sharey=True)
    if n_layers == 1:
        axes = [axes]
    cmap = plt.get_cmap('cool', n_columns)

    for li, ax in enumerate(axes):
        for ci in range(n_columns):
            key = f"hgrn/beta/L{li}_C{ci}"
            vals = [h.get(key, float('nan')) for h in beta_history]
            if any(not np.isnan(v) for v in vals):
                ax.plot(step_history, vals, color=cmap(ci), label=f"C{ci}", linewidth=1.5)
        ax.set_title(f'Layer {li}')
        ax.set_xlabel('Step')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        if li == 0:
            ax.set_ylabel('β (forget lower bound)')
            ax.legend(fontsize=7, loc='upper left')

    fig.suptitle('HGRN β Dynamics', fontsize=12)
    plt.tight_layout()
    return fig


def plot_diversity_components(
    div_history: dict[str, list[float]],
    step_history: list[int],
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
    ax1.set_ylabel('Loss value')
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
    norm_history: list[dict[str, float]],
    n_layers: int,
    n_columns: int,
    step: int,
) -> plt.Figure:
    if not norm_history:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    data = np.zeros((n_layers, n_columns))
    for li in range(n_layers):
        for ci in range(n_columns):
            key = f"hidden_norm/L{li}_C{ci}"
            vals = [h.get(key, float('nan')) for h in norm_history]
            valid = [v for v in vals if not np.isnan(v)]
            data[li, ci] = np.mean(valid) if valid else 0.0

    fig, ax = plt.subplots(figsize=(max(4, n_columns * 0.9), max(3, n_layers * 0.7)))
    im = ax.imshow(data, cmap='viridis', aspect='auto')
    ax.set_xticks(range(n_columns))
    ax.set_xticklabels([f"Col {ci}" for ci in range(n_columns)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {li}" for li in range(n_layers)])
    plt.colorbar(im, ax=ax, label='Mean L2 Norm')
    ax.set_title(f'Hidden State Norm Heatmap | step={step:,}')
    plt.tight_layout()
    return fig


def plot_gate_saturation(
    gate_buffer: list[list[float]],
    n_layers: int,
    step: int,
) -> plt.Figure:
    fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 3), sharey=True)
    if n_layers == 1:
        axes = [axes]

    arr = np.array(gate_buffer) if gate_buffer else np.zeros((1, n_layers))
    for li, ax in enumerate(axes):
        if li < arr.shape[1]:
            vals = arr[:, li]
            ax.hist(vals, bins=20, range=(0, 1), color='steelblue', edgecolor='white')
        ax.set_title(f'Layer {li}')
        ax.set_xlabel('Gate value')
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color='red', linestyle='--', linewidth=1, label='ideal')
        ax.grid(True, alpha=0.3)
        if li == 0:
            ax.set_ylabel('Count')
            ax.legend(fontsize=7)

    fig.suptitle(f'Attention Gate Saturation | step={step:,}', fontsize=11)
    plt.tight_layout()
    return fig


def plot_grad_norm_per_layer(
    grad_norm_history: list[dict[str, float]],
    step_history: list[int],
    n_layers: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap('plasma', n_layers)

    for li in range(n_layers):
        key = f"grad_norm/layer_{li}"
        vals = [h.get(key, float('nan')) for h in grad_norm_history]
        valid_steps = [s for s, v in zip(step_history, vals) if not np.isnan(v)]
        valid_vals = [v for v in vals if not np.isnan(v)]
        if valid_vals:
            ax.plot(valid_steps, valid_vals, color=cmap(li), label=f"L{li}", linewidth=1.5)

    ax.set_title('Gradient Norm per Layer')
    ax.set_xlabel('Step')
    ax.set_ylabel('||∇||')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    # FIX: log scale только если есть ненулевые значения
    try:
        ax.set_yscale('log')
    except Exception:
        pass
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_grad_norm_per_layer(model, n_layers: int) -> dict[str, float]:
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


def _compute_hidden_norms(
    h: torch.Tensor,
    n_layers: int,
    n_columns: int,
) -> dict[str, float]:
    result = {}
    with torch.no_grad():
        for li in range(n_layers):
            for ci in range(n_columns):
                nrm = h[li, ci].float().norm(dim=-1).mean().item()
                result[f"hidden_norm/L{li}_C{ci}"] = nrm
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(config):
    run_name = (
        config.get('name', None)
        or config.get('log', {}).get('name', None)
        or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    config.setdefault('log', {})['name'] = run_name
    print(f'Run name: {run_name}')

    VIS_ENABLED = config.get('visualize', True)

    rng    = np.random.default_rng(config['seed'])
    device = get_device(config.get('device', None))
    dtype  = get_dtype(config.get('dtype', None))
    n_envs = config['n_envs']

    gen_cfg = config['gens'][config['gen']]
    gen = StoreDistractQueryGenerator(
        **gen_cfg, n_envs=n_envs,
        seed=rng.integers(1_000_000),
        ignore_index=CE_ignore_index,
    )

    rnn_type = config['model']
    rnn_cfg  = config['models'][rnn_type]
    is_fusion = (rnn_type == 'grnn_fusion')

    match rnn_type:
        case 'grnn_fusion':
            from knitwork.models.grnn_fusion import build_fusion_from_config
            rnn = build_fusion_from_config(rnn_cfg, gen.n_tokens, gen.V)
        case 'rnn':
            from knitwork.models.gru import GruBaseline
            rnn = GruBaseline(**rnn_cfg, input_size=gen.n_tokens, output_size=gen.V)
        case 'grnn':
            from knitwork.models.grnn import GridRnn
            rnn = GridRnn(**rnn_cfg, input_size=gen.n_tokens, output_size=gen.V)
        case 'grnn_loss':
            from knitwork.models.grnn_loss import GridRnnLoss
            rnn = GridRnnLoss(**rnn_cfg, input_size=gen.n_tokens, output_size=gen.V)
        case 'grnn_reservoir':
            from knitwork.models.grnn_reservoir import GridRnnReservoir
            rnn = GridRnnReservoir(**rnn_cfg, input_size=gen.n_tokens, output_size=gen.V)
        case 'grnn_hgrn':
            from knitwork.models.hgrn_grnn import HGRN_GridRnn
            rnn = HGRN_GridRnn(**rnn_cfg, input_size=gen.n_tokens, output_size=gen.V)
        case _:
            raise ValueError(f"Неизвестный тип модели: {rnn_type}")

    rnn = rnn.to(device=device, dtype=dtype)
    from debug_nan import run_nan_diagnostics
    run_nan_diagnostics(rnn, gen, device, nn.CrossEntropyLoss(ignore_index=CE_ignore_index))
    has_diversity  = is_fusion and rnn.cfg.diversity_loss.enabled
    has_hgrn       = is_fusion and rnn.cfg.hgrn.enabled
    has_reservoir  = is_fusion and rnn.cfg.reservoir.enabled

    print(f'[FUSION]    {"ON" if is_fusion else "OFF"}')
    print(f'[HGRN]      {"ON" if has_hgrn else "OFF"}')
    print(f'[RESERVOIR] {"ON" if has_reservoir else "OFF"}')
    print(f'[DIV_LOSS]  {"ON" if has_diversity else "OFF"}')

    n_layers  = rnn.n_layers
    n_columns = rnn.n_columns

    attn_vis = AttnFlowVisualizer(n_layers=n_layers, n_columns=n_columns, buffer_size=100)
    cka_vis  = CKAVisualizer(n_layers=n_layers, n_columns=n_columns, buffer_size=50)
    next_vis_step = VIS_INTERVAL

    gate_buffer:        list[list[float]] = []
    col_sim_buffer:     list[dict]        = []
    hidden_norm_buffer: list[dict]        = []
    beta_buffer:        list[dict]        = []
    beta_step_buffer:   list[int]         = []
    div_history:        dict[str, list]   = defaultdict(list)
    div_step_history:   list[int]         = []
    grad_norm_buffer:   list[dict]        = []
    grad_step_buffer:   list[int]         = []

    reservoir_sr_info: dict[str, float] = {}
    if has_reservoir:
        reservoir_sr_info = rnn.get_reservoir_spectral_radii()
        print(f'Reservoir SR: {reservoir_sr_info}')

    lr_cfg = config['lr']
    lr     = lr_cfg['val']

    wm_lr_cfg, wm_lr_schedule = extracted(lr_cfg['warmup'], 'schedule')
    dc_lr_cfg, dc_lr_schedule = extracted(lr_cfg['decay'],  'schedule')
    wm_lr = DynamicParameter(
        val=1e-5 * lr, tar=lr, **wm_lr_cfg,
        scheduler=Scheduler(wm_lr_schedule),
    )
    dc_lr = DynamicParameter(val=lr, **dc_lr_cfg, scheduler=Scheduler(dc_lr_schedule))

    def get_lr() -> float:
        return wm_lr.val if not wm_lr.scheduler.is_infinite else dc_lr.val

    def step_lr() -> bool:
        return wm_lr.step() if not wm_lr.scheduler.is_infinite else dc_lr.step()

    optim   = torch.optim.RMSprop(rnn.parameters(), lr=get_lr())
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)

    rollout_len = config['rollout_len']
    batch_size  = gen.n_envs * rollout_len
    n_steps     = int(config['n_steps'])
    step        = 0

    log_stats_schedule   = Scheduler(int(config['log']['schedule']))
    print_stats_schedule = Scheduler(int(config['log']['print_schedule']))

    curriculum_cfg, curriculum_schedule = extracted(config['curriculum'], 'schedule')
    curriculum_step_schedule = CurriculumScheduler(
        **curriculum_cfg, scheduler=Scheduler(curriculum_schedule),
    )

    logger = create_logger(config)
    if logger is not None and hasattr(logger, 'name'):
        logger.name = run_name

    stats       = Tracker(lr=2e-4)
    fps_counter = FpsCounter()

    rnn_state     = None
    batch_y:       list[torch.Tensor] = []
    batch_y_gt:    list[torch.Tensor] = []
    batch_sq_gaps: list[torch.Tensor] = []
    batch_div:     list[dict]          = []

    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}

        rnn_state = rnn.reset_state(rnn_state, obs['reset_mask'])
        x = obs['tokens'].view(-1, 1)

        capture = VIS_ENABLED and (step >= next_vis_step - gen.n_envs)

        if capture or has_diversity:
            result = rnn(x, rnn_state, return_attn=True)
            y, rnn_state, extras = result[0], result[1], result[2]
        else:
            result = rnn(x, rnn_state)
            y, rnn_state = result[0], result[1]
            extras = {}

        div_losses: Optional[dict] = None
        if has_diversity and extras:
            div_losses = rnn.compute_diversity_loss(extras)
            batch_div.append({k: v.detach() for k, v in div_losses.items()})

        if capture and extras:
            attn_weights = extras.get('attn_weights', [])
            if attn_weights:
                attn_vis.update(attn_weights)

            h_for_cka = rnn_state[0] if isinstance(rnn_state, tuple) else rnn_state
            cka_vis.update(h_for_cka)

            # FIX: берём sigmoid(gate_logits) для визуализации насыщения
            gate_probs = [
                torch.sigmoid(g).detach().float().mean().item()
                for g in extras.get('gate_logits', [])
                if g is not None
            ]
            if gate_probs:
                gate_buffer.append(gate_probs)

            if is_fusion and isinstance(rnn_state, torch.Tensor):
                col_sim = rnn.get_column_cosine_similarities(rnn_state)
                col_sim_buffer.append(col_sim)

            if isinstance(rnn_state, torch.Tensor) and rnn_state.ndim == 4:
                hn = _compute_hidden_norms(rnn_state, n_layers, n_columns)
                hidden_norm_buffer.append(hn)

            if has_hgrn:
                betas = rnn.get_hgrn_betas()
                beta_buffer.append(betas)
                beta_step_buffer.append(step)

        y_gt = obs['targets']
        sq_gaps = obs['sq_gaps']

        # FIX: вычисляем маску активных токенов на этом шаге
        m_active_step = y_gt != CE_ignore_index

        batch_y.append(y)
        batch_y_gt.append(y_gt)
        batch_sq_gaps.append(sq_gaps)

        # Визуализация
        if step >= next_vis_step and logger is not None and VIS_ENABLED:
            _run_visualizations(
                logger=logger, step=step,
                attn_vis=attn_vis, cka_vis=cka_vis,
                gate_buffer=gate_buffer,
                col_sim_buffer=col_sim_buffer,
                hidden_norm_buffer=hidden_norm_buffer,
                beta_buffer=beta_buffer, beta_step_buffer=beta_step_buffer,
                div_history=div_history, div_step_history=div_step_history,
                grad_norm_buffer=grad_norm_buffer, grad_step_buffer=grad_step_buffer,
                n_layers=n_layers, n_columns=n_columns,
                has_hgrn=has_hgrn, has_reservoir=has_reservoir,
                reservoir_sr_info=reservoir_sr_info,
            )
            gate_buffer.clear()
            col_sim_buffer.clear()
            hidden_norm_buffer.clear()
            next_vis_step += VIS_INTERVAL

        step += gen.n_envs

        if step % batch_size == 0:
            y_cat     = torch.cat(batch_y,      dim=0)
            y_gt_cat  = torch.cat(batch_y_gt,   dim=0)
            # FIX: sq_gaps уже отфильтрован генератором — НЕ применяем m_active!
            sq_gaps_all = torch.cat(batch_sq_gaps, dim=0).float()

            ce_loss = loss_fn(y_cat, y_gt_cat)

            div_mean: dict[str, torch.Tensor] = {}
            total_div = torch.tensor(0.0, device=device, dtype=y_cat.dtype)
            if batch_div:
                for key in batch_div[0]:
                    div_mean[key] = torch.stack([d[key] for d in batch_div]).mean()
                total_div = div_mean.get('total', total_div)
                for key, val in div_mean.items():
                    div_history[f'div/{key}'].append(val.item())
                div_step_history.append(step)

            total_loss = ce_loss + total_div

            with torch.no_grad():
                m_active    = y_gt_cat != CE_ignore_index
                y_active    = y_cat[m_active]
                y_gt_active = y_gt_cat[m_active]
                
                acc = (y_active.argmax(dim=-1) == y_gt_active).float()

                # FIX: генератор возвращает sq_gaps уже в соответствии с active tokens
                # sq_gaps_all.shape[0] == acc.shape[0] (оба равны числу активных токенов)
                # УБИРАЕМ фильтрацию [m_active] — она уже применена генератором!
                sq_gaps_active = sq_gaps_all
                
                # Защитная проверка (можно убрать после тестирования)
                if sq_gaps_active.shape[0] != acc.shape[0]:
                    print(
                        f"⚠ WARNING: sq_gaps {sq_gaps_active.shape[0]} != "
                        f"acc {acc.shape[0]} — генератор вернул некорректный размер"
                    )
                    # Fallback: если генератор ошибся и вернул все токены
                    if sq_gaps_active.shape[0] == m_active.shape[0]:
                        sq_gaps_active = sq_gaps_active[m_active]

                # Маски для store/query/distract
                mask_store    = sq_gaps_active < -1.0
                mask_query    = sq_gaps_active > 0.0
                mask_distract = (~mask_store) & (~mask_query)
                mask_misses   = sq_gaps_active < 0.0

                def safe_mean(t, mask):
                    return t[mask].mean() if mask.any() else torch.tensor(float('nan'))

                acc_store    = safe_mean(acc, mask_store)
                acc_query    = safe_mean(acc, mask_query)
                acc_distract = safe_mean(acc, mask_distract)
                acc_miss     = safe_mean(acc, mask_misses)
                acc_non_miss = safe_mean(acc, ~mask_misses)

                sq_non_miss = sq_gaps_active[~mask_misses]
                if sq_non_miss.numel() > 0:
                    acc_up_half = acc[sq_gaps_active > sq_non_miss.mean()].mean()
                else:
                    acc_up_half = torch.tensor(float('nan'))

                acc_mean = acc.mean()

            # Backward
            optim.zero_grad()
            total_loss.backward()

            grad_norm_layer = _compute_grad_norm_per_layer(rnn, n_layers)
            grad_norm_buffer.append(grad_norm_layer)
            grad_step_buffer.append(step)

            grad_norm = nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('⚠ Nan/Inf grad — шаг пропущен')

            if step_lr():
                optim.param_groups[0]['lr'] = get_lr()

            stat_dict = {
                "Loss":         to_numpy(ce_loss,      copy=False),
                "Acc":          to_numpy(acc_mean,     copy=False),
                "Acc/store":    to_numpy(acc_store,    copy=False),
                "Acc/query":    to_numpy(acc_query,    copy=False),
                "Acc/distract": to_numpy(acc_distract, copy=False),
                "Acc-":         to_numpy(acc_miss,     copy=False),
                "Acc+":         to_numpy(acc_non_miss, copy=False),
                "Acc++":        to_numpy(acc_up_half,  copy=False),
                "|Grad|":       to_numpy(grad_norm,    copy=False),
                "LR":           get_lr(),
            }

            if div_mean:
                for key, val in div_mean.items():
                    stat_dict[f"div/{key}"] = val.item()

            if has_hgrn:
                try:
                    betas = rnn.get_hgrn_betas()
                    for li in range(n_layers):
                        lb = [v for k, v in betas.items() if f"L{li}_" in k]
                        if lb:
                            stat_dict[f"hgrn/beta_mean/L{li}"] = float(np.mean(lb))
                except Exception:
                    pass

            for k, v in grad_norm_layer.items():
                stat_dict[k] = v

            # NEW: reservoir utilization metrics
            if has_reservoir and isinstance(rnn_state, torch.Tensor):
                try:
                    res_util = rnn.get_reservoir_utilization(rnn_state)
                    for k, v in res_util.items():
                        stat_dict[k] = v
                except Exception:
                    pass

            stats.put(stat_dict)

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear()
            batch_y_gt.clear()
            batch_sq_gaps.clear()
            batch_div.clear()

        if curriculum_step_schedule.tick(metrics=stats, n_steps=gen.n_envs):
            K = 10
            dT, dp_store, dp_query = 1.0 / K, -0.0014 / K, -0.0005 / K
            gen.set_metaparams(
                T       = gen.T + dT,
                p_store = max(gen.p_store + dp_store, 0.10),
                p_query = max(gen.p_query + dp_query, 0.25),
            )

        if print_stats_schedule.tick(gen.n_envs):
            metrics = {"global_step": step} | stats.get()
            fps     = fps_counter.fps(n_iters=step, start=True)
            print(
                f'[{format_readable_num(step)}/'
                f'{format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)}fps |'
                f' L:{metrics["Loss"]:.3f}'
                f' A:{metrics["Acc"]:.3f}'
                f' Aq:{metrics.get("Acc/query", float("nan")):.3f}'
                f' As:{metrics.get("Acc/store", float("nan")):.3f}'
                f' Ad:{metrics.get("Acc/distract", float("nan")):.3f}'
                f' A-:{metrics["Acc-"]:.3f}'
                f' A++:{metrics["Acc++"]:.3f}'
            )

        if log_stats_schedule.tick(gen.n_envs) and logger is not None:
            fps     = fps_counter.fps(n_iters=step, start=True)
            metrics = {
                "global_step":   step,
                "fps":           fps,
                "curr_step":     curriculum_step_schedule.cnt_accepted,
                "curr_schedule": curriculum_step_schedule.scheduler.schedule,
            } | stats.get()
            metrics['gen'] = gen.get_stats()
            for k, v in reservoir_sr_info.items():
                metrics[k] = v
            logger.track(flatten_dict(metrics))

    fps = fps_counter.fps(n_iters=step)
    print(f'Done. {format_readable_num(fps)} fps')


# ═══════════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def _run_visualizations(
    *,
    logger,
    step: int,
    attn_vis,
    cka_vis,
    gate_buffer: list,
    col_sim_buffer: list,
    hidden_norm_buffer: list,
    beta_buffer: list,
    beta_step_buffer: list,
    div_history: dict,
    div_step_history: list,
    grad_norm_buffer: list,
    grad_step_buffer: list,
    n_layers: int,
    n_columns: int,
    has_hgrn: bool,
    has_reservoir: bool,
    reservoir_sr_info: dict,
) -> None:

    # 1. AttnFlow (работало — оставляем)
    try:
        attn_vis.log(logger, step=step)
    except Exception as e:
        print(f'[VIS] AttnFlow error: {e}')

    # 2. CKA (работало — оставляем)
    try:
        cka_vis.log(logger, step=step)
    except Exception as e:
        print(f'[VIS] CKA error: {e}')

    # 3. Gate saturation
    if gate_buffer:
        try:
            fig = plot_gate_saturation(gate_buffer, n_layers, step)
            _log_figure(logger, fig, 'vis/gate_saturation', step)
        except Exception as e:
            print(f'[VIS] Gate saturation error: {e}')

        arr = np.array(gate_buffer)
        for li in range(min(arr.shape[1], n_layers)):
            try:
                logger.track(float(arr[:, li].mean()), name=f"attn_gate/L{li}", step=step)
                logger.track(float(arr[:, li].std()),  name=f"attn_gate_std/L{li}", step=step)
            except Exception:
                pass

    # 4. Column cosine similarity heatmap
    if col_sim_buffer:
        try:
            avg_sim = defaultdict(list)
            for d in col_sim_buffer:
                for k, v in d.items():
                    avg_sim[k].append(v)
            avg_sim_scalar = {k: float(np.mean(vs)) for k, vs in avg_sim.items()}
            fig = plot_col_similarity_heatmap(avg_sim_scalar, n_layers, n_columns, step)
            _log_figure(logger, fig, 'vis/col_similarity_heatmap', step)
            for k, v in avg_sim_scalar.items():
                try:
                    logger.track(v, name=k, step=step)
                except Exception:
                    pass
        except Exception as e:
            print(f'[VIS] ColSim heatmap error: {e}')

    # 5. Hidden norm heatmap
    if hidden_norm_buffer:
        try:
            fig = plot_hidden_norm_heatmap(hidden_norm_buffer, n_layers, n_columns, step)
            _log_figure(logger, fig, 'vis/hidden_norm_heatmap', step)
            for li in range(n_layers):
                for ci in range(n_columns):
                    key = f"hidden_norm/L{li}_C{ci}"
                    vals = [h.get(key) for h in hidden_norm_buffer if key in h]
                    if vals:
                        try:
                            logger.track(float(np.mean(vals)), name=key, step=step)
                        except Exception:
                            pass
        except Exception as e:
            print(f'[VIS] Hidden norm heatmap error: {e}')

    # 6. HGRN beta dynamics
    if has_hgrn and beta_buffer and len(beta_buffer) > 1:
        try:
            fig = plot_beta_dynamics(beta_buffer, beta_step_buffer, n_layers, n_columns)
            _log_figure(logger, fig, 'vis/beta_dynamics', step)
            if beta_buffer:
                last_betas = beta_buffer[-1]
                for k, v in last_betas.items():
                    try:
                        logger.track(v, name=k, step=step)
                    except Exception:
                        pass
        except Exception as e:
            print(f'[VIS] Beta dynamics error: {e}')

    # 7. Diversity loss curves
    if div_step_history:
        try:
            fig = plot_diversity_components(div_history, div_step_history)
            _log_figure(logger, fig, 'vis/diversity_loss_curves', step)
        except Exception as e:
            print(f'[VIS] Diversity curves error: {e}')

    # 8. Gradient norm per layer
    if grad_norm_buffer and len(grad_norm_buffer) > 1:
        try:
            fig = plot_grad_norm_per_layer(grad_norm_buffer, grad_step_buffer, n_layers)
            _log_figure(logger, fig, 'vis/grad_norm_per_layer', step)
        except Exception as e:
            print(f'[VIS] Grad norm vis error: {e}')

    # 9. Reservoir SR (scalar)
    if has_reservoir:
        for k, v in reservoir_sr_info.items():
            try:
                logger.track(v, name=k, step=step)
            except Exception:
                pass


if __name__ == "__main__":
    run_experiment(runner=main)