"""Unified text experiment — supports all models."""
from __future__ import annotations

from functools import partial
import importlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from knitwork.common.dynamic_param import DynamicParameter
from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging_alt import start_logger
from knitwork.common.numpy import get_seed
from knitwork.common.scheduler import create_scheduler
from knitwork.common.torch import DynamicLearningRate, to_loggable_metrics
from knitwork.common.tracker import Tracker
from knitwork.common.status import write_status
from knitwork.common.utils import (
    CE_ignore_index, FpsCounter, flatten_dict,
    format_readable_num, get_device, get_dtype,
    to_numpy, to_torch,
)
from knitwork.gens.text import TextGenerator, load_dataset, tokenize
from knitwork.visualization.attn_flow import AttnFlowVisualizer
from knitwork.visualization.cka import CKAVisualizer

VIS_INTERVAL = 10_000_000


# Model registry

_REGISTRY: dict[str, tuple[str, str] | None] = {
    'rnn':            ('knitwork.models.gru',           'GruBaseline'),
    'grnn':           ('knitwork.models.grnn',           'GridRnn'),
    'grnn_err':       ('knitwork.models.grnn_err',       'GridRnn'),
    'grnn2':          ('knitwork.models.grnn2',          'GridRnn2'),
    'grnn_lru':       ('knitwork.models.grnn_lru',       'GridLRU'),
    'grnn_lru_wide':  ('knitwork.models.grnn_lru',       'GridLRU'),
    'hgrnn':          ('knitwork.models.hgrnn',          'HopfieldGridRnn'),
    'hgrnn_lru':      ('knitwork.models.hgrnn_lru',      'HopfieldGridLRU'),
    'hgrn_grnn':      ('knitwork.models.hgrn_grnn',      'HGRN_GridRnn'),
    'grnn_fw':        ('knitwork.models.grnn_fw',        'GridRnnFW'),
    'grnn_reservoir': ('knitwork.models.grnn_reservoir', 'GridRnnReservoir'),
    'grnn_loss':      ('knitwork.models.grnn_loss',      'GridRnnLoss'),
    'grnn_engram':    ('knitwork.models.engram_grnn',    'EngramGridRnn'),
    'grnn_prec_delta': ('knitwork.models.grnn_prec_delta', 'GridRnnPrecDelta'),
    'grnn_ema_mem':   ('knitwork.models.grnn_ema_mem',   'GridRnnEmaMem'),
    'grnn_fusion':    None,  # factory
    # config aliases
    'grnn_res':       ('knitwork.models.grnn_reservoir', 'GridRnnReservoir'),
    'grnn_delta':     ('knitwork.models.grnn_delta',    'GridDelta'),
    'grnn_delta_wide':('knitwork.models.grnn_delta',    'GridDelta'),
    'grnn_harmonic':  ('knitwork.models.grnn_harmonic', 'HarmonicGridRNN'),
    # external baselines
    'delta_net':      ('knitwork.models.baseline.delta_net', 'DeltaNet'),
    'hgrn2':          ('knitwork.models.baseline.hgrn2',     'HGRN2'),
    'mlstm':          ('knitwork.models.baseline.mlstm',     'mLSTM'),
}


def main(config):
    _default_name = f"{config['model']}"
    run_name = config.get('name') or config.get('log', {}).get('name') or _default_name
    config.setdefault('log', {})['name'] = run_name
    print(f'Run name: {run_name}')

    rng = np.random.default_rng(config['seed'])
    device = get_device(config.get('device'))
    dtype = get_dtype(config.get('dtype'))
    n_envs = config['n_envs']

    gen_cfg   = config['gens'][config['gen']]
    data_path = Path(gen_cfg['path']).expanduser()
    data, charset = tokenize(load_dataset(data_path))
    n_chars = charset.size

    train_data = data
    eval_cfg = config.get('eval', {})
    do_eval = eval_cfg.get('enabled', False)

    if do_eval:
        from knitwork.gens.text import split_train_test
        eval_schedule = create_scheduler(eval_cfg['schedule'])
        max_rollout = int(eval_cfg.get('max_rollout', 1e+8))
        train_frac = 1.0 - eval_cfg['split']
        train_data, val_data = split_train_test(data, train_frac=train_frac)
        val_gen = TextGenerator(val_data, n_envs=n_envs, ignore_index=CE_ignore_index, seed=get_seed(rng))

    gen = TextGenerator(train_data, n_envs=n_envs, ignore_index=CE_ignore_index, seed=get_seed(rng))

    rnn_type = config['model']
    rnn_cfg = config[rnn_type]
    rnn = build_model(rnn_type, rnn_cfg, n_chars)
    rnn = rnn.to(device=device, dtype=dtype)
    print(f'Model on {next(rnn.parameters()).device} | dtype {next(rnn.parameters()).dtype}')

    use_vae = getattr(rnn, 'use_vae', False)
    has_grid = hasattr(rnn, 'n_layers') and hasattr(rnn, 'n_columns')
    has_harmonic = hasattr(rnn, 'mem_layers') and hasattr(rnn, 'flatten_extras_stats')

    # Visualizers
    vis_cfg = config.get('vis', {})
    vis_enabled = vis_cfg.get('enabled', False)
    if vis_enabled:
        vis_schedule = create_scheduler(vis_cfg['schedule'])
        if has_grid:
            attn_vis = AttnFlowVisualizer(n_layers=rnn.n_layers, n_columns=rnn.n_columns, buffer_size=100)
            cka_vis = CKAVisualizer(n_layers=rnn.n_layers, n_columns=rnn.n_columns, buffer_size=50)
        else:
            attn_vis, cka_vis = None, None
    gate_buffer: list = []


    # KL annealing
    kl_cfg = config.get('kl_anneal', {})
    kl_steps = int(kl_cfg.get('steps', 50_000))
    kl_max = float(kl_cfg.get('max_weight', 1.0))
    kl_anneal = lambda step: kl_max if kl_steps == 0 else kl_max * min(1.0, step / kl_steps)

    lr = DynamicLearningRate(name=f'LR', **config['lr'])
    optim = torch.optim.RMSprop(rnn.parameters(), lr=lr.val)
    lr.connect_to_optimiser(optim)

    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)

    # p_reset schedule
    p_reset = DynamicParameter(**gen_cfg['reset_prob'])

    rollout_len = config['rollout_len']
    batch_size = gen.n_envs * rollout_len
    n_steps = int(config['n_steps'])
    step = 0

    logger = start_logger(
        config, tracker=config['trackers'], 
        printer=partial(print_metrics, max_steps=n_steps, use_vae=use_vae, lr=lr)
    )

    ln_2 = np.log(2.0)
    rnn_state = None
    batch_y: list = []
    batch_y_gt: list = []
    batch_kl: list = []
    batch_harmonic: list = []   # harmonic model diagnostics
    acc_by_pos: dict[int, float] = {}

    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}

        rnd_reset  = torch.from_numpy(rng.random(gen.n_envs) < p_reset.val).to(device)
        reset_mask = torch.logical_or(obs['reset_mask'], rnd_reset)
        rnn_state  = rnn.reset_state(rnn_state, reset_mask)
        x = obs['tokens'].view(-1, 1)

        capture = vis_enabled and (step >= next_vis_step - gen.n_envs)
        capture_extras = capture or has_harmonic
        y, rnn_state, extras, kl = model_forward(rnn, x, rnn_state, capture=capture_extras)

        if has_harmonic and extras:
            batch_harmonic.append(rnn.flatten_extras_stats(extras))

        if capture and extras and attn_vis is not None:
            attn_vis.update(extras['attn_weights'])
            h_for_cka = rnn_state[0] if isinstance(rnn_state, tuple) else rnn_state
            if cka_vis is not None:
                cka_vis.update(h_for_cka)
            gate_vals = [
                g.detach().sigmoid().mean().item()
                for g in extras.get('gates', [])
                if isinstance(g, torch.Tensor)
            ]
            if gate_vals:
                gate_buffer.append(gate_vals)

        if vis_enabled and step >= next_vis_step and logger is not None:
            if attn_vis is not None:
                attn_vis.log(logger, step=step)
            if cka_vis is not None:
                cka_vis.log(logger, step=step)
            if gate_buffer:
                arr = np.array(gate_buffer)
                for li in range(min(arr.shape[1], rnn.n_layers)):
                    logger.track(float(arr[:, li].mean()), name=f"attn_gate/L{li}", step=step)
            gate_buffer.clear()
            next_vis_step += VIS_INTERVAL

        batch_y.append(y)
        batch_y_gt.append(obs['targets'])
        batch_kl.append(kl if kl is not None else torch.tensor(0.0, device=device, dtype=dtype))

        step += gen.n_envs

        if step % batch_size == 0:
            y_cat  = torch.cat(batch_y, dim=0)
            y_gt_cat = torch.cat(batch_y_gt, dim=0)
            m_active = y_gt_cat != CE_ignore_index

            ce_loss = loss_fn(y_cat, y_gt_cat)
            kl_mean = torch.stack(batch_kl).mean()
            kl_scale = kl_anneal(step)
            total_loss = ce_loss + kl_scale * kl_mean

            with torch.no_grad():
                logits_a = y_cat[m_active]
                gt_a = y_gt_cat[m_active]
                acc = (logits_a.argmax(dim=-1) == gt_a).float()
                bpc = ce_loss / ln_2
                perplexity = torch.exp(ce_loss)
                update_intra_word_metrics(acc_by_pos, obs['tokens'].view(-1), acc)

            optim.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('Nan/Inf grad — step skipped')

            p_reset.step()
            lr.step()

            metrics = {
                'Loss': ce_loss,
                'BPC': bpc,
                'Perplexity': perplexity,
                'Acc': acc,
                '|Grad|': grad_norm,
                'LR': lr.val,
                'T': 1.0 / p_reset.val,
            }
            if use_vae:
                metrics['KL'] = kl_mean
                metrics['KL_scale'] = kl_scale
            for pos, val in list(acc_by_pos.items())[:4]:
                metrics[f'Acc[{pos}]'] = val
            if has_harmonic and batch_harmonic:
                keys = batch_harmonic[0].keys()
                for k in keys:
                    vals = [d[k] for d in batch_harmonic if k in d]
                    if vals:
                        metrics[k] = float(np.mean(vals))
                batch_harmonic.clear()
            metrics = to_loggable_metrics(metrics)
            logger.accumulate(metrics, key='slow')

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear(); batch_y_gt.clear(); batch_kl.clear()

        logger.log(step, flush=True)
        # if log_stats_schedule.tick(gen.n_envs):
        #     fps = fps_counter.fps(n_iters=step, start=True)
        #     metrics = {'global_step': step, 'fps': fps} | stats.get()
        #     metrics['gen'] = gen.get_stats()
        #     write_status(step, metrics)
        #     if logger is not None:
        #         logger.track(flatten_dict(metrics))
        #         if has_grid:
        #             log_col_similarity(rnn, rnn_state, logger, step)

        if do_eval and eval_schedule.tick(gen.n_envs):
            run_eval(
                step, rnn=rnn, gen=val_gen, logger=logger, n_envs=n_envs, max_rollout=max_rollout,
                device=device, ln_2=ln_2, context_window=None
            )

    # fps = fps_counter.fps(n_iters=step)
    # print(f'Done. {format_readable_num(fps)} fps')

def run_eval(
        step: int, *, rnn, gen, logger, n_envs, max_rollout, device, ln_2,
        context_window=None
):
    """Evaluate on val set; if context_window>0, also run context-memory probe."""
    rnn.eval()
    val_state = None
    val_loss_acc  = []
    val_char_acc  = []

    if context_window is not None:
        cw_ix = np.zeros(n_envs, dtype=np.int64)
        cw_ix_bpc = np.zeros(context_window)
        cw_ix_cnt = np.zeros(context_window)

    with torch.no_grad():
        for _ in range(max_rollout):
            obs = gen.next()
            obs = {k: to_torch(v, device=device) for k, v in obs.items()}

            # reset at dataset wrap
            val_state = rnn.reset_state(val_state, obs['reset_mask'])

            # context window reset: zero state for envs at boundary
            if context_window is not None:
                cw_reset = torch.from_numpy(cw_ix == 0).to(device)
                val_state = rnn.reset_state(val_state, cw_reset)

            x = obs['tokens'].view(-1, 1)
            y, val_state, *_ = model_forward(rnn, x, val_state, capture=False)

            targets = obs['targets']
            valid = targets != CE_ignore_index
            if valid.any():
                ce = nn.functional.cross_entropy(y[valid], targets[valid], reduction='none')
                val_loss_acc.append(ce.mean().item())
                acc_t = (y[valid].argmax(dim=-1) == targets[valid]).float().mean().item()
                val_char_acc.append(acc_t)

                if context_window is not None:
                    cw_ix_bpc[cw_ix] += (ce / ln_2).cpu().numpy()
                    cw_ix_cnt[cw_ix] += 1.0
                    cw_ix = (cw_ix + 1) % context_window

            val_state = rnn.detach_state(val_state)

    rnn.train()

    if not val_loss_acc:
        return

    # val_bpc  = float(np.mean(val_loss_acc)) / ln_2
    # val_loss = float(np.mean(val_loss_acc))
    # val_acc  = float(np.mean(val_char_acc))
    # print(f'[EVAL step={format_readable_num(step)}]'
    #         f' BPC:{val_bpc:.3f} Acc:{val_acc:.3f}')
    # if logger is not None:
    #     logger.track(val_bpc,  name='val/BPC',  step=step)
    #     logger.track(val_loss, name='val/Loss', step=step)
    #     logger.track(val_acc,  name='val/Acc',  step=step)

    # if context_window > 0:
    #     bpc_curve = np.array([
    #         float(np.mean(cw_ix_bpc[p])) if cw_ix_bpc[p] else float('nan')
    #         for p in range(context_window)
    #     ])
    #     # log key percentiles numerically
    #     for label, frac in [('p0', 0.0), ('p25', 0.25), ('p50', 0.50), ('p75', 0.75), ('p100', 0.99)]:
    #         idx = min(int(frac * context_window), context_window - 1)
    #         if not np.isnan(bpc_curve[idx]):
    #             if logger is not None:
    #                 logger.track(float(bpc_curve[idx]),
    #                                 name=f'val/ctx/{label}', step=step)
    #     # visual plot
    #     if logger is not None:
    #         try:
    #             import matplotlib
    #             matplotlib.use('Agg')
    #             import matplotlib.pyplot as plt
    #             from knitwork.exps.sdq._viz import log_figure
    #             fig, ax = plt.subplots(figsize=(8, 4))
    #             valid_mask = ~np.isnan(bpc_curve)
    #             ax.plot(np.where(valid_mask)[0], bpc_curve[valid_mask], lw=1.5)
    #             ax.set_xlabel('Position in context window (tokens)')
    #             ax.set_ylabel('BPC')
    #             ax.set_title(f'Context memory probe (window={context_window}) step={format_readable_num(step)}')
    #             ax.grid(True, alpha=0.3)
    #             plt.tight_layout()
    #             log_figure(logger, fig, 'val/ctx/bpc_curve', step)
    #             plt.close(fig)
    #         except Exception as e:
    #             print(f'[EVAL ctx plot] {e}')


def build_model(rnn_type: str, rnn_cfg: dict, n_chars: int):
    if rnn_type == 'grnn_fusion':
        from knitwork.models.grnn_fusion import build_fusion_from_config
        return build_fusion_from_config(rnn_cfg, n_chars, n_chars)
    entry = _REGISTRY.get(rnn_type)
    if entry is None:
        raise ValueError(f'Unknown model type: {rnn_type!r}')
    mod_path, cls_name = entry
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(**rnn_cfg, input_size=n_chars, output_size=n_chars)


def log_col_similarity(rnn, state, logger, step: int) -> None:
    """Log max/mean pairwise cosine similarity between column activations (last layer)."""
    # Column collapse monitoring
    try:
        h = state[0] if isinstance(state, tuple) else state
        if not isinstance(h, torch.Tensor) or h.ndim != 4:
            return
        H    = rnn.hidden_size
        acts = h[-1, :, :, :H].mean(dim=1).detach().float()   # [cols, H]
        norm = acts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        acts = acts / norm
        sim  = acts @ acts.T                                    # [cols, cols]
        n    = sim.shape[0]
        mask = sim.new_ones(n, n, dtype=torch.bool).triu(diagonal=1)
        pairs = sim[mask]
        logger.track(float(pairs.max()),  name='col_sim/max',  step=step)
        logger.track(float(pairs.mean()), name='col_sim/mean', step=step)
    except Exception as e:
        print(f'[col_sim] {e}')


def model_forward(rnn, x, state, *, capture: bool):
    """Forward normalizer"""
    result = rnn(x, state, return_attn=True) if capture else rnn(x, state)
    y, state = result[0], result[1]
    if len(result) == 2:
        return y, state, {}, None
    if len(result) == 3:
        third = result[2]
        if isinstance(third, dict):
            return y, state, third, None
        return y, state, {}, third
    if len(result) == 4:
        return y, state, result[2], result[3]
    return y, state, {}, None


def update_intra_word_metrics(
    acc_by_pos: dict[int, float],
    tokens: torch.Tensor,   # [B]
    acc:    torch.Tensor,   # [B]
    lr: float = 0.01,
    space_idx: int = 0,
) -> None:
    """Intra-word accuracy"""
    pos = 0
    for t, a in zip(tokens.tolist(), acc.tolist()):
        if t == space_idx:
            pos = 0
        else:
            old = acc_by_pos.get(pos, a)
            acc_by_pos[pos] = old + lr * (a - old)
            pos += 1


def print_metrics(step, metrics, max_steps, use_vae, lr):
    m = metrics
    fps = m['perf/fps']
    kl_s = f' KL:{m.get("KL", 0):.2e}(x{m.get("KL_scale", 0):.2f}) |' if use_vae else ''
    print(
        f'[{format_readable_num(step)}/{format_readable_num(max_steps, frac=0)}]'
        f' {format_readable_num(fps, frac=0)}fps |'
        f' LR:{int(100*m["LR"]/lr.base_val)}%'
        f' T:{int(m["T"])} |{kl_s}'
        f' L:{m["Loss"]:.3f}'
        f' BPC:{m["BPC"]:.3f}'
        f' A:{m["Acc"]:.3f}'
    )


if __name__ == '__main__':
    run_experiment(runner=main)
