"""Unified text experiment — supports all models."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn

from knitwork.common.dynamic_param import DynamicParameter
from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging_alt import start_logger
from knitwork.common.numpy import get_seed
from knitwork.common.scheduler import create_scheduler
from knitwork.common.torch import DynamicLearningRate, to_loggable_metrics, to_numpy
from knitwork.common.status import write_status
from knitwork.common.tracking import SplitEmaTracker
from knitwork.common.utils import (
    CE_ignore_index, count_learnable_params, dont_throw, format_readable_num, get_device, 
    get_dtype,
)
from knitwork.gens.text import TextGenerator, load_dataset, tokenize
from knitwork.models.utils import build_model, model_forward


def main(config):
    _default_name = f"{config['model']}"
    run_name = config.get('name') or config['log'].get('name') or _default_name

    rng = np.random.default_rng(config['seed'])
    device = get_device(config.get('device'))
    dtype = get_dtype(config.get('dtype'))
    n_envs = config['n_envs']

    gen_cfg   = config['gens'][config['gen']]
    data_path = Path(gen_cfg['path']).expanduser()
    data, charset = tokenize(load_dataset(data_path))
    n_chars = charset.size
    # print(f"{charset.tobytes().decode('utf-8')!r}")
    space_token = charset.tobytes().decode('utf-8').find(' ')

    train_data = data
    eval_cfg = config.get('eval', {})
    do_eval = eval_cfg.get('enabled', False)
    do_eval_on_start = eval_cfg.get('on_start', False)

    if do_eval:
        from knitwork.gens.text import split_train_test
        eval_schedule = create_scheduler(eval_cfg['schedule'])
        max_rollout = int(eval_cfg.get('max_rollout', 1e+8))
        context_window = eval_cfg.get('context_window', None)
        train_frac = 1.0 - eval_cfg['split']
        train_data, val_data = split_train_test(data, train_frac=train_frac)
        val_gen = TextGenerator(val_data, n_envs=n_envs, ignore_index=CE_ignore_index, seed=get_seed(rng), device=device)
        max_rollout = min(
            max_rollout, 
            max(100*config['rollout_len'], round(len(val_data) / n_envs))
        )

    gen = TextGenerator(train_data, n_envs=n_envs, ignore_index=CE_ignore_index, seed=get_seed(rng), device=device)

    rnn_type = config['model']
    rnn_cfg = config[rnn_type]
    rnn = build_model(rnn_type, rnn_cfg, n_chars)
    rnn = rnn.to(device=device, dtype=dtype)
    print(f'Model on {next(rnn.parameters()).device} | dtype {next(rnn.parameters()).dtype}')

    run_name = f"{run_name}_{count_learnable_params(rnn, as_str=True)}"
    config['log']['name'] = run_name
    print(f'Run name: {run_name}')

    rollout_len = config['rollout_len']
    batch_size = gen.n_envs * rollout_len
    n_steps, step_size = int(config['n_steps']), gen.n_envs

    if config.get('adapt_to_bsz', None) == 'auto':
        # factor explressing an update frequency relative to the "default" rollout=32 bsz=512
        # NB: see its usages for how it affects schedules of some trackers. 
        # Such adaptation is not ideal, and reasonable only in a short range
        update_freq_alpha = (batch_size / 32 / 512)**0.5
        n_steps = int(n_steps * update_freq_alpha)
        config['log']['schedule'] *= update_freq_alpha
        config['eval']['schedule'] *= update_freq_alpha
        config['lr']['schedule'] /= update_freq_alpha
        config['lr']['warmup']['schedule'] /= update_freq_alpha
        gen_cfg['reset_prob']['schedule'] /= update_freq_alpha
        config['trackers']['slow'] *= update_freq_alpha
        if do_eval:
            eval_schedule = create_scheduler(eval_cfg['schedule'])

    use_vae = getattr(rnn, 'use_vae', False)
    has_grid = hasattr(rnn, 'n_layers') and hasattr(rnn, 'n_columns')
    has_harmonic = hasattr(rnn, 'mem_layers') and hasattr(rnn, 'flatten_extras_stats')

    inspect_scheduler = create_scheduler(config.get('inspect_schedule'))
    vis_inspect_scheduler = create_scheduler(config.get('vis_inspect_schedule'))
    if not vis_inspect_scheduler.is_infinite and has_grid:
        from knitwork.visualization.attn_flow import AttnFlowVisualizerNew
        from knitwork.visualization.cka import CKAVisualizerNew
        attn_vis = AttnFlowVisualizerNew(n_layers=rnn.n_layers, n_columns=rnn.n_columns, lr=0.01)
        cka_vis = CKAVisualizerNew(n_layers=rnn.n_layers, n_columns=rnn.n_columns, lr=0.01)

    def _inject_visualizations(step, *, scalars, figures):
        if vis_inspect_scheduler.is_infinite:
            return
        if has_grid and 'val/Loss' in scalars:
            # log figures only with eval schedule
            figures |= attn_vis.get_figures()
            figures |= cka_vis.get_figures()

    # KL annealing
    kl_cfg = config.get('kl_anneal', {})
    kl_steps = int(kl_cfg.get('steps', 50_000))
    kl_max = float(kl_cfg.get('max_weight', 1.0))
    kl_anneal = lambda step: kl_max if kl_steps == 0 else kl_max * min(1.0, step / kl_steps)

    lr = DynamicLearningRate(name=f'LR', **config['lr'])
    optim = torch.optim.RMSprop(rnn.parameters(), lr=lr.val)
    lr.connect_to_optimiser(optim)

    # p_reset schedule
    gen_cfg['reset_prob']['val'] /= rollout_len
    gen_cfg['reset_prob']['tar'] /= rollout_len
    p_reset = DynamicParameter(**gen_cfg['reset_prob'])

    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)

    dump_status_enabled = config.get('dump_status', False)
    def dump_status(step, *, scalars, figures):
        write_status(step, metrics)

    _print_short_summary = partial(print_short_summary, max_steps=n_steps, use_vae=use_vae, lr=lr)
    log_callbacks = [_print_short_summary, _inject_visualizations]
    if dump_status_enabled:
        log_callbacks.append(dump_status)
    logger = start_logger(
        config, tracker=config['trackers'],
        suppress_printing=True, callbacks=log_callbacks
    )

    in_word_acc = SplitEmaTracker(bins=config['inspect_n_in_word_acc'], lr=0.01)
    in_word_acc.ixs = torch.zeros(step_size, dtype=torch.int64, device=device)

    ln_2 = np.log(2.0)
    step, i_update = 0, 0
    rnn_state = None
    batch_y, batch_y_gt, batch_kl = [], [], []
    batch_in_word_pos = []

    def _run_eval(step):
        run_eval(
            step, rnn=rnn, gen=val_gen, logger=logger, n_envs=n_envs, max_rollout=max_rollout,
            device=device, context_window=context_window
        )

    if do_eval and do_eval_on_start:
        _run_eval(step)
        logger.log(step, flush=True, force=True)

    while step < n_steps:
        obs = gen.next()

        rnd_reset = torch.rand(gen.n_envs, device=device, generator=gen.rng) < p_reset.val
        reset_mask = torch.logical_or(obs['reset_mask'], rnd_reset)
        rnn_state  = rnn.reset_state(rnn_state, reset_mask)
        x = obs['tokens'].view(-1, 1)

        collect_vis_data = vis_inspect_scheduler.tick(step_size)
        y, rnn_state, extras, kl = model_forward(rnn, x, rnn_state, capture=collect_vis_data or has_harmonic)

        if has_harmonic and extras:
            harmonic_stats = to_loggable_metrics(rnn.flatten_extras_stats(extras))
            logger.accumulate(harmonic_stats, key='slow')

        if collect_vis_data and extras and has_grid:
            attn_vis.update(extras['attn_weights'])
            h_for_cka = rnn_state[0] if isinstance(rnn_state, tuple) else rnn_state
            cka_vis.update(h_for_cka)

            gate_metrics = {
                f'attn_gate/L{li}': g.detach().sigmoid().mean()
                for li, g in enumerate(extras.get('gates', []))
            }
            logger.accumulate(gate_metrics, key='fast')

        batch_y.append(y)
        batch_y_gt.append(obs['targets'])
        batch_in_word_pos.append(in_word_acc.ixs)
        if use_vae:
            batch_kl.append(kl if kl is not None else torch.tensor(0.0, device=device, dtype=dtype))

        step += step_size
        in_word_acc.ixs = torch.where(x.view(-1) == space_token, 0, in_word_acc.ixs + 1)

        if step % batch_size == 0:
            y_cat  = torch.cat(batch_y, dim=0)
            y_gt_cat = torch.cat(batch_y_gt, dim=0)
            m_active = y_gt_cat != CE_ignore_index

            total_loss = ce_loss = loss_fn(y_cat, y_gt_cat)
            if use_vae:
                kl_mean = torch.stack(batch_kl).mean()
                kl_scale = kl_anneal(step)
                total_loss = total_loss + kl_scale * kl_mean

            with torch.no_grad():
                logits_a = y_cat[m_active]
                gt_a = y_gt_cat[m_active]
                acc = (logits_a.argmax(dim=-1) == gt_a).float()
                bpc = ce_loss / ln_2
                perplexity = torch.exp(ce_loss)
                update_in_word_acc(in_word_acc, batch_in_word_pos, acc, m_active)

            optim.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('Nan/Inf grad — step skipped')

            p_reset.step()
            lr.step()
            i_update += 1

            metrics = {
                'Loss': ce_loss,
                'BPC': bpc,
                'Perplexity': perplexity,
                'Acc': acc,
                '|Grad|': grad_norm,
                'LR': lr.val,
                'T': min(1e+6, 1.0 / p_reset.val),
                'Upd': i_update,
            }
            if use_vae:
                metrics['KL'] = kl_mean
                metrics['KL_scale'] = kl_scale
            metrics = to_loggable_metrics(metrics)
            logger.accumulate(metrics, key='slow')
            logger.accumulate(in_word_acc.get(split=True), key='fast')

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear(); batch_y_gt.clear(); batch_kl.clear()

        if has_grid and inspect_scheduler.tick(step_size):
            log_col_similarity(rnn, rnn_state, logger)
            log_lru_spectrum(rnn, logger)
            log_attn_beta(rnn, logger)

        if do_eval and eval_schedule.tick(step_size):
            _run_eval(step)

        logger.log(step, flush=True)

    if do_eval and eval_schedule.tick(step_size):
        _run_eval(step)
    logger.log(step, flush=True, force=True)
    logger.finish()


@torch.no_grad()
def run_eval(
        step: int, *, rnn, gen, logger, n_envs, max_rollout, device,
        context_window=None
):
    """Evaluate on val set; if context_window>0, also run context-memory probe."""
    rnn.eval()
    state = None
    ce_loss, acc, tot_cnt = 0.0, 0.0, 0

    if context_window is not None:
        cw_ix = torch.zeros(n_envs, dtype=torch.int64, device=device)
        cw_ix_ce = torch.zeros(context_window, device=device)
        cw_ix_cnt = torch.zeros(context_window, device=device) + 1.0e-9

    for _ in range(max_rollout):
        obs = gen.next()

        # reset at dataset wrap [and at context window boundary]
        reset_mask = obs['reset_mask']
        if context_window is not None:
            reset_mask = torch.logical_or(reset_mask, cw_ix == 0)
        state = rnn.reset_state(state, reset_mask)

        x = obs['tokens'].view(-1, 1)
        y, state, *_ = model_forward(rnn, x, state, capture=False)

        targets = obs['targets']
        valid = targets != CE_ignore_index
        y, targets = y[valid], targets[valid]
        tot_cnt += y.shape[0]
        if y.shape[0] == 0:
            continue

        ce = nn.functional.cross_entropy(y, targets, reduction='none')
        ce_loss = ce_loss + ce.sum()
        acc = acc + (y.argmax(dim=-1) == targets).sum()
        if context_window is not None:
            _cw_ix = cw_ix[valid]
            cw_ix_ce.index_add_(0, _cw_ix, ce)
            cw_ix_cnt += torch.bincount(_cw_ix, minlength=context_window)
            cw_ix[valid] = (_cw_ix + 1) % context_window

    rnn.train()

    if tot_cnt == 0:
        print('No valid data for evaluation!')
        return

    ce_loss, acc = ce_loss / tot_cnt, acc / tot_cnt

    ln_2 = np.log(2.0)
    metrics = to_loggable_metrics({
        'Loss': ce_loss,
        'BPC': ce_loss / ln_2,
        'Acc': acc,
    })
    logger.accumulate(metrics, prefix='val', key='eval')

    if context_window is not None:
        cw_ix_bpc = to_numpy(cw_ix_ce / cw_ix_cnt / ln_2)
        # log key percentiles numerically
        label_fracs = [('p0', 0.0), ('p25', 0.25), ('p50', 0.50), ('p75', 0.75), ('p100', 1.0)]
        def _frac_to_ix(frac):
            return round(frac * (context_window-1))
        metrics = {
            f'BPC_{label}': cw_ix_bpc[_frac_to_ix(frac)]
            for label, frac in label_fracs
        }
        logger.accumulate(metrics, prefix='val.context_window', key='list')

        from knitwork.visualization.context_analysis import plot_bpc_by_context_pos
        figures = {
            'bpc_curve': plot_bpc_by_context_pos(cw_ix_bpc, step=step),
        }
        logger.accumulate(figures, prefix='val.context_window', key='list')


@torch.no_grad()
@dont_throw('col_sim')
def log_col_similarity(rnn, state, logger):
    """Log max/mean pairwise cosine similarity between column activations (last layer)."""
    # Column collapse monitoring
    h = state[0] if isinstance(state, tuple) else state
    if not isinstance(h, torch.Tensor) or h.ndim != 4:
        return

    H = rnn.hidden_size
    acts = h[-1, :, :, :H].mean(dim=1)
    norm = acts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    acts = acts / norm

    # [cols, cols]
    sim = acts @ acts.T
    mask = torch.ones_like(sim, dtype=torch.bool).triu(diagonal=1)
    sim = sim[mask]

    metrics = {
        'max': sim.max(), 
        'avg': sim.mean(),
    }
    metrics = to_loggable_metrics(metrics)
    logger.accumulate(metrics, prefix='col_sim', key='fast')


@torch.no_grad()
@dont_throw('LRU spectrum')
def log_lru_spectrum(rnn, logger):
    if not hasattr(rnn, 'cells'):
        return {}

    metrics = {}
    for li, row in enumerate(rnn.cells):
        for ci, cell in enumerate(row):
            lru = getattr(cell, 'lru', cell)
            if not hasattr(lru, 'nu'):
                continue
            if hasattr(lru, '_lambda_gamma'):
                lam_re, lam_im, _ = lru._lambda_gamma()
                r = torch.sqrt(lam_re ** 2 + lam_im ** 2)
            else:
                r = torch.exp(-torch.exp(lru.nu))
            entropy = -(r * torch.log(r + 1e-8)).sum()

            metrics[f'r_avg/L{li}_C{ci}'] = r.mean()
            metrics[f'r_min/L{li}_C{ci}'] = r.min()
            metrics[f'r_max/L{li}_C{ci}'] = r.max()
            metrics[f'r_H/L{li}_C{ci}'] = entropy

    metrics = to_loggable_metrics(metrics)
    logger.accumulate(metrics, prefix='lru', key='fast')


@torch.no_grad()
@dont_throw('attn beta')
def log_attn_beta(rnn, logger):
    if not hasattr(rnn, 'attn'):
        return

    metrics = {}
    for li, attn in enumerate(rnn.attn):
        lb = getattr(attn, 'log_beta', None)
        if lb is None:
            continue
        beta = lb.exp().detach().float()
        if beta.ndim == 2:
            for ci in range(beta.shape[0]):
                metrics[f'L{li}_C{ci}'] = beta[ci].mean()
        else:
            metrics[f'L{li}'] = beta.mean()
    metrics = to_loggable_metrics(metrics)
    logger.accumulate(metrics, prefix='attn_beta', key='fast')


@torch.no_grad()
def update_in_word_acc(in_word_acc, batch_in_word_pos, acc, m_active):
    acc = to_numpy(acc, copy=False).ravel()
    # take only "active" samples to align with acc
    in_word_pos = torch.concat(batch_in_word_pos)[m_active.view(-1)]
    in_word_pos = to_numpy(in_word_pos, copy=False)
    batch_in_word_pos.clear()

    # merge all "outer" positions into the last bin
    in_word_pos = np.minimum(in_word_pos, in_word_acc.n_bins - 1)

    in_word_acc.put({'in_word_stats/Acc': acc}, ixs=in_word_pos)


def print_short_summary(step, *, scalars, figures, max_steps, use_vae, lr):
    m = scalars
    if 'Loss' in m:
        # Train data is available
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

    if 'val/Loss' in m:
        # Val data is available
        print(
            f'[{format_readable_num(step)}/{format_readable_num(max_steps, frac=0)} EVAL]'
            f' L:{m["val/Loss"]:.3f}'
            f' BPC:{m["val/BPC"]:.3f}'
            f' A:{m["val/Acc"]:.3f}'
        )


if __name__ == '__main__':
    run_experiment(runner=main)
