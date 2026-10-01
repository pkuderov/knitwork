"""Unified text experiment — supports all models."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import copy

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
from knitwork.gens.text import (
    TextGenerator, load_dataset, split_train_val_test, tokenize, tokenize_splits,
)
from knitwork.models.utils import build_model


def main(config):
    torch.set_float32_matmul_precision('high')

    default_name = config['model']
    name_sfx = config.get('name') or config['log'].get('name') or ''

    rng = np.random.default_rng(config['seed'])
    device = get_device(config.get('device'))
    dtype = get_dtype(config.get('dtype'))
    n_envs = config['n_envs']

    gen_cfg   = config['gens'][config['gen']]
    data_path = Path(gen_cfg['path']).expanduser()
    raw = load_dataset(data_path)

    eval_cfg = config.get('eval', {})
    do_eval = eval_cfg.get('enabled', False)
    do_eval_on_start = eval_cfg.get('on_start', False)
    if do_eval:
        # contiguous train|val|test; vocab from train only; test is touched once at the end
        splits = split_train_val_test(raw, eval_cfg['val_size'], eval_cfg['test_size'])
        (train_data, val_data, test_data), charset = tokenize_splits(*splits)
        eval_schedule = create_scheduler(eval_cfg['schedule'])
        context_window = eval_cfg.get('context_window', None)

        def _make_eval_gen(d):
            g = TextGenerator(d, n_envs=n_envs, ignore_index=CE_ignore_index, seed=get_seed(rng), device=device)
            return torch.compile(g.to(device))

        val_gen, test_gen = _make_eval_gen(val_data), _make_eval_gen(test_data)
        # one full pass over each eval split
        max_rollout = {k: len(d) // n_envs for k, d in [('val', val_data), ('test', test_data)]}
        print(f'Split (tokens): train {len(train_data):,} | val {len(val_data):,} | test {len(test_data):,}')
    else:
        train_data, charset = tokenize(raw)
    n_chars = charset.size
    space_token = charset.tobytes().decode('utf-8').find(' ')

    gen = TextGenerator(train_data, n_envs=n_envs, ignore_index=CE_ignore_index, seed=get_seed(rng), device=device)
    gen = torch.compile(gen.to(device))

    config['model_cfg'] = config['model'].replace('.', '_')
    config['model'] = config['model'].split('.', 1)[0]
    if config['model'] == 'transformer':
        raise ValueError('Use run_offline.py for the model=transformer')

    wrapper_cfg = config[f'{config["wrapper_model"]}_wrapper'] | dict(
        input_size=n_chars, output_size=n_chars,
        dtype=dtype, device=device
    )
    model = build_model(
        wrapper_type=config['wrapper_model'],
        wrapper_cfg=wrapper_cfg,
        rnn_type=config['model'],
        rnn_cfg=config[config['model_cfg']],
    )
    model = model.to(device=device, dtype=dtype)
    if config.get('compile', False):
        model = torch.compile(model)
    rnn = model.rnn
    print(f'Model on {next(model.parameters()).device} | dtype {next(model.parameters()).dtype}')

    run_name = f"{default_name}_{count_learnable_params(model, as_str=True)} {name_sfx}"
    config['log']['name'] = run_name
    print(f'Run name: {run_name}')

    rollout_len = config['rollout_len']
    batch_size = gen.n_envs * rollout_len
    n_steps, step_size = int(config['n_steps']), gen.n_envs

    use_vae = getattr(rnn, 'use_vae', False)
    has_grid = hasattr(rnn, 'n_layers') and hasattr(rnn, 'n_columns')
    has_harmonic = hasattr(rnn, 'mem_layers') and hasattr(rnn, 'flatten_extras_stats')
    communication_cfg = config.get('communication', {})
    comm_loss_enabled = None
    comm_loss_weight = float(communication_cfg.get('loss_weight', 0.0))
    comm_entropy_weight = float(communication_cfg.get('entropy_weight', 0.0))

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

    lr = DynamicLearningRate(name='LR', **config['lr'])
    optim = torch.optim.RMSprop(model.parameters(), lr=lr.val)
    lr.connect_to_optimiser(optim)

    # p_reset schedule
    gen_cfg['reset_prob']['val'] /= rollout_len
    gen_cfg['reset_prob']['tar'] /= rollout_len
    p_reset = DynamicParameter(name='1/T', **gen_cfg['reset_prob'])

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
    iter, step, i_update = 0, 0, 0
    state = None
    batch_y, batch_y_gt = [], []
    batch_kl, batch_comm_loss, batch_comm_entropy = 0.0, 0.0, 0.0
    batch_in_word_pos = []

    best = dict(loss=float('inf'), step=0, state=None)

    # full-continuity eval is the primary metric; the windowed one (state reset every `window` tokens)
    # matches the training context regime and is what checkpoints are selected on (when enabled)
    window = eval_cfg.get('window')

    def _eval_pass(step, gen_, tag, n_roll):
        kw = dict(model=model, gen=gen_, logger=logger, n_envs=n_envs, max_rollout=n_roll, device=device)
        res = run_eval(step, prefix=tag, context_window=context_window if tag == 'val' else None, **kw)
        if window:
            res = run_eval(step, prefix=f'{tag}_w{int(window)}', context_window=int(window), **kw)
        return res

    def _run_eval(step):
        res = _eval_pass(step, val_gen, 'val', max_rollout['val'])
        if res is not None and res['Loss'] < best['loss']:
            best.update(loss=res['Loss'], step=step, state=copy.deepcopy(model.state_dict()))

    def _run_test():
        # single test pass: best-val checkpoint (reported) and last weights (for reference)
        last_state = copy.deepcopy(model.state_dict())
        for tag, sd in [('test_last', last_state), ('test', best['state'])]:
            if sd is None:
                continue
            model.load_state_dict(sd)
            _eval_pass(step, test_gen, tag, max_rollout['test'])
        model.load_state_dict(last_state)
        print(f'Test done: best val Loss {best["loss"]:.4f} at step {best["step"]:,}')

    if do_eval and do_eval_on_start:
        _run_eval(step)
        logger.log(step, flush=True, force=True)

    while step < n_steps:
        obs = gen.next()

        rnd_reset = torch.rand(gen.n_envs, device=device, generator=gen.rng) < p_reset.val
        reset_mask = torch.logical_or(obs['reset_mask'], rnd_reset)
        state  = rnn.reset_state(state, reset_mask)
        x = step_tokens(obs['tokens'], rnn)

        capture_details = inspect_scheduler.tick(step_size)
        capture_vis_data = vis_inspect_scheduler.tick(step_size)
        capture = capture_details or capture_vis_data or has_harmonic

        y, state, info = model(x, state, capture=capture)

        if capture:
            # FIXME: not yet supported after rework
            if has_harmonic and False:
                harmonic_stats = to_loggable_metrics(rnn.flatten_extras_stats(extras))
                logger.accumulate(harmonic_stats, key='slow')

            if has_grid:
                cka_vis.update(state['h'])
            if 'attn_weights' in info:
                attn_vis.update(info['attn_weights'])
            if 'gates' in info:
                gate_metrics = {
                    f'attn_gate/L{li}': g.sigmoid().mean()
                    for li, g in enumerate(info['gates'])
                }
                logger.accumulate(gate_metrics, key='fast')

        batch_y.append(y)
        batch_y_gt.append(obs['targets'])
        batch_in_word_pos.append(in_word_acc.ixs)
        if use_vae and info.get('kl') is not None:
            batch_kl += info['kl'].mean()
        if comm_loss_enabled or (comm_loss_enabled is None and 'comm_loss' in info):
            comm_loss_enabled = True
            batch_comm_loss += torch.stack(info['comm_loss']).mean()
            batch_comm_entropy += torch.stack(info['comm_entropy']).mean()

        iter += 1
        step += step_size
        in_word_acc.ixs = torch.where(x.view(-1) == space_token, 0, in_word_acc.ixs + 1)

        if step % batch_size == 0:
            y_cat  = torch.cat(batch_y, dim=0)
            y_gt_cat = torch.cat(batch_y_gt, dim=0)
            m_active = y_gt_cat != CE_ignore_index

            total_loss = ce_loss = loss_fn(y_cat, y_gt_cat)
            if use_vae:
                kl = batch_kl / rollout_len
                kl_scale = kl_anneal(step)
                total_loss = total_loss + kl_scale * kl
            if comm_loss_enabled:
                comm_loss = batch_comm_loss / rollout_len
                comm_entropy = batch_comm_entropy / rollout_len
                total_loss = total_loss + comm_loss_weight * comm_loss - comm_entropy_weight * comm_entropy

            with torch.no_grad():
                logits_a = y_cat[m_active]
                gt_a = y_gt_cat[m_active]
                acc = (logits_a.argmax(dim=-1) == gt_a).float()
                bpc = ce_loss / ln_2
                perplexity = torch.exp(ce_loss)
                update_in_word_acc(in_word_acc, batch_in_word_pos, acc, m_active)

            optim.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                metrics['KL'] = kl
                metrics['KL_scale'] = kl_scale
            if comm_loss_enabled:
                metrics['L_comm'] = comm_loss
                metrics['H_comm'] = comm_entropy
            metrics = to_loggable_metrics(metrics)
            logger.accumulate(metrics, key='slow')
            logger.accumulate(in_word_acc.get(split=True), key='fast')

            state = rnn.detach_state(state)
            batch_y.clear(); batch_y_gt.clear();
            batch_kl, batch_comm_loss, batch_comm_entropy = 0.0, 0.0, 0.0

        if has_grid and inspect_scheduler.tick(step_size):
            log_col_similarity(rnn, state, logger)
            log_lru_spectrum(rnn, logger)
            log_attn_beta(rnn, logger)

        if do_eval and eval_schedule.tick(step_size):
            _run_eval(step)

        logger.log(step, flush=True)

    if do_eval:
        _run_eval(step)
        _run_test()
        # +1 so the final eval/test metrics are flushed even if the last step was already flushed
        step += 1
    logger.log(step, flush=True, force=True)
    logger.finish()


def step_tokens(tokens, rnn):
    # cores take a step input as (B, 1) [batch-first, grnn_lru] or (1, B) [sequence dim first]
    return tokens.view(-1, 1) if getattr(rnn, 'batch_first', False) else tokens.view(1, -1)


@torch.no_grad()
def run_eval(
        step: int, *, model, gen, logger, n_envs, max_rollout, device,
        context_window=None, prefix='val'
):
    """One full pass over an eval split; if context_window is set, also run context-memory probe."""
    model.eval()
    gen.reset()
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
        state = model.rnn.reset_state(state, reset_mask)

        x = step_tokens(obs['tokens'], model.rnn)
        y, state, _ = model(x, state, capture=False)

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

    model.train()

    if tot_cnt == 0:
        print('No valid data for evaluation!')
        return None

    ce_loss, acc = ce_loss / tot_cnt, acc / tot_cnt

    ln_2 = np.log(2.0)
    metrics = to_loggable_metrics({
        'Loss': ce_loss,
        'BPC': ce_loss / ln_2,
        'Acc': acc,
    })
    logger.accumulate(metrics, prefix=prefix, key='eval')

    if context_window is not None and prefix == 'val':
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

    return {'Loss': float(ce_loss), 'BPC': float(ce_loss) / ln_2, 'Acc': float(acc)}


@torch.no_grad()
@dont_throw('col_sim')
def log_col_similarity(rnn, state, logger):
    """Log max/mean pairwise cosine similarity between column activations (last layer)."""
    # Column collapse monitoring
    h = state['h']
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
    if not hasattr(rnn, 'cells') or not isinstance(rnn.cells, nn.ModuleList):
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
        lb = getattr(attn, 'pi_logtemp', None)
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
            f' LR:{int(100*m["LR"]/lr.special_base_val)}%'
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
