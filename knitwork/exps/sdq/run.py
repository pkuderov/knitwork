"""Store-Distract-Query experiment with the unified core-model API."""
from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch import nn

from knitwork.common.curriculum import CurriculumScheduler
from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging_alt import start_logger
from knitwork.common.numpy import get_seed
from knitwork.common.scheduler import create_scheduler
from knitwork.common.torch import DynamicLearningRate, to_loggable_metrics, to_torch
from knitwork.common.utils import (
    CE_ignore_index, count_learnable_params, dont_throw, format_readable_num,
    get_device, get_dtype,
)
from knitwork.gens.sdq import StoreDistractQueryGenerator
from knitwork.models.utils import build_model


def main(config):
    torch.set_float32_matmul_precision('high')

    default_name = config['model']
    name_sfx = config.get('name') or config['log'].get('name') or ''

    rng = np.random.default_rng(config['seed'])
    device = get_device(config.get('device'))
    dtype = get_dtype(config.get('dtype'))
    n_envs = config['n_envs']

    gen_cfg = config['gens'][config['gen']]
    gen = StoreDistractQueryGenerator(
        **gen_cfg,
        n_envs=n_envs,
        seed=get_seed(rng),
        ignore_index=CE_ignore_index,
    )

    config['model_cfg'] = config['model'].replace('.', '_')
    config['model'] = config['model'].split('.', 1)[0]
    wrapper_cfg = config[f'{config["wrapper_model"]}_wrapper'] | dict(
        input_size=gen.n_tokens,
        output_size=gen.V,
        dtype=dtype,
        device=device,
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

    run_name = f'{default_name}_{count_learnable_params(model, as_str=True)} {name_sfx}'
    config['log']['name'] = run_name
    print(f'Run name: {run_name}')

    rollout_len = config['rollout_len']
    batch_size = gen.n_envs * rollout_len
    n_steps, step_size = int(config['n_steps']), gen.n_envs

    has_grid = hasattr(rnn, 'n_layers') and hasattr(rnn, 'n_columns')
    communication_cfg = config.get('communication', {})
    comm_loss_enabled = None
    comm_loss_weight = float(communication_cfg.get('loss_weight', 0.0))
    comm_entropy_weight = float(communication_cfg.get('entropy_weight', 0.0))

    inspect_scheduler = create_scheduler(config.get('inspect_schedule'))
    vis_inspect_scheduler = create_scheduler(config.get('vis_inspect_schedule'))
    if not vis_inspect_scheduler.is_infinite and has_grid:
        from knitwork.visualization.attn_flow import AttnFlowVisualizerNew
        from knitwork.visualization.cka import CKAVisualizerNew
        attn_vis = AttnFlowVisualizerNew(
            n_layers=rnn.n_layers,
            n_columns=rnn.n_columns,
            lr=0.01,
        )
        cka_vis = CKAVisualizerNew(n_layers=rnn.n_layers, n_columns=rnn.n_columns, lr=0.01)

    def inject_visualizations(step, *, scalars, figures):
        if vis_inspect_scheduler.is_infinite or not has_grid:
            return
        figures |= attn_vis.get_figures()
        figures |= cka_vis.get_figures()

    lr = DynamicLearningRate(name='LR', **config['lr'])
    optim = torch.optim.RMSprop(model.parameters(), lr=lr.val)
    lr.connect_to_optimiser(optim)
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)
    curriculum = CurriculumScheduler(**config['curriculum'])

    print_summary = partial(print_short_summary, max_steps=n_steps, lr=lr)
    logger = start_logger(
        config,
        tracker=config['trackers'],
        suppress_printing=True,
        callbacks=[print_summary, inject_visualizations],
    )

    step, i_update = 0, 0
    state = None
    batch_y, batch_y_gt, batch_sq_gaps = [], [], []
    batch_comm_loss, batch_comm_entropy = 0.0, 0.0
    curriculum_metrics = None

    while step < n_steps:
        obs = {
            key: to_torch(value, device=device)
            for key, value in gen.next().items()
        }
        state = rnn.reset_state(state, obs['reset_mask'])
        x = obs['tokens'].view(-1, 1)

        capture_details = inspect_scheduler.tick(step_size)
        capture_vis_data = vis_inspect_scheduler.tick(step_size)
        capture = capture_details or capture_vis_data
        y, state, info = model(x, state, capture=capture)

        if capture:
            if has_grid:
                cka_vis.update(state['h'])
            if 'attn_weights' in info:
                attn_vis.update(info['attn_weights'])

        batch_y.append(y)
        batch_y_gt.append(obs['targets'])
        batch_sq_gaps.append(obs['sq_gaps'])
        if comm_loss_enabled or (comm_loss_enabled is None and 'comm_loss' in info):
            comm_loss_enabled = True
            batch_comm_loss += torch.stack(info['comm_loss']).mean()
            batch_comm_entropy += torch.stack(info['comm_entropy']).mean()

        step += step_size

        if step % batch_size == 0:
            y_cat = torch.cat(batch_y, dim=0)
            y_gt_cat = torch.cat(batch_y_gt, dim=0)
            sq_gaps = torch.cat(batch_sq_gaps, dim=0).float()
            m_active = y_gt_cat != CE_ignore_index

            total_loss = ce_loss = loss_fn(y_cat, y_gt_cat)
            if comm_loss_enabled:
                comm_loss = batch_comm_loss / rollout_len
                comm_entropy = batch_comm_entropy / rollout_len
                total_loss = (
                    total_loss
                    + comm_loss_weight * comm_loss
                    - comm_entropy_weight * comm_entropy
                )

            with torch.no_grad():
                logits_a = y_cat[m_active]
                targets_a = y_gt_cat[m_active]
                acc = (logits_a.argmax(dim=-1) == targets_a).float()
                # The generator emits one gap for each query, in target order.
                gap_metrics = sq_gap_metrics(acc, sq_gaps)

            optim.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('Nan/Inf grad — step skipped')

            lr.step()
            i_update += 1

            metrics = {
                'Loss': ce_loss,
                'Acc': acc.mean(),
                '|Grad|': grad_norm,
                'LR': lr.val,
                'T': gen.T,
                'Upd': i_update,
                **gap_metrics,
            }
            if comm_loss_enabled:
                metrics['L_comm'] = comm_loss
                metrics['H_comm'] = comm_entropy
            metrics = to_loggable_metrics(metrics)
            curriculum_metrics = metrics
            logger.accumulate(metrics, key='slow')
            logger.accumulate(gen.get_stats(), prefix='gen', key='fast')

            state = rnn.detach_state(state)
            batch_y.clear()
            batch_y_gt.clear()
            batch_sq_gaps.clear()
            batch_comm_loss, batch_comm_entropy = 0.0, 0.0

        if has_grid and inspect_scheduler.tick(step_size):
            log_col_similarity(rnn, state, logger)
            log_lru_spectrum(rnn, logger)
            log_attn_beta(rnn, logger)

        if curriculum_metrics is not None:
            axes = curriculum.tick(metrics=curriculum_metrics, n_steps=step_size)
            if any(axes.values()):
                scale = 10
                gen.set_metaparams(
                    T=gen.T + 1.0 / scale if axes['T'] else gen.T,
                    p_store=(
                        max(gen.p_store - 0.0014 / scale, 0.10)
                        if axes['p_store'] else gen.p_store
                    ),
                    p_query=(
                        max(gen.p_query - 0.0005 / scale, 0.25)
                        if axes['p_query'] else gen.p_query
                    ),
                )
                logger.accumulate(
                    {
                        'step': curriculum.cnt_accepted,
                        'schedule': curriculum.scheduler.schedule,
                    },
                    prefix='curriculum',
                    key='fast',
                )

        logger.log(step, flush=True)

    logger.log(step, flush=True, force=True)
    logger.finish()


@torch.no_grad()
def sq_gap_metrics(acc, sq_gaps):
    def safe_mean(values, mask):
        if mask.any():
            return values[mask].mean()
        return torch.tensor(float('nan'), device=values.device)

    mask_miss = sq_gaps < 0
    mask_non_miss = ~mask_miss
    non_miss_gaps = sq_gaps[mask_non_miss]
    upper_half = (
        sq_gaps > non_miss_gaps.mean()
        if non_miss_gaps.numel() else torch.zeros_like(mask_miss)
    )
    return {
        'Acc-': safe_mean(acc, mask_miss),
        'Acc+': safe_mean(acc, mask_non_miss),
        'Acc++': safe_mean(acc, upper_half),
    }


@torch.no_grad()
@dont_throw('col_sim')
def log_col_similarity(rnn, state, logger):
    h = state['h']
    if not isinstance(h, torch.Tensor) or h.ndim != 4:
        return

    acts = h[-1].mean(dim=1)
    acts = acts / acts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = acts @ acts.T
    sim = sim[torch.ones_like(sim, dtype=torch.bool).triu(diagonal=1)]
    logger.accumulate(
        to_loggable_metrics({'max': sim.max(), 'avg': sim.mean()}),
        prefix='col_sim',
        key='fast',
    )


@torch.no_grad()
@dont_throw('LRU spectrum')
def log_lru_spectrum(rnn, logger):
    if not hasattr(rnn, 'cells') or not isinstance(rnn.cells, nn.ModuleList):
        return

    metrics = {}
    for li, row in enumerate(rnn.cells):
        for ci, cell in enumerate(row):
            lru = getattr(cell, 'lru', cell)
            if not hasattr(lru, 'nu'):
                continue
            r = torch.exp(-torch.exp(lru.nu))
            metrics[f'r_avg/L{li}_C{ci}'] = r.mean()
            metrics[f'r_min/L{li}_C{ci}'] = r.min()
            metrics[f'r_max/L{li}_C{ci}'] = r.max()

    logger.accumulate(to_loggable_metrics(metrics), prefix='lru', key='fast')


@torch.no_grad()
@dont_throw('attn beta')
def log_attn_beta(rnn, logger):
    if not hasattr(rnn, 'attn'):
        return

    metrics = {}
    for li, attn in enumerate(rnn.attn):
        beta = attn.pi_logtemp.exp()
        metrics[f'L{li}'] = beta.mean()
    logger.accumulate(to_loggable_metrics(metrics), prefix='attn_beta', key='fast')


def print_short_summary(step, *, scalars, figures, max_steps, lr):
    if 'Loss' not in scalars or 'Acc' not in scalars:
        return

    print(
        f'[{format_readable_num(step)}/{format_readable_num(max_steps, frac=0)}]'
        f' {format_readable_num(scalars["perf/fps"], frac=0)}fps |'
        f' LR:{int(100 * scalars["LR"] / lr.base_val)}%'
        f' T:{scalars["T"]:.1f} |'
        f' L:{scalars["Loss"]:.3f}'
        f' A:{scalars["Acc"]:.3f}'
        f' A-:{scalars.get("Acc-", float("nan")):.3f}'
        f' A+:{scalars.get("Acc+", float("nan")):.3f}'
        f' A++:{scalars.get("Acc++", float("nan")):.3f}'
    )


if __name__ == '__main__':
    run_experiment(runner=main)
