"""Unified text experiment — supports all models."""
from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from knitwork.common.config import extracted
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
    'grnn_fusion':    None,  # factory
    # config aliases
    'grnn_res':       ('knitwork.models.grnn_reservoir', 'GridRnnReservoir'),
}


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


# Forward normalizer

def model_forward(rnn, x, state, *, capture: bool):
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


# Intra-word accuracy

def update_intra_word_metrics(
    acc_by_pos: dict[int, float],
    tokens: torch.Tensor,   # [B]
    acc:    torch.Tensor,   # [B]
    lr: float = 0.01,
    space_idx: int = 0,
) -> None:
    pos = 0
    for t, a in zip(tokens.tolist(), acc.tolist()):
        if t == space_idx:
            pos = 0
        else:
            old = acc_by_pos.get(pos, a)
            acc_by_pos[pos] = old + lr * (a - old)
            pos += 1


def main(config):
    run_name = (
        config.get('name')
        or config.get('log', {}).get('name')
        or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    config.setdefault('log', {})['name'] = run_name
    print(f'Run name: {run_name}')

    vis_enabled = config.get('visualize', True)
    rng    = np.random.default_rng(config['seed'])
    device = get_device(config.get('device'))
    dtype  = get_dtype(config.get('dtype'))
    n_envs = config['n_envs']

    gen_cfg   = config['gens'][config['gen']]
    data_path = Path(gen_cfg['path']).expanduser()
    data, charset = tokenize(load_dataset(data_path))
    n_chars = charset.size

    gen = TextGenerator(data, n_envs=n_envs, ignore_index=CE_ignore_index,
                        seed=rng.integers(1_000_000))

    rnn_type = config['model']
    rnn_cfg  = config['models'][rnn_type]
    rnn = build_model(rnn_type, rnn_cfg, n_chars)
    rnn = rnn.to(device=device, dtype=dtype)
    print(f'Model on {next(rnn.parameters()).device} | dtype {next(rnn.parameters()).dtype}')

    use_vae = getattr(rnn, 'use_vae', False)
    has_grid = hasattr(rnn, 'n_layers') and hasattr(rnn, 'n_columns')

    # Visualizers
    attn_vis = AttnFlowVisualizer(
        n_layers=rnn.n_layers, n_columns=rnn.n_columns, buffer_size=100
    ) if has_grid and vis_enabled else None
    cka_vis = CKAVisualizer(
        n_layers=rnn.n_layers, n_columns=rnn.n_columns, buffer_size=50
    ) if has_grid and vis_enabled else None
    next_vis_step = VIS_INTERVAL
    gate_buffer: list = []

    # KL annealing
    kl_cfg   = config.get('kl_anneal', {})
    kl_steps = int(kl_cfg.get('steps', 50_000))
    kl_max   = float(kl_cfg.get('max_weight', 1.0))
    kl_anneal = lambda step: kl_max if kl_steps == 0 else kl_max * min(1.0, step / kl_steps)

    # LR
    lr_cfg = config['lr']
    lr     = lr_cfg['val']
    wm_lr_cfg, wm_lr_schedule = extracted(lr_cfg['warmup'], 'schedule')
    dc_lr_cfg, dc_lr_schedule = extracted(lr_cfg['decay'],  'schedule')
    wm_lr = DynamicParameter(val=1e-5*lr, tar=lr, **wm_lr_cfg, scheduler=Scheduler(wm_lr_schedule))
    dc_lr = DynamicParameter(val=lr, **dc_lr_cfg, scheduler=Scheduler(dc_lr_schedule))

    def get_lr():
        return wm_lr.val if not wm_lr.scheduler.is_infinite else dc_lr.val

    def step_lr():
        return wm_lr.step() if not wm_lr.scheduler.is_infinite else dc_lr.step()

    optim   = torch.optim.RMSprop(rnn.parameters(), lr=get_lr())
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)

    # p_reset schedule
    p_reset_cfg, p_reset_schedule = extracted(gen_cfg['reset_prob'], 'schedule')
    p_reset = DynamicParameter(**p_reset_cfg, scheduler=Scheduler(int(p_reset_schedule)))

    rollout_len = config['rollout_len']
    batch_size  = gen.n_envs * rollout_len
    n_steps     = int(config['n_steps'])
    step        = 0

    log_stats_schedule   = Scheduler(int(config['log']['schedule']))
    print_stats_schedule = Scheduler(int(config['log']['print_schedule']))

    logger = create_logger(config)
    stats       = Tracker(lr=2e-4)
    fps_counter = FpsCounter()

    ln_2 = np.log(2.0)
    rnn_state   = None
    batch_y:    list = []
    batch_y_gt: list = []
    batch_kl:   list = []
    acc_by_pos: dict[int, float] = {}

    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}

        rnd_reset  = torch.from_numpy(rng.random(gen.n_envs) < p_reset.val).to(device)
        reset_mask = torch.logical_or(obs['reset_mask'], rnd_reset)
        rnn_state  = rnn.reset_state(rnn_state, reset_mask)
        x = obs['tokens'].view(-1, 1)

        capture = vis_enabled and (step >= next_vis_step - gen.n_envs)
        y, rnn_state, extras, kl = model_forward(rnn, x, rnn_state, capture=capture)

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
            y_cat    = torch.cat(batch_y,    dim=0)
            y_gt_cat = torch.cat(batch_y_gt, dim=0)
            m_active = y_gt_cat != CE_ignore_index

            ce_loss    = loss_fn(y_cat, y_gt_cat)
            kl_mean    = torch.stack(batch_kl).mean()
            kl_scale   = kl_anneal(step)
            total_loss = ce_loss + kl_scale * kl_mean

            with torch.no_grad():
                logits_a = y_cat[m_active]
                gt_a     = y_gt_cat[m_active]
                acc      = (logits_a.argmax(dim=-1) == gt_a).float()
                bpc        = ce_loss / ln_2
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
            if step_lr():
                optim.param_groups[0]['lr'] = get_lr()

            stat_dict = {
                'Loss':       to_numpy(ce_loss,     copy=False),
                'BPC':        to_numpy(bpc,         copy=False),
                'Perplexity': to_numpy(perplexity,  copy=False),
                'Acc':        to_numpy(acc.mean(),  copy=False),
                '|Grad|':     to_numpy(grad_norm,   copy=False),
                'LR':         get_lr(),
                'T':          1.0 / p_reset.val,
            }
            if use_vae:
                stat_dict['KL']       = to_numpy(kl_mean, copy=False)
                stat_dict['KL_scale'] = kl_scale
            for pos, val in list(acc_by_pos.items())[:4]:
                stat_dict[f'Acc[{pos}]'] = val
            stats.put(stat_dict)

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear(); batch_y_gt.clear(); batch_kl.clear()

        if print_stats_schedule.tick(gen.n_envs):
            m   = {'global_step': step} | stats.get()
            fps = fps_counter.fps(n_iters=step, start=True)
            kl_s = f' KL:{m.get("KL", 0):.2e}(x{m.get("KL_scale", 0):.2f}) |' if use_vae else ''
            print(
                f'[{format_readable_num(step)}/{format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)}fps |'
                f' LR:{int(100*m["LR"]/lr)}%'
                f' T:{int(m["T"])} |{kl_s}'
                f' L:{m["Loss"]:.3f}'
                f' BPC:{m["BPC"]:.3f}'
                f' A:{m["Acc"]:.3f}'
            )

        if log_stats_schedule.tick(gen.n_envs) and logger is not None:
            fps = fps_counter.fps(n_iters=step, start=True)
            metrics = {'global_step': step, 'fps': fps} | stats.get()
            metrics['gen'] = gen.get_stats()
            logger.track(flatten_dict(metrics))

    fps = fps_counter.fps(n_iters=step)
    print(f'Done. {format_readable_num(fps)} fps')


if __name__ == '__main__':
    run_experiment(runner=main)
