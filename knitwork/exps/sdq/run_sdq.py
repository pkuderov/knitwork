"""Unified SDQ experiment — supports all models."""
from __future__ import annotations

import importlib
from datetime import datetime

import numpy as np
import torch
from torch import nn

from knitwork.common.curriculum import CurriculumScheduler
from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging import create_logger
from knitwork.common.scheduler import create_scheduler
from knitwork.common.torch import DynamicLearningRate
from knitwork.common.tracker import Tracker
from knitwork.common.status import write_status
from knitwork.common.utils import (
    CE_ignore_index, FpsCounter, flatten_dict,
    format_readable_num, get_device, get_dtype,
    to_numpy, to_torch,
)
from knitwork.gens.sdq import StoreDistractQueryGenerator
from knitwork.exps.sdq._viz import VizManager


#  Model registry 

_REGISTRY: dict[str, tuple[str, str] | None] = {
    'rnn':            ('knitwork.models.gru',           'GruBaseline'),
    'grnn':           ('knitwork.models.grnn',           'GridRnn'),
    'grnn_err':       ('knitwork.models.grnn_err',       'GridRnn'),
    'grnn2':          ('knitwork.models.grnn2',          'GridRnn2'),
    'grnn_eq':        ('knitwork.models.grnn_eq',        'EquilibriumGridRnnCoT'),
    'grnn_eq1':       ('knitwork.models.grnn_eq1',       'EquilibriumGridRnnCoT'),
    'grnn_lru':       ('knitwork.models.grnn_lru',       'GridLRU'),
    'grnn_lru_wide':  ('knitwork.models.grnn_lru',       'GridLRU'),
    'hgrnn':          ('knitwork.models.hgrnn',          'HopfieldGridRnn'),
    'hgrnn_lru':      ('knitwork.models.hgrnn_lru',      'HopfieldGridLRU'),
    'hgrn_grnn':      ('knitwork.models.hgrn_grnn',      'HGRN_GridRnn'),
    'grnn_fw':        ('knitwork.models.grnn_fw',        'GridRnnFW'),
    'grnn_reservoir': ('knitwork.models.grnn_reservoir', 'GridRnnReservoir'),
    'grnn_loss':      ('knitwork.models.grnn_loss',      'GridRnnLoss'),
    'grnn_disc':      ('knitwork.models.grnn_disc',      'GridRnnNoveltyGate'),
    'grnn_adv_loss':  ('knitwork.models.grnn_adv_loss',  'GridRnn'),
    'grnn_engram':    ('knitwork.models.engram_grnn',    'EngramGridRnn'),
    'grnn_prec_delta': ('knitwork.models.grnn_prec_delta', 'GridRnnPrecDelta'),
    'grnn_ema_mem':   ('knitwork.models.grnn_ema_mem',   'GridRnnEmaMem'),
    'grnn_fusion':    None,  # factory
    # config aliases
    'grnn_hgrn':      ('knitwork.models.hgrn_grnn',  'HGRN_GridRnn'),
    'grnn_lru_hop':   ('knitwork.models.hgrnn_lru',  'HopfieldGridLRU'),
    'grnn_delta':     ('knitwork.models.grnn_delta',    'GridDelta'),
    'grnn_delta_wide':('knitwork.models.grnn_delta',    'GridDelta'),
    'grnn_harmonic':  ('knitwork.models.grnn_harmonic', 'HarmonicGridRNN'),
}


def build_model(rnn_type: str, rnn_cfg: dict, gen):
    if rnn_type == 'grnn_fusion':
        from knitwork.models.grnn_fusion import build_fusion_from_config
        return build_fusion_from_config(rnn_cfg, gen.n_tokens, gen.V)
    entry = _REGISTRY.get(rnn_type)
    if entry is None:
        raise ValueError(f'Unknown model type: {rnn_type!r}')
    mod_path, cls_name = entry
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(**rnn_cfg, input_size=gen.n_tokens, output_size=gen.V)


#  Forward normalizer 

def model_forward(rnn, x, state, *, capture: bool):
    """Normalize model forward to (logits, state, extras, aux_loss)."""
    if capture:
        result = rnn(x, state, return_attn=True)
    else:
        result = rnn(x, state)
    y     = result[0]
    state = result[1]
    # grnn2 returns (y, h, kl) or (y, h, extras, kl)
    if len(result) == 2:
        return y, state, {}, None
    if len(result) == 3:
        third = result[2]
        if isinstance(third, dict):
            return y, state, third, None
        return y, state, {}, third  # kl_loss
    if len(result) == 4:
        return y, state, result[2], result[3]
    return y, state, {}, None


#  LRU spectrum logging 

def log_lru_spectrum(rnn, logger, step: int) -> None:
    if not hasattr(rnn, 'cells'):
        return
    try:
        for li, row in enumerate(rnn.cells):
            for ci, cell in enumerate(row):
                lru = getattr(cell, 'lru', cell)
                if not hasattr(lru, 'nu'):
                    continue
                r = torch.exp(-torch.exp(lru.nu)).detach().float()
                entropy = -(r * torch.log(r + 1e-8)).sum().item()
                logger.track(float(r.mean()), name=f"lru/r_mean/L{li}_C{ci}", step=step)
                logger.track(entropy, name=f"lru/entropy/L{li}_C{ci}", step=step)
    except Exception as e:
        print(f'[LRU spectrum] {e}')


#  KL annealing 

def make_kl_anneal(cfg: dict):
    steps = int(cfg.get('steps', 50_000))
    max_w = float(cfg.get('max_weight', 1.0))
    def get(step: int) -> float:
        if steps == 0:
            return max_w
        return max_w * min(1.0, step / steps)
    return get


#  Acc metrics from sq_gaps

def sq_gap_metrics(acc: torch.Tensor, sq_gaps: torch.Tensor) -> dict:
    def safe_mean(t, mask):
        return t[mask].mean() if mask.any() else torch.tensor(float('nan'))
    mask_store    = sq_gaps < -1.0
    mask_query    = sq_gaps > 0.0
    mask_distract = ~mask_store & ~mask_query
    mask_miss     = sq_gaps < 0.0
    sq_non_miss   = sq_gaps[~mask_miss]
    acc_up_half   = acc[sq_gaps > sq_non_miss.mean()] if sq_non_miss.numel() > 0 else acc
    return {
        'Acc/store':    safe_mean(acc, mask_store),
        'Acc/query':    safe_mean(acc, mask_query),
        'Acc/distract': safe_mean(acc, mask_distract),
        'Acc-':         safe_mean(acc, mask_miss),
        'Acc+':         safe_mean(acc, ~mask_miss),
        'Acc++':        acc_up_half.mean(),
    }


#  Main 

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

    gen_cfg = config['gens'][config['gen']]
    gen = StoreDistractQueryGenerator(
        **gen_cfg, n_envs=n_envs,
        seed=rng.integers(1_000_000),
        ignore_index=CE_ignore_index,
    )

    rnn_type = config['model']
    rnn_cfg  = config['models'][rnn_type]
    rnn = build_model(rnn_type, rnn_cfg, gen)
    rnn = rnn.to(device=device, dtype=dtype)
    print(f'Model on {next(rnn.parameters()).device} | dtype {next(rnn.parameters()).dtype}')

    # Feature detection
    has_diversity  = hasattr(rnn, 'compute_diversity_loss')
    has_act_loss   = hasattr(rnn, 'act_loss_weight')
    has_hgrn_betas = hasattr(rnn, 'get_hgrn_betas')
    has_reservoir  = hasattr(rnn, 'get_reservoir_spectral_radii')
    has_lru        = hasattr(rnn, 'lru_r_per_col')
    has_harmonic   = hasattr(rnn, 'mem_layers') and hasattr(rnn, 'flatten_extras_stats')
    use_vae        = getattr(rnn, 'use_vae', False)
    is_fusion      = rnn_type == 'grnn_fusion'

    reservoir_sr_info: dict = {}
    if has_reservoir:
        reservoir_sr_info = rnn.get_reservoir_spectral_radii()
        print(f'Reservoir SR: {reservoir_sr_info}')

    kl_anneal = make_kl_anneal(config.get('kl_anneal', {}))

    lr = DynamicLearningRate(name=f'LR', **config['lr'])
    optim = torch.optim.RMSprop(rnn.parameters(), lr=lr.val)
    lr.connect_to_optimiser(optim)

    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)

    rollout_len = config['rollout_len']
    batch_size  = gen.n_envs * rollout_len
    n_steps     = int(config['n_steps'])
    step        = 0

    log_stats_schedule = create_scheduler(config['log']['schedule'])
    print_stats_schedule = create_scheduler(config['log']['print_schedule'])
    curriculum_step = CurriculumScheduler(**config['curriculum'])

    logger = create_logger(config)
    stats       = Tracker(lr=2e-4)
    fps_counter = FpsCounter()

    has_grid = hasattr(rnn, 'n_layers') and hasattr(rnn, 'n_columns')
    viz = VizManager(rnn.n_layers, rnn.n_columns) if (vis_enabled and has_grid) else None

    rnn_state     = None
    batch_y:       list = []
    batch_y_gt:    list = []
    batch_sq_gaps: list = []
    batch_kl:      list = []
    batch_div:     list = []
    batch_harmonic: list = []   # harmonic model diagnostics

    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}
        rnn_state = rnn.reset_state(rnn_state, obs['reset_mask'])
        x = obs['tokens'].view(-1, 1)

        capture = vis_enabled and viz is not None and (step >= viz.next_step - gen.n_envs)
        need_extras = capture or has_diversity or has_act_loss or has_harmonic

        y, rnn_state, extras, kl = model_forward(rnn, x, rnn_state, capture=need_extras)

        # Diversity loss (detached for buffer)
        if has_diversity and extras:
            div = rnn.compute_diversity_loss(extras)
            batch_div.append({k: v.detach() for k, v in div.items()})

        # Harmonic model diagnostics (lightweight scalar dicts, no tensors)
        if has_harmonic and extras:
            batch_harmonic.append(rnn.flatten_extras_stats(extras))

        if capture and viz is not None:
            viz.update(step, extras, rnn_state, has_hgrn=has_hgrn_betas, has_fusion=is_fusion, rnn=rnn)

        batch_y.append(y)
        batch_y_gt.append(obs['targets'])
        batch_sq_gaps.append(obs['sq_gaps'])
        batch_kl.append(kl if kl is not None else torch.tensor(0.0, device=device, dtype=dtype))

        # Visualization flush
        if vis_enabled and viz is not None and step >= viz.next_step and logger is not None:
            viz.flush(logger, step, has_hgrn=has_hgrn_betas, has_reservoir=has_reservoir,
                      reservoir_sr_info=reservoir_sr_info)

        step += gen.n_envs

        if step % batch_size == 0:
            y_cat     = torch.cat(batch_y,      dim=0)
            y_gt_cat  = torch.cat(batch_y_gt,   dim=0)
            sq_gaps   = torch.cat(batch_sq_gaps, dim=0).float()
            m_active  = y_gt_cat != CE_ignore_index

            ce_loss = loss_fn(y_cat, y_gt_cat)

            # Aux losses
            kl_mean    = torch.stack(batch_kl).mean()
            kl_scale   = kl_anneal(step)
            total_loss = ce_loss + kl_scale * kl_mean

            div_mean: dict = {}
            if batch_div:
                for key in batch_div[0]:
                    div_mean[key] = torch.stack([d[key] for d in batch_div]).mean()
                total_loss = total_loss + div_mean.get('total', torch.tensor(0.0, device=device))
                if viz is not None:
                    viz.update_div(step, {k: v.item() for k, v in div_mean.items()})

            if has_act_loss and extras.get('act_iters'):
                total_loss = total_loss + rnn.act_loss(extras['act_iters'])

            with torch.no_grad():
                acc = (y_cat[m_active].argmax(dim=-1) == y_gt_cat[m_active]).float()
                gap_metrics = sq_gap_metrics(acc, sq_gaps[:acc.shape[0]] if sq_gaps.shape[0] != acc.shape[0] else sq_gaps)

            optim.zero_grad()
            total_loss.backward()

            if viz is not None:
                grad_norms = viz.update_grad_norms(step, rnn)
            else:
                grad_norms = {}

            grad_norm = nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('Nan/Inf grad — step skipped')

            lr.step()

            stat_dict = {
                'Loss':   to_numpy(ce_loss,    copy=False),
                'Acc':    to_numpy(acc.mean(), copy=False),
                '|Grad|': to_numpy(grad_norm,  copy=False),
                'LR':     lr.val,
                **{k: to_numpy(v, copy=False) for k, v in gap_metrics.items()},
            }
            if use_vae:
                stat_dict['KL']       = to_numpy(kl_mean, copy=False)
                stat_dict['KL_scale'] = kl_scale
            if div_mean:
                stat_dict.update({f'div/{k}': v.item() for k, v in div_mean.items()})
            stat_dict.update(grad_norms)
            if has_hgrn_betas:
                try:
                    betas = rnn.get_hgrn_betas()
                    for li in range(rnn.n_layers):
                        lb = [v for k, v in betas.items() if f'L{li}_' in k]
                        if lb:
                            stat_dict[f'hgrn/beta_mean/L{li}'] = float(np.mean(lb))
                except Exception:
                    pass
            if has_reservoir:
                try:
                    res_util = rnn.get_reservoir_utilization(rnn_state)
                    stat_dict.update(res_util)
                except Exception:
                    pass
            if has_act_loss and extras.get('act_iters'):
                for li, iters in enumerate(extras['act_iters']):
                    stat_dict[f'eq/iters/L{li}'] = float(iters.float().mean())
            if has_harmonic and batch_harmonic:
                # average harmonic diagnostics over rollout steps
                keys = batch_harmonic[0].keys()
                for k in keys:
                    vals = [d[k] for d in batch_harmonic if k in d]
                    if vals:
                        stat_dict[k] = float(np.mean(vals))
                batch_harmonic.clear()
            stats.put(stat_dict)

            if has_lru and logger is not None and step % (batch_size * 100) == 0:
                log_lru_spectrum(rnn, logger, step)

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear(); batch_y_gt.clear(); batch_sq_gaps.clear()
            batch_kl.clear(); batch_div.clear()

        axes = curriculum_step.tick(metrics=stats, n_steps=gen.n_envs)
        if any(axes.values()):
            K = 10
            gen.set_metaparams(
                T       = gen.T       + 1.0 / K        if axes['T']       else gen.T,
                p_store = max(gen.p_store - 0.0014 / K, 0.10) if axes['p_store'] else gen.p_store,
                p_query = max(gen.p_query - 0.0005 / K, 0.25) if axes['p_query'] else gen.p_query,
            )

        if print_stats_schedule.tick(gen.n_envs):
            m   = {'global_step': step} | stats.get()
            fps = fps_counter.fps(n_iters=step, start=True)
            kl_s = f' KL:{m.get("KL", 0):.2e}(x{m.get("KL_scale", 0):.2f}) |' if use_vae else ''
            print(
                f'[{format_readable_num(step)}/{format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)}fps |'
                f' LR:{int(100*m["LR"]/lr.base_val)}% |{kl_s}'
                f' L:{m["Loss"]:.3f}'
                f' A:{m["Acc"]:.3f}'
                f' Aq:{m.get("Acc/query", float("nan")):.3f}'
                f' As:{m.get("Acc/store", float("nan")):.3f}'
                f' A++:{m.get("Acc++", float("nan")):.3f}'
            )

        if log_stats_schedule.tick(gen.n_envs):
            fps = fps_counter.fps(n_iters=step, start=True)
            metrics = {
                'global_step':   step,
                'fps':           fps,
                'curr_step':     curriculum_step.cnt_accepted,
                'curr_schedule': curriculum_step.scheduler.schedule,
            } | stats.get()
            metrics['gen'] = gen.get_stats()
            for k, v in reservoir_sr_info.items():
                metrics[k] = v
            write_status(step, metrics)
            if logger is not None:
                logger.track(flatten_dict(metrics))

    fps = fps_counter.fps(n_iters=step)
    print(f'Done. {format_readable_num(fps)} fps')


if __name__ == '__main__':
    run_experiment(runner=main)
