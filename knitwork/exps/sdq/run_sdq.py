"""Unified SDQ experiment — supports all models."""
from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint as _ckpt
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
from knitwork.visualization.cka import linear_cka
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
    'grnn_fix':       ('knitwork.models.grnn_fix',       'GridRnnFix'),
    'grnn_fix_v3':    ('knitwork.models.grnn_fix_v3',    'GridRnnFixV3'),
    'grnn_fix_v4':    ('knitwork.models.grnn_fix_v4',    'GridRnnFixV4'),
    'hgrnn_fix_v4':   ('knitwork.models.hgrnn_fix_v4',   'HopfieldGridRnnFixV4'),
    'grnn_fix_v4_8c': ('knitwork.models.grnn_fix_v4',    'GridRnnFixV4'),
    'grnn_fix_v4_6c': ('knitwork.models.grnn_fix_v4',    'GridRnnFixV4'),
    'grnn_fix_v4_12c_reg': ('knitwork.models.grnn_fix_v4', 'GridRnnFixV4'),
    'grnn_fix_v4_12c_act': ('knitwork.models.grnn_fix_v4', 'GridRnnFixV4'),
    'grnn_fix_v4_12c_reg2': ('knitwork.models.grnn_fix_v4', 'GridRnnFixV4'),
    'grnn_robust':        ('knitwork.models.grnn_robust', 'GridRnnRobust'),
    'grnn_robust_concat': ('knitwork.models.grnn_robust', 'GridRnnRobust'),
    'grnn_bal_causal':    ('knitwork.models.grnn_balance', 'GridRnnBalance'),
    'grnn_bal_lb':        ('knitwork.models.grnn_balance', 'GridRnnBalance'),
    'grnn_early':         ('knitwork.models.grnn_fix_v4',  'GridRnnFixV4'),
    'grnn_noaux':         ('knitwork.models.grnn_fix_v4',  'GridRnnFixV4'),
    'grnn_x':             ('knitwork.models.grnn_route',   'GridRnnRoute'),
    'grnn_big':           ('knitwork.models.grnn_fix_v4',  'GridRnnFixV4'),
    'grnn_big_noaux':     ('knitwork.models.grnn_fix_v4',  'GridRnnFixV4'),
    'grnn_big_lb':        ('knitwork.models.grnn_balance', 'GridRnnBalance'),
    'grnn_bal_causal2':   ('knitwork.models.grnn_balance', 'GridRnnBalance'),
    'grnn_ts_flat':       ('knitwork.models.grnn_fix_v4',  'GridRnnFixV4'),
    'grnn_ts_wide':       ('knitwork.models.grnn_fix_v4',  'GridRnnFixV4'),
    'grnn_concat':        ('knitwork.models.grnn_fix_v4',  'GridRnnFixV4'),
    'grnn_redo':          ('knitwork.models.grnn_plastic', 'GridRnnPlastic'),
    'grnn_route_lb':      ('knitwork.models.grnn_route',   'GridRnnRoute'),
    'grnn_route_topk':    ('knitwork.models.grnn_route',   'GridRnnRoute'),
    'grnn_route_noise':   ('knitwork.models.grnn_route',   'GridRnnRoute'),
    'grnn_fix_v4_2m': ('knitwork.models.grnn_fix_v4',    'GridRnnFixV4'),
    'grnn_fix_v4_10m':('knitwork.models.grnn_fix_v4',    'GridRnnFixV4'),
    'grnn_feedback':  ('knitwork.models.grnn_feedback',  'GridRnnFeedback'),
    'grnn_attn_cost': ('knitwork.models.grnn_attn_cost', 'GridRnnAttnCost'),
    'grnn_fix_v5':    ('knitwork.models.grnn_fix_v5',    'GridRnnFixV5'),
    'hgrnn_fix':      ('knitwork.models.hgrnn_fix',      'HopfieldGridRnnFix'),
    'grnn_fusion':    None,  # factory
    # config aliases
    'grnn_hgrn':      ('knitwork.models.hgrn_grnn',  'HGRN_GridRnn'),
    'grnn_lru_hop':   ('knitwork.models.hgrnn_lru',  'HopfieldGridLRU'),
    'grnn_delta':     ('knitwork.models.grnn_delta',    'GridDelta'),
    'grnn_delta_wide':('knitwork.models.grnn_delta',    'GridDelta'),
    'grnn_harmonic':  ('knitwork.models.grnn_harmonic', 'HarmonicGridRNN'),
    # external baselines
    'delta_net':      ('knitwork.models.baseline.delta_net', 'DeltaNet'),
    'hgrn2':          ('knitwork.models.baseline.hgrn2',     'HGRN2'),
    'mlstm':          ('knitwork.models.baseline.mlstm',     'mLSTM'),
    'grnn_base':      ('knitwork.models.grnn_base',          'GridRnnBase'),
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

def _supports_attn(rnn) -> bool:
    flag = getattr(rnn, '_supports_return_attn', None)
    if flag is None:
        import inspect
        flag = 'return_attn' in inspect.signature(rnn.forward).parameters
        rnn._supports_return_attn = flag
    return flag


def model_forward(rnn, x, state, *, capture: bool):
    """Normalize model forward to (logits, state, extras, aux_loss)."""
    capture = capture and _supports_attn(rnn)
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


def _ckpt_step(rnn, x, state):
    """Per-timestep gradient checkpointing: recompute step internals in backward.
    Cuts BPTT activation memory at rollout>>1 (enabled via rnn.grad_checkpoint)."""
    def fn(state_):
        y, new_state, _extras, aux = model_forward(rnn, x, state_, capture=False)
        aux_t = aux if isinstance(aux, torch.Tensor) \
            else torch.zeros((), device=y.device, dtype=y.dtype)
        return y, new_state, aux_t
    y, new_state, aux = _ckpt.checkpoint(fn, state, use_reentrant=False)
    return y, new_state, {}, aux


#  Checkpointing

def save_checkpoint(rnn, config, step, ckpt_dir: Path, rnn_type: str) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f'step_{step}.pt'
    torch.save({
        'model_state': rnn.state_dict(), 'config': config,
        'step': step, 'model_type': rnn_type,
    }, path)
    print(f'[checkpoint] saved {path}')


#  LRU spectrum + column collapse monitoring

def log_col_similarity(rnn, state, logger, step: int) -> None:
    """Log max/mean pairwise cosine similarity between column activations (last layer)."""
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


def log_col_cka(rnn, state, logger, step: int) -> None:
    """Log pairwise linear CKA between column states (last layer).

    col_sim above cosine-compares batch-*averaged* activations and stays flat while
    columns collapse: it missed a 12C run where 7 columns sat at CKA 0.51-0.83.
    CKA compares the full [B, H] representations, which is what the offline column
    analysis reports, so training and analysis finally track the same quantity.
    """
    try:
        h = state[0] if isinstance(state, tuple) else state
        if not isinstance(h, torch.Tensor) or h.ndim != 4:
            return
        acts = h[-1, :, :, :rnn.hidden_size].detach().float().cpu().numpy()  # [C, B, H]
        C = acts.shape[0]
        pairs = [linear_cka(acts[i], acts[j])
                 for i in range(C) for j in range(i + 1, C)]
        if not pairs:
            return
        pairs = np.array(pairs)
        logger.track(float(pairs.max()),  name='col_cka/max',  step=step)
        logger.track(float(pairs.mean()), name='col_cka/mean', step=step)
        # effective number of non-redundant columns
        frac = float((pairs > 0.6).mean())
        logger.track(frac, name='col_cka/frac_gt_06', step=step)
        _collapse_gate(frac, step)
    except Exception as e:
        print(f'[col_cka] {e}')


# collapse gate: warn once when redundant column pairs stay above threshold past the
# midpoint of training. The 12C baseline sat at 18/66 = 0.27 and nobody noticed for 61M
# steps because col_sim (the only detector at the time) stayed flat.
_COLLAPSE_STATE = {'warned': False, 'total': None, 'hits': 0}


def _collapse_gate(frac: float, step: int, thresh: float = 0.2, patience: int = 3) -> None:
    st = _COLLAPSE_STATE
    if st['warned'] or st['total'] is None or step < 0.5 * st['total']:
        return
    st['hits'] = st['hits'] + 1 if frac > thresh else 0
    if st['hits'] >= patience:
        st['warned'] = True
        print(
            f'[collapse] WARNING at step {step:,}: col_cka/frac_gt_06={frac:.3f} > {thresh}'
            f' for {patience} consecutive checks -- columns are collapsing.'
            f' Raise aux_act_weight / col_dropout, or check that the aux schedule is live.'
        )


def log_routing(rnn, logger, step: int) -> None:
    """Log the column->column communication graph.

    Five 12C arms were tuned without anyone looking at the routing matrix. Self-attention
    is already masked out, so the degenerate mode is a hub: every column reading the same
    source. in_max is the share of total attention the most-read column receives (1/C for
    a uniform graph), in_eff is the effective number of distinct sources.
    """
    try:
        mats = [a.last_attn for a in getattr(rnn, 'attn', []) if getattr(a, 'last_attn', None) is not None]
        if not mats:
            return
        A = torch.stack(mats).mean(dim=0).float().cpu().numpy()      # [C_q, C_k]
        C = A.shape[0]
        in_mass = A.sum(axis=0)
        share = in_mass / (in_mass.sum() + 1e-9)
        ent = float(np.exp(-(share * np.log(share + 1e-12)).sum()))
        logger.track(float(share.max()), name='route/in_max_share', step=step)
        logger.track(ent,                name='route/in_eff_cols',  step=step)
        # sanity check on the hard diagonal mask: must stay 0
        logger.track(float(np.trace(A) / C), name='route/diag_frac', step=step)
        if hasattr(rnn, 'redo_count'):
            logger.track(float(rnn.redo_count), name='route/redo_count', step=step)
    except Exception as e:
        print(f'[route] {e}')


def log_lru_spectrum(rnn, logger, step: int) -> None:
    if not hasattr(rnn, 'cells'):
        return
    try:
        for li, row in enumerate(rnn.cells):
            for ci, cell in enumerate(row):
                lru = getattr(cell, 'lru', cell)
                if not hasattr(lru, 'nu'):
                    continue
                if hasattr(lru, '_lambda_gamma'):
                    # true |lambda|, incl. retention floors (FloorLRUCell)
                    lam_re, lam_im, _ = lru._lambda_gamma()
                    r = torch.sqrt(lam_re ** 2 + lam_im ** 2).detach().float()
                else:
                    r = torch.exp(-torch.exp(lru.nu)).detach().float()
                entropy = -(r * torch.log(r + 1e-8)).sum().item()
                logger.track(float(r.mean()), name=f"lru/r_mean/L{li}_C{ci}", step=step)
                logger.track(float(r.min()),  name=f"lru/r_min/L{li}_C{ci}", step=step)
                logger.track(float(r.max()),  name=f"lru/r_max/L{li}_C{ci}", step=step)
                logger.track(entropy, name=f"lru/entropy/L{li}_C{ci}", step=step)
    except Exception as e:
        print(f'[LRU spectrum] {e}')


def log_attn_beta(rnn, logger, step: int) -> None:
    # beta evolution for models with learnable attention temperature
    if not hasattr(rnn, 'attn'):
        return
    try:
        for li, attn in enumerate(rnn.attn):
            lb = getattr(attn, 'log_beta', None)
            if lb is None:
                continue
            beta = lb.exp().detach().float()
            if beta.ndim == 2:      # [C, heads] per-column temperature
                for ci in range(beta.shape[0]):
                    logger.track(float(beta[ci].mean()), name=f"attn_beta/L{li}_C{ci}", step=step)
            else:                   # [heads]
                logger.track(float(beta.mean()), name=f"attn_beta/L{li}", step=step)
    except Exception as e:
        print(f'[attn beta] {e}')


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
    _default_name = f"knitwork_{config['model']}_sdq"
    run_name = config.get('name') or config.get('log', {}).get('name') or _default_name
    if not run_name.startswith('knitwork_'):
        run_name = 'knitwork_' + run_name
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
    # aux schedules count env-steps (like n_steps); the model's own counter cannot see
    # them and is silently frozen under gradient checkpointing, so the loop drives it
    drives_aux_clock = hasattr(rnn, 'set_env_step')
    if not drives_aux_clock and hasattr(rnn, 'aux_tick_scale'):
        rnn.aux_tick_scale = gen.n_envs
    print(f'Model on {next(rnn.parameters()).device} | dtype {next(rnn.parameters()).dtype}')

    # Feature detection
    has_diversity  = hasattr(rnn, 'compute_diversity_loss')
    has_act_loss   = hasattr(rnn, 'act_loss_weight')
    has_hgrn_betas = hasattr(rnn, 'get_hgrn_betas')
    has_reservoir  = hasattr(rnn, 'get_reservoir_spectral_radii')
    has_lru        = hasattr(rnn, 'lru_r_per_col') or hasattr(rnn, 'attn')
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
    _COLLAPSE_STATE.update(total=n_steps, warned=False, hits=0)

    log_stats_schedule = create_scheduler(config['log']['schedule'])
    print_stats_schedule = create_scheduler(config['log']['print_schedule'])
    curriculum_step = CurriculumScheduler(**config['curriculum'])

    logger = create_logger(config)
    stats       = Tracker(lr=2e-4)
    fps_counter = FpsCounter()

    # checkpointing (disabled by default)
    ckpt_cfg      = config.get('checkpoint', {})
    ckpt_enabled  = ckpt_cfg.get('enabled', False)
    ckpt_dir      = Path(ckpt_cfg.get('save_dir', 'runs/checkpoints')) / run_name
    ckpt_save_at  = sorted(int(float(s)) for s in ckpt_cfg.get('save_at', []))
    ckpt_every    = int(float(ckpt_cfg.get('save_every', 0)))
    ckpt_saved: set = set()
    ckpt_next     = ckpt_every if ckpt_every > 0 else None

    has_grid    = hasattr(rnn, 'n_layers') and hasattr(rnn, 'n_columns')
    vis_interval = int(config.get('vis_interval', 10_000_000))
    viz = VizManager(rnn.n_layers, rnn.n_columns, vis_interval=vis_interval) if (vis_enabled and has_grid) else None

    rnn_state     = None
    batch_y:       list = []
    batch_y_gt:    list = []
    batch_sq_gaps: list = []
    batch_kl:      list = []
    batch_div:     list = []
    batch_harmonic: list = []   # harmonic model diagnostics

    aux_iter = 0
    while step < n_steps:
        if drives_aux_clock:
            # aux_on must be constant across a whole rollout: backward recomputes all
            # rollout_len checkpointed steps at once, and if the flag flipped in between,
            # the recompute saves a different number of tensors than the original pass
            rnn.set_env_step(step, aux_on=(
                (aux_iter // rollout_len) % max(rnn.aux_every, 1) == 0
            ))
            aux_iter += 1
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}
        rnn_state = rnn.reset_state(rnn_state, obs['reset_mask'])
        x = obs['tokens'].view(-1, 1)

        capture = vis_enabled and viz is not None and (step >= viz.next_step - gen.n_envs)
        need_extras = capture or has_diversity or has_act_loss or has_harmonic

        if getattr(rnn, 'grad_checkpoint', False) and rnn.training and not need_extras:
            y, rnn_state, extras, kl = _ckpt_step(rnn, x, rnn_state)
        else:
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
            # specialisation update: use sq_gaps to determine dominant phase
            sq = obs['sq_gaps'].float()
            n_s = (sq < -1.0).sum().item()
            n_q = (sq > 0.0).sum().item()
            n_d = len(sq) - n_s - n_q
            phase = max([('store', n_s), ('query', n_q), ('distract', n_d)], key=lambda x: x[1])[0]
            viz.update_specialisation(phase, rnn_state, extras.get('attn_weights'))

        batch_y.append(y)
        batch_y_gt.append(obs['targets'])
        batch_sq_gaps.append(obs['sq_gaps'])
        batch_kl.append(kl if kl is not None else torch.tensor(0.0, device=device, dtype=dtype))

        # Visualization flush
        if vis_enabled and viz is not None and step >= viz.next_step and logger is not None:
            viz.flush(logger, step, has_hgrn=has_hgrn_betas, has_reservoir=has_reservoir,
                      reservoir_sr_info=reservoir_sr_info)

        step += gen.n_envs

        if ckpt_enabled:
            due = [s for s in ckpt_save_at if s <= step and s not in ckpt_saved]
            if ckpt_next is not None and step >= ckpt_next:
                due.append(step)
                ckpt_next += ckpt_every
            for s in due:
                save_checkpoint(rnn, config, step, ckpt_dir, rnn_type)
                ckpt_saved.add(s)

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

            # after the optimizer step, never mid-forward: revival rewrites weights
            if hasattr(rnn, 'apply_redo'):
                rnn.apply_redo(optim)

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
                log_attn_beta(rnn, logger, step)
            has_col_state = hasattr(rnn, 'n_columns') and hasattr(rnn, 'hidden_size')
            if has_col_state and logger is not None and step % (batch_size * 100) == 0:
                log_col_similarity(rnn, rnn_state, logger, step)
                log_col_cka(rnn, rnn_state, logger, step)
                log_routing(rnn, logger, step)

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
