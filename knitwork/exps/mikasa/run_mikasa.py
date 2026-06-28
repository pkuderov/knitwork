from __future__ import annotations

import importlib
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging import create_logger
from knitwork.common.scheduler import create_scheduler
from knitwork.common.torch import DynamicLearningRate
from knitwork.common.tracker import Tracker
from knitwork.common.utils import (
    FpsCounter, flatten_dict,
    format_readable_num, get_device, get_dtype,
    to_numpy, to_torch,
)
from knitwork.env.mikasa_wrapper import MikasakWrapper
from knitwork.exps.sdq._viz import VizManager

# ---------------------------------------------------------------------------
# Model registries

# For Discrete observation spaces — pass tokens directly to existing models
_REGISTRY_DISCRETE: dict[str, tuple[str, str]] = {
    'rnn':            ('knitwork.models.gru',           'GruBaseline'),
    'grnn':           ('knitwork.models.grnn',           'GridRnn'),
    'grnn_lru':       ('knitwork.models.grnn_lru',       'GridLRU'),
    'grnn_lru_wide':  ('knitwork.models.grnn_lru',       'GridLRU'),
    'hgrnn':          ('knitwork.models.hgrnn',          'HopfieldGridRnn'),
    'hgrnn_lru':      ('knitwork.models.hgrnn_lru',      'HopfieldGridLRU'),
    'hgrn_grnn':      ('knitwork.models.hgrn_grnn',      'HGRN_GridRnn'),
    'grnn_fw':        ('knitwork.models.grnn_fw',        'GridRnnFW'),
    'grnn_reservoir': ('knitwork.models.grnn_reservoir', 'GridRnnReservoir'),
    'grnn_loss':      ('knitwork.models.grnn_loss',      'GridRnnLoss'),
    'grnn_engram':    ('knitwork.models.engram_grnn',    'EngramGridRnn'),
    'grnn_prec_delta':('knitwork.models.grnn_prec_delta','GridRnnPrecDelta'),
    'grnn_ema_mem':   ('knitwork.models.grnn_ema_mem',   'GridRnnEmaMem'),
    'grnn_delta':     ('knitwork.models.grnn_delta',     'GridDelta'),
    'grnn_harmonic':  ('knitwork.models.grnn_harmonic',  'HarmonicGridRNN'),
    # external baselines
    'delta_net':      ('knitwork.models.baseline.delta_net', 'DeltaNet'),
    'hgrn2':          ('knitwork.models.baseline.hgrn2',     'HGRN2'),
    'mlstm':          ('knitwork.models.baseline.mlstm',     'mLSTM'),
}

# For MultiDiscrete / Box / Tuple observation spaces — linear encoder replaces Embedding
_REGISTRY_CONTINUOUS: dict[str, tuple[str, str]] = {
    'rnn':       ('knitwork.models.rl_wrappers', 'GruBaselineContinuous'),
    'grnn':      ('knitwork.models.rl_wrappers', 'GridRnnContinuous'),
    'grnn_lru':  ('knitwork.models.rl_wrappers', 'GridLRUContinuous'),
    'hgrnn':     ('knitwork.models.rl_wrappers', 'HopfieldGridRnnContinuous'),
    'hgrnn_lru': ('knitwork.models.rl_wrappers', 'HopfieldGridLRUContinuous'),
}


def build_model(
    model_type: str,
    model_cfg: dict,
    obs_type: str,
    n_tokens: int,
    obs_dim: int,
    n_actions: int,
):
    if obs_type == 'discrete':
        registry = _REGISTRY_DISCRETE
        extra = {'input_size': n_tokens, 'output_size': n_actions}
    else:
        registry = _REGISTRY_CONTINUOUS
        extra = {'obs_dim': obs_dim, 'output_size': n_actions}

    entry = registry.get(model_type)
    if entry is None:
        raise ValueError(
            f'Model {model_type!r} not available for obs_type={obs_type!r}. '
            f'Available: {sorted(registry)}'
        )
    mod_path, cls_name = entry
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(**model_cfg, **extra)


# ---------------------------------------------------------------------------
# PPO constants

GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
VALUE_COEF   = 0.5
MAX_GRAD_NORM = 0.5
PPO_EPOCHS   = 4


def compute_gae(
    rewards:    torch.Tensor,  # [T, B]
    values:     torch.Tensor,  # [T+1, B]
    dones:      torch.Tensor,  # [T, B]  terminated | truncated — resets GAE carry-over
    terminated: torch.Tensor,  # [T, B]  true terminal only — zeros bootstrap
) -> tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    advs = torch.zeros_like(rewards)
    gae  = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(T)):
        mask_cont = 1.0 - dones[t]        # 0 on any done — stops GAE carry-over
        mask_boot = 1.0 - terminated[t]   # 0 only on true termination — zeros V(s_next)
        delta   = rewards[t] + GAMMA * values[t + 1] * mask_boot - values[t]
        gae     = delta + GAMMA * GAE_LAMBDA * mask_cont * gae
        advs[t] = gae
    returns = advs + values[:T]
    return advs, returns


# ---------------------------------------------------------------------------
# Main

def main(config):
    env_id = config['env']
    _env_slug = env_id.replace('popgym-', '').replace('-v0', '').replace('-', '_')
    _default_name = f"knitwork_{config['model']}_{_env_slug}"
    run_name = config.get('name') or config.get('log', {}).get('name') or _default_name
    if not run_name.startswith('knitwork_'):
        run_name = 'knitwork_' + run_name
    config.setdefault('log', {})['name'] = run_name
    print(f'Run name: {run_name}')

    rng          = np.random.default_rng(config['seed'])
    device       = get_device(config.get('device', None))
    dtype        = get_dtype(config.get('dtype', None))
    n_envs       = config['n_envs']
    entropy_coef = config.get('entropy_coef', 0.05)

    # env
    env_id    = config['env']
    async_envs = config.get('async_envs', True)
    gen = MikasakWrapper(
        env_id=env_id,
        n_envs=n_envs,
        seed=int(rng.integers(1_000_000)),
        async_envs=async_envs,
    )
    print(
        f'Env: {env_id}'
        f'  obs_type={gen.obs_type}'
        f'  obs_dim={gen.obs_dim}'
        f'  n_tokens={gen.n_tokens}'
        f'  n_actions={gen.n_actions}'
    )

    # model
    model_type = config['model']
    model_cfg  = config['models'][model_type]
    rnn = build_model(
        model_type=model_type,
        model_cfg=model_cfg,
        obs_type=gen.obs_type,
        n_tokens=gen.n_tokens,
        obs_dim=gen.obs_dim,
        n_actions=gen.n_actions,
    )
    rnn = rnn.to(device=device, dtype=dtype)

    actor_hidden = rnn.hidden_size
    critic = nn.Linear(actor_hidden, 1).to(device=device, dtype=dtype)

    def extract_h_top(state) -> torch.Tensor:
        # normalize heterogeneous state shapes to [B, actor_hidden]
        if hasattr(rnn, 'get_top_h'):
            return rnn.get_top_h(state)[:, :actor_hidden]
        h = state[0] if isinstance(state, tuple) else state
        if h.ndim == 3:
            return h[-1][:, :actor_hidden]
        return h[-1, 0, :, :actor_hidden]

    # Aux observation-reconstruction head for a specific column (anti-collapse regulariser)
    aux_col_idx = int(config.get('aux_col_idx', 2))
    aux_loss_w  = float(config.get('aux_col_weight', 0.0))
    _n_cols     = getattr(rnn, 'n_columns', 0)
    aux_enabled = aux_loss_w > 0 and _n_cols > aux_col_idx
    if aux_enabled:
        _obs_out = gen.n_tokens if gen.obs_type == 'discrete' else gen.obs_dim
        aux_head = nn.Linear(actor_hidden, _obs_out).to(device=device, dtype=dtype)
        print(f'Aux col loss enabled: col={aux_col_idx}  weight={aux_loss_w}  out={_obs_out}')
    else:
        aux_head = None

    all_params = (
        list(rnn.parameters()) + list(critic.parameters())
        + (list(aux_head.parameters()) if aux_head is not None else [])
    )
    lr = DynamicLearningRate(name=f'LR', **config['lr'])
    optim = torch.optim.RMSprop(all_params, lr=lr.val, eps=1e-5)
    lr.connect_to_optimiser(optim)

    rollout_len  = config['rollout_len']
    n_steps      = int(config['n_steps'])
    vis_interval = int(config.get('vis_interval', 10_000_000))
    step         = 0

    log_stats_schedule   = create_scheduler(config['log']['schedule'])
    print_stats_schedule = create_scheduler(config['log']['print_schedule'])

    logger = create_logger(config)
    stats       = Tracker(lr=2e-4)
    fps_counter = FpsCounter()

    rnn_state   = None
    is_discrete = gen.obs_type == 'discrete'
    is_grid     = hasattr(rnn, 'cells')  # Grid-type model with per-column cells

    def _obs_to_model_input(obs_np: np.ndarray) -> torch.Tensor:
        t = to_torch(obs_np, device=device)
        if is_discrete:
            return t.view(-1, 1)   # [B, 1] int64
        return t.to(dtype)         # [B, obs_dim] float

    # Pre-allocate rollout buffers to avoid repeated torch.stack / alloc
    _obs_shape = (rollout_len, n_envs, 1) if is_discrete else (rollout_len, n_envs, gen.obs_dim)
    _obs_dtype = torch.int64 if is_discrete else dtype
    obs_buf   = torch.zeros(_obs_shape, dtype=_obs_dtype, device=device)
    act_buf   = torch.zeros(rollout_len, n_envs, dtype=torch.int64, device=device)
    lp_buf    = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)
    rew_buf   = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)
    done_buf  = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)
    term_buf  = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)
    reset_buf = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)
    val_buf   = torch.zeros(rollout_len, n_envs, dtype=dtype, device=device)

    viz = VizManager(rnn.n_layers, rnn.n_columns, vis_interval=vis_interval) if (is_grid and logger is not None) else None
    ep_step = torch.zeros(n_envs, dtype=torch.long, device=device)

    while step < n_steps:
        h_init = rnn_state

        with torch.no_grad():
            for t in range(rollout_len):
                raw        = gen.observe()
                reset_mask = to_torch(raw['reset_mask'], device=device)
                obs_in     = _obs_to_model_input(raw['obs'])

                rnn_state = rnn.reset_state(rnn_state, reset_mask)
                logits, rnn_state = rnn(obs_in, rnn_state)

                value = critic(extract_h_top(rnn_state)).squeeze(-1)

                dist      = Categorical(F.softmax(logits, dim=-1))
                actions   = dist.sample()
                log_probs = dist.log_prob(actions)

                rewards_np, dones_np, term_np = gen.step(to_numpy(actions))
                rewards = to_torch(rewards_np, device=device).to(dtype)
                dones   = to_torch(dones_np,   device=device).to(dtype)
                term    = to_torch(term_np,    device=device).to(dtype)

                obs_buf[t]   = obs_in
                act_buf[t]   = actions
                lp_buf[t]    = log_probs
                rew_buf[t]   = rewards
                done_buf[t]  = dones
                term_buf[t]  = term
                reset_buf[t] = reset_mask.to(dtype)
                val_buf[t]   = value

                if viz is not None:
                    # RepeatFirst phases: step-0 = store, terminal = query, middle = distract
                    if ep_step[0].item() == 0:
                        phase = 'store'
                    elif dones[0].bool():
                        phase = 'query'
                    else:
                        phase = 'distract'
                    viz.update(step, {}, rnn_state, has_hgrn=False, has_fusion=False, rnn=rnn)
                    viz.update_specialisation(phase, rnn_state, None)
                    ep_step = (ep_step + 1) * (1 - dones.long())

                step += n_envs

            # bootstrap value
            raw        = gen.observe()
            reset_mask = to_torch(raw['reset_mask'], device=device)
            obs_last   = _obs_to_model_input(raw['obs'])
            rnn_state  = rnn.reset_state(rnn_state, reset_mask)
            _, rnn_last = rnn(obs_last, rnn_state)
            value_last  = critic(extract_h_top(rnn_last)).squeeze(-1).detach()

        values_with_boot = torch.cat([val_buf, value_last.unsqueeze(0)], dim=0)

        advs, returns = compute_gae(rew_buf, values_with_boot, done_buf, term_buf)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # PPO update
        total_policy_loss = total_value_loss = total_entropy = total_aux_loss = 0.0
        n_updates = 0

        for _ in range(PPO_EPOCHS):
            h = h_init

            for t in range(rollout_len):
                h = rnn.reset_state(h, reset_buf[t])

                logits, h = rnn(obs_buf[t], h)

                value_new = critic(extract_h_top(h)).squeeze(-1)
                dist_new  = Categorical(F.softmax(logits, dim=-1))
                new_lp    = dist_new.log_prob(act_buf[t])
                entropy   = dist_new.entropy().mean()

                ratio = (new_lp - lp_buf[t]).exp()
                adv_t = advs[t]
                p_loss = -torch.min(
                    ratio * adv_t,
                    ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t,
                ).mean()
                v_loss = F.mse_loss(value_new, returns[t])

                loss = p_loss + VALUE_COEF * v_loss - entropy_coef * entropy

                if aux_head is not None:
                    # reconstruct current obs from column aux_col_idx of last layer
                    col_h = h[-1, aux_col_idx, :, :actor_hidden]  # [B, H] real part
                    if is_discrete:
                        aux_loss = F.cross_entropy(aux_head(col_h), obs_buf[t].squeeze(-1))
                    else:
                        aux_loss = F.mse_loss(aux_head(col_h), obs_buf[t].to(dtype))
                    loss = loss + aux_loss_w * aux_loss
                    total_aux_loss += aux_loss.item()

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, MAX_GRAD_NORM)
                optim.step()

                total_policy_loss += p_loss.item()
                total_value_loss  += v_loss.item()
                total_entropy     += entropy.item()
                n_updates += 1

                h = rnn.detach_state(h)

        rnn_state = rnn.detach_state(rnn_state)

        if viz is not None and step >= viz.next_step and logger is not None:
            viz.flush(logger, step, has_hgrn=False, has_reservoir=False, reservoir_sr_info={})

        lr.step()

        _stat = {
            'PolicyLoss' : total_policy_loss / max(n_updates, 1),
            'ValueLoss'  : total_value_loss  / max(n_updates, 1),
            'Entropy'    : total_entropy     / max(n_updates, 1),
            'MeanReward' : to_numpy(rew_buf.mean()),
            'LR'         : lr.val,
        }
        if aux_head is not None:
            _stat['AuxLoss'] = total_aux_loss / max(n_updates, 1)
        stats.put(_stat)

        if print_stats_schedule.tick(n_envs * rollout_len):
            m   = stats.get()
            fps = fps_counter.fps(n_iters=step, start=True)
            ep  = gen.get_stats()
            print(
                f'[{format_readable_num(step)} / {format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)} fps |'
                f' LR:{int(100*m["LR"]/lr.base_val)}% |'
                f' PL:{m["PolicyLoss"]:.3f}'
                f' VL:{m["ValueLoss"]:.3f}'
                f' H:{m["Entropy"]:.2f}'
                f' R:{m["MeanReward"]:.3f}'
                + (f' EpRet:{ep["ep_return"]:.3f}' if ep else '')
            )

        if log_stats_schedule.tick(n_envs * rollout_len) and logger is not None:
            fps = fps_counter.fps(n_iters=step, start=True)
            metrics = {'global_step': step, 'fps': fps} | stats.get()
            ep = gen.get_stats()
            if ep:
                metrics['env'] = ep
            logger.track(flatten_dict(metrics))

    fps = fps_counter.fps(n_iters=step)
    print(format_readable_num(fps))
    gen.close()


if __name__ == '__main__':
    run_experiment(runner=main)
