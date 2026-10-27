"""Configuration, accounting and checkpoint primitives for the six controls."""

import copy
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from knitwork.common.state_size import state_floats
from knitwork.common.utils import count_learnable_params
from knitwork.models.utils import build_model


CONDITIONS = tuple(
    f'{routing}_{readout}'
    for routing in ('dynamic', 'static', 'none')
    for readout in ('column0', 'mean')
)


def resolve_config(config):
    cfg = copy.deepcopy(config)
    condition = cfg['model'].split('.', 1)[1]
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown condition: {condition}')
    if cfg['model'].split('.', 1)[0] != 'grnn_communication':
        raise ValueError('This runner only supports grnn_communication')
    key = cfg['model'].replace('.', '_')
    cfg[key].update(cfg['topologies'][cfg['topology']])
    routing, readout = condition.split('_', 1)
    if (cfg[key]['routing'], cfg[key]['readout']) != (routing, readout):
        raise ValueError('Model condition name must agree with routing/readout settings')
    if cfg[key]['hidden_size'] % cfg[key]['n_attn_heads']:
        raise ValueError('Width must divide the head count exactly; silent rounding is forbidden')
    recipe = cfg['recipes'][cfg['recipe']]
    cfg[key]['noise_std'] = recipe['noise_std']
    cfg['communication'] = {
        'loss_weight': recipe['loss_weight'],
        'entropy_weight': recipe['entropy_weight'],
    }
    for name in ('seed', 'data_seed'):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f'{name} must be an explicit nonnegative integer')
    batch = cfg['n_envs'] * cfg['rollout_len']
    if cfg['n_envs'] < 1 or cfg['rollout_len'] < 1 or int(cfg['n_steps']) < batch:
        raise ValueError('Require positive streams/rollout and at least one full update')
    if int(cfg['checkpoint']['every_tokens']) < 1:
        raise ValueError('checkpoint.every_tokens must be positive')
    if int(cfg['eval']['schedule']) < 1 or int(cfg['log']['schedule']) < 1:
        raise ValueError('Evaluation/logging schedules must be positive')
    if cfg['eval']['n_envs'] < 1:
        raise ValueError('Evaluation stream count must be positive')
    if cfg['exp'] == 'sdq' and (cfg['eval']['sdq_steps'] < 1 or cfg['eval']['fixed_gap_examples'] < 1 or not cfg['eval']['fixed_gaps']):
        raise ValueError('SDQ evaluation requires positive steps, examples and fixed gaps')
    cfg['condition'] = condition
    cfg['run_id'] = f"{cfg['exp']}/{cfg['topology']}/{cfg['recipe']}/{condition}/seed{cfg['seed']}-data{cfg['data_seed']}"
    cfg['effective_tokens'] = int(cfg['n_steps']) // batch * batch
    cfg['log']['name'] = cfg['run_id']
    return cfg


def make_model(cfg, *, device='cpu', vocab=None):
    if vocab is None:
        vocab = 27 if cfg['exp'] == 'text8' else 65
    output = vocab if cfg['exp'] == 'text8' else cfg['generator']['n_vals']
    return build_model(
        wrapper_type='token',
        wrapper_cfg={'input_size': vocab, 'output_size': output, 'dtype': torch.float32, 'device': device},
        rnn_type='grnn_communication',
        rnn_cfg=cfg[cfg['model'].replace('.', '_')],
    ).to(device=device, dtype=torch.float32)


def accounting(model):
    rnn = model.rnn
    L, C, H = rnn.n_layers, rnn.n_columns, rnn.hidden_size
    V_in, V_out = model.input_size, model.output_size
    # Same partial dense forward convention as the manuscript, including both adapters.
    dense = 2 * ((10 * L * C + 2) * H**2 + 2 * H * (C * (C + 1) + (L - 1) * C**2) + V_out * H)
    return {
        'parameters_nominal_trainable': count_learnable_params(model),
        'parameters_gru': count_learnable_params(rnn.cells),
        'parameters_router': count_learnable_params(rnn.attn),
        'parameters_adapters': V_in * H + (H + 1) * V_out,
        'state_scalars': state_floats(rnn),
        'state_scalars_with_path_to_readout': L * H if rnn.routing == 'none' and rnn.readout == 'column0' else L * C * H,
        'dense_forward_flops_per_token': dense,
        'dense_flops_scope': 'forward matrix products only; excludes elementwise operations, backward and optimizer',
        'parameter_caveat': 'Masked single-source routes have zero task gradient in Q/K; unreachable columns are deliberate in none_column0. Nominal parameter equality is not effective-capacity equality.',
    }


def config_fingerprint(cfg):
    canonical = copy.deepcopy(cfg)
    for key in ('resume', 'output_dir', 'log', 'accounting', 'runtime'):
        canonical.pop(key, None)
    canonical['checkpoint'].pop('every_tokens', None)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def tensor_digest(state_dict):
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def scheduler_state(scheduler):
    return {name: getattr(scheduler, name) for name in ('schedule', 'remain', 'slowdown', 'total_events')}


def restore_scheduler(scheduler, saved):
    for name, value in saved.items():
        setattr(scheduler, name, value)


def parameter_state(parameter):
    saved = {}
    for name in ('param', 'warmup'):
        part = getattr(parameter, name)
        if part is None:
            saved[name] = None
        else:
            saved[name] = {key: value for key, value in vars(part).items() if key != 'scheduler'}
            saved[name]['scheduler'] = scheduler_state(part.scheduler)
    return saved


def restore_parameter(parameter, saved):
    for name in ('param', 'warmup'):
        if saved[name] is None:
            setattr(parameter, name, None)
        else:
            part = getattr(parameter, name)
            if part is None:
                raise ValueError('Resume requires the original schedule configuration')
            part.__dict__.update({key: value for key, value in saved[name].items() if key != 'scheduler'})
            restore_scheduler(part.scheduler, saved[name]['scheduler'])


def rng_state():
    return {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }


def restore_rng(saved):
    torch.set_rng_state(saved['torch'].cpu())
    if saved['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in saved['cuda']])
    np.random.set_state(saved['numpy'])
    random.setstate(saved['python'])
