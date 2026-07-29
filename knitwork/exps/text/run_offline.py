"""Offline-batched Transformer text experiment.

The Transformer receives a full TBPTT rollout in one call while preserving the
online generator, reset curriculum, optimizer, and primary training metrics.
"""
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
from knitwork.common.tracking import SplitEmaTracker
from knitwork.common.utils import (
    CE_ignore_index, count_learnable_params, format_readable_num,
    get_device, get_dtype,
)
from knitwork.gens.text import TextGenerator, load_dataset, tokenize
from knitwork.models.utils import build_model


def next_rollout(gen, rollout_len):
    observations = [gen.next() for _ in range(rollout_len)]
    return {
        key: torch.stack([obs[key] for obs in observations])
        for key in observations[0]
    }


def main(config):
    torch.set_float32_matmul_precision('high')

    default_name = f'{config["model"]}.offline'
    name_sfx = config.get('name') or config['log'].get('name') or ''
    config['offline'] = True

    rng = np.random.default_rng(config['seed'])
    device = get_device(config.get('device'))
    dtype = get_dtype(config.get('dtype'))
    n_envs = config['n_envs']
    rollout_len = config['rollout_len']

    gen_cfg = config['gens'][config['gen']]
    data_path = Path(gen_cfg['path']).expanduser()
    data, charset = tokenize(load_dataset(data_path))
    n_chars = charset.size
    space_token = charset.tobytes().decode('utf-8').find(' ')

    train_data = data
    eval_cfg = config.get('eval', {})
    do_eval = eval_cfg.get('enabled', False)
    do_eval_on_start = eval_cfg.get('on_start', False)
    if do_eval:
        from knitwork.gens.text import split_train_test
        eval_schedule = create_scheduler(eval_cfg['schedule'])
        max_rollout = int(eval_cfg.get('max_rollout', 1e8))
        context_window = eval_cfg.get('context_window')
        train_data, val_data = split_train_test(
            data, train_frac=1.0 - eval_cfg['split']
        )
        val_gen = TextGenerator(
            val_data, n_envs=n_envs, ignore_index=CE_ignore_index,
            seed=get_seed(rng), device=device,
        ).to(device)
        max_rollout = min(
            max_rollout,
            max(100 * rollout_len, round(len(val_data) / n_envs)),
        )

    gen = TextGenerator(
        train_data, n_envs=n_envs, ignore_index=CE_ignore_index,
        seed=get_seed(rng), device=device,
    ).to(device)

    config['model_cfg'] = config['model'].replace('.', '_')
    config['model'] = config['model'].split('.', 1)[0]
    if config['model'] != 'transformer':
        raise ValueError('run_offline.py currently supports only model=transformer')

    wrapper_cfg = config[f'{config["wrapper_model"]}_wrapper'] | dict(
        input_size=n_chars,
        output_size=n_chars,
        dtype=dtype,
        device=device,
    )
    model = build_model(
        wrapper_type=config['wrapper_model'],
        wrapper_cfg=wrapper_cfg,
        rnn_type=config['model'],
        rnn_cfg=config[config['model_cfg']],
    ).to(device=device, dtype=dtype)
    if config.get('compile', False):
        model = torch.compile(model)
    rnn = model.rnn
    print(f'Model on {next(model.parameters()).device} | dtype {next(model.parameters()).dtype}')

    run_name = f'{default_name}_{count_learnable_params(model, as_str=True)} {name_sfx}'
    config['log']['name'] = run_name
    print(f'Run name: {run_name}')

    batch_size = n_envs * rollout_len
    n_steps = int(config['n_steps'])
    n_updates = n_steps // batch_size
    effective_steps = n_updates * batch_size
    print(
        f'Offline budget: {effective_steps:,} trained tokens in {n_updates:,} updates'
        f' ({n_steps - effective_steps:,} trailing tokens omitted)'
    )

    lr = DynamicLearningRate(name='LR', **config['lr'])
    optim = torch.optim.RMSprop(model.parameters(), lr=lr.val)
    lr.connect_to_optimiser(optim)

    gen_cfg['reset_prob']['val'] /= rollout_len
    gen_cfg['reset_prob']['tar'] /= rollout_len
    p_reset = DynamicParameter(**gen_cfg['reset_prob'])
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)

    _print_short_summary = partial(
        print_short_summary, max_steps=n_steps, lr=lr,
    )
    logger = start_logger(
        config, tracker=config['trackers'],
        suppress_printing=True, callbacks=[_print_short_summary],
    )
    in_word_acc = SplitEmaTracker(
        bins=config['inspect_n_in_word_acc'], lr=0.01,
    )
    in_word_acc.ixs = torch.zeros(n_envs, dtype=torch.int64, device=device)

    def _run_eval(step):
        run_eval(
            step, model=model, gen=val_gen, logger=logger,
            n_envs=n_envs, max_rollout=max_rollout,
            rollout_len=rollout_len, device=device,
            context_window=context_window,
        )

    if do_eval and do_eval_on_start:
        _run_eval(0)
        logger.log(0, flush=True, force=True)

    state = None
    step = 0
    for i_update in range(1, n_updates + 1):
        obs = next_rollout(gen, rollout_len)
        reset_mask = obs['reset_mask'] | (
            torch.rand(
                rollout_len, n_envs,
                device=device, generator=gen.rng,
            ) < p_reset.val
        )
        x = obs['tokens'].transpose(0, 1)
        in_word_pos = []
        for tokens in obs['tokens']:
            in_word_pos.append(in_word_acc.ixs)
            in_word_acc.ixs = torch.where(
                tokens == space_token, 0, in_word_acc.ixs + 1,
            )
        y, state, _ = model(x, state, reset_mask=reset_mask)

        y = y.reshape(-1, n_chars)
        targets = obs['targets'].reshape(-1)
        ce_loss = loss_fn(y, targets)

        with torch.no_grad():
            active = targets != CE_ignore_index
            acc = (y[active].argmax(dim=-1) == targets[active]).float()
            bpc = ce_loss / np.log(2.0)
            perplexity = torch.exp(ce_loss)

        optim.zero_grad()
        ce_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(grad_norm):
            optim.step()
        else:
            print('Nan/Inf grad — step skipped')

        state = rnn.detach_state(state)
        p_reset.step()
        lr.step()
        step += batch_size

        metrics = to_loggable_metrics({
            'Loss': ce_loss,
            'BPC': bpc,
            'Perplexity': perplexity,
            'Acc': acc,
            '|Grad|': grad_norm,
            'LR': lr.val,
            'T': min(1e6, 1.0 / p_reset.val),
            'Upd': i_update,
        })
        logger.accumulate(metrics, key='slow')
        update_in_word_acc(in_word_acc, in_word_pos, acc, active)
        logger.accumulate(in_word_acc.get(split=True), key='fast')

        if do_eval and eval_schedule.tick(batch_size):
            _run_eval(step)
        logger.log(step, flush=True)

    if do_eval and eval_schedule.tick(batch_size):
        _run_eval(step)
    logger.log(step, flush=True, force=True)
    logger.finish()


@torch.no_grad()
def run_eval(
        step, *, model, gen, logger, n_envs, max_rollout,
        rollout_len, device, context_window,
):
    model.eval()
    state = None
    ce_loss = 0.0
    acc = 0.0
    total = 0
    cw_ix = torch.zeros(n_envs, dtype=torch.int64, device=device)

    for start in range(0, max_rollout, rollout_len):
        T = min(rollout_len, max_rollout - start)
        obs = next_rollout(gen, T)
        reset_mask = obs['reset_mask']
        if context_window is not None:
            reset_steps = []
            for reset in reset_mask:
                reset_steps.append(reset | (cw_ix == 0))
                cw_ix = (cw_ix + 1) % context_window
            reset_mask = torch.stack(reset_steps)

        y, state, _ = model(
            obs['tokens'].transpose(0, 1), state,
            reset_mask=reset_mask,
        )
        targets = obs['targets'].reshape(-1)
        y = y.reshape(-1, y.shape[-1])
        valid = targets != CE_ignore_index
        y, targets = y[valid], targets[valid]
        if len(y) == 0:
            continue

        ce = nn.functional.cross_entropy(y, targets, reduction='sum')
        ce_loss += ce
        acc += (y.argmax(dim=-1) == targets).sum()
        total += len(y)

    model.train()
    if total == 0:
        print('No valid data for evaluation!')
        return

    ce_loss /= total
    acc /= total
    metrics = to_loggable_metrics({
        'Loss': ce_loss,
        'BPC': ce_loss / np.log(2.0),
        'Acc': acc,
    })
    logger.accumulate(metrics, prefix='val', key='eval')


def print_short_summary(step, *, scalars, figures, max_steps, lr):
    if 'Loss' in scalars:
        print(
            f'[{format_readable_num(step)}/{format_readable_num(max_steps, frac=0)}]'
            f' {format_readable_num(scalars["perf/fps"], frac=0)}fps |'
            f' LR:{int(100 * scalars["LR"] / lr.base_val)}%'
            f' T:{int(scalars["T"])} |'
            f' L:{scalars["Loss"]:.3f}'
            f' BPC:{scalars["BPC"]:.3f}'
            f' A:{scalars["Acc"]:.3f}'
        )
    if 'val/Loss' in scalars:
        print(
            f'[{format_readable_num(step)}/{format_readable_num(max_steps, frac=0)} EVAL]'
            f' L:{scalars["val/Loss"]:.3f}'
            f' BPC:{scalars["val/BPC"]:.3f}'
            f' A:{scalars["val/Acc"]:.3f}'
        )


@torch.no_grad()
def update_in_word_acc(in_word_acc, in_word_pos, acc, active):
    positions = torch.stack(in_word_pos).reshape(-1)[active]
    in_word_acc.put(
        {'in_word_stats/Acc': to_numpy(acc, copy=False).ravel()},
        ixs=to_numpy(positions, copy=False),
    )


if __name__ == '__main__':
    run_experiment(runner=main)
