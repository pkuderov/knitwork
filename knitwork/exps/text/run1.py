from pathlib import Path
from datetime import datetime

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


def main(config):
    run_name = (
        config.get('name', None)
        or config.get('log', {}).get('name', None)
        or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    config.setdefault('log', {})['name'] = run_name
    print(f'Run name: {run_name}')

    rng = np.random.default_rng(config['seed'])
    device = get_device(config.get('device', None))
    dtype = get_dtype(config.get('dtype', None))

    n_envs = config['n_envs']

    gen_cfg = config['gens'][config['gen']]

    data_path = Path(gen_cfg['path']).expanduser()
    data, ds_charset = tokenize(load_dataset(data_path))
    n_chars = ds_charset.size

    gen = TextGenerator(
        data, n_envs=n_envs, ignore_index=CE_ignore_index,
        seed=rng.integers(1_000_000)
    )

    rnn_type = config['model']
    rnn_cfg = config['models'][rnn_type]
    match rnn_type:
        case 'rnn':
            from knitwork.models.gru import GruBaseline
            rnn_fn = GruBaseline
        case 'grnn':
            from knitwork.models.grnn import GridRnn
            rnn_fn = GridRnn
        case 'grnn_err':
            from knitwork.models.grnn_err import GridRnn
            rnn_fn = GridRnn
        case 'hgrnn':
            from knitwork.models.hgrnn import HopfieldGridRnn
            rnn_fn = HopfieldGridRnn

    rnn = rnn_fn(**rnn_cfg, input_size=n_chars, output_size=n_chars)
    rnn = rnn.to(device=device, dtype=dtype)
    print(
        f'Model is on "{next(rnn.parameters()).device}"'
        f' having "{next(rnn.parameters()).dtype}" dtype'
    )

    lr_cfg = config['lr']
    lr = lr_cfg['val']
    print(f"Base LR: {lr}")

    wm_lr_cfg, wm_lr_schedule = extracted(lr_cfg['warmup'], 'schedule')
    dc_lr_cfg, dc_lr_schedule = extracted(lr_cfg['decay'], 'schedule')
    wm_lr = DynamicParameter(
        val=1e-5*lr, tar=lr, **wm_lr_cfg,
        scheduler=Scheduler(wm_lr_schedule)
    )
    dc_lr = DynamicParameter(
        val=lr, **dc_lr_cfg,
        scheduler=Scheduler(dc_lr_schedule)
    )
    def get_lr():
        return wm_lr.val if not wm_lr.scheduler.is_infinite else dc_lr.val
    def step_lr():
        return wm_lr.step() if not wm_lr.scheduler.is_infinite else dc_lr.step()

    optim = torch.optim.RMSprop(rnn.parameters(), lr=get_lr())
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)

    rollout_len = config['rollout_len']
    batch_size = gen.n_envs * rollout_len

    n_steps = int(config['n_steps'])
    step = 0
    log_stats_schedule = Scheduler(int(config['log']['schedule']))
    print_stats_schedule = Scheduler(int(config['log']['print_schedule']))

    p_reset_cfg, p_reset_decay_schedule = extracted(gen_cfg['reset_prob'], 'schedule')
    p_reset = DynamicParameter(
        **p_reset_cfg, scheduler=Scheduler(int(p_reset_decay_schedule))
    )

    logger = create_logger(config)

    stats = Tracker(lr=2e-4)
    fps_counter = FpsCounter()

    rnn_state = None
    batch_y = []
    batch_y_gt = []
    ln_2 = np.log(2.0)

    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}

        rnd_reset = torch.from_numpy(rng.random(gen.n_envs) < p_reset.val).to(device=device)
        reset_mask = torch.logical_or(obs['reset_mask'], rnd_reset)

        rnn_state = rnn.reset_state(rnn_state, reset_mask)
        x = obs['tokens'].view(-1, 1)
        y, rnn_state = rnn(x, rnn_state)

        batch_y.append(y)
        batch_y_gt.append(obs['targets'])

        step += gen.n_envs

        if step % batch_size == 0:
            y = torch.cat(batch_y, dim=0)
            y_gt = torch.cat(batch_y_gt, dim=0)
            m_active = y_gt != CE_ignore_index

            loss = loss_fn(y, y_gt)
            with torch.no_grad():
                acc = (y[m_active].argmax(dim=-1) == y_gt[m_active]).float()
                bpc = loss / ln_2
                perplexity = torch.exp(loss)
            acc = acc.mean()

            optim.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('Nan loss')

            stats.put({
                "Loss": to_numpy(loss, copy=False),
                "BPC": to_numpy(bpc, copy=False),
                "Perplexity": to_numpy(perplexity, copy=False),
                "Acc": to_numpy(acc, copy=False),
                "|Grad|": to_numpy(grad_norm, copy=False),
                "LR": get_lr(),
                "T": 1 / p_reset.val,
            })

            p_reset.step()
            if step_lr():
                optim.param_groups[0]['lr'] = get_lr()

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear()
            batch_y_gt.clear()

        if print_stats_schedule.tick(gen.n_envs):
            metrics = {"global_step": step} | stats.get()
            fps = fps_counter.fps(n_iters=step, start=True)
            print(
                f'[{format_readable_num(step)} / {format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)} fps |'
                f' LR: {int(100*metrics["LR"]/lr)}%  '
                f' T: {int(metrics["T"])} | '
                f' L: {metrics["Loss"]:.3f}, A: {metrics["Acc"]:.3f}'
            )

        if log_stats_schedule.tick(gen.n_envs) and logger is not None:
            fps = fps_counter.fps(n_iters=step, start=True)
            metrics = {
                "global_step": step, "fps": fps,
            } | stats.get()
            gen_stats = gen.get_stats()
            metrics['gen'] = gen_stats
            logger.track(flatten_dict(metrics))

    fps = fps_counter.fps(n_iters=step)
    print(format_readable_num(fps))


if __name__ == "__main__":
    run_experiment(runner=main)