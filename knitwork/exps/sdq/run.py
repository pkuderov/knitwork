import numpy as np
import torch
from torch import nn

from knitwork.common.curriculum import CurriculumScheduler
from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging import create_logger
from knitwork.common.scheduler import create_scheduler
from knitwork.common.torch import DynamicLearningRate
from knitwork.common.tracker import Tracker
from knitwork.common.utils import CE_ignore_index, FpsCounter, flatten_dict, format_readable_num, get_device, get_dtype, to_numpy, to_torch
from knitwork.gens.sdq import StoreDistractQueryGenerator


def main(config):
    rng = np.random.default_rng(config['seed'])
    device = get_device(config.get('device', None))
    dtype = get_dtype(config.get('dtype', None))

    n_envs=config['n_envs']

    gen_cfg = config['gens'][config['gen']]
    gen = StoreDistractQueryGenerator(
        **gen_cfg, n_envs=n_envs, seed=rng.integers(1_000_000),
        ignore_index=CE_ignore_index
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

    rnn = rnn_fn(**rnn_cfg, input_size=gen.n_tokens, output_size=gen.V)
    rnn = rnn.to(device=device, dtype=dtype)
    print(
        f'Model is on "{next(rnn.parameters()).device}"'
        f' having "{next(rnn.parameters()).dtype}" dtype'
    )

    lr = DynamicLearningRate(name=f'LR', **config['lr'])
    optim = torch.optim.RMSprop(rnn.parameters(), lr=lr.val)
    lr.connect_to_optimiser(optim)

    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)

    rollout_len = config['rollout_len']
    batch_size = gen.n_envs * rollout_len

    n_steps = int(config['n_steps'])
    step = 0
    log_stats_schedule = create_scheduler(config['log']['schedule'])
    print_stats_schedule = create_scheduler(config['log']['print_schedule'])
    curriculum_step_schedule = CurriculumScheduler(**config['curriculum'])

    logger = create_logger(config)

    stats = Tracker(lr=2e-4)
    fps_counter = FpsCounter()

    rnn_state = None
    batch_y = []
    batch_y_gt = []
    batch_sq_gaps = []

    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}

        rnn_state = rnn.reset_state(rnn_state, obs['reset_mask'])
        x = obs['tokens'].view(-1, 1)
        y, rnn_state = rnn(x, rnn_state)

        batch_y.append(y)
        batch_y_gt.append(obs['targets'])
        batch_sq_gaps.append(obs['sq_gaps'])

        step += gen.n_envs

        if step % batch_size == 0:
            y = torch.cat(batch_y, dim=0)
            y_gt = torch.cat(batch_y_gt, dim=0)
            sq_gaps = torch.cat(batch_sq_gaps, dim=0).float()
            m_active = y_gt != CE_ignore_index

            loss = loss_fn(y, y_gt)
            with torch.no_grad():
                acc = (y[m_active].argmax(dim=-1) == y_gt[m_active]).float()

            mask_misses = sq_gaps < 0.0
            acc_miss = acc[mask_misses].mean()
            acc_non_miss = acc[~mask_misses].mean()
            acc_up_half = acc[sq_gaps > sq_gaps[~mask_misses].mean()].mean()
            acc = acc.mean()

            optim.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('Nan loss')

            lr.step()

            stats.put({
                "Loss": to_numpy(loss, copy=False), 
                "Acc": to_numpy(acc, copy=False),
                "Acc-": to_numpy(acc_miss, copy=False),
                "Acc+": to_numpy(acc_non_miss, copy=False),
                "Acc++": to_numpy(acc_up_half, copy=False),
                "|Grad|": to_numpy(grad_norm, copy=False),
                "LR": lr.val,
            })

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear()
            batch_y_gt.clear()
            batch_sq_gaps.clear()

        if curriculum_step_schedule.tick(metrics=stats, n_steps=gen.n_envs):
            K = 10
            dT, dp_store, dp_query = 1.0, -0.0014, -0.0005
            dT, dp_store, dp_query = dT/K, dp_store/K, dp_query/K

            gen.set_metaparams(
                T=gen.T + dT,
                p_store=max(gen.p_store + dp_store, 0.10),
                p_query=max(gen.p_query + dp_query, 0.25)
            )

        if print_stats_schedule.tick(gen.n_envs):
            metrics = {"global_step": step} | stats.get()
            fps = fps_counter.fps(n_iters=step, start=True)
            print(
                f'[{format_readable_num(step)} / {format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)} fps |'
                f' LR: {int(100*metrics["LR"]/lr.base_val)}% | '
                f' L: {metrics["Loss"]:.3f}, A: {metrics["Acc"]:.3f}'
                f' A-: {metrics["Acc-"]:.3f}, A+: {metrics["Acc+"]:.3f},'
                f' A++: {metrics["Acc++"]:.3f}'
            )
            # from pprint import pprint
            # pprint(gen.get_stats(), sort_dicts=False, indent=4)

        if log_stats_schedule.tick(gen.n_envs) and logger is not None:
            fps = fps_counter.fps(n_iters=step, start=True)
            metrics = {
                "global_step": step, "fps": fps, 
                "curr_step": curriculum_step_schedule.cnt_accepted,
                "curr_schedule": curriculum_step_schedule.scheduler.schedule,
            } | stats.get()
            gen_stats = gen.get_stats()
            metrics['gen'] = gen_stats
            logger.track(flatten_dict(metrics))
    
    fps = fps_counter.fps(n_iters=step)
    print(format_readable_num(fps))


if __name__ == "__main__":
    run_experiment(runner=main)