"""Multimodal Digit-Sum (MDS) experiment — GridRNN with per-column modality routing."""
from __future__ import annotations

import importlib

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
from knitwork.gens.multimodal_sum import MultimodalDigitSumGenerator
from knitwork.exps.multimodal_sum._viz import (
    plot_attn_heatmap, plot_sum_confusion, plot_sum_probe_r2,
    estimate_sum_location_r2, log_figure,
)
from knitwork.visualization.cka import CKAVisualizer


#  Model registry

_REGISTRY: dict[str, tuple[str, str]] = {
    'grnn_multimodal':        ('knitwork.models.grnn_multimodal', 'GridRnnMultimodal'),
    # naive-concat baseline (same class, signal_columns=[0,0] in config)
    'grnn_multimodal_concat': ('knitwork.models.grnn_multimodal', 'GridRnnMultimodal'),
    # v2: hard-gated buffer isolation (attn-masked) + concat (not mean-pool) signal head
    'grnn_multimodal_v2':     ('knitwork.models.grnn_multimodal_v2', 'GridRnnMultimodalV2'),
}


def build_model(rnn_type: str, rnn_cfg: dict, gen):
    mod_path, cls_name = _REGISTRY[rnn_type]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(
        **rnn_cfg,
        image_feat_dim=gen.image_dim, audio_feat_dim=gen.audio_dim,
        buffer_feat_dim=gen.buffer_dim,
        output_size=gen.n_sum_classes,
    )


#  Column-role similarity (signal vs buffer specialisation diagnostic)

def log_col_similarity_by_role(rnn, state, logger, step: int) -> None:
    try:
        if not isinstance(state, torch.Tensor) or state.ndim != 4:
            return
        h = state[-1]  # top layer: [cols, batch, hidden]
        acts = h.mean(dim=1).detach().float()  # [cols, hidden]
        norm = acts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        acts = acts / norm
        sim = (acts @ acts.T).cpu().numpy()

        sig = list(rnn.signal_columns)
        buf = list(rnn.buffer_columns)

        if sig[0] != sig[1]:
            logger.track(float(sim[sig[0], sig[1]]), name='col_sim/signal_signal', step=step)
        if len(buf) > 1:
            vals = [sim[i, j] for a, i in enumerate(buf) for j in buf[a + 1:]]
            logger.track(float(np.mean(vals)), name='col_sim/buffer_buffer', step=step)
        cross = [sim[i, j] for i in sig for j in buf]
        if cross:
            logger.track(float(np.mean(cross)), name='col_sim/signal_buffer', step=step)
    except Exception as e:
        print(f'[col_sim] {e}')


#  Modality / buffer ablation eval (does the model actually use each input?)

@torch.no_grad()
def run_ablation_eval(rnn, gen_cfg, *, device, dtype, seed, n_envs=32, steps=200):
    conditions = ('full', 'no_image', 'no_audio', 'no_buffer')
    results = {}
    was_training = rnn.training
    rnn.eval()
    eval_gen_cfg = {**gen_cfg, 'split': 'test'}
    for cond in conditions:
        gen = MultimodalDigitSumGenerator(
            **eval_gen_cfg, n_envs=n_envs, seed=seed, ignore_index=CE_ignore_index,
        )
        state = None
        correct, total = 0, 0
        for _ in range(steps):
            obs = {k: to_torch(v, device=device) for k, v in gen.next().items()}
            state = rnn.reset_state(state, obs['reset_mask'])

            image_feat  = obs['image_feat'].to(dtype=dtype)
            audio_feat  = obs['audio_feat'].to(dtype=dtype)
            buffer_feat = obs['buffer_feat'].to(dtype=dtype)
            if cond == 'no_image':
                image_feat = torch.zeros_like(image_feat)
            elif cond == 'no_audio':
                audio_feat = torch.zeros_like(audio_feat)
            elif cond == 'no_buffer':
                buffer_feat = torch.zeros_like(buffer_feat)

            y, state = rnn(image_feat, audio_feat, buffer_feat, state)
            m = obs['target'] != CE_ignore_index
            if m.any():
                correct += int((y[m].argmax(dim=-1) == obs['target'][m]).sum())
                total += int(m.sum())
        results[cond] = correct / total if total > 0 else float('nan')
    rnn.train(was_training)
    return results


#  Main

def main(config):
    _default_name = f"knitwork_{config['model']}_mdsum"
    run_name = config.get('name') or config.get('log', {}).get('name') or _default_name
    if not run_name.startswith('knitwork_'):
        run_name = 'knitwork_' + run_name
    config.setdefault('log', {})['name'] = run_name
    print(f'Run name: {run_name}')

    rng    = np.random.default_rng(config['seed'])
    device = get_device(config.get('device'))
    dtype  = get_dtype(config.get('dtype'))
    n_envs = config['n_envs']

    gen_cfg = config['gens'][config['gen']]
    gen = MultimodalDigitSumGenerator(
        **gen_cfg, n_envs=n_envs,
        seed=rng.integers(1_000_000),
        ignore_index=CE_ignore_index,
    )
    chance_acc = 1.0 / gen.n_sum_classes

    rnn_type = config['model']
    rnn_cfg  = config['models'][rnn_type]
    rnn = build_model(rnn_type, rnn_cfg, gen)
    rnn = rnn.to(device=device, dtype=dtype)
    print(f'Model on {next(rnn.parameters()).device} | dtype {next(rnn.parameters()).dtype}')
    print(f'Chance-level accuracy: {chance_acc:.4f} ({gen.n_sum_classes} classes)')

    lr = DynamicLearningRate(name='LR', **config['lr'])
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

    vis_interval      = int(config.get('vis_interval', 2_000_000))
    ablation_interval = int(config.get('ablation_interval', 2_000_000))
    ablation_steps    = int(config.get('ablation_steps', 200))
    ablation_envs     = int(config.get('ablation_envs', 32))
    next_vis_step      = vis_interval
    next_ablation_step = ablation_interval

    logger = create_logger(config)
    stats       = Tracker(lr=2e-4)
    fps_counter = FpsCounter()
    cka_vis = CKAVisualizer(n_layers=rnn.n_layers, n_columns=rnn.n_columns, buffer_size=50)

    rnn_state = None
    batch_y:        list = []
    batch_y_gt:     list = []
    batch_n_events: list = []
    last_attn_weights = None
    last_sum_probe = None   # (state_snapshot, sum_now_np)

    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}
        rnn_state = rnn.reset_state(rnn_state, obs['reset_mask'])

        capture_attn = logger is not None and step >= next_vis_step - gen.n_envs
        if capture_attn:
            y, rnn_state, extras = rnn(
                obs['image_feat'].to(dtype=dtype),
                obs['audio_feat'].to(dtype=dtype),
                obs['buffer_feat'].to(dtype=dtype),
                rnn_state, return_attn=True,
            )
            if extras.get('attn_weights'):
                last_attn_weights = [
                    w.detach().cpu().numpy() for w in extras['attn_weights'] if w is not None
                ]
            cka_vis.update(rnn_state)
            last_sum_probe = (rnn_state.detach(), to_numpy(obs['sum_now'], copy=False))
        else:
            y, rnn_state = rnn(
                obs['image_feat'].to(dtype=dtype),
                obs['audio_feat'].to(dtype=dtype),
                obs['buffer_feat'].to(dtype=dtype),
                rnn_state,
            )

        batch_y.append(y)
        batch_y_gt.append(obs['target'])
        batch_n_events.append(obs['n_events'])

        step += gen.n_envs

        if step % batch_size == 0:
            y_cat        = torch.cat(batch_y,        dim=0)
            y_gt_cat     = torch.cat(batch_y_gt,     dim=0)
            n_events_cat = torch.cat(batch_n_events, dim=0)
            m_active = y_gt_cat != CE_ignore_index

            loss = loss_fn(y_cat, y_gt_cat)

            with torch.no_grad():
                acc = (y_cat[m_active].argmax(dim=-1) == y_gt_cat[m_active]).float()
                acc_mean = acc.mean() if acc.numel() > 0 else torch.tensor(float('nan'))

                # accuracy bucketed by number of digits accumulated so far
                # (memory-capacity diagnostic: does acc degrade as the sum grows?)
                event_acc = {}
                for k in range(gen.max_events_per_episode + 1):
                    bmask = m_active & (n_events_cat == k)
                    if bmask.any():
                        event_acc[f'Acc/n_events_{k}'] = (
                            y_cat[bmask].argmax(dim=-1) == y_gt_cat[bmask]
                        ).float().mean().item()

            optim.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('Nan/Inf grad — step skipped')
            lr.step()

            stats.put({
                'Loss':       to_numpy(loss,      copy=False),
                'Acc':        to_numpy(acc_mean,  copy=False),
                '|Grad|':     to_numpy(grad_norm, copy=False),
                'LR':         lr.val,
                'chance_acc': chance_acc,
                **event_acc,
            })

            if logger is not None and step % (batch_size * 20) == 0:
                log_col_similarity_by_role(rnn, rnn_state, logger, step)

            if capture_attn and logger is not None:
                if last_attn_weights:
                    fig = plot_attn_heatmap(
                        last_attn_weights, rnn.n_layers, rnn.n_columns, rnn.signal_columns, step
                    )
                    log_figure(logger, fig, 'viz/attn_heatmap', step)
                y_true_np = to_numpy(y_gt_cat[m_active], copy=False)
                y_pred_np = to_numpy(y_cat[m_active].argmax(dim=-1), copy=False)
                fig = plot_sum_confusion(y_true_np, y_pred_np, gen.n_sum_classes, step)
                log_figure(logger, fig, 'viz/sum_confusion', step)

                try:
                    cka_vis.log(logger, step=step)
                except Exception as e:
                    print(f'[CKA] {e}')

                if last_sum_probe is not None:
                    state_snap, sum_now_np = last_sum_probe
                    r2 = estimate_sum_location_r2(state_snap, sum_now_np)
                    fig = plot_sum_probe_r2(r2, rnn.signal_columns, step)
                    log_figure(logger, fig, 'viz/sum_probe_r2', step)
                    for li in range(r2.shape[0]):
                        for ci in range(r2.shape[1]):
                            logger.track(float(r2[li, ci]), name=f'sum_probe_r2/L{li}_C{ci}', step=step)

                next_vis_step += vis_interval

            if step >= next_ablation_step:
                abl = run_ablation_eval(
                    rnn, gen_cfg, device=device, dtype=dtype,
                    seed=int(rng.integers(1_000_000)),
                    n_envs=ablation_envs, steps=ablation_steps,
                )
                print(f'[ablation @ {format_readable_num(step)}] ' + ' '.join(
                    f'{k}:{v:.3f}' for k, v in abl.items()
                ))
                if logger is not None:
                    for k, v in abl.items():
                        logger.track(v, name=f'ablation/{k}', step=step)
                next_ablation_step += ablation_interval

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear(); batch_y_gt.clear(); batch_n_events.clear()

        axes = curriculum_step.tick(metrics=stats, n_steps=gen.n_envs)
        if any(axes.values()):
            K = 10
            gen.set_metaparams(
                T=gen.T + 1.0 / K if axes['T'] else gen.T,
                # p_store axis drives arrival sparsity (analogue of SDQ's store rate)
                p_arrive=max(gen.p_arrive - 0.0014 / K, 0.05) if axes['p_store'] else gen.p_arrive,
                p_query=max(gen.p_query - 0.0005 / K, 0.10) if axes['p_query'] else gen.p_query,
            )

        if print_stats_schedule.tick(gen.n_envs):
            m   = {'global_step': step} | stats.get()
            fps = fps_counter.fps(n_iters=step, start=True)
            print(
                f'[{format_readable_num(step)}/{format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)}fps |'
                f' LR:{int(100 * m["LR"] / lr.base_val)}% |'
                f' L:{m["Loss"]:.3f}'
                f' A:{m["Acc"]:.3f} (chance {chance_acc:.3f})'
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
            write_status(step, metrics)
            if logger is not None:
                logger.track(flatten_dict(metrics))

    fps = fps_counter.fps(n_iters=step)
    print(f'Done. {format_readable_num(fps)} fps')


if __name__ == '__main__':
    run_experiment(runner=main)
