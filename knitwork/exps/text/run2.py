# run2.py ─ тренировка GridRnn2 с VAE + TimeGate + ColDropout
# + метрики и визуализации (AttnFlow, CKA, gate stats)
from __future__ import annotations

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

# ── Визуализации ──────────────────────────────────────────────────────────────
from knitwork.visualization.attn_flow import AttnFlowVisualizer
from knitwork.visualization.cka import CKAVisualizer

VIS_INTERVAL = 10_000_000


def main(config):
    # ── Имя запуска ───────────────────────────────────────────────────────
    run_name = (
        config.get('name', None)
        or config.get('log', {}).get('name', None)
        or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    config.setdefault('log', {})['name'] = run_name
    print(f'Run name: {run_name}')

    VIS_ENABLED = config.get('visualize', True)

    rng    = np.random.default_rng(config['seed'])
    device = get_device(config.get('device', None))
    dtype  = get_dtype(config.get('dtype', None))
    n_envs = config['n_envs']

    # ── Датасет ───────────────────────────────────────────────────────────
    gen_cfg   = config['gens'][config['gen']]
    data_path = Path(gen_cfg['path']).expanduser()
    data, ds_charset = tokenize(load_dataset(data_path))
    n_chars = ds_charset.size

    gen = TextGenerator(
        data, n_envs=n_envs,
        ignore_index=CE_ignore_index,
        seed=rng.integers(1_000_000)
    )

    # ── Выбор модели ──────────────────────────────────────────────────────
    rnn_type = config['model']
    rnn_cfg  = config['models'][rnn_type]

    match rnn_type:
        case 'rnn':
            from knitwork.models.gru import GruBaseline
            rnn_fn = GruBaseline
        case 'grnn':
            from knitwork.models.grnn import GridRnn
            rnn_fn = GridRnn
        case 'grnn2':
            from knitwork.models.grnn2 import GridRnn2
            rnn_fn = GridRnn2
        case 'grnn_err':
            from knitwork.models.grnn_err import GridRnn
            rnn_fn = GridRnn
        case 'hgrnn':
            from knitwork.models.hgrnn import HopfieldGridRnn
            rnn_fn = HopfieldGridRnn
        case 'grnn_loss':
            from knitwork.models.grnn_loss import GridRnnLoss
            rnn_fn = GridRnnLoss
        case 'grnn_res':
            from knitwork.models.grnn_reservoir import GridRnnReservoir
            rnn_fn = GridRnnReservoir
        case 'grnn_engram':
            from knitwork.models.engram_grnn import EngramGridRnn
            rnn_fn = EngramGridRnn
        case _:
            raise ValueError(f"Неизвестный тип модели: {rnn_type}")

    rnn = rnn_fn(**rnn_cfg, input_size=n_chars, output_size=n_chars)
    rnn = rnn.to(device=device, dtype=dtype)
    print(
        f'Model on "{next(rnn.parameters()).device}"'
        f' dtype="{next(rnn.parameters()).dtype}"'
    )

    # ── Флаг: является ли модель GridRnn2 (другой forward) ───────────────
    is_grnn2 = rnn_type == 'grnn2'

    # ── Визуализаторы ─────────────────────────────────────────────────────
    attn_vis = AttnFlowVisualizer(
        n_layers=rnn.n_layers, n_columns=rnn.n_columns, buffer_size=100
    )
    cka_vis = CKAVisualizer(
        n_layers=rnn.n_layers, n_columns=rnn.n_columns, buffer_size=50
    )
    next_vis_step = VIS_INTERVAL
    gate_buffer: list = []

    # ── Learning rate ─────────────────────────────────────────────────────
    lr_cfg = config['lr']
    lr     = lr_cfg['val']
    print(f"Base LR: {lr}")

    wm_lr_cfg, wm_lr_schedule = extracted(lr_cfg['warmup'], 'schedule')
    dc_lr_cfg, dc_lr_schedule = extracted(lr_cfg['decay'],  'schedule')
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

    optim   = torch.optim.RMSprop(rnn.parameters(), lr=get_lr())
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)

    # ── KL Annealing для VAE ──────────────────────────────────────────────
    kl_anneal_cfg   = config.get('kl_anneal', {})
    kl_anneal_steps = int(kl_anneal_cfg.get('steps', 50_000))
    kl_anneal_max   = float(kl_anneal_cfg.get('max_weight', 1.0))

    def get_kl_anneal(step: int) -> float:
        if kl_anneal_steps == 0:
            return kl_anneal_max
        return kl_anneal_max * min(1.0, step / kl_anneal_steps)

    # ── Расписания ────────────────────────────────────────────────────────
    rollout_len  = config['rollout_len']
    batch_size   = gen.n_envs * rollout_len
    n_steps      = int(config['n_steps'])
    step         = 0

    log_stats_schedule   = Scheduler(int(config['log']['schedule']))
    print_stats_schedule = Scheduler(int(config['log']['print_schedule']))

    p_reset_cfg, p_reset_decay_schedule = extracted(
        gen_cfg['reset_prob'], 'schedule'
    )
    p_reset = DynamicParameter(
        **p_reset_cfg,
        scheduler=Scheduler(int(p_reset_decay_schedule))
    )

    logger = create_logger(config)
    if logger is not None:
        if hasattr(logger, 'name'):
            logger.name = run_name
        elif hasattr(logger, 'run') and hasattr(logger.run, 'name'):
            logger.run.name = run_name

    stats       = Tracker(lr=2e-4)
    fps_counter = FpsCounter()

    rnn_state   = None
    batch_y     = []
    batch_y_gt  = []
    batch_kl    = []
    batch_sq_gaps = []   # заполняется только если obs содержит 'sq_gaps'
    ln_2 = np.log(2.0)

    # ── Главный цикл ──────────────────────────────────────────────────────
    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}

        rnd_reset  = torch.from_numpy(
            rng.random(gen.n_envs) < p_reset.val
        ).to(device=device)
        reset_mask = torch.logical_or(obs['reset_mask'], rnd_reset)

        rnn_state = rnn.reset_state(rnn_state, reset_mask)
        x = obs['tokens'].view(-1, 1)

        # ── Forward pass: обычный или с захватом внимания для визуализации
        capture = (step >= next_vis_step - gen.n_envs)

        if capture:
            # ── Режим захвата: запрашиваем веса внимания (return_attn=True)
            if is_grnn2:
                y, rnn_state, extras, kl_loss = rnn(x, rnn_state, return_attn=True)
            else:
                y, rnn_state, extras = rnn(x, rnn_state, return_attn=True)
                kl_loss = torch.tensor(0.0, device=device, dtype=y.dtype)

            # Обновляем AttnFlow буфер
            attn_vis.update(extras["attn_weights"])

            # Обновляем CKA буфер (берём скрытые состояния верхнего уровня)
            h_for_cka = (
                rnn_state[0] if isinstance(rnn_state, tuple) else rnn_state
            )
            cka_vis.update(h_for_cka)

            # Собираем средние значения gate-ов по слоям
            gate_vals = [
                g.detach().sigmoid().mean().item()
                for g in extras["gates"]
            ]
            gate_buffer.append(gate_vals)

        else:
            # ── Обычный режим
            if is_grnn2:
                y, rnn_state, kl_loss = rnn(x, rnn_state)
            else:
                result       = rnn(x, rnn_state)
                y, rnn_state = result[0], result[1]
                kl_loss      = torch.tensor(0.0, device=device, dtype=y.dtype)

        batch_y.append(y)
        batch_y_gt.append(obs['targets'])
        batch_kl.append(kl_loss)

        # sq_gaps — опционально (если генератор их отдаёт)
        if 'sq_gaps' in obs:
            batch_sq_gaps.append(obs['sq_gaps'])

        # ── Сброс визуализации ────────────────────────────────────────────
        if step >= next_vis_step and logger is not None and VIS_ENABLED:
            attn_vis.log(logger, step=step)
            cka_vis.log(logger, step=step)

            if gate_buffer:
                gate_arr = np.array(gate_buffer)   # [T, n_layers]
                for li in range(rnn.n_layers):
                    logger.track(
                        float(gate_arr[:, li].mean()),
                        name=f"attn_gate/layer_{li}",
                        step=step,
                    )
            gate_buffer.clear()
            next_vis_step += VIS_INTERVAL

        step += gen.n_envs

        # ── Обновление весов раз в rollout_len шагов ──────────────────────
        if step % batch_size == 0:
            y_cat    = torch.cat(batch_y,    dim=0)
            y_gt_cat = torch.cat(batch_y_gt, dim=0)
            m_active = y_gt_cat != CE_ignore_index

            # Основной CE-лосс
            ce_loss = loss_fn(y_cat, y_gt_cat)

            # KL-лосс (только для grnn2)
            kl_mean  = torch.stack(batch_kl).mean()
            kl_scale = get_kl_anneal(step)
            total_loss = ce_loss + kl_scale * kl_mean

            with torch.no_grad():
                logits_active = y_cat[m_active]
                gt_active     = y_gt_cat[m_active]
                acc_all = (logits_active.argmax(dim=-1) == gt_active).float()

                bpc        = ce_loss / ln_2
                perplexity = torch.exp(ce_loss)

            acc = acc_all.mean()

            # ── Метрики по sq_gaps (если доступны) ───────────────────────
            has_gaps = len(batch_sq_gaps) > 0
            if has_gaps:
                sq_gaps    = torch.cat(batch_sq_gaps, dim=0).float()
                # m_active применяем к sq_gaps тоже
                sq_gaps_a  = sq_gaps[m_active] if sq_gaps.shape[0] == y_cat.shape[0] \
                             else sq_gaps

                mask_misses    = sq_gaps_a < 0.0
                acc_miss       = acc_all[mask_misses].mean()   if mask_misses.any()  else acc
                acc_non_miss   = acc_all[~mask_misses].mean()  if (~mask_misses).any() else acc
                gap_mean       = sq_gaps_a[~mask_misses].mean() if (~mask_misses).any() else sq_gaps_a.mean()
                mask_up_half   = sq_gaps_a > gap_mean
                acc_up_half    = acc_all[mask_up_half].mean()  if mask_up_half.any() else acc

            optim.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('⚠ Nan/Inf grad — шаг пропущен')

            p_reset.step()
            if step_lr():
                optim.param_groups[0]['lr'] = get_lr()

            # ── Записываем в Tracker ──────────────────────────────────────
            stat_dict = {
                "Loss":       to_numpy(ce_loss,    copy=False),
                "KL":         to_numpy(kl_mean,    copy=False),
                "KL_scale":   kl_scale,
                "BPC":        to_numpy(bpc,        copy=False),
                "Perplexity": to_numpy(perplexity, copy=False),
                "Acc":        to_numpy(acc,         copy=False),
                "|Grad|":     to_numpy(grad_norm,  copy=False),
                "LR":         get_lr(),
                "T":          1 / p_reset.val,
            }
            if has_gaps:
                stat_dict["Acc-"]  = to_numpy(acc_miss,     copy=False)
                stat_dict["Acc+"]  = to_numpy(acc_non_miss,  copy=False)
                stat_dict["Acc++"] = to_numpy(acc_up_half,   copy=False)
            stats.put(stat_dict)

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear()
            batch_y_gt.clear()
            batch_kl.clear()
            batch_sq_gaps.clear()

        # ── Вывод в консоль ───────────────────────────────────────────────
        if print_stats_schedule.tick(gen.n_envs):
            metrics  = {"global_step": step} | stats.get()
            fps      = fps_counter.fps(n_iters=step, start=True)
            kl_str   = (
                f' KL: {metrics.get("KL", 0):.2e} (x{metrics.get("KL_scale", 0):.2f}) |'
                if is_grnn2 else ''
            )
            gap_str = (
                f' A-: {metrics["Acc-"]:.3f},'
                f' A+: {metrics["Acc+"]:.3f},'
                f' A++: {metrics["Acc++"]:.3f}'
                if "Acc-" in metrics else ''
            )
            print(
                f'[{format_readable_num(step)} / {format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)} fps |'
                f' LR: {int(100*metrics["LR"]/lr)}%'
                f' T: {int(metrics["T"])} |'
                f'{kl_str}'
                f' L: {metrics["Loss"]:.3f},'
                f' A: {metrics["Acc"]:.3f}'
                f'{gap_str}'
            )

        # ── Логирование ───────────────────────────────────────────────────
        if log_stats_schedule.tick(gen.n_envs) and logger is not None:
            fps     = fps_counter.fps(n_iters=step, start=True)
            metrics = {
                "global_step": step,
                "fps":         fps,
            } | stats.get()
            metrics['gen'] = gen.get_stats()
            logger.track(flatten_dict(metrics))

    fps = fps_counter.fps(n_iters=step)
    print(format_readable_num(fps))


if __name__ == "__main__":
    run_experiment(runner=main)