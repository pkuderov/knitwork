# knitwork/exps/sdq/run_lru_grnn.py
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from datetime import datetime

from knitwork.common.config import extracted
from knitwork.common.curriculum import CurriculumScheduler
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
from knitwork.gens.sdq import StoreDistractQueryGenerator

from knitwork.visualization.attn_flow import AttnFlowVisualizer
from knitwork.visualization.cka import CKAVisualizer

VIS_INTERVAL = 10_000_000


# ─────────────────────────────────────────────────────────────────────────────
# LRU-специфичные утилиты логирования
# ─────────────────────────────────────────────────────────────────────────────

def log_lru_spectrum(rnn, logger, step: int):
    """
    Логирует распределение |λ| = exp(−exp(ν)) для каждой LRU-ячейки.

    |λ| близко к 1 → долгосрочная память (медленная динамика).
    |λ| близко к 0 → кратковременная память (быстрая динамика).
    Diversity (entropy) показывает, насколько разнообразны временны́е масштабы.
    """
    for li, row_cells in enumerate(rnn.cells):
        for ci, cell in enumerate(row_cells):
            with torch.no_grad():
                r = torch.exp(-torch.exp(cell.nu_log))
            logger.track(float(r.mean()), name=f"lru/l{li}c{ci}_mean",    step=step)
            logger.track(float(r.min()),  name=f"lru/l{li}c{ci}_min",     step=step)
            logger.track(float(r.max()),  name=f"lru/l{li}c{ci}_max",     step=step)
            r_c     = r.clamp(1e-6, 1 - 1e-6)
            entropy = -(r_c * r_c.log() + (1 - r_c) * (1 - r_c).log()).mean()
            logger.track(float(entropy),  name=f"lru/l{li}c{ci}_entropy", step=step)


def log_assoc_matrix(rnn, logger, step: int, h_state: torch.Tensor, n_show: int = 8):
    """
    Логирует матрицу косинусных сходств Re-части состояния (last layer, col 0).
    Помогает видеть разделение паттернов в ассоциативной памяти.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        from PIL import Image as PILImage
    except ImportError:
        return

    z   = h_state[-1, 0, :n_show, :rnn.hidden_size]
    z_n = torch.nn.functional.normalize(z.float().detach().cpu(), dim=-1)
    sim = (z_n @ z_n.T).numpy()

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(f"Cosine sim (step {step // 1000}k)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    pil_img = PILImage.open(buf)

    try:
        from aim import Image as AimImage
        logger.track(AimImage(pil_img), name="assoc/cosine_matrix", step=step)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────────────────────────────────────────

def main(config):
    import os
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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

    # ── Генератор ─────────────────────────────────────────────────────────
    gen_cfg = config['gens'][config['gen']]
    gen = StoreDistractQueryGenerator(
        **gen_cfg, n_envs=n_envs,
        seed=rng.integers(1_000_000),
        ignore_index=CE_ignore_index,
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
        case 'grnn_err':
            from knitwork.models.grnn_err import GridRnn
            rnn_fn = GridRnn
        case 'hgrnn':
            from knitwork.models.hgrnn import HopfieldGridRnn
            rnn_fn = HopfieldGridRnn
        case 'grnn_fw':
            from knitwork.models.grnn_fw import GridRnnFW
            rnn_fn = GridRnnFW
        case 'grnn_reservoir':
            from knitwork.models.grnn_reservoir import GridRnnReservoir
            rnn_fn = GridRnnReservoir
        case 'grnn_hgrn':
            from knitwork.models.hgrn_grnn import HGRN_GridRnn
            rnn_fn = HGRN_GridRnn
        case 'grnn2':
            from knitwork.models.grnn2 import GridRnn2
            rnn_fn = GridRnn2
        case 'grnn_engram':
            from knitwork.models.engram_grnn import EngramGridRnn
            rnn_fn = EngramGridRnn
        case 'grnn_loss':
            from knitwork.models.grnn_loss import GridRnnLoss
            rnn_fn = GridRnnLoss
        case 'grnn_eq':
            from knitwork.models.grnn_eq import EquilibriumGridRnnCoT
            rnn_fn = EquilibriumGridRnnCoT
        case 'grnn_disc':
            from knitwork.models.grnn_disc import GridRnnNoveltyGate
            rnn_fn = GridRnnNoveltyGate
        case 'grnn_adv_loss':
            from knitwork.models.grnn_adv_loss import GridRnn
            rnn_fn = GridRnn
        case 'grnn_lru':
            from knitwork.models.grnn_lru import GridLRU
            rnn_fn = GridLRU
        case 'grnn_lru_wide':
            from knitwork.models.grnn_lru import GridLRU
            rnn_fn = GridLRU
        case 'grnn_lru_hop':
            from knitwork.models.hgrnn_lru import HopfieldGridLRU
            rnn_fn = HopfieldGridLRU
        case _:
            raise ValueError(f"Неизвестный тип модели: {rnn_type}")

    rnn = rnn_fn(**rnn_cfg, input_size=gen.n_tokens, output_size=gen.V)
    rnn = rnn.to(device=device, dtype=dtype)
    print(
        f'Model: "{next(rnn.parameters()).device}"'
        f' | dtype "{next(rnn.parameters()).dtype}"'
    )

    is_lru_hop = (rnn_type == 'grnn_lru_hop')

    # ── Визуализаторы ─────────────────────────────────────────────────────
    attn_vis = AttnFlowVisualizer(
        n_layers=rnn.n_layers, n_columns=rnn.n_columns, buffer_size=100,
    )
    cka_vis = CKAVisualizer(
        n_layers=rnn.n_layers, n_columns=rnn.n_columns, buffer_size=50,
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
        val=1e-5 * lr, tar=lr, **wm_lr_cfg,
        scheduler=Scheduler(wm_lr_schedule),
    )
    dc_lr = DynamicParameter(
        val=lr, **dc_lr_cfg,
        scheduler=Scheduler(dc_lr_schedule),
    )

    def get_lr() -> float:
        return wm_lr.val if not wm_lr.scheduler.is_infinite else dc_lr.val

    def step_lr() -> bool:
        return wm_lr.step() if not wm_lr.scheduler.is_infinite else dc_lr.step()

    # AdamW — рекомендуется для LRU: меньше чувствителен к lr,
    # weight_decay не затрагивает nu_log/theta_log (скаляры, не матрицы)
    optim   = torch.optim.AdamW(rnn.parameters(), lr=get_lr(), weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)

    # ── Ассоциативный лосс — annealing ────────────────────────────────────
    assoc_cfg        = config.get('assoc_loss', {})
    assoc_enabled    = assoc_cfg.get('enabled', False) and is_lru_hop
    assoc_steps      = int(assoc_cfg.get('steps',       80_000))
    assoc_max_weight = float(assoc_cfg.get('max_weight', 0.3))

    def get_assoc_weight(current_step: int) -> float:
        if not assoc_enabled or assoc_steps == 0:
            return assoc_max_weight if assoc_enabled else 0.0
        return assoc_max_weight * min(1.0, current_step / assoc_steps)

    # ── Расписания ────────────────────────────────────────────────────────
    rollout_len  = config['rollout_len']
    batch_size   = gen.n_envs * rollout_len
    n_steps      = int(config['n_steps'])
    step         = 0

    log_stats_schedule   = Scheduler(int(config['log']['schedule']))
    print_stats_schedule = Scheduler(int(config['log']['print_schedule']))

    curriculum_cfg, curriculum_schedule = extracted(config['curriculum'], 'schedule')
    curriculum_step_schedule = CurriculumScheduler(
        **curriculum_cfg, scheduler=Scheduler(curriculum_schedule),
    )

    logger = create_logger(config)
    if logger is not None:
        if hasattr(logger, 'name'):
            logger.name = run_name
        elif hasattr(logger, 'run') and hasattr(logger.run, 'name'):
            logger.run.name = run_name

    stats       = Tracker(lr=2e-4)
    fps_counter = FpsCounter()

    rnn_state     = None
    batch_y       = []
    batch_y_gt    = []
    batch_sq_gaps = []
    batch_assoc   = []

    # ── Главный цикл ──────────────────────────────────────────────────────
    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}

        # reset_state: умножение на keep-маску, без clone, без detach
        rnn_state = rnn.reset_state(rnn_state, obs['reset_mask'])

        x          = obs['tokens'].view(-1, 1)
        store_mask = obs.get('store_mask', None)
        query_mask = obs.get('query_mask', None)

        # capture=True только 1 раз за VIS_INTERVAL — добавляет return_attn,
        # не меняет ничего в логике обучения
        capture = (
            VIS_ENABLED
            and logger is not None
            and step >= next_vis_step - gen.n_envs
        )

        # ── Forward pass ──────────────────────────────────────────────────
        # Единый вызов — capture влияет только на return_attn
        if is_lru_hop:
            result = rnn(
                x, rnn_state,
                return_attn=capture,
                return_assoc_loss=assoc_enabled,
                store_mask=store_mask,
                query_mask=query_mask,
            )
            # Распаковка: порядок определяется комбинацией флагов
            if capture and assoc_enabled:
                y, rnn_state, extras, assoc_loss = result
            elif capture:
                y, rnn_state, extras = result
                assoc_loss = torch.tensor(0.0, device=device, dtype=dtype)
            elif assoc_enabled:
                y, rnn_state, assoc_loss = result
                extras = None
            else:
                y, rnn_state = result
                assoc_loss = torch.tensor(0.0, device=device, dtype=dtype)
                extras = None

            # Буферизуем визуализационные данные (только при capture)
            if capture and extras is not None:
                # extras["attn_weights"]: list[Tensor(num_heads, batch, cols, cols)]
                # AttnFlowVisualizer.update() ожидает list[(cols, cols)] — усредняем
                attn_weights_2d = [
                    aw.detach().float().mean(dim=(0, 1))   # (num_heads, batch, cols, cols) → (cols, cols)
                    for aw in extras["attn_weights"]
                ]
                attn_vis.update(attn_weights_2d)

                # CKAVisualizer.update() ожидает сырой тензор с .detach() —
                # передаём rnn_state как есть, визуализатор сам его разбирает
                cka_vis.update(rnn_state.detach())

                gate_buffer.append([
                    g.detach().sigmoid().mean().item()
                    for g in extras["gates"]
                ])

        else:
            y, rnn_state = rnn(x, rnn_state)
            assoc_loss = torch.tensor(0.0, device=device, dtype=dtype)

        # ✅ batch и step обновляются ВСЕГДА — независимо от capture
        batch_y.append(y)
        batch_y_gt.append(obs['targets'])
        batch_sq_gaps.append(obs['sq_gaps'])
        batch_assoc.append(assoc_loss)
        step += gen.n_envs

        # ── Визуализация ──────────────────────────────────────────────────
        if step >= next_vis_step and logger is not None and VIS_ENABLED:
            attn_vis.log(logger, step=step)
            cka_vis.log(logger, step=step)

            if gate_buffer:
                gate_arr = np.array(gate_buffer)
                for li in range(rnn.n_layers):
                    logger.track(
                        float(gate_arr[:, li].mean()),
                        name=f"attn_gate/layer_{li}",
                        step=step,
                    )
                gate_buffer.clear()

            if is_lru_hop:
                log_lru_spectrum(rnn, logger, step=step)
                if rnn_state is not None:
                    log_assoc_matrix(rnn, logger, step=step, h_state=rnn_state)

            next_vis_step += VIS_INTERVAL

        # ── Обновление весов ──────────────────────────────────────────────
        if step % batch_size == 0:
            y_cat    = torch.cat(batch_y,       dim=0)
            y_gt_cat = torch.cat(batch_y_gt,    dim=0)
            sq_gaps  = torch.cat(batch_sq_gaps, dim=0).float()
            m_active = y_gt_cat != CE_ignore_index

            ce_loss    = loss_fn(y_cat, y_gt_cat)
            assoc_mean = torch.stack(batch_assoc).mean()
            assoc_w    = get_assoc_weight(step)
            total_loss = ce_loss + assoc_w * assoc_mean

            with torch.no_grad():
                acc = (
                    y_cat[m_active].argmax(dim=-1) == y_gt_cat[m_active]
                ).float()

            mask_misses  = sq_gaps < 0.0
            acc_miss = (
                acc[mask_misses].mean()
                if mask_misses.any()
                else acc.new_tensor(float('nan'))
            )
            acc_non_miss = (
                acc[~mask_misses].mean()
                if (~mask_misses).any()
                else acc.new_tensor(float('nan'))
            )

            non_miss_gaps = sq_gaps[~mask_misses]
            if non_miss_gaps.numel() > 10:
                median_gap   = non_miss_gaps.median()
                acc_up_half  = acc[sq_gaps > median_gap].mean()
                acc_low_half = acc[(sq_gaps <= median_gap) & (~mask_misses)].mean()
            else:
                acc_up_half  = acc.new_tensor(float('nan'))
                acc_low_half = acc.new_tensor(float('nan'))

            acc = acc.mean()

            optim.zero_grad()
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optim.step()
            else:
                print('⚠ Nan/Inf grad — шаг пропущен')

            if step_lr():
                optim.param_groups[0]['lr'] = get_lr()

            # ✅ detach ТОЛЬКО здесь — после backward()
            # Это единственное место Truncated BPTT:
            # граф обрывается раз в rollout_len шагов
            rnn_state = rnn.detach_state(rnn_state)

            stat_dict = {
                "Loss":    to_numpy(ce_loss,      copy=False),
                "Acc":     to_numpy(acc,           copy=False),
                "Acc-":    to_numpy(acc_miss,      copy=False),
                "Acc+":    to_numpy(acc_non_miss,  copy=False),
                "Acc++":   to_numpy(acc_up_half,   copy=False),
                "Acc+low": to_numpy(acc_low_half,  copy=False),
                "|Grad|":  to_numpy(grad_norm,     copy=False),
                "LR":      get_lr(),
            }
            if assoc_enabled:
                stat_dict["L_assoc"]      = to_numpy(assoc_mean, copy=False)
                stat_dict["assoc_weight"] = assoc_w

            stats.put(stat_dict)

            batch_y.clear()
            batch_y_gt.clear()
            batch_sq_gaps.clear()
            batch_assoc.clear()

        # ── Curriculum ────────────────────────────────────────────────────
        if curriculum_step_schedule.tick(metrics=stats, n_steps=gen.n_envs):
            K = 10
            gen.set_metaparams(
                T       = gen.T + 1.0 / K,
                p_store = max(gen.p_store - 0.0014 / K, 0.10),
                p_query = max(gen.p_query - 0.0005 / K, 0.25),
            )

        # ── Вывод в консоль ───────────────────────────────────────────────
        if print_stats_schedule.tick(gen.n_envs):
            metrics = {"global_step": step} | stats.get()
            fps     = fps_counter.fps(n_iters=step, start=True)
            assoc_str = (
                f' L_a:{metrics.get("L_assoc", 0):.3f}'
                f'(x{metrics.get("assoc_weight", 0):.2f}) |'
                if assoc_enabled else ''
            )
            print(
                f'[{format_readable_num(step)} /'
                f' {format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)} fps |'
                f' LR:{int(100 * metrics["LR"] / lr)}% |'
                f'{assoc_str}'
                f' L:{metrics["Loss"]:.3f}'
                f' A:{metrics["Acc"]:.3f}'
                f' A-:{metrics["Acc-"]:.3f}'
                f' A+:{metrics["Acc+"]:.3f}'
                f' A++:{metrics["Acc++"]:.3f}'
            )

        # ── Логирование ───────────────────────────────────────────────────
        if log_stats_schedule.tick(gen.n_envs) and logger is not None:
            fps     = fps_counter.fps(n_iters=step, start=True)
            metrics = {
                "global_step":   step,
                "fps":           fps,
                "curr_step":     curriculum_step_schedule.cnt_accepted,
                "curr_schedule": curriculum_step_schedule.scheduler.schedule,
            } | stats.get()
            metrics['gen'] = gen.get_stats()
            logger.track(flatten_dict(metrics))


if __name__ == "__main__":
    run_experiment(runner=main)