# run_sdq2.py — SDQ-тренировка v2
# + полная поддержка EquilibriumGridRnnCoT с equilibrium-метриками
# + curriculum, sq_gaps метрики, AttnFlow/CKA визуализации
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from datetime import datetime
from collections import defaultdict

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


def main(config):

    # ── Имя запуска ───────────────────────────────────────────────────
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

    # ── Генератор ─────────────────────────────────────────────────────
    gen_cfg = config['gens'][config['gen']]
    gen = StoreDistractQueryGenerator(
        **gen_cfg, n_envs=n_envs,
        seed=rng.integers(1_000_000),
        ignore_index=CE_ignore_index,
    )

    # ── Выбор модели ──────────────────────────────────────────────────
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
        case 'grnn_eq1':
            from knitwork.models.grnn_eq1 import EquilibriumGridRnnCoT
            rnn_fn = EquilibriumGridRnnCoT
        case 'grnn_disc':
            from knitwork.models.grnn_disc import GridRnnNoveltyGate
            rnn_fn = GridRnnNoveltyGate
        case 'grnn_adv_loss':
            from knitwork.models.grnn_adv_loss import GridRnn
            rnn_fn = GridRnn
        case _:
            raise ValueError(f"Неизвестный тип модели: {rnn_type}")

    rnn = rnn_fn(**rnn_cfg, input_size=gen.n_tokens, output_size=gen.V)
    rnn = rnn.to(device=device, dtype=dtype)
    print(
        f'Model is on "{next(rnn.parameters()).device}"'
        f' having "{next(rnn.parameters()).dtype}" dtype'
    )

    # ── Флаги модели ──────────────────────────────────────────────────
    is_grnn2   = rnn_type == 'grnn2'
    is_grnn_eq = rnn_type == 'grnn_eq1'
    vae_active = is_grnn2 and (rnn_cfg.get('vae_latent_dim') is not None)

    print(f'[VAE] {"АКТИВЕН" if vae_active else "ВЫКЛЮЧЕН"} (is_grnn2={is_grnn2})')
    print(f'[EQ]  {"АКТИВЕН" if is_grnn_eq else "ВЫКЛЮЧЕН"}')

    # ── Визуализаторы ─────────────────────────────────────────────────
    attn_vis = AttnFlowVisualizer(
        n_layers=rnn.n_layers, n_columns=rnn.n_columns, buffer_size=100
    )
    cka_vis = CKAVisualizer(
        n_layers=rnn.n_layers, n_columns=rnn.n_columns, buffer_size=50
    )
    next_vis_step = VIS_INTERVAL
    gate_buffer:       list = []

    # Буферы equilibrium-метрик за период визуализации
    eq_iters_buffer:     list[list] = []   # [ [layer0, layer1, ...], ... ]
    eq_delta_buffer:     list[list] = []
    eq_conv_buffer:      list[list] = []
    eq_halt_buffer:      list[list] = []
    eq_ponder_buffer:    list[float] = []
    eq_residual_buffer:  list[float] = []

    # ── Learning rate ──────────────────────────────────────────────────
    lr_cfg = config['lr']
    lr     = lr_cfg['val']
    print(f"Base LR: {lr}")

    wm_lr_cfg, wm_lr_schedule = extracted(lr_cfg['warmup'], 'schedule')
    dc_lr_cfg, dc_lr_schedule = extracted(lr_cfg['decay'],  'schedule')
    wm_lr = DynamicParameter(
        val=1e-5*lr, tar=lr, **wm_lr_cfg,
        scheduler=Scheduler(wm_lr_schedule),
    )
    dc_lr = DynamicParameter(
        val=lr, **dc_lr_cfg,
        scheduler=Scheduler(dc_lr_schedule),
    )

    def get_lr():
        return wm_lr.val if not wm_lr.scheduler.is_infinite else dc_lr.val

    def step_lr():
        return wm_lr.step() if not wm_lr.scheduler.is_infinite else dc_lr.step()

    optim   = torch.optim.RMSprop(rnn.parameters(), lr=get_lr())
    loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=CE_ignore_index)

    # ── KL Annealing ──────────────────────────────────────────────────
    kl_anneal_cfg   = config.get('kl_anneal', {})
    kl_anneal_steps = int(kl_anneal_cfg.get('steps', 50_000))
    kl_anneal_max   = float(kl_anneal_cfg.get('max_weight', 1.0))

    def get_kl_anneal(current_step: int) -> float:
        if not vae_active:
            return 0.0
        if kl_anneal_steps == 0:
            return kl_anneal_max
        return kl_anneal_max * min(1.0, current_step / kl_anneal_steps)

    # ── Расписания ─────────────────────────────────────────────────────
    rollout_len  = config['rollout_len']
    batch_size   = gen.n_envs * rollout_len
    n_steps      = int(config['n_steps'])
    step         = 0

    log_stats_schedule   = Scheduler(int(config['log']['schedule']))
    print_stats_schedule = Scheduler(int(config['log']['print_schedule']))

    curriculum_cfg, curriculum_schedule = extracted(config['curriculum'], 'schedule')
    curriculum_step_schedule = CurriculumScheduler(
        **curriculum_cfg, scheduler=Scheduler(curriculum_schedule)
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
    batch_kl      = []
    # Equilibrium: храним extras из каждого шага rollout
    batch_eq_metrics: list[dict] = []

    # ── Главный цикл ───────────────────────────────────────────────────
    while step < n_steps:
        obs = gen.next()
        obs = {k: to_torch(v, device=device) for k, v in obs.items()}

        rnn_state = rnn.reset_state(rnn_state, obs['reset_mask'])
        x = obs['tokens'].view(-1, 1)

        # ── Capture mode для визуализации ─────────────────────────────
        capture = (step >= next_vis_step - gen.n_envs)

        # ── Forward pass ──────────────────────────────────────────────
        if is_grnn_eq:
            # grnn_eq всегда возвращает extras (нужны для eq_loss)
            y, rnn_state, extras = rnn(x, rnn_state, return_attn=True)
            kl_loss = torch.tensor(0.0, device=device, dtype=y.dtype)
            batch_eq_metrics.append(extras)

            if capture:
                attn_vis.update(extras["attn_weights"])
                h_for_cka = rnn_state[0] if isinstance(rnn_state, tuple) else rnn_state
                cka_vis.update(h_for_cka)

                gate_vals = [
                    g.detach().sigmoid().mean().item()
                    for g in extras["gates"]
                    if g is not None and g.numel() > 1
                ]
                if gate_vals:
                    gate_buffer.append(gate_vals)

                # Буферизуем equilibrium-метрики для визуализации
                n_l = rnn.n_layers
                eq_iters_buffer.append([
                    extras["act_iters"][li].float().mean().item()
                    for li in range(n_l)
                ])
                if extras["eq_delta_norms"]:
                    eq_delta_buffer.append(extras["eq_delta_norms"])
                    eq_conv_buffer.append(extras["eq_convergence_rate"])
                    eq_halt_buffer.append(extras["eq_halt_probs"])
                eq_ponder_buffer.append(extras.get("eq_ponder_cost", 0.0))
                eq_residual_buffer.append(
                    extras["eq_residual_loss"].item()
                    if torch.is_tensor(extras.get("eq_residual_loss"))
                    else 0.0
                )

        elif is_grnn2:
            if capture:
                y, rnn_state, extras, kl_loss = rnn(x, rnn_state, return_attn=True)
                attn_vis.update(extras["attn_weights"])
                h_for_cka = rnn_state[0] if isinstance(rnn_state, tuple) else rnn_state
                cka_vis.update(h_for_cka)
                gate_vals = [
                    g.detach().sigmoid().mean().item()
                    for g in extras["gates"]
                ]
                gate_buffer.append(gate_vals)
            else:
                y, rnn_state, kl_loss = rnn(x, rnn_state)
        else:
            if capture:
                result = rnn(x, rnn_state, return_attn=True)
                y, rnn_state, extras = result[0], result[1], result[2]
                kl_loss = torch.tensor(0.0, device=device, dtype=y.dtype)
                attn_vis.update(extras["attn_weights"])
                h_for_cka = rnn_state[0] if isinstance(rnn_state, tuple) else rnn_state
                cka_vis.update(h_for_cka)
                gate_vals = [
                    g.detach().sigmoid().mean().item()
                    for g in extras["gates"]
                ]
                gate_buffer.append(gate_vals)
            else:
                result       = rnn(x, rnn_state)
                y, rnn_state = result[0], result[1]
                kl_loss      = torch.tensor(0.0, device=device, dtype=y.dtype)

        batch_y.append(y)
        batch_y_gt.append(obs['targets'])
        batch_sq_gaps.append(obs['sq_gaps'])
        batch_kl.append(kl_loss)

        # ── Сброс визуализации ─────────────────────────────────────────
        if step >= next_vis_step and logger is not None and VIS_ENABLED:
            attn_vis.log(logger, step=step)
            cka_vis.log(logger, step=step)

            # Gate stats
            if gate_buffer:
                gate_arr = np.array(gate_buffer)   # [T, n_layers]
                for li in range(min(gate_arr.shape[1], rnn.n_layers)):
                    logger.track(
                        float(gate_arr[:, li].mean()),
                        name=f"attn_gate/layer_{li}",
                        step=step,
                    )
            gate_buffer.clear()

            # ── Equilibrium визуализация ───────────────────────────────
            if is_grnn_eq and eq_iters_buffer:
                iters_arr = np.array(eq_iters_buffer)   # [T, n_layers]

                # 1. Средние итерации по слоям
                for li in range(rnn.n_layers):
                    logger.track(
                        float(iters_arr[:, li].mean()),
                        name=f"eq/mean_iters/layer_{li}",
                        step=step,
                    )

                # 2. Невязки и convergence rate
                if eq_delta_buffer:
                    delta_arr = np.array(eq_delta_buffer)   # [T, n_layers]
                    conv_arr  = np.array(eq_conv_buffer)
                    halt_arr  = np.array(eq_halt_buffer)
                    for li in range(rnn.n_layers):
                        logger.track(
                            float(delta_arr[:, li].mean()),
                            name=f"eq/delta_norm/layer_{li}",
                            step=step,
                        )
                        logger.track(
                            float(conv_arr[:, li].mean()),
                            name=f"eq/convergence_rate/layer_{li}",
                            step=step,
                        )
                        logger.track(
                            float(halt_arr[:, li].mean()),
                            name=f"eq/halt_prob/layer_{li}",
                            step=step,
                        )

                # 3. Ponder cost и residual loss
                if eq_ponder_buffer:
                    logger.track(
                        float(np.mean(eq_ponder_buffer)),
                        name="eq/ponder_cost_mean",
                        step=step,
                    )
                if eq_residual_buffer:
                    logger.track(
                        float(np.mean(eq_residual_buffer)),
                        name="eq/residual_loss_mean",
                        step=step,
                    )

                # 4. Histogram итераций (если logger поддерживает)
                if hasattr(logger, 'track_histogram'):
                    all_iters = iters_arr.flatten()
                    logger.track_histogram(
                        all_iters, name="eq/iters_hist", step=step
                    )

            eq_iters_buffer.clear()
            eq_delta_buffer.clear()
            eq_conv_buffer.clear()
            eq_halt_buffer.clear()
            eq_ponder_buffer.clear()
            eq_residual_buffer.clear()

            next_vis_step += VIS_INTERVAL

        step += gen.n_envs

        # ── Обновление весов ───────────────────────────────────────────
        if step % batch_size == 0:
            y_cat    = torch.cat(batch_y,    dim=0)
            y_gt_cat = torch.cat(batch_y_gt, dim=0)
            sq_gaps  = torch.cat(batch_sq_gaps, dim=0).float()
            m_active = y_gt_cat != CE_ignore_index

            ce_loss = loss_fn(y_cat, y_gt_cat)

            # KL
            kl_mean  = torch.stack(batch_kl).mean()
            kl_scale = get_kl_anneal(step)

            # Equilibrium auxiliary loss
            eq_loss_val  = torch.tensor(0.0, device=device, dtype=y_cat.dtype)
            eq_iters_mean_per_layer  = []
            eq_delta_mean_per_layer  = []
            eq_conv_mean_per_layer   = []
            eq_halt_mean_per_layer   = []
            eq_ponder_mean           = 0.0
            eq_residual_mean         = 0.0

            if is_grnn_eq and batch_eq_metrics:
                # Суммируем auxiliary losses через rollout
                ponder_list   = []
                residual_list = []
                # Сборка per-layer статистик
                iters_per_layer  = defaultdict(list)
                delta_per_layer  = defaultdict(list)
                conv_per_layer   = defaultdict(list)
                halt_per_layer   = defaultdict(list)

                for em in batch_eq_metrics:
                    p = em.get("eq_ponder_cost_tensor")
                    r = em.get("eq_residual_loss")
                    if p is not None:
                        ponder_list.append(p)
                    if r is not None:
                        residual_list.append(r)

                    for li, it in enumerate(em.get("act_iters", [])):
                        iters_per_layer[li].append(it.float().mean().item())
                    for li, d in enumerate(em.get("eq_delta_norms", [])):
                        delta_per_layer[li].append(d)
                    for li, c in enumerate(em.get("eq_convergence_rate", [])):
                        conv_per_layer[li].append(c)
                    for li, hp in enumerate(em.get("eq_halt_probs", [])):
                        halt_per_layer[li].append(hp)

                if ponder_list:
                    ponder_mean_t = torch.stack(ponder_list).mean()
                    eq_ponder_mean = ponder_mean_t.item()
                    eq_loss_val = eq_loss_val + rnn.act_loss_weight * ponder_mean_t

                if residual_list:
                    residual_mean_t = torch.stack(residual_list).mean()
                    eq_residual_mean = residual_mean_t.item()
                    eq_loss_val = eq_loss_val + rnn.eq_residual_weight * residual_mean_t

                for li in range(rnn.n_layers):
                    eq_iters_mean_per_layer.append(
                        float(np.mean(iters_per_layer[li])) if iters_per_layer[li] else 0.0
                    )
                    eq_delta_mean_per_layer.append(
                        float(np.mean(delta_per_layer[li])) if delta_per_layer[li] else 0.0
                    )
                    eq_conv_mean_per_layer.append(
                        float(np.mean(conv_per_layer[li])) if conv_per_layer[li] else 0.0
                    )
                    eq_halt_mean_per_layer.append(
                        float(np.mean(halt_per_layer[li])) if halt_per_layer[li] else 0.0
                    )

            total_loss = ce_loss + kl_scale * kl_mean + eq_loss_val

            with torch.no_grad():
                acc = (
                    y_cat[m_active].argmax(dim=-1) == y_gt_cat[m_active]
                ).float()

            mask_misses  = sq_gaps < 0.0
            acc_miss     = acc[mask_misses].mean()    if mask_misses.any()  else torch.tensor(float('nan'))
            acc_non_miss = acc[~mask_misses].mean()   if (~mask_misses).any() else torch.tensor(float('nan'))
            sq_non_miss  = sq_gaps[~mask_misses]
            acc_up_half  = (
                acc[sq_gaps > sq_non_miss.mean()].mean()
                if sq_non_miss.numel() > 0 else torch.tensor(float('nan'))
            )
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

            # ── Tracker ───────────────────────────────────────────────
            stat_dict = {
                "Loss":      to_numpy(ce_loss,     copy=False),
                "Acc":       to_numpy(acc,          copy=False),
                "Acc-":      to_numpy(acc_miss,     copy=False),
                "Acc+":      to_numpy(acc_non_miss, copy=False),
                "Acc++":     to_numpy(acc_up_half,  copy=False),
                "|Grad|":    to_numpy(grad_norm,    copy=False),
                "LR":        get_lr(),
            }
            if vae_active:
                stat_dict["KL"]       = to_numpy(kl_mean, copy=False)
                stat_dict["KL_scale"] = kl_scale

            # Equilibrium метрики в tracker
            if is_grnn_eq:
                stat_dict["EQ_loss"]     = float(eq_loss_val.item())
                stat_dict["EQ_ponder"]   = eq_ponder_mean
                stat_dict["EQ_residual"] = eq_residual_mean
                for li in range(rnn.n_layers):
                    stat_dict[f"EQ_iters_L{li}"]  = eq_iters_mean_per_layer[li] if li < len(eq_iters_mean_per_layer) else 0.0
                    stat_dict[f"EQ_delta_L{li}"]  = eq_delta_mean_per_layer[li] if li < len(eq_delta_mean_per_layer) else 0.0
                    stat_dict[f"EQ_conv_L{li}"]   = eq_conv_mean_per_layer[li]  if li < len(eq_conv_mean_per_layer)  else 0.0
                    stat_dict[f"EQ_halt_L{li}"]   = eq_halt_mean_per_layer[li]  if li < len(eq_halt_mean_per_layer)  else 0.0

            stats.put(stat_dict)

            rnn_state = rnn.detach_state(rnn_state)
            batch_y.clear()
            batch_y_gt.clear()
            batch_sq_gaps.clear()
            batch_kl.clear()
            batch_eq_metrics.clear()

        # ── Curriculum ────────────────────────────────────────────────
        if curriculum_step_schedule.tick(metrics=stats, n_steps=gen.n_envs):
            K = 10
            dT, dp_store, dp_query = 1.0/K, -0.0014/K, -0.0005/K
            gen.set_metaparams(
                T       = gen.T + dT,
                p_store = max(gen.p_store + dp_store, 0.10),
                p_query = max(gen.p_query + dp_query, 0.25),
            )

        # ── Вывод в консоль ───────────────────────────────────────────
        if print_stats_schedule.tick(gen.n_envs):
            metrics = {"global_step": step} | stats.get()
            fps     = fps_counter.fps(n_iters=step, start=True)

            kl_str = (
                f' KL:{metrics.get("KL",0):.2e}(x{metrics.get("KL_scale",0):.2f}) |'
                if vae_active else ''
            )
            eq_str = ''
            if is_grnn_eq:
                iters_str = ' '.join(
                    f'L{li}:{metrics.get(f"EQ_iters_L{li}", 0):.1f}'
                    for li in range(rnn.n_layers)
                )
                conv_str = ' '.join(
                    f'{metrics.get(f"EQ_conv_L{li}", 0):.2f}'
                    for li in range(rnn.n_layers)
                )
                eq_str = (
                    f' EQ[iters:{iters_str}]'
                    f' EQ[conv:{conv_str}]'
                    f' EQ_res:{metrics.get("EQ_residual", 0):.2e} |'
                )

            print(
                f'[{format_readable_num(step)} /'
                f' {format_readable_num(n_steps, frac=0)}]'
                f' {format_readable_num(fps, frac=0)} fps |'
                f' LR:{int(100*metrics["LR"]/lr)}% |'
                f'{kl_str}'
                f'{eq_str}'
                f' L:{metrics["Loss"]:.3f},'
                f' A:{metrics["Acc"]:.3f}'
                f' A-:{metrics["Acc-"]:.3f},'
                f' A+:{metrics["Acc+"]:.3f},'
                f' A++:{metrics["Acc++"]:.3f}'
            )

        # ── Логирование ───────────────────────────────────────────────
        if log_stats_schedule.tick(gen.n_envs) and logger is not None:
            fps     = fps_counter.fps(n_iters=step, start=True)
            metrics = {
                "global_step":   step,
                "fps":           fps,
                "curr_step":     curriculum_step_schedule.cnt_accepted,
                "curr_schedule": curriculum_step_schedule.scheduler.schedule,
            } | stats.get()
            metrics['gen'] = gen.get_stats()

            # Детальные eq-метрики в logger напрямую (не через flatten)
            if is_grnn_eq:
                for li in range(rnn.n_layers):
                    metrics[f"eq/mean_iters/layer_{li}"]      = metrics.pop(f"EQ_iters_L{li}", 0.0)
                    metrics[f"eq/delta_norm/layer_{li}"]       = metrics.pop(f"EQ_delta_L{li}", 0.0)
                    metrics[f"eq/convergence_rate/layer_{li}"] = metrics.pop(f"EQ_conv_L{li}",  0.0)
                    metrics[f"eq/halt_prob/layer_{li}"]        = metrics.pop(f"EQ_halt_L{li}",  0.0)
                metrics["eq/ponder_cost"]   = metrics.pop("EQ_ponder",   0.0)
                metrics["eq/residual_loss"] = metrics.pop("EQ_residual", 0.0)
                metrics["eq/total_loss"]    = metrics.pop("EQ_loss",     0.0)

            logger.track(flatten_dict(metrics))

    fps = fps_counter.fps(n_iters=step)
    print(format_readable_num(fps))


if __name__ == "__main__":
    run_experiment(runner=main)