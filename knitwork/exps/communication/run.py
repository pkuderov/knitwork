"""Full-budget, resumable Text8/SDQ training for matched communication controls."""

import copy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from knitwork.common.dynamic_param import DynamicParameter
from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging_alt import start_logger
from knitwork.common.scheduler import create_scheduler
from knitwork.common.torch import DynamicLearningRate, to_loggable_metrics
from knitwork.common.utils import CE_ignore_index, get_device
from knitwork.exps.communication.evaluation import evaluate_fixed_gaps, evaluate_sdq, evaluate_text
from knitwork.exps.communication.protocol import (
    accounting, atomic_torch_save, config_fingerprint, file_digest, make_model,
    parameter_state, resolve_config, restore_parameter, restore_rng,
    restore_scheduler, rng_state, scheduler_state, tensor_digest,
)
from knitwork.exps.communication.runtime import PauseRequested, StopController
from knitwork.exps.sdq.run import sq_gap_metrics
from knitwork.gens.sdq import StoreDistractQueryGenerator
from knitwork.gens.text import TextGenerator, load_dataset, split_train_val_test, tokenize_splits


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def unwrap(model):
    return getattr(model, '_orig_mod', model)


def cpu_weights(model):
    return {key: value.detach().cpu().clone() for key, value in unwrap(model).state_dict().items()}


def generator_state(gen, task):
    if task == 'text8':
        return {'pos': gen.pos.clone(), 'rng': gen.rng.get_state()}
    return copy.deepcopy(gen.__dict__)


def restore_generator(gen, saved, task):
    if task == 'text8':
        gen.pos.copy_(saved['pos'])
        gen.rng.set_state(saved['rng'].cpu())
    else:
        gen.__dict__.update(saved)


def write_json(path, value):
    temporary = Path(path).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(config, *, stop_after_updates=None):
    cfg = resolve_config(config)
    device = get_device(cfg['device'])
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(cfg.get('deterministic', False))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    directory = Path(cfg['output_dir']) / cfg['run_id']
    resume = cfg.get('resume')
    if resume == 'auto':
        resume = directory / 'latest.pt'
        if not resume.exists():
            resume = directory / 'previous.pt'
        if not resume.exists():
            resume = None
    if directory.exists() and (directory / 'metadata.json').exists() and not resume:
        raise FileExistsError(f'Existing run: {directory}; set --resume=auto or choose another output_dir')
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = torch.load(resume, map_location='cpu', weights_only=False) if resume else None
    if snapshot and snapshot['config_fingerprint'] != config_fingerprint(cfg):
        raise ValueError('Checkpoint/config mismatch; task, model, data, seed, recipe and budgets must match')

    validation = test = None
    if cfg['exp'] == 'text8':
        path = Path(os.path.expandvars(cfg['dataset']['path'])).expanduser()
        raw = load_dataset(path)
        splits = split_train_val_test(raw, cfg['dataset']['val_size'], cfg['dataset']['test_size'])
        (train, validation, test), charset = tokenize_splits(*splits)
        data_identity = {'sha256': file_digest(path), 'sizes': [len(part) for part in splits], 'charset': charset.tolist()}
        gen = TextGenerator(
            train, n_envs=cfg['n_envs'], device=device, seed=cfg['data_seed'], ignore_index=CE_ignore_index,
        ).to(device)
        model = make_model(cfg, device=device, vocab=len(charset))
    elif cfg['exp'] == 'sdq':
        gen = StoreDistractQueryGenerator(
            **cfg['generator'], n_envs=cfg['n_envs'], seed=cfg['data_seed'], ignore_index=CE_ignore_index,
        )
        data_identity = {'generator': cfg['generator'], 'seed': cfg['data_seed']}
        model = make_model(cfg, device=device, vocab=gen.n_tokens)
    else:
        raise ValueError('exp must be text8 or sdq')
    if snapshot and snapshot['data_identity'] != data_identity:
        raise ValueError('Dataset or generator changed since the checkpoint')
    cfg['accounting'] = accounting(model)
    low, high = cfg['parameter_range']
    if not low <= cfg['accounting']['parameters_nominal_trainable'] <= high:
        raise ValueError('Parameter count is outside the specified 1M range')

    source_root = Path(__file__).resolve().parents[3]
    sources = [
        'knitwork/models/grnn_core.py', 'knitwork/models/grnn_communication.py',
        'knitwork/models/wrappers.py', 'knitwork/gens/sdq.py', 'knitwork/gens/text.py',
        'knitwork/exps/communication/run.py', 'knitwork/exps/communication/protocol.py',
        'knitwork/exps/communication/evaluation.py', 'knitwork/exps/communication/runtime.py',
        'knitwork/common/dynamic_param.py', 'knitwork/common/scheduler.py',
        'knitwork/common/logging_alt.py', 'knitwork/common/tracking.py',
        'knitwork/common/torch.py', 'knitwork/common/utils.py', 'knitwork/models/utils.py',
    ]
    hashes = {name: file_digest(source_root / name) for name in sources}
    if snapshot and snapshot['source_hashes'] != hashes:
        raise ValueError('Training/model source changed since checkpoint; exact resume is not guaranteed')
    metadata = {
        'run_id': cfg['run_id'], 'status': 'running', 'config_fingerprint': config_fingerprint(cfg),
        'source_hashes': hashes, 'data_identity': data_identity,
        'initialization_sha256': tensor_digest(model.state_dict()), 'accounting': cfg['accounting'],
        'hardware': str(device), 'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
        'requested_tokens': int(cfg['n_steps']), 'effective_tokens': cfg['effective_tokens'],
        'training_seconds_scope': 'synchronized training loop including Python/compile overhead; excludes evaluation, checkpoint writing and remote logging',
    }
    if device.type == 'cuda':
        metadata['hardware'] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        metadata['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata['git_commit'] = None
    write_json(directory / 'metadata.json', metadata)
    (directory / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

    optim = torch.optim.RMSprop(model.parameters(), lr=cfg['lr']['val'])
    lr = DynamicLearningRate(**copy.deepcopy(cfg['lr']))
    lr.connect_to_optimiser(optim)
    reset = None
    if cfg['exp'] == 'text8':
        reset_cfg = copy.deepcopy(cfg['reset_prob'])
        reset_cfg['val'] /= cfg['rollout_len']
        reset_cfg['tar'] /= cfg['rollout_len']
        reset = DynamicParameter(**reset_cfg)
    curriculum = create_scheduler(cfg.get('curriculum', {}).get('schedule'))
    eval_schedule = create_scheduler(cfg['eval']['schedule'])
    state = None
    step = updates = skipped = 0
    training_seconds = evaluation_seconds = checkpoint_seconds = 0.0
    best = {'loss': float('inf'), 'step': 0, 'weights': None}
    final_val = None
    final_test = best_test = probes = None
    last_eval_step = -1
    last_checkpoint_step = 0
    pending_evaluation = False
    control = StopController(cfg.get('runtime', {}).get('deadline'))
    last_progress = 0.0

    def progress(phase, *, force=False):
        nonlocal last_progress
        now = time.time()
        if force or now - last_progress >= 10:
            row = {
                'run_id': cfg['run_id'], 'pid': os.getpid(), 'phase': phase,
                'tokens': step, 'updates': updates, 'updated_at': now,
                'training_seconds': training_seconds,
                'last_checkpoint_tokens': last_checkpoint_step,
            }
            if device.type == 'cuda':
                row['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            write_json(directory / 'progress.json', row)
            last_progress = now

    metrics_path = directory / 'metrics.jsonl'
    if snapshot and metrics_path.exists():
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        metrics_path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows if row['step'] <= snapshot['step']))

    def local_log(log_step, *, scalars, figures):
        row = {'step': log_step, 'metrics': {key: float(value) for key, value in scalars.items() if np.isfinite(value)}}
        with metrics_path.open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        print(f"{cfg['run_id']} tokens={log_step:,} loss={scalars.get('Loss', float('nan')):.4f}", flush=True)

    # Tracking SDK initialization must not change the training randomness.
    before_logger_rng = rng_state()
    logger = start_logger(cfg, tracker=cfg['trackers'], suppress_printing=True, callbacks=[local_log])
    restore_rng(before_logger_rng)

    if snapshot:
        model.load_state_dict(snapshot['model'])
        optim.load_state_dict(snapshot['optimizer'])
        restore_parameter(lr, snapshot['lr'])
        if reset is not None:
            restore_parameter(reset, snapshot['reset'])
        restore_scheduler(curriculum, snapshot['curriculum'])
        restore_scheduler(eval_schedule, snapshot['eval_schedule'])
        restore_generator(gen, snapshot['generator'], cfg['exp'])
        state = snapshot['recurrent_state']
        if state is not None:
            state = {key: value.to(device) for key, value in state.items()}
            # Preserve the public message as a view of the restored top layer.
            state['out'] = state['h'][-1]
        step, updates, skipped = snapshot['step'], snapshot['updates'], snapshot['skipped']
        training_seconds = snapshot['training_seconds']
        evaluation_seconds = snapshot['evaluation_seconds']
        checkpoint_seconds = snapshot['checkpoint_seconds']
        best, final_val = snapshot['best'], snapshot['final_val']
        last_eval_step = snapshot['last_eval_step']
        last_checkpoint_step = step
        pending_evaluation = snapshot.get('pending_evaluation', False)
        logger.tracker = snapshot['logger_tracker']
        logger.last_flush_step = snapshot['logger_last_flush_step']
        logger.next_flush = snapshot['logger_next_flush']
        restore_rng(snapshot['rng'])
        print(f'Resumed {resume} at {step:,} tokens / {updates:,} updates', flush=True)
    control.install()
    progress('starting', force=True)
    if cfg['compile']:
        model = torch.compile(model)
    if best['weights'] is not None:
        atomic_torch_save({'model': best['weights'], 'step': best['step']}, directory / 'best.pt')

    batch_size = cfg['n_envs'] * cfg['rollout_len']
    checkpoint_interval = int(cfg['checkpoint']['every_tokens'])

    def save_checkpoint():
        nonlocal checkpoint_seconds, last_checkpoint_step
        synchronize(device)
        started = time.perf_counter()
        payload = {
            'format_version': 1, 'config_fingerprint': config_fingerprint(cfg), 'source_hashes': hashes,
            'data_identity': data_identity, 'model': cpu_weights(model), 'optimizer': optim.state_dict(),
            'recurrent_state': state, 'generator': generator_state(gen, cfg['exp']),
            'rng': rng_state(), 'lr': parameter_state(lr),
            'reset': parameter_state(reset) if reset is not None else None,
            'curriculum': scheduler_state(curriculum), 'eval_schedule': scheduler_state(eval_schedule),
            'step': step, 'updates': updates, 'skipped': skipped, 'best': best,
            'final_val': final_val, 'last_eval_step': last_eval_step,
            'pending_evaluation': pending_evaluation,
            'training_seconds': training_seconds, 'evaluation_seconds': evaluation_seconds,
            'checkpoint_seconds': checkpoint_seconds, 'logger_tracker': logger.tracker,
            'logger_last_flush_step': logger.last_flush_step, 'logger_next_flush': logger.next_flush,
        }
        latest = directory / 'latest.pt'
        if latest.exists():
            temporary = directory / 'previous.pt.tmp'
            shutil.copyfile(latest, temporary)
            temporary.replace(directory / 'previous.pt')
        atomic_torch_save(payload, latest)
        if cfg['checkpoint'].get('keep_history', False):
            shutil.copyfile(latest, directory / f'checkpoint-{step:012d}.pt')
        checkpoint_seconds += time.perf_counter() - started
        last_checkpoint_step = step
        progress('checkpoint_saved', force=True)

    def evaluate(*, final=False):
        nonlocal evaluation_seconds, final_val, last_eval_step, best, pending_evaluation
        synchronize(device)
        started = time.perf_counter()
        progress('evaluation', force=True)
        control.check()
        if cfg['exp'] == 'text8':
            segment_path = directory / 'val_final_segments.npz' if final else None
            final_val = evaluate_text(model, validation, cfg['eval'], segment_path=segment_path, check_stop=control.check)
        else:
            final_val = evaluate_sdq(
                model, cfg['eval_generator'], cfg['eval'],
                result_path=directory / 'val_final_gaps.npz' if final else None, check_stop=control.check,
            )
        logger.set_summary(final_val, prefix='val', key='eval')
        last_eval_step = step
        pending_evaluation = False
        if final_val['Loss'] < best['loss']:
            best = {'loss': final_val['Loss'], 'step': step, 'weights': cpu_weights(model)}
            atomic_torch_save({'model': best['weights'], 'step': step}, directory / 'best.pt')
        synchronize(device)
        evaluation_seconds += time.perf_counter() - started

    completed = False
    try:
        try:
            while step < cfg['effective_tokens']:
                control.check()
                if pending_evaluation:
                    evaluate()
                progress('training')
                synchronize(device)
                started = time.perf_counter()
                ys, targets, gaps = [], [], []
                comm_loss = comm_entropy = 0.0
                model.train()
                for _ in range(cfg['rollout_len']):
                    obs = gen.next()
                    if cfg['exp'] == 'text8':
                        reset_mask = obs['reset_mask'] | (torch.rand(cfg['n_envs'], device=device, generator=gen.rng) < reset.val)
                    else:
                        obs = {key: torch.from_numpy(value).to(device) for key, value in obs.items()}
                        reset_mask = obs['reset_mask']
                        gaps.append(obs['sq_gaps'])
                    state = model.rnn.reset_state(state, reset_mask)
                    y, state, info = model(obs['tokens'].view(1, -1), state)
                    ys.append(y)
                    targets.append(obs['targets'])
                    comm_loss = comm_loss + torch.stack(info['comm_loss']).mean()
                    comm_entropy = comm_entropy + torch.stack(info['comm_entropy']).mean()
                    step += cfg['n_envs']
                    if curriculum.tick(cfg['n_envs']):
                        gen.set_metaparams(
                            T=min(gen.T + 0.125, cfg['curriculum']['max_T']),
                            p_store=max(gen.p_store - 0.00014, 0.10),
                            p_query=max(gen.p_query - 0.00005, 0.25),
                        )
                y = torch.cat(ys)
                target = torch.cat(targets)
                valid = target != CE_ignore_index
                ce = F.cross_entropy(y, target, ignore_index=CE_ignore_index) if valid.any() else y.sum() * 0
                comm_loss = comm_loss / cfg['rollout_len']
                comm_entropy = comm_entropy / cfg['rollout_len']
                total = ce + cfg['communication']['loss_weight'] * comm_loss - cfg['communication']['entropy_weight'] * comm_entropy
                optim.zero_grad(set_to_none=True)
                total.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isfinite(grad):
                    optim.step()
                else:
                    skipped += 1
                lr.step()
                if reset is not None:
                    reset.step()
                updates += 1
                with torch.no_grad():
                    acc = (y[valid].argmax(-1) == target[valid]).float()
                    metrics = {'Loss': ce, 'Acc': acc.mean(), '|Grad|': grad, 'LR': lr.val, 'L_comm': comm_loss, 'H_comm': comm_entropy}
                    if cfg['exp'] == 'text8':
                        metrics['BPC'] = ce / np.log(2)
                        metrics['T'] = 1 / reset.val
                    else:
                        metrics.update(sq_gap_metrics(acc, torch.cat(gaps).float()))
                        metrics['T'] = gen.T
                    logger.accumulate(to_loggable_metrics(metrics), key='slow')
                state = model.rnn.detach_state(state)
                synchronize(device)
                training_seconds += time.perf_counter() - started
                progress('training', force=updates == 1)
                pending_evaluation = eval_schedule.tick(batch_size) and step < cfg['effective_tokens']
                control.check()
                if pending_evaluation:
                    evaluate()
                logger.set_summary(
                    {
                        'updates': updates, 'skipped_updates': skipped,
                        'training_seconds': training_seconds, 'evaluation_seconds': evaluation_seconds,
                        'checkpoint_seconds': checkpoint_seconds,
                        'dense_forward_flops_estimate': step * cfg['accounting']['dense_forward_flops_per_token'],
                    }, prefix='resource', key='eval',
                )
                logger.log(step, flush=True)
                if step // checkpoint_interval > last_checkpoint_step // checkpoint_interval:
                    save_checkpoint()
                if stop_after_updates is not None and updates >= stop_after_updates:
                    break
            if step >= cfg['effective_tokens']:
                control.check()
                evaluate(final=True)
                synchronize(device)
                final_evaluation_started = time.perf_counter()
                if cfg['exp'] == 'text8':
                    if cfg['eval'].get('test_on_finish', True):
                        final_test = evaluate_text(model, test, cfg['eval'], segment_path=directory / 'test_final_segments.npz', check_stop=control.check)
                        logger.set_summary(final_test, prefix='test_final', key='eval')
                        final_weights = cpu_weights(model)
                        unwrap(model).load_state_dict(best['weights'])
                        try:
                            best_test = evaluate_text(
                                model, test, cfg['eval'], segment_path=directory / 'test_best_segments.npz', check_stop=control.check,
                            )
                            logger.set_summary(best_test, prefix='test_best', key='eval')
                        finally:
                            unwrap(model).load_state_dict(final_weights)
                else:
                    probes = evaluate_fixed_gaps(
                        model, cfg['eval_generator'], cfg['eval'], result_path=directory / 'fixed_gap_final.npz', check_stop=control.check,
                    )
                    logger.set_summary(probes, prefix='fixed_gap', key='eval')
                synchronize(device)
                evaluation_seconds += time.perf_counter() - final_evaluation_started
                atomic_torch_save({'model': cpu_weights(model), 'step': step}, directory / 'final.pt')
                completed = True
        except PauseRequested as error:
            print(f'Pausing at {step:,} tokens: {error}', flush=True)
        # Save before remote logging so a tracking delay cannot lose session progress.
        save_checkpoint()
        # Flush final evaluations even if training already flushed at this token count.
        logger.flush(step)
        save_checkpoint()
        summary = {
            'run_id': cfg['run_id'], 'completed': completed, 'tokens': step, 'updates': updates,
            'pause_reason': control.reason,
            'skipped_updates': skipped, 'final_validation': final_val, 'best_validation_step': best['step'],
            'final_test': final_test, 'best_validation_test': best_test, 'fixed_gap_evaluation': probes,
            'training_seconds': training_seconds, 'evaluation_seconds': evaluation_seconds,
            'checkpoint_seconds': checkpoint_seconds,
        }
        if device.type == 'cuda':
            summary['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        write_json(directory / 'summary.json', summary)
        metadata['status'] = 'complete' if completed else 'paused'
        write_json(directory / 'metadata.json', metadata)
        progress(metadata['status'], force=True)
        return summary
    finally:
        control.restore()
        logger.finish()


if __name__ == '__main__':
    run_experiment(runner=main)
