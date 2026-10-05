"""Query-only associative recall with full-sequence BPTT and fixed held-out splits."""

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging_alt import start_logger
from knitwork.common.state_size import state_floats
from knitwork.common.utils import count_learnable_params, get_device, get_dtype
from knitwork.gens.mqar import IGNORE_INDEX, SOURCE_COMMIT, SOURCE_URL, build_splits, epoch_batches
from knitwork.models.utils import build_model


def sequence_objective(model, inputs, labels):
    """Sum CE only on query keys; retain the graph through every preceding token.

    The existing token wrapper supplies the embedding/core/head. Computing the
    vocabulary projection only at scored queries avoids an unused large head
    projection at store/filler steps. No answers are fed back into the sequence.
    """
    batch, length = inputs.shape
    state = model.init_state(batch)
    batch_first = getattr(model.rnn, 'batch_first', False)
    # one embedding / head call per sequence instead of per step: same math, one dense table gradient
    embedded = model.embedding(inputs)  # [B, T, H]
    query_steps = set(torch.where((labels != IGNORE_INDEX).any(0))[0].tolist())
    feats, targets, rows = [], [], []
    for step in range(length):
        x = embedded[:, step]  # [B, H]
        features, state, _ = model.rnn(x[:, None] if batch_first else x[None], state)
        if step in query_steps:
            mask = labels[:, step] != IGNORE_INDEX
            feats.append(features[mask])
            targets.append(labels[mask, step])
            rows.append(torch.where(mask)[0])
    if not feats:
        raise ValueError('A batch must contain at least one scored MQAR query')
    logits = model.head(torch.cat(feats))  # [Q, V]
    target = torch.cat(targets)
    hits = logits.detach().argmax(-1) == target
    misses = torch.zeros(batch, device=inputs.device, dtype=torch.long).index_add_(0, torch.cat(rows), (~hits).long())
    return F.cross_entropy(logits, target, reduction='sum'), hits.sum(), (misses == 0).sum()


@torch.no_grad()
def evaluate(model, segments, *, batch_size, device, seed=1000):
    model.eval()
    by_segment = {}
    totals = dict(loss=0.0, correct=0, exact=0, queries=0, sequences=0)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    # Match fixed state initializations between repeated evaluations without changing training RNG.
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        for segment in segments:
            stats = dict(loss=0.0, correct=0, exact=0, queries=0, sequences=0)
            for start in range(0, len(segment['inputs']), batch_size):
                inputs = segment['inputs'][start:start + batch_size].to(device)
                labels = segment['labels'][start:start + batch_size].to(device)
                loss, correct, exact = sequence_objective(model, inputs, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite evaluation loss')
                stats['loss'] += loss.item()
                stats['correct'] += correct.item()
                stats['exact'] += exact.item()
                stats['queries'] += int((labels != IGNORE_INDEX).sum())
                stats['sequences'] += len(inputs)
            by_segment[segment['key']] = summarize(stats)
            for key in totals:
                totals[key] += stats[key]
    return summarize(totals) | {'by_segment': by_segment}


def summarize(stats):
    return {
        'Loss': stats['loss'] / stats['queries'],
        'Acc': stats['correct'] / stats['queries'],
        'Exact_match': stats['exact'] / stats['sequences'],
        'queries': stats['queries'], 'sequences': stats['sequences'],
    }


def log_metrics(metrics):
    result = {key: value for key, value in metrics.items() if key != 'by_segment'}
    for segment, values in metrics['by_segment'].items():
        result.update({f'{segment}/{key}': value for key, value in values.items()})
    return result


def allocated_state_floats(state):
    # Count each underlying allocation once, including cached outputs but excluding aliased views.
    storages = {}
    for value in state.values():
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes() // value.element_size()
    return sum(storages.values())


def model_state_info(model, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    # A single core step reveals steady-state buffers; preserve training randomness.
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        state = model.init_state(1)
        initial = allocated_state_floats(state)
        tokens = torch.zeros(1, 1, dtype=torch.long, device=device)
        _, state, _ = model.rnn(model.embedding(tokens), state)
        return {
            'carried_state_floats': state_floats(model.rnn),
            'initial_allocated_state_floats': initial,
            'allocated_state_floats': allocated_state_floats(state),
        }


def main(config):
    config = copy.deepcopy(config)
    if config['seed'] is None:
        raise ValueError('An explicit model seed is required')
    torch.manual_seed(config['seed'])
    torch.set_float32_matmul_precision('high')
    device, dtype = get_device(config['device']), get_dtype(config['dtype'])
    train_cfg, eval_cfg = config['training'], config['eval']
    for key in ('epochs', 'batch_size'):
        if not isinstance(train_cfg[key], int) or train_cfg[key] < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    limit = train_cfg.get('max_updates')
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError('training.max_updates must be null or a positive integer')
    if eval_cfg['schedule'] < 1 or eval_cfg['batch_size'] < 1:
        raise ValueError('Evaluation schedule and batch size must be positive')
    if train_cfg['learning_rate'] <= 0 or train_cfg['grad_clip'] <= 0:
        raise ValueError('Learning rate and gradient clipping must be positive')
    splits = build_splits(config['data'])
    model_key = config['model'].replace('.', '_')
    model = build_model(
        wrapper_type='token', rnn_type=config['model'].split('.', 1)[0],
        wrapper_cfg=dict(
            input_size=config['data']['vocab_size'], output_size=config['data']['vocab_size'],
            dtype=dtype, device=device,
        ),
        rnn_cfg=config[model_key],
    ).to(device=device, dtype=dtype)
    config['model_info'] = {
        'params': count_learnable_params(model), 'core_params': count_learnable_params(model.rnn),
    } | model_state_info(model, device)
    if config.get('compile', False):
        # the per-token core step is launch-bound; compiling it is ~3-4x faster, results are numerically equivalent
        model.rnn = torch.compile(model.rnn)
    config['runtime'] = {
        'device': str(device), 'dtype': str(dtype), 'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
    }
    config['protocol'] = {
        'generator_source': SOURCE_URL, 'generator_commit': SOURCE_COMMIT, 'num_passes': 1,
        'target_alignment': 'at_query_key', 'bptt': 'full_sequence',
        'split_segments': {
            split: [{key: segment[key] for key in ('key', 'seed', 'input_seq_len', 'num_kv_pairs', 'num_examples')} for segment in group]
            for split, group in splits.items()
        },
    }
    config['log'].setdefault('name', f'{config["model"]}_seed{config["seed"]}')
    output = None
    if config.get('output_dir'):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = Path(config['output_dir']).expanduser() / f'{stamp}_{model_key}_seed{config["seed"]}'
        output.mkdir(parents=True, exist_ok=False)
        config['run_dir'] = str(output)
        (output / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'{config["model"]}: {config["model_info"]["params"]:,} params; full-sequence MQAR BPTT')
    if output:
        print(f'Run directory: {output}')
    logger = start_logger(config, tracker=config['trackers'])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    eval_kwargs = dict(batch_size=eval_cfg['batch_size'], device=device, seed=eval_cfg['state_seed'])
    rng = np.random.default_rng(config['seed'])
    best_loss, best_state, best_update = float('inf'), None, 0
    updates, tokens, queries, train_seconds, skipped = 0, 0, 0, 0.0, 0
    total_updates = train_cfg['epochs'] * sum(
        (len(segment['inputs']) + train_cfg['batch_size'] - 1) // train_cfg['batch_size']
        for segment in splits['train']
    )
    total_updates = min(total_updates, limit) if limit is not None else total_updates
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    def set_status(state, **extra):
        print(f'STATUS {state} update={updates}/{total_updates} skipped={skipped}', flush=True)
        if output:
            (output / 'status.json').write_text(json.dumps({'status': state, 'updates': updates, 'total_updates': total_updates, 'skipped_steps': skipped, **extra}))

    set_status('training')
    try:
        for epoch in range(1, train_cfg['epochs'] + 1):
            for index, indices in epoch_batches(splits['train'], train_cfg['batch_size'], rng):
                segment = splits['train'][index]
                inputs, labels = (segment[key][indices].to(device) for key in ('inputs', 'labels'))
                n_queries = int((labels != IGNORE_INDEX).sum())
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                loss, correct, _ = sequence_objective(model, inputs, labels)
                loss = loss / n_queries
                if torch.isfinite(loss):
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
                else:
                    grad_norm = torch.tensor(float('nan'))
                # a non-finite step is skipped (as in the text runner) instead of killing the run
                warmup = train_cfg.get('warmup_updates') or 0
                for group in optimizer.param_groups:
                    group['lr'] = train_cfg['learning_rate'] * (min(1.0, (updates + 1) / warmup) if warmup else 1.0)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                else:
                    skipped += 1
                    if skipped >= 100 and skipped > 0.5 * (updates + 1):
                        raise FloatingPointError(f'{skipped} of {updates + 1} steps skipped: non-finite loss/gradients')
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                train_seconds += time.perf_counter() - start
                updates += 1
                tokens += inputs.numel()
                queries += n_queries
                logger.accumulate(
                    {
                        'Loss': loss.item() if torch.isfinite(loss) else 0.0, 'Acc': correct.item() / n_queries,
                        'Grad_norm': grad_norm.item() if torch.isfinite(grad_norm) else 0.0,
                        'update': updates, 'epoch': epoch,
                    }, prefix='train', key='slow',
                )
                # exact cumulative counter: the slow tracker above is smoothed
                logger.accumulate({'skipped_steps_exact': skipped, 'skipped_fraction': skipped / updates}, prefix='train', key='eval')
                if updates % eval_cfg['schedule'] == 0 or updates == total_updates:
                    validation = evaluate(model, splits['val'], **eval_kwargs)
                    logger.accumulate(log_metrics(validation), prefix='val', key='eval')
                    if validation['Loss'] < best_loss:
                        best_loss, best_update = validation['Loss'], updates
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                        if output:
                            torch.save({'model': best_state, 'update': updates, 'validation': validation}, output / 'best.pt')
                logger.log(tokens, flush=True)
                if updates == total_updates:
                    break
            if updates == total_updates:
                break
        if output:
            torch.save({'model': model.state_dict(), 'update': updates}, output / 'last.pt')
        if best_state is None:
            raise RuntimeError('No finite validation loss was recorded; refusing to run test')
        set_status('training_done_validated', best_update=best_update, best_val_loss=best_loss)
        model.load_state_dict(best_state)
        # No test access during checkpoint selection; include length/capacity extrapolation bins.
        test = evaluate(model, splits['test'], **eval_kwargs)
        logger.accumulate(log_metrics(test), prefix='test', key='eval')
        result = {
            'best_update': best_update, 'best_val_loss': best_loss, 'test': test,
            'model_info': config['model_info'], 'protocol': config['protocol'],
            'runtime': config['runtime'],
            'training_updates': updates, 'skipped_steps': skipped, 'training_tokens': tokens, 'training_queries': queries,
            'training_seconds': train_seconds, 'training_tokens_per_second': tokens / train_seconds,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        }
        if output:
            (output / 'metrics.json').write_text(json.dumps(result, indent=2))
        logger.flush(tokens)
        set_status('complete', best_update=best_update, test_acc=test['Acc'])
        print(f'Best update {best_update}; test query accuracy {test["Acc"]:.4f}')
        return result
    except BaseException as error:
        set_status('failed', error=repr(error))
        raise
    finally:
        logger.finish()


if __name__ == '__main__':
    run_experiment(runner=main)
