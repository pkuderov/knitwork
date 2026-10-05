"""Next-observation prediction on aliased cubes; no RL or map extraction objective."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from knitwork.common.entrypoint import run_experiment
from knitwork.common.logging_alt import start_logger
from knitwork.common.utils import count_learnable_params, get_device, get_dtype
from knitwork.env.aliased_cube import AliasedCube
from knitwork.exps.aliased_cubes.model import ActionObservationModel, reactive_probabilities


@torch.no_grad()
def evaluate(model, walks, *, batch_size, device, report_after=30, seed=1000):
    model.eval()
    n_sequences, sequence_length = walks['observations'].shape
    n_transitions = sequence_length - 1
    hits = torch.zeros(n_transitions, device=device)
    loss = torch.zeros((), device=device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    # Fixed eval initialization without disturbing the training RNG stream.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for start in range(0, n_sequences, batch_size):
            obs = torch.as_tensor(walks['observations'][start:start + batch_size], device=device)
            actions = torch.as_tensor(walks['actions'][start:start + batch_size], device=device)
            state = model.rnn.init_state(obs.shape[0])
            for step in range(n_transitions):
                logits, state = model(obs[:, step], actions[:, step], state)
                target = obs[:, step + 1]
                loss += F.cross_entropy(logits, target, reduction='sum')
                hits[step] += (logits.argmax(-1) == target).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite evaluation loss')
    by_step = (hits / n_sequences).cpu().tolist()
    after = min(report_after, n_transitions - 1)
    return {
        'Loss': float(loss.cpu()) / (n_sequences * n_transitions),
        'Acc': sum(by_step) / n_transitions,
        'Acc_after_context': sum(by_step[after:]) / len(by_step[after:]),
        'context_steps': after,
        'Acc_by_step': by_step,
    }


def evaluate_reactive(probabilities, walks, report_after):
    obs, actions = walks['observations'], walks['actions']
    probabilities = probabilities.numpy()
    predicted = probabilities[obs[:, :-1], actions]
    targets = obs[:, 1:]
    by_step = (predicted.argmax(-1) == targets).mean(0)
    target_probs = np.take_along_axis(predicted, targets[..., None], axis=-1)[..., 0]
    after = min(report_after, actions.shape[1] - 1)
    return {
        'Loss': float(-np.log(target_probs).mean()),
        'Acc': float(by_step.mean()),
        'Acc_after_context': float(by_step[after:].mean()),
        'context_steps': after,
    }


def scalar_metrics(metrics):
    return {key: value for key, value in metrics.items() if key != 'Acc_by_step'}


def main(config):
    config = copy.deepcopy(config)
    if config['seed'] is None:
        raise ValueError('An explicit seed is required for this experiment')
    torch.manual_seed(config['seed'])
    torch.set_float32_matmul_precision('high')
    device, dtype = get_device(config['device']), get_dtype(config['dtype'])
    data_cfg, train_cfg, eval_cfg = config['data'], config['training'], config['eval']
    for key in ('iterations', 'batch_size', 'rollout_len'):
        if train_cfg[key] < 1:
            raise ValueError(f'training.{key} must be positive')
    if eval_cfg['schedule'] < 1 or eval_cfg['batch_size'] < 1 or eval_cfg['report_after'] < 0:
        raise ValueError('Invalid evaluation schedule, batch size or context')
    env = AliasedCube(**config['environment'])
    walks = {
        split: env.sample_walks(data_cfg[f'{split}_sequences'], data_cfg['sequence_length'], data_cfg[f'{split}_seed'])
        for split in ('train', 'val', 'test')
    }
    config['environment_info'] = {
        'source': env.source, 'fingerprint': env.fingerprint,
        'n_states': env.n_states, 'n_actions': env.n_actions, 'n_observations': env.n_observations,
    }
    model_key = config['model'].replace('.', '_')
    model_type = config['model'].split('.', 1)[0]
    model = ActionObservationModel(
        n_observations=env.n_observations, n_actions=env.n_actions,
        rnn_type=model_type, rnn_cfg=config[model_key], dtype=dtype, device=device,
    ).to(device=device, dtype=dtype)
    config['params'] = count_learnable_params(model)
    if config.get('compile', False):
        model = torch.compile(model)
    output = None
    if config.get('output_dir'):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = Path(config['output_dir']).expanduser() / f'{stamp}_{model_key}_seed{config["seed"]}'
        output.mkdir(parents=True, exist_ok=False)
        config['run_dir'] = str(output)
        (output / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'{config["model"]}: {config["params"]:,} params | {env.source} | map {env.fingerprint[:12]}')
    if output:
        print(f'Run directory: {output}')
    logger = start_logger(config, tracker=config['trackers'])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    reactive = reactive_probabilities(walks['train'], env.n_observations, env.n_actions)
    eval_kwargs = dict(
        batch_size=eval_cfg['batch_size'], device=device,
        report_after=eval_cfg['report_after'], seed=eval_cfg['state_seed'],
    )
    best_loss, best_state, best_iteration = float('inf'), None, 0
    n_steps, rng = 0, np.random.default_rng(config['seed'])
    order, cursor = np.empty(0, dtype=np.int64), 0
    try:
        for iteration in range(1, train_cfg['iterations'] + 1):
            if cursor >= len(order):
                order = rng.permutation(len(walks['train']['observations']))
                cursor = 0
            indices = order[cursor:cursor + train_cfg['batch_size']]
            cursor += len(indices)
            obs = torch.as_tensor(walks['train']['observations'][indices], device=device)
            actions = torch.as_tensor(walks['train']['actions'][indices], device=device)
            model.train()
            state = model.rnn.init_state(len(indices))
            optimizer.zero_grad(set_to_none=True)
            n_transitions = actions.shape[1]
            train_loss, correct = torch.zeros((), device=device), torch.zeros((), device=device)
            for start in range(0, n_transitions, train_cfg['rollout_len']):
                losses = []
                for step in range(start, min(start + train_cfg['rollout_len'], n_transitions)):
                    logits, state = model(obs[:, step], actions[:, step], state)
                    target = obs[:, step + 1]
                    losses.append(F.cross_entropy(logits, target))
                    correct += (logits.detach().argmax(-1) == target).sum()
                loss = torch.stack(losses).sum() / n_transitions
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                train_loss += loss.detach()
                state = model.rnn.detach_state(state)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            if not torch.isfinite(grad_norm):
                raise FloatingPointError('Non-finite gradients')
            optimizer.step()
            n_steps += len(indices) * n_transitions
            logger.accumulate(
                {
                    'Loss': train_loss.item(), 'Acc': correct.item() / (len(indices) * n_transitions),
                    'Grad_norm': grad_norm.item(), 'iteration': iteration,
                }, prefix='train', key='slow',
            )
            if iteration % eval_cfg['schedule'] == 0 or iteration == train_cfg['iterations']:
                validation = evaluate(model, walks['val'], **eval_kwargs)
                logger.accumulate(scalar_metrics(validation), prefix='val', key='eval')
                if validation['Loss'] < best_loss:
                    best_loss, best_iteration = validation['Loss'], iteration
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    if output:
                        torch.save({'model': best_state, 'iteration': iteration, 'validation': validation}, output / 'best.pt')
            logger.log(n_steps, flush=True)
        if output:
            torch.save({'model': model.state_dict(), 'iteration': iteration}, output / 'last.pt')
        model.load_state_dict(best_state)
        # Test is used once, after all checkpoint selection has finished.
        test = evaluate(model, walks['test'], **eval_kwargs)
        reactive_val = evaluate_reactive(reactive, walks['val'], eval_cfg['report_after'])
        reactive_test = evaluate_reactive(reactive, walks['test'], eval_cfg['report_after'])
        logger.accumulate(scalar_metrics(test), prefix='test', key='eval')
        logger.accumulate(reactive_val, prefix='reactive_val', key='eval')
        logger.accumulate(reactive_test, prefix='reactive_test', key='eval')
        result = {
            'best_iteration': best_iteration, 'best_val_loss': best_loss,
            'test': test, 'reactive_val': reactive_val, 'reactive_test': reactive_test,
            'environment': config['environment_info'], 'training_transitions': n_steps,
        }
        if output:
            (output / 'metrics.json').write_text(json.dumps(result, indent=2))
        # Test metrics must flush even if training already flushed at the same step.
        logger.flush(n_steps)
        print(f'Best iteration {best_iteration}; test accuracy {test["Acc"]:.4f}')
        return result
    finally:
        logger.finish()


if __name__ == '__main__':
    run_experiment(runner=main)
