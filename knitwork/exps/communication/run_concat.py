"""Run none_concat using the original training/evaluation/checkpoint protocol."""

import copy
from contextlib import contextmanager
from pathlib import Path

from knitwork.common.entrypoint import run_experiment
from knitwork.exps.communication import protocol, run as training
from knitwork.models.utils import REGISTRY, WRAPPER_REGISTRY


ROOT = Path(__file__).resolve().parents[3]


def accounting(model):
    counts = protocol.accounting(model)
    core = model.rnn
    extra_inputs = (core.n_columns - 1) * core.hidden_size
    counts['parameters_adapters'] += extra_inputs * model.output_size
    counts['dense_forward_flops_per_token'] += 2 * extra_inputs * model.output_size
    counts['parameter_caveat'] = (
        'All columns have a path to prediction through independently learned head blocks. '
        'Upper single-source masked attention has zero task gradient in Q/K. '
        'The concat head adds parameters and output-head FLOPs while preserving carried state.'
    )
    return counts


@contextmanager
def registered_model():
    previous = (REGISTRY['grnn_communication'], WRAPPER_REGISTRY['token'], protocol.CONDITIONS, training.accounting)
    REGISTRY['grnn_communication'] = 'grnn_concat.GridRnn'
    WRAPPER_REGISTRY['token'] = 'grnn_concat.TokenModel'
    protocol.CONDITIONS = (*protocol.CONDITIONS, 'none_concat')
    training.accounting = accounting
    try:
        yield
    finally:
        REGISTRY['grnn_communication'], WRAPPER_REGISTRY['token'], protocol.CONDITIONS, training.accounting = previous


def prepare_config(config):
    cfg = copy.deepcopy(config)
    if cfg['model'] != 'grnn_communication.none_concat' or cfg['topology'] != 'L2C4':
        raise ValueError('Concat control requires L2C4 and none_concat')
    core = cfg['topologies'][cfg['topology']]
    if (core['n_layers'], core['n_columns'], core['hidden_size']) != (2, 4, 136):
        raise ValueError('Concat control preserves L2C4/H136 and its 1088 carried state values')
    if cfg.get('readout_initialization') != 'mean_equivalent':
        raise ValueError('Concat head must start with the prediction of mean readout')
    identity = {
        'kind': 'concatenated-readout-none',
        'source_hashes': {name: protocol.file_digest(ROOT / name) for name in (
            'knitwork/models/grnn_concat.py',
            'knitwork/exps/communication/run_concat.py',
        )},
    }
    if 'implementation' in cfg and cfg['implementation'] != identity:
        raise ValueError('Concat implementation changed; use the original sources to resume')
    cfg['implementation'] = identity
    with registered_model():
        protocol.resolve_config(cfg)
    return cfg


def main(config, *, stop_after_updates=None):
    cfg = prepare_config(config)
    with registered_model():
        return training.main(cfg, stop_after_updates=stop_after_updates)


if __name__ == '__main__':
    run_experiment(runner=main)
