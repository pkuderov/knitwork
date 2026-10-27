"""Run the widened one-column control without changing the original runner sources."""

import copy
from contextlib import contextmanager
from pathlib import Path

from knitwork.common.entrypoint import run_experiment
from knitwork.exps.communication import protocol, run as training
from knitwork.exps.communication.protocol import file_digest, resolve_config
from knitwork.models.utils import REGISTRY


ROOT = Path(__file__).resolve().parents[3]


def accounting(model):
    counts = protocol.accounting(model)
    counts['parameter_caveat'] = (
        'One column: all carried state has a path to prediction; no unused columns. '
        'Upper single-source attention has zero task gradient in Q/K. '
        'Width matches the nominal parameter budget, not the L2C4 state or compute budget.'
    )
    return counts


def prepare_config(config):
    cfg = copy.deepcopy(config)
    if cfg['model'] != 'grnn_communication.none_column0' or cfg['topology'] != 'L2C1':
        raise ValueError('The widened control requires L2C1 and none_column0')
    core = cfg['topologies'][cfg['topology']]
    if (core['n_layers'], core['n_columns']) != (2, 1):
        raise ValueError('L2C1 must contain two layers and exactly one column')
    identity = {
        'kind': 'single-column-none',
        'source_hashes': {name: file_digest(ROOT / name) for name in (
            'knitwork/models/grnn_single_column.py',
            'knitwork/exps/communication/run_single_column.py',
        )},
    }
    if 'implementation' in cfg and cfg['implementation'] != identity:
        raise ValueError('Single-column implementation changed; use the original sources to resume')
    # The original fingerprint includes this identity, so extension changes reject resume.
    cfg['implementation'] = identity
    resolve_config(cfg)
    return cfg


@contextmanager
def registered_model():
    previous = (REGISTRY['grnn_communication'], training.accounting)
    REGISTRY['grnn_communication'] = 'grnn_single_column.GridRnn'
    training.accounting = accounting
    try:
        yield
    finally:
        REGISTRY['grnn_communication'], training.accounting = previous


def main(config, *, stop_after_updates=None):
    cfg = prepare_config(config)
    with registered_model():
        return training.main(cfg, stop_after_updates=stop_after_updates)


if __name__ == '__main__':
    run_experiment(runner=main)
