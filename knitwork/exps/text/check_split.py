"""Sanity checks for the leak-free train/val/test protocol.

uv run python -m knitwork.exps.text.check_split [path] [val_size] [test_size]
"""
from __future__ import annotations

import os
import sys

import numpy as np

from knitwork.gens.text import load_dataset, split_train_val_test, tokenize_splits


def main(path='$MY_HOME/data/text/text8.txt', val_size=5e6, test_size=5e6):
    raw = load_dataset(path)
    train, val, test = split_train_val_test(raw, float(val_size), float(test_size))
    n = len(raw)
    assert len(train) + len(val) + len(test) == n
    # contiguous, ordered, no overlap
    assert np.array_equal(np.concatenate([train, val, test]), raw)
    (t_tok, v_tok, s_tok), chars = tokenize_splits(train, val, test)
    assert np.array_equal(chars, np.unique(train)), 'vocab must come from train only'
    assert max(t_tok.max(), v_tok.max(), s_tok.max()) < len(chars)
    # decoded text of each split equals the raw split
    for tok, ref in [(t_tok, train), (v_tok, val), (s_tok, test)]:
        assert np.array_equal(chars[tok], ref)
    # unigram BPC of val under train frequencies as a reference for the initial loss
    p = np.bincount(t_tok, minlength=len(chars)) / len(t_tok)
    bpc = -np.log2(p[v_tok]).mean()
    print(f'train {len(train):,} | val {len(val):,} | test {len(test):,} | vocab {len(chars)}')
    print(f'uniform BPC {np.log2(len(chars)):.3f} | unigram val BPC {bpc:.3f}')
    print('OK')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args:
        args[0] = os.path.expandvars(args[0])
    main(*args)
