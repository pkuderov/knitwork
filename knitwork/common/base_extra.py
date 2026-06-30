from typing import OrderedDict

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


def iterate(x):
    """
    Return flattened iterable over the dict-like structure:
    a) For dict of np.ndarray, output iterable of arrays
    b) For dict of box spaces, output iterable of box specs (low, high, dtype)

    The order of iteration is according to DFS (depth first).
    """
    if isinstance(x, np.ndarray):
        yield x
    elif isinstance(x, spaces.Box):
        yield x.low, x.high, x.dtype
    elif isinstance(x, (dict, OrderedDict, spaces.Dict)):
        for k in x:
            yield from iterate(x[k])


def pprint_shape(*xs, key='', depth=0, indent='  ', **named_xs):
    """
        Pretty print structure of value[-s] or gymnasium space[-s].
        Usage:
            pprint_shape(x)
            pprint_shape(x, key='the name')
            pprint_shape(x, y, z, key='will be [{index}]{key} for all')
            pprint_shape(i, j, k, a=x, b=y, c=z, key='only for positional args')

        Useful to print multi-component observations/actions, which
        may have dict-based hierarchical structure — it will
        print shapes/lengths of all components with indentations,
        e.g. `pprint(x, key="the name")`:

            the name:
              <subkey1>: <shape>
              <dict subkey2>:
                <subsubkey1>: <shape>
                <subsubkey2>: <shape>
              <subkey3>: <shape>

        It is also useful as a general-purpose smart shape/len printer since it
        gracefully accepts arbitrary number of numpy ndarrays, torch tensors,
        python lists/tuples/sets and dicts.
    """
    is_single_val = len(xs) == 1 and len(named_xs) == 0
    if is_single_val:
        _pprint_shape(xs[0], key=key, depth=depth, indent=indent)
        return

    # unpack and call pprint for each one
    _key = f' {key}' if key else ''
    kv_pairs = [
        (f'[{i}]{_key}', _x)
        for i, _x in enumerate(xs)
    ]
    kv_pairs.extend(named_xs.items())
    for k, v in kv_pairs:
        _pprint_shape(v, key=k, depth=depth, indent=indent)
        print()


def _pprint_shape(x, key, depth, indent):
    """Pretty print structure of a single value."""
    prefix = indent * depth
    if not key:
        key = '<>'
    prefix = f'{prefix}{key}: '

    if isinstance(x, (dict, gym.spaces.Dict)):
        print(f'{prefix}')
        for subkey in x:
            _pprint_shape(x[subkey], key=subkey, depth=depth + 1, indent=indent)
    elif isinstance(x, (np.ndarray, torch.Tensor, gym.spaces.Box)):
        if isinstance(x, torch.Tensor):
            tp = 'torch'
        elif isinstance(x, np.ndarray): 
            tp = 'np'
        else:
            tp = 'gym'
        print(f'{prefix}{x.shape}        | {tp}.{x.dtype}')
    elif isinstance(x, (list, tuple, set)):
        print(f'{prefix}{len(x)}        | {type(x)}')
    else:
        print(f'{prefix}{type(x)}')
