from timeit import default_timer
from typing import OrderedDict

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


timer = default_timer
CE_ignore_index = -100


def pprint_shape(*xs, key='', depth=0, indent='  ', **named_xs):
    # noinspection SpellCheckingInspection
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
        is_torch = isinstance(x, torch.Tensor)
        tp = 'torch' if is_torch else 'np'
        print(f'{prefix}{x.shape}        | {tp}.{x.dtype}')
    elif isinstance(x, (list, tuple, set)):
        print(f'{prefix}{len(x)}        | {type(x)}')
    else:
        print(f'{prefix}{type(x)}')


def print_with_timestamp(start_time: float, *args):
    """Extend regular print with the '[<elapsed seconds>]' prefix. """
    elapsed_sec = timer() - start_time
    if elapsed_sec < 1:
        time_format = '5.3f'
    elif elapsed_sec < 10:
        time_format = '5.2f'
    elif elapsed_sec < 1000:
        time_format = '5.1f'
    else:
        time_format = '5.0f'
    print(f'[{elapsed_sec:{time_format}}]', *args)


def to_readable_num(x):
    """
    For a number `x`, return a tuple (x_, suffix) represeting
    a shortened human-readable form suitable for printing, e.g.:

    >>>x_, suffix = to_readable_num(10_000)
    (10, "k")
    >>>print(f'{x_:.0f}{suffix}')
    "10k"
    >>>to_readable_num(23_987_555)
    (23.987555, "M")
    >>>print(f'{x_:.2f}{suffix}')
    "23.99M"
    """
    suffixes = ['', 'k', 'M', 'B']
    i = 0
    while abs(x) > 1000.0 or i >= len(suffixes):
        x = x / 1000.0
        i += 1

    return x, suffixes[i]

def format_readable_num(x, frac: int = 2):
    x_, sx = to_readable_num(x)
    return f'{x_:.{frac}f}{sx}'


class FpsCounter:
    def __init__(self):
        self.total_time = 0.0
        self.n_iters = 0
        self.start_time = None
        self.start()

    def start(self):
        self.start_time = timer()

    def stop(self):
        if self.start_time is None:
            return
        self.total_time += timer() - self.start_time
        self.n_iters += 1
        self.start_time = None

    def fps(self, n_iters=None, start=False):
        self.stop()
        if n_iters is None:
            n_iters = self.n_iters
        fps = n_iters / self.total_time
        if start:
            self.start()
        return fps

    def print(self, n_iters=None):
        fps = self.fps(n_iters)
        fps, fps_sx = to_readable_num(fps)
        print(f'FPS: {fps:.2f}{fps_sx}')


def get_device(device: str = None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def get_dtype(dtype: str = None, default=torch.float):
    return getattr(torch, dtype) if dtype is not None else default


def to_torch(x, device=None, copy=True):
    if isinstance(x, np.ndarray):
        if copy:
            x = x.copy()
        return torch.from_numpy(x).to(device)

    if isinstance(x, torch.Tensor) or x is None:
        return x

    x = torch.tensor(x, device=device)
    if copy:
        x = x.clone()
    return x


def to_numpy(x, copy=True):
    if isinstance(x, torch.Tensor):
        if copy:
            x = x.clone()
        # force is a shorthand for detach+cpu+...
        x = x.numpy(force=True)
    elif x is not None:
        x = np.array(x)
    return x

def isnone(x, default):
    """Return x if it's not None, or default value instead."""
    return x if x is not None else default


def ensure_list(arr):
    """Wrap single value to list or return list as it is."""
    if arr is not None and not isinstance(arr, list):
        arr = [arr]
    return arr


def safe_div(num, denom, default=0.0):
    """
    Return num / denom or just default itself if denom ~= 0 preventing NaNs.
    NB: it is not perfect for many borderline cases and is expected to
        be used in very simple straightforward cases as a handy shortcut
    """
    return num / denom if num != 0 and abs(denom) > 1e-9 else default


# ============================== Traversing ===================================

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


def flatten_dict(d, keep_prefix=True):
    """
    Return flattened dict. The order of iteration is according to DFS (depth first).
    """
    # '' -> keep prefix | None -> do not keep it
    init_prefix = '' if keep_prefix else None
    flattened_pairs = iterate_dict(d, prefix=init_prefix)
    return dict(flattened_pairs)


def iterate_dict(x, prefix=None):
    """Return an iterator of pairs (key, value) over the dict `x` in a DFS order."""
    for k, v in x.items():
        if prefix is not None:
            delimiter = '/' if prefix else ''
            k = f'{prefix}{delimiter}{k}'
        if isinstance(v, dict):
            yield from iterate_dict(v, prefix=k)
        else:
            yield k, v

# =============================================================================


def _count_linear_params(in_dim: int, out_dim: int, bias: bool = True) -> int:
    b = int(bias)
    return (in_dim + b) * out_dim


def _count_mha_params(hid_dim: int, bias: bool = True) -> int:
    b = int(bias)
    return 4 * (hid_dim + b) * hid_dim


def _count_rnn_cell_params(in_dim: int, hid_dim: int, *, bias: bool = True, cell: str = 'gru') -> int:
    cell = cell.lower()
    assert cell in ('gru', 'lstm')
    g = 3 if cell == 'gru' else 4

    w_ih = _count_linear_params(in_dim=in_dim, out_dim=hid_dim, bias=bias)
    w_hh = _count_linear_params(in_dim=hid_dim, out_dim=hid_dim, bias=bias)
    return g * (w_ih + w_hh)


def count_rnn_params(
        *, in_dim: int, hid_dim: int, out_dim: int = None, 
        n_layers: int = 1, bias: bool = None, cell: str = 'gru'
    ) -> int:
    # first layer has different input dim, while the others are just hidden -> hidden
    first = _count_rnn_cell_params(in_dim=in_dim, hid_dim=hid_dim, cell=cell, bias=bias)
    others = _count_rnn_cell_params(in_dim=hid_dim, hid_dim=hid_dim, cell=cell, bias=bias)
    out_proj = _count_linear_params(in_dim=hid_dim, out_dim=out_dim, bias=bias) if out_dim is not None else 0
    return first + (n_layers - 1) * others + out_proj


def count_grid_rnn_params(
        *, in_dim: int, hid_dim: int, out_dim: int = None, 
        n_layers: int = 1, n_columns: int = 1, bias: bool = None, cell: str = 'gru'
    ) -> int:
    # first col has both input and output
    first_col = count_rnn_params(
        in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, n_layers=n_layers, bias=bias, cell=cell
    )
    # while the other cols has in_dim = 0 (empty input) and out_dim = 0 (no output)
    other_cols = count_rnn_params(
        in_dim=0, hid_dim=hid_dim, out_dim=0, n_layers=n_layers, bias=bias, cell=cell
    )

    row_mha = _count_mha_params(hid_dim=hid_dim, bias=bias)

    return first_col + (n_columns - 1) * other_cols + n_layers * row_mha


def convert_hidden_size(
    *,
    base_hid_dim: int,
    in_dim: int, out_dim: int,
    n_layers: int, n_columns: int = 1,
    cell: str = 'gru', type: str = 'rnn',
    bias: bool = True, min_hidden: int = 4,
) -> int:
    """Convert reference 1-layer RNN cell hidden size to l-layer n-column RNN/gRNN hidden size."""
    # Target: recurrent params of 1-layer reference
    target = count_rnn_params(in_dim=in_dim, hid_dim=base_hid_dim, out_dim=out_dim, n_layers=1, bias=bias, cell=cell)

    from functools import partial
    shared_params = dict(in_dim=in_dim, out_dim=out_dim, n_layers=n_layers, bias=bias, cell=cell)
    if type == 'rnn':
        cnt_fn = partial(count_rnn_params, **shared_params)
    else:
        cnt_fn = partial(count_grid_rnn_params, **shared_params, n_columns=n_columns)

    low = 1
    high = base_hid_dim
    while low + 1 < high:
        mid = (low + high) // 2
        low, high = (mid, high) if cnt_fn(hid_dim=mid) < target else (low, mid)

    low_cnt, high_cnt = cnt_fn(hid_dim=low), cnt_fn(hid_dim=high)
    h = low if abs(low_cnt - target) < abs(high_cnt - target) else high

    return max(min_hidden, h)
