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


def flatten_dict(d, keep_prefix=True):
    """
    Return flattened dict. The order of iteration is according to DFS (depth first).
    """
    # '' -> keep prefix | None -> do not keep it
    init_prefix = '' if keep_prefix else None
    flattened_pairs = iterate_dict(d, prefix=init_prefix)
    return dict(flattened_pairs)


def prefix_dict(x, prefix=None, delimiter='/'):
    """Return a copy of the dict with top-level keys prefixed."""
    if prefix is None:
        return x
    return {
        f'{prefix}{delimiter}{k}': v
        for k, v in x.items()
    }


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


def to_readable_size(bytes):
    i = 0
    while bytes >= 1024:
        bytes /= 1024
        i += 1

    size_units = ['B', 'KB', 'MB', 'GB', 'TB']
    sfx = size_units[i]
    return bytes, sfx


def format_sec(sec):
    if sec < 1:
        time_format = '5.3f'
    elif sec < 10:
        time_format = '5.2f'
    elif sec < 1000:
        time_format = '5.1f'
    else:
        time_format = '5.0f'
    return time_format


def print_with_timestamp(elapsed_sec: float, *args):
    """Extend regular print with the '[<elapsed seconds>]' prefix. """
    time_format = format_sec(elapsed_sec)
    print(f'[{elapsed_sec:{time_format}}s]', *args)
