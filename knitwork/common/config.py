from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ryaml

from knitwork.common.base import ensure_list


TKeyPathValue = tuple[list, Any]


def ns_to_dict(ns, high_level=True):
    if high_level and ns is None:
        return {}
    if not isinstance(ns, SimpleNamespace):
        return ns

    return {
        k: ns_to_dict(v, high_level=False)
        for k, v in ns.__dict__.items()
    }


def parse_str(s: str, try_eval=False) -> Any:
    """Parse string value to the most appropriate type."""
    # noinspection PyShadowingNames
    def boolify(s):
        if s in ('True', 'true'):
            return True
        if s in ('False', 'false'):
            return False
        raise ValueError('Not a boolean value!')

    assert isinstance(s, str)

    # NB: order of casters is important (from most specific to most general)
    casters = (boolify, int, float)
    # NB: try/except is widely accepted pythonic way to parse things
    if try_eval:
        casters = casters + (literal_eval,)
    for caster in casters:
        try:
            return caster(s)
        except (ValueError, SyntaxError):
            pass
    return s


def split_key_to_path(key: str) -> list[str]:
    """
    Split a string containing a dot-delimited path over nested dicts 
    to the list of corresponding dict keys.
    """
    # parse key tokens as they can represent array indices
    # NB: skip empty key tokens (useful for wandb sweep params)
    key_path = [
        parse_str(key_token)
        for key_token in key.split('.')
        if key_token != ''
    ]
    return key_path


def override_config(
        config: dict,
        overrides: dict[str, Any] | tuple[str, Any] | TKeyPathValue | list[TKeyPathValue]
) -> None:
    """
    Apply the number of overrides to the content of the config dictionary.
    Overrides are defined as a dict {"key.path.to.value": value}
    or as a list of tuples (["key", "path", "to", "value"], value).
    """
    if isinstance(overrides, dict):
        overrides = [
            (split_key_to_path(key), value) 
            for key, value in overrides.items()
        ]
    elif isinstance(overrides, tuple) and isinstance(overrides[0], str):
        overrides = [(split_key_to_path(overrides[0]), overrides[1])]
    else:
        overrides = ensure_list(overrides)

    for key_path, value in overrides:
        c = config
        for key_token in key_path[:-1]:
            c = c[key_token]
        c[key_path[-1]] = value


def filtered(d: dict, keys_to_remove, depth: int) -> dict:
    """
    Return a shallow copy of the provided dictionary without the items
    that match `keys_to_remove`.

    The `depth == 1` means filtering `d` itself,
        `depth == 2` — with its dict immediate descendants
        and so on.
    """
    if not isinstance(d, dict) or depth <= 0:
        return d

    return {
        k: filtered(v, keys_to_remove, depth - 1)
        for k, v in d.items()
        if k not in keys_to_remove
    }


def extracted(d: dict, *keys: str) -> tuple:
    """
    Return a tuple containing the copy of the filtered out dictionary
    and all the corresponding [to the keys] extracted values
    (or None if a specified key was absent).

    Examples
    --------
    >>> extracted({'a': 1, 'b': 2, 'c': 3}, 'a', 'c')
    ({'b': 2}, 1, 3)
    """
    values = tuple([d.get(k, None) for k in keys])
    filtered_dict = filtered(d, keys, depth=1)
    return (filtered_dict, ) + values


def load_config(path):
    path = Path(path).expanduser().resolve().as_posix()
    with open(path, 'r') as f:
        return ryaml.load(f)


# ==================== resolve absolute or relative quantity ====================
# quantities can be specified as absolute or relative to some baseline value

def resolve_absolute_quantity(abs_or_relative: int | float, *, baseline: int) -> int:
    """
    Convert passed quantity to the absolute quantity regarding its type and the baseline value.
    Here we consider that ints relate to the absolute quantities and floats
    relate to the relative quantities (relative to the `baseline` value).

    Examples:
        resolve_absolute_quantity(10, 20) -> 10
        resolve_absolute_quantity(1.25, 20) -> 25


    Parameters
    ----------
    abs_or_relative: int or float
        The value to convert. If it's int then it's returned as is. Otherwise, it's
        converted to the absolute system relative to the `baseline` value
    baseline: int
        The baseline for the relative number system.

    Returns
    -------
        Integer value in the absolute quantities system
    """

    if isinstance(abs_or_relative, float):
        relative = abs_or_relative
        return int(baseline * relative)
    elif isinstance(abs_or_relative, int):
        absolute = abs_or_relative
        return absolute
    else:
        raise TypeError(f'Function does not support type {type(abs_or_relative)}')


def resolve_relative_quantity(abs_or_relative: int | float, *, baseline: int) -> float:
    """See `resolve_absolute_quantity` - this method is the opposite of it."""

    if isinstance(abs_or_relative, float):
        relative = abs_or_relative
        return relative
    elif isinstance(abs_or_relative, int):
        absolute = abs_or_relative
        return absolute / baseline
    else:
        raise TypeError(f'Function does not support type {type(abs_or_relative)}')
