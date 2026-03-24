from __future__ import annotations

from ast import literal_eval
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from knitwork.common.config import TKeyPathValue, load_config, override_config


def run_experiment(*, runner, arg_parser: ArgumentParser | None = None):
    """
    THE MAIN entry point for starting a program.
        1) resolves run args
        2) reads config
        3) sets any execution params
        4) passes execution handling to the runner.
    """
    arg_parser = arg_parser or default_run_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()

    if args.math_threads > 0:
        # manually set math parallelization as it usually only slows things down for us
        set_number_cpu_threads_for_math(
            num_threads=args.math_threads, cpu_affinity=args.cpu_affinity,
            with_torch=args.with_torch
        )

    config_path = Path(args.config_filepath)
    config = load_config(config_path)
    override_config(config, overrides=parse_arg_list(unknown_args))
    
    return runner(config)


def set_number_cpu_threads_for_math(
        *, num_threads: int, with_torch: bool = False, cpu_affinity: str | None
):
    # Set cpu threads for math libraries: affects math operations parallelization capability
    # NB: most of the time multithreaded math only slows things down, 
    #   and it is enabled by default in numpy and scipy, but not in torch.
    os.environ['OMP_NUM_THREADS'] = f'{num_threads}'
    os.environ['OPENBLAS_NUM_THREADS'] = f'{num_threads}'
    os.environ['MKL_NUM_THREADS'] = f'{num_threads}'
    if with_torch:
        import torch
        torch.set_num_threads(num_threads)

    if cpu_affinity:
        # Math libraries also love to set cpu affinity, restricting
        # which CPU cores your sub-processes can run on... So, tell them explicitly to shut up :)
        # Setting these variables doesn't affect the number of threads, btw
        os.environ['OMP_PLACES'] = cpu_affinity


def default_run_arg_parser() -> ArgumentParser:
    """
    Returns default run command parser.

    Instead of creating a new one for your specific purposes, you can create a default one
    and then extend it by adding new arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(dest='config_filepath')
    parser.add_argument('--math_threads', dest='math_threads', type=int, default=1)
    parser.add_argument('--with_torch', dest='with_torch', action='store_true', default=True)
    parser.add_argument(
        '--cpu_affty', dest='cpu_affinity', default=None, 
        help="Set None or in a form of {low:high}` core indices, e.g. `{0:3}`. Default: None"
    )
    # set how many cores each process should use
    parser.add_argument('--icpu_affty', dest='ind_cpu_affinity', type=int, default=None)
    return parser


def parse_arg_list(args: list[str]) -> list[TKeyPathValue]:
    """Parse a list of command line arguments to the list of key-value pairs."""
    return list(map(parse_arg, args))


def parse_arg(arg: str | tuple[str, Any]) -> TKeyPathValue:
    """Parse a single command line argument to the key-value pair."""
    try:
        if isinstance(arg, str):
            # raw arg string: "key=value"

            # "--key=value" --> ["--key", "value"]
            key_path, value = arg.split('=', maxsplit=1)

            # "--key" --> "key"
            key_path = key_path.removeprefix('--')

            # parse value represented as str
            value = parse_str(value)
        else:
            # tuple ("key", value) from wandb config of the sweep single run
            # we assume that the passed value is already correctly parsed
            key_path, value = arg
    except:
        print(arg)
        raise

    # parse key tokens as they can represent array indices
    # NB: skip empty key tokens (see [1] in the end of the file for an explanation)
    key_path = [
        parse_str(key_token)
        for key_token in key_path.split('.')
        if key_token
    ]

    return key_path, value


def parse_str(s: str) -> Any:
    """Parse string value to the most appropriate type."""
    # noinspection PyShadowingNames
    def boolify(s):
        if s in ('True', 'true'):
            return True
        if s in ('False', 'false'):
            return False
        raise ValueError('Not a boolean value!')

    # NB: try/except is widely accepted pythonic way to parse things
    assert isinstance(s, str)

    # NB: order of casters is important (from most specific to most general)
    for caster in (boolify, int, float, literal_eval):
        try:
            return caster(s)
        except (ValueError, SyntaxError):
            pass
    return s


# [1]: Using sweeps we have a problem with config logging. All parameters provided to
# a run from the sweep via run args are logged to wandb automatically. At the same time,
# when we also log our compiled config dictionary, its content is flattened such that
# each param key is represented as `path.to.nested.dict.key`. Note that we declare
# params in a sweep config the same way. Therefore, each sweep run will have such params
# visibly duplicated in wandb and there's no correct way to distinguish them
# (although, wandb itself does it)! Also, only sweep runs will have params duplicated.
# Simple runs don't have the duplicate entry because they don't have sweep param args.
#
# Problem: when you want to filter or group by a param in wandb interface,
# you cannot be sure which of the duplicated entries to select, while they're different
# — the only entry that is presented in all runs [either sweep or simple] is the entry
# from our config, not from a sweep.
#
# Solution: That's why we introduced a trick - you are allowed to specify sweep param
# with insignificant additional dots (e.g. `path..to...key.`) to de-duplicate entries.
# We ignore these dots [or empty path elements introduced by them after split-by-dots]
# while parsing the nested key path.
