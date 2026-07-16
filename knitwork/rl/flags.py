import numpy as np
import torch


class RlFlags:
    TERMINATED = 1
    TRUNCATED = 2
    # Derivative "flag": TERM | TRUNC for easier binary checking.
    DONE = 3
    # New episode started
    RESET = 4


def to_flags_np(term: np.ndarray, trunc: np.ndarray, reset: bool | torch.Tensor = False):
    # bitwise OR is same as sum here
    res = term.view(np.uint8) | (trunc.view(np.uint8) << 1)
    if reset is True:
        res = res | RlFlags.RESET
    elif reset is False:
        ...
    else:
        res = res | (reset.view(np.uint8) << 2)
    return res

def to_flags_torch(term: torch.Tensor, trunc: torch.Tensor, reset: bool | torch.Tensor = False):
    # bitwise OR is same as sum here
    res = term.view(torch.uint8) | (trunc.view(torch.uint8) << 1)
    if reset is True:
        res = res | RlFlags.RESET
    elif reset is False:
        ...
    else:
        res = res | (reset.view(torch.uint8) << 2)
    return res
