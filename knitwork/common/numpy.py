import numpy as np
from numpy.random import Generator


def get_seed(rng: Generator = None):
    if rng is None:
        return get_seed(np.random.default_rng())
    return int(rng.integers(1_000_000))


def stochastic_round(x, rng: Generator):
    """Stochastically round value to the nearest integers."""
    n = int(x)
    # treat fractional part as probability
    frac_prob = x - n
    return n + (rng.random() < frac_prob)
