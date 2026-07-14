from __future__ import annotations

from functools import partial

from knitwork.common.base import isnone
from knitwork.common.dynamic_param import DynamicParameter
import numpy as np
import torch
from torch import nn, softmax


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


def to_loggable_metrics(stats):
    # bind gpu values into a single tensor (try reducing each before the bind)
    res = {}
    if not stats:
        return res

    gpu_keys, gpu_vals = [], []
    for k, v in stats.items():
        if v is None:
            # filter out all None values
            ...
        elif isinstance(v, torch.Tensor):
            # filter out non-finite values
            if torch.isfinite(v).all():
                gpu_keys.append(k)
                gpu_vals.append(v.detach().mean())
        else:
            res[k] = np.mean(v) if isinstance(v, np.ndarray) else v

    # single transfer to the cpu
    gpu_vals = torch.stack(gpu_vals).cpu()
    # then unbind
    for i, k in enumerate(gpu_keys):
        res[k] = gpu_vals[i].item()
    return res


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def chain(*args):
    def _chain(x):
        if isinstance(x, (list, tuple)):
            yield from x
        else:
            yield x
    
    return tuple(
        x
        for arg in args
        for x in _chain(arg)
    )


def make_layers(
        name, input_size, layers,
        activation=nn.SiLU, std=None, bias=None,
        out_logits=False, is_output=False,
        rng=None, print_module=True,
):
    """
    Create and initialize MLP layers returning the created module (Linear or Sequential) 
    and its output dim. If layers are empty, returns None and input_size.

    Activation defines activation after each layer.
    Bias and std define the initialization (std means gain).
    Out logits means that activation func is not added to the last layer.
    Is output means that the initialization of the last layer will have much 
        smaller default std (if not set explicitly).
    """
    act_name = activation.__name__.lower()
    modules = []
    for il, output_size in enumerate(layers):
        is_last_layer = il == len(layers) - 1
        is_output_layer = is_last_layer and is_output

        _std = std if not is_output_layer else isnone(std, 0.01)

        modules.append(
            init_layer(
                nn.Linear(input_size, output_size),
                std=_std, bias=bias, activation=act_name, rng=rng,
            )
        )
        if not (is_last_layer and out_logits):
            modules.append(activation())
        input_size = output_size

    if len(modules) > 1:
        modules = nn.Sequential(*modules)
    elif len(modules) == 1:
        modules = modules[0]
    else:
        modules = None

    if print_module:
        maybe_print_module(name, modules)
    return input_size, modules


def init_layer(
        layer, std=None, bias=None, activation=None, rng=None
):
    bias = isnone(bias, 0.0)
    if std is None:
        std = nn.init.calculate_gain(
            'relu' if activation == 'silu' else isnone(activation, 'linear')
        )

    if isinstance(layer, (nn.GRUCell, nn.LSTMCell)):
        for name, param in layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, bias)
                # bias_hh + bias_ih are concatenated [i, f, g, o]
                # bump forget gate (second quarter)
                n = param.shape[0]
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)
            elif "weight" in name:
                maybe_with_generator(nn.init.orthogonal_, rng)(param, std)
    else:
        # we assume it's Linear layer with the specified activation afterwards
        maybe_with_generator(nn.init.orthogonal_, rng)(layer.weight, std)
        if hasattr(layer, 'bias') and layer.bias is not None:
            torch.nn.init.constant_(layer.bias, bias)

    return layer


def get_detached_params(m):
    params = {k: v.detach() for k, v in m.named_parameters()}
    buffers = dict(m.named_buffers())
    return params | buffers


def maybe_print_module(name, module):
    if module is None:
        return
    print(f'{name}: ', module)


def maybe_with_generator(fn, rng):
    if rng is not None:
        return partial(fn, generator=rng)
    return fn


def fw(module, x):
    """Call module if it's not None or return input untouched."""
    return module(x) if module is not None else x


def safe_mean(x, default):
    x = x if len(x) > 0 else x.new([default])
    return x.mean()


def to_softmax_distr(q, softmax_temp):
    probs = softmax(q / softmax_temp, dim=-1)
    return torch.distributions.Categorical(probs)


def get_weights_ema_step_fn(weights_ema_lr):
    def ema_step_fn(avg_model_parameter, model_parameter, _):
        lr = weights_ema_lr
        p, avg_p = model_parameter, avg_model_parameter
        return (1.0 - lr) * avg_p + lr * p

    return ema_step_fn

def working_set_penalty(x: torch.Tensor, low=None, high=None):
    # Calculate how far values exceed the maximum boundary
    overshoot = torch.relu(x - high) if high is not None else x.new_zeros((1,))
    # Calculate how far values drop below the minimum boundary
    undershoot = torch.relu(low - x) if low is not None else x.new_zeros((1,))

    # L2 penalty (quadratic): Penalizes larger deviations much harder
    return torch.mean(overshoot**2 + undershoot**2)


def normalize_entropy(h, size):
    return h / torch.log(h.new([size]))


def get_entropy(p, dim=-1, keepdim=False, normalize=False):
    single_val = p.shape[dim] == 1
    if normalize and not single_val:
        p = p / (p.sum(-1, keepdim=True) + 1e-6)

    def _entr(p):
        return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

    H = _entr(p) if not single_val else _entr(p) + _entr(1.0 - p)
    n = max(2, p.shape[dim])
    return H / torch.log(H.new([n]))


class DynamicLearningRate(DynamicParameter):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('name', 'LR')
        super().__init__(*args, **kwargs)
        self.optimiser = None

    def connect_to_optimiser(self, optimiser):
        self.optimiser = optimiser

    def step(self):
        if super().step():
            for pg in self.optimiser.param_groups:
                pg['lr'] = self.val


def huber_from_diff(diff, delta=1.0, reduction='mean'):
    """
    Compute Huber loss given the difference between two tensors.

    NB: Base torch implementation accepts two separate tensors, while
    this function accepts their difference.
    """
    # Ensure delta is a scalar tensor with proper dtype/device
    delta = torch.as_tensor(delta, dtype=diff.dtype, device=diff.device)
    abs_diff = diff.abs()

    # Vectorized computation:
    # 0.5 * min(|d|, δ)^2 + δ * (|d| - min(|d|, δ))
    quadratic = torch.minimum(abs_diff, delta)
    loss = 0.5 * quadratic * quadratic + delta * (abs_diff - quadratic)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
