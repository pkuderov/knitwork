"""A stacked, single-column LRU control using the Grid-LRU cell exactly.

There is no inter-column router or recurrent feedback from emitted messages.
"""

import math

import torch
from torch import nn

from knitwork.models.grnn_lru_core import LruBank


class LruCore(nn.Module):
    batch_first = True
    has_attn = False

    def __init__(self, *, hidden_size, n_layers, horizon, dtype, device):
        super().__init__()
        if type(hidden_size) is not int or hidden_size < 1 or type(n_layers) is not int or n_layers < 1:
            raise ValueError('hidden_size and n_layers must be positive integers')
        self.hidden_size, self.n_layers = hidden_size, n_layers
        self.dtype, self.device = dtype, device
        self.cells = LruBank(n_layers=n_layers, n_columns=1, hidden_size=hidden_size, horizon=horizon)

    def forward(self, x, state, **_):
        if x.shape[1] != 1:
            raise ValueError('Expected batch-first input with one token per step')
        hidden = []
        for layer in range(self.n_layers):
            out, h = self.cells(layer, x.transpose(0, 1), state['h'][layer].unsqueeze(0))
            hidden.append(h[0])
            x = out
        return x[:, 0], {'h': torch.stack(hidden)}, {}

    def init_state(self, bsz):
        h = torch.empty(self.n_layers, bsz, 2 * self.hidden_size, dtype=self.dtype, device=self.device)
        h.normal_(0, 0.01 / math.sqrt(self.hidden_size))
        return {'h': h}

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            return self.init_state(reset_mask.shape[0] if reset_mask is not None else bsz)
        return {'h': state['h'] * (~reset_mask.flatten())[None, :, None]}

    def detach_state(self, state):
        return None if state is None else {'h': state['h'].detach()}

    def carried_state_floats(self):
        return 2 * self.n_layers * self.hidden_size
