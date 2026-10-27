"""One-column, widened none control using the established GRU/message updates."""

import torch
from torch import nn

from knitwork.models.grnn_core import GruBank2
from knitwork.models.grnn_communication import ControlledMessagePassingLayer, GridRnn as ControlledGridRnn


class SingleColumnMessagePassing(ControlledMessagePassingLayer):
    def forward(self, q, k, v, return_weights=False):
        message, info = super().forward(q, k, v, return_weights=return_weights)
        if self.training and k.shape[0] == 1:
            # A deterministic one-source route has zero entropy; log(1) cannot normalize it.
            info['comm_entropy'] = message.new_zeros(1)
        return message, info


class GridRnn(ControlledGridRnn):
    def __init__(
        self, *, hidden_size, n_layers, n_columns, n_attn_heads,
        dtype, device, routing='none', readout='column0', use_bias=True,
        ln_msg=True, bank=2, mha=2, noise_std=0.0, top_k=None,
        n_inputs=1, n_outputs=1,
    ):
        nn.Module.__init__(self)
        if (n_columns, n_inputs, n_outputs, routing, readout) != (1, 1, 1, 'none', 'column0'):
            raise ValueError('Single-column control requires C=1, one input/output and none_column0')
        if bank != 2 or mha != 2 or top_k is not None or not use_bias:
            raise ValueError('Single-column control requires bank=2, mha=2, dense projections and bias')
        if n_layers < 1 or hidden_size < 1 or n_attn_heads < 1 or hidden_size % n_attn_heads:
            raise ValueError('Require positive layers/width/heads and width divisible by heads')
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.n_attn_heads = n_attn_heads
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dtype = dtype
        self.device = device
        self.routing = routing
        self.readout = readout
        print(f'GridRNN of {n_layers}L x 1C GRU cells w/ {hidden_size} hidden units')
        self.cells = GruBank2(n_layers, 1, hidden_size, bias=use_bias)
        self.attn = nn.ModuleList()
        for layer_index in range(n_layers):
            n_kv = 2 if layer_index == 0 else 1
            layer = SingleColumnMessagePassing(
                hidden_size, num_heads=n_attn_heads, ln_msg=ln_msg,
                n_q=1, n_kv=n_kv, noise_std=noise_std, top_k=None,
            )
            layer.routing = 'none'
            layer.register_buffer('local_sources', torch.ones(1, n_kv, dtype=torch.bool), persistent=False)
            self.attn.append(layer)
