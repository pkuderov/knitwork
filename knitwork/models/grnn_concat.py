"""Independent columns with a learned linear prediction over concatenated states."""

import torch
from torch import nn

from knitwork.common.utils import count_learnable_params
from knitwork.models.grnn_communication import GridRnn as ControlledGridRnn
from knitwork.models.wrappers import TokenModel as BaseTokenModel


class GridRnn(ControlledGridRnn):
    def __init__(self, *, routing='none', readout='concat', **kwargs):
        if routing != 'none' or readout != 'concat':
            raise ValueError('This control requires none routing and concat readout')
        super().__init__(routing='none', readout='column0', **kwargs)
        self.readout = 'concat'

    def forward(self, x, state, *, capture=False, **kwargs):
        _, state, info = super().forward(x, state, capture=capture, **kwargs)
        top = state['h'][-1]
        output = top.permute(1, 0, 2).reshape(top.shape[1], self.n_columns * self.hidden_size)
        return output, state, info


class TokenModel(BaseTokenModel):
    def __init__(self, *, input_size, output_size, rnn, rnn_fn, dtype, device):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype
        self.device = device
        self.rnn = rnn_fn(dtype=dtype, device=device, **rnn)
        self.hidden_size = self.embedding_size = self.rnn.hidden_size
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        original = nn.Linear(self.hidden_size, output_size)
        self.head = nn.Linear(self.rnn.n_columns * self.hidden_size, self.output_size)
        # Start with the same prediction as mean readout; column weights can then diverge.
        with torch.no_grad():
            self.head.weight.copy_(original.weight.repeat(1, self.rnn.n_columns) / self.rnn.n_columns)
            self.head.bias.copy_(original.bias)
        print(f'Concat model parameter count: {count_learnable_params(self):,}')
