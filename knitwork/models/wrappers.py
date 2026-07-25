from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import format_readable_num


class TokenModel(nn.Module):
    def __init__(
            self, *,
            input_size, hidden_size, output_size,
            rnn_core: dict, rnn_core_fn,
            dtype, device
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, self.embedding_size)

        self.rnn_core = rnn_core_fn(
            hidden_size=hidden_size,
            n_inputs=1, n_outputs=1,
            dtype=dtype, device=device,
            **rnn_core
        )

        # Head reads from the top layer, 0-th column (the external column)
        self.head = nn.Linear(self.hidden_size, self.output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total model param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, state: dict, *, out_attn=False):
        x = self.embedding(tokens.view(-1))
        z, state, info = self.rnn_core(x, state, out_attn=out_attn)
        y = self.head(z)

        return y, state, info

    def reset_state(self, state, reset_mask):
        return self.rnn_core.reset_state(state, reset_mask)

    def detach_state(self, state):
        return self.rnn_core.detach_state(state)

    def init_state(self, bsz):
        return self.rnn_core.init_state(bsz)
