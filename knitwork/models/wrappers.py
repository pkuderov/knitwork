from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import format_readable_num


class TokenModel(nn.Module):
    def __init__(
            self, *,
            input_size, output_size,
            rnn: dict, rnn_fn,
            dtype, device
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype
        self.device = device

        # make first to get the real hidden size (could be mod n_attn_heads)
        self.rnn = rnn_fn(
            n_inputs=1, n_outputs=1,
            dtype=dtype, device=device,
            **rnn
        )
        self.hidden_size = self.embedding_size = self.rnn.hidden_size

        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.head = nn.Linear(self.hidden_size, self.output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total model param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, state: dict, *, capture=False, **kwargs):
        # (B, 1) -> (B, 1, E) -> (1, B, E)
        x = self.embedding(tokens)
        x = x.transpose(0, 1)

        z, state, info = self.rnn(x, state, capture=capture, **kwargs)
        y = self.head(z)

        return y, state, info

    def reset_state(self, state, reset_mask):
        return self.rnn.reset_state(state, reset_mask)

    def detach_state(self, state):
        return self.rnn.detach_state(state)

    def init_state(self, bsz):
        return self.rnn.init_state(bsz)

    @property
    def has_attn(self):
        return getattr(self.rnn, 'has_attn', False)
