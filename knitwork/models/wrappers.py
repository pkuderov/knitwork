from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import count_learnable_params, format_readable_num


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
            dtype=dtype, device=device,
            **rnn
        )
        self.hidden_size = self.embedding_size = self.rnn.hidden_size

        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.head = nn.Linear(self.hidden_size, self.output_size)

        print(f'Param count: {count_learnable_params(self, as_str=True)}')

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


class RLTokenModel(nn.Module):
    """Token-input recurrent actor-critic built around a core model."""

    def __init__(
            self, *,
            input_size, output_size,
            rnn: dict, rnn_fn,
            dtype, device,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.rnn = rnn_fn(dtype=dtype, device=device, **rnn)
        self.hidden_size = self.rnn.hidden_size

        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.policy_head = nn.Linear(self.hidden_size, output_size)
        self.value_head = nn.Linear(self.hidden_size, 1)

        print(f'Param count: {count_learnable_params(self, as_str=True)}')

    def forward(self, tokens, state, *, capture=False, **kwargs):
        x = self.embedding(tokens).transpose(0, 1)
        z, state, info = self.rnn(x, state, capture=capture, **kwargs)
        return self.policy_head(z), self.value_head(z).squeeze(-1), state, info

    def reset_state(self, state, reset_mask):
        return self.rnn.reset_state(state, reset_mask)

    def detach_state(self, state):
        return self.rnn.detach_state(state)


class RLVectorModel(nn.Module):
    """Vector-input recurrent actor-critic built around a core model."""

    def __init__(
            self, *,
            input_size, output_size,
            rnn: dict, rnn_fn,
            dtype, device,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.rnn = rnn_fn(dtype=dtype, device=device, **rnn)
        self.hidden_size = self.rnn.hidden_size

        self.encoder = nn.Linear(input_size, self.hidden_size)
        self.policy_head = nn.Linear(self.hidden_size, output_size)
        self.value_head = nn.Linear(self.hidden_size, 1)

        print(f'Param count: {count_learnable_params(self, as_str=True)}')

    def forward(self, obs, state, *, capture=False, **kwargs):
        x = self.encoder(obs.to(self.dtype)).unsqueeze(0)
        z, state, info = self.rnn(x, state, capture=capture, **kwargs)
        return self.policy_head(z), self.value_head(z).squeeze(-1), state, info

    def reset_state(self, state, reset_mask):
        return self.rnn.reset_state(state, reset_mask)

    def detach_state(self, state):
        return self.rnn.detach_state(state)
