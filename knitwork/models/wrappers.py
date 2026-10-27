from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import count_learnable_params, format_readable_num


def _init_rl_heads(model, policy_gain):
    if policy_gain is None:
        return
    nn.init.orthogonal_(model.policy_head.weight, gain=policy_gain)
    nn.init.zeros_(model.policy_head.bias)
    nn.init.orthogonal_(model.value_head.weight, gain=1.0)
    nn.init.zeros_(model.value_head.bias)


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
        # (B, 1) -> (B, 1, E)
        x = self.embedding(tokens)

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
            dtype, device, feature_norm=False, policy_gain=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.rnn = rnn_fn(dtype=dtype, device=device, **rnn)
        self.hidden_size = self.rnn.hidden_size
        self.feature_norm = feature_norm

        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.policy_head = nn.Linear(self.hidden_size, output_size)
        self.value_head = nn.Linear(self.hidden_size, 1)
        _init_rl_heads(self, policy_gain)

        print(f'Param count: {count_learnable_params(self, as_str=True)}')

    def forward(self, tokens, state, *, capture=False, **kwargs):
        x = self.embedding(tokens)
        if not getattr(self.rnn, 'batch_first', False):
            x = x.transpose(0, 1)
        z, state, info = self.rnn(x, state, capture=capture, **kwargs)
        if self.feature_norm:
            z = torch.nn.functional.rms_norm(z, (self.hidden_size,))
        return self.policy_head(z), self.value_head(z).squeeze(-1), state, info

    def reset_state(self, state, reset_mask):
        return self.rnn.reset_state(state, reset_mask)

    def detach_state(self, state):
        return self.rnn.detach_state(state)


class RLVectorModel(nn.Module):
    """Vector-input recurrent actor-critic built around a core model.

    readout='columns' feeds all top-layer grid columns to the heads (default: core output only);
    obs_skip=True also feeds the encoded current observation to the heads.
    policy='ordinal' scores ordered actions k by -scale * (k - mu)^2 from two head outputs (counting tasks).
    aux_lags=(k, ...) adds a head predicting the observations k steps back from the head features (aux memory loss).
    """

    def __init__(
            self, *,
            input_size, output_size,
            rnn: dict, rnn_fn,
            dtype, device, feature_norm=False, policy_gain=None,
            readout='core', obs_skip=False, policy='categorical', aux_lags=(),
    ):
        super().__init__()
        assert readout in ('core', 'columns')
        assert policy in ('categorical', 'ordinal')
        self.policy = policy
        self.output_size = output_size
        self.dtype = dtype
        self.device = device
        self.rnn = rnn_fn(dtype=dtype, device=device, **rnn)
        self.hidden_size = self.rnn.hidden_size
        self.feature_norm = feature_norm
        self.readout = readout
        self.obs_skip = obs_skip

        n_feat = getattr(self.rnn, 'n_columns', 1) if readout == 'columns' else 1
        head_size = self.hidden_size * (n_feat + int(obs_skip))
        self.encoder = nn.Linear(input_size, self.hidden_size)
        self.policy_head = nn.Linear(head_size, 2 if policy == 'ordinal' else output_size)
        self.value_head = nn.Linear(head_size, 1)
        self.aux_lags = tuple(aux_lags)
        self.aux_head = nn.Linear(head_size, input_size * len(self.aux_lags)) if self.aux_lags else None
        _init_rl_heads(self, policy_gain)

        print(f'Param count: {count_learnable_params(self, as_str=True)}')

    def forward(self, obs, state, *, capture=False, **kwargs):
        x = self.encoder(obs.to(self.dtype)).unsqueeze(1)
        if not getattr(self.rnn, 'batch_first', False):
            x = x.transpose(0, 1)
        z, state, info = self.rnn(x, state, capture=capture, **kwargs)
        B = z.shape[0]
        # [B, n_feat, H]: core output or all top columns, optionally plus the encoded observation
        z = self._top_columns(state) if self.readout == 'columns' and hasattr(self.rnn, 'n_columns') else z.view(B, 1, -1)
        if self.obs_skip:
            z = torch.cat([z, x.reshape(B, 1, -1)], dim=1)
        if self.feature_norm:
            z = torch.nn.functional.rms_norm(z, (self.hidden_size,))
        z = z.flatten(1)
        # aux predictions [B, K * obs_dim], else the head features; made inside forward so CUDA graphs capture the head
        info['aux'] = self.aux_head(z) if self.aux_head is not None else z
        logits = self.policy_head(z)
        if self.policy == 'ordinal':
            # mu in [0, A-1]; scale 0.01 exp(.) starts near-uniform over A ~ 26 actions, exp lets it sharpen
            A = self.output_size
            mu = (A - 1) * torch.sigmoid(logits[:, :1])
            scale = 0.01 * torch.exp(logits[:, 1:].clamp(max=8.0))
            k = torch.arange(A, device=z.device, dtype=z.dtype)
            logits = -scale * (k - mu) ** 2
        return logits, self.value_head(z).squeeze(-1), state, info

    @staticmethod
    def _top_columns(state):
        # grid-LRU family keeps top outputs as outs [L, B, C, H]; grid-GRU keeps h [L, C, B, H]
        if 'outs' in state:
            return state['outs'][-1]
        return state['h'][-1].transpose(0, 1)

    def reset_state(self, state, reset_mask):
        return self.rnn.reset_state(state, reset_mask)

    def detach_state(self, state):
        return self.rnn.detach_state(state)
