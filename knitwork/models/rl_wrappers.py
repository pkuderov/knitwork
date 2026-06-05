"""Continuous-input wrappers for knitwork recurrent models.

Each wrapper replaces the internal nn.Embedding with nn.Linear so that
float observation vectors [B, obs_dim] can be used directly. The recurrent
core (grid cells, attention, etc.) is inherited unchanged.
"""
from __future__ import annotations

import torch
from torch import nn

from knitwork.models.grnn      import GridRnn
from knitwork.models.grnn_lru  import GridLRU
from knitwork.models.hgrnn     import HopfieldGridRnn
from knitwork.models.hgrnn_lru import HopfieldGridLRU
from knitwork.models.gru       import GruBaseline


def _patch_embedding(model, obs_dim: int):
    """Replace nn.Embedding with nn.Linear in-place."""
    model.embedding = nn.Linear(obs_dim, model.embedding_size, bias=False)


# ---------------------------------------------------------------------------
# GridRnn — grid_step_* returns (h, extras)

class GridRnnContinuous(GridRnn):
    def __init__(self, *, obs_dim: int, **kwargs):
        super().__init__(input_size=1, **kwargs)
        _patch_embedding(self, obs_dim)

    def forward(self, obs: torch.Tensor, h=None, return_attn=False):
        x = self.embedding(obs.float())             # [B, embedding_size]
        if self.use_postmsg:
            h, extras = self.grid_step_postmsg(x, h=h, return_attn=return_attn)
        else:
            h, extras = self.grid_step_premsg(x, h=h), {}
        z = h[-1][0]
        y = self.head(z)
        if return_attn:
            return y, h, extras
        return y, h


# ---------------------------------------------------------------------------
# GridLRU — grid_step_* returns (h, last_out, extras); use last_out[0] as z

class GridLRUContinuous(GridLRU):
    def __init__(self, *, obs_dim: int, **kwargs):
        super().__init__(input_size=1, **kwargs)
        _patch_embedding(self, obs_dim)

    def forward(self, obs: torch.Tensor, h=None, return_attn=False):
        x = self.embedding(obs.float())             # [B, embedding_size]
        if self.use_postmsg:
            h, last_out, extras = self.grid_step_postmsg(x, h=h, return_attn=return_attn)
        else:
            h, last_out, extras = self.grid_step_premsg(x, h=h)
        z = last_out[0]                             # top layer, col 0  [B, H]
        y = self.head(z)
        if return_attn:
            return y, h, extras
        return y, h


# ---------------------------------------------------------------------------
# HopfieldGridRnn — LSTM state (h, c); grid_step returns (h, c)

class HopfieldGridRnnContinuous(HopfieldGridRnn):
    def __init__(self, *, obs_dim: int, **kwargs):
        super().__init__(input_size=1, **kwargs)
        _patch_embedding(self, obs_dim)

    def forward(self, obs: torch.Tensor, state=None):
        x = self.embedding(obs.float())             # [B, embedding_size]
        h, c = state
        if self.use_postmsg:
            h, c = self.grid_step_postmsg(x, h=h, c=c)
        else:
            h, c = self.grid_step_premsg(x, h=h, c=c)
        y = self.head(h[-1][0])
        return y, (h, c)


# ---------------------------------------------------------------------------
# HopfieldGridLRU — state is h tensor; _grid_step_* returns (h, attn, gates)

class HopfieldGridLRUContinuous(HopfieldGridLRU):
    def __init__(self, *, obs_dim: int, **kwargs):
        super().__init__(input_size=1, **kwargs)
        _patch_embedding(self, obs_dim)

    def forward(self, obs: torch.Tensor, state=None, return_attn=False):
        x = self.embedding(obs.float())             # [B, embedding_size]
        if self.use_postmsg:
            h_new, all_attn, all_gates = self._grid_step_postmsg(x, h=state)
        else:
            h_new, all_attn, all_gates = self._grid_step_premsg(x, h=state)
        z = h_new[-1, 0, :, :self.hidden_size]      # Re-part, top layer, col 0
        y = self.head(z)
        if return_attn:
            return y, h_new, {"attn_weights": all_attn, "gates": all_gates}
        return y, h_new


# ---------------------------------------------------------------------------
# GruBaseline — GRU; state is [layers, B, H]

class GruBaselineContinuous(GruBaseline):
    def __init__(self, *, obs_dim: int, **kwargs):
        super().__init__(input_size=1, **kwargs)
        _patch_embedding(self, obs_dim)

    def forward(self, obs: torch.Tensor, h0=None):
        x = self.embedding(obs.float())             # [B, embedding_size]
        x = x.unsqueeze(0)                          # [1, B, embedding_size]
        y, hN = self.rnn(x, h0)
        logits = self.head(y.squeeze(0))
        return logits, hN
