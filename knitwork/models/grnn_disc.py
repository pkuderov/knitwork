from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.grnn import MessagePassingLayer


class NoveltyGate(nn.Module):
    """Discrete gate in {0, 0.5, 1} based on cosine novelty of message vs hidden state."""

    GATE_LOW  = 0.1
    GATE_MID  = 0.4
    GATE_HIGH = 0.6

    def __init__(self, hidden_size: int, low_thresh: float = 0.25, high_thresh: float = 0.65):
        super().__init__()
        self.low_thresh  = low_thresh
        self.high_thresh = high_thresh

        # learned novelty scorer: [h_new; msg] -> scalar correction
        self.novelty_proj = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        # small initial blend weight: raw cosine dominates early
        self.blend = nn.Parameter(torch.tensor(0.1))

    def _raw_novelty(self, h_new: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        # cosine distance mapped from [-1,1] to [0,1]; [cols, batch, 1]
        cos_sim = F.cosine_similarity(h_new, msg, dim=-1, eps=1e-8)  # [cols, batch]
        return ((1.0 - cos_sim) / 2.0).unsqueeze(-1)

    def _discretize(self, score: torch.Tensor) -> torch.Tensor:
        # straight-through: forward=discrete {low,mid,high}, backward=continuous
        lo, hi = self.low_thresh, self.high_thresh
        discrete = torch.where(
            score < lo,
            torch.full_like(score, self.GATE_LOW),
            torch.where(score > hi, torch.full_like(score, self.GATE_HIGH),
                        torch.full_like(score, self.GATE_MID)),
        )
        return score + (discrete - score).detach()

    def forward(self, h_new: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        # returns discrete gate [cols, batch, 1] in {GATE_LOW, GATE_MID, GATE_HIGH}
        raw     = self._raw_novelty(h_new, msg)
        learned = self.novelty_proj(torch.cat([h_new, msg], dim=-1))
        blend   = torch.sigmoid(self.blend)
        score   = (1.0 - blend) * raw + blend * learned
        return self._discretize(score)


class GridRnnNoveltyGate(nn.Module):
    """GridRNN with NoveltyGate replacing the standard attention gate."""

    def __init__(
        self, *,
        input_size: int, embedding_size: int, output_size: int,
        hidden_size: int, n_layers: int, n_columns: int, n_attn_heads: int,
        col_identities: bool,
        use_bias: bool = True, dropout: float = 0.0,
        novelty_low: float = 0.25, novelty_high: float = 0.65,
    ):
        super().__init__()
        assert n_columns > 1
        self.input_size     = input_size
        self.embedding_size = embedding_size
        self.output_size    = output_size
        self.n_layers       = n_layers
        self.n_columns      = n_columns
        self.n_attn_heads   = n_attn_heads

        self.embedding   = nn.Embedding(input_size, embedding_size)
        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        print(f'GridRNN-NoveltyGate | {n_layers}L x {n_columns}C | hidden={self.hidden_size}')

        self.cells         = nn.ModuleList()
        self.attn          = nn.ModuleList()
        self.novelty_gates = nn.ModuleList()
        for layer in range(n_layers):
            self.cells.append(nn.ModuleList([
                nn.GRUCell(
                    input_size=self._cell_input_dim(layer, ic),
                    hidden_size=self.hidden_size, bias=use_bias, dtype=torch.float64,
                )
                for ic in range(n_columns)
            ]))
            n_part = n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(self.hidden_size, n_attn_heads, n_part))
            self.novelty_gates.append(NoveltyGate(self.hidden_size, novelty_low, novelty_high))

        self.head = nn.Linear(self.hidden_size, output_size)
        print(f'Param count: {format_readable_num(sum(p.numel() for p in self.parameters() if p.requires_grad))}')

    def forward(self, tokens: torch.Tensor, h=None, return_attn: bool = False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2 and tokens.shape[1] == 1
        x = self.embedding(tokens.view(-1))
        h, extras = self.grid_step(x, h=h, return_attn=return_attn)
        y = self.head(h[-1][0])  # top layer, 0-th col
        if return_attn:
            return y, h, extras
        return y, h

    def grid_step(self, x: torch.Tensor, *, h: torch.Tensor, return_attn: bool = True):
        h_n, attn_list, gate_list = [], [], []
        x_in = self._prepare_grid_input(x)
        for cells, attn, nov_gate, hl in zip(self.cells, self.attn, self.novelty_gates, h):
            hl_n = torch.stack(
                [cells[ic](x_in[ic], hl[ic]) for ic in range(self.n_columns)], dim=0
            )  # [cols, batch, hidden]
            msg, attn_w = attn(hl_n, return_weights=return_attn)
            g    = nov_gate(hl_n, msg)           # [cols, batch, 1] in {low, mid, high}
            hl_n = (1.0 - g) * hl_n + g * msg
            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)
            x_in = hl_n
        return torch.stack(h_n, dim=0), {"attn_weights": attn_list, "gates": gate_list}

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        return self.hidden_size

    def _prepare_grid_input(self, x: torch.Tensor):
        bsz, _ = x.shape
        dummy = torch.zeros(bsz, 1, device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)

    def init_state(self, bsz: int) -> torch.Tensor:
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype,
        )

    def reset_state(self, state, reset_mask):
        if state is None:
            return self.init_state(reset_mask.shape[0])
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state
        state = state.clone()
        state[:, :, ixs, :] *= 0.0
        return state

    def detach_state(self, state):
        return state.detach() if state is not None else None
