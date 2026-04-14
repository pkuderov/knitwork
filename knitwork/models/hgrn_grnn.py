from __future__ import annotations

import math
import torch
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.grnn import MessagePassingLayer


class HGRUCell(nn.Module):
    """Hierarchically Gated Recurrent Unit cell."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        beta_init: float = 0.0,
        use_bias: bool = True,
        dtype=torch.float64,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dtype = dtype

        # forget gate: lambda_t = sigmoid(W_f*x + U_f*h) * (1-beta) + beta
        self.W_f = nn.Linear(input_size, hidden_size, bias=use_bias, dtype=dtype)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        # output gate: o_t = sigmoid(W_o*x + U_o*h)
        self.W_o = nn.Linear(input_size, hidden_size, bias=use_bias, dtype=dtype)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        # content candidate: c_t = tanh(W_c*x + U_c*(o_t * h))
        self.W_c = nn.Linear(input_size, hidden_size, bias=use_bias, dtype=dtype)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)

        # learnable lower bound for forget gate; stored pre-sigmoid so beta in (0,1)
        self.beta_raw = nn.Parameter(
            torch.tensor(self._inv_sigmoid(beta_init + 1e-6), dtype=dtype)
        )
        self._reset_parameters()

    @staticmethod
    def _inv_sigmoid(x: float) -> float:
        x = max(1e-6, min(1.0 - 1e-6, x))
        return math.log(x / (1.0 - x))

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.ndim == 2:
                nn.init.orthogonal_(p)
            elif p.ndim == 1 and 'bias' in name:
                nn.init.zeros_(p)

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_raw)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: [B, input_size], h: [B, hidden_size]
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h))
        c_t = torch.tanh(self.W_c(x) + self.U_c(o_t * h))
        # lambda in [beta, 1]: higher beta -> slower forgetting (upper layers)
        raw_f = torch.sigmoid(self.W_f(x) + self.U_f(h))
        lam_t = raw_f * (1.0 - self.beta) + self.beta
        return lam_t * h + (1.0 - lam_t) * c_t   # [B, hidden_size]


class HGRN_GridRnn(nn.Module):
    """Grid RNN with HGRU cells and layer-wise monotone forget-gate lower bounds."""

    def __init__(
        self,
        *,
        input_size: int,
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        n_columns: int,
        n_attn_heads: int,
        messaging: str = "post",
        col_identities: bool,
        use_bias: bool = True,
        dropout: float = 0.0,
        beta_min: float = 0.0,
        beta_max: float = 0.99,
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, embedding_size)

        self.n_layers = n_layers
        assert n_columns > 1, "n_columns must be > 1"
        self.n_columns = n_columns
        self.n_attn_heads = n_attn_heads
        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        self.use_postmsg = (messaging == "post")

        print(
            f'GridRNN (HGRN-adapted) {n_layers}L x {n_columns}C HGRU'
            f' hidden={self.hidden_size}'
        )

        # beta linearly spaced: layer 0 -> beta_min, layer L-1 -> beta_max
        if n_layers > 1:
            betas = [
                beta_min + (beta_max - beta_min) * i / (n_layers - 1)
                for i in range(n_layers)
            ]
        else:
            betas = [beta_min]
        print(f'  Layer betas: {[round(b, 3) for b in betas]}')

        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()

        for layer_idx in range(n_layers):
            row = nn.ModuleList([
                HGRUCell(
                    input_size=self._cell_input_dim(layer_idx, icol),
                    hidden_size=self.hidden_size,
                    beta_init=betas[layer_idx],
                    use_bias=use_bias,
                    dtype=torch.float64,
                )
                for icol in range(n_columns)
            ])
            self.cells.append(row)

            n_participants = n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(
                self.hidden_size, num_heads=n_attn_heads, n_participants=n_participants,
            ))
            if self.use_postmsg:
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        # output gate before head (HGRN-style)
        self.final_output_gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid(),
        )
        self.head = nn.Linear(self.hidden_size, output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, h=None, return_attn: bool = False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2
        bsz, n_features = tokens.shape
        assert n_features == 1, "expected input features = 1 (token ids)"

        x = self.embedding(tokens.view(-1))   # [B, embedding_size]

        if self.use_postmsg:
            h, extras = self.grid_step_postmsg(x, h=h, return_attn=return_attn)
        else:
            h, extras = self.grid_step_premsg(x, h=h), {}

        z = h[-1][0]                          # top layer, first col [B, H]
        z = self.final_output_gate(z) * z
        y = self.head(z)
        if return_attn:
            return y, h, extras
        return y, h

    def grid_step_postmsg(self, x: torch.Tensor, *, h, return_attn: bool = True):
        h_n, attn_list, gate_list = [], [], []
        x = self._prepare_grid_input(x)

        for cells, attn, attn_gate, hl in zip(self.cells, self.attn, self.attn_gates, h):
            hl_n = torch.stack([
                self.cell_forward(cells, x, hl, ix_col=ic)
                for ic in range(self.n_columns)
            ], dim=0)   # [cols, B, H]

            msg, attn_w = attn(hl_n, return_weights=return_attn)
            g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
            hl_n = (1.0 - g) * hl_n + g * msg

            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)
            x = hl_n

        return torch.stack(h_n, dim=0), {"attn_weights": attn_list, "gates": gate_list}

    def grid_step_premsg(self, x: torch.Tensor, *, h):
        h_n = []
        x = self._prepare_grid_input(x)
        first_row = True

        for cells, attn, hl in zip(self.cells, self.attn, h):
            msg, _ = attn(hl, return_weights=False)
            if first_row:
                x = [torch.cat([xc, msgc], dim=-1) for xc, msgc in zip(x, msg)]
            else:
                x = torch.cat([x, msg], dim=-1)   # type: ignore

            hl_n = torch.stack([
                self.cell_forward(cells, x, hl, ix_col=ic)
                for ic in range(self.n_columns)
            ], dim=0)

            h_n.append(hl_n)
            x = hl_n
            first_row = False

        return torch.stack(h_n, dim=0)

    def cell_forward(self, cells, x, h, *, ix_col: int) -> torch.Tensor:
        return cells[ix_col](x[ix_col], h[ix_col])

    def reset_state(self, state, reset_mask):
        if state is None:
            return self.init_state(reset_mask.shape[0])
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state
        h = state.clone()
        h[:, :, ixs, :] *= 0.0
        return h

    def detach_state(self, state):
        return state.detach() if state is not None else None

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        hsz = self.hidden_size
        if not self.use_postmsg:
            hsz += self.hidden_size
        return hsz

    def _prepare_grid_input(self, x: torch.Tensor) -> list:
        bsz, _ = x.shape
        in_dim = self._cell_input_dim(ix_layer=0, ix_col=1)
        dummy = torch.zeros(bsz, in_dim, device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)

    def init_state(self, bsz: int) -> torch.Tensor:
        # [layers, cols, batch, hidden]
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device,
            dtype=self.head.weight.dtype,
        )
