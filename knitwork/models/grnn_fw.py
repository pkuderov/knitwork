"""GridRNN with Fast Weights (Ba et al. 2016)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from knitwork.common.utils import format_readable_num, to_torch


class FastWeightMessagePassing(nn.Module):
    """Hebbian fast-weight matrix A for cross-column message passing."""

    def __init__(
        self, *,
        hidden_size: int,
        n_columns: int,
        decay: float = 0.9,
        fw_lr: float = 0.5,
        n_participants: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_columns   = n_columns
        self.decay       = decay
        self.fw_lr       = fw_lr
        self.scale       = hidden_size ** -0.5

        self.norm   = nn.LayerNorm(hidden_size)
        self.proj_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_v = nn.Linear(hidden_size, hidden_size, bias=False)
        for proj in (self.proj_k, self.proj_q, self.proj_v):
            nn.init.normal_(proj.weight, 0.0, 0.01)

        self.ids = None
        if n_participants is not None:
            self.ids = nn.Parameter(torch.zeros(n_participants, hidden_size))
            nn.init.normal_(self.ids, 0.0, 0.01)

    def forward(self, h, A, return_weights=False):
        # h: (cols, batch, hidden),  A: (batch, hidden, hidden)
        n_cols, batch, hsz = h.shape
        h_flat = h.reshape(n_cols * batch, hsz)
        k = self.proj_k(h_flat).view(n_cols, batch, hsz)
        q = self.proj_q(h_flat).view(n_cols, batch, hsz)
        v = self.proj_v(h_flat).view(n_cols, batch, hsz)

        if self.ids is not None:
            k = k + self.ids.unsqueeze(1)
            q = q + self.ids.unsqueeze(1)

        # Hebbian write: A += v_j ⊗ k_j  for each column j
        delta_A = torch.zeros_like(A)
        for col_j in range(n_cols):
            k_j = F.normalize(k[col_j], dim=-1)   # (batch, hidden)
            v_j = F.normalize(v[col_j], dim=-1)
            delta_A += torch.bmm(v_j.unsqueeze(2), k_j.unsqueeze(1))
        A_new = self.decay * A + (self.fw_lr / n_cols) * delta_A

        # content-based retrieval
        msgs = []
        for col_i in range(n_cols):
            q_i = F.normalize(q[col_i], dim=-1)
            msgs.append(torch.bmm(A_new, q_i.unsqueeze(2)).squeeze(2))
        h_msg = self.norm(torch.stack(msgs, dim=0))   # (cols, batch, hidden)

        # pseudo attention weights for visualization  [n_cols, n_cols]
        attn_w = None
        if return_weights:
            q_mat  = F.normalize(q.mean(dim=1), dim=-1)
            k_mat  = F.normalize(k.mean(dim=1), dim=-1)
            scores = torch.matmul(q_mat, k_mat.T)
            attn_w = torch.softmax(scores / self.scale, dim=-1).detach()

        return h_msg, A_new, attn_w


class GridRnnFW(nn.Module):
    """GridRNN with fast-weight associative memory instead of attention."""

    def __init__(
        self, *,
        input_size, embedding_size, output_size,
        hidden_size: int,
        n_layers: int, n_columns: int,
        n_attn_heads,   # unused, kept for config compatibility
        messaging: str = "post",
        col_identities,
        fw_decay: float = 0.9,
        fw_lr: float = 0.5,
        use_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size     = input_size
        self.embedding_size = embedding_size
        self.output_size    = output_size
        self.embedding      = nn.Embedding(input_size, embedding_size)

        self.n_layers  = n_layers
        assert n_columns > 1
        self.n_columns = n_columns
        self.fw_decay  = fw_decay
        self.fw_lr     = fw_lr
        self.hidden_size = hidden_size

        print(
            f'GridRNN-FW {n_layers}L x {n_columns}C'
            f' | hidden={hidden_size} decay={fw_decay} fw_lr={fw_lr}'
        )

        self.cells      = nn.ModuleList()
        self.fw_layers  = nn.ModuleList()
        self.attn_gates = nn.ModuleList()

        for layer in range(n_layers):
            row = nn.ModuleList([
                nn.GRUCell(
                    input_size  = self._cell_input_dim(layer, icol),
                    hidden_size = self.hidden_size,
                    bias        = use_bias,
                    dtype       = torch.float64,
                )
                for icol in range(n_columns)
            ])
            self.cells.append(row)

            n_participants = n_columns if col_identities else None
            self.fw_layers.append(FastWeightMessagePassing(
                hidden_size    = self.hidden_size,
                n_columns      = n_columns,
                decay          = fw_decay,
                fw_lr          = fw_lr,
                n_participants = n_participants,
            ))
            self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        self.head = nn.Linear(self.hidden_size, output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, state=None, return_attn: bool = False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2 and tokens.shape[1] == 1

        x = self.embedding(tokens.view(-1))   # (batch, emb)
        h_new, A_new, extras = self._grid_step(x, state=state, return_attn=return_attn)

        y = self.head(h_new[-1][0])   # top layer, col 0
        new_state = (h_new, A_new)

        if return_attn:
            return y, new_state, extras
        return y, new_state

    def _grid_step(self, x, *, state, return_attn: bool):
        h, A = state if state is not None else (None, None)
        if h is None:
            bsz = x.shape[0]
            h = self._init_h(bsz)
            A = self._init_A(bsz)

        h_n, A_n, attn_list, gate_list = [], [], [], []
        x_list = self._prepare_grid_input(x)   # list[Tensor(batch, dim)]

        for cells, fw, gate_lin, hl, Al in zip(self.cells, self.fw_layers, self.attn_gates, h, A):
            hl_n = torch.stack([
                cells[ic](x_list[ic], hl[ic]) for ic in range(self.n_columns)
            ], dim=0)   # (cols, batch, hidden)

            msg, Al_new, attn_w = fw(hl_n, Al, return_weights=return_attn)

            g    = torch.sigmoid(gate_lin(torch.cat([hl_n, msg], dim=-1)))  # (cols, batch, 1)
            hl_n = (1 - g) * hl_n + g * msg                                # (cols, batch, hidden)

            h_n.append(hl_n)
            A_n.append(Al_new)
            attn_list.append(attn_w)
            gate_list.append(g)
            x_list = hl_n

        h_n = torch.stack(h_n, dim=0)   # (n_layers, cols, batch, hidden)
        A_n = torch.stack(A_n, dim=0)   # (n_layers, batch, hidden, hidden)
        return h_n, A_n, {"attn_weights": attn_list, "gates": gate_list}

    def reset_state(self, state, reset_mask):
        if state is None:
            return None
        h, A = state
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state
        h = h.clone(); h[:, :, ixs, :] = 0.0
        A = A.clone(); A[:, ixs, :, :] = 0.0
        return (h, A)

    def detach_state(self, state):
        if state is None:
            return state
        h, A = state
        return (h.detach(), A.detach())

    def _init_h(self, bsz: int) -> torch.Tensor:
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype,
        )

    def _init_A(self, bsz: int) -> torch.Tensor:
        return torch.zeros(
            self.n_layers, bsz, self.hidden_size, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype,
        )

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        return self.hidden_size

    def _prepare_grid_input(self, x):
        bsz   = x.shape[0]
        in_dim = self._cell_input_dim(ix_layer=0, ix_col=1)
        dummy  = torch.zeros(bsz, in_dim, device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)
