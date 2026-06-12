"""GridRNN with Preconditioned Delta Rule fast weights.

Delta rule: error-corrective KV update (erase old, write new).
Preconditioning: Adam-style second moment of keys scales each update,
normalizing the effective learning rate per key dimension.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch


class PrecDeltaMessagePassing(nn.Module):
    """Cross-column message passing via preconditioned delta rule."""

    def __init__(
        self, *,
        hidden_size: int,
        n_columns: int,
        delta_lr: float = 0.01,
        delta_decay: float = 0.99,
        beta2: float = 0.999,
        eps: float = 1e-8,
        n_participants: int | None = None,
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_columns    = n_columns
        self.delta_lr     = delta_lr
        self.delta_decay  = delta_decay
        self.beta2        = beta2
        self.eps          = eps
        self.scale        = hidden_size ** 0.5

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

    def forward(self, h, A, v2, return_weights: bool = False):
        # h:  (cols, batch, H)
        # A:  (batch, H, H)  — fast-weight matrix
        # v2: (batch, H)     — second moment of keys (Adam estimate)
        n_cols, batch, H = h.shape
        h_flat = h.reshape(n_cols * batch, H)
        k = self.proj_k(h_flat).view(n_cols, batch, H)
        q = self.proj_q(h_flat).view(n_cols, batch, H)
        v = self.proj_v(h_flat).view(n_cols, batch, H)

        if self.ids is not None:
            k = k + self.ids.unsqueeze(1)
            q = q + self.ids.unsqueeze(1)

        # compute all errors from the same initial A, then apply one joint update
        # (parallel delta rule — avoids chained Jacobians through sequential A reads)
        v2_new = v2
        delta_A = torch.zeros_like(A)
        for col_j in range(n_cols):
            k_j = F.normalize(k[col_j], dim=-1)   # (batch, H)
            v_j = F.normalize(v[col_j], dim=-1)

            # second moment: running EMA of k^2; detach — running stat, not differentiable param
            v2_new = self.beta2 * v2_new + (1 - self.beta2) * (k_j.detach() ** 2)  # (batch, H)

            # prediction error from the same initial A (parallel, not sequential)
            error = torch.bmm(A.detach(), k_j.unsqueeze(2)).squeeze(2) - v_j       # (batch, H)

            # preconditioned key: scale by 1/sqrt(v2)
            k_prec = k_j / (v2_new.sqrt() + self.eps)                              # (batch, H)

            delta_A = delta_A + torch.bmm(
                error.unsqueeze(2),    # (batch, H, 1)
                k_prec.unsqueeze(1),   # (batch, 1, H)
            )

        # single joint update: decay + preconditioned delta writes
        A = self.delta_decay * A - self.delta_lr * delta_A

        # content-based retrieval
        msgs = []
        for col_i in range(n_cols):
            q_i = F.normalize(q[col_i], dim=-1)
            msgs.append(torch.bmm(A, q_i.unsqueeze(2)).squeeze(2))
        h_msg = self.norm(torch.stack(msgs, dim=0))   # (cols, batch, H)

        attn_w = None
        if return_weights:
            q_mat  = F.normalize(q.mean(dim=1), dim=-1)
            k_mat  = F.normalize(k.mean(dim=1), dim=-1)
            scores = torch.matmul(q_mat, k_mat.T)
            attn_w = torch.softmax(scores / self.scale, dim=-1).detach()

        return h_msg, A, v2_new, attn_w


class GridRnnPrecDelta(nn.Module):
    """GridRNN with preconditioned delta rule KV memory."""

    def __init__(
        self, *,
        input_size, embedding_size, output_size,
        hidden_size: int,
        n_layers: int, n_columns: int,
        n_attn_heads,         # unused, kept for config compat
        messaging: str = "post",
        col_identities: bool,
        delta_lr: float = 0.01,
        delta_decay: float = 0.99,
        beta2: float = 0.999,
        eps: float = 1e-8,
        use_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size     = input_size
        self.embedding_size = embedding_size
        self.output_size    = output_size
        self.embedding      = nn.Embedding(input_size, embedding_size)
        self.n_layers       = n_layers
        assert n_columns > 1
        self.n_columns  = n_columns
        self.hidden_size = hidden_size

        print(
            f'GridRNN-PrecDelta {n_layers}L x {n_columns}C'
            f' | hidden={hidden_size} delta_lr={delta_lr} decay={delta_decay} beta2={beta2}'
        )

        self.cells      = nn.ModuleList()
        self.pd_layers  = nn.ModuleList()
        self.attn_gates = nn.ModuleList()

        for layer in range(n_layers):
            row = nn.ModuleList([
                nn.GRUCell(
                    input_size  = self._cell_input_dim(layer, ic),
                    hidden_size = hidden_size,
                    bias        = use_bias,
                    dtype       = torch.float64,
                )
                for ic in range(n_columns)
            ])
            self.cells.append(row)
            n_parts = n_columns if col_identities else None
            self.pd_layers.append(PrecDeltaMessagePassing(
                hidden_size    = hidden_size,
                n_columns      = n_columns,
                delta_lr       = delta_lr,
                delta_decay    = delta_decay,
                beta2          = beta2,
                eps            = eps,
                n_participants = n_parts,
            ))
            self.attn_gates.append(nn.Linear(2 * hidden_size, 1))

        self.head = nn.Linear(hidden_size, output_size)
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens, state=None, return_attn: bool = False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2 and tokens.shape[1] == 1
        x = self.embedding(tokens.view(-1))
        h_new, A_new, v2_new, extras = self._grid_step(x, state=state, return_attn=return_attn)
        y = self.head(h_new[-1][0])
        new_state = (h_new, A_new, v2_new)
        if return_attn:
            return y, new_state, extras
        return y, new_state

    def _grid_step(self, x, *, state, return_attn: bool):
        if state is not None:
            h, A, v2 = state
        else:
            bsz = x.shape[0]
            h  = self._init_h(bsz)
            A  = self._init_A(bsz)
            v2 = self._init_v2(bsz)

        h_n, A_n, v2_n, attn_list, gate_list = [], [], [], [], []
        x_list = self._prepare_grid_input(x)

        for i, (cells, pd, gate_lin) in enumerate(zip(self.cells, self.pd_layers, self.attn_gates)):
            hl_n = torch.stack([
                cells[ic](x_list[ic], h[i][ic]) for ic in range(self.n_columns)
            ], dim=0)   # (cols, batch, H)

            msg, Al_new, v2l_new, attn_w = pd(hl_n, A[i], v2[i], return_weights=return_attn)
            g    = torch.sigmoid(gate_lin(torch.cat([hl_n, msg], dim=-1)))  # (cols, batch, 1)
            hl_n = (1 - g) * hl_n + g * msg

            h_n.append(hl_n)
            A_n.append(Al_new)
            v2_n.append(v2l_new)
            attn_list.append(attn_w)
            gate_list.append(g)
            x_list = hl_n

        h_n  = torch.stack(h_n,  dim=0)   # (n_layers, cols, batch, H)
        A_n  = torch.stack(A_n,  dim=0)   # (n_layers, batch, H, H)
        v2_n = torch.stack(v2_n, dim=0)   # (n_layers, batch, H)
        return h_n, A_n, v2_n, {"attn_weights": attn_list, "gates": gate_list}

    def reset_state(self, state, reset_mask):
        if state is None:
            return None
        h, A, v2 = state
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state
        h  = h.clone();  h[:, :, ixs, :]  = 0.0
        A  = A.clone();  A[:, ixs, :, :]  = 0.0
        v2 = v2.clone(); v2[:, ixs, :]    = 0.0
        return (h, A, v2)

    def detach_state(self, state):
        if state is None:
            return None
        h, A, v2 = state
        return (h.detach(), A.detach(), v2.detach())

    def _init_h(self, bsz: int) -> torch.Tensor:
        dev, dt = self.head.weight.device, self.head.weight.dtype
        return torch.zeros(self.n_layers, self.n_columns, bsz, self.hidden_size, device=dev, dtype=dt)

    def _init_A(self, bsz: int) -> torch.Tensor:
        dev, dt = self.head.weight.device, self.head.weight.dtype
        return torch.zeros(self.n_layers, bsz, self.hidden_size, self.hidden_size, device=dev, dtype=dt)

    def _init_v2(self, bsz: int) -> torch.Tensor:
        # init to 1 so preconditioning is neutral at episode start (avoids k/eps blow-up)
        dev, dt = self.head.weight.device, self.head.weight.dtype
        return torch.ones(self.n_layers, bsz, self.hidden_size, device=dev, dtype=dt)

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        return self.hidden_size

    def _prepare_grid_input(self, x):
        bsz   = x.shape[0]
        dummy = torch.zeros(bsz, self._cell_input_dim(0, 1), device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)
