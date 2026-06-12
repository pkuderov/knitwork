"""GridRNN with EMA Surprise Momentum Memory.

Writes to fast-weight matrix in proportion to prediction surprise momentum:
high surprise → strong write; low surprise → weak write.
Forgetting is adaptive: proportional to current matrix "fullness".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch


class SurpriseMemoryPassing(nn.Module):
    """Fast-weight KV memory with EMA-based surprise-weighted writes."""

    def __init__(
        self, *,
        hidden_size: int,
        n_columns: int,
        ema_beta: float = 0.9,
        lam_base: float = 0.01,
        n_participants: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_columns   = n_columns
        self.ema_beta    = ema_beta
        self.lam_base    = lam_base
        self.scale       = hidden_size ** 0.5

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

    def forward(self, h, A, m_prev, return_weights: bool = False):
        # h:      (cols, batch, H)
        # A:      (batch, H, H) — fast-weight matrix
        # m_prev: (batch,)      — surprise EMA
        n_cols, batch, H = h.shape
        h_flat = h.reshape(n_cols * batch, H)
        k = self.proj_k(h_flat).view(n_cols, batch, H)
        q = self.proj_q(h_flat).view(n_cols, batch, H)
        v = self.proj_v(h_flat).view(n_cols, batch, H)

        if self.ids is not None:
            k = k + self.ids.unsqueeze(1)
            q = q + self.ids.unsqueeze(1)

        # accumulate delta errors and mean surprise across columns
        surprise_acc = torch.zeros(batch, device=h.device, dtype=h.dtype)
        delta_A      = torch.zeros_like(A)

        for col_j in range(n_cols):
            k_j    = F.normalize(k[col_j], dim=-1)
            v_j    = F.normalize(v[col_j], dim=-1)
            error  = torch.bmm(A, k_j.unsqueeze(2)).squeeze(2) - v_j   # (batch, H)
            surprise_acc += (error ** 2).mean(dim=-1)                   # (batch,)
            delta_A      += torch.bmm(error.unsqueeze(2), k_j.unsqueeze(1))

        surprise = surprise_acc / n_cols   # (batch,) mean across columns

        # EMA of surprise — tracks "how surprising" recent inputs are
        m_new = self.ema_beta * m_prev + (1 - self.ema_beta) * surprise   # (batch,)

        # write strength: normalize to [0, 1] relative to current batch max
        alpha = m_new / (m_new.detach().max() + 1e-6)   # (batch,)

        # adaptive forgetting: stronger when matrix is "fuller"
        # ||A||_F / H as a rough capacity measure
        fullness = A.detach().norm(dim=(-2, -1)) / H    # (batch,)
        lam      = self.lam_base * fullness.clamp(0.0, 1.0)   # (batch,)

        # update: forget proportionally + delta write scaled by surprise
        A_new = (1 - lam.view(batch, 1, 1)) * A \
              - alpha.view(batch, 1, 1) * delta_A

        # content-based retrieval
        msgs = []
        for col_i in range(n_cols):
            q_i = F.normalize(q[col_i], dim=-1)
            msgs.append(torch.bmm(A_new, q_i.unsqueeze(2)).squeeze(2))
        h_msg = self.norm(torch.stack(msgs, dim=0))   # (cols, batch, H)

        attn_w = None
        if return_weights:
            q_mat  = F.normalize(q.mean(dim=1), dim=-1)
            k_mat  = F.normalize(k.mean(dim=1), dim=-1)
            scores = torch.matmul(q_mat, k_mat.T)
            attn_w = torch.softmax(scores / self.scale, dim=-1).detach()

        return h_msg, A_new, m_new, attn_w


class GridRnnEmaMem(nn.Module):
    """GridRNN with EMA surprise-weighted fast-weight memory."""

    def __init__(
        self, *,
        input_size, embedding_size, output_size,
        hidden_size: int,
        n_layers: int, n_columns: int,
        n_attn_heads,         # unused, kept for config compat
        messaging: str = "post",
        col_identities: bool,
        ema_beta: float = 0.9,
        lam_base: float = 0.01,
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
        self.n_columns   = n_columns
        self.hidden_size = hidden_size

        print(
            f'GridRNN-EmaMem {n_layers}L x {n_columns}C'
            f' | hidden={hidden_size} ema_beta={ema_beta} lam_base={lam_base}'
        )

        self.cells      = nn.ModuleList()
        self.mem_layers = nn.ModuleList()
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
            self.mem_layers.append(SurpriseMemoryPassing(
                hidden_size    = hidden_size,
                n_columns      = n_columns,
                ema_beta       = ema_beta,
                lam_base       = lam_base,
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
        h_new, A_new, m_new, extras = self._grid_step(x, state=state, return_attn=return_attn)
        y = self.head(h_new[-1][0])
        new_state = (h_new, A_new, m_new)
        if return_attn:
            return y, new_state, extras
        return y, new_state

    def _grid_step(self, x, *, state, return_attn: bool):
        if state is not None:
            h, A, m = state
        else:
            bsz = x.shape[0]
            h = self._init_h(bsz)
            A = self._init_A(bsz)
            m = self._init_m(bsz)

        h_n, A_n, m_n, attn_list, gate_list = [], [], [], [], []
        x_list = self._prepare_grid_input(x)

        for i, (cells, mem, gate_lin) in enumerate(zip(self.cells, self.mem_layers, self.attn_gates)):
            hl_n = torch.stack([
                cells[ic](x_list[ic], h[i][ic]) for ic in range(self.n_columns)
            ], dim=0)   # (cols, batch, H)

            msg, Al_new, ml_new, attn_w = mem(hl_n, A[i], m[i], return_weights=return_attn)
            g    = torch.sigmoid(gate_lin(torch.cat([hl_n, msg], dim=-1)))  # (cols, batch, 1)
            hl_n = (1 - g) * hl_n + g * msg

            h_n.append(hl_n)
            A_n.append(Al_new)
            m_n.append(ml_new)
            attn_list.append(attn_w)
            gate_list.append(g)
            x_list = hl_n

        h_n = torch.stack(h_n, dim=0)   # (n_layers, cols, batch, H)
        A_n = torch.stack(A_n, dim=0)   # (n_layers, batch, H, H)
        m_n = torch.stack(m_n, dim=0)   # (n_layers, batch)
        return h_n, A_n, m_n, {"attn_weights": attn_list, "gates": gate_list}

    def reset_state(self, state, reset_mask):
        if state is None:
            return None
        h, A, m = state
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state
        h = h.clone(); h[:, :, ixs, :] = 0.0
        A = A.clone(); A[:, ixs, :, :] = 0.0
        m = m.clone(); m[:, ixs]       = 0.0
        return (h, A, m)

    def detach_state(self, state):
        if state is None:
            return None
        h, A, m = state
        return (h.detach(), A.detach(), m.detach())

    def _init_h(self, bsz: int) -> torch.Tensor:
        dev, dt = self.head.weight.device, self.head.weight.dtype
        return torch.zeros(self.n_layers, self.n_columns, bsz, self.hidden_size, device=dev, dtype=dt)

    def _init_A(self, bsz: int) -> torch.Tensor:
        dev, dt = self.head.weight.device, self.head.weight.dtype
        return torch.zeros(self.n_layers, bsz, self.hidden_size, self.hidden_size, device=dev, dtype=dt)

    def _init_m(self, bsz: int) -> torch.Tensor:
        dev, dt = self.head.weight.device, self.head.weight.dtype
        return torch.zeros(self.n_layers, bsz, device=dev, dtype=dt)

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        return self.hidden_size

    def _prepare_grid_input(self, x):
        bsz   = x.shape[0]
        dummy = torch.zeros(bsz, self._cell_input_dim(0, 1), device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)
