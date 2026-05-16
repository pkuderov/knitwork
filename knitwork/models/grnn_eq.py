from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.grnn import MessagePassingLayer


class EquilibriumCell(nn.Module):
    """GRUCell that iterates to approximate fixed point: h* = GRU(x, h*)."""

    def __init__(self, input_size: int, hidden_size: int, use_bias: bool = True,
                 max_iters: int = 8, tol: float = 1e-3):
        super().__init__()
        self.cell      = nn.GRUCell(input_size, hidden_size, bias=use_bias)
        self.max_iters = max_iters
        self.tol       = tol

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor,
        halt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = h_prev
        n_iters = torch.zeros(x.shape[0], device=x.device, dtype=torch.int32)
        for _ in range(self.max_iters):
            h_new     = self.cell(x, h)
            converged = (h_new - h).norm(dim=-1) < self.tol
            not_done  = ~converged
            if halt_mask is not None:
                not_done = not_done & ~halt_mask
            h       = torch.where(not_done.unsqueeze(-1), h_new, h)
            n_iters += not_done.int()
            if not not_done.any():
                break
        return h, n_iters


class HaltingUnit(nn.Module):
    """Sigmoid halting probability from hidden state."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, -2.0)  # start with low halt prob

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(h).squeeze(-1))


class ChainOfThoughtGRU(nn.Module):
    """Accumulates a separate thought vector across time steps."""

    def __init__(self, hidden_size: int, thought_size: int):
        super().__init__()
        self.cell = nn.GRUCell(hidden_size, thought_size)
        self.norm = nn.LayerNorm(thought_size)

    def forward(self, h_top: torch.Tensor, thought: torch.Tensor) -> torch.Tensor:
        return self.norm(self.cell(h_top, thought))


class EquilibriumGridRnnCoT(nn.Module):
    """GridRNN with Adaptive Computation Time (ACT) + Chain-of-Thought buffer."""

    def __init__(
        self, *,
        input_size: int, embedding_size: int, output_size: int,
        hidden_size: int, n_layers: int, n_columns: int, n_attn_heads: int,
        col_identities: bool,
        thought_size: int | None = None,
        use_bias: bool = True, dropout: float = 0.0,
        max_eq_iters: int = 8, eq_tol: float = 1e-3,
        act_eps: float = 0.01, act_loss_weight: float = 1e-3,
    ):
        super().__init__()
        assert n_columns > 1
        self.input_size      = input_size
        self.embedding_size  = embedding_size
        self.output_size     = output_size
        self.n_layers        = n_layers
        self.n_columns       = n_columns
        self.n_attn_heads    = n_attn_heads
        self.act_eps         = act_eps
        self.act_loss_weight = act_loss_weight
        self.max_eq_iters    = max_eq_iters

        self.embedding   = nn.Embedding(input_size, embedding_size)
        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        self.thought_size = thought_size or self.hidden_size
        print(
            f'EqGridRNN-CoT | {n_layers}L x {n_columns}C'
            f' | hidden={self.hidden_size} | thought={self.thought_size}'
            f' | max_eq_iters={max_eq_iters}'
        )

        self.cells   = nn.ModuleList()
        self.attn    = nn.ModuleList()
        self.gates   = nn.ModuleList()
        self.halters = nn.ModuleList()
        for layer in range(n_layers):
            self.cells.append(nn.ModuleList([
                EquilibriumCell(
                    input_size=self._cell_input_dim(layer, ic),
                    hidden_size=self.hidden_size, use_bias=use_bias,
                    max_iters=max_eq_iters, tol=eq_tol,
                )
                for ic in range(n_columns)
            ]))
            n_part = n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(self.hidden_size, n_attn_heads, n_part))
            self.gates.append(nn.Linear(2 * self.hidden_size, 1))
            self.halters.append(HaltingUnit(self.hidden_size))

        self.cot  = ChainOfThoughtGRU(self.hidden_size, self.thought_size)
        self.head = nn.Linear(self.hidden_size + self.thought_size, output_size)
        print(f'Param count: {format_readable_num(sum(p.numel() for p in self.parameters() if p.requires_grad))}')

    def forward(self, tokens: torch.Tensor, state=None, return_attn: bool = False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2 and tokens.shape[1] == 1
        bsz = tokens.shape[0]

        if state is None:
            h, thought = self.init_hidden(bsz), self.init_thought(bsz)
        else:
            h, thought = state

        x = self.embedding(tokens.view(-1))  # [batch, emb]
        h_new, act_iters, attn_info, gate_info = self._grid_step(x, h, return_gates=return_attn)

        h_top   = h_new[-1, 0]               # [batch, hidden]
        thought = self.cot(h_top, thought)
        y       = self.head(torch.cat([h_top, thought], dim=-1))

        new_state = (h_new, thought)
        if return_attn:
            return y, new_state, {"attn_weights": attn_info, "gates": gate_info, "act_iters": act_iters}
        return y, new_state

    def act_loss(self, act_iters_list) -> torch.Tensor:
        return self.act_loss_weight * sum(it.float().mean() for it in act_iters_list)

    def _grid_step(self, x: torch.Tensor, h: torch.Tensor, return_gates: bool = False):
        bsz, device, dtype = x.shape[0], x.device, x.dtype
        x_in     = self._prepare_grid_input(x, bsz, device, dtype)
        h_n:       List[torch.Tensor] = []
        act_iters: List[torch.Tensor] = []
        attn_info: List               = []
        gate_info: List               = []

        for layer_idx, (cells_row, attn_mod, gate, halter) in enumerate(
            zip(self.cells, self.attn, self.gates, self.halters)
        ):
            hl = h[layer_idx]  # [cols, batch, hidden]

            # ACT accumulation state
            halt_acc  = torch.zeros(bsz, device=device, dtype=dtype)
            halt_mask = torch.zeros(bsz, device=device, dtype=torch.bool)
            h_acc     = torch.zeros_like(hl)
            n_iters   = torch.zeros(bsz, device=device, dtype=torch.int32)
            h_layer   = hl

            for act_step in range(self.max_eq_iters):
                h_layer_new = torch.stack(
                    [cells_row[ic].cell(x_in[ic], h_layer[ic]) for ic in range(self.n_columns)],
                    dim=0,
                )  # [cols, batch, hidden]

                p_halt  = halter(h_layer_new.mean(dim=0))  # [batch,]
                is_last = (act_step == self.max_eq_iters - 1)
                p_use   = torch.where(
                    is_last | (halt_acc + p_halt >= 1.0 - self.act_eps),
                    1.0 - halt_acc, p_halt,
                )

                # intermediate steps detached — only final carries gradient
                w      = p_use.unsqueeze(0).unsqueeze(-1)  # [1, batch, 1]
                h_acc += w * (h_layer_new if is_last else h_layer_new.detach())

                halt_acc  = halt_acc + p_use
                n_iters  += (~halt_mask).int()
                halt_mask = halt_mask | (halt_acc >= 1.0 - self.act_eps)
                h_layer   = h_layer_new
                if halt_mask.all():
                    break

            hl_n = h_acc  # [cols, batch, hidden]

            msg, attn_w = attn_mod(hl_n, return_weights=return_gates)
            g_raw = gate(torch.cat([hl_n, msg], dim=-1))
            g     = torch.sigmoid(g_raw)
            hl_n  = (1.0 - g) * hl_n + g * msg

            h_n.append(hl_n)
            act_iters.append(n_iters)
            attn_info.append(attn_w)
            if return_gates:
                gate_info.append(g_raw)
            x_in = hl_n

        return torch.stack(h_n, dim=0), act_iters, attn_info, gate_info

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        return self.hidden_size

    def _prepare_grid_input(self, x: torch.Tensor, bsz: int,
                            device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        dummy = torch.zeros(bsz, 1, device=device, dtype=dtype)
        return [x] + [dummy] * (self.n_columns - 1)

    def init_hidden(self, bsz: int) -> torch.Tensor:
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype,
        )

    def init_thought(self, bsz: int) -> torch.Tensor:
        return torch.zeros(
            bsz, self.thought_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype,
        )

    def reset_state(self, state, reset_mask):
        if state is None:
            bsz = reset_mask.shape[0]
            return self.init_hidden(bsz), self.init_thought(bsz)
        h, thought = state
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return h, thought
        h       = h.clone();       h[:, :, ixs, :] = 0.0
        thought = thought.clone(); thought[ixs, :]  = 0.0
        return h, thought

    def detach_state(self, state):
        if state is None:
            return None
        h, thought = state
        return h.detach(), thought.detach()
