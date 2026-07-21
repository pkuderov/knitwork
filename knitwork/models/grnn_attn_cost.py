from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import convert_hidden_size, format_readable_num, to_torch
from knitwork.models.grnn import MessagePassingLayer


class GridRnnAttnCost(nn.Module):
    # Base GRNN (post-messaging, input-conditioned gate) + attention-cost penalty.
    # The mixing gate g in every model collapses to a shared constant (~0.5-0.6);
    # here an aux loss makes attention "expensive": a monotone cost(g) grows with how
    # open the gate is, summed over all gates, so the model keeps attention closed and
    # opens it only where it pays off in CE.
    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size=None, base_hidden_size=None,
            n_layers: int, n_columns: int,
            n_attn_heads, col_identities=True,
            attn_cost_weight: float = 0.02, cost_kind: str = 'linear',
            use_bias=True, dropout=0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, embedding_size)

        self.n_layers = n_layers
        assert n_columns > 1
        self.n_columns = n_columns
        self.n_attn_heads = n_attn_heads
        self.attn_cost_weight = attn_cost_weight
        assert cost_kind in ('linear', 'quad', 'logbarrier')
        self.cost_kind = cost_kind
        self.use_aux = attn_cost_weight > 0

        if hidden_size is not None:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = convert_hidden_size(
                base_hid_dim=base_hidden_size,
                in_dim=embedding_size, out_dim=output_size,
                n_layers=n_layers, n_columns=n_columns, type='grnn'
            )
        self.hidden_size -= self.hidden_size % n_attn_heads
        H = self.hidden_size
        print(
            f'GridRnnAttnCost {n_layers}L x {n_columns}C GRU hidden={H}'
            f' cost={cost_kind} w={attn_cost_weight}'
        )

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        for layer in range(n_layers):
            self.cells.append(nn.ModuleList(
                nn.GRUCell(self._cell_input_dim(layer, ic), H, bias=use_bias)
                for ic in range(n_columns)
            ))
            n_participants = n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(H, num_heads=n_attn_heads, n_participants=n_participants))
            self.attn_gates.append(nn.Linear(2 * H, 1))

        self.head = nn.Linear(H, output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def _cell_input_dim(self, layer: int, ix_col: int) -> int:
        if layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        return self.hidden_size

    def _gate_cost(self, g: torch.Tensor) -> torch.Tensor:
        # g: [C, B, 1] in (0, 1); monotone increasing cost of opening attention
        if self.cost_kind == 'quad':
            c = g.pow(2)
        elif self.cost_kind == 'logbarrier':
            c = -torch.log1p(-g.clamp(max=1 - 1e-4))
        else:
            c = g
        return c.sum(dim=0).mean()   # sum over columns, mean over batch

    def forward(self, tokens: torch.Tensor, h=None, return_attn=False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2 and tokens.shape[1] == 1
        x = self.embedding(tokens.view(-1))              # [B, E]

        h_n, extras, aux = self.grid_step(x, h=h, return_attn=return_attn)
        y = self.head(h_n[-1][0])
        if return_attn:
            return (y, h_n, extras, aux) if self.use_aux else (y, h_n, extras)
        return (y, h_n, aux) if self.use_aux else (y, h_n)

    def grid_step(self, x, *, h, return_attn=False):
        h_n, attn_list, gate_list = [], [], []
        # base grnn input layout: col 0 gets the token embedding, buffers a dummy zero
        bsz = x.shape[0]
        dummy = torch.zeros(bsz, 1, device=x.device, dtype=x.dtype)
        x = [x] + [dummy] * (self.n_columns - 1)

        aux = 0.0
        for cells, attn, attn_gate, hl in zip(self.cells, self.attn, self.attn_gates, h):
            hl_n = torch.stack([
                cells[ic](x[ic], hl[ic]) for ic in range(self.n_columns)
            ], dim=0)                                    # [C, B, H]

            msg, attn_w = attn(hl_n, return_weights=return_attn)
            g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))  # [C, B, 1]
            hl_n = (1 - g) * hl_n + g * msg

            if self.use_aux and self.training:
                aux = aux + self._gate_cost(g)

            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)
            x = hl_n

        h_n = torch.stack(h_n, dim=0)
        extras = {"attn_weights": attn_list, "gates": gate_list}
        if self.use_aux:
            aux = self.attn_cost_weight * (aux / self.n_layers) if self.training \
                else torch.zeros((), device=h_n.device, dtype=h_n.dtype)
        return h_n, extras, aux

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
        return state if state is None else state.detach()

    def init_state(self, bsz):
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype
        )
