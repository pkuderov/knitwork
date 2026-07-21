from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import convert_hidden_size, format_readable_num, to_torch
from knitwork.models.grnn import MessagePassingLayer


class GridRnnFeedback(nn.Module):
    # Base GRNN + top-down feedback: every buffer column (c >= 1) receives, as its
    # layer-0 input, its own top-layer output from the previous step (projected back
    # to embedding_size) plus a learnable per-column seed. The seed breaks symmetry so
    # buffer columns specialise differently; at reset (prev state = 0) the input is
    # exactly the seed. Column 0 keeps the external token embedding. Post-messaging only.
    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size=None, base_hidden_size=None,
            n_layers: int, n_columns: int,
            n_attn_heads, col_identities=True,
            seed_scale: float = 1.0, fb_init: float = 0.1,
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
        print(f'GridRnnFeedback {n_layers}L x {n_columns}C GRU hidden={H}')

        # per-buffer-column top-down feedback projection H -> E (small init: seeds
        # dominate early, feedback grows in as columns specialise)
        self.fb_proj = nn.ModuleList(
            nn.Linear(H, embedding_size, bias=False) for _ in range(n_columns - 1)
        )
        for proj in self.fb_proj:
            nn.init.normal_(proj.weight, 0.0, fb_init * (1 / H) ** 0.5)
        # distinct initial input value per buffer column -> different specialisation
        self.col_seeds = nn.Parameter(torch.empty(n_columns - 1, embedding_size))
        nn.init.orthogonal_(self.col_seeds)
        self.col_seeds.data *= seed_scale

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = embedding_size if layer == 0 else H
            self.cells.append(nn.ModuleList(
                nn.GRUCell(in_dim, H, bias=use_bias) for _ in range(n_columns)
            ))
            n_participants = n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(H, num_heads=n_attn_heads, n_participants=n_participants))
            self.attn_gates.append(nn.Linear(2 * H, 1))

        self.head = nn.Linear(H, output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, h=None, return_attn=False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2 and tokens.shape[1] == 1
        x = self.embedding(tokens.view(-1))              # [B, E]

        h_n, extras = self.grid_step(x, h=h, return_attn=return_attn)
        z = h_n[-1][0]                                    # top layer, external column
        y = self.head(z)
        if return_attn:
            return y, h_n, extras
        return y, h_n

    def grid_step(self, x, *, h, return_attn=False):
        h_n, attn_list, gate_list = [], [], []
        # layer-0 inputs: [token emb] + [seed_c + feedback from prev top layer]
        prev_top = h[-1]                                 # [C, B, H]
        xl = [x]
        for c in range(1, self.n_columns):
            fb = self.fb_proj[c - 1](prev_top[c])        # [B, E]
            xl.append(self.drop(self.col_seeds[c - 1] + fb))
        x = xl

        for cells, attn, attn_gate, hl in zip(self.cells, self.attn, self.attn_gates, h):
            hl_n = torch.stack([
                cells[ic](x[ic], hl[ic]) for ic in range(self.n_columns)
            ], dim=0)                                    # [C, B, H]

            msg, attn_w = attn(hl_n, return_weights=return_attn)
            g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
            hl_n = (1 - g) * hl_n + g * msg

            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)
            x = hl_n

        h_n = torch.stack(h_n, dim=0)
        return h_n, {"attn_weights": attn_list, "gates": gate_list}

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
