from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.grnn_fix import ColumnAttention


class GridRnnFixV3(nn.Module):
    # GridRnnFix v3: sharper beta init, RMSNorm between layers, per-column gates
    # with staggered init, and column-specialization aux losses (diversity,
    # gate variance, activity decorrelation). See architecture_analysis.md §7.1.
    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size,
            n_layers: int, n_columns: int,
            n_attn_heads,
            beta_scale: float = 3.0,
            aux_div_weight: float = 0.05,
            aux_gate_weight: float = 0.02,
            aux_act_weight: float = 0.02,
            aux_every: int = 8,
            gate_std_target: float = 0.15,
            use_bias = True, dropout = 0.0
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
        self.hidden_size = hidden_size - hidden_size % n_attn_heads

        self.aux_div_weight = aux_div_weight
        self.aux_gate_weight = aux_gate_weight
        self.aux_act_weight = aux_act_weight
        self.aux_every = max(int(aux_every), 1)
        self.gate_std_target = gate_std_target
        self.use_aux = (aux_div_weight + aux_gate_weight + aux_act_weight) > 0
        self._aux_tick = 0

        print(
            f'GridRnnFixV3 {n_layers}L x {n_columns}C GRU'
            f' hidden={self.hidden_size} beta_scale={beta_scale} aux={self.use_aux}'
        )

        self.col_input_projs = nn.ModuleList(
            nn.Linear(embedding_size, embedding_size, bias=False)
            for _ in range(n_columns)
        )
        for proj in self.col_input_projs:
            nn.init.orthogonal_(proj.weight)

        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = embedding_size if layer == 0 else self.hidden_size
            self.cells.append(nn.ModuleList(
                nn.GRUCell(in_dim, self.hidden_size, bias=use_bias)
                for _ in range(n_columns)
            ))
            self.attn.append(ColumnAttention(
                self.hidden_size, num_heads=n_attn_heads, beta_scale=beta_scale
            ))
            # per-column gates, staggered init to break gate symmetry
            gates = nn.ModuleList(
                nn.Linear(2 * self.hidden_size, 1) for _ in range(n_columns)
            )
            for ic, gate in enumerate(gates):
                nn.init.constant_(gate.bias, -2.5 - 0.5 * ic)
            self.attn_gates.append(gates)

        # keep upper-layer input in a healthy range (anti tanh-saturation)
        self.mid_norms = nn.ModuleList(
            nn.RMSNorm(self.hidden_size) for _ in range(max(n_layers - 1, 0))
        )

        self.head = nn.Linear(self.n_columns * self.hidden_size, output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, h=None, return_attn=False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2
        x = self.embedding(tokens.view(-1))                      # [B, E]

        h_n, o_top, extras, aux = self.grid_step(x, h=h, return_attn=return_attn)

        y = self.head(o_top)
        if return_attn:
            return (y, h_n, extras, aux) if self.use_aux else (y, h_n, extras)
        return (y, h_n, aux) if self.use_aux else (y, h_n)

    def grid_step(self, x, *, h, return_attn=False):
        h_n, attn_list, gate_list = [], [], []
        aux_div = aux_gate = aux_act = 0.0
        # aux is computed once per aux_every calls (~once per optimizer step)
        do_aux = self.use_aux and self.training and (self._aux_tick % self.aux_every == 0)
        if self.use_aux and self.training:
            self._aux_tick += 1
        x = torch.stack([proj(x) for proj in self.col_input_projs], dim=0)  # [C, B, E]

        o = None
        for layer, (cells, attn, gates, hl) in enumerate(
            zip(self.cells, self.attn, self.attn_gates, h)
        ):
            hl_n = torch.stack([
                cells[ic](x[ic], hl[ic]) for ic in range(self.n_columns)
            ], dim=0)                                            # [C, B, H]

            msg, attn_w = attn(hl_n, return_weights=return_attn)
            gate_in = torch.cat([hl_n, msg], dim=-1)             # [C, B, 2H]
            g = torch.stack([
                torch.sigmoid(gates[ic](gate_in[ic])) for ic in range(self.n_columns)
            ], dim=0)                                            # [C, B, 1]
            o = hl_n + g * msg

            if do_aux:
                d, gv, a = self._layer_aux(hl_n, hl, g)
                aux_div, aux_gate, aux_act = aux_div + d, aux_gate + gv, aux_act + a

            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)
            x = self.mid_norms[layer](o) if layer < self.n_layers - 1 else o

        h_n = torch.stack(h_n, dim=0)
        o_top = o.permute(1, 0, 2).reshape(o.shape[1], -1)       # [B, C*H]

        aux = None
        if self.use_aux:
            if do_aux:
                # scale by aux_every: keeps effective weight constant after
                # averaging over the rollout window in the runner
                aux = self.aux_every * (
                    self.aux_div_weight * aux_div
                    + self.aux_gate_weight * aux_gate
                    + self.aux_act_weight * aux_act
                ) / self.n_layers
            else:
                aux = torch.zeros((), device=o.device, dtype=o.dtype)
        extras = {"attn_weights": attn_list, "gates": gate_list}
        return h_n, o_top, extras, aux

    def _layer_aux(self, hl_n, hl, g):
        C = self.n_columns
        iu, ju = torch.triu_indices(C, C, offset=1, device=hl_n.device)

        # (1) feature decorrelation (Barlow-style): kills CKA-like redundancy
        # between columns, which mean-state cosine cannot see
        z = hl_n - hl_n.mean(dim=1, keepdim=True)                # [C, B, H]
        z = z / (z.std(dim=1, keepdim=True) + 1e-6)
        B = z.shape[1]
        cross = torch.einsum('cbh,dbk->cdhk', z, z) / B          # [C, C, H, H]
        div = cross[iu, ju].pow(2).mean()

        # (2) gate diversity: push per-column mean gates apart
        gm = g.mean(dim=(1, 2))                                  # [C]
        gate = F.relu(self.gate_std_target - gm.std())

        # (3) activity decorrelation: columns should update at different times
        u = (hl_n - hl).norm(dim=-1)                             # [C, B]
        u = u - u.mean(dim=1, keepdim=True)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-6)
        corr = u @ u.T                                           # [C, C]
        act = F.relu(corr[iu, ju]).mean()

        return div, gate, act

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
        if state is None:
            return state
        return state.detach()

    def init_state(self, bsz):
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype
        )
