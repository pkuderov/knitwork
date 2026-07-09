from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.grnn_fix_v4 import PerColumnAttention


class HopfieldGridRnnFixV4(nn.Module):
    # Hopfield counterpart of GridRnnFixV4: LSTM cells with dual memory —
    # working h carries the (additively) mixed message recurrently, cell
    # state c stays protected from messages. Per-column attention, input-
    # conditioned gates, forget-gate timescale stagger, aux losses as in v4.
    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size,
            n_layers: int, n_columns: int,
            n_attn_heads,
            beta_scale: float = 3.0,
            timescale_spread: float = 1.0,
            aux_div_weight: float = 0.05,
            aux_gate_weight: float = 0.1,
            aux_act_weight: float = 0.02,
            aux_sat_weight: float = 0.02,
            aux_every: int = 8,
            gate_std_target: float = 0.15,
            sat_target: float = 0.8,
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
        self.aux_sat_weight = aux_sat_weight
        self.aux_every = max(int(aux_every), 1)
        self.gate_std_target = gate_std_target
        self.sat_target = sat_target
        self.use_aux = (aux_div_weight + aux_gate_weight
                        + aux_act_weight + aux_sat_weight) > 0
        self._aux_tick = 0
        if n_layers > 1:
            self._div_layer_w = [0.5 + 1.5 * l / (n_layers - 1) for l in range(n_layers)]
        else:
            self._div_layer_w = [1.0]

        print(
            f'HopfieldGridRnnFixV4 {n_layers}L x {n_columns}C LSTM'
            f' hidden={self.hidden_size} beta_scale={beta_scale}'
            f' ts_spread={timescale_spread} aux={self.use_aux}'
        )

        self.col_input_projs = nn.ModuleList(
            nn.Linear(embedding_size, embedding_size, bias=False)
            for _ in range(n_columns)
        )
        for proj in self.col_input_projs:
            nn.init.orthogonal_(proj.weight)

        H = self.hidden_size
        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = embedding_size if layer == 0 else H
            cells = nn.ModuleList(
                nn.LSTMCell(in_dim, H, bias=use_bias)
                for _ in range(n_columns)
            )
            # multi-timescale prior via forget-gate bias (LSTM layout: i,f,g,o)
            # f -> 1 remembers longer (slow column); f -> 0 forgets fast
            if use_bias and n_columns > 1:
                for ic, cell in enumerate(cells):
                    shift = timescale_spread * (2 * ic / (n_columns - 1) - 1)
                    with torch.no_grad():
                        cell.bias_ih[H:2 * H] += shift
            self.cells.append(cells)

            self.attn.append(PerColumnAttention(
                H, num_heads=n_attn_heads, n_columns=n_columns, beta_scale=beta_scale
            ))
            gates = nn.ModuleList(
                nn.Linear(2 * H + in_dim, 1) for _ in range(n_columns)
            )
            for ic, gate in enumerate(gates):
                nn.init.constant_(gate.bias, -2.5 - 0.5 * ic)
            self.attn_gates.append(gates)

        self.mid_norms = nn.ModuleList(
            nn.RMSNorm(H) for _ in range(max(n_layers - 1, 0))
        )

        self.head = nn.Linear(self.n_columns * H, output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(self, tokens: torch.Tensor, state=None, return_attn=False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2
        x = self.embedding(tokens.view(-1))                      # [B, E]
        h, c = state

        h_n, c_n, attn_list, gate_list = [], [], [], []
        aux_div = aux_gate = aux_act = aux_sat = 0.0
        do_aux = self.use_aux and self.training and (self._aux_tick % self.aux_every == 0)
        if self.use_aux and self.training:
            self._aux_tick += 1
        x = torch.stack([proj(x) for proj in self.col_input_projs], dim=0)  # [C, B, E]

        hl_mix = None
        for layer, (cells, attn, gates, hl, cl) in enumerate(
            zip(self.cells, self.attn, self.attn_gates, h, c)
        ):
            hl_cols, cl_cols = [], []
            for ic in range(self.n_columns):
                h_ic, c_ic = cells[ic](x[ic], (hl[ic], cl[ic]))
                hl_cols.append(h_ic)
                cl_cols.append(c_ic)
            hl_new = torch.stack(hl_cols, dim=0)                 # [C, B, H]
            cl_new = torch.stack(cl_cols, dim=0)

            msg, attn_w = attn(hl_new, return_weights=return_attn)
            gate_in = torch.cat([hl_new, msg, x], dim=-1)        # [C, B, 2H+in]
            g = torch.stack([
                torch.sigmoid(gates[ic](gate_in[ic])) for ic in range(self.n_columns)
            ], dim=0)                                            # [C, B, 1]
            # additive message into working h; memory c stays untouched
            hl_mix = hl_new + g * msg

            if do_aux:
                d, gv, a = self._layer_aux(hl_new, hl, g)
                aux_div = aux_div + self._div_layer_w[layer] * d
                aux_gate, aux_act = aux_gate + gv, aux_act + a
                if layer > 0:
                    aux_sat = aux_sat + F.relu(hl_new.abs().mean() - self.sat_target)

            h_n.append(hl_mix)
            c_n.append(cl_new)
            attn_list.append(attn_w)
            gate_list.append(g)
            x = self.mid_norms[layer](hl_mix) if layer < self.n_layers - 1 else hl_mix

        h_n = torch.stack(h_n, dim=0)
        c_n = torch.stack(c_n, dim=0)

        z = hl_mix.permute(1, 0, 2).reshape(hl_mix.shape[1], -1)  # [B, C*H]
        y = self.head(z)

        aux = None
        if self.use_aux:
            if do_aux:
                aux = self.aux_every * (
                    self.aux_div_weight * aux_div
                    + self.aux_gate_weight * aux_gate
                    + self.aux_act_weight * aux_act
                    + self.aux_sat_weight * aux_sat
                ) / self.n_layers
            else:
                aux = torch.zeros((), device=z.device, dtype=z.dtype)

        extras = {"attn_weights": attn_list, "gates": gate_list}
        if return_attn:
            return (y, (h_n, c_n), extras, aux) if self.use_aux else (y, (h_n, c_n), extras)
        return (y, (h_n, c_n), aux) if self.use_aux else (y, (h_n, c_n))

    def _layer_aux(self, hl_n, hl, g):
        C = self.n_columns
        iu, ju = torch.triu_indices(C, C, offset=1, device=hl_n.device)

        z = hl_n - hl_n.mean(dim=1, keepdim=True)                # [C, B, H]
        z = z / (z.std(dim=1, keepdim=True) + 1e-6)
        B = z.shape[1]
        cross = torch.einsum('cbh,dbk->cdhk', z, z) / B          # [C, C, H, H]
        div = cross[iu, ju].pow(2).mean()

        gm = g.mean(dim=(1, 2))                                  # [C]
        gate = F.relu(self.gate_std_target - gm.std())

        u = (hl_n - hl).norm(dim=-1)                             # [C, B]
        u = u - u.mean(dim=1, keepdim=True)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-6)
        corr = u @ u.T                                           # [C, C]
        act = F.relu(corr[iu, ju]).mean()

        return div, gate, act

    def reset_state(self, state, reset_mask):
        if state is None:
            return self.init_state(reset_mask.shape[0])
        h, c = state
        keep = (~reset_mask.bool()).to(dtype=h.dtype, device=h.device)  # [B]
        keep = keep[None, None, :, None]
        return (h * keep, c * keep)

    def detach_state(self, state):
        if state is None:
            return state
        h, c = state
        return (h.detach(), c.detach())

    def init_state(self, bsz: int):
        shape = (self.n_layers, self.n_columns, bsz, self.hidden_size)
        device = self.head.weight.device
        dtype = self.head.weight.dtype
        return (
            torch.zeros(*shape, device=device, dtype=dtype),
            torch.zeros(*shape, device=device, dtype=dtype),
        )
