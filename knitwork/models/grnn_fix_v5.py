from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.common.utils import format_readable_num, to_torch


class FastFloorLRUCell(nn.Module):
    # slim LRU for storage columns: merged B projection (1 matmul), no D
    # feedthrough, GLU output nonlinearity, guaranteed retention floor,
    # damped nu/theta gradients (double-exp reparam blows |grad| otherwise)
    def __init__(self, input_size, hidden_size, *, r_floor: float = 0.9,
                 grad_scale: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        r_free = torch.rand(hidden_size) * 0.9 + 0.05
        self.nu = nn.Parameter(torch.log(-torch.log(r_free)))
        self.theta = nn.Parameter(torch.log(2 * math.pi * torch.rand(hidden_size) + 1e-8))
        self.register_buffer('r_floor', torch.tensor(float(r_floor)))
        if grad_scale and grad_scale != 1.0:
            self.nu.register_hook(lambda g: g * grad_scale)
            self.theta.register_hook(lambda g: g * grad_scale)

        self.B = nn.Linear(input_size, 2 * hidden_size, bias=False)
        self.C = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.G = nn.Linear(2 * hidden_size, hidden_size)   # GLU gate
        nn.init.xavier_normal_(self.B.weight)
        nn.init.xavier_normal_(self.C.weight)

    def _lambda_gamma(self):
        r_free = torch.exp(-torch.exp(self.nu))              # (0, 1)
        r = self.r_floor + (1.0 - self.r_floor) * r_free     # >= r_floor
        phi = torch.exp(self.theta)
        lam_re = r * torch.cos(phi)
        lam_im = r * torch.sin(phi)
        gamma = torch.sqrt(torch.clamp(1.0 - r * r, min=1e-6))
        return lam_re, lam_im, gamma

    def forward(self, u, h):
        # u: [B, in], h: [B, 2H]
        H = self.hidden_size
        h_re, h_im = h[:, :H], h[:, H:]
        lam_re, lam_im, gamma = self._lambda_gamma()
        bu = self.B(u)
        new_re = lam_re * h_re - lam_im * h_im + gamma * bu[:, :H]
        new_im = lam_re * h_im + lam_im * h_re + gamma * bu[:, H:]
        h_n = torch.cat([new_re, new_im], dim=-1)            # [B, 2H]
        # GLU: per-step nonlinearity the linear recurrence lacks
        y = self.C(h_n) * torch.sigmoid(self.G(h_n))         # [B, H]
        return y, h_n


class GridRnnFixV5(nn.Module):
    # v5.1 hybrid: fast GRU columns (nonlinear binding) + slow LRU storage
    # columns with retention floors + frozen reservoir hub. Lessons from v5.0:
    # pure linear columns cannot bind (SDQ Acc++ 0.40), nu/theta gradients
    # explode (|grad| 8), 5 small matmuls per LRU cell are slow.
    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size,
            n_layers: int, n_columns: int,
            n_attn_heads,
            n_lru_cols: int = 1,
            beta_scale: float = 3.0,
            timescale_spread: float = 1.0,
            r_floor_min: float = 0.7,
            r_floor_max: float = 0.95,
            lru_grad_scale: float = 0.1,
            reservoir_mult: int = 2,
            spectral_radius: float = 0.95,
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
        assert 0 <= n_lru_cols < n_columns
        self.n_columns = n_columns
        self.n_lru = n_lru_cols
        self.n_gru = n_columns - n_lru_cols
        self.n_attn_heads = n_attn_heads
        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        H = self.hidden_size
        self.res_size = reservoir_mult * H

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
            f'GridRnnFixV5 {n_layers}L x {n_columns}C hybrid'
            f' ({self.n_gru} GRU + {self.n_lru} LRU) hidden={H}'
            f' res={self.res_size} floors=[{r_floor_min},{r_floor_max}]'
        )

        self.col_input_projs = nn.ModuleList(
            nn.Linear(embedding_size, embedding_size, bias=False)
            for _ in range(n_columns)
        )
        for proj in self.col_input_projs:
            nn.init.orthogonal_(proj.weight)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ParameterList()
        self.res_in = nn.ModuleList()
        self.res_hh = nn.ModuleList()
        self.res_read = nn.ModuleList()
        for layer in range(n_layers):
            in_dim = embedding_size if layer == 0 else H
            cells = nn.ModuleList()
            for ic in range(n_columns):
                if ic < self.n_gru:
                    # fast/medium GRU columns: multiplicative binding
                    cell = nn.GRUCell(in_dim, H, bias=use_bias)
                    if use_bias and self.n_gru > 1:
                        shift = timescale_spread * (2 * ic / max(self.n_gru - 1, 1) - 1)
                        with torch.no_grad():
                            cell.bias_ih[H:2 * H] += shift
                else:
                    # slow LRU storage columns with retention floors
                    il = ic - self.n_gru
                    floor = (r_floor_min
                             + (r_floor_max - r_floor_min) * il / max(self.n_lru - 1, 1))
                    cell = FastFloorLRUCell(in_dim, H, r_floor=floor,
                                            grad_scale=lru_grad_scale)
                cells.append(cell)
            self.cells.append(cells)

            self.attn.append(HubColumnAttention(
                H, num_heads=n_attn_heads,
                n_receivers=n_columns, n_sources=n_columns + 1,
                beta_scale=beta_scale,
            ))
            self.attn_gates.append(nn.Parameter(
                torch.tensor([-2.5 - 0.5 * ic for ic in range(n_columns)])
            ))
            w_in = nn.Linear(in_dim, self.res_size, bias=False)
            w_hh = nn.Linear(self.res_size, self.res_size, bias=False)
            with torch.no_grad():
                eig = torch.linalg.eigvals(w_hh.weight).abs().max()
                w_hh.weight *= spectral_radius / eig.clamp(min=1e-6)
            w_in.weight.requires_grad_(False)
            w_hh.weight.requires_grad_(False)
            self.res_in.append(w_in)
            self.res_hh.append(w_hh)
            self.res_read.append(nn.Linear(self.res_size, H))

        self.mid_norms = nn.ModuleList(
            nn.RMSNorm(H) for _ in range(max(n_layers - 1, 0))
        )
        self.head = nn.Linear(self.n_columns * H, output_size)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f'Param count: {format_readable_num(n_train)}'
              f' (+{format_readable_num(n_total - n_train)} frozen)')

    def forward(self, tokens: torch.Tensor, state=None, return_attn=False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2
        x = self.embedding(tokens.view(-1))                      # [B, E]
        h, r = state                                             # [L,C,B,2H], [L,B,R]
        H = self.hidden_size

        h_n, r_n, attn_list, gate_list = [], [], [], []
        aux_div = aux_gate = aux_act = aux_sat = 0.0
        do_aux = self.use_aux and self.training and (self._aux_tick % self.aux_every == 0)
        if self.use_aux and self.training:
            self._aux_tick += 1
        x = self.drop(torch.stack(
            [proj(x) for proj in self.col_input_projs], dim=0))  # [C, B, E]

        o = None
        for layer, (cells, attn, gates, hl, rl) in enumerate(
            zip(self.cells, self.attn, self.attn_gates, h, r)
        ):
            y_cols, h_cols = [], []
            for ic in range(self.n_columns):
                if ic < self.n_gru:
                    y_ic = cells[ic](x[ic], hl[ic][:, :H])       # GRU: y == state
                    h_ic = F.pad(y_ic, (0, H))
                else:
                    y_ic, h_ic = cells[ic](x[ic], hl[ic])        # LRU: [B,2H] state
                y_cols.append(y_ic)
                h_cols.append(h_ic)
            y = torch.stack(y_cols, dim=0)                       # [C, B, H]
            hl_new = torch.stack(h_cols, dim=0)                  # [C, B, 2H]

            rl_new = torch.tanh(self.res_in[layer](x.mean(dim=0)) + self.res_hh[layer](rl))
            hub = self.res_read[layer](rl_new)                   # [B, H]

            sources = torch.cat([y, hub.unsqueeze(0)], dim=0)    # [C+1, B, H]
            msg, attn_w = attn(y, sources, return_weights=return_attn)
            g = torch.sigmoid(gates).view(self.n_columns, 1, 1)  # [C, 1, 1]
            o = y + g * msg

            if do_aux:
                d, gv, a = self._layer_aux(y, hl_new, hl, g)
                aux_div = aux_div + self._div_layer_w[layer] * d
                aux_gate, aux_act = aux_gate + gv, aux_act + a
                if layer > 0 and self.n_gru > 0:
                    # tanh saturation concerns only the GRU columns
                    aux_sat = aux_sat + F.relu(y[:self.n_gru].abs().mean() - self.sat_target)

            h_n.append(hl_new)
            r_n.append(rl_new)
            attn_list.append(attn_w)
            gate_list.append(g)
            x = self.drop(self.mid_norms[layer](o)) if layer < self.n_layers - 1 else o

        h_n = torch.stack(h_n, dim=0)
        r_n = torch.stack(r_n, dim=0)

        o_top = o.permute(1, 0, 2).reshape(o.shape[1], -1)       # [B, C*H]
        yhat = self.head(o_top)

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
                aux = torch.zeros((), device=o.device, dtype=o.dtype)

        extras = {"attn_weights": attn_list, "gates": gate_list}
        state = (h_n, r_n)
        if return_attn:
            return (yhat, state, extras, aux) if self.use_aux else (yhat, state, extras)
        return (yhat, state, aux) if self.use_aux else (yhat, state)

    def _layer_aux(self, y, hl_new, hl, g):
        C = self.n_columns
        iu, ju = torch.triu_indices(C, C, offset=1, device=y.device)

        z = y - y.mean(dim=1, keepdim=True)                      # [C, B, H]
        z = z / (z.std(dim=1, keepdim=True) + 1e-6)
        B = z.shape[1]
        cross = torch.einsum('cbh,dbk->cdhk', z, z) / B          # [C, C, H, H]
        div = cross[iu, ju].pow(2).mean()

        gm = g.mean(dim=(1, 2))                                  # [C]
        gate = F.relu(self.gate_std_target - gm.std())

        u = (hl_new - hl).norm(dim=-1)                           # [C, B]
        u = u - u.mean(dim=1, keepdim=True)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-6)
        corr = u @ u.T                                           # [C, C]
        act = F.relu(corr[iu, ju]).mean()

        return div, gate, act

    def reset_state(self, state, reset_mask):
        if state is None:
            return self.init_state(reset_mask.shape[0])
        h, r = state
        keep = (~reset_mask.bool()).to(dtype=h.dtype, device=h.device)  # [B]
        return (h * keep[None, None, :, None], r * keep[None, :, None])

    def detach_state(self, state):
        if state is None:
            return state
        h, r = state
        return (h.detach(), r.detach())

    def init_state(self, bsz: int):
        device = self.head.weight.device
        dtype = self.head.weight.dtype
        return (
            torch.zeros(self.n_layers, self.n_columns, bsz, 2 * self.hidden_size,
                        device=device, dtype=dtype),
            torch.zeros(self.n_layers, bsz, self.res_size, device=device, dtype=dtype),
        )


class HubColumnAttention(nn.Module):
    # per-column attention with an extra frozen-hub source:
    # queries = C columns, keys/values = C columns + hub
    def __init__(self, dim, num_heads, n_receivers, n_sources, beta_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        xavier_alpha = (1 / dim) ** 0.5
        self.ids_q = nn.Parameter(torch.empty(n_receivers, 1, dim))
        self.ids_k = nn.Parameter(torch.empty(n_sources, 1, dim))
        nn.init.normal_(self.ids_q, 0.0, 0.1 * xavier_alpha)
        nn.init.normal_(self.ids_k, 0.0, 0.1 * xavier_alpha)

        base = math.log(beta_scale / math.sqrt(self.head_dim))
        spread = torch.linspace(math.log(0.5), math.log(2.0), n_receivers)
        self.log_beta = nn.Parameter(base + spread[:, None].repeat(1, num_heads))

        nn.init.normal_(self.out_proj.weight, 0.0, 0.001)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h_recv, h_all, return_weights: bool = False):
        # h_recv: [C, B, D], h_all: [S, B, D]
        C, B, D = h_recv.shape
        S = h_all.shape[0]
        q = self.W_q(h_recv + self.ids_q).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        k = self.W_k(h_all + self.ids_k).view(S, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        v = self.W_v(h_all).view(S, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)

        beta = self.log_beta.exp().T.unsqueeze(1).unsqueeze(-1)  # [heads, 1, C, 1]
        attn = torch.softmax(beta * torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        out = torch.matmul(attn, v)                              # [heads, B, C, hd]
        out = out.permute(2, 1, 0, 3).contiguous().view(C, B, D)
        attn_w = attn.mean(dim=(0, 1)) if return_weights else None
        return self.out_proj(out), attn_w
