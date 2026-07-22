from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.common.utils import format_readable_num, to_torch


class GridRnnFixV4(nn.Module):
    # v4: per-column attention (query/key identities + per-column beta),
    # input-conditioned gates, multi-timescale column init. Task-agnostic:
    # no benchmark-specific signals. Carries over all v3 fixes and aux losses.
    # beta_floor guards against long-run attention flattening; aux_div_max_weight
    # + aux_div_ramp_steps target the single worst-colliding column pair instead
    # of only the uniform average.
    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size,
            n_layers: int, n_columns: int,
            n_attn_heads,
            beta_scale: float = 3.0,
            beta_floor: float = 0.0,
            timescale_spread: float = 1.0,
            aux_div_weight: float = 0.05,
            aux_div_max_weight: float = 0.1,
            aux_div_ramp_steps: int = 0,
            aux_gate_weight: float = 0.1,
            aux_act_weight: float = 0.02,
            aux_sat_weight: float = 0.02,
            aux_every: int = 8,
            gate_std_target: float = 0.15,
            sat_target: float = 0.8,
            optim: bool = False,
            div_latent_dim: int = 32,
            aux_batch_frac: float = 0.25,
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
        self.aux_div_max_weight = aux_div_max_weight
        self.aux_div_ramp_steps = max(int(aux_div_ramp_steps), 0)
        self.aux_gate_weight = aux_gate_weight
        self.aux_act_weight = aux_act_weight
        self.aux_sat_weight = aux_sat_weight
        self.aux_every = max(int(aux_every), 1)
        self.gate_std_target = gate_std_target
        self.sat_target = sat_target
        self.use_aux = (aux_div_weight + aux_div_max_weight + aux_gate_weight
                        + aux_act_weight + aux_sat_weight) > 0
        self._aux_tick = torch.zeros((1,), dtype=torch.int64)

        # --optim: single switch for all memory optimizations (see docs/grnn_fix_v4.md).
        # Backward-compatible: default False keeps the original behavior/weights.
        self.optim = optim
        self.grad_checkpoint = optim          # honored by run scripts (per-step ckpt)
        self.div_latent_dim = div_latent_dim
        self.aux_batch_frac = aux_batch_frac
        if optim:
            # optimized variant drops the activity-decorrelation aux loss
            self.aux_act_weight = 0.0
        # optional per-column attention mask [C, C] bool (True=allowed); inference-only
        # ablation, does not affect training when None
        self.attn_col_mask = None
        # Barlow weight grows with depth: top-layer redundancy is the failure mode
        if n_layers > 1:
            self._div_layer_w = [0.5 + 1.5 * l / (n_layers - 1) for l in range(n_layers)]
        else:
            self._div_layer_w = [1.0]

        print(
            f'GridRnnFixV4 {n_layers}L x {n_columns}C GRU'
            f' hidden={self.hidden_size} beta_scale={beta_scale}'
            f' ts_spread={timescale_spread} aux={self.use_aux} optim={self.optim}'
        )

        self.col_input_projs = nn.ModuleList(
            nn.Linear(embedding_size, embedding_size, bias=False)
            for _ in range(n_columns)
        )
        for proj in self.col_input_projs:
            nn.init.orthogonal_(proj.weight)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        H = self.hidden_size
        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ParameterList()
        for layer in range(n_layers):
            in_dim = embedding_size if layer == 0 else H
            cells = nn.ModuleList(
                nn.GRUCell(in_dim, H, bias=use_bias)
                for _ in range(n_columns)
            )
            # multi-timescale prior: stagger update-gate bias across columns
            # (z -> 1 keeps old state = slow column; z -> 0 rewrites = fast)
            if use_bias and n_columns > 1:
                for ic, cell in enumerate(cells):
                    shift = timescale_spread * (2 * ic / (n_columns - 1) - 1)
                    with torch.no_grad():
                        cell.bias_ih[H:2 * H] += shift
            self.cells.append(cells)

            self.attn.append(PerColumnAttention(
                H, num_heads=n_attn_heads, n_columns=n_columns, beta_scale=beta_scale,
                beta_floor=beta_floor,
            ))
            # gates degenerated to constants in v3/v4/hgrnn-v4 runs: use an honest
            # learnable scalar per column (staggered init), params go back to H
            self.attn_gates.append(nn.Parameter(
                torch.tensor([-2.5 - 0.5 * ic for ic in range(n_columns)])
            ))

        self.mid_norms = nn.ModuleList(
            nn.RMSNorm(H) for _ in range(max(n_layers - 1, 0))
        )

        # optim: Barlow decorrelation runs in a low-dim latent to avoid the
        # [C,C,H,H] cross-covariance tensor. Fixed orthonormal projection (buffer,
        # not learnable) so the penalty can't be gamed by rotating features away.
        if self.optim:
            d = min(div_latent_dim, H)
            proj = torch.empty(H, d)
            nn.init.orthogonal_(proj)
            self.register_buffer('div_proj', proj)          # [H, d]
        # optim: mean-pool over columns -> narrow head (fewer params + activations)
        head_in = H if self.optim else self.n_columns * H
        self.head = nn.Linear(head_in, output_size)

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
        aux_div = aux_div_max = aux_gate = aux_act = aux_sat = 0.0
        # Increment tick BEFORE reading do_aux so both passes of grad checkpointing
        # agree: use_reentrant=False runs the original pass under no_grad and the
        # recompute pass under enable_grad, so gate the increment on grad state to
        # advance the counter exactly once (on the original pass).
        if self.use_aux and self.training:
            if not self.grad_checkpoint or not torch.is_grad_enabled():
                self._aux_tick += 1
        do_aux = self.use_aux and self.training and (self._aux_tick % self.aux_every == 0).any()
        x = self.drop(torch.stack(
            [proj(x) for proj in self.col_input_projs], dim=0))  # [C, B, E]

        o = None
        for layer, (cells, attn, gates, hl) in enumerate(
            zip(self.cells, self.attn, self.attn_gates, h)
        ):
            hl_n = torch.stack([
                cells[ic](x[ic], hl[ic]) for ic in range(self.n_columns)
            ], dim=0)                                            # [C, B, H]

            msg, attn_w = attn(hl_n, return_weights=return_attn, col_mask=self.attn_col_mask)
            g = torch.sigmoid(gates).view(self.n_columns, 1, 1)  # [C, 1, 1]
            o = hl_n + g * msg

            if do_aux:
                d, dmax, gv, a = self._layer_aux(hl_n, hl, g)
                aux_div = aux_div + self._div_layer_w[layer] * d
                aux_div_max = aux_div_max + self._div_layer_w[layer] * dmax
                aux_gate, aux_act = aux_gate + gv, aux_act + a
                if layer > 0:
                    # anti tanh-saturation: penalize upper-layer |h| above target
                    aux_sat = aux_sat + F.relu(hl_n.abs().mean() - self.sat_target)

            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)
            x = self.drop(self.mid_norms[layer](o)) if layer < self.n_layers - 1 else o

        h_n = torch.stack(h_n, dim=0)
        if self.optim:
            o_top = o.mean(dim=0)                                # [B, H]
        else:
            o_top = o.permute(1, 0, 2).reshape(o.shape[1], -1)   # [B, C*H]

        aux = None
        if self.use_aux:
            if do_aux:
                div_w, div_max_w = self._div_weights()
                aux = self.aux_every * (
                    div_w * aux_div
                    + div_max_w * aux_div_max
                    + self.aux_gate_weight * aux_gate
                    + self.aux_act_weight * aux_act
                    + self.aux_sat_weight * aux_sat
                ) / self.n_layers
            else:
                aux = torch.zeros((), device=o.device, dtype=o.dtype)
        extras = {"attn_weights": attn_list, "gates": gate_list}
        return h_n, o_top, extras, aux

    def _div_weights(self):
        # ramp diversity weights up over training: on long SDQ runs col_sim/max
        # keeps climbing in the *second* half (a specific pair collapsing)
        # even while the constant-weight mean term holds col_sim/mean near 0
        if self.aux_div_ramp_steps <= 0:
            return self.aux_div_weight, self.aux_div_max_weight
        # _aux_tick is a CPU tensor (compile-friendly); read as a python float so the
        # returned weights stay device-agnostic (they later multiply CUDA aux terms)
        tick = self._aux_tick.item() if torch.is_tensor(self._aux_tick) else self._aux_tick
        frac = min(1.0, tick / self.aux_div_ramp_steps)
        return self.aux_div_weight * frac, self.aux_div_max_weight * frac

    def _layer_aux(self, hl_n, hl, g):
        C = self.n_columns
        iu, ju = torch.triu_indices(C, C, offset=1, device=hl_n.device)

        # (1) Barlow-style feature decorrelation between columns
        z = hl_n - hl_n.mean(dim=1, keepdim=True)                # [C, B, H]
        z = z / (z.std(dim=1, keepdim=True) + 1e-6)
        if self.optim:
            # subsample batch, then project H -> d_latent to shrink the cross tensor
            # from [C,C,H,H] to [C,C,d,d] (the dominant memory term at large H)
            B_full = z.shape[1]
            k = max(1, int(B_full * self.aux_batch_frac))
            if k < B_full:
                idx = torch.randperm(B_full, device=z.device)[:k]
                z = z[:, idx, :]
            z = z @ self.div_proj                                # [C, B', d]
        B = z.shape[1]
        cross = torch.einsum('cbh,dbk->cdhk', z, z) / B          # [C, C, D, D], D=H or d
        pair_div = cross[iu, ju].pow(2).mean(dim=(-1, -2))       # [n_pairs]
        div = pair_div.mean()
        # target the single most-collapsed pair specifically (e.g. a CKA~0.8
        # pair hiding behind a healthy col_sim/mean) instead of only the
        # uniformly-weighted average over all pairs
        div_max = pair_div.max()

        # (2) gate diversity: push per-column mean gates apart
        gm = g.mean(dim=(1, 2))                                  # [C]
        gate = F.relu(self.gate_std_target - gm.std())

        # (3) activity decorrelation: columns update at different times
        u = (hl_n - hl).norm(dim=-1)                             # [C, B]
        u = u - u.mean(dim=1, keepdim=True)
        u = u / (u.norm(dim=1, keepdim=True) + 1e-6)
        corr = u @ u.T                                           # [C, C]
        act = F.relu(corr[iu, ju]).mean()

        return div, div_max, gate, act

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


class PerColumnAttention(nn.Module):
    # column mixing where each column attends differently: learnable per-column
    # query/key identities and per-(column, head) beta; tiny out_proj, NO post-norm
    def __init__(self, dim, num_heads, n_columns, beta_scale: float = 1.0,
                 beta_floor: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_columns = n_columns
        self.beta_floor = beta_floor

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # per-column identities: each column asks its own question
        xavier_alpha = (1 / dim) ** 0.5
        self.ids_q = nn.Parameter(torch.empty(n_columns, 1, dim))
        self.ids_k = nn.Parameter(torch.empty(n_columns, 1, dim))
        nn.init.normal_(self.ids_q, 0.0, 0.1 * xavier_alpha)
        nn.init.normal_(self.ids_k, 0.0, 0.1 * xavier_alpha)

        # per-(column, head) sharpness: columns read with different selectivity;
        # staggered init from broad (0.5x) to sharp (2x) around beta_scale.
        # beta_floor is a guaranteed minimum (cf. FastFloorLRUCell's r_floor in
        # grnn_fix_v5): long runs otherwise let most columns' beta decay toward
        # near-uniform attention, so init targets beta_scale net of the floor.
        target = max(beta_scale / math.sqrt(self.head_dim) - beta_floor, 1e-4)
        base = math.log(target)
        spread = torch.linspace(math.log(0.5), math.log(2.0), n_columns)
        self.log_beta = nn.Parameter(
            base + spread[:, None].repeat(1, num_heads)          # [C, heads]
        )

        nn.init.normal_(self.out_proj.weight, 0.0, 0.001)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h, return_weights: bool = False, col_mask=None):
        # h: [C, B, D]
        C, B, D = h.shape
        q = self.W_q(h + self.ids_q).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        k = self.W_k(h + self.ids_k).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        v = self.W_v(h).view(C, B, self.num_heads, self.head_dim).permute(2, 1, 0, 3)

        # beta indexed by the query (receiving) column  [heads, 1, C, 1]
        beta = self.beta_floor + self.log_beta.exp().T.unsqueeze(1).unsqueeze(-1)
        logits = beta * torch.matmul(q, k.transpose(-2, -1))     # [heads, B, C_q, C_k]
        if col_mask is not None:
            # col_mask: [C, C] bool, True = query i may attend to key j (ablation)
            logits = logits.masked_fill(~col_mask.to(logits.device)[None, None], float('-inf'))
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)                              # [heads, B, C, hd]
        out = out.permute(2, 1, 0, 3).contiguous().view(C, B, D)
        attn_w = attn.mean(dim=(0, 1)) if return_weights else None
        return self.out_proj(out), attn_w
