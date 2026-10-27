from __future__ import annotations

from collections import defaultdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.models.grnn_core import StochasticMessagePassingLayer


class GridRnn(nn.Module):
    has_attn = True
    batch_first = True  # step input is (B, n_inputs, H)

    def __init__(
            self, *,
            hidden_size, n_layers, n_columns, n_inputs=1, n_outputs=1,
            horizon, fb_norm=False, x_add=False, feedback=True, write_topk=0,
            routing='static', n_attn_heads=4, selective_rank=0, comm_rounds=1, read_attn=False,
            route_topk=None, column_horizons=False, input_all_layers=False,
            write_mode='input', write_key_size=32, write_balance=0.0, write_free=0.0,
            input_gate_rank=0, gru_columns=0, route_query='prev', write_alloc=False,
            layer_strides=None, mem_slots=0, mem_stride=8, share_columns=False, backbone=False,
            raw_columns=0, cell_act='silu',
            dtype, device,
    ):
        super().__init__()
        assert n_columns > 1
        assert 0 < n_inputs <= n_columns
        assert 0 < n_outputs <= n_columns

        self.n_inputs = n_inputs
        # parameter-free RMS norm on the top-layer output fed back as the next step's message: bounds the residual loop
        self.fb_norm = fb_norm
        # x_add: also add the external input to every layer-0 column input (bypasses routing dilution)
        # feedback=False: drop the previous top-layer outputs from the layer-0 routing sources
        self.x_add = x_add
        self.feedback = feedback
        # write_topk=k > 0: per step and layer only the k best-scoring columns update their state
        # (content-based sparse writes); the rest keep their state but are still read
        assert 0 <= write_topk <= n_columns
        self.write_topk = write_topk
        # write_mode='state': score = <q(column input), k(column state)>, keys shared across columns so
        # columns compete by content; the key of a column moves with what it stored
        # write_balance=u: aux-loss-free load balancing, a selection-only bias nudged by u toward even load
        # write_free=a: selection prefers columns with low state RMS (unused memory)
        assert write_mode in ('input', 'state')
        self.write_mode, self.write_balance, self.write_free = write_mode, write_balance, write_free
        # routing='dynamic': per-step query-key attention between columns instead of the static table
        # routing='table': the dynamic layer's projections and LayerNorm with query-independent weights (control)
        assert routing in ('static', 'dynamic', 'table')
        self.routing = routing
        # comm_rounds=n > 1: n - 1 further exchanges among the columns' received messages (shared weights)
        assert comm_rounds in (1, 2, 3) and (comm_rounds == 1 or routing != 'static')
        self.comm_rounds = comm_rounds
        # route_query='state': dynamic-routing queries are a projection of the column's LRU state (route by memory content)
        # route_query='current': dynamic-routing queries come from this step (layer 0: the input, layer l: layer l - 1
        # outputs) instead of the column's previous output; with feedback=False routing is then feed-forward in time
        assert route_query in ('prev', 'current', 'state') and (route_query == 'prev' or routing == 'dynamic')
        self.route_query = route_query
        # write_alloc: soft content-addressed write, columns compete for their input by matching the external input
        # against what they last emitted; cell_in is scaled by softmax over columns * C (uniform split = no change)
        self.write_alloc = write_alloc
        # layer_strides=[s_0, ..]: layer l updates its state and output only every s_l steps (clockwork hierarchy),
        # otherwise it holds them; gradients cross a long gap in gap / s_l steps
        assert layer_strides is None or len(layer_strides) == n_layers
        self.layer_strides = layer_strides
        # mem_slots=M: every mem_stride steps the core output is written into a ring of M snapshot slots; the slots,
        # tagged with their age, are extra layer-0 routing sources (explicit, time-addressed episodic memory)
        assert mem_slots == 0 or routing == 'dynamic'
        self.mem_slots, self.mem_stride = mem_slots, mem_stride
        # backbone: the external input first passes a single wide LRU (same width), the grid routes on its output
        self.backbone = backbone
        # raw_columns=k: the last k columns of layer 0 take the external input only (no routed message):
        # a plain LRU stream inside the grid, still a routing source for the other columns
        assert 0 <= raw_columns < n_columns
        self.raw_columns = raw_columns
        # read_attn: the core output is column 0 plus attention from the external input over top-layer columns
        self.read_attn = read_attn
        # route_topk=k: dynamic routing reads only the k best sources per column and head (sparse reads)
        assert route_topk is None or routing == 'dynamic'
        # input_all_layers: the external input is a routing source in every layer, not only layer 0
        self.input_all_layers = input_all_layers
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.dtype = dtype
        self.device = device
        print(
            f'GridRNN-LRU of {n_layers}L x {n_columns}C '
            f'w/ {self.hidden_size} hidden units'
        )

        self.cells = LruBank(
            n_layers=n_layers, n_columns=n_columns, hidden_size=self.hidden_size, horizon=horizon,
            selective_rank=selective_rank, input_gate_rank=input_gate_rank, gru_columns=gru_columns,
            column_horizons=column_horizons, share_columns=share_columns, act=cell_act,
        )
        if backbone:
            self.backbone_cell = LruBank(n_layers=1, n_columns=1, hidden_size=self.hidden_size, horizon=horizon)

        def make_comm(n_q, n_kv):
            if routing == 'dynamic':
                return StochasticMessagePassingLayer(
                    hidden_size, num_heads=n_attn_heads, ln_msg=True, n_q=n_q, n_kv=n_kv, top_k=route_topk,
                )
            if routing == 'table':
                return TableMessagePassingLayer(hidden_size, n_q=n_q, n_kv=n_kv)
            return StaticCommunication(hidden_size, n_q=n_q, n_kv=n_kv)

        C = n_columns
        self.comm = self.attn = nn.ModuleList(
            make_comm(C, C + n_inputs + mem_slots if layer == 0 else C + (n_inputs if input_all_layers else 0))
            for layer in range(n_layers)
        )
        if mem_slots:
            self.mem_age = nn.Parameter(torch.randn(mem_slots, hidden_size) * 0.02)
        if comm_rounds > 1:
            self.comm2 = nn.ModuleList(make_comm(C, C) for _ in range(n_layers))
        if read_attn:
            self.read = StochasticMessagePassingLayer(hidden_size, num_heads=n_attn_heads, ln_msg=True, n_q=n_inputs, n_kv=C)
        if route_query == 'state':
            self.state_q = nn.Parameter(torch.randn(n_layers, 2 * hidden_size, hidden_size) / math.sqrt(2 * hidden_size))
        if write_alloc:
            L, H, K = n_layers, hidden_size, write_key_size
            self.alloc_q = nn.Parameter(torch.randn(L, H, K) / math.sqrt(H))
            # zero key projection: uniform allocation at init; column ids break the symmetry later
            self.alloc_k = nn.Parameter(torch.zeros(L, H, K))
            self.alloc_ids = nn.Parameter(torch.randn(L, C, 1, K) * 0.1)
        if write_topk:
            L, H = n_layers, hidden_size
            if write_mode == 'state':
                self.write_q = nn.Parameter(torch.randn(L, H, write_key_size) / math.sqrt(H))
                self.write_k = nn.Parameter(torch.randn(L, H, write_key_size) / math.sqrt(H))
            else:
                self.write_w = nn.Parameter(torch.randn(L, C, 1, H) / math.sqrt(H))
            # sigmoid(2) ~ 0.88: selected columns start as almost full writes
            self.write_b = nn.Parameter(torch.full((L, C, 1, 1), 2.0))
            self.register_buffer('write_bias', torch.zeros(L, C, 1, 1))

    def forward(self, x, state, *, capture=False, **_):
        # x: (B, In, H)
        assert x.shape[1] == self.n_inputs
        # tp == t prev == t-1
        # outs: (L, B, C, H)
        outs_tp = state['outs']
        # h: (L, C, B, 2H)
        h_tp = state['h']

        # (B, C, H)
        # extend prev output (as internal input) with external input
        x_int, x_ext = state['out'], x
        if self.backbone:
            # (In, B, H) -> (B, In, H) backbone output replaces the external input
            x_ext, hb = self.backbone_cell(0, x.transpose(0, 1), state['hb'])
        if not self.feedback:
            x_int = torch.zeros_like(x_int)
        out = torch.cat([x_int, x_ext], dim=1)
        timed = bool(self.layer_strides) or self.mem_slots > 0
        t = state['t'] if timed else None  # (B,) steps since the episode start
        if self.mem_slots:
            M, S = self.mem_slots, self.mem_stride
            # (M, B) slot age in writes, 0 = written at the last write step
            last = (t - 1).clamp(min=0) // S
            age = (last[None] - torch.arange(M, device=t.device)[:, None]) % M
            # (B, C + In + M, H): snapshots tagged with their age join the layer-0 sources
            mem_tp = state['mem']
            out = torch.cat([out, (mem_tp + self.mem_age[age]).transpose(0, 1)], dim=1)

        h_t, outs_t = [], []
        info = defaultdict(list)

        for layer in range(self.n_layers):
            msg_in = out
            if layer > 0 and self.input_all_layers:
                msg_in = torch.cat([out, x_ext], dim=1)
            # in: (B, C, H), out: (C, B, H)
            if self.routing == 'static':
                msg_out, comm_info = self.comm[layer](outs_tp[layer], msg_in, msg_in, return_weights=capture)
            else:
                # queries from the column's previous output, keys/values from the sources: (C|Ckv, B, H)
                q, kv = outs_tp[layer].transpose(0, 1), msg_in.transpose(0, 1)
                if self.route_query == 'current':
                    # (C, B, H): layer 0 asks with the input, upper layers with the layer below at this step
                    q = x_ext.sum(1)[None].expand_as(q) if layer == 0 else out.transpose(0, 1)
                elif self.route_query == 'state':
                    # (C, B, 2H) @ (2H, H): ask with what the column holds
                    q = h_tp[layer] @ self.state_q[layer]
                msg_out, comm_info = self.comm[layer](q, kv, kv, return_weights=capture)
            for _ in range(self.comm_rounds - 1):
                # (C, B, H): columns exchange what they received, residual
                msg2, _ = self.comm2[layer](msg_out, msg_out, msg_out)
                msg_out = msg_out + msg2

            cell_in = msg_out
            if layer == 0 and self.x_add:
                # (B, In, H) -> (1, B, H), broadcast over columns
                cell_in = cell_in + x_ext.sum(1)[None]
            if layer == 0 and self.raw_columns:
                k = self.raw_columns
                cell_in = torch.cat([cell_in[:-k], x_ext.sum(1)[None].expand(k, -1, -1)])
            if self.write_alloc:
                # (B, K) query from the external input, (C, B, K) keys from the columns' previous outputs
                q = x_ext.sum(1) @ self.alloc_q[layer]
                k = outs_tp[layer].transpose(0, 1) @ self.alloc_k[layer] + self.alloc_ids[layer]
                w = torch.softmax((q * k).sum(-1, keepdim=True) / math.sqrt(k.shape[-1]), dim=0) * self.n_columns
                cell_in = cell_in * w
            hl_tp = h_tp[layer]
            # in: (C, B, H), out: (B, C, H)
            write = None
            if self.write_topk:
                write = self._write_gate(layer, cell_in, outs_tp[layer], hl_tp)
            cell_out, hl_t = self.cells(layer, cell_in, hl_tp, write=write)
            if self.layer_strides and self.layer_strides[layer] > 1:
                # clockwork: off-beat steps hold the layer's state and output
                upd = t % self.layer_strides[layer] == 0
                hl_t = torch.where(upd[None, :, None], hl_t, hl_tp)
                cell_out = torch.where(upd[:, None, None], cell_out, outs_tp[layer])
            out = cell_out

            for k, v in comm_info.items():
                info[k].append(v)
            h_t.append(hl_t)
            outs_t.append(out)

        h_t = torch.stack(h_t, dim=0)
        outs_t = torch.stack(outs_t, dim=0)
        y = outs_t[-1][:, 0]
        if self.read_attn:
            # (In, B, H) queries over (C, B, H) top columns -> (B, H)
            kv = outs_t[-1].transpose(0, 1)
            read, _ = self.read(x_ext.transpose(0, 1), kv, kv)
            y = y + read.sum(0)
        out = F.rms_norm(outs_t[-1], (self.hidden_size,)) if self.fb_norm else outs_t[-1]
        state = {'h': h_t, 'outs': outs_t, 'out': out}
        if self.backbone:
            state['hb'] = hb
        # key order must match init_state: CUDA graphs rebuild the state from a flat leaf list
        if timed:
            state['t'] = t + 1
        if self.mem_slots:
            # write the core output into slot (t // S) % M on every S-th step: (M, B, H)
            slot = (t // S) % M
            write = (torch.arange(M, device=t.device)[:, None] == slot[None]) & (t % S == 0)[None]
            state['mem'] = torch.where(write[..., None], y[None], mem_tp)
        return y, state, info

    def _write_gate(self, layer, cell_in, outs_tp, h_tp):
        # cell_in: (C, B, H), outs_tp: (B, C, H), h_tp: (C, B, 2H) -> write strength (C, B, 1)
        if self.write_mode == 'state':
            q = cell_in @ self.write_q[layer]
            k = outs_tp.transpose(0, 1) @ self.write_k[layer]
            score = (q * k).sum(-1, keepdim=True) / math.sqrt(q.shape[-1]) + self.write_b[layer]
        else:
            score = (cell_in * self.write_w[layer]).sum(-1, keepdim=True) + self.write_b[layer]
        # selection-only terms: they decide which columns write, not how strongly
        sel = score.detach() + self.write_bias[layer]
        if self.write_free:
            sel = sel - self.write_free * h_tp.pow(2).mean(-1, keepdim=True).sqrt()
        kth = sel.topk(self.write_topk, dim=0).values[-1:]
        mask = sel >= kth
        if self.write_balance:
            with torch.no_grad():
                load = mask.float().mean(dim=(1, 2), keepdim=True)
                self.write_bias[layer] += self.write_balance * torch.sign(self.write_topk / self.n_columns - load)
        return torch.sigmoid(score) * mask

    def init_state(self, bsz):
        L, B, C, H = self.n_layers, bsz, self.n_columns, self.hidden_size
        small = 0.01 / math.sqrt(self.hidden_size)

        h = torch.empty(L, C, B, 2*H, device=self.device, dtype=self.dtype).normal_(0.0, small)
        outs = h.new_empty(L, B, C, H).normal_(0.0, small)

        state = {'h': h, 'outs': outs, 'out': outs[-1]}
        if self.backbone:
            state['hb'] = h.new_zeros(1, B, 2 * H)
        if self.layer_strides or self.mem_slots:
            state['t'] = torch.zeros(B, dtype=torch.long, device=self.device)
        if self.mem_slots:
            state['mem'] = h.new_zeros(self.mem_slots, B, H)
        return state

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)

        # (B,) -> 4D tensor
        keep = ~reset_mask.flatten()
        keep_h = keep[None, None, :, None]
        keep_outs = keep[None, :, None, None]
        h, outs = state['h'] * keep_h, state['outs'] * keep_outs
        out = F.rms_norm(outs[-1], (self.hidden_size,)) if self.fb_norm else outs[-1]
        new = {'h': h, 'outs': outs, 'out': out}
        if 'hb' in state:
            new['hb'] = state['hb'] * keep[None, :, None]
        if 't' in state:
            new['t'] = state['t'] * keep
        if 'mem' in state:
            new['mem'] = state['mem'] * keep[None, :, None]
        return new

    def detach_state(self, state):
        if state is None:
            return state
        return {key: value.detach() for key, value in state.items()}


class StaticCommunication(nn.Module):
    """Communication with learned, query-independent (=static) routing."""
    def __init__(
            self, dim, n_q=None, n_kv=None,
    ):
        super().__init__()
        assert n_q is not None and n_kv is not None
        self.dim = dim

        # Store one Cq->Ckv routing table per head.
        self.pi_route_logits = nn.Parameter(torch.empty(n_q, n_kv))

        self.reset_parameters()

    def forward(self, q, k, v, return_weights: bool = False):
        Cq, Ckv = self.pi_route_logits.shape
        # (B, Ckv, D)
        B, _, D = v.shape

        # broadcast batch dim: (B, Cq, Ckv)
        pi_route = torch.softmax(self.pi_route_logits, dim=-1).unsqueeze(0).expand(B, Cq, Ckv)
        # (B, Cq, Ckv) x (B, Ckv, D) --> (B, Cq, D)
        msg = torch.bmm(pi_route, v)

        info = {}
        if return_weights:
            info['attn_weights'] = pi_route.detach().mean(0)

        # to (C, B, D)
        msg = torch.transpose_copy(msg, 0, 1)
        return msg, info

    @torch.no_grad()
    def reset_parameters(self):
        self.init_logits_positive_diagonal()

    @torch.no_grad()
    def init_logits_near_zero(self):
        nn.init.normal_(self.pi_route_logits, 0.0, 0.01 / math.sqrt(self.dim))

    @torch.no_grad()
    def init_logits_positive_diagonal(self):
        self.init_logits_near_zero()
        n_q, n_kv = self.pi_route_logits.shape
        n = min(n_q, n_kv)
        ixs = torch.arange(n, device=self.pi_route_logits.device)
        # set diag elems
        self.pi_route_logits[ixs, ixs] = 1.0


class TableMessagePassingLayer(nn.Module):
    """Dynamic-layer value/output projections and LayerNorm, but query-independent routing weights."""
    def __init__(self, dim, n_q, n_kv):
        super().__init__()
        self.pi_route_logits = nn.Parameter(torch.zeros(n_q, n_kv))
        n = min(n_q, n_kv)
        with torch.no_grad():
            self.pi_route_logits[torch.arange(n), torch.arange(n)] = 1.0
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ln_msg = nn.LayerNorm(dim)

    def forward(self, q, k, v, return_weights: bool = False):
        # v: (Ckv, B, D) -> msg (Cq, B, D)
        pi_route = torch.softmax(self.pi_route_logits, dim=-1)
        msg = torch.einsum('qk,kbd->qbd', pi_route, F.silu(self.v_proj(v)))
        msg = self.ln_msg(self.out_proj(msg))
        info = {'attn_weights': pi_route.detach()} if return_weights else {}
        return msg, info


class LruBank(nn.Module):
    """LxC independent complex LRUs with real-packed state and residual output."""

    def __init__(
            self, *, n_layers, n_columns, hidden_size, horizon, selective_rank=0, input_gate_rank=0, gru_columns=0,
            column_horizons=False, share_columns=False, act='silu',
    ):
        super().__init__()
        hz_min, hz_max = horizon
        if hz_min <= 0 or hz_max <= 0 or hz_min > hz_max:
            raise ValueError('horizon bounds must be positive and ordered')

        self.n_layers = n_layers
        self.n_columns = n_columns
        self.hidden_size = hidden_size
        self.horizon = horizon
        # column_horizons: column c draws its horizons from the c-th of C log-spaced bands of [hz_min, hz_max];
        # 'reverse' gives column 0 (the readout) the longest band
        self.column_horizons = column_horizons
        # act='linear': column output y + x without the silu, a linear state readout
        assert act in ('silu', 'linear')
        self.act = act
        self.hz_min, self.hz_max = hz_min, hz_max

        # TBD: we can pre-process some of values for all layers at layer=0
        L, C, H = n_layers, n_columns, hidden_size 
        self.log_r = nn.Parameter(torch.empty(L, C, 1, H))
        self.theta = nn.Parameter(torch.empty(L, C, 1, H))

        # share_columns: one input/output projection per layer for all columns; columns differ by horizons only
        Cw = 1 if share_columns else C
        self.weight_in = nn.Parameter(torch.empty(L, Cw, H, 2*H))
        self.weight_out = nn.Parameter(torch.empty(L, Cw, 2*H, H))
        # selective_rank=r > 0: input-dependent step size s = softplus(x W1 W2 + b), lambda_t = lambda^s
        self.selective_rank = selective_rank
        if selective_rank:
            R = selective_rank
            self.dt_down = nn.Parameter(torch.randn(L, C, H, R) / math.sqrt(H))
            # zero up-projection and softplus(b) = 1: starts as the plain LRU
            self.dt_up = nn.Parameter(torch.zeros(L, C, R, H))
            self.dt_bias = nn.Parameter(torch.full((L, C, 1, H), math.log(math.expm1(1.0))))
        # input_gate_rank=r > 0: per-channel write gate g = 2 sigmoid(x W1 W2) on the input, fixed lambda
        self.input_gate_rank = input_gate_rank
        if input_gate_rank:
            R = input_gate_rank
            self.gate_down = nn.Parameter(torch.randn(L, C, H, R) / math.sqrt(H))
            # zero up-projection: g = 1, starts as the plain LRU
            self.gate_up = nn.Parameter(torch.zeros(L, C, R, H))
        # gru_columns=k > 0: the first k columns of every layer are GRU cells on the real half of the state
        assert 0 <= gru_columns < n_columns
        self.gru_columns = gru_columns
        if gru_columns:
            K = gru_columns
            self.gru_wx = nn.Parameter(torch.empty(L, K, H, 3 * H).uniform_(-1 / math.sqrt(H), 1 / math.sqrt(H)))
            self.gru_wh = nn.Parameter(torch.empty(L, K, H, 3 * H).uniform_(-1 / math.sqrt(H), 1 / math.sqrt(H)))
            self.gru_b = nn.Parameter(torch.zeros(L, K, 1, 3 * H))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.hidden_size)
        # radius = exp(-1 / horizon) = exp(-exp(log_r)).
        nn.init.uniform_(self.log_r, -math.log(self.hz_max), -math.log(self.hz_min))
        if self.column_horizons:
            C = self.n_columns
            with torch.no_grad():
                # [1, C, 1, 1] band edges in log horizon; u in [0, 1) places each channel inside its band
                band = torch.arange(C).flip(0) if self.column_horizons == 'reverse' else torch.arange(C)
                lo = math.log(self.hz_min) + band / C * math.log(self.hz_max / self.hz_min)
                u = torch.rand_like(self.log_r)
                self.log_r.copy_(-(lo.view(1, C, 1, 1) + u * math.log(self.hz_max / self.hz_min) / C))
        nn.init.uniform_(self.theta, 0.0, 2.0 * math.pi)

        for parameter in (self.weight_in, self.weight_out):
            nn.init.uniform_(parameter, -bound, bound)

    def _lambda_gamma(self, layer):
        theta = self.theta[layer]
        decay = torch.exp(self.log_r[layer])
        radius = torch.exp(-decay)
        return (
            radius * torch.cos(theta), 
            radius * torch.sin(theta),
            torch.sqrt(-torch.expm1(-2.0 * decay)),
        )

    def forward(self, layer, x, h, write=None):
        # x: [C, B, H]
        # h: [C, B, 2H] with real then imaginary components.
        C, B, H = x.shape
        # w_in: [C, H, 2H]

        if self.selective_rank:
            # [C, B, H] step size; s -> 0 keeps the state (no decay, no write), s >> 1 resets and writes
            s = F.softplus(torch.bmm(torch.bmm(x, self.dt_down[layer]), self.dt_up[layer]) + self.dt_bias[layer])
            decay = torch.exp(self.log_r[layer]) * s
            radius, theta = torch.exp(-decay), self.theta[layer] * s
            lam_re, lam_im = radius * torch.cos(theta), radius * torch.sin(theta)
            gamma = torch.sqrt((-torch.expm1(-2.0 * decay)).clamp_min(1e-6))
        else:
            # C, 1, H
            lam_re, lam_im, gamma = self._lambda_gamma(layer)
        h_re, h_im = h.chunk(2, -1)

        # [C, B, H] x [C, H, 2H] -> [C, B, 2H]
        u_re, u_im = torch.bmm(x, self.weight_in[layer].expand(C, -1, -1)).chunk(2, -1)
        if self.input_gate_rank:
            # [C, B, H] gate in (0, 2), shared by real and imaginary parts
            g = 2 * torch.sigmoid(torch.bmm(torch.bmm(x, self.gate_down[layer]), self.gate_up[layer]))
            u_re, u_im = g * u_re, g * u_im

        new_re = lam_re * h_re - lam_im * h_im + gamma * u_re
        new_im = lam_re * h_im + lam_im * h_re + gamma * u_im
        h_n = torch.cat((new_re, new_im), dim=-1)
        if self.gru_columns:
            K = self.gru_columns
            # [K, B, H] GRU state kept in the real half, the imaginary half stays zero
            hg = h_re[:K]
            gx = torch.bmm(x[:K], self.gru_wx[layer]) + self.gru_b[layer]
            gh = torch.bmm(hg, self.gru_wh[layer])
            xr, xz, xn = gx.chunk(3, -1)
            hr, hz, hn = gh.chunk(3, -1)
            r, z = torch.sigmoid(xr + hr), torch.sigmoid(xz + hz)
            hg = (1 - z) * torch.tanh(xn + r * hn) + z * hg
            h_n = torch.cat((torch.cat((hg, torch.zeros_like(hg)), -1), h_n[K:]), dim=0)
        if write is not None:
            # [C, B, 1] sparse write gate; every column is still read
            h_n = h + write * (h_n - h)

        # A real projection of packed state implements real(C h).
        y = torch.bmm(h_n, self.weight_out[layer].expand(C, -1, -1))
        out = (F.silu(y) if self.act == 'silu' else y) + x

        # to (B, C, H)
        out = torch.transpose_copy(out, 0, 1)
        return out, h_n
