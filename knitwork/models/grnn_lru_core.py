from __future__ import annotations

from collections import defaultdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.common.torch import normalize_entropy, print_max_norm


class GridRnn(nn.Module):
    has_attn = True

    def __init__(
            self, *,
            hidden_size, n_layers, n_columns, n_inputs=1, n_outputs=1,
            mha, n_attn_heads=1, noise_std, horizon, 
            use_bias=True, ln_msg=True,
            dtype, device,
    ):
        super().__init__()
        assert n_columns > 1
        assert 0 < n_inputs <= n_columns
        assert 0 < n_outputs <= n_columns
        assert n_attn_heads == 1

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.n_attn_heads = n_attn_heads
        self.dtype = dtype
        self.device = device
        print(
            f'GridRNN-LRU of {n_layers}L x {n_columns}C '
            f'w/ {self.hidden_size} hidden units'
        )
        self.cells = LruBank(
            n_layers=n_layers, n_columns=n_columns,
            hidden_size=self.hidden_size, bias=use_bias,
            horizon=horizon,
        )
        mhas = [
            None, None, None,
            StaticMessagePassingLayer,
        ]
        if not 0 <= mha < len(mhas):
            raise ValueError('mha must be between 0 and 3')
        self.comm = self.attn = nn.ModuleList()
        mha_cls = mhas[mha]
        mha_kwargs = {'noise_std': noise_std} if mha in (2, 3) else {}
        for layer in range(n_layers):
            n_kv = self.n_columns + self.n_inputs if layer == 0 else self.n_columns
            self.comm.append(mha_cls(
                self.hidden_size, num_heads=n_attn_heads, ln_msg=ln_msg,
                n_q=self.n_columns, n_kv=n_kv,
                **mha_kwargs,
            ))

    def forward(self, x, state, *, capture=False, **_):
        # x shape: (In, B, H)
        assert x.shape[0] == self.n_inputs
        # h shape: (L, B, C, H); 
        # tp == t prev == t-1
        h_tp, outs_tp = state['h'], state['outs']

        # (B, C, H)
        # extend prev state and internal input w/ ext input
        x_int, x_ext = state['out'], x.transpose(0, 1)
        out = torch.cat([x_int, x_ext], dim=1)

        h_t, outs_t = [], []
        info = defaultdict(list)

        for layer in range(self.n_layers):
            msg_in = out
            msg_out, comm_info = self.comm[layer](outs_tp[layer], msg_in, msg_in, return_weights=capture)

            cell_in = msg_out
            hl_tp = h_tp[layer]
            cell_out, hl_t = self.cells(layer, cell_in, hl_tp)
            out = cell_out

            print_max_norm(f'[l{layer}]', out, mx=1.0e+2)

            for k, v in comm_info.items():
                info[k].append(v)
            h_t.append(hl_t)
            outs_t.append(out)
    
        h_t = torch.stack(h_t, dim=0)
        outs_t = torch.stack(outs_t, dim=0)
        y = outs_t[-1][:, 0]
        state = {'h': h_t, 'outs': outs_t, 'out': outs_t[-1]}
        return y, state, info

    def init_state(self, bsz):
        small = 0.01 / math.sqrt(self.hidden_size)
        h = small * torch.randn(
            self.n_layers, bsz, self.n_columns, 2 * self.hidden_size,
            device=self.device, dtype=self.dtype
        )
        outs = small * h.new_zeros(
            self.n_layers, bsz, self.n_columns, self.hidden_size,
        )
        return {'h': h, 'outs': outs, 'out': outs[-1]}

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)

        # (L, B, C, H)
        keep = ~reset_mask.flatten()
        keep_ = keep[None, :, None, None]
        h, outs = state['h'] * keep_, state['outs'] * keep_
        return {'h': h, 'outs': outs, 'out': outs[-1]}

    def detach_state(self, state):
        if state is None:
            return state
        return {key: value.detach() for key, value in state.items()}


class StaticMessagePassingLayer(nn.Module):
    """Message passing with learned, query-independent routing."""
    def __init__(
            self, dim, num_heads, ln_msg=True, n_q=None, n_kv=None,
            noise_std=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads == 1, "Such simplified message passing doesn't need multihead"
        assert n_q is not None and n_kv is not None

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.noise_std = noise_std
        # self.value_proj = nn.Linear(dim, dim)
        # self.out_proj = nn.Linear(dim, dim)
        # self.ln_msg = nn.LayerNorm(dim) if ln_msg else None

        # Store one Cq->Ckv routing table per head.
        self.pi_route_logits = nn.Parameter(torch.empty(n_q, n_kv))
        # self.pi_logtemp = nn.Parameter(torch.empty(1, n_q, 1))

        self.reset_parameters()

    def forward(self, q, k, v, return_weights: bool = False):
        Cq, Ckv = self.pi_route_logits.shape
        # (B, Ckv, D)
        B, _, D = v.shape

        # broadcast batch dim: (B, Cq, Ckv)
        logits = self.pi_route_logits.unsqueeze(0).expand(B, Cq, Ckv)
        # beta = F.softplus(self.pi_logtemp)
        # pi_route = torch.softmax(beta * logits, dim=-1)
        pi_route = torch.softmax(logits, dim=-1)

        # (B, Cq, Ckv) x (B, Ckv, D) --> (B, Cq, D)
        msg = torch.bmm(pi_route, v)
        # if self.ln_msg is not None:
        #     msg = self.ln_msg(msg)

        info = {}
        if return_weights:
            info['attn_weights'] = pi_route.detach().mean(0)

        return msg, info

    def forward_old(self, q, k, v, return_weights: bool = False):
        Cq, Ckv = self.pi_route_logits.shape
        B, _, D = v.shape

        # (B, Ckv, D)
        v = self.value_proj(v)

        # broadcast batch dim: (B, Cq, Ckv)
        logits = self.pi_route_logits.unsqueeze(0).expand(B, Cq, Ckv)
        if self.training and self.noise_std > 0.0:
            logits = logits + self.noise_std * torch.randn_like(logits)
        beta = F.softplus(self.pi_logtemp)
        pi_route = torch.softmax(beta * logits, dim=-1)

        # (B, Cq, Ckv) * (B, Ckv, D) --> (B, Cq, D)... BMM?
        msg = torch.matmul(pi_route, v)
        msg = self.out_proj(msg)
        if self.ln_msg is not None:
            msg = self.ln_msg(msg)

        info = {}
        if self.training:
            prob_comm = 1.0 - pi_route.diagonal(dim1=-2, dim2=-1)
            comm_loss = prob_comm
            if Ckv > Cq:
                # Add a weighted column-specific extra cost for external input.
                extra_w = torch.arange(Cq, dtype=pi_route.dtype, device=pi_route.device).view(1, -1)
                x_ext_prob_weighted = pi_route[..., -1] * (extra_w * 2 - 1)
                comm_loss = comm_loss + x_ext_prob_weighted
            entropy = -(pi_route * torch.log(pi_route.clamp_min(torch.finfo(pi_route.dtype).tiny))).sum(dim=-1)
            info |= {
                'comm_loss': comm_loss.mean(),
                'comm_entropy': normalize_entropy(entropy.mean(), Ckv),
            }
        if return_weights:
            info['attn_weights'] = pi_route.detach().mean(0)

        return msg, info

    @torch.no_grad()
    def reset_parameters(self):
        small = 0.01 / math.sqrt(self.dim)
        # nn.init.eye_(self.value_proj.weight)
        # nn.init.zeros_(self.value_proj.bias)

        # nn.init.eye_(self.out_proj.weight)
        # self.out_proj.weight.add_(torch.randn_like(self.out_proj.weight) * small)

        # nn.init.zeros_(self.out_proj.bias)
        self.init_logits_near_zero()
        # nn.init.constant_(self.pi_logtemp, math.log(math.expm1(1.0)))

    @torch.no_grad()
    def init_logits_near_zero(self):
        nn.init.normal_(self.pi_route_logits, 0.0, 0.01 / math.sqrt(self.dim))

    @torch.no_grad()
    def init_logits_positive_diagonal(self):
        nn.init.zeros_(self.pi_route_logits)
        n_q, n_kv = self.pi_route_logits.shape
        n = min(n_q, n_kv)
        ixs = torch.arange(n, device=self.pi_route_logits.device)
        # set diag elems
        self.pi_route_logits[ixs, ixs] = 1.0


class LruBank(nn.Module):
    """LxC independent complex LRUs with real-packed state and residual output."""

    def __init__(self, *, n_layers, n_columns, hidden_size, horizon, bias=True):
        super().__init__()
        hz_min, hz_max = horizon
        if hz_min <= 0 or hz_max <= 0 or hz_min > hz_max:
            raise ValueError('horizon bounds must be positive and ordered')

        self.n_layers = n_layers
        self.n_columns = n_columns
        self.hidden_size = hidden_size
        self.use_bias = bias
        self.horizon = horizon
        self.hz_min, self.hz_max = hz_min, hz_max

        L, C, H = n_layers, n_columns, hidden_size 
        self.log_r = nn.Parameter(torch.empty(L, C, H))
        self.theta = nn.Parameter(torch.empty(L, C, H))

        self.weight_in = nn.Parameter(torch.empty(L, C, H, 2*H))
        self.weight_out = nn.Parameter(torch.empty(L, C, 2*H, H))
        if bias:
            self.bias_out = nn.Parameter(torch.empty(L, C, 1, H))
        else:
            self.register_parameter('bias_out', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.hidden_size)
        # radius = exp(-1 / horizon) = exp(-exp(log_r)).
        nn.init.uniform_(self.log_r, -math.log(self.hz_max), -math.log(self.hz_min))
        nn.init.uniform_(self.theta, 0.0, 2.0 * math.pi)

        for parameter in (self.weight_in, self.weight_out):
            nn.init.uniform_(parameter, -bound, bound)
        for parameter in (self.bias_out,):
            if parameter is not None:
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

    def forward_new(self, layer, x, h):
        # x: [B, C, H]
        # w_in: [C, H, 2H]
        B, C, H = x.shape

        beta = 0.1
        re = beta * x + (1-beta)*h[..., :H]
        im = beta * x - (1-beta)*h[..., H:]
        h_n = torch.cat([re, im], -1)
        y = re
        out = F.silu(y) + x
        # print(x.shape, out.shape, y.shape, h_n.shape)
        # assert False
        return out, h_n

    def forward(self, layer, x, h):
        # x: [B, C, H]
        # w_in: [C, H, 2H]
        B, C, H = x.shape

        # [C, B, H] x [C, H, 2H] -> [C, B, 2H] -> [B, C, 2H]
        drive = torch.bmm(x.transpose(0, 1), self.weight_in[layer])
        b_re, b_im = drive.transpose(0, 1).chunk(2, dim=-1)
        # b_re, b_im = drive.chunk(2, dim=-1)

        # h: [B, C, 2H] with real then imaginary components.
        h_re, h_im = h.chunk(2, dim=-1)
        lam_re, lam_im, gamma = self._lambda_gamma(layer)
        lam_re = lam_re.unsqueeze(0)
        lam_im = lam_im.unsqueeze(0)
        gamma = gamma.unsqueeze(0)

        new_re = lam_re * h_re - lam_im * h_im + gamma * b_re
        new_im = lam_re * h_im + lam_im * h_re + gamma * b_im
        h_n = torch.cat((new_re, new_im), dim=-1)

        # A real projection of packed state implements real(C h).
        y = torch.bmm(h_n.transpose(0, 1), self.weight_out[layer])
        if self.use_bias:
            y = y + self.bias_out[layer]
        y = y.transpose(0, 1)
        out = F.silu(y) + x
        # print(x.shape, out.shape, y.shape, h_n.shape)
        # assert False
        return out, h_n
