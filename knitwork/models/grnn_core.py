from __future__ import annotations
from collections import defaultdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from knitwork.common.torch import normalize_entropy


class GridRnn(nn.Module):
    has_attn = True

    def __init__(
            self, *,
            hidden_size, n_layers: int, n_columns: int,
            n_inputs: int = 1, n_outputs: int = 1,
            n_attn_heads, use_bias=True,
            ln_msg=True, attn_gate=True, self_feeding=False, pre_msg=False, msg_alter_state=False,
            bank=2, mha=0,
            noise_std=0.0,
            dtype, device,
    ):
        super().__init__()
        assert n_columns > 1
        assert 0 < n_inputs <= n_columns
        assert 0 < n_outputs <= n_columns

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_columns = n_columns
        self.n_attn_heads = n_attn_heads
        self.self_feeding = self_feeding

        self.use_attn_gate = attn_gate
        self.msg_alter_state = msg_alter_state

        self.dtype = dtype
        self.device = device

        # Hidden size should be a multiply of the n_attn_heads
        self.hidden_size -= self.hidden_size % self.n_attn_heads
        print(
            f'GridRNN of {self.n_layers}L x {self.n_columns}C GRU cells'
            f' w/ {self.hidden_size} hidden units'
        )

        # Build a grid of cells: layers x columns
        banks = [GruBank, GruBank1, GruBank2]
        self.cells = banks[bank](self.n_layers, self.n_columns, self.hidden_size, bias=use_bias)

        mhas = [MessagePassingLayer, MessagePassingLayer1, StochasticMessagePassingLayer]
        mha = mhas[mha]
        mha_kwargs = {}
        if mha == StochasticMessagePassingLayer:
            mha_kwargs |= {'noise_std': noise_std}

        self.attn = nn.ModuleList()
        for layer in range(self.n_layers):
            self.attn.append(mha(
                self.hidden_size, num_heads=self.n_attn_heads, ln_msg=ln_msg, n_participants=self.n_columns,
                **mha_kwargs,
            ))

        self.pre_msg = pre_msg
        if not self.pre_msg and self.use_attn_gate:
            self.attn_gates = nn.ModuleList()
            for layer in range(self.n_layers):
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

    def forward(self, x: torch.Tensor, state: dict, *, capture=False, **_):
        # x shape: (In, B, H)
        assert x.shape[0] == self.n_inputs
        # h shape: (L, C, B, H)
        h, x_int = state['h'], state['out']
        h_new = []

        info = defaultdict(list)
        use_gates = not self.pre_msg and self.use_attn_gate

        # (C, B, H)
        x = self._prepare_grid_input(x, x_int)

        for layer in range(self.n_layers):
            hl = h[layer]
            if self.pre_msg:
                msg, comm_info = self.attn[layer](hl, x, x, return_weights=capture)
                x = msg

            hl_n = self.cells(layer, x, hl)
            msg = hl_n
            if not self.pre_msg:
                msg, comm_info = self.attn[layer](hl_n, hl_n, hl_n, return_weights=capture)
                if use_gates:
                    g = torch.sigmoid(self.attn_gates[layer](
                        torch.cat([hl_n, msg], dim=-1)
                    ))
                    msg = torch.lerp(hl_n, msg, g)

            # either communication msg alters state or not
            hl_n = msg if self.msg_alter_state else hl_n

            for k, v in comm_info.items():
                info[k].append(v)
            if capture and use_gates:
                info['gates'].append(g.detach())

            h_new.append(hl_n)
            x = msg

        h_new = torch.stack(h_new, dim=0)

        # top (=last) layer, first col as grid output
        # y = hl_n[0]
        y = x[0]

        state = {'h': h_new, 'out': x}
        return y, state, info

    def cell_forward(self, cells, x, h, *, ix_col):
        cells, x, h = cells[ix_col], x[ix_col], h[ix_col]
        return cells(x, h)

    def _cell_input_dim(self, ix_layer: int, ix_col) -> int:
        if ix_layer == 0:
            # only the first col gets non-empty external input, 
            # the others get dummy 1-dim zero tensor
            return self.embedding_size if ix_col == 0 else 1

        hsz = self.hidden_size
        if not self.use_postmsg:
            # RNN input: [x; h_mix]
            hsz += self.hidden_size
        return hsz

    def _prepare_grid_input(self, x_ext: torch.Tensor, x_int: torch.Tensor):
        if self.self_feeding:
            # h shape: (L, C, B, H)
            x_int = x_int[self.n_inputs:]
        else:
            n_internal_cols = self.n_columns - self.n_inputs
            x_int = x_ext.new_zeros(n_internal_cols, *x_ext.shape[1:])

        # (col, batch, features), cat over cols
        x_ext = torch.cat([x_ext, x_int], dim=0)
        return x_ext

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)

        # (L, C, B, H)
        keep = (~reset_mask.flatten())
        h = state['h'] * keep[None, None, :, None]
        out = state['out'] * keep[None, :, None]
        return {'h': h, 'out': out}

    def detach_state(self, state):
        if state is None:
            return state
        return {'h': state['h'].detach(), 'out': state['out'].detach()}

    def init_state(self, bsz):
        h = torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.device, dtype=self.dtype
        )
        return {'h': h, 'out': h[-1]}


class MessagePassingLayer(nn.Module):
    """Default MHA + layer norm."""
    def __init__(self, dim, num_heads, ln_msg=True, n_participants=None):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=False)
        self.ln_msg = nn.LayerNorm(dim) if ln_msg else None

        xavier_alpha = (1 / dim) ** 0.5
        # learnable identities "bias" to distinguish self-attention participants
        self.ids = None
        if n_participants is not None:
            # (col, batch, dim)
            self.ids = nn.Parameter(torch.empty(2, n_participants, 1, dim))
            # init them with different near-zero vectors
            nn.init.normal_(self.ids, 0.0, 0.01 * xavier_alpha)

        # Set very small out_proj to make the initial "message" negligible
        nn.init.normal_(self.mha.out_proj.weight, 0.0, 0.01 * xavier_alpha)
        nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, q, k, v, return_weights: bool = False):
        # qkv: (C, B, D)
        if self.ids is not None:
            q = q + self.ids[0]
            k = k + self.ids[1]

        msg, attn_w = self.mha(q, k, v, need_weights=return_weights, average_attn_weights=True)
        if self.ln_msg is not None:
            msg = self.ln_msg(msg)

        info = {}
        if return_weights:
            info['attn_weights'] = attn_w.detach().mean(dim=0)
        return msg, info


class MessagePassingLayer1(nn.Module):
    """Compared to naive version, has an "self-to-self" bias for communication init."""
    def __init__(self, dim, num_heads, ln_msg=True, n_participants=None):
        super().__init__()
        self.dim = dim
        self.mha = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=False)
        self.ln_msg = nn.LayerNorm(dim) if ln_msg else None

        # Learnable identities distinguish communication participants.
        # (q/k, C, B, D)
        self.ids = nn.Parameter(torch.empty(2, n_participants, 1, dim)) if n_participants is not None else None

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        H = self.dim
        xavier_alpha = (1 / H) ** 0.5
        near_zero_xavier_alpha = 0.01 * xavier_alpha

        W_q, W_k, W_v = self.mha.in_proj_weight.split(H, dim=0)

        # Same projection for Q and K -> initial content-based self preference.
        nn.init.orthogonal_(W_q, gain=xavier_alpha)
        W_k.copy_(W_q)
        # Preserve the selected token's representation.
        nn.init.eye_(W_v)

        # Near-identity final feature projection.
        nn.init.eye_(self.mha.out_proj.weight)
        self.mha.out_proj.weight.add_(
            torch.randn_like(self.mha.out_proj.weight) * near_zero_xavier_alpha
        )
        if self.mha.in_proj_bias is not None:
            nn.init.zeros_(self.mha.in_proj_bias)
        if self.mha.out_proj.bias is not None:
            nn.init.zeros_(self.mha.out_proj.bias)

        if self.ids is not None:
            # init them with different near-zero vectors
            nn.init.normal_(self.ids, 0.0, near_zero_xavier_alpha)

    def forward(self, q, k, v, return_weights: bool = False):
        # qkv: (C, B, D)
        if self.ids is not None:
            q = q + self.ids[0]
            k = k + self.ids[1]

        msg, attn_w = self.mha(q, k, v, need_weights=return_weights, average_attn_weights=True)
        if self.ln_msg is not None:
            msg = self.ln_msg(msg)

        info = {}
        if return_weights:
            info['attn_weights'] = attn_w.detach().mean(dim=0)
        return msg, info


class StochasticMessagePassingLayer(nn.Module):
    """MHA-1 with stochastic routing and a free-cost diagonal route."""
    def __init__(
            self, dim, num_heads, ln_msg=True, n_participants=None,
            noise_std=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.noise_std = noise_std
        self.mha = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=False)
        self.ln_msg = nn.LayerNorm(dim) if ln_msg else None

        # Learnable identities distinguish communication participants.
        # (q/k, C, B, D)
        self.ids = nn.Parameter(torch.empty(2, n_participants, 1, dim)) if n_participants is not None else None

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        H = self.dim
        xavier_alpha = (1 / H) ** 0.5
        near_zero_xavier_alpha = 0.01 * xavier_alpha

        W_q, W_k, W_v = self.mha.in_proj_weight.split(H, dim=0)

        # Match MHA-1's initial content-based self preference.
        nn.init.orthogonal_(W_q, gain=xavier_alpha)
        W_k.copy_(W_q)
        nn.init.eye_(W_v)

        nn.init.eye_(self.mha.out_proj.weight)
        self.mha.out_proj.weight.add_(
            torch.randn_like(self.mha.out_proj.weight) * near_zero_xavier_alpha
        )
        if self.mha.in_proj_bias is not None:
            nn.init.zeros_(self.mha.in_proj_bias)
        if self.mha.out_proj.bias is not None:
            nn.init.zeros_(self.mha.out_proj.bias)

        if self.ids is not None:
            nn.init.normal_(self.ids, 0.0, near_zero_xavier_alpha)

    def forward(self, q, k, v, return_weights: bool = False):
        # qkv: (C, B, D)
        C, B, H = q.shape
        if self.ids is not None:
            q = q + self.ids[0]
            k = k + self.ids[1]

        W_q, W_k, W_v = self.mha.in_proj_weight.split(H, dim=0)
        b_q, b_k, b_v = self.mha.in_proj_bias.split(H, dim=0)

        q = F.linear(q, W_q, b_q)
        k = F.linear(k, W_k, b_k)
        v = F.linear(v, W_v, b_v)

        # (C, B, H) -> (B, heads, C, head_dim)
        q, k, v = map(self.split_heads, (q, k, v))

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.training and self.noise_std > 0.0:
            logits = logits + self.noise_std * torch.randn_like(logits)
        pi_route = torch.softmax(logits, dim=-1)

        msg = torch.matmul(pi_route, v)
        # back to (C, B, H)
        msg = msg.permute(2, 0, 1, 3).reshape(C, B, H)
        msg = F.linear(msg, self.mha.out_proj.weight, self.mha.out_proj.bias)
        if self.ln_msg is not None:
            msg = self.ln_msg(msg)

        info = {}
        if self.training:
            prob_comm = 1.0 - pi_route.diagonal(dim1=-2, dim2=-1)
            entropy = -(pi_route * torch.log(pi_route.clamp_min(torch.finfo(pi_route.dtype).tiny))).sum(dim=-1)
            info |= {
                'comm_loss': prob_comm.mean(),
                'comm_entropy': normalize_entropy(entropy.mean(), C),
            }
        if return_weights:
            info['attn_weights'] = pi_route.detach().mean(dim=(0, 1))

        return msg, info

    def split_heads(self, x):
        # (C, B, H) -> (B, heads, C, head_dim)
        C, B, _ = x.shape
        return x.view(C, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)


class GruBank2(nn.Module):
    """
    LxC independent GRU cells evaluated together layer by layer.
    Replace a porion of bmms (rz) with less batched larger mms (xh concat over feature dim).
    It has lesser intermediate memory usage, but more copying (two concats).
    """

    def __init__(self, n_layers: int, n_columns: int, hidden_size: int, *, bias: bool = True):
        super().__init__()

        self.n_layers = n_layers
        self.n_columns = n_columns
        self.hidden_size = hidden_size
        self.use_bias = bias

        # Stored in the orientation directly consumed by bmm:
        # [C, B, H] @ [C, H, 3H] -> [C, B, 3H]
        self.weight_rz = nn.Parameter(torch.empty(n_layers, n_columns, 2 * hidden_size, 2 * hidden_size))
        self.weight_n = nn.Parameter(torch.empty(n_layers, n_columns, 2 * hidden_size, hidden_size))
        if self.use_bias:
            # Singleton batch dimension allows direct broadcasting.
            self.bias_rz = nn.Parameter(torch.empty(n_layers, n_columns, 1, 2 * hidden_size))
            self.bias_n = nn.Parameter(torch.empty(n_layers, n_columns, 1, hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Same initialization as nn.GRUCell.
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, layer, x: torch.Tensor, h: torch.Tensor):
        # x: [C, B, H]
        # h: [C, B, H]
        xh = torch.cat((x, h), dim=-1)
        if self.use_bias:
            rz = torch.bmm(xh, self.weight_rz[layer]) + self.bias_rz[layer]
        else:
            rz = torch.bmm(xh, self.weight_rz[layer])

        rg, ug = torch.sigmoid(rz).chunk(2, dim=-1)

        xr = torch.cat((x, rg * h), dim=-1)
        if self.use_bias:
            pre_ng = torch.bmm(xr, self.weight_n[layer]) + self.bias_n[layer]
        else:
            pre_ng = torch.bmm(xr, self.weight_n[layer])

        ng = torch.tanh(pre_ng)

        # [C, B, H]
        return (h - ng) * ug + ng


class GruBank1(nn.Module):
    """
    LxC independent GRU cells evaluated together layer by layer.
    Compared to the naive impl, in/h weights are merged along bmm batch dim. 
    Exchanges two separate bmm calls with [x, h] concat + single batched bmm.
    """

    def __init__(self, n_layers: int, n_columns: int, hidden_size: int, *, bias: bool = True):
        super().__init__()

        self.n_layers = n_layers
        self.n_columns = n_columns
        self.hidden_size = hidden_size
        self.use_bias = bias

        # Stored in the orientation directly consumed by bmm:
        # [C, B, H] @ [C, H, 3H] -> [C, B, 3H]
        self.weight = nn.Parameter(torch.empty(n_layers, 2 * n_columns, hidden_size, 3 * hidden_size))
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(n_layers, 2 * n_columns, 1, 3 * hidden_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Same initialization as nn.GRUCell.
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, layer, x: torch.Tensor, h: torch.Tensor):
        # x: [C, B, H]
        # h: [C, B, H]
        C, B, H = x.shape
        xh = torch.cat((x, h), dim=0)
        if self.use_bias:
            gates = torch.bmm(xh, self.weight[layer]) + self.bias[layer]
        else:
            gates = torch.bmm(xh, self.weight[layer])
        
        input_gates, hidden_gates = gates.view(2, C, B, 3 * H).unbind(0)

        # PyTorch gate order: reset, update, new.
        x_r, x_z, x_n = input_gates.chunk(3, dim=-1)
        h_r, h_z, h_n = hidden_gates.chunk(3, dim=-1)

        reset_gate = torch.sigmoid(x_r + h_r)
        update_gate = torch.sigmoid(x_z + h_z)
        new_gate = torch.tanh(x_n + reset_gate * h_n)

        # [C, B, hidden_size]
        return (h - new_gate) * update_gate + new_gate


class GruBank(nn.Module):
    """
    LxC independent GRU cells evaluated together layer by layer.
    The most naively implemented bmm generalization of gru cell.
    """

    def __init__(self, n_layers: int, n_columns: int, hidden_size: int, *, bias: bool = True):
        super().__init__()

        self.n_layers = n_layers
        self.n_columns = n_columns
        self.hidden_size = hidden_size
        self.use_bias = bias

        # Stored in the orientation directly consumed by bmm:
        # [C, B, H] @ [C, H, 3H] -> [C, B, 3H]
        self.weight_ih = nn.Parameter(torch.empty(n_layers, n_columns, hidden_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(n_layers, n_columns, hidden_size, 3 * hidden_size))
        if self.use_bias:
            # Singleton batch dimension allows direct broadcasting.
            self.bias_ih = nn.Parameter(torch.empty(n_layers, n_columns, 1, 3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(n_layers, n_columns, 1, 3 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Same initialization as nn.GRUCell.
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, layer, x: torch.Tensor, h: torch.Tensor):
        # x: [C, B, H]
        # h: [C, B, H]
        if self.use_bias:
            input_gates = torch.bmm(x, self.weight_ih[layer]) + self.bias_ih[layer]
            hidden_gates = torch.bmm(h, self.weight_hh[layer]) + self.bias_hh[layer]
        else:
            input_gates = torch.bmm(x, self.weight_ih[layer])
            hidden_gates = torch.bmm(h, self.weight_hh[layer])

        # PyTorch gate order: reset, update, new.
        x_r, x_z, x_n = input_gates.chunk(3, dim=-1)
        h_r, h_z, h_n = hidden_gates.chunk(3, dim=-1)

        reset_gate = torch.sigmoid(x_r + h_r)
        update_gate = torch.sigmoid(x_z + h_z)
        new_gate = torch.tanh(x_n + reset_gate * h_n)

        # [C, B, hidden_size]
        return (h - new_gate) * update_gate + new_gate
