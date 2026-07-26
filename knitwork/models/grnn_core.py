from __future__ import annotations
import math

import torch
from torch import nn

from knitwork.common.utils import format_readable_num


class GridRnn(nn.Module):
    has_attn = True

    def __init__(
            self, *,
            hidden_size, n_layers: int, n_columns: int,
            n_inputs: int = 1, n_outputs: int = 1,
            n_attn_heads, ln_msg=True, use_bias=True,
            self_feeding: bool = False,
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

        self.dtype = dtype
        self.device = device

        # Hidden size should be a multiply of the n_attn_heads
        self.hidden_size -= self.hidden_size % self.n_attn_heads
        print(
            f'GridRNN of {self.n_layers}L x {self.n_columns}C GRU cells'
            f' w/ {self.hidden_size} hidden units'
        )

        # Build a grid of cells: layers x columns
        # self.cells = GruBank(self.n_layers, self.n_columns, self.hidden_size, bias=use_bias)
        # self.cells = GruBank2(self.n_layers, self.n_columns, self.hidden_size, bias=use_bias)
        self.cells = GruBank3(self.n_layers, self.n_columns, self.hidden_size, bias=use_bias)

        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        for layer in range(self.n_layers):
            self.attn.append(MessagePassingLayer(
                self.hidden_size, num_heads=self.n_attn_heads, ln_msg=ln_msg, n_participants=self.n_columns
            ))
            self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

    def forward(self, x: torch.Tensor, state: dict, *, capture=False, **_):
        # x shape: (In, B, H)
        assert x.shape[0] == self.n_inputs
        # h shape: (L, C, B, H)
        h = state['h']
        h_new = []

        if capture:
            attn_ws, gate_vs = [], []
        # (C, B, H)
        x = self._prepare_grid_input(x, h)

        for layer in range(self.n_layers):
            hl_n = self.cells(layer, x, h[layer])

            msg, attn_w = self.attn[layer](hl_n, hl_n, hl_n, return_weights=capture)
            g = torch.sigmoid(self.attn_gates[layer](
                torch.cat([hl_n, msg], dim=-1)
            ))
            hl_n = torch.lerp(hl_n, msg, g)

            h_new.append(hl_n)
            if capture:
                attn_ws.append(attn_w)
                gate_vs.append(g.detach())
            x = hl_n

        h_new = torch.stack(h_new, dim=0)

        # top (=last) layer, first col as grid output
        y = h_new[-1][0]
        state = {'h': h_new}
        info = {}
        if capture:
            info |= {"attn_weights": attn_ws, "gates": gate_vs}

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

    def _prepare_grid_input(self, x: torch.Tensor, h: torch.Tensor):
        if self.self_feeding:
            internal_input = h[-1, :, self.n_inputs:]
        else:
            n_internal_cols = self.n_columns - self.n_inputs
            internal_input = x.new_zeros(n_internal_cols, *x.shape[1:])

        # (col, batch, features), cat over cols
        x = torch.cat([x, internal_input], dim=0)
        return x

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)

        keep = (~reset_mask.flatten())[None, None, :, None]
        h = state['h'] * keep
        return {'h': h}

    def detach_state(self, state):
        if state is None:
            return state
        return {'h': state['h'].detach()}

    def init_state(self, bsz):
        h = torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.device, dtype=self.dtype
        )
        return {'h': h}


class MessagePassingLayer(nn.Module):
    def __init__(self, dim, num_heads, ln_msg=True, n_participants=None):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=False)
        self.ln_msg = nn.LayerNorm(dim) if ln_msg else None

        xavier_alpha = (1 / dim) ** 0.5
        # learnable identities "bias" to distinguish self-attention participants
        self.ids = None
        if n_participants is not None:
            # (col, batch, dim)
            self.ids = nn.Parameter(torch.empty(n_participants, 1, dim))
            # init them with different near-zero vectors
            nn.init.normal_(self.ids, 0.0, 0.01 * xavier_alpha)

        # Set very small out_proj to make the initial "message" negligible
        nn.init.normal_(self.mha.out_proj.weight, 0.0, 0.01 * xavier_alpha)
        nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, q, k, v, return_weights: bool = False):
        # qkv: (C, B, D)
        if self.ids is not None:
            q = q + self.ids
            k = k + self.ids

        msg, attn_w = self.mha(q, k, v, need_weights=return_weights, average_attn_weights=True)

        # Layer norm ensures we are in a good range
        if return_weights:
            attn_w = attn_w.detach().mean(dim=0)
        if self.ln_msg is not None:
            msg = self.ln_msg(msg)
        return msg, attn_w


class GruBank2(nn.Module):
    """LxC independent GRU cells evaluated together layer by layer."""

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
    """LxC independent GRU cells evaluated together layer by layer."""

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


class GruBank3(nn.Module):
    """LxC independent GRU cells evaluated together layer by layer."""

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

        # [C, B, hidden_size]
        return (h - ng) * ug + ng
