from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.grnn import MessagePassingLayer
from knitwork.models.engram import EngramMemory, EngramState


class EngramGridRnn(nn.Module):
    """GridRNN with per-cell Engram associative memory."""

    def __init__(
        self,
        *,
        input_size: int,
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int,
        n_columns: int,
        n_attn_heads: int,
        messaging: str = "post",
        col_identities: bool,
        use_bias: bool = True,
        dropout: float = 0.0,
        n_engram_slots: int = 16,
        engram_top_k: int = 4,
        engram_hebb_lr: float = 0.1,
        engram_gate_write: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, self.embedding_size)

        self.n_layers = n_layers
        assert n_columns > 1
        self.n_columns = n_columns
        self.n_attn_heads = n_attn_heads
        self.n_engram_slots = n_engram_slots

        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        print(
            f'EngramGridRNN of {n_layers}L x {n_columns}C GRU cells'
            f' w/ {self.hidden_size} hidden units'
            f' + {n_engram_slots} engram slots per cell'
            f' (top-{engram_top_k} sparse retrieval)'
        )

        self.use_postmsg = (messaging == "post")

        self.cells = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        self.engrams = nn.ModuleList()

        for layer in range(n_layers):
            self.cells.append(nn.ModuleList([
                nn.GRUCell(
                    input_size=self._cell_input_dim(layer, icol),
                    hidden_size=self.hidden_size,
                    bias=use_bias,
                    dtype=torch.float64,
                )
                for icol in range(n_columns)
            ]))

            n_participants = n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(
                self.hidden_size, num_heads=n_attn_heads, n_participants=n_participants,
            ))

            if self.use_postmsg:
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

            self.engrams.append(nn.ModuleList([
                EngramMemory(
                    hidden_size=self.hidden_size,
                    n_slots=n_engram_slots,
                    top_k=engram_top_k,
                    hebb_lr=engram_hebb_lr,
                    gate_write=engram_gate_write,
                    dtype=torch.float64,
                )
                for _ in range(n_columns)
            ]))

        self.head = nn.Linear(self.hidden_size, self.output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Param count: {format_readable_num(param_count)}')

    def forward(
        self,
        tokens: torch.Tensor,
        state: Optional[EngramState] = None,
        return_attn: bool = False,
    ) -> tuple:
        tokens = to_torch(tokens)
        assert tokens.ndim == 2
        bsz = tokens.shape[0]

        x = self.embedding(tokens.view(-1))  # [B, E]

        if state is None:
            state = self.init_state(bsz, x.device, x.dtype)

        if self.use_postmsg:
            new_state, extras = self.grid_step_postmsg(x, state=state, return_attn=return_attn)
        else:
            new_state, extras = self.grid_step_premsg(x, state=state), {}

        z = new_state.h[-1][0]  # top layer, first col: [B, H]
        y = self.head(z)

        if return_attn:
            return y, new_state, extras
        return y, new_state

    def grid_step_postmsg(
        self,
        x: torch.Tensor,
        *,
        state: EngramState,
        return_attn: bool = True,
    ) -> tuple[EngramState, dict]:
        h = state.h   # [layers, cols, B, H]
        M = state.M

        h_n_layers, M_n_layers = [], []
        attn_list, gate_list, engram_attn_list = [], [], []

        x_input = self._prepare_grid_input(x)

        for layer_i, (cells, attn_msg, attn_gate, engram_row) in enumerate(
            zip(self.cells, self.attn, self.attn_gates, self.engrams)
        ):
            hl = h[layer_i]   # [cols, B, H]
            Ml = M[layer_i]

            hl_n_cols, Ml_n_cols, layer_engram_attn = [], [], []

            for col_i in range(self.n_columns):
                h_prev = hl[col_i]   # [B, H]
                M_prev = Ml[col_i]   # [B, S, H]

                r, M_new, eng_attn = engram_row[col_i](h_prev, M_prev)
                x_aug = torch.cat([x_input[col_i], r], dim=-1)  # [B, input_dim+H]
                h_new = cells[col_i](x_aug, h_prev)             # [B, H]

                hl_n_cols.append(h_new)
                Ml_n_cols.append(M_new)
                layer_engram_attn.append(eng_attn)

            hl_n = torch.stack(hl_n_cols, dim=0)  # [cols, B, H]

            msg, attn_w = attn_msg(hl_n, return_weights=return_attn)
            g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
            hl_n = (1.0 - g) * hl_n + g * msg

            h_n_layers.append(hl_n)
            M_n_layers.append(Ml_n_cols)
            attn_list.append(attn_w)
            gate_list.append(g)
            engram_attn_list.append(layer_engram_attn)

            x_input = hl_n

        new_state = EngramState(h=torch.stack(h_n_layers, dim=0), M=M_n_layers)
        extras = {
            'attn_weights': attn_list,
            'gates':        gate_list,
            'engram_attn':  engram_attn_list,  # [layer][col] -> [B, n_slots]
            'h_layers':     h_n_layers,
        }
        return new_state, extras

    def grid_step_premsg(
        self,
        x: torch.Tensor,
        *,
        state: EngramState,
    ) -> EngramState:
        h = state.h
        M = state.M

        h_n_layers, M_n_layers = [], []
        x_input = self._prepare_grid_input(x)

        for layer_i, (cells, attn_msg, engram_row) in enumerate(
            zip(self.cells, self.attn, self.engrams)
        ):
            hl = h[layer_i]
            Ml = M[layer_i]

            msg, _ = attn_msg(hl, return_weights=False)

            if layer_i == 0:
                # x_input is a list
                x_aug_list = [torch.cat([xc, msgc], dim=-1) for xc, msgc in zip(x_input, msg)]
            else:
                # x_input is a contiguous tensor [cols, B, H]
                x_aug_list = [torch.cat([x_input[c], msg[c]], dim=-1) for c in range(self.n_columns)]

            hl_n_cols, Ml_n_cols = [], []

            for col_i in range(self.n_columns):
                h_prev = hl[col_i]
                M_prev = Ml[col_i]

                r, M_new, _ = engram_row[col_i](h_prev, M_prev)
                x_col_r = torch.cat([x_aug_list[col_i], r], dim=-1)
                h_new = cells[col_i](x_col_r, h_prev)

                hl_n_cols.append(h_new)
                Ml_n_cols.append(M_new)

            hl_n = torch.stack(hl_n_cols, dim=0)
            h_n_layers.append(hl_n)
            M_n_layers.append(Ml_n_cols)
            x_input = hl_n

        return EngramState(h=torch.stack(h_n_layers, dim=0), M=M_n_layers)

    def init_state(self, bsz: int, device, dtype) -> EngramState:
        h = torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=device, dtype=dtype,
        )
        M = [
            [engram.init_memory(bsz, device, dtype) for engram in layer_engrams]
            for layer_engrams in self.engrams
        ]
        return EngramState(h=h, M=M)

    def reset_state(
        self,
        state: Optional[EngramState],
        reset_mask: torch.Tensor,
    ) -> EngramState:
        if state is None:
            return self.init_state(
                reset_mask.shape[0], reset_mask.device, self.head.weight.dtype,
            )

        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state

        h_new = state.h.clone()
        h_new[:, :, ixs, :] = 0.0

        M_new = [
            [M_col.clone().index_fill_(0, ixs, 0.0) for M_col in layer_M]
            for layer_M in state.M
        ]
        return EngramState(h=h_new, M=M_new)

    def detach_state(self, state: Optional[EngramState]) -> Optional[EngramState]:
        if state is None:
            return None
        return EngramState(
            h=state.h.detach(),
            M=[[M_col.detach() for M_col in layer_M] for layer_M in state.M],
        )

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        # GRU input = [x_col ; engram_retrieval r], so +hidden_size vs base
        if ix_layer == 0:
            base = self.embedding_size if ix_col == 0 else 1
        else:
            base = self.hidden_size
            if not self.use_postmsg:
                base += self.hidden_size  # pre-msg concatenates msg too
        return base + self.hidden_size

    def _prepare_grid_input(self, x: torch.Tensor) -> list:
        bsz, _ = x.shape
        dummy = torch.zeros(bsz, 1, device=x.device, dtype=x.dtype)
        return [x] + [dummy] * (self.n_columns - 1)
