from __future__ import annotations

import torch
from torch import nn

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.grnn import MessagePassingLayer


class EmbeddingVAE(nn.Module):
    """VAE bottleneck over token embeddings; returns latent vector + KL loss."""

    def __init__(self, vocab_size: int, embed_dim: int,
                 latent_dim: int, kl_weight: float = 1e-3):
        super().__init__()
        self.embed_dim  = embed_dim
        self.latent_dim = latent_dim
        self.kl_weight  = kl_weight

        self.embedding  = nn.Embedding(vocab_size, embed_dim)
        self.fc_mu      = nn.Linear(embed_dim, latent_dim)
        self.fc_log_var = nn.Linear(embed_dim, latent_dim)
        self.fc_decode  = nn.Linear(latent_dim, embed_dim)

        nn.init.normal_(self.fc_decode.weight, 0.0, 0.01)
        nn.init.zeros_(self.fc_decode.bias)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # reparameterization trick; deterministic at eval
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        return mu

    def forward(self, token_ids: torch.Tensor):
        e       = self.embedding(token_ids)        # [batch, embed_dim]
        mu      = self.fc_mu(e)                    # [batch, latent_dim]
        log_var = self.fc_log_var(e)               # [batch, latent_dim]
        z       = self.reparameterize(mu, log_var) # [batch, latent_dim]
        x       = self.fc_decode(z)                # [batch, embed_dim]
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return x, kl * self.kl_weight


class ColumnDelayGate(nn.Module):
    """Mix each column j with its left neighbour j-1 from the previous time step."""

    def __init__(self, hidden_size: int, n_columns: int, delay_scale: float = -2.0):
        super().__init__()
        self.n_columns = n_columns
        # one gate per column except column 0
        self.gates = nn.ModuleList([
            nn.Linear(2 * hidden_size, hidden_size)
            for _ in range(n_columns - 1)
        ])
        for g in self.gates:
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, delay_scale)

    def forward(self,
                h_new: torch.Tensor,   # [cols, batch, hidden]
                h_prev: torch.Tensor,  # [cols, batch, hidden]
                ) -> torch.Tensor:
        cols = [h_new[0]]
        for j in range(1, self.n_columns):
            combined = torch.cat([h_new[j], h_prev[j - 1]], dim=-1)
            g = torch.sigmoid(self.gates[j - 1](combined))
            cols.append((1.0 - g) * h_new[j] + g * h_prev[j - 1])
        return torch.stack(cols, dim=0)  # [cols, batch, hidden]


class ColumnDropout(nn.Module):
    """Drop entire columns with probability drop_prob during training."""

    def __init__(self, n_columns: int, drop_prob: float = 0.1, keep_first: bool = True):
        super().__init__()
        self.n_columns  = n_columns
        self.drop_prob  = drop_prob
        self.keep_first = keep_first

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [cols, batch, hidden]
        if not self.training or self.drop_prob == 0.0:
            return h
        start = 1 if self.keep_first else 0
        keep  = torch.rand(self.n_columns - start, device=h.device) > self.drop_prob
        scale = 1.0 / (1.0 - self.drop_prob + 1e-8)
        result = h.clone()
        for i, col_idx in enumerate(range(start, self.n_columns)):
            result[col_idx] = 0.0 if not keep[i] else result[col_idx] * scale
        return result


# ---
class GridRnn2(nn.Module):
    def __init__(
            self, *,
            input_size, embedding_size, output_size,
            hidden_size: int,
            n_layers: int, n_columns: int,
            n_attn_heads, messaging: str = "post", col_identities,
            use_bias=True, dropout=0.0,
            vae_latent_dim: int | None = None,
            vae_kl_weight: float = 1e-3,
            use_time_gate: bool = True,
            time_gate_delay_scale: float = -2.0,
            col_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.input_size     = input_size
        self.embedding_size = embedding_size
        self.output_size    = output_size
        self.n_layers       = n_layers
        self.n_columns      = n_columns
        self.n_attn_heads   = n_attn_heads

        assert n_columns > 1

        self.hidden_size = hidden_size - hidden_size % n_attn_heads
        self.use_postmsg = messaging == "post"

        # embedding: VAE or plain
        self.use_vae = vae_latent_dim is not None
        if self.use_vae:
            self.embedding_vae = EmbeddingVAE(
                vocab_size=input_size,
                embed_dim=embedding_size,
                latent_dim=vae_latent_dim,
                kl_weight=vae_kl_weight,
            )
        else:
            self.embedding = nn.Embedding(input_size, embedding_size)

        # grid of GRU cells + attention layers
        self.cells      = nn.ModuleList()
        self.attn       = nn.ModuleList()
        self.attn_gates = nn.ModuleList()

        for layer in range(self.n_layers):
            row = (
                nn.GRUCell(
                    input_size=self._cell_input_dim(layer, icol),
                    hidden_size=self.hidden_size,
                    bias=use_bias,
                    dtype=torch.float64,
                )
                for icol in range(self.n_columns)
            )
            self.cells.append(nn.ModuleList(row))

            n_participants = self.n_columns if col_identities else None
            self.attn.append(MessagePassingLayer(
                self.hidden_size, num_heads=self.n_attn_heads, n_participants=n_participants,
            ))
            if self.use_postmsg:
                self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1))

        # column time-gate (one per layer)
        self.use_time_gate = use_time_gate
        self.time_gates = nn.ModuleList([
            ColumnDelayGate(self.hidden_size, self.n_columns, time_gate_delay_scale)
            for _ in range(self.n_layers)
        ]) if use_time_gate else None

        self.col_dropout = ColumnDropout(self.n_columns, col_drop_prob, keep_first=True)
        self.head = nn.Linear(self.hidden_size, self.output_size)

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f'GridRNN2: {self.n_layers}L x {self.n_columns}C,'
            f' hidden={self.hidden_size}'
            f' | params={format_readable_num(param_count)}'
        )

    def forward(self, tokens: torch.Tensor, h=None, return_attn=False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2 and tokens.shape[1] == 1

        kl_loss = torch.tensor(0.0, dtype=self.head.weight.dtype, device=self.head.weight.device)
        if self.use_vae:
            x, kl_loss = self.embedding_vae(tokens.view(-1))
        else:
            x = self.embedding(tokens.view(-1))  # [batch, embed_dim]

        if self.use_postmsg:
            h, extras = self._grid_step_postmsg(x, h=h, return_attn=return_attn)
        else:
            h, extras = self._grid_step_premsg(x, h=h), {}

        z = h[-1][0]
        y = self.head(z)

        if return_attn:
            return y, h, extras, kl_loss
        return y, h, kl_loss

    def _grid_step_postmsg(self, x, *, h, return_attn=True):
        h_n, attn_list, gate_list = [], [], []
        x = self._prepare_grid_input(x)

        for layer_idx, (cells, attn, attn_gate, hl) in enumerate(
                zip(self.cells, self.attn, self.attn_gates, h)):

            hl_n = torch.stack([
                self._cell_forward(cells, x, hl, ix_col=c)
                for c in range(self.n_columns)
            ], dim=0)  # [cols, batch, hidden]

            msg, attn_w = attn(hl_n, return_weights=return_attn)
            g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
            hl_n = (1 - g) * hl_n + g * msg

            if self.use_time_gate and self.time_gates is not None:
                hl_n = self.time_gates[layer_idx](hl_n, hl)

            hl_n = self.col_dropout(hl_n)
            h_n.append(hl_n)
            attn_list.append(attn_w)
            gate_list.append(g)
            x = hl_n

        h_n = torch.stack(h_n, dim=0)  # [layers, cols, batch, hidden]
        return h_n, {"attn_weights": attn_list, "gates": gate_list}

    def _grid_step_premsg(self, x, *, h):
        h_n = []
        x = self._prepare_grid_input(x)
        first_row = True

        for layer_idx, (cells, attn, hl) in enumerate(zip(self.cells, self.attn, h)):
            msg, _ = attn(hl, return_weights=False)
            if first_row:
                x = [torch.cat([xc, msgc], -1) for xc, msgc in zip(x, msg)]
            else:
                x = torch.cat([x, msg], dim=-1)

            hl_n = torch.stack([
                self._cell_forward(cells, x, hl, ix_col=c)
                for c in range(self.n_columns)
            ], dim=0)

            if self.use_time_gate and self.time_gates is not None:
                hl_n = self.time_gates[layer_idx](hl_n, hl)
            hl_n = self.col_dropout(hl_n)

            h_n.append(hl_n)
            x = hl_n
            first_row = False

        return torch.stack(h_n, dim=0)

    def _cell_forward(self, cells, x, h, *, ix_col):
        return cells[ix_col](x[ix_col], h[ix_col])

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

    def _cell_input_dim(self, ix_layer: int, ix_col: int) -> int:
        if ix_layer == 0:
            return self.embedding_size if ix_col == 0 else 1
        hsz = self.hidden_size
        if not self.use_postmsg:
            hsz += self.hidden_size
        return hsz

    def _prepare_grid_input(self, x):
        xl = [x]
        bsz, _ = x.shape
        in_dim = self._cell_input_dim(ix_layer=0, ix_col=1)
        dummy = torch.zeros(bsz, in_dim, device=x.device, dtype=x.dtype)
        for _ in range(1, self.n_columns):
            xl.append(dummy)
        return xl

    def init_state(self, bsz):
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device,
            dtype=self.head.weight.dtype,
        )
