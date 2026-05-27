from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass, field
from typing import Optional

from knitwork.common.utils import format_readable_num, to_torch
from knitwork.models.fusion_cells import HGRUCell, BatchedHGRUColumns, BatchedReservoirColumns


# --- sub-configs ---

@dataclass
class HGRNConfig:
    enabled: bool = True
    beta_min: float = 0.01
    beta_max: float = 0.95
    use_final_output_gate: bool = True
    learnable_beta: bool = True


@dataclass
class ReservoirConfig:
    enabled: bool = True
    n_reservoir_cols: int = 2
    spectral_radius: float = 0.9
    reservoir_scale: float = 0.1
    spectral_radii: Optional[list] = None   # if None — all equal spectral_radius


@dataclass
class DiversityLossConfig:
    enabled: bool = True
    total_weight: float = 0.05
    cosine_weight: float = 0.3
    cov_weight: float = 0.2
    var_weight: float = 0.1
    gate_entropy_weight: float = 0.05
    layer_weights: Optional[list] = None
    cosine_margin: float = 0.5
    var_threshold: float = 0.1
    compute_every_n: int = 1


@dataclass
class FusionConfig:
    embedding_size: int = 64
    hidden_size: int = 192
    n_layers: int = 5
    n_columns: int = 5
    n_attn_heads: int = 4
    messaging: str = "post"
    use_bias: bool = True
    dropout: float = 0.0
    col_identities: bool = True
    all_cols_get_input: bool = True   # all columns see input via separate projections

    hgrn: HGRNConfig = field(default_factory=HGRNConfig)
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    diversity_loss: DiversityLossConfig = field(default_factory=DiversityLossConfig)

    use_cross_attention: bool = True


# --- message passing with residual ---

class MessagePassingLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, n_participants: Optional[int] = None):
        super().__init__()
        self.mha  = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=False)
        self.norm = nn.LayerNorm(dim)
        alpha = (1 / dim) ** 0.5
        self.ids = None
        if n_participants is not None:
            self.ids = nn.Parameter(torch.empty(n_participants, 1, dim))
            nn.init.normal_(self.ids, 0.0, 0.01 * alpha)
        nn.init.normal_(self.mha.out_proj.weight, 0.0, 0.01 * alpha)
        nn.init.zeros_(self.mha.out_proj.bias)

    def forward(
        self,
        h: torch.Tensor,           # (cols, batch, hidden)
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        qh = kh = (h + self.ids) if self.ids is not None else h
        h_mixed, attn_w = self.mha(qh, kh, h, average_attn_weights=True)
        if return_weights and attn_w is not None:
            attn_w = attn_w.mean(dim=0)
        return self.norm(h + h_mixed), attn_w


# --- cross-attention: trainable columns read from reservoir ---

class TrainableReservoirCrossAttention(nn.Module):
    def __init__(self, hidden_size, n_trainable, n_reservoir, num_heads=2, dtype=torch.float64):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True, dtype=dtype
        )
        self.norm = nn.LayerNorm(hidden_size, dtype=dtype)
        self.gate = nn.Linear(hidden_size * 2, 1, dtype=dtype)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h_trainable, h_reservoir):
        # h_trainable: (batch, n_trainable, hidden)
        # h_reservoir: (batch, n_reservoir, hidden)
        read, _ = self.cross_attn(h_trainable, h_reservoir, h_reservoir)
        g = torch.sigmoid(self.gate(torch.cat([h_trainable, read], dim=-1)))  # (batch, n_t, 1)
        return self.norm(h_trainable + g * read)


# --- diversity loss ---

class ColumnDiversityLoss(nn.Module):
    def __init__(self, cfg: DiversityLossConfig, n_layers: int):
        super().__init__()
        self.cfg = cfg
        if cfg.layer_weights is not None:
            lw = torch.tensor(cfg.layer_weights, dtype=torch.float32)
        else:
            lw = torch.tensor(
                [0.5 + 1.5 * i / max(n_layers - 1, 1) for i in range(n_layers)],
                dtype=torch.float32,
            )
        self.register_buffer('layer_weights', lw)

    def cosine_diversity(self, h: torch.Tensor, lw: float) -> torch.Tensor:
        cols = h.shape[0]
        h_norm = F.normalize(h.float(), dim=-1)
        loss, count = h.new_zeros(()), 0
        for i in range(cols):
            for j in range(i + 1, cols):
                sim = (h_norm[i] * h_norm[j]).sum(dim=-1).mean()
                loss = loss + F.relu(sim - self.cfg.cosine_margin)
                count += 1
        return lw * self.cfg.cosine_weight * (loss / max(count, 1))

    def covariance_diversity(self, h: torch.Tensor, lw: float) -> torch.Tensor:
        cols, bsz, d = h.shape
        loss = h.new_zeros(())
        eye  = torch.eye(d, device=h.device, dtype=h.dtype)
        for c in range(cols):
            z   = h[c].float()
            z   = z - z.mean(dim=0, keepdim=True)
            cov = (z.T @ z) / max(bsz - 1, 1)
            loss = loss + (cov * (1.0 - eye)).pow(2).sum() / d
        return lw * self.cfg.cov_weight * (loss / cols)

    def variance_loss(self, h: torch.Tensor, lw: float) -> torch.Tensor:
        cols = h.shape[0]
        loss = h.new_zeros(())
        for c in range(cols):
            std  = h[c].float().var(dim=0).clamp(min=0.0).sqrt()
            loss = loss + F.relu(self.cfg.var_threshold - std).mean()
        return lw * self.cfg.var_weight * (loss / cols)

    def gate_entropy_loss(self, gate_logits: list[torch.Tensor]) -> torch.Tensor:
        if not gate_logits:
            return torch.tensor(0.0)
        eps   = 1e-6
        total = gate_logits[0].new_zeros(())
        for g_logit in gate_logits:
            g     = torch.sigmoid(g_logit).clamp(eps, 1.0 - eps)
            H     = -(g * g.log() + (1.0 - g) * (1.0 - g).log())
            total = total + (-H.mean())   # maximize gate entropy
        return self.cfg.gate_entropy_weight * total / max(len(gate_logits), 1)

    def forward(
        self,
        h_per_layer: list[torch.Tensor],
        gate_logits: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        dev, dt = h_per_layer[0].device, h_per_layer[0].dtype
        cos_t = torch.zeros((), device=dev, dtype=dt)
        cov_t = torch.zeros((), device=dev, dtype=dt)
        var_t = torch.zeros((), device=dev, dtype=dt)

        for li, h in enumerate(h_per_layer):
            lw    = self.layer_weights[li].item()
            cos_t = cos_t + self.cosine_diversity(h, lw)
            cov_t = cov_t + self.covariance_diversity(h, lw)
            var_t = var_t + self.variance_loss(h, lw)

        gate_t = self.gate_entropy_loss(gate_logits)
        total  = (cos_t + cov_t + var_t + gate_t) * self.cfg.total_weight
        return {
            'cosine': cos_t, 'covariance': cov_t,
            'variance': var_t, 'gate_entropy': gate_t, 'total': total,
        }


# --- reservoir utilization metrics ---

def compute_reservoir_utilization(
    h_res: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    result = {}
    with torch.no_grad():
        h = h_res.float()
        bsz, hid = h.shape

        h_centered = h - h.mean(dim=0, keepdim=True)
        cov = (h_centered.T @ h_centered) / max(bsz - 1, 1)
        try:
            eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
            pr = eigvals.sum().pow(2) / (eigvals.pow(2).sum() + 1e-10)
            result['participation_ratio']      = pr.item()
            result['participation_ratio_norm'] = pr.item() / hid
        except Exception:
            pass

        if baseline is not None:
            diff = (h - baseline.float()).norm(dim=-1).mean()
            base = baseline.float().norm(dim=-1).mean()
            result['input_sensitivity'] = (diff / (base + 1e-10)).item()

        abs_h = h.abs().mean(dim=0)
        abs_h = abs_h / (abs_h.sum() + 1e-10)
        ent   = -(abs_h * (abs_h + 1e-10).log()).sum()
        result['activation_entropy_norm'] = (ent / math.log(hid)).item()

    return result


# --- main model ---

class GridRnnFusion(nn.Module):
    """
    GridRNN Fusion v3: HGRN trainable columns + frozen reservoir + cross-attention + diversity loss.

    Column layout: [trainable_0, ..., trainable_{n-1}, res_0, ..., res_{m-1}]
    """

    def __init__(self, *, input_size: int, output_size: int, cfg: FusionConfig):
        super().__init__()
        self.cfg         = cfg
        self.input_size  = input_size
        self.output_size = output_size

        rcfg  = cfg.reservoir
        n_res = rcfg.n_reservoir_cols if rcfg.enabled else 0
        assert cfg.n_columns > 1
        if rcfg.enabled:
            assert 0 < n_res < cfg.n_columns

        self.n_layers         = cfg.n_layers
        self.n_columns        = cfg.n_columns
        self.n_reservoir_cols = n_res
        self.n_trainable_cols = cfg.n_columns - n_res

        self.embedding   = nn.Embedding(input_size, cfg.embedding_size)
        self.hidden_size = cfg.hidden_size - cfg.hidden_size % cfg.n_attn_heads

        # per-column input projections (orthogonal init)
        self.col_input_projs = nn.ModuleList([
            nn.Linear(cfg.embedding_size, cfg.embedding_size, bias=True, dtype=torch.float64)
            for _ in range(cfg.n_columns)
        ])
        for proj in self.col_input_projs:
            nn.init.orthogonal_(proj.weight)
            nn.init.zeros_(proj.bias)

        # HGRN beta schedule: increases from beta_min (lower layers) to beta_max (upper layers)
        hcfg = cfg.hgrn
        if hcfg.enabled and cfg.n_layers > 1:
            self._layer_betas = [
                [
                    max(hcfg.beta_min,
                        hcfg.beta_min + (hcfg.beta_max - hcfg.beta_min) * li / (cfg.n_layers - 1))
                    for _ in range(self.n_trainable_cols)
                ]
                for li in range(cfg.n_layers)
            ]
        else:
            self._layer_betas = [
                [max(0.01, hcfg.beta_min)] * self.n_trainable_cols
                for _ in range(cfg.n_layers)
            ]

        # reservoir spectral radii: multi-scale memory if not specified
        if rcfg.spectral_radii is not None:
            assert len(rcfg.spectral_radii) == n_res
            sr_list = rcfg.spectral_radii
        elif n_res == 1:
            sr_list = [rcfg.spectral_radius]
        elif n_res == 2:
            sr_list = [0.7, 0.95]
        elif n_res == 3:
            sr_list = [0.6, 0.85, 0.97]
        else:
            sr_list = [0.5 + 0.49 * i / (n_res - 1) for i in range(n_res)]

        print(
            f'[GridRnnFusion v3] {cfg.n_layers}L x {cfg.n_columns}C '
            f'({self.n_trainable_cols} HGRU + {n_res} reservoir) '
            f'| hidden={self.hidden_size} | SR={sr_list}'
        )

        cell_in  = cfg.embedding_size
        inter_in = self.hidden_size

        self.trainable_cells = nn.ModuleList()
        self.reservoir_cells = nn.ModuleList()
        self.cross_attns     = nn.ModuleList() if cfg.use_cross_attention and n_res > 0 else None
        self.attn            = nn.ModuleList()
        self.attn_gates      = nn.ModuleList()

        for li in range(cfg.n_layers):
            in_dim = cell_in if li == 0 else inter_in

            if hcfg.enabled:
                self.trainable_cells.append(BatchedHGRUColumns(
                    n_cols=self.n_trainable_cols,
                    input_size=in_dim,
                    hidden_size=self.hidden_size,
                    beta_inits=self._layer_betas[li],
                    use_bias=cfg.use_bias,
                    learnable_beta=hcfg.learnable_beta,
                    use_layer_norm=True,
                    dtype=torch.float64,
                ))
            else:
                self.trainable_cells.append(BatchedHGRUColumns(
                    n_cols=self.n_trainable_cols,
                    input_size=in_dim,
                    hidden_size=self.hidden_size,
                    beta_inits=[0.5] * self.n_trainable_cols,
                    use_bias=cfg.use_bias,
                    learnable_beta=False,
                    use_layer_norm=False,
                    dtype=torch.float64,
                ))

            if n_res > 0:
                self.reservoir_cells.append(BatchedReservoirColumns(
                    n_cols=n_res,
                    input_size=in_dim,
                    hidden_size=self.hidden_size,
                    spectral_radii=sr_list,
                    reservoir_scale=rcfg.reservoir_scale,
                    dtype=torch.float64,
                ))

            if cfg.use_cross_attention and n_res > 0:
                self.cross_attns.append(TrainableReservoirCrossAttention(
                    hidden_size=self.hidden_size,
                    n_trainable=self.n_trainable_cols,
                    n_reservoir=n_res,
                    num_heads=min(2, cfg.n_attn_heads),
                    dtype=torch.float64,
                ))

            n_part = cfg.n_columns if cfg.col_identities else None
            self.attn.append(MessagePassingLayer(self.hidden_size, num_heads=cfg.n_attn_heads, n_participants=n_part))
            self.attn_gates.append(nn.Linear(2 * self.hidden_size, 1, dtype=torch.float64))

        self.final_output_gate = None
        if hcfg.enabled and hcfg.use_final_output_gate:
            self.final_output_gate = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.float64),
                nn.Sigmoid(),
            )

        self.head = nn.Linear(self.hidden_size, output_size, dtype=torch.float64)

        dcfg = cfg.diversity_loss
        self.diversity_loss_fn = ColumnDiversityLoss(dcfg, cfg.n_layers) if dcfg.enabled else None

        self.register_buffer('_res_baseline', torch.zeros(1, self.hidden_size, dtype=torch.float64))

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        print(
            f'  Params: total={format_readable_num(total)}'
            f' | trainable={format_readable_num(trainable)}'
            f' | frozen={format_readable_num(frozen)}'
            f' ({100 * frozen / max(total, 1):.0f}% frozen)'
        )

    # --- state management ---

    def init_state(self, bsz: int) -> torch.Tensor:
        return torch.zeros(
            self.n_layers, self.n_columns, bsz, self.hidden_size,
            device=self.head.weight.device, dtype=self.head.weight.dtype,
        )

    def reset_state(self, state, reset_mask):
        if state is None:
            return self.init_state(reset_mask.shape[0])
        ixs = torch.nonzero(reset_mask).flatten()
        if ixs.numel() == 0:
            return state
        state = state.clone()
        state[:, :, ixs, :] = 0.0
        return state

    def detach_state(self, state):
        return state.detach() if state is not None else None

    # --- forward ---

    def forward(self, tokens: torch.Tensor, h=None, return_attn: bool = False):
        tokens = to_torch(tokens)
        assert tokens.ndim == 2 and tokens.shape[1] == 1
        bsz = tokens.shape[0]

        if h is None:
            h = self.init_state(bsz)

        x       = self.embedding(tokens.view(-1))   # (bsz, emb_size)
        h, extras = self._grid_step(x, h=h, return_attn=return_attn)

        z = h[-1, 0]   # top layer, first trainable column
        if self.final_output_gate is not None:
            z = self.final_output_gate(z) * z
        y = self.head(z)

        if return_attn:
            return y, h, extras
        return y, h

    def _grid_step(self, x, *, h, return_attn):
        n_t = self.n_trainable_cols
        n_r = self.n_reservoir_cols

        attn_list, gate_logit_list, h_layer_list = [], [], []
        h_new_layers = []

        # each column sees input through its own orthogonal projection
        x_cols = torch.stack([proj(x) for proj in self.col_input_projs], dim=1)  # (batch, n_cols, emb)

        for li in range(self.n_layers):
            hl = h[li]   # (n_cols, batch, hidden) — recurrent state

            if li == 0:
                x_t_batch = x_cols                                        # (batch, n_cols, emb)
            else:
                x_t_batch = h_new_layers[li - 1].permute(1, 0, 2)        # (batch, n_cols, hidden)

            h_t_in = hl[:n_t].permute(1, 0, 2)   # (batch, n_t, hidden)
            x_t_in = x_t_batch[:, :n_t, :]

            h_t_new = self._batched_trainable_forward(li, x_t_in, h_t_in)  # (batch, n_t, hidden)

            if n_r > 0:
                h_r_in  = hl[n_t:].permute(1, 0, 2)
                x_r_in  = x_t_batch[:, n_t:, :]
                h_r_new = self._batched_reservoir_forward(li, x_r_in, h_r_in)  # (batch, n_r, hidden)

                if self.cross_attns is not None:
                    h_t_new = self.cross_attns[li](h_t_new, h_r_new)

                h_all = torch.cat([h_t_new, h_r_new], dim=1)   # (batch, n_cols, hidden)
            else:
                h_all = h_t_new

            h_all_seq = h_all.permute(1, 0, 2)   # (n_cols, batch, hidden)
            msg, attn_w = self.attn[li](h_all_seq, return_weights=return_attn)

            msg_t      = msg[:n_t]
            h_t_seq    = h_t_new.permute(1, 0, 2)                              # (n_t, batch, hidden)
            gate_logit = self.attn_gates[li](torch.cat([h_t_seq, msg_t], dim=-1))
            g          = torch.sigmoid(gate_logit)
            h_t_merged = (1.0 - g) * h_t_seq + g * msg_t

            if n_r > 0:
                h_r_seq = h_r_new.permute(1, 0, 2)
                h_layer = torch.cat([h_t_merged, h_r_seq], dim=0)
            else:
                h_layer = h_t_merged

            h_new_layers.append(h_layer)
            attn_list.append(attn_w)
            gate_logit_list.append(gate_logit)
            h_layer_list.append(h_layer)

        h_tensor = torch.stack(h_new_layers, dim=0)
        extras = {
            'attn_weights': attn_list,
            'gate_logits':  gate_logit_list,
            'gates':        [torch.sigmoid(g) for g in gate_logit_list],
            'h_layers':     h_layer_list,
        }
        return h_tensor, extras

    def _batched_trainable_forward(
        self,
        li: int,
        x_cols: torch.Tensor,   # (batch, n_t, in_dim)
        h_cols: torch.Tensor,   # (batch, n_t, hidden)
    ) -> torch.Tensor:          # (batch, n_t, hidden)
        cell: BatchedHGRUColumns = self.trainable_cells[li]
        n_t = cell.n_cols

        x_t = x_cols.permute(1, 2, 0)   # (n_t, in_dim, batch)
        h_t = h_cols.permute(1, 2, 0)   # (n_t, hid, batch)

        def gx(W, b):
            out = torch.bmm(W, x_t).permute(0, 2, 1)   # (n_t, batch, hid)
            return out + b.unsqueeze(1) if b is not None else out

        def gh(U, h_src=None):
            src = h_src if h_src is not None else h_t
            return torch.bmm(U, src).permute(0, 2, 1)  # (n_t, batch, hid)

        o_t = torch.sigmoid(gx(cell.W_o, cell.b_o) + gh(cell.U_o))
        h_p = h_cols.permute(1, 0, 2)                          # (n_t, batch, hid)

        # content gate uses o_t * h_p as recurrent input
        oh    = (o_t * h_p).permute(0, 2, 1)                   # (n_t, hid, batch)
        c_raw = gx(cell.W_c, cell.b_c) + gh(cell.U_c, oh)

        if cell.ln_c is not None:
            c_normed = torch.stack([cell.ln_c[i](c_raw[i]) for i in range(n_t)], dim=0)
        else:
            c_normed = c_raw
        c_t = torch.tanh(c_normed)

        f_raw = gx(cell.W_f, cell.b_f) + gh(cell.U_f)
        betas = cell.betas.view(n_t, 1, 1)
        lam_t = torch.sigmoid(f_raw) * (1.0 - betas) + betas   # forget gate with beta floor

        h_new = lam_t * h_p + (1.0 - lam_t) * c_t             # (n_t, batch, hid)
        return h_new.permute(1, 0, 2)                           # (batch, n_t, hid)

    def _batched_reservoir_forward(
        self,
        li: int,
        x_cols: torch.Tensor,   # (batch, n_r, in_dim)
        h_cols: torch.Tensor,   # (batch, n_r, hidden)
    ) -> torch.Tensor:          # (batch, n_r, hidden)
        cell: BatchedReservoirColumns = self.reservoir_cells[li]
        n_r, hid = cell.n_cols, cell.hidden_size

        x_t = x_cols.permute(1, 2, 0)   # (n_r, in_dim, batch)
        h_t = h_cols.permute(1, 2, 0)   # (n_r, hid, batch)

        gates = (
            torch.bmm(cell.W_ih, x_t) + torch.bmm(cell.W_hh, h_t) + cell.b.unsqueeze(-1)
        ).permute(0, 2, 1)              # (n_r, batch, 3*hid)

        r = torch.sigmoid(gates[..., :hid])
        z = torch.sigmoid(gates[..., hid:2 * hid])

        h_p = h_cols.permute(1, 0, 2)                                             # (n_r, batch, hid)
        n_x = torch.bmm(cell.W_ih[:, 2 * hid:], x_t).permute(0, 2, 1)
        rh  = (r * h_p).permute(0, 2, 1)                                          # (n_r, hid, batch)
        n_h = torch.bmm(cell.W_hh[:, 2 * hid:], rh).permute(0, 2, 1)
        n   = torch.tanh(n_x + n_h + cell.b[:, 2 * hid:].unsqueeze(1))

        h_new = (1.0 - z) * n + z * h_p                                           # (n_r, batch, hid)
        return h_new.permute(1, 0, 2)                                              # (batch, n_r, hid)

    # --- diversity loss ---

    def compute_diversity_loss(self, extras: dict) -> dict[str, torch.Tensor]:
        zero = torch.tensor(0.0)
        if self.diversity_loss_fn is None:
            return {k: zero for k in ('cosine', 'covariance', 'variance', 'gate_entropy', 'total')}
        h_layers    = extras.get('h_layers', [])
        gate_logits = extras.get('gate_logits', [])
        if not h_layers:
            return {k: zero for k in ('cosine', 'covariance', 'variance', 'gate_entropy', 'total')}
        return self.diversity_loss_fn(h_layers, gate_logits)

    # --- inspection utilities ---

    def get_hgrn_betas(self) -> dict[str, float]:
        result = {}
        if not self.cfg.hgrn.enabled:
            return result
        for li, cell in enumerate(self.trainable_cells):
            if isinstance(cell, BatchedHGRUColumns):
                result.update(cell.get_betas_dict(li))
        return result

    def get_reservoir_spectral_radii(self) -> dict[str, float]:
        result = {}
        if not self.cfg.reservoir.enabled:
            return result
        for li, cell in enumerate(self.reservoir_cells):
            hid = self.hidden_size
            for ci in range(cell.n_cols):
                radii = []
                for gi in range(3):
                    block = cell.W_hh[ci, gi * hid:(gi + 1) * hid].float()
                    ev    = torch.linalg.eigvals(block).abs().max().item()
                    radii.append(ev)
                col_idx = self.n_trainable_cols + ci
                result[f"reservoir/sr/L{li}_C{col_idx}_mean"] = float(sum(radii) / len(radii))
                result[f"reservoir/sr/L{li}_C{col_idx}_max"]  = float(max(radii))
        return result

    def get_reservoir_utilization(self, h: torch.Tensor) -> dict[str, float]:
        result = {}
        if not self.cfg.reservoir.enabled:
            return result
        with torch.no_grad():
            for li in range(self.n_layers):
                for ci_res in range(self.n_reservoir_cols):
                    ci      = self.n_trainable_cols + ci_res
                    h_res   = h[li, ci]
                    baseline = self._res_baseline.expand(h_res.shape[0], -1)
                    for k, v in compute_reservoir_utilization(h_res, baseline).items():
                        result[f"reservoir/util/L{li}_C{ci}/{k}"] = v
        return result

    def get_column_cosine_similarities(self, h: torch.Tensor) -> dict[str, float]:
        result = {}
        with torch.no_grad():
            for li in range(self.n_layers):
                for ci in range(self.n_columns):
                    for cj in range(ci + 1, self.n_columns):
                        sim = F.cosine_similarity(
                            h[li, ci].float(), h[li, cj].float(), dim=-1
                        ).mean().item()
                        result[f"col_sim/L{li}_C{ci}_C{cj}"] = sim
        return result


# --- factory ---

def build_fusion_from_config(raw_cfg: dict, input_size: int, output_size: int) -> GridRnnFusion:
    fusion_cfg = FusionConfig(
        embedding_size=raw_cfg.get('embedding_size', 64),
        hidden_size=raw_cfg.get('hidden_size', 192),
        n_layers=raw_cfg.get('n_layers', 5),
        n_columns=raw_cfg.get('n_columns', 5),
        n_attn_heads=raw_cfg.get('n_attn_heads', 4),
        messaging=raw_cfg.get('messaging', 'post'),
        use_bias=raw_cfg.get('use_bias', True),
        dropout=raw_cfg.get('dropout', 0.0),
        col_identities=raw_cfg.get('col_identities', True),
        all_cols_get_input=raw_cfg.get('all_cols_get_input', True),
        use_cross_attention=raw_cfg.get('use_cross_attention', True),
        hgrn=HGRNConfig(**raw_cfg.get('hgrn', {})),
        reservoir=ReservoirConfig(**raw_cfg.get('reservoir', {})),
        diversity_loss=DiversityLossConfig(**raw_cfg.get('diversity_loss', {})),
    )
    return GridRnnFusion(input_size=input_size, output_size=output_size, cfg=fusion_cfg)
