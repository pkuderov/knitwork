from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DiversityLossConfig:
    cosine_weight:       float = 0.05
    cov_weight:          float = 0.04
    var_weight:          float = 0.02
    gate_entropy_weight: float = 0.01
    layer_weights:       Optional[list] = None
    cosine_margin:       float = 0.3
    var_threshold:       float = 1.0


class ColumnDiversityLoss(nn.Module):
    """4 components: cosine, covariance, variance, gate entropy."""

    def __init__(self, cfg: DiversityLossConfig, n_layers: int):
        super().__init__()
        self.cfg = cfg
        lw = (
            torch.tensor(cfg.layer_weights, dtype=torch.float32)
            if cfg.layer_weights is not None
            else torch.ones(n_layers, dtype=torch.float32)
        )
        self.register_buffer('layer_weights', lw)

    def cosine_diversity(self, h: torch.Tensor, lw: float) -> torch.Tensor:
        # h: [cols, B, H]
        cols = h.shape[0]
        loss, count = h.new_zeros(1).squeeze(), 0
        for i in range(cols):
            for j in range(i + 1, cols):
                sim = F.cosine_similarity(h[i], h[j], dim=-1).mean()
                loss = loss + F.relu(sim - self.cfg.cosine_margin)
                count += 1
        return lw * self.cfg.cosine_weight * (loss / max(count, 1))

    def covariance_diversity(self, h: torch.Tensor, lw: float) -> torch.Tensor:
        # h: [cols, B, H]
        cols, bsz, d = h.shape
        loss = h.new_zeros(1).squeeze()
        eye = torch.eye(d, device=h.device, dtype=h.dtype)
        for c in range(cols):
            z = h[c] - h[c].mean(dim=0, keepdim=True)
            cov = (z.T @ z) / (bsz - 1)
            loss = loss + (cov * (1.0 - eye)).pow(2).sum() / d
        return lw * self.cfg.cov_weight * (loss / cols)

    def variance_loss(self, h: torch.Tensor, lw: float) -> torch.Tensor:
        # h: [cols, B, H]
        cols = h.shape[0]
        loss = h.new_zeros(1).squeeze()
        for c in range(cols):
            loss = loss + F.relu(self.cfg.var_threshold - h[c].var(dim=0).sqrt()).mean()
        return lw * self.cfg.var_weight * (loss / cols)

    def gate_entropy_loss(self, gates: list[torch.Tensor]) -> torch.Tensor:
        eps   = 1e-6
        total = torch.tensor(0.0, device=gates[0].device, dtype=gates[0].dtype)
        for g in gates:
            gc = g.clamp(eps, 1.0 - eps)
            H  = -(gc * gc.log() + (1.0 - gc) * (1.0 - gc).log())
            total = total - H.mean()
        return self.cfg.gate_entropy_weight * total / max(len(gates), 1)

    def forward(
        self,
        h_per_layer: list[torch.Tensor],  # list of [cols, B, H]
        gates: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        dev, dt = h_per_layer[0].device, h_per_layer[0].dtype
        cos = cov = var = torch.tensor(0.0, device=dev, dtype=dt)
        for i, h in enumerate(h_per_layer):
            lw = self.layer_weights[i].item()
            cos = cos + self.cosine_diversity(h, lw)
            cov = cov + self.covariance_diversity(h, lw)
            var = var + self.variance_loss(h, lw)
        gate = self.gate_entropy_loss(gates)
        return {'cosine': cos, 'covariance': cov, 'variance': var,
                'gate_entropy': gate, 'total': cos + cov + var + gate}


class ColumnDiversityAnalyzer:
    """Post-training column specialization metrics (no learnable params)."""

    @staticmethod
    def _center(X: torch.Tensor) -> torch.Tensor:
        return X - X.mean(0, keepdim=True)

    @staticmethod
    def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        X = ColumnDiversityAnalyzer._center(X).double()
        Y = ColumnDiversityAnalyzer._center(Y).double()
        hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
        hsic_xx = (X @ X.T).pow(2).sum().sqrt()
        hsic_yy = (Y @ Y.T).pow(2).sum().sqrt()
        return (hsic_xy / (hsic_xx * hsic_yy + 1e-10)).item()

    @staticmethod
    def cosine_similarity_matrix(states: torch.Tensor) -> torch.Tensor:
        # states: [cols, N, D] -> [cols, cols]
        means  = states.mean(dim=1)
        normed = means / means.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return normed @ normed.T

    @staticmethod
    def rsa(X: torch.Tensor, Y: torch.Tensor) -> float:
        N = X.shape[0]
        def rdm(M):
            M = F.normalize(M.float(), dim=-1)
            sim = M @ M.T
            idx = torch.triu_indices(N, N, offset=1)
            return (1 - sim)[idx[0], idx[1]]
        def spearman(a, b):
            ra = a.argsort().argsort().float()
            rb = b.argsort().argsort().float()
            ra -= ra.mean(); rb -= rb.mean()
            return (ra * rb).sum() / (ra.norm() * rb.norm() + 1e-10)
        return spearman(rdm(X), rdm(Y)).item()

    @staticmethod
    def effective_rank(X: torch.Tensor) -> float:
        X  = ColumnDiversityAnalyzer._center(X).float()
        sv = torch.linalg.svdvals(X)
        sv = sv / (sv.sum() + 1e-10)
        return (-(sv * (sv + 1e-10).log()).sum()).exp().item()

    @staticmethod
    def activation_entropy(X: torch.Tensor, bins: int = 50) -> float:
        X = X.float()
        entropies = []
        for d in range(X.shape[1]):
            hist = torch.histc(X[:, d], bins=bins)
            p    = hist / (hist.sum() + 1e-10)
            entropies.append(-(p * (p + 1e-10).log()).sum().item())
        return sum(entropies) / len(entropies)

    @staticmethod
    def pca_subspace_angle(X: torch.Tensor, Y: torch.Tensor, k: int = 8) -> float:
        def basis(M, k):
            M = ColumnDiversityAnalyzer._center(M).float()
            return torch.linalg.svd(M, full_matrices=False)[2][:k]
        sigma = torch.linalg.svdvals(basis(X, k) @ basis(Y, k).T).clamp(-1, 1)
        return (torch.acos(sigma) * 180 / math.pi).mean().item()

    @classmethod
    def full_report(cls, hidden_states: torch.Tensor, layer_idx: int = 0) -> Dict:
        C = hidden_states.shape[0]
        cka_v, rsa_v, ang_v = [], [], []
        for i in range(C):
            for j in range(i + 1, C):
                cka_v.append(cls.cka(hidden_states[i], hidden_states[j]))
                rsa_v.append(cls.rsa(hidden_states[i], hidden_states[j]))
                ang_v.append(cls.pca_subspace_angle(hidden_states[i], hidden_states[j]))
        report = {
            f"layer{layer_idx}/cka_mean":       sum(cka_v) / len(cka_v),
            f"layer{layer_idx}/cka_max":        max(cka_v),
            f"layer{layer_idx}/rsa_mean":       sum(rsa_v) / len(rsa_v),
            f"layer{layer_idx}/pca_angle_mean": sum(ang_v) / len(ang_v),
        }
        for c in range(C):
            report[f"layer{layer_idx}/col{c}/erank"]   = cls.effective_rank(hidden_states[c])
            report[f"layer{layer_idx}/col{c}/entropy"] = cls.activation_entropy(hidden_states[c])
        report[f"layer{layer_idx}/cosine_sim_matrix"] = cls.cosine_similarity_matrix(hidden_states).cpu()
        return report

    @staticmethod
    def top_token_analysis(
        hidden_states: torch.Tensor,
        tokens: torch.Tensor,
        vocab,
        top_k: int = 20,
    ) -> Dict[int, List[str]]:
        C = hidden_states.shape[0]
        result = {}
        for c in range(C):
            norms = hidden_states[c].norm(dim=-1)
            score = {t.item(): norms[tokens == t].mean().item()
                     for t in tokens.unique() if (tokens == t).any()}
            top = sorted(score, key=score.get, reverse=True)[:top_k]
            result[c] = [vocab[t] if vocab else str(t) for t in top]
        return result


class ColumnSpecializationLoss(nn.Module):
    """4 components: decorr (VICReg), variance hinge, cosine, whiten (SVD).

    Input h: [layers, cols, batch, hidden]. Returns (total_loss, details_dict).
    """

    def __init__(
        self, *,
        hidden_size: int,
        n_columns: int,
        n_layers: int,
        lambda_decorr: float = 1.0,
        lambda_var: float = 0.5,
        lambda_cosine: float = 0.3,
        lambda_whiten: float = 0.1,
        target_layers: Optional[List[int]] = None,
        use_squared_decorr: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_columns = n_columns
        self.n_layers = n_layers
        self.lambda_decorr = lambda_decorr
        self.lambda_var    = lambda_var
        self.lambda_cosine = lambda_cosine
        self.lambda_whiten = lambda_whiten
        self.target_layers = target_layers or list(range(n_layers))
        self.use_squared_decorr = use_squared_decorr

    def _decorr_loss(self, h: torch.Tensor) -> torch.Tensor:
        # h: [C, B, D]
        C, B, D = h.shape
        loss, n_pairs = h.new_zeros(1), 0
        for i in range(C):
            for j in range(i + 1, C):
                hi = h[i] - h[i].mean(0, keepdim=True)
                hj = h[j] - h[j].mean(0, keepdim=True)
                cov = (hi.T @ hj) / (B - 1)
                loss = loss + (cov.pow(2) if self.use_squared_decorr else cov.abs()).sum() / D
                n_pairs += 1
        return loss / max(n_pairs, 1)

    def _var_loss(self, h: torch.Tensor) -> torch.Tensor:
        # h: [C, B, D]
        C = h.shape[0]
        loss = h.new_zeros(1)
        for i in range(C):
            loss = loss + F.relu(1.0 - h[i].std(dim=0)).mean()
        return loss / C

    def _cosine_loss(self, h: torch.Tensor) -> torch.Tensor:
        means = F.normalize(h.mean(dim=1), dim=-1)         # [C, D]
        gram  = means @ means.T
        mask  = torch.triu(torch.ones(h.shape[0], h.shape[0], device=h.device), diagonal=1).bool()
        return gram[mask].pow(2).mean()

    def _whiten_loss(self, h: torch.Tensor) -> torch.Tensor:
        C, B, D = h.shape
        k = min(16, D // 4, B - 1)
        if k < 2:
            return h.new_zeros(1)
        bases = []
        for i in range(C):
            hi = h[i] - h[i].mean(0, keepdim=True)
            try:
                bases.append(torch.linalg.svd(hi, full_matrices=False)[2][:k])
            except Exception:
                return h.new_zeros(1)
        loss, n_pairs = h.new_zeros(1), 0
        for i in range(C):
            for j in range(i + 1, C):
                loss = loss + (bases[i] @ bases[j].T).pow(2).sum() / k
                n_pairs += 1
        return loss / max(n_pairs, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # h: [layers, cols, batch, hidden]
        total, details = h.new_zeros(1), {}
        for li in self.target_layers:
            hl = h[li]
            ld = self._decorr_loss(hl)
            lv = self._var_loss(hl)
            lc = self._cosine_loss(hl)
            lw = self._whiten_loss(hl)
            layer_loss = (
                self.lambda_decorr * ld + self.lambda_var * lv
                + self.lambda_cosine * lc + self.lambda_whiten * lw
            )
            total = total + layer_loss
            details.update({
                f"spec/layer{li}/decorr": ld.item(),
                f"spec/layer{li}/var":    lv.item(),
                f"spec/layer{li}/cosine": lc.item(),
                f"spec/layer{li}/whiten": lw.item(),
                f"spec/layer{li}/total":  layer_loss.item(),
            })
        total = total / max(len(self.target_layers), 1)
        details["spec/total"] = total.item()
        return total, details
