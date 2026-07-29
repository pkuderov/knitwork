"""mLSTM baseline — matrix LSTM with exponential gating (Beck et al., NeurIPS 2024, arXiv 2405.04517)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _mLSTMLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        H = hidden_size
        self.H   = H
        self.W_q = nn.Linear(input_size, H, bias=False)
        self.W_k = nn.Linear(input_size, H, bias=False)
        self.W_v = nn.Linear(input_size, H, bias=False)
        self.W_i = nn.Linear(input_size, 1, bias=True)   # log-domain input gate
        self.W_f = nn.Linear(input_size, 1, bias=True)   # forget gate (log-sigmoid)
        self.norm = nn.RMSNorm(H)
        self.ff   = nn.Sequential(nn.Linear(H, H * 2), nn.GELU(), nn.Linear(H * 2, H))
        nn.init.normal_(self.ff[-1].weight, std=0.01 / (H ** 0.5))
        nn.init.zeros_(self.ff[-1].bias)
        nn.init.constant_(self.W_f.bias, 3.0)   # high initial forget = long memory

    def forward(self, x: torch.Tensor, state):
        # x: [B, H],  state: (C [B,H,H], n [B,H], m [B,1])
        C, n, m = state
        q  = self.W_q(x) / (self.H ** 0.5)   # [B, H] scaled query
        k  = self.W_k(x) / (self.H ** 0.5)   # [B, H] scaled key
        v  = self.W_v(x)                       # [B, H]
        li = self.W_i(x)                       # [B, 1] log input gate
        lf = F.logsigmoid(self.W_f(x))        # [B, 1] log forget gate

        # numerically stable gate combination in log-space
        m_new = torch.max(lf + m, li)          # [B, 1]
        f_g   = torch.exp(lf + m - m_new)      # [B, 1] forget gate
        i_g   = torch.exp(li - m_new)          # [B, 1] input gate

        C = f_g.unsqueeze(-1) * C + i_g.unsqueeze(-1) * torch.bmm(
            v.unsqueeze(-1), k.unsqueeze(1)    # outer product v ⊗ k  [B, H, H]
        )
        n = f_g * n + i_g * k                  # [B, H] normaliser

        # stabilised readout
        n_dot_q  = (n * q).sum(-1, keepdim=True).abs().clamp(min=1.0)
        h_out    = torch.bmm(C, q.unsqueeze(-1)).squeeze(-1) / n_dot_q  # [B, H]
        h_out    = self.norm(h_out) + x
        h_out    = h_out + self.ff(h_out)
        return h_out, (C, n, m_new)


class mLSTM(nn.Module):
    """mLSTM baseline.

    State per layer: (C ∈ R^{H×H}, n ∈ R^H, m ∈ R^1).
    Update: C_t = f_t C_{t-1} + i_t (v ⊗ k);  gates in log-space for stability.
    """

    def __init__(
        self,
        *,
        input_size: int,
        embedding_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers    = n_layers

        self.embedding  = nn.Embedding(input_size, embedding_size)
        self.input_proj = nn.Linear(embedding_size, hidden_size)
        self.layers     = nn.ModuleList([
            _mLSTMLayer(hidden_size, hidden_size) for _ in range(n_layers)
        ])
        self.norm_out   = nn.RMSNorm(hidden_size)
        self.head       = nn.Linear(hidden_size, output_size)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'mLSTM {n_layers}L  hidden={hidden_size}  params={n_params:,}')

    def forward(self, tokens: torch.Tensor, h=None):
        # tokens: [B, 1]
        x = self.embedding(tokens.view(-1))  # [B, emb]
        x = self.input_proj(x)               # [B, H]
        B = x.shape[0]

        if h is None:
            h = self.init_state(B, x.device)
        C_list, n_list, m_list, y_prev = h

        new_C, new_n, new_m, new_y = [], [], [], []
        for layer, C, n, m in zip(self.layers, C_list, n_list, m_list):
            x, (C_new, n_new, m_new) = layer(x, (C, n, m))
            new_C.append(C_new)
            new_n.append(n_new)
            new_m.append(m_new)
            new_y.append(x)

        y_stk  = torch.stack(new_y)  # [L, B, H]
        logits = self.head(self.norm_out(x))
        return logits, (new_C, new_n, new_m, y_stk)

    def init_state(self, batch_size: int, device):
        H = self.hidden_size
        C = [torch.zeros(batch_size, H, H, device=device) for _ in self.layers]
        n = [torch.zeros(batch_size, H,    device=device) for _ in self.layers]
        m = [torch.zeros(batch_size, 1,    device=device) for _ in self.layers]
        y = torch.zeros(self.n_layers, batch_size, H, device=device)
        return (C, n, m, y)

    def reset_state(self, h, mask: torch.Tensor):
        if h is None:
            return h
        C_list, n_list, m_list, y = h
        mf = (~mask.bool()).float()
        return (
            [C * mf[:, None, None] for C in C_list],
            [n * mf[:, None]       for n in n_list],
            [m * mf[:, None]       for m in m_list],
            y * mf[None, :, None],
        )

    def detach_state(self, h):
        if h is None:
            return h
        C_list, n_list, m_list, y = h
        return (
            [C.detach() for C in C_list],
            [n.detach() for n in n_list],
            [m.detach() for m in m_list],
            y.detach(),
        )

    def get_top_h(self, h) -> torch.Tensor:
        """Last-layer output for critic: [B, H]."""
        *_, y = h
        return y[-1]


class mLSTMCore(nn.Module):
    """Feature-level mLSTM core for use with model wrappers."""
    has_attn = False

    def __init__(
            self, *,
            hidden_size, n_layers,
            dtype, device,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dtype = dtype
        self.device = device

        self.layers = nn.ModuleList([
            _mLSTMLayer(hidden_size, hidden_size)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.RMSNorm(hidden_size)

        print(f'mLSTM core {n_layers}L w/ {hidden_size} hidden units')

    def forward(self, x: torch.Tensor, state: dict, **_):
        assert x.shape[0] == 1
        x = x.squeeze(0)
        if state is None:
            state = self.init_state(x.shape[0])

        new_C, new_n, new_m = [], [], []
        for layer, C, n, m in zip(
                self.layers, state['C'], state['n'], state['m']
        ):
            x, (C, n, m) = layer(x, (C, n, m))
            new_C.append(C)
            new_n.append(n)
            new_m.append(m)

        state = {'C': new_C, 'n': new_n, 'm': new_m}
        return self.norm_out(x), state, {}

    def reset_state(self, state=None, reset_mask=None, *, bsz=None):
        if state is None:
            bsz = reset_mask.shape[0] if reset_mask is not None else bsz
            return self.init_state(bsz)

        keep_matrix = (~reset_mask.flatten())[:, None, None]
        keep_vector = keep_matrix.squeeze(-1)
        return {
            'C': [C * keep_matrix for C in state['C']],
            'n': [n * keep_vector for n in state['n']],
            'm': [m * keep_vector for m in state['m']],
        }

    def detach_state(self, state):
        if state is None:
            return state
        return {
            key: [value.detach() for value in values]
            for key, values in state.items()
        }

    def init_state(self, bsz):
        matrices = [
            torch.zeros(
                bsz, self.hidden_size, self.hidden_size,
                device=self.device, dtype=self.dtype,
            )
            for _ in self.layers
        ]
        vectors = [
            torch.zeros(
                bsz, self.hidden_size,
                device=self.device, dtype=self.dtype,
            )
            for _ in self.layers
        ]
        scalars = [
            torch.zeros(
                bsz, 1,
                device=self.device, dtype=self.dtype,
            )
            for _ in self.layers
        ]
        return {'C': matrices, 'n': vectors, 'm': scalars}
