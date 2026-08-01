# GridRnnFixV5

v5.1 — a hybrid grid: **fast GRU columns + slow LRU storage columns + a frozen reservoir hub**. A revision after v5.0 failed (SDQ Acc++ 0.40@70M vs 0.777@50M for v4), where all columns were linear LRU. v5.0 diagnosis: (a) linear recurrence has nothing to **bind** with — "write V if K arrived" needs multiplicative gating, which GRU has every step and LRU has none of (Acc/distract 0.95, Acc/query 0.49); (b) the double-exponential reparametrization lambda = exp(-exp(nu)) blows up gradients (|Grad| = 8 at a clip of 1.0 — every step got cut down by 8x); (c) 4-5 separate matmuls per LRU cell per step — 4.5k fps vs ~10k for v4.

## Key mechanism

Division of labor by cell type: binding is handled by nonlinear GRU, long-term storage by linear LRU with a guaranteed retention floor:

```python
if ic < self.n_gru:
    cell = nn.GRUCell(in_dim, H)          # fast/medium: multiplicative binding
    cell.bias_ih[H:2*H] += shift          # timescale stagger (v4)
else:
    cell = FastFloorLRUCell(in_dim, H, r_floor=floor)   # slow storage, |lambda| >= 0.9
```

State is packed into a shared tensor [L, C, B, 2H]: GRU columns use the first half, LRU columns use both (re/im).

## FastFloorLRUCell — a faster, stabilized LRU

```python
self.B = nn.Linear(input_size, 2 * hidden_size, bias=False)  # merged B_re+B_im: 1 matmul
# no D feedthrough
y = self.C(h_n) * torch.sigmoid(self.G(h_n))                 # GLU: nonlinearity per step
self.nu.register_hook(lambda g: g * grad_scale)              # 0.1: damp double-exp gradients
```

Three fixes against v5.0: a merged B projection and dropping D (fewer kernel launches and parameters), a GLU output nonlinearity (a plain LRU has none), and a nu/theta gradient damper x0.1 (fixes |Grad|=8 without needing param groups in the runner).

## Important implementation details

The rest of the machinery is from v4/v5.0: per-column attention with a hub source (`HubColumnAttention`, C receivers x C+1 sources), scalar gates, additive message with protected recurrent state, RMSNorm between layers, concat-readout, aux losses (Barlow with depth-scaled weight, gate-std, activity; the saturation penalty applies only to GRU columns). The `lru/r_*` diagnostics in the runners report the true |lambda| of LRU columns including the floor; `attn_beta/L_C` tracks temperature evolution.

## Hyperparameters

| Parameter | Description |
|---|---|
| `n_lru_cols` | 1 — how many columns from the end are LRU storage |
| `hidden_size` | 60 at 2L x 3C (2 GRU + 1 LRU) — ~196K active parameters |
| `r_floor_min/max` | 0.9 / 0.95 — retention floors of LRU columns |
| `lru_grad_scale` | 0.1 — nu/theta gradient damper |
| `timescale_spread` | 1.0 — spread of the GRU columns' update-gate bias |
