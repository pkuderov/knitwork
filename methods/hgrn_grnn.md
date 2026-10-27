# HGRN_GridRnn

The model addresses the problem of uniform "forgetting" in a standard GRU within the Grid RNN context: all network layers reset state at the same rate, which hinders simultaneous storage of local and long-term patterns. The key idea is to replace GRUCell with HGRUCell (Hierarchically Gated Recurrent Unit), where the forget gate λ has a learnable lower bound β specific to each layer. Lower layers (β ≈ 0) behave like a standard GRU and quickly overwrite state, while upper layers (β → 1) almost never forget — in the spirit of hierarchical recurrent networks HGRN.

## Key mechanism

HGRUCell introduces three gates instead of two in GRU: output gate `o_t`, content candidate `c_t`, and forget gate `λ_t` with lower bound β.

```python
# output gate controls how much of h_{t-1} enters content computation  [B, H]
o_t = torch.sigmoid(self.W_o(x) + self.U_o(h))
# candidate content uses gated previous state                           [B, H]
c_t = torch.tanh(self.W_c(x) + self.U_c(o_t * h))
# forget gate bounded below by beta: lambda in [beta, 1]               [B, H]
raw_f = torch.sigmoid(self.W_f(x) + self.U_f(h))
lam_t = raw_f * (1.0 - self.beta) + self.beta
# state update
h_new = lam_t * h + (1.0 - lam_t) * c_t
```

β is stored as `beta_raw` in pre-sigmoid space (`β = sigmoid(beta_raw)`), guaranteeing β ∈ (0, 1) for any parameter value.

## Important implementation details

**Hierarchical β assignment per layer.** During initialization β values are linearly distributed from `beta_min` (bottom layer) to `beta_max` (top layer):

```python
# layer 0 -> beta_min, layer L-1 -> beta_max
betas = [
    beta_min + (beta_max - beta_min) * i / (n_layers - 1)
    for i in range(n_layers)
]
```

This hard-codes the hierarchy of temporal horizons: lower layers track the current token, upper layers track long-term context.

**Output gate at the top of the full grid.** On top of the final state a `final_output_gate` is added — an additional sigmoid block that scales the representation before the head:

```python
gate = self.final_output_gate(z)   # [B, H]
z = gate * z
y = self.head(z)
```

**Post-messaging with a merge gate.** After attention-based message passing between columns, the original state and message are mixed via a learnable gate:

```python
g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))   # [cols, B, 1]
hl_n = (1.0 - g) * hl_n + g * msg
```

## Hyperparameters

| Parameter | Description |
|---|---|
| `beta_min` | Lower bound of λ for the bottom layer (≈ 0 — high forgetting rate) |
| `beta_max` | Lower bound of λ for the top layer (≈ 0.99 — long-term memory) |
| `messaging` | `"post"` — attention after cell step; `"pre"` — before step (changes input dimension) |
| `col_identities` | Whether to add learnable positional bias in attention to distinguish columns |

## Results

### SDQ (Store-Distract-Query, hard)

| Configuration | H | Cols / Layers | β range | Acc | Acc++ | Loss | Steps |
|---|---|---|---|---|---|---|---|
| grnn\_hgru H=128 (`sdq-hgru`) | 128 | 4 / 3 | 0.0–0.99 | **0.965** | **0.930** | **0.092** | ~45M |
| grnn\_hgru 5col H=116 (`sdq-hgru`) | 116 | 5 / 3 | 0.0–0.99 | 0.717 | 0.416 | 0.765 | ~14M |
| grnn\_hgrn H=160 4col 5l (`grid-rnn-sdq`) | 160 | 4 / 5 | 0.0–0.99 | 0.738 | 0.604 | 0.640 | ~210M |

The 4 col / 3 layer configuration shows the best result among all tested models (Acc=0.965), slightly outperforming the base grnn 4/3 (Acc=0.960) at the same number of steps. Hierarchical β distribution (lower layers — fast memory, upper layers — slow memory) is more effective than uniform GRU for associative tasks. The 5-column variant (14M steps) is not complete; results are preliminary.

### Text experiments (shakespeare)

| Configuration | H | Cols / Layers | Acc | BPC | PPL | Steps |
|---|---|---|---|---|---|---|
| grnn\_hgru H=128 (`text-hgru`) | 128 | 4 / 3 | 0.635 | 1.686 | 3.22 | ~70M |
| grnn\_hgru 5col H=116 (`text-hgru`) | 116 | 5 / 3 | **0.636** | **1.680** | **3.20** | ~70M |
| grnn\_hgru\_reservoir H=128 (`text-hgru`) | 128 | 5 / 3 (3+2 res.) | 0.628 | 1.730 | 3.32 | ~70M |

On shakespeare HGRN cells outperform the GRU counterpart of the same depth (BPC=1.686 vs 1.721, PPL=3.22 vs 3.30). The 5-column variant (H=116) is slightly better (BPC=1.680, PPL=3.20) despite the smaller hidden size. Adding reservoir columns to HGRU reduces quality (BPC=1.730 vs 1.686).
