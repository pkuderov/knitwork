# HopfieldGridRnnFix

Fixed variant of `hgrnn` (Hopfield Grid RNN on LSTM cells). The original already had two strong mechanisms — a message-protected memory `c` and a learnable temperature beta — but kept the base attention's defects: LayerNorm after a tiny-init projection (scale-1 noise from the first step), convex averaging `(1-g)h + g*msg`, input only into column 0, and readout from a single column. HopfieldGridRnnFix removes the post-norm, makes the message additive with a gate closed at init, feeds input into all columns through orthogonal projections, and reads the concatenation of top-layer columns.

## Key mechanism

Additive message into the working state `h`; the memory `c` is never mixed:

```python
h_ic, c_ic = cells[ic](x[ic], (hl[ic], cl[ic]))   # LSTM per column
...
msg, _ = attn(hl_new)                              # no post-norm inside
g = torch.sigmoid(attn_gate(torch.cat([hl_new, msg], dim=-1)))
hl_mix = hl_new + g * msg    # additive, not convex
c_n.append(cl_new)           # long-term memory stays clean
```

The additive term preserves column identity (a convex mixture is a consensus operator that pulls states together), while `c` carries long-term memory across all steps without contamination.

## Important implementation details

Reuses `ColumnAttention` from `grnn_fix.py` — Hopfield attention with a learnable beta per head, a tiny-init `out_proj`, and no LayerNorm; the gate has bias -3 (g ~ 0.05 at init). Input goes into all columns:

```python
x = torch.stack([proj(x) for proj in self.col_input_projs], dim=0)  # [C, B, E]
```

Readout is the concatenation of top-layer `hl_mix` columns (in `hgrnn` only `h[-1][0]` was read, and the other columns' contribution had to squeeze through the gate):

```python
z = hl_mix.permute(1, 0, 2).reshape(hl_mix.shape[1], -1)   # [B, C*H]
y = self.head(z)
```

## Hyperparameters

| Parameter | Description |
|---|---|
| `hidden_size` | 104 at 1L x 2C gives ~200K parameters; LSTM costs 4/3 of GRU at the same H |
| `n_columns` | 2 — the shape of the historical SDQ winner; capacity lives in H, not column count |
| `n_attn_heads` | H is truncated to a multiple of head count; beta is init as 1/sqrt(d_k) and learned |
