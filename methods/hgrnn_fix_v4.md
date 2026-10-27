# HopfieldGridRnnFixV4

Hopfield version of `grnn_fix_v4`: the same per-column inter-column attention architecture, but on LSTM cells with **dual memory**. Motivation: the family's best-ever SDQ result historically belonged to hgrnn (Acc++ 0.870 at 215K), and its edge came from isolating long-term memory `c` from messages. This model checks whether the gains stack: v4's mechanics (per-column identities and beta, input-aware gate, timescales, aux losses) plus LSTM's memory protection.

## Key mechanism

Dual memory: the working state `h` recurrently carries the mixed message, the cell state `c` is never mixed:

```python
h_ic, c_ic = cells[ic](x[ic], (hl[ic], cl[ic]))   # LSTM per column
msg, _ = attn(hl_new)                              # PerColumnAttention from v4
g = torch.sigmoid(gates[ic](cat([hl_new, msg, x])))
hl_mix = hl_new + g * msg    # goes into recurrent h AND upward
c_n.append(cl_new)           # long-term memory: never mixed
```

Difference from `grnn_fix_v4` (GRU): there the recurrent state is fully protected and the message lives only on the upward path; here the message **enters the recurrent** `h` (columns can write into each other's working memory through the gate), but its errors can never overwrite `c`.

## Important implementation details

Multi-timescale via the LSTM forget gate (instead of the GRU update gate):

```python
# f -> 1 remembers longer (slow column); f -> 0 forgets fast
shift = timescale_spread * (2 * ic / (n_columns - 1) - 1)
cell.bias_ih[H:2 * H] += shift    # LSTM bias layout: [i, f, g, o]
```

Everything else is identical to v4: `PerColumnAttention` (Q/K identities + per-(column, head) beta), RMSNorm between layers, concat-readout, four aux losses (Barlow with depth-scaled weight, gate-std, activity decorrelation, upper-layer anti-saturation) every `aux_every` calls.

## Hyperparameters

| Parameter | Description |
|---|---|
| `hidden_size` | 56 at 2L x 3C gives ~204K parameters (LSTM costs 4/3 of GRU) |
| `beta_scale` | 3.0 — same as v4; per-column spread 0.5x-2x |
| `timescale_spread` | 1.0 — amplitude of the forget-bias shift across columns |
