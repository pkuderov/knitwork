# GridRnnRoute (`grnn_route_lb`, `grnn_route_topk`, `grnn_route_noise`)

## Summary

Five 12C arms were designed, run for 61M steps each and analysed without anyone ever
looking at the column-to-column attention matrix. Every regularizer so far acts on
representations (`aux_div`, `aux_act`) or on the readout (`aux_head`); none acts on the
communication graph. `GridRnnRoute` makes routing a first-class object.

One fact shapes the whole design: **self-attention is already structurally impossible**.
The base `PerColumnAttention` masks the diagonal to `-inf` (`grnn_fix_v4.py`), because a
column's own state is carried by the residual `o = hl_n + g * msg`. So the degenerate mode
worth worrying about is not "every column reads itself" but "every column reads the same
hub" — and a hub is exactly what a two-tier structure of 2 specialists plus 10 blanks
looks like from the routing side.

Three independent knobs, all off by default, one enabled per config arm.

## Key mechanism

**In-degree load balance** (`aux_route_weight`). The Switch-Transformer auxiliary loss
applied to who *receives* attention rather than who is routed to an expert. Same
non-saturating form as `GridRnnBalance._lb_aux`: 0 for a uniform graph, C−1 for a single
hub.

```python
P = attn.mean(dim=(0, 1, 2))                                   # [C_k] mean received
f = F.one_hot(attn.argmax(dim=-1), C).to(attn.dtype).mean(dim=(0, 1, 2))
total = total + C * (f.detach() * P).sum() - 1.0
```

**Hard capacity** (`top_k`). A query column may read at most `top_k` of the other C−1.
This is the one mechanism here that is a constraint rather than a penalty — five arms of
soft penalties were each partly gamed, and a masked logit cannot be traded against the
task loss at any exchange rate.

```python
kth = logits.topk(self.top_k, dim=-1).values[..., -1:]
logits = logits.masked_fill(logits < kth, float('-inf'))
```

**Relative routing noise** (`noise_std`). Breaks the rich-get-richer loop by which a column
that wins attention early gets more gradient, gets better, and wins more.

```python
scale = logits.detach().std().clamp_min(1e-6)
logits = logits + self.noise_std * scale * torch.randn_like(logits)
```

The scaling is the point. `PerColumnAttention` has a learnable per-(column, head) `beta`
multiplying the logits, so noise of fixed variance is defeated for free: the task gradient
pushes `beta` up until the noise is negligible, and you get a "stochastic" attention that
is deterministic in practice. A perturbation proportional to the logits' own spread is
scale-invariant, so there is no escape by rescaling. `std` is detached — it is a measuring
stick, not something to shrink.

## Instrumentation

`PerColumnAttention` now always stashes a detached `[C_q, C_k]` mean as `last_attn`, and
`run_sdq.py` logs `route/in_max_share` (share of attention received by the most-read
column, 1/C when uniform), `route/in_eff_cols` (effective number of distinct sources) and
`route/diag_frac` (must stay 0 — a sanity check on the diagonal mask). This is available
for every model, not only the routing arms.

The graph-carrying stash used by the load-balance term is opt-in via `needs_attn_graph`:
holding a tensor with an autograd graph across a gradient-checkpoint boundary changes what
gets packed and raises `CheckpointError: A different number of tensors was saved`.

## Hyperparameters

| Name | Default | Notes |
|---|---|---|
| `aux_route_weight` | `0.0` (arm: `0.05`) | Loss ranges over [0, C−1]; 0.05 caps the contribution near 0.55. |
| `top_k` | `0` (off; arm: `3`) | Sources per query column, out of C−1 = 11 available. |
| `noise_std` | `0.0` (off; arm: `0.3`) | Fraction of the logit standard deviation. Training only. |
