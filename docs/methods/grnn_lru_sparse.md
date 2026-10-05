# Sparse-connectivity Grid LRU

**Summary.** Grid LRU with restricted or adaptive column communication, used as the control and base for the LRU variants. Instead of dense routing between all columns, columns can be connected as a star around column 0, a star with ring, blocks, or clockwork horizons; routing can be softmax or entmax15, and the hub can read with a learned (dynamic) query. File: `knitwork/models/grnn_lru_sparse.py`. Configurations: [LRU follow-up](../experiments/lru_followup.md), [sparse-LRU mid](../experiments/lru_sparse_mid.md).

## Key mechanism

A topology mask fixes which columns may read which; the hub row can additionally condition its routing on a query/key score.

```python
logits = pi_route_logits.masked_fill(~allowed, float('-inf'))
weights = torch.cat((logits[:1].softmax(-1), entmax15(logits[1:])), dim=0)  # entmax15 variant
```

## Hyperparameters

- `topology`: `dense`, `star`, `star_ring`; `routing_activation`: `softmax` or `entmax15` (dense only).
- `hub_query_size` (8) and `hub_query_init` (`zero` makes initial routing equal the static star).
- `fb_norm`: parameter-free RMS normalization of the top-layer feedback, which prevents the NaN growth seen in untrained Grid LRU.
