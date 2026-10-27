# Many-column grid with structured communication (grnn_many)

`knitwork/models/grnn_many.py`, `--model=grnn_many.C16`. In the RL study the one-layer dynamic grid (4 columns, per-column horizon
bands) was the best memory core, but adding columns hurt: with per-column weights the columns get narrower, and all-to-all attention dilutes
each column's sources. This core makes columns cheap (one shared LRU input/output projection, columns differ only by their horizon band and
routing ids) so that 16+ columns fit, and replaces all-to-all routing by structured schemes. Communication stays recurrent and
state-dependent (queries from the previous outputs) and one hop per step, the two properties the grid could not lose.

## Key mechanism

```python
q, kv = outs_prev, rms(outs_prev)                         # (C, B, H) state-dependent queries, fed-back sources
msg = comm(q, [kv, x])                                    # full | workspace | groups
mask = topk_c(<q(x), k(outs_prev_c)> + bias_c, active_k)  # active_k > 0: column 0 + k others update
out, h = lru(msg + x, h, write=sigmoid(score) * mask)     # held columns keep h exactly (no decay)
y = out[0]
```

- `comm='full'` (M1): every column attends to all columns and the input.
- `active_k=k` (M2): mixture-of-experts style writes; inactive columns hold their state without decay, so a stored item does not fade with
  episode length. The selection bias is moved toward even load (aux-loss-free) only in the gradient replay.
- `comm='workspace'` (M3): `n_slots` shared slots read the top `slot_topk` sources (competition), columns then read the slots and the input.
- `comm='groups'` (M4): full attention inside groups of `group_size`; the first column of each group is a hub that also attends to the other
  hubs (cross-group path: two hops).

## Hyperparameters

- Activation memory of the 128-step BPTT scales with `C * H`, which is the binding limit on a small GPU; `hidden_size` must not equal
  `n_envs` (the state batch dim would be ambiguous).
- `route_key_size`, `active_balance`: router key width and the bias step of the load balancing.
