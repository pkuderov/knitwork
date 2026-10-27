# Multi-timescale LRU bank with attention read (grnn_bank)

`knitwork/models/grnn_lru_bank.py`, `--model=grnn_bank.L2C4`. A cardinal alternative to the dynamic grid for POPGym RL. The grid's useful
part on memory tasks is content-based selection between columns with different timescales; its harmful part for RL is that this selection sits
inside the recurrence (routed feedback and queries from the previous outputs), which lengthens the gradient path. Here the C columns are
independent stacks of complex LRUs (one horizon band per column, as in `column_horizons`). They never exchange messages, and the only
content-based step is one attention read at the output, outside the recurrence.

## Key mechanism

```python
inp = x.expand(C, B, H)                                    # every column gets the input (write='all')
w = softmax_c(<q(x), k(out_prev_c) + id_c>) * C            # write='attn': columns compete for the input (soft slots)
for layer in range(L):
    out, h[layer] = cells(layer, inp * w, h[layer])        # column-local LRU stack, no routing
tokens = cat([x, out_top]) + col_emb                       # (C + 1, B, H)
y = LayerNorm(x + MHA(query=tokens[0], keys=tokens))       # content-based read, one hop
```

## Hyperparameters

- `write='all' | 'attn'`: hypothesis H1 (read-only addressing) vs H2 (also content-addressed writes). The write key projection starts at zero,
  so allocation starts uniform; `write_key_size` is the key width.
- `write_heads=n`: n write heads, each with a value projection `V_h x` and its own allocation; column input `sum_h w_hc V_h x / n`.
- `write_feedback=True`: the write query is `q(x) + q_y(y_prev)` (previous read output, zero-initialised), so allocation depends on the
  model's own state as in the grid's recurrent routing.
- `column_horizons=True` by default: column c initialises its horizons in the c-th log band of `horizon`.
- Parameter-matched to the L2C4 dynamic grid (~1.03M with the RL wrapper) at `hidden_size=168`.
