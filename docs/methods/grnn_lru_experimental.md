# Grid LRU memory and access variants

**Summary.** Five single-mechanism extensions of the star-connected, dynamic-hub Grid LRU (`grnn_lru_sparse` base, L2C4, H180). Column 0 is the controller and the only output; each variant changes how peripheral memory is written or read, with the task loss only. File: `knitwork/models/grnn_lru_experimental.py`. Experiment setup: [LRU memory and access variants](../experiments/lru_architecture_variants.md).

## Key mechanism

- `write_gate`: the hub gates peripheral updates, `h = lerp(h_old, h_new, gate)`, so a closed gate preserves state exactly.
- `query_read`: the hub reads narrow (rank 16) query-conditioned values from peripheral private state.
- `associative`: one peripheral column is replaced by a delta-rule matrix memory (`M += beta * k (v - k M)`).
- `rotations`: Givens rotations exchange recurrent state between hub and peripherals (24 angles).
- `two_reads`: a second hub read over fresh peripheral outputs, with shared weights.

```python
next_memory = memory + key.unsqueeze(-1) * (rate * (value - key @ memory)).unsqueeze(-2)  # associative
```

## Hyperparameters

- `mechanism`: one of the five names above; `memory_key_size`/`memory_value_size` (16/32) and `memory_column` for `associative`; `read_rank` for `query_read`.
- `bounded_feedback` (default off): normalized private read added as a small residual (`residual_scale=0.1`) to the original message, and RMS-normalized inter-layer messages; added after unstable gradients in MQAR runs.
