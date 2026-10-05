# Grid LRU with addressable slots

**Summary.** `associative` stacks all facts on one matrix, and `write_gate` freezes a whole vector. Here one peripheral column per layer is replaced by `S=8` separate key/value slots (plus a usage scalar). Column 0 stays the controller and the only output: it reads slots by content and decides whether and where to write, so a distractor token can leave stored facts untouched. Addressable memory is from NTM/DNC; this adapts it to the Grid with the task loss only (no auxiliary losses, no discrete estimators). File: `knitwork/models/grnn_lru_slots.py`.

## Key mechanism

Slot state is `[B, S, key + value + usage]` per layer. The hub context forms the read query; the write key, value and gate come from the slot column's routed message and the hub context.

```python
logits = beta_r * cos(q, slot_keys) - 4 * (1 - usage)      # read: empty slots suppressed
read = softmax(logits) @ slot_values
step = gate * write_weight                                    # [B, S, 1] rewrite strength
values = lerp(values, new_value, step)                        # untouched slots keep step ~ 0
```

Three write-address variants share this read:

- `slot_content`: write weights = softmax over key cosine (existing fact is updated, empty slots win ties by a fixed order).
- `slot_alloc`: a learned gate mixes content addressing with DNC-style allocation to the least used slot.
- `slot_erase_add`: content addressing, but values use NTM erase/add vectors, so one dimension of a fact can change without overwriting the rest.

## Hyperparameters

- `n_slots=8`, `slot_key_size=16`, `slot_value_size=32`: carried state 4,024 floats vs 3,960 for the control (L2C4, H180); MQAR weights about 3.80M vs 4.00M (the slot column replaces an LRU column).
- `slot_write_bias=-1`: initial write gate about 0.27.
- Read and write sharpness are learned per layer, initialised to 8.
- Config: `knitwork/exps/mqar/config/lru_slots_mid/`; compare against `lru_architecture_mid/dynamic_control.yaml` (MQAR, 8 epochs).
