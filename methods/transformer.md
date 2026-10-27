# Transformer baseline (rolling KV memory)

Causal Transformer with RoPE evaluated in the same streaming setup as the recurrent models (`baseline/transformer.py`,
`TransformerCore`; trained with `exps/text/run_offline.py` because it consumes a full TBPTT rollout per call). Attention covers a rolling memory of
the last `mem_len` tokens, which acts as the "state".

## Key mechanism

```python
state = {'kv': [(k, v)] * L, 'valid': [B, mem_len], 'pos': [B]}   # k, v: [B, mem_len, H]
```

State per sequence: `2 * L * mem_len * H` floats (KV cache); `valid`/`pos` are bookkeeping and not counted.

## Hyperparameters

`hidden_size`, `n_layers`, `n_heads=4`, `d_ff` (2H here), `mem_len=256`.
