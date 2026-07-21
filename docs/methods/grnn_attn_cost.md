# GridRnnAttnCost

Base GridRNN (post-messaging, input-conditioned gate) plus an **attention-cost penalty**. Across every attention variant the mixing gate `g` quickly collapses to a shared constant (~0.5-0.6): the model mixes columns everywhere by a fixed amount instead of learning *where* attention actually helps. Here an auxiliary loss makes attention "expensive": a monotonically increasing `cost(g)` is summed over all gates, so the CE gradient only opens a gate where the message pays for itself. The result is sparse, selective inter-column attention.

## Key mechanism

The gate is the standard input-conditioned mix; the penalty is a monotone function of how open each gate is, summed over columns and averaged over layers:

```python
g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))  # [C, B, 1]
hl_n = (1 - g) * hl_n + g * msg
# aux cost, summed over gates -> the model keeps attention closed unless it helps
aux = aux + self._gate_cost(g)          # cost(g).sum(cols).mean(batch)
...
aux = self.attn_cost_weight * (aux / self.n_layers)
```

`cost_kind` selects the shape: `linear` (`g`, cheap and stable), `quad` (`g**2`), or `logbarrier` (`-log(1-g)`, extremely expensive as `g -> 1`). The model returns the aux as its 4th forward output; the SDQ/text runners fold it into the loss automatically. Attention weights and gates are returned in `extras`, so the existing per-layer `AttnFlowVisualizer` heatmaps (all columns, all layers) directly show which gates the penalty leaves open.

## Hyperparameters

| Parameter | Description |
|---|---|
| `attn_cost_weight` | 0.02 — strength of the penalty; higher -> sparser gates (set 0 to disable) |
| `cost_kind` | `linear` \| `quad` \| `logbarrier` — how steeply cost grows with gate openness |
| `col_identities` | true — per-column identities in `MessagePassingLayer` |
