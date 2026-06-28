# GridRnnNoveltyGate (grnn_disc)

The model addresses the problem of useless message passing between Grid RNN columns: the standard continuous gate cannot explicitly ignore messages that carry no new information. The core idea is to replace the linear attention gate with a `NoveltyGate` that computes the novelty of a message relative to the current hidden state via cosine distance, then quantizes the gate to a discrete set `{0, 0.5, 1}` using a straight-through estimator: `0` — the message carries old information and is ignored, `0.5` — partial update, `1` — full replacement of state with the new message.

## Key mechanism

```python
# novelty = cosine distance, mapped to [0, 1]
cos_sim = F.cosine_similarity(h_new, msg, dim=-1, eps=1e-8)  # [cols, batch]
novelty = (1.0 - cos_sim) / 2.0                              # [cols, batch]

# straight-through discretization: forward=discrete, backward=continuous
discrete = torch.where(score < lo, GATE_LOW,
           torch.where(score > hi, GATE_HIGH, GATE_MID))
return score + (discrete - score).detach()                   # [cols, batch, 1]
```

The operation `score + (discrete - score).detach()` returns a discrete value in the forward pass, but during backprop the gradient flows through the continuous `score`, bypassing the non-differentiable branching.

## Important implementation details

**Blending cosine and learned novelty:**

```python
# blend raw cosine novelty with learned correction
raw     = self._raw_novelty(h_new, msg)                             # [cols, batch, 1]
learned = self.novelty_proj(torch.cat([h_new, msg], dim=-1))        # [cols, batch, 1]
blend   = torch.sigmoid(self.blend)                                 # scalar in (0,1)
score   = (1.0 - blend) * raw + blend * learned
```

`self.blend` is a learnable scalar initialized at `0.1`, so at the start of training the simple cosine distance dominates.

**Gate application in grid step:**

```python
# discrete novelty gate replaces standard sigmoid gate
g    = nov_gate(hl_n, msg)          # [cols, batch, 1] in {0.0, 0.5, 1.0}
hl_n = (1.0 - g) * hl_n + g * msg  # selective state update
```

## Hyperparameters

| Parameter | Description |
|---|---|
| `novelty_low` | Lower novelty threshold; messages with score below → gate=0 (ignored) |
| `novelty_high` | Upper threshold; messages above → gate=1 (full replacement) |
| `GATE_LOW / GATE_MID / GATE_HIGH` | Fixed discrete gate values: `0.1`, `0.4`, `0.6`; not literally `0/0.5/1`, which softens extreme updates |

## Results

### SDQ (Store-Distract-Query, hard)

Three runs in `grid-rnn-sdq` with different discretization thresholds:

| Configuration | novelty\_low / high | Acc | Acc++ | Loss | Steps |
|---|---|---|---|---|---|
| grnn disc gate (base) | default | 0.621 | 0.306 | 1.002 | ~63M |
| grnn disc gate 0.7 (high threshold) | —  / 0.7 | 0.647 | 0.343 | 0.960 | ~69M |
| grnn disc gate 0.1 / 0.4 / 0.6 | 0.1 / — | 0.655 | 0.299 | 1.001 | ~40M |

All three NoveltyGate variants are significantly worse than baseline grnn 2/1 (Acc=0.734). The discrete straight-through gate hinders training: the model cannot smoothly tune the degree of message blending. Raising the upper threshold to 0.7 gives a small improvement (Acc=0.647 vs 0.621), but overall discretization hurts more than it helps.
