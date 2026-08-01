# GridRnn

GridRnn addresses associative memory and language modeling by organizing recurrent cells into a two-dimensional grid of "layers × columns". Instead of a single hidden vector, the model maintains a state matrix `[layers, cols, batch, hidden]`, where columns at each layer exchange information through an attention mechanism (post- or pre-messaging). The input token is fed only to the zeroth column of the first layer; other columns in the first layer receive a zero dummy input but are enriched through attention. Predictions are built from the state of the top layer, zeroth column.

## Key mechanism

Post-messaging: after an independent GRU step across all columns, attention is run and its result is blended in via learnable gates.

```python
# hl_n: [cols, batch, hidden] — states after GRU step
hl_n = torch.stack([cell_forward(cells, x, hl, ix_col=c)
                    for c in range(self.n_columns)], dim=0)

msg, attn_w = attn(hl_n, return_weights=return_attn)
g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
hl_n = (1 - g) * hl_n + g * msg          # gated mixing
```

Gate `g` allows each column to independently decide how much to incorporate the aggregated message from neighbors.

## Important implementation details

`MessagePassingLayer` adds learnable column identifiers to queries and keys, so attention can distinguish participants:

```python
# ids: [n_cols, 1, dim] — learnable per-column bias
if self.ids is not None:
    qh = kh = qh + self.ids
h_mixed, attn_w = self.mha(qh, kh, vh, average_attn_weights=True)
return self.norm(h_mixed), attn_w
```

Initializing `out_proj` close to zero makes initial messages negligible, stabilizing the start of training.

---

The first-layer input is stored as a list (different dimensionalities across columns), while from the second layer onward it is a dense tensor `[cols, batch, hidden]`:

```python
def _prepare_grid_input(self, x):
    xl = [x]   # col 0: embedding, shape [batch, embed_dim]
    dummy = torch.zeros(bsz, 1, device=x.device, dtype=x.dtype)
    for _ in range(1, self.n_columns):
        xl.append(dummy)   # cols 1..C-1: dummy 1-dim input
    return xl
```

This allows different input sizes for the first column (embedding) and all others.

## Hyperparameters

| Parameter | Description |
|---|---|
| `base_hidden_size` | Reference hidden size for a single-layer GRU; `hidden_size` is chosen automatically to match the parameter count |
| `n_columns` | Number of columns in the grid; must be > 1 |
| `messaging` | `"post"` — attention after GRU step (with gates); `"pre"` — attention before GRU step (concatenation) |
| `col_identities` | Whether to add learnable column identifiers to attention |
| `n_attn_heads` | `hidden_size` is rounded down to the nearest multiple of this number |

## Results

GridRnn was tested in two main configurations: small (~78K parameters, 2 columns / 1 layer) and full (H=128, 4–5 columns / 3 layers).

### SDQ (Store-Distract-Query, hard)

| Configuration | H | Col. / Layers | Acc | Acc++ | Loss | Steps |
|---|---|---|---|---|---|---|
| 2 col., 1 layer (`grid-rnn-sdq`) | 115 | 2 / 1 | 0.734 | 0.494 | 0.686 | ~85M |
| 4 col., 3 layers (`sdq-gru`) | 128 | 4 / 3 | **0.960** | **0.917** | **0.107** | ~57M |

### Text experiments

| Configuration | H | Col. / Layers | Dataset | Acc | BPC | PPL | Steps |
|---|---|---|---|---|---|---|---|
| 2 col., 1 layer (`grid-rnn-text`) | 128 | 2 / 1 | text8 | 0.553 | 2.088 | 4.25 | ~34M |
| 2 col., 1 layer (`grid-rnn-text`) | 128 | 2 / 1 | shakespeare | 0.589 | 1.954 | 3.88 | ~146M |
| 4 col., 3 layers (`text-gru`) | 128 | 4 / 3 | shakespeare | **0.629** | **1.721** | **3.30** | ~70M |
| 5 col., 3 layers (`text-gru`) | 116 | 5 / 3 | shakespeare | 0.622 | 1.756 | 3.38 | ~70M |

Moving from 2 to 4 columns with a comparable parameter count yields +0.42 in Acc++ on SDQ and a BPC drop from 1.954 to 1.721 on shakespeare (PPL: 3.88→3.30). Increasing the number of columns matters more than increasing the number of layers: adding a 5th column while reducing H barely affects quality.
