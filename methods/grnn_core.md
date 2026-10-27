# GridRnn (grnn_core)

The reference Grid RNN core used for all current `--model=grnn` runs (registered in
`knitwork/models/utils.py` as `grnn -> grnn_core.GridRnn`). Instead of one wide recurrent
state, it keeps an `L x C` grid of **independent narrow GRU cells** and lets them exchange
information through a learned attention router. A plain GRU of the same parameter budget
has to squeeze everything into a single vector; here the state is split into `C` columns
that each keep their own thing, and a routing layer decides, per step, which column reads
from which. The whole model stays a recurrent network — cost per token is constant, there
is no growing KV cache — but it gains a content-addressed communication channel that a
monolithic GRU does not have.

## State and data flow

State is a dict with two entries:

```python
h    # [L, C, B, H] — hidden state of every cell in the grid
out  # [C, B, H]    — h[-1], the top layer, used as next step's message pool
```

One step of `forward` (`x` arrives as `[n_inputs, B, H]`; `H` equals the embedding size,
`TokenModel` sets `embedding_size = rnn.hidden_size`):

```python
msg = torch.cat([x_int, x_ext], dim=0)          # [C + n_inputs, B, H]
for layer in range(self.n_layers):
    hl = h[layer]
    x, comm_info = self.attn[layer](hl, msg, msg)  # queries = this layer's state
    hln = self.cells(layer, x, hl)                 # GRU update, [C, B, H]
    msg = hln                                      # next layer reads this layer's output
    hn.append(hln)

y = hn[-1][0]                                   # readout: column 0 of the top layer
state = {'h': hn, 'out': hn[-1]}
```

Two things are worth spelling out because they are easy to misread:

- **Within a step, layers are feed-forward.** Layer `l` queries the message pool produced
  by layer `l-1` *at the same timestep*. So depth adds within-step compute, like layers of
  an MLP, not extra memory horizons.
- **Recurrence closes through the top.** The pool the *first* layer sees is
  `cat([state['out'], x_ext])`, and `state['out']` is the top layer's previous output. The
  top layer therefore feeds back into the bottom layer with a one-step delay, which is the
  only path by which depth becomes temporal depth.
- Queries are `C`, keys/values are `C + n_inputs`. The external input participates as an
  extra "column" that everyone may attend to, but that nobody writes to.

## GRU banks (`bank=0|1|2`)

`GruBank`, `GruBank1`, `GruBank2` are three implementations of the same thing: `L x C`
independent GRU cells, evaluated one layer at a time with batched matmuls. All three are
mathematically identical to `nn.GRUCell` and use its initialization (`U(-1/sqrt(H), 1/sqrt(H))`),
including the standard update written in the fused form:

```python
return (h - new_gate) * update_gate + new_gate   # == (1 - z) * n + z * h
```

They differ only in memory/copy trade-offs:

| `bank` | Class | Layout |
|---|---|---|
| 0 | `GruBank` | naive: separate `weight_ih` / `weight_hh`, two `bmm` of `[C,B,H] @ [C,H,3H]` |
| 1 | `GruBank1` | `x` and `h` concatenated along the **batch** dim of `bmm`, one call over `2C` |
| 2 | `GruBank2` | `x` and `h` concatenated along the **feature** dim; `r,z` from one `[C,B,2H] @ [C,2H,2H]`, then `n` from a second `bmm` |

All production configs use `bank: 2`. It computes the reset/update gates first and only
then builds `[x; r*h]` for the candidate, which halves the size of the largest intermediate
at the cost of two concats.

## Message passing (`mha=0|1|2`)

| `mha` | Class | Idea |
|---|---|---|
| 0 | `MessagePassingLayer` | plain `nn.MultiheadAttention` + LayerNorm, `out_proj` initialized near zero so the initial message is negligible |
| 1 | `MessagePassingLayer1` | same, but `W_k = W_q` (initial content-based self-preference) and `W_v = I`, `out_proj ≈ I` so the routed vector passes through unchanged |
| 2 | `StochasticMessagePassingLayer` | the production router — reimplements attention by hand to expose the routing distribution |

Only `mha=2` returns communication diagnostics, and only `mha=2` accepts `n_q`/`n_kv`
separately (the other two take `n_participants` and are therefore not usable with an
asymmetric key set — they are kept for reference, not for current configs).

## Key mechanism: `StochasticMessagePassingLayer`

The router borrows `nn.MultiheadAttention`'s parameters but runs the math explicitly:

```python
q = q + self.ids[0][:Cq]          # learnable per-column identities (query side)
k = k + self.ids[1][:Ckv]         # and key side, so columns are distinguishable
q = F.silu(F.linear(q, W_q, b_q))  # SiLU on all three projections
k = F.silu(F.linear(k, W_k, b_k))
v = F.silu(F.linear(v, W_v, b_v))

logits = q @ k.transpose(-2, -1) / sqrt(head_dim)
if self.training and self.noise_std > 0.0:
    logits = logits + self.noise_std * torch.randn_like(logits)   # stochastic routing
beta = F.softplus(self.pi_logtemp)        # learnable inverse temperature, per query column
pi_route = torch.softmax(beta * logits, dim=-1)
msg = F.linear((pi_route @ v).reshape(Cq, B, H), out_proj.weight, out_proj.bias)
```

Four deliberate choices:

- **Learnable identities.** Columns are otherwise interchangeable — the grid has no
  positional structure of its own, so `ids` is what makes "column 3" a stable address.
- **Learnable temperature `beta`.** `pi_logtemp` has shape `[1, 1, n_q, 1]`, i.e. each
  *receiving* column learns how sharp its own routing is: some columns can commit to a
  single source, others can average. Initialized to `softplus(x) = 1`, i.e. ordinary softmax.
- **Routing noise.** `noise_std` perturbs the logits during training only. It keeps the
  argmax from locking in early and acts as exploration over the discrete routing choice.
- **Near-identity initialization.** `W_v = I` and `out_proj = I + N(0, small)` mean that at
  step zero a message is (a SiLU of) the selected column's own state, not a random
  projection of it. `W_q`/`W_k` get Xavier, biases zero.

The message is LayerNormed (`ln_msg`) and handed to the GRU bank as its input `x` — it does
**not** bypass the cell as a residual.

## Communication regularizers

Only computed in training mode, returned in `info` and consumed by the runners
(`knitwork/exps/{sdq,text}/run.py`), which add `loss_weight * comm_loss - entropy_weight * comm_entropy`
to the task loss and log them as `L_comm` / `H_comm`:

```python
prob_comm = 1.0 - pi_route.diagonal(dim1=-2, dim2=-1)   # mass NOT sent to self
comm_loss = prob_comm
if Ckv > Cq:
    extra_w = torch.arange(Cq).view(1, -1)
    x_ext_prob_weighted = pi_route[..., -1] * (extra_w * 2 - 1)
    comm_loss = prob_comm + x_ext_prob_weighted
entropy = -(pi_route * log pi_route).sum(-1)
info = {'comm_loss': comm_loss.mean(), 'comm_entropy': normalize_entropy(entropy.mean(), Ckv)}
```

- `prob_comm` is a **cost of talking**: minimizing it pushes each column toward its own
  diagonal entry, so cross-column communication has to earn its place against the task loss.
- The `extra_w * 2 - 1` term prices attention to the *external input* per column:
  the weight is `-1` for column 0 (it is rewarded for reading the input) and `+1, +3, +5, ...`
  for the rest (they are increasingly penalized). This is what makes column 0 the
  input/readout column and pushes the others toward being internal memory.
- `comm_entropy` is normalized by `log(Ckv)` and enters with a *negative* weight, i.e. it is
  a bonus that counteracts premature collapse to a hard route.

## Hyperparameters

| Parameter | Notes |
|---|---|
| `n_layers`, `n_columns` | the grid; `n_columns > 1` is asserted. `LxC` is the main capacity knob |
| `hidden_size` | per-cell width; **silently rounded down** to a multiple of `n_attn_heads` |
| `n_attn_heads` | 4 in every current config |
| `n_inputs`, `n_outputs` | must be `<= n_columns`. `n_inputs` extends the key set; `n_outputs` is stored and asserted but the readout is hardwired to `hn[-1][0]` |
| `bank` | 0/1/2 GRU-bank layout, production is 2 |
| `mha` | 0/1/2 router, production is 2 |
| `noise_std` | routing noise, train only. 0.1 in the small configs, 0.02 (SDQ) / 0.05 (text8) in the ~10M ones |
| `ln_msg` | LayerNorm on the message, `true` everywhere |
| `communication.loss_weight` / `entropy_weight` | global config section, `5e-2` / `5e-3` |

Reference ~10M-parameter configurations (`exps/*/config/large.yaml`), all with
`n_attn_heads: 4`, `bank: 2`, `mha: 2`:

| Config | `n_layers` | `n_columns` | `hidden_size` |
|---|---:|---:|---:|
| `grnn_L1C8` | 1 | 8 | 440 |
| `grnn_L2C4` | 2 | 4 | 424 |
| `grnn_L2C8` | 2 | 8 | 312 |
| `grnn_L2C16` | 2 | 16 | 224 |
| `grnn_L3C4` | 3 | 4 | 344 |

## Dead code

`cell_forward`, `_cell_input_dim` and `_prepare_grid_input` are inherited from an earlier
revision and are never called. They reference attributes this class never assigns
(`self.embedding_size`, `self.use_postmsg`, `self.self_feeding`) and would raise
`AttributeError` if invoked — do not use them as documentation of current behavior.
