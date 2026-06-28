#import "_template_html.typ": template
#show: template.with(title: "grnn_fw")

= GridRnnFW
GridRNN with Fast Weights (Ba et al. 2016). Addresses the problem of
short-term associative memory: a standard GRU remembers the past only
through the hidden state, while fast weights store explicit pairwise
associations between column states in a matrix `A`. Matrix `A` is
updated by Hebbian rule at each step and read via content-based
retrieval by each column\'s query, creating \"inter-step\" memory faster
than parameters but slower than activations.

== Key Mechanism
Hebbian update of matrix `A` and retrieval by query:

```python
# Hebbian write: outer product of value and key for each column  [batch, hidden, hidden]
delta_A = zeros_like(A)
for col_j in range(n_cols):
    k_j = F.normalize(k[col_j], dim=-1)   # (batch, hidden)
    v_j = F.normalize(v[col_j], dim=-1)
    delta_A += bmm(v_j.unsqueeze(2), k_j.unsqueeze(1))

A_new = decay * A + (fw_lr / n_cols) * delta_A   # exponential decay

# Content-based retrieval: each column reads from A via its query
msgs = []
for col_i in range(n_cols):
    q_i = F.normalize(q[col_i], dim=-1)
    msgs.append(bmm(A_new, q_i.unsqueeze(2)).squeeze(2))  # (batch, hidden)
h_msg = stack(msgs, dim=0)   # (cols, batch, hidden)
```

Each column writes the association `v ⊗ k` into matrix `A`, then all
columns read from it via their queries `q`. Parameters `k`, `q`, `v` are
learnable linear projections (analogous to heads in attention, but
without softmax).

== Important Implementation Details
#strong[Model state] — extended with matrix `A` per layer:

```python
# state = (h, A)
# h : (n_layers, n_cols, batch, hidden)
# A : (n_layers, batch, hidden, hidden)
```

#strong[Pseudo-attention weights] for compatibility with the visualizer
are computed from batch-averaged keys and queries:

```python
# attn_w[i,j] ≈ similarity between col_i query and col_j key  [n_cols, n_cols]
q_mat = F.normalize(q.mean(dim=1), dim=-1)
k_mat = F.normalize(k.mean(dim=1), dim=-1)
scores = matmul(q_mat, k_mat.T)
attn_w = softmax(scores / scale, dim=-1)
```

#strong[Gated merge] — as in the base grnn.py, applied on top of
fast-weight retrieval:

```python
g = sigmoid(gate_lin(cat([hl_n, msg], dim=-1)))   # (cols, batch, 1)
hl_n = (1 - g) * hl_n + g * msg                   # (cols, batch, hidden)
```

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`fw_decay`],
  [λ — exponential decay of matrix `A`; 0.9 \= slow forgetting, \<0.5 \=
  short-term memory],
  [`fw_lr`],
  [η — write rate into `A`; scales the strength of each Hebbian update],
  [`col_identities`],
  [If True — adds learnable identity vectors to column keys and
  queries],
)

== Results
=== SDQ (Store-Distract-Query, hard)
#table(
  columns: 9,
  inset: 6pt,
  [Experiment], [H], [Cols / Layers], [fw\_decay], [fw\_lr], [Acc],
  [Acc++], [Loss], [Steps],
  [grnn fw sdq (`grid-rnn-sdq`)],
  [128],
  [3 / 2],
  [0.9],
  [0.5],
  [#strong[0.703]],
  [#strong[0.407]],
  [#strong[0.760]],
  [~48M],
)

Fast Weights achieve Acc\=0.703 in ~48M steps, close to the baseline
grnn 2/1 (Acc\=0.734 in 85M). The Hebbian matrix A provides an explicit
key-value mechanism on top of the standard GRU state. However, the model
topology (3 columns, 2 layers) without additional columns cannot fully
compete with deeper configurations (Acc\=0.960 at 4 cols / 3 layers).
