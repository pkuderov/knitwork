#import "_template.typ": template
#show: template.with(title: "fusion_cells")

= BatchedHGRUColumns
`BatchedHGRUColumns` solves the problem of efficient parallel
computation of multiple HGRU columns in Grid RNN without a Python loop
over columns. Instead of `n_cols` separate `nn.Linear` modules, all
weight matrices are packed into three-dimensional parameters of shape
`[n_cols, out, in]`, allowing all columns to be processed in a single
`torch.bmm` call with ~3–5× speedup over an element-wise loop. Each
column uses its own learnable parameter `beta` — a lower bound on the
forget gate that defines its temporal scale.

== Key mechanism
```python
# x: (B, in)  h: (B, n_cols, hid)
x_t = x.T.unsqueeze(0).expand(self.n_cols, -1, -1)   # (n_cols, in, B)
h_t = h.permute(1, 2, 0)                               # (n_cols, hid, B)

# batched matmul for all columns simultaneously
def gate_x(W, b):
    out = torch.bmm(W, x_t).permute(0, 2, 1)           # (n_cols, B, hid)
    return out + b.unsqueeze(1) if b is not None else out

lam_t = torch.sigmoid(gate_x(W_f, b_f) + gate_h(U_f)) * (1.0 - betas) + betas
return (lam_t * h_perm + (1.0 - lam_t) * c_t).permute(1, 0, 2)   # (B, n_cols, hid)
```

`beta` is parameterized via logit (`beta_raw`), guaranteeing a value in
`(0, 1)` and allowing each column to learn its own temporal niche.

== Important implementation details
#strong[Hierarchical forget gate structure.] The formula
`λ = sigmoid(...) * (1 - β) + β` ensures a lower bound of λ at level β:
even with maximum \"forgetting\", the column retains fraction β of its
previous state, creating a hierarchy of temporal scales:

```python
betas = torch.sigmoid(self.beta_raw).view(n_cols, 1, 1)
lam_t = torch.sigmoid(gate_x(W_f, b_f) + gate_h(U_f)) * (1.0 - betas) + betas
```

#strong[Weight initialization.] Each weight matrix of each column is
initialized orthogonally and independently, reducing initial correlation
between columns:

```python
for name in ['W_f', 'W_o', 'W_c', 'U_f', 'U_o', 'U_c']:
    p = getattr(self, name)
    for i in range(self.n_cols):
        nn.init.orthogonal_(p[i])
```

#strong[`BatchedReservoirColumns`.] The second class in the file
implements frozen GRU columns with spectral radius normalization of
recurrent matrices. Parameters have `requires_grad=False` — the network
operates as an echo state machine (ESN), adding nonlinear diversity
without training.

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`beta_inits`],
  [Initial β values for each column; different values define different
  temporal scales from the first step],
  [`learnable_beta`],
  [If `False` — β is fixed and the hierarchy is set manually via
  `beta_inits`],
  [`use_layer_norm`],
  [LayerNorm applied to candidate state `c_raw` before `tanh`,
  stabilizing dynamics with large hidden sizes],
  [`spectral_radii`],
  [(for `BatchedReservoirColumns`) Spectral radius of the recurrent
  matrix for each reservoir column; values \< 1 ensure echo-state
  behavior],
)

== Results
`BatchedHGRUColumns` and `BatchedReservoirColumns` are components used
inside `grnn_fusion`. No standalone experiments were conducted. Results
for the integrated model are given in
#link("grnn_fusion.md")[grnn\_fusion.md]: SDQ Acc\=0.831, Acc++\=0.708.
