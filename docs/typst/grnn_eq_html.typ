#import "_template_html.typ": template
#show: template.with(title: "grnn_eq")

= EquilibriumGridRnnCoT (grnn\_eq)
The model addresses the problem of limited \"thinking depth\" within a
single time step: a standard Grid RNN makes exactly one pass through
each layer, while some inputs require more computation steps. The core
idea combines three mechanisms: `EquilibriumCell` searches for an
approximate fixed point of a GRU by iteratively applying
`h* = GRU(x, h*)` until convergence; `ACT` (Adaptive Computation Time)
dynamically decides when to stop iterations for each example in the
batch; `ChainOfThoughtGRU` accumulates a separate \"thought\" state on
top of the upper grid layer, giving the model an explicit working buffer
for multi-step reasoning.

== Key Mechanism
```python
# iterative equilibrium search with ACT halting
for act_step in range(self.max_eq_iters):
    h_layer_new = torch.stack([cell.cell(x_in[ic], h_layer[ic])
                               for ic, cell in enumerate(cells_row)], dim=0)
    h_pool  = h_layer_new.mean(dim=0)     # [batch, hidden]
    p_halt  = halter(h_pool)              # [batch,] halting probability

    is_last = (act_step == self.max_eq_iters - 1)
    p_use   = torch.where(is_last | (halt_acc + p_halt >= 1.0 - act_eps),
                          1.0 - halt_acc, p_halt)

    w      = p_use.unsqueeze(0).unsqueeze(-1)   # [1, batch, 1]
    h_acc += w * (h_layer_new if is_last else h_layer_new.detach())
    halt_mask = halt_mask | (halt_acc >= 1.0 - act_eps)
    if halt_mask.all():
        break
```

At each step, ACT weights cell outputs by halting probability `p_use`;
intermediate steps are detached — only the final step carries gradient,
saving memory.

== Important Implementation Details
#strong[Chain-of-Thought buffer:]

```python
# thought state accumulates across time steps
h_top   = h_new[-1, 0]                          # [batch, hidden]
thought = self.cot(h_top, thought)               # [batch, thought_size]
y       = self.head(torch.cat([h_top, thought], dim=-1))
```

`ChainOfThoughtGRU` is a separate GRUCell with LayerNorm that updates
the `thought` vector from step to step. The head reads the concatenation
`[h_top; thought]`, allowing the model to use information from previous
\"thoughts\".

#strong[HaltingUnit:]

```python
# initialized with near-zero weights and bias=-2 => initially almost never halts
self.proj = nn.Linear(hidden_size, 1)
nn.init.zeros_(self.proj.weight)
nn.init.constant_(self.proj.bias, -2.0)
```

The bias initialization of `-2` corresponds to a starting halting
probability of `sigmoid(-2) ≈ 0.12`, encouraging the model to perform
several iterations before converging.

#strong[ACT loss:]

```python
# ponder penalty: minimizes unnecessary computation steps
def act_loss(self, act_iters_list) -> torch.Tensor:
    total = sum(it.float().mean() for it in act_iters_list)
    return self.act_loss_weight * total
```

The penalty is proportional to the average number of iterations across
all layers and examples — a trade-off between accuracy and computational
cost.

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`max_eq_iters`],
  [Maximum number of fixed-point search iterations; limits computation
  depth],
  [`eq_tol`],
  [Convergence threshold on the difference norm `‖h_new − h‖`; used only
  in `EquilibriumCell.forward`, not in the ACT loop of the main grid
  step],
  [`act_eps`],
  [ACT tolerance: stop when `halt_acc ≥ 1 − eps`; smaller value →
  stricter],
  [`act_loss_weight`],
  [Weight of the iteration count penalty; typically `1e-3`],
  [`thought_size`],
  [Size of the `thought` vector; defaults to `hidden_size`; can be
  reduced to save parameters],
)

== Results
=== Text experiments (shakespeare)
#table(
  columns: 8,
  inset: 6pt,
  [Experiment], [H], [Cols / Layers], [max\_eq\_iters], [Acc], [BPC],
  [PPL], [Steps],
  [grnn\_eq H\=128 text (`text-gru-eq`)],
  [128],
  [4 / 3],
  [4],
  [#strong[0.588]],
  [#strong[1.950]],
  [#strong[3.86]],
  [~19M],
)

Run terminated early (19M out of 70M steps). BPC\=1.950 is comparable to
the baseline grnn 2/1 on shakespeare (BPC\=1.954 over 146M steps) —
similar quality with significantly fewer steps, but higher per-step
compute due to the ACT loop. The second run (`69f2a083`) recorded no
metrics.
