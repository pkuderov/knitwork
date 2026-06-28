#import "_template_html.typ": template
#show: template.with(title: "grnn_eq1")

= EquilibriumGridRnnCoT
GridRNN with Adaptive Computation Time (ACT) and Chain-of-Thought (CoT).
Addresses the problem of fixed computation depth: instead of a single
pass through the grid, each layer performs a variable number of
iterations until reaching a fixed point (equilibrium). The ACT mechanism
dynamically decides when to stop, and the CoT-GRU accumulates
\"thoughts\" on top of the upper layer as slow long-term memory. Three
auxiliary losses drive the network toward equilibrium (MSE residual),
computational efficiency (ponder cost), and uniform participation of all
columns in attention (entropy regularizer).

== Key Mechanism
ACT loop with weighted state accumulation and adaptive halting:

```python
# ACT loop per layer  [cols, batch, hidden]
for act_step in range(self.max_eq_iters):
    h_layer_new = stack([cell(x_in[ic], h_layer[ic]) for ic, cell in enumerate(cells_row)])

    # Anderson mixing for faster convergence
    if self.anderson_beta > 0.0 and act_step >= 1:
        h_layer_new = h_layer_new + self.anderson_beta * (h_layer_new - h_layer_prev)

    # cross-column attention every attn_every steps
    if is_attn_step:
        msg, attn_w = attn_mod(h_layer_new, return_weights=True)
        g = sigmoid(gate_mod(cat([h_layer_new, msg], dim=-1)))
        h_layer_new = (1.0 - g) * h_layer_new + g * msg

    # halting probability
    p_halt = halter(h_layer_new.mean(dim=0))      # (batch,)
    p_use  = where(is_last | halt_acc + p_halt >= 1 - eps, 1 - halt_acc, p_halt)

    # weighted accumulation  [1, batch, 1]
    h_acc = h_acc + p_use.unsqueeze(0).unsqueeze(-1) * h_layer_new
    halt_mask = halt_mask | (halt_acc >= 1 - eps)
    if halt_mask.all():
        break
```

At each ACT step, the halting probability `p_halt` is computed and
states are weighted-summed into `h_acc`. The loop terminates when all
examples in the batch have halted or `max_eq_iters` is exhausted.

== Important Implementation Details
#strong[Equilibrium residual loss] — a direct signal for reaching the
fixed point:

```python
# penalty: one more GRU step from h_acc should not change it  [cols, batch, hidden]
h_acc_detached = h_acc.detach()
h_check = stack([cell(x_in[ic], h_acc_detached[ic]) for ic, cell in enumerate(cells_row)])
residual_loss = F.mse_loss(h_check, h_acc_detached)
```

#strong[Ponder cost (FIX-1)] — normalized iteration count, not
accumulated probability (which always equals 1.0):

```python
# n_iters counts real steps per example  [batch,]
n_iters += (~halt_mask).float()
ponder_cost = n_iters.mean() / self.max_eq_iters   # range [1/max .. 1]
```

#strong[Chain-of-Thought] — a separate GRUCell on top of the upper
layer, output concatenated with `h_top` before the head:

```python
h_top   = h_new[-1, 0]                            # (batch, hidden)
thought = self.cot(h_top, thought)                 # GRUCell + LayerNorm
y       = self.head(cat([h_top, thought], dim=-1))
```

#strong[Column participation loss (FIX-5)] — penalty for non-uniform
attention between columns:

```python
# maximize entropy of mean attention weights → uniform column participation
attn_w_mean = stack(attn_w_accum).mean(0)         # (cols, cols)
col_entropy  = -(attn_w_mean * (attn_w_mean + 1e-8).log()).sum(-1).mean()
participation_loss = -col_entropy                  # minimize negative entropy
```

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`max_eq_iters`],
  [Maximum number of equilibrium iterations per layer; default 12],
  [`eq_tol`],
  [Residual norm threshold for convergence logging (does not affect
  halting — that is controlled by ACT)],
  [`act_eps`],
  [ACT epsilon: halt when `halt_acc >= 1 - eps`; default 0.01],
  [`act_loss_weight`],
  [Weight of ponder cost in total loss],
  [`eq_residual_weight`],
  [Weight of equilibrium MSE residual; default 1e-2],
  [`col_participation_weight`],
  [Weight of column entropy regularizer; default 1e-3],
  [`anderson_beta`],
  [Anderson mixing coefficient for faster convergence; 0.0 \= disabled],
  [`attn_every`],
  [Attention is applied every N iterations inside the ACT loop],
  [`thought_size`],
  [Size of the CoT state; defaults to `hidden_size`],
)

== Results
=== SDQ (Store-Distract-Query, hard)
Six runs in `grid-rnn-sdq`, configuration H\=128, 4 columns, 4 layers,
max\_eq\_iters\=12:

#table(
  columns: 6,
  inset: 6pt,
  [Run], [Acc], [Acc++], [Loss], [Steps], [Note],
  [grnn equilibr v.2 sdq eq metric],
  [0.636],
  [0.323],
  [1.014],
  [~76M],
  [best in series],
  [grnn equilibr v.2 sdq eq iters\=12 (3b26dae3)],
  [0.623],
  [0.296],
  [1.022],
  [~47M],
  [],
  [grnn equilibr v.2 sdq eq iters\=12 (48ba7c7d)],
  [0.394],
  [0.102],
  [1.640],
  [~6M],
  [early stop],
  [grnn equilibr v.2 sdq eq iters\=12 (67748060)],
  [0.336],
  [0.101],
  [1.821],
  [~3M],
  [early stop],
  [grnn equilibr v.1 sdq (3f16bb52)],
  [0.429],
  [0.105],
  [1.572],
  [~11M],
  [],
  [grnn equilibr v.1 sdq (c07d8d61)],
  [0.414],
  [0.106],
  [1.609],
  [~8M],
  [],
)

All variants are significantly worse than the baseline grnn 2/1
(Acc\=0.734). The best result — Acc\=0.636 at 76M steps — does not even
reach the baseline configuration threshold. Numerous early stops
indicate training instability with ACT+CoT+multiple auxiliary losses.
The adaptive iteration count mechanism provides no advantage on SDQ,
where the key factor is associative storage quality, not \"thinking
depth\".
