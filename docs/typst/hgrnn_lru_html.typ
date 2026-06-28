#import "_template_html.typ": template
#show: template.with(title: "hgrnn_lru")

= HopfieldGridLRU
The model combines three ideas to improve Grid RNN on associative memory
and language modeling tasks: (1) Linear Recurrent Unit (LRU) with
diagonal parameterization instead of LSTM — fewer parameters, stable
gradients on long sequences; (2) Modern Hopfield Network for message
passing between columns; (3) an auxiliary contrastive associative loss
that explicitly trains the network to separate \"write\" and \"read\"
patterns. LRU operates in complex state space: state is stored as
`[Re | Im]` — a compact form from which the Hopfield layer sees only the
Re part.

== Key mechanism
LRUCell parameterizes the diagonal transition matrix via `nu_log` and
`theta_log`, ensuring |λ| ∈ (0,1) by construction:

```python
# stable reparametrization of lambda: |lambda| = exp(-exp(nu)), angle = exp(theta)
r      = torch.exp(-torch.exp(self.nu_log))     # |lambda| in (0,1)
theta  = torch.exp(self.theta_log)
lam_re = r * torch.cos(theta)
lam_im = r * torch.sin(theta)
# gamma normalizes input contribution inversely to memory strength
gamma  = torch.sqrt(torch.clamp(1.0 - r * r, min=1e-6))

# complex state update (expanded to 4 real ops, no overhead)
new_re = lam_re * h_re - lam_im * h_im + gamma * bx_re  # [B, H]
new_im = lam_re * h_im + lam_im * h_re + gamma * bx_im  # [B, H]
# output: Re(C * h_new) + D * x
y = self.C_re(new_re) - self.C_im(new_im) + self.D(x)
```

`gamma = sqrt(1 - |λ|²)` — the key difference from S4: the stronger the
\"memory\" (|λ| → 1), the weaker the influence of the new input.

== Important implementation details
#strong[Detach Im part when assembling state.] The Im part does not
participate in the Hopfield exchange and gate, so its gradient is
already accounted for inside LRUCell. Accumulating the graph through Im
on long rollouts would lead to OOM:

```python
hl_im_stop  = hl_full[:, :, self.hidden_size:].detach()   # Im-part detached
hl_full_new = torch.cat([hl_re_gated, hl_im_stop], dim=-1)
```

#strong[Associative contrastive loss.] Penalizes closeness of
representations for random \"write/read\" pairs and encourages closeness
for matching pairs:

```python
sim_matrix = torch.matmul(h_query, h_store.T)   # (n, n) cosine similarity
cos_pos    = sim_matrix.diagonal()               # positive pairs
cos_neg    = sim_matrix.masked_fill(eye_mask, -1.0).max(dim=-1).values
loss       = (-cos_pos + F.relu(cos_neg + margin)).mean()
```

#strong[PositionwiseFFN after each LRU.] LRUCell is linear recurrence.
FFN with Pre-LN and GELU adds block nonlinearity, as in Orvieto et al.
2023:

```python
# Pre-LN + GELU FFN with residual connection
self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * expansion),
                         nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(dim * expansion, dim), nn.Dropout(dropout))
def forward(self, x): return x + self.net(x)
```

#strong[reset\_state without clone.] State reset is implemented via
multiplication by a keep mask — cheaper than `clone()` + indexing:

```python
keep = (~reset_mask).to(dtype=state.dtype, device=state.device)
return state * keep.view(1, 1, -1, 1)   # broadcast over (layers, cols, batch, 2*hid)
```

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`lru_r_min`, `lru_r_max`],
  [],
  [`lru_max_phase`],
  [Maximum initial angle θ; 2π/3 gives diversity of initial
  frequencies],
  [`ffn_expansion`],
  [Expansion factor in FFN (usually 2–4)],
  [`attn_dropout`],
  [Dropout on Hopfield layer attention weights],
  [`log_beta`],
  [Learnable attention scale (one per head); initialized as
  `log(1/√d_k)`],
)

== Results
=== SDQ (Store-Distract-Query, hard)
#table(
  columns: 7,
  inset: 6pt,
  [Configuration], [H], [Cols / Layers], [Acc], [Acc++], [Loss],
  [Steps],
  [grnn\_hopfield H\=128 (`sdq-hgru-hopfield`)],
  [128],
  [4 / 3],
  [#strong[0.967]],
  [#strong[0.932]],
  [#strong[0.087]],
  [~40M],
  [grnn\_hopfield 5col H\=116 (`sdq-hgru-hopfield`)],
  [116],
  [5 / 3],
  [0.628],
  [0.284],
  [0.974],
  [~12M],
)

The 4 col / 3 layer configuration is the best result among all tested
models on SDQ: Acc\=0.967, Acc++\=0.932 in just 40M steps, outpacing
grnn\_hgru (0.965 at 45M) and the base grnn (0.960 at 57M). The
combination of LRU (diagonal recurrence) + Hopfield attention (learnable
β) + contrastive loss delivers the maximum Acc++. The 5-column run was
stopped early (12M).

=== Text experiments (shakespeare)
#table(
  columns: 7,
  inset: 6pt,
  [Configuration], [H], [Cols / Layers], [Acc], [BPC], [PPL], [Steps],
  [grnn\_hopfield H\=128 (`text-hgru-hopfield`)],
  [128],
  [4 / 3],
  [#strong[0.636]],
  [#strong[1.686]],
  [#strong[3.22]],
  [~70M],
  [grnn\_hopfield 5col H\=116 (`text-hgru-hopfield`)],
  [116],
  [5 / 3],
  [0.635],
  [1.690],
  [3.23],
  [~70M],
  [grnn\_lru\_hop H\=104 (`text-lru`)],
  [104],
  [3 / 3],
  [0.330],
  [3.844],
  [14.35],
  [~17M],
)

On shakespeare the 4/3 configuration results (BPC\=1.686, PPL\=3.22) are
nearly identical to grnn\_hgru (BPC\=1.686) — Hopfield vs standard
attention shows no difference in text experiments. The grnn\_lru\_hop
variant in `text-lru` was stopped very early and diverged (BPC\=3.844,
PPL\=14.35).

=== MIKASA / POPGym (stopped at 15M/200M, ~7.5%)
#table(
  columns: 6,
  inset: 6pt,
  [Environment], [Memory type], [EpRet], [H], [FPS], [Result],
  [RepeatFirstEasy],
  [Object],
  [~−0.4…−0.6],
  [1.18],
  [338],
  [oscillates, stopped],
)

EpRet oscillates with large amplitude (−0.4 ↔ −0.6) with no clear trend.
Entropy H\=1.18 is consistently high — the model actively explores. The
high entropy compared to grnn\_lru (H\=1.07) is explained by Hopfield
attention: learnable β is slower to determinize the attention
distribution. Nevertheless, no progress in EpRet over 7.5% of training.

Run stopped early. RepeatPreviousEasy (\#024) did not start (was
pending).
