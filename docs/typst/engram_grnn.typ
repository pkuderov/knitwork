#import "_template.typ": template
#show: template.with(title: "engram_grnn")

= EngramGridRnn
Grid RNN where each cell is augmented with associative memory
(`EngramMemory`). At each step, the cell first retrieves a relevant
vector from memory via sparse cosine attention, concatenates it with the
input, and passes it to GRU. After updating the hidden state, the memory
is rewritten using Hebbian rule. Messages between columns are passed
through multi-head attention (`post` and `pre` modes).

== Key mechanism
Memory read and Hebbian write are implemented in `EngramMemory.forward`:

```python
# read → hebbian write  [B, H], [B, S, H]
def forward(self, h, M):
    r, attn = self.read(h, M)
    return r, self.write(h, M, attn), attn
```

`read` computes sparse cosine attention over slots, aggregates values by
weighted sum, and applies a read gate. `write` shifts slots toward the
current hidden state proportional to attention weights.

== Important implementation details
#strong[Sparse cosine attention] — top-K masking zeros out weak slots
before softmax:

```python
# sparse cosine attention over memory slots  [B, S]
q_norm = F.normalize(query.unsqueeze(1), dim=-1)   # [B, 1, H]
scores = (q_norm * F.normalize(M, dim=-1)).sum(-1)
threshold = scores.topk(self.top_k, dim=-1).values[:, -1:]
scores = scores.masked_fill(scores < threshold, float('-inf'))
attn = torch.softmax(scores, dim=-1)
```

#strong[Hebbian write] with gate and slot normalization:

```python
# Hebbian delta rule; w in [0,1] from write_gate  [B, S, H]
delta = h.unsqueeze(1) - M
lr = self.hebb_lr * w.unsqueeze(-1)
M_new = M + lr * attn.unsqueeze(-1) * delta
return M_new / M_new.norm(dim=-1, keepdim=True).clamp(min=1.0)
```

Normalization `clamp(min=1.0)` does not shrink short vectors, but
prevents explosive growth of long ones.

#strong[GRU input] — concatenation of the input token/message and the
retrieval vector `r`:

```python
# augment cell input with engram retrieval  [B, input_dim + H]
r, M_new, eng_attn = engram_row[col_i](h_prev, M_prev)
x_aug = torch.cat([x_input[col_i], r], dim=-1)
h_new = cells[col_i](x_aug, h_prev)
```

#strong[Post-message gate] mixes the original hidden state with column
messages:

```python
# gated message mixing across columns  [cols, B, H]
msg, attn_w = attn_msg(hl_n, return_weights=return_attn)
g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
hl_n = (1.0 - g) * hl_n + g * msg
```

== Hyperparameters
#table(
  columns: 2,
  inset: 6pt,
  [Parameter], [Description],
  [`n_engram_slots`],
  [Number of memory slots per cell; more slots — higher capacity, but
  more expensive attention],
  [`engram_top_k`],
  [How many slots participate in softmax; controls retrieval sparsity],
  [`engram_hebb_lr`],
  [Memory update rate via Hebbian rule; at high values memory is quickly
  overwritten],
  [`engram_gate_write`],
  [Enables write gate: the cell can suppress writing if the current
  input is irrelevant],
  [`messaging`],
  [`post` — messages after GRU step (with gate); `pre` — before GRU step
  (without gate)],
  [`col_identities`],
  [Adds column positional embeddings to the attention mechanism],
)

== Results
Configuration: H\=128, 4 columns, 4 layers, 16 engram slots,
hebb\_lr\=0.1, top\_k\=4.

=== SDQ (Store-Distract-Query, hard)
#table(
  columns: 5,
  inset: 6pt,
  [Experiment], [Acc], [Acc++], [Loss], [Steps],
  [grnn engram sdq (`grid-rnn-sdq`)],
  [#strong[0.819]],
  [#strong[0.664]],
  [#strong[0.439]],
  [~116M],
)

Engram memory gives a noticeable improvement over baseline grnn 2/1
(Acc\=0.734), but falls short of deeper configurations without Engram:
grnn 4/3 reaches Acc\=0.960, and grnn\_loss 4/4 — Acc\=0.862. The
Hebbian write mechanism is useful, but the gain from explicit memory
slots is offset by the growth in grid parameter count.

=== Text experiments
#table(
  columns: 6,
  inset: 6pt,
  [Experiment], [Dataset], [Acc], [BPC], [PPL], [Steps],
  [engram 4×4 text (`grid-rnn-text`)],
  [text8],
  [0.571],
  [2.077],
  [4.22],
  [~60M],
)

On text8 the result is close to baseline grnn 2/1 (BPC\=2.088). The
Engram mechanism does not give meaningful improvement on text tasks
within this training budget.
