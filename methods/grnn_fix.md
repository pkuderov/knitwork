# GridRnnFix

Fixed variant of Grid RNN's inter-column attention exchange. The base `grnn` suffers from four defects (see `architecture_analysis.md` sec 2): convex averaging pulls columns together (col_sim ~ 0.93), LayerNorm after a tiny-init projection turns a "negligible" message into scale-1 noise, columns without input are born collapsed, and the message overwrites the GRU's single recurrent state. GridRnnFix applies all sec 7.1 fixes at once: additive message, gate closed at init, no post-norm, learnable temperature beta, input into all columns via orthogonal projections, protected recurrent state, and concat-readout over columns.

## Key mechanism

The message is added to the layer output, not to the recurrent state — column memory is never mixed in:

```python
# pure recurrence, then additive gated message  [C, B, H]
hl_n = torch.stack([cells[ic](x[ic], hl[ic]) for ic in range(self.n_columns)], dim=0)
msg, attn_w = attn(hl_n, return_weights=return_attn)
g = torch.sigmoid(attn_gate(torch.cat([hl_n, msg], dim=-1)))
o = hl_n + g * msg      # goes up to the next layer / readout
h_n.append(hl_n)        # recurrent state stays unmixed
```

Each column's recurrent dynamics stay clean (analogous to the protected `c` in an LSTM), while the message-enriched representation `o` feeds the next layer and the readout.

## Important implementation details

Symmetry breaking: each column receives the embedding through its own orthogonal projection (in `grnn` only col0 sees the input, the rest start identical):

```python
self.col_input_projs = nn.ModuleList(
    nn.Linear(embedding_size, embedding_size, bias=False) for _ in range(n_columns)
)
x = torch.stack([proj(x) for proj in self.col_input_projs], dim=0)  # [C, B, E]
```

The gate is closed at init — the model turns on attention as it becomes useful, instead of fighting noise from the start:

```python
nn.init.constant_(gate.bias, -3.0)   # sigmoid(-3) ~ 0.05 at init
```

`ColumnAttention` is Hopfield-style with no post-norm: a tiny-init `out_proj` genuinely makes the message zero at init (in `grnn` LayerNorm cancelled this out), and beta is learned per head:

```python
beta = self.log_beta.exp().view(self.num_heads, 1, 1, 1)
attn = torch.softmax(beta * torch.matmul(q, k.transpose(-2, -1)), dim=-1)
nn.init.normal_(self.out_proj.weight, 0.0, 0.001)   # no norm after this
```

Readout — concatenation of top-layer columns instead of reading a single column `h[-1][0]`:

```python
self.head = nn.Linear(self.n_columns * self.hidden_size, output_size)
```

## Hyperparameters

| Parameter | Description |
|---|---|
| `hidden_size` | 64 at 2L x 3C gives ~201K parameters (SDQ and text8) |
| `n_layers` | >=2 is meaningful: with 1 layer the message only affects the readout |
| `n_attn_heads` | H is truncated to a multiple of head count; beta is learned per head |
