# RIMs (Recurrent Independent Mechanisms)

Baseline for the modular-recurrence comparison (Goyal et al., ICLR 2021, arXiv 1909.10893). The hidden state is split
into `n_modules` independent LSTM cells. At every step the modules compete for the input through attention over
`[input, null]`: only the `k` modules that attend least to the null input are **active** and update their state, the
others keep it unchanged. Active modules then exchange information through attention among all modules.
Our `baseline/rims.py` is a compact step-wise reimplementation of the official `RIMCell`
(github.com/dido1998/Recurrent-Independent-Mechanisms) with the same `Core` interface as the other models; it is not a copy of that code.

## Key mechanism

```python
att  = (q(h) @ keys.T / sqrt(d_key)).softmax(-1)          # [B, n, S+1], last source is null
idx  = (1 - att[..., -1]).topk(k, dim=1).indices          # active modules
inp  = (att @ vals) * mask                                 # inactive modules get zero input
h_rnn, c_rnn = block_diag_lstm(inp, h, c)
h_g  = mask * h_rnn + (1 - mask) * h_rnn.detach()          # gradients only into active modules
h_new = mask * (comm_attention(h_g) + h_g) + (1 - mask) * h
```

State per sequence: `h` and `c`, each `L * n_modules * module_size` floats. The core output is the concatenated modules of the last layer,
so `hidden_size = n_modules * module_size`.

## Hyperparameters

- `n_modules`, `n_active` (6 / 3 as in the papers), `module_size` (tuned to the parameter budget with `knitwork.common.state_size`).
- `d_key`, `d_comm`, `n_comm_heads`: attention sizes; dropout inside attention (used in the official code) is omitted.
