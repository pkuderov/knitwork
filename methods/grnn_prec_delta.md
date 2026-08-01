# GridRnnPrecDelta

Grid RNN with parallel preconditioned Delta Rule. Standard Fast Weights (`grnn_fw`) accumulate interference: the additive update `A += v⊗k` does not erase the old association for the same key. Delta rule fixes this — before writing, the current prediction of `A` for key `k` is subtracted. Preconditioning eliminates the second drawback of a fixed learning rate: Adam-style second moment of keys `v2` normalizes the update step per dimension, equalizing the learning speed across different key variances. For backpropagation stability, all errors are computed from the same frozen matrix `A.detach()` (parallel variant, without chained Jacobians through sequential A reads).

## Key Mechanism

Compute errors for all columns from the original `A`, collect the total `delta_A`, apply a single update with decay:

```python
# parallel delta rule: errors from frozen A, single joint update  [B, H, H]
delta_A = torch.zeros_like(A)
for col_j in range(n_cols):
    k_j    = F.normalize(k[col_j], dim=-1)
    v_j    = F.normalize(v[col_j], dim=-1)

    # second moment (running stat — detached from graph)
    v2_new = beta2 * v2_new + (1 - beta2) * k_j.detach() ** 2   # (B, H)

    # prediction error from frozen A — avoids chained Jacobians
    error  = torch.bmm(A.detach(), k_j.unsqueeze(2)).squeeze(2) - v_j  # (B, H)

    # preconditioned key: scale by 1/sqrt(v2)
    k_prec = k_j / (v2_new.sqrt() + eps)                               # (B, H)

    delta_A += torch.bmm(error.unsqueeze(2), k_prec.unsqueeze(1))

# decay existing A, then apply preconditioned delta writes
A = delta_decay * A - delta_lr * delta_A
```

Using `A.detach()` is intentional: A is a recurrent state, not a parameter; chaining gradients through sequential reads and writes of A creates exploding Jacobians when `delta_lr≥0.1`.

## Important Implementation Details

**Retrieval** — dot-product read from updated A:

```python
# content-based retrieval after all writes  [cols, B, H]
msgs = [torch.bmm(A, F.normalize(q[i], dim=-1).unsqueeze(2)).squeeze(2)
        for i in range(n_cols)]
h_msg = self.norm(torch.stack(msgs, dim=0))
```

Gradient flows through A only at the retrieval stage, not the write stage. This separation is a standard technique in linear attention.

**Initialization of v2 to ones:**

```python
# init to ones — neutral preconditioning at episode start (avoids k/eps blow-up)
def _init_v2(self, bsz):
    return torch.ones(self.n_layers, bsz, self.hidden_size, device=dev, dtype=dt)
```

With `v2=0`, the initial values `k_prec = k/(0 + eps)` explode. With `v2=1`, initial preconditioning is neutral: `k_prec ≈ k`.

**Model state** — triple `(h, A, v2)`:

```python
# state = (h, A, v2)
# h:  (n_layers, n_cols, B, H)
# A:  (n_layers, B, H, H)
# v2: (n_layers, B, H)   ← second moment, reset together with A
```

## Hyperparameters

| Parameter | Description |
|---|---|
| `delta_lr` | Write rate into `A`; safe range 0.005–0.05; at `delta_lr≥0.1` without `delta_decay<1`, the matrix grows and NaN gradients appear |
| `delta_decay` | L2-decay of `A` before each update; 0.99 ensures bounded norm; 1.0 — pure delta rule without forgetting (unstable for long sequences) |
| `beta2` | Second moment decay; 0.999 — slow curvature accumulation (like Adam), 0.9 — faster adaptation to key distribution shifts |
| `eps` | Denominator stabilizer in `k_prec`; with standard `v2≈1` has virtually no effect; important only when `v2→0` (does not occur with ones initialization) |
