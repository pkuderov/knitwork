# GridRnnPlastic (`grnn_redo`)

## Summary

Every regularizer in this codebase shapes the column structure through the loss. The 12C
study showed that this is the wrong lever for the problem: `col_cka/frac_gt_06` reaches its
final level by roughly **0.6M steps — 1% of training — and is then flat for the remaining
60M**. Whatever commitment the network makes in that first percent, it keeps. A loss
weight applied afterwards is arguing with a decision that has already been made.

`GridRnnPlastic` adds the one mechanism that can undo such a commitment instead of
penalizing it: ReDo-style periodic re-initialization of columns whose contribution has
collapsed.

The columns that die are not random. Causal delta correlates negatively with column index
in **all five** 12C arms (Spearman ρ from −0.22 to −0.84), and the fast half of the grid
holds 76% of the causal mass on average. The multi-timescale prior staggers the update-gate
bias linearly from fast (C0) to slow (C11), and slow columns are useless on the short
episodes early in the curriculum — so they never earn gradient, and by the time long gaps
appear they are already dead. Revival gives them a second draw against a network that is
already trained, where more niches exist than at step 0.

## Key mechanism

```python
dead = (share < self.redo_threshold).nonzero().flatten().tolist()
dead = sorted(dead, key=lambda c: float(share[c]))[:self.redo_max_cols]
for c in dead:
    for layer in range(self.n_layers):
        cell = self.cells[layer][c]
        for name, p in cell.named_parameters():
            nn.init.xavier_uniform_(p) if p.dim() > 1 else nn.init.zeros_(p)
        a = self.attn[layer]
        a.ids_q[c].normal_(0.0, 0.1 * alpha)
        a.ids_k[c].normal_(0.0, 0.1 * alpha)
if optimizer is not None:
    for p in touched:
        optimizer.state.pop(p, None)
```

Three details decide whether this works at all:

**Reset the attention identity, not just the cell.** `ids_q[c]` and `ids_k[c]` determine
whether any other column ever reads this one. A revived cell that nobody reads receives no
useful gradient and dies again within a few thousand steps.

**Reset the optimizer state.** Adam moments accumulated over tens of millions of steps
encode the old, dead direction and would drag the fresh weights straight back. The ReDo
paper resets optimizer state for revived units for exactly this reason.

**Never re-initialize mid-forward.** `_readout_aux` only takes a detached snapshot of the
per-column share; the mutation happens in `apply_redo()`, which `run_sdq.py` calls *after*
`optimizer.step()`. Rewriting weights between forward and backward would have backward
differentiate activations that no longer match the parameters that produced them.

## Hyperparameters

| Name | Default | Notes |
|---|---|---|
| `redo_every` | `0` (off; arm: `5e6`) | Env-steps between revival checks. |
| `redo_threshold` | `0.02` | Readout share below which a column counts as dead. Uniform share at C=12 is 0.083. |
| `redo_max_cols` | `1` | Columns revived per event. More than one at a time destabilizes the readout. |

## Interaction with the aux clock

This model was what exposed a defect in the aux scheduling: with `optim: true` the model's
internal step counter never advanced, because `torch.utils.checkpoint(..., use_reentrant=False)`
runs *both* passes grad-enabled, while the counter's guard assumed the reentrant semantics
where the first pass is under `no_grad`. `run_sdq.py` now drives the clock via
`set_env_step()`. See `docs/reviews/column_collapse_12c.md` §2.3.
