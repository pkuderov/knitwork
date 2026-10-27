# Memory of the matrix-state baselines

DeltaNet, mLSTM and HGRN2 carry a matrix state per layer (`H x H`, `H x H*expand`). With truncated BPTT over a 64-step rollout and 512 parallel
streams, autograd keeps the state of every step, so activation memory is `copies x state x layers x steps`. On the mid configs (about 1M
parameters) the measured peaks were 56 GB (DeltaNet), 56 GB (mLSTM) and 39 GB (HGRN2), which forced these runs to queue one after another.

## What was changed

1. **Lazy reset (`delta_net`, `mlstm`).** `reset_state` used to return `state * keep`, a full copy saved for backward. It now stores the keep mask
   in `state['keep']`; the layer applies it through per-environment scalars (gates), see the method docs. Mathematically the same reset.
2. **Scaling before the outer product.** The write `b * (dv k^T)` is computed as `((b dv) k^T)`, so the `[B, H, H]` product is not saved.
3. **Activation checkpointing over time (`grad_checkpoint=k`, text runner).** The rollout is processed in segments of `k` steps under
   `torch.utils.checkpoint`; states are kept only at segment borders and the segment is recomputed in backward. Off by default; `k` must divide
   `rollout_len`; plain recurrent cores only. Reset masks are drawn outside the segment, so recomputation is deterministic.

## Measurements (n_envs=16, 64-step rollout, scaled x32 to 512 streams; gradients compared to the original implementation)

| Model | Variant | Peak memory | Time | Relative gradient difference |
| --- | --- | ---: | ---: | ---: |
| DeltaNet / mLSTM | original | 1.00x (52.8 GiB) | 1.0x | - |
| | lazy reset | 0.35x (18.6 GiB) | 0.83-0.95x | 2e-5 |
| | checkpoint 8 | 0.16x | 1.3-1.5x | 0 |
| | lazy reset + checkpoint 8 | 0.08x (4.5 GiB) | 1.2-1.4x | 2e-5 |
| HGRN2 | original | 1.00x (37 GiB) | 1.0x | - |
| | lazy reset | 0.97x | 0.86x | 0 |
| | checkpoint 8 | 0.26x | 1.27x | 0 |

Checkpoint segments of 16 steps were worse than 8 in both memory and time. The queue runs the heavy baselines with `grad_checkpoint=8`
(about 5 GB for DeltaNet and mLSTM, 10 GB for HGRN2), so they no longer wait for a nearly empty card.

## Caveats

- The peak is activation memory only; model and optimizer states are small. Measurements were done in eager mode on a laptop GPU; the original
  numbers match the server peaks (56 GB measured, 52.8 GiB predicted), so the scaling by the number of streams holds.
- Results of runs with and without these changes are numerically equivalent but not bit-identical.
- Tests: `tests/test_baseline_memory.py` (lazy reset equals explicit multiplication; checkpointed gradients equal plain ones).
