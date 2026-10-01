# Sparse LRU: mid configurations

Prepared on 2026-10-01. Implementation: `knitwork/models/grnn_lru_sparse.py`, registered as `grnn_lru_sparse`. The original `grnn_lru_core.py` remains the control implementation. Each variant has a standalone YAML in `knitwork/exps/text/config/lru_mid/`; no changes to the existing `mid.yaml` are needed.

All variants retain the first-column bottleneck: the prediction head receives only column 0 of the last layer. Communication has no auxiliary losses, stochastic gates, or value projections. The current mid baseline's parameter-free feedback RMS normalization (`fb_norm: true`) is shared by every variant; no variant-specific normalization is added. These are experimental configurations, not established improvements.

## Variants

All configurations use L2C4, H180, initial LRU horizons `[1, 1000]`. Parameter counts below include the token embedding and prediction head for the 27-character Text8 vocabulary.

| YAML | Change relative to control | Parameters | Required carried floats per sequence |
| --- | --- | ---: | ---: |
| `control.yaml` | Original dense static routing and dense LRU projections; identical initialization, outputs and gradients at the same seed | 1,049,463 | 3,600 |
| `self_heavy.yaml` | Initialize peripheral rows with 70% self-weight; hub initialization unchanged | 1,049,463 | 3,600 |
| `star.yaml` | Peripheral columns read themselves, the hub and the external input; hub reads every source | 1,049,463 | 3,600 |
| `star_ring.yaml` | Star plus one directed peripheral neighbor: 1 reads 2, 2 reads 3, 3 reads 1 | 1,049,463 | 3,600 |
| `clockwork.yaml` | Periods `[1,1,2,2]`, offsets `[0,0,0,1]`; hub updates every token | 1,049,463 | 4,320 |
| `block.yaml` | Two blocks in each LRU input/output projection, with channel shuffle between layers | 531,063 | 3,600 |
| `entmax.yaml` | Static entmax-1.5 peripheral routing; dense softmax hub; peripheral diagonal logits initialized to 0.1 | 1,049,463 | 3,600 |
| `dynamic_hub.yaml` | Star peripherals plus content-dependent softmax reading by the hub, query/key width 8 | 1,055,223 | 3,960 |

The five architectural changes are star connectivity, clockwork, block projections, entmax, and dynamic hub reading. Control, self-heavy initialization and star-with-neighbor are additional comparisons. Each variant changes only its listed mechanism, except the deliberately paired block/shuffle and dynamic-hub/star configurations.

`self_heavy` assigns 15% total weight to external inputs and 15% to other columns in layer 0; upper layers assign the remaining 30% to other columns. Star masks preserve the initial dense routing's self-weight and external-input weight, redistributing the removed peripheral communication mass across allowed peer connections. Thus masking does not silently increase the initial self/input weight.

Clockwork initializes every column on the first token, then alternates active sets `{0,1,2}` and `{0,1,3}`. On skipped steps, both the complex hidden state and residual output remain unchanged. Input/output cell projections are computed only for active columns. Batch-element resets clear that sequence's tensors while retaining the common clock phase; they do not restart a separate per-sequence clock. The horizon is measured in active updates, so period-2 columns initially remember roughly twice as many tokens. The finite phase cycle is compatible with `torch.compile`; initial and steady-state phases can require separate compiled graphs.

Block projections split each column's 180 channels into two independent groups of 90. Each input/output projection operates within its own group. The inter-layer permutation interleaves the groups, allowing the next layer to combine channels from both. This halves projection parameters without shrinking the recurrent state. This configuration is a compression comparison, not a parameter-matched quality comparison. Projection initialization uses the block width as fan-in.

Entmax can learn exact zero routing weights without an auxiliary sparsity loss. Its hub remains dense to keep information readable through column 0, and the softer peripheral initialization keeps external input readable initially. Routing is still query-independent for the peripheral columns.

Dynamic hub queries use the previous output of their own column 0 plus a fresh signal: the mean external embedding at layer 0 or the current lower-layer hub output above it. Keys project the available source messages, and values remain unprojected. Only the hub's routing depends on content; peripheral routing stays static and masked.

Routing masks and entmax currently use dense `bmm` for message aggregation. They enforce sparse connections/weights but do not implement sparse GPU kernels. Clockwork and block projections reduce cell projection work; whether they improve throughput must be measured after compilation and warm-up.

The state column reports the minimum information the implemented transition needs, using `knitwork.common.state_size`. With feedback normalization, all variants currently allocate 5,040 floating-point state elements per sequence after a step (`h`, all raw layer outputs, and normalized feedback). Static variants do not need the cached raw outputs; dynamic reading additionally needs each layer's raw hub output. Clockwork needs every cached raw output, from which normalized feedback can be recomputed. Clockwork also carries one shared integer phase. Without feedback normalization the allocated tensors contain 4,320 floats, with `out` an alias; dynamic reading then needs only 3,780 independent floats. This distinction matters when comparing actual memory versus required state size.

## Common protocol

The shared settings are copied from the current `mid.yaml`: float32, 512 streams, TBPTT 64, RMSprop with the existing learning-rate and reset schedules, up to 1B training tokens, contiguous validation/test splits of 5M tokens each, validation every 20M training tokens, and additional evaluation with state reset every 1,024 tokens. The runner selects the best weights using windowed validation when `eval.window` is enabled; final evaluation reports both best-validation and last weights on test. The existing full-continuity evaluation also remains enabled.

Default seed is 42. The text runner now seeds PyTorch as well as the NumPy generator when a seed is provided. `communication.loss_weight` and `communication.entropy_weight` are explicitly zero in these configurations. Keep the data split, learning-rate/reset schedules, batch size and training budget identical across variants.

The inherited dataset path contains `$MY_HOME`; supply an explicit path on the command line, since the runner does not expand environment variables in YAML paths. Comet is the inherited logger default; use `--log.logger=None` for local checks or runs without an external tracker.

The shared feedback normalization addresses the existing unnormalized residual-feedback instability. Keep it enabled consistently across this series. Inspect both full-continuity and windowed metrics; any unstable continuous metric should be recorded rather than treated as a successful result. No variant-specific stabilization has been added.

## Launch

From the repository root, run one variant:

```bash
uv run python -m knitwork.exps.text.run \
  knitwork/exps/text/config/lru_mid/star.yaml \
  --gens.text8.path=/absolute/path/to/text8.txt \
  --seed=42
```

For a preliminary, sequential series of 50M tokens each, with validation every 10M tokens:

```bash
for variant in control self_heavy star star_ring clockwork block entmax dynamic_hub; do
  uv run python -m knitwork.exps.text.run \
    "knitwork/exps/text/config/lru_mid/${variant}.yaml" \
    --gens.text8.path=/absolute/path/to/text8.txt \
    --seed=42 --n_steps=5e7 --eval.schedule=1e7
done
```

This loop launches actual training and may take substantial time; it was prepared but not run. The 50M-token budget is a preliminary comparison, not a final quality estimate. Repeat promising variants and control with the same additional seeds, for example 43 and 44, before drawing conclusions.

Check parameter/state accounting for any configuration:

```bash
uv run python -m knitwork.common.state_size knitwork/exps/text/config/lru_mid/dynamic_hub.yaml
```

Record validation BPC at the same token budgets, windowed validation/test BPC, full-continuity stability, training throughput after compile warm-up, and peak GPU memory. Distinguish block compression from parameter-matched results and clockwork's frozen state from routing sparsity. Routing visualizations can show weight sparsity; they do not establish semantic specialization.

## Local verification

```bash
uv run python -m unittest discover -s tests -p test_lru_sparse.py -v
```

Tests cover exact control compatibility, masks and initialization masses, skipped projections and frozen states, state reset/detach, block projections against an expanded block-diagonal dense reference, channel permutation, the entmax Jacobian, content-dependent hub reading, all eight mid configurations' forward/backward passes, and full-graph Dynamo tracing with the eager backend. Short runner checks use synthetic text and no external logger; they verify training, validation and test plumbing, not model quality.

Verified locally on 2026-10-01: all 14 tests passed; all eight H180 configurations completed a 128-token synthetic-text train/validation/test runner check on CPU with `fb_norm: true`; all eight variants also passed CPU Inductor full-graph forward/backward checks at H12, including the clockwork phases. Full Text8 training and GPU throughput/memory measurements have not been run.
