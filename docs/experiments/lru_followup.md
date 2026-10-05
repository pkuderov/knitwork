# LRU follow-up configurations

The 2026-10-02 follow-up implements the experiments proposed after reviewing `knitwork-aistat`. Every Grid-LRU variant predicts through column 0. The experiment configurations use the task cross-entropy; they do not add communication or specialization losses. These are prepared experiments, with correctness and pipeline checks, rather than trained benchmark results.

## New mechanisms and Text8 configurations

Implementation: [grnn_lru_sparse.py](../../knitwork/models/grnn_lru_sparse.py). The eight new standalone configurations are in [text/config/lru_mid](../../knitwork/exps/text/config/lru_mid). The existing control, star, block and clockwork configurations remain available in the same directory.

| Configuration | Mechanism | Text8 H | Total parameters, V=27 |
| --- | --- | ---: | ---: |
| `dynamic_hub_top.yaml` | Dynamic reading only in the top layer; `hub_layers: [1]` uses zero-based indices | 180 | 1,052,343 |
| `dynamic_hub_zero.yaml` | Query starts at zero, key stays random; initial routing matches the static star | 180 | 1,055,223 |
| `dynamic_hub_self_heavy.yaml` | Dynamic hub plus self-heavy initialization of the star's peripheral rows | 180 | 1,055,223 |
| `dense_small.yaml` | Dense control with a weight budget close to block H180 | 128 | 533,311 |
| `block_wide.yaml` | Grouped projections with a weight budget close to dense H180 | 254 | 1,050,099 |
| `block_dense_hub.yaml` | Dense input/output projections in col0; two projection groups in each peripheral column | 180 | 660,663 |
| `clockwork_light.yaml` | Periods `[1,1,1,2]`: only one peripheral column updates every other token | 180 | 1,049,463 |
| `clockwork_output.yaml` | Every column records every token; peripheral output projections follow `[1,1,2,2]` | 180 | 1,049,463 |

The dense H180 control has 1,049,463 parameters; the existing block H180 has 531,063. Compare block H180 against both dense H180 and dense H128, and block H254 against dense H180. Changing H also changes the state and central bottleneck width; a similar number of weights is one comparison axis, rather than an equivalence of memory or computation. The mixed dense-hub bank stores only the peripheral blocks and the hub's dense matrices, rather than masking full matrices.

`hub_layers` defaults to all layers when `hub_query_size > 0`, preserving the original dynamic-hub variant. Top-layer routing adds 2,880 weights instead of 5,760. `hub_query_init: zero` initializes the query projection to zero and isolates construction of query/key from the global CPU RNG: common recurrent parameters, embedding and head are initialized identically to the same-seed static star. The query receives a gradient on the first update; the key becomes trainable through the nonzero query afterward. Default dynamic initialization preserves the earlier behavior.

`clockwork_mode: state`, the default, freezes both hidden state and cached output for inactive columns. `clockwork_mode: output` advances all complex states and input projections, then computes output projections only for scheduled columns; the others retain cached messages. The hub always updates, the first step initializes every output, and per-example resets clear both hidden state and cached messages while preserving the batch clock phase. In the steady cycle, light clockwork performs 87.5% of full column updates. Output-only clockwork performs all input projections and 75% of output projections, approximately 87.5% of the dense projection arithmetic. These operation fractions are not measured speedups.

The new Text8 configurations retain the existing 1B-token, 512-stream, TBPTT64, RMSprop and reset/LR schedules. They select checkpoints on window-1024 validation and use `knitwork-aistat` as the default Comet project. Existing control configurations still have their original logging project; override it when grouping a new series.

```bash
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/lru_mid/dynamic_hub_top.yaml
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/lru_mid/dynamic_hub_self_heavy.yaml
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/lru_mid/block_dense_hub.yaml
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/lru_mid/clockwork_output.yaml

# Existing dense control in the same logging project.
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/lru_mid/control.yaml --log.project=knitwork-aistat
```

Set `--gens.text8.path=/absolute/path/to/text8.txt` if the configured `$MY_HOME` data path is unsuitable, and use `--seed=1337` / `--seed=31337` for repeats. These commands are full experiments; they were not executed as part of implementation.

## Standalone MQAR suite

[mqar/config/lru_mid](../../knitwork/exps/mqar/config/lru_mid) contains 20 standalone configurations. They share exactly the data, training and checkpoint protocol in [mid.yaml](../../knitwork/exps/mqar/config/mid.yaml): vocabulary 8192, independent train/val/test data, 32 epochs, batch 32, Adam at 0.001, query-only loss, full-sequence BPTT and validation selection before test. Logging is local by default. The shared `mid.yaml` and tiny `smoke.yaml` also expose all model sections for `--model` overrides.

| Configuration | Role | H | Full MQAR parameters |
| --- | --- | ---: | ---: |
| `grnn_lru.yaml` | Original dense Grid-LRU L2C4 | 180 | 3,997,028 |
| `control.yaml` | Equivalent dense core through the sparse-model class | 180 | 3,997,028 |
| `grnn.yaml` | Grid-GRU L2C4 with attention, nearest full parameter budget among widths divisible by 4 | 156 | 3,935,344 |
| `grnn_core_mid.yaml` | Original Text8 mid Grid-GRU core, for a core-budget comparison | 136 | 3,279,544 |
| `gru.yaml` | Monolithic GRU L2, matched full parameter budget | 211 | 4,002,000 |
| `star.yaml`, `star_ring.yaml`, `self_heavy.yaml`, `entmax.yaml` | Original routing ablations | 180 | 3,997,028 |
| `dynamic_hub.yaml`, `dynamic_hub_zero.yaml`, `dynamic_hub_self_heavy.yaml` | Dynamic reading in both layers | 180 | 4,002,788 |
| `dynamic_hub_top.yaml` | Dynamic reading only at the top | 180 | 3,999,908 |
| `clockwork.yaml`, `clockwork_light.yaml`, `clockwork_output.yaml` | Temporal computation ablations | 180 | 3,997,028 |
| `block.yaml` | Existing two-group block model | 180 | 3,478,628 |
| `dense_small.yaml` | Full-budget control for block H180 | 161 | 3,478,100 |
| `block_wide.yaml` | Full-budget block comparison with dense H180 | 204 | 4,019,684 |
| `block_dense_hub.yaml` | Dense hub with grouped peripheral projections | 180 | 3,608,228 |

MQAR widths for `dense_small` and `block_wide` differ from Text8 because the vocabulary embedding and head contribute many more weights. Block H204 is the closest even width to the dense H180 total, within 0.57%; dense H161 is within 0.02% of block H180. GRNN H156 is within 1.55% of the original Grid-LRU total; GRNN H136 retains the earlier core width and has a smaller complete model. Report both core and full counts, state size and measured resources.

GRNN uses the existing `grnn` registry core, `bank: 2`, `mha: 2`, four heads, message normalization and `noise_std: 0.05`. Its communication statistics are not included in MQAR's objective. The MQAR runner already adapts tokens to the core's time-first layout, while Grid-LRU receives batch-first tokens. The MQAR dynamic-hub configuration now explicitly uses `topology: star`, matching its Text8 counterpart and the static-star control; earlier `mid.yaml` omitted this setting.

```bash
# Controls.
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_mid/grnn.yaml
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_mid/grnn_lru.yaml
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_mid/gru.yaml

# Test variants; replace the filename to select any row above.
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_mid/dynamic_hub_top.yaml
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_mid/clockwork_output.yaml

# Tiny CPU checks with the same mechanisms, small hidden size and vocabulary.
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/smoke.yaml --model=grnn.L2C4
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/smoke.yaml --model=grnn_lru_sparse.block_dense_hub

# Vary model/order randomness while keeping data and evaluation seeds fixed.
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_mid/dynamic_hub_top.yaml --seed=1337
```

To use Comet when launching, add `--log.logger=comet` and, if desired, `--log.project=knitwork-aistat`. Authentication uses the repository's normal environment configuration. Full MQAR runs were not launched during implementation.

## Validation

Forty local tests pass. They check exact preservation of the original dense control, zero-query equivalence including embedding/head initialization, gradients from late MQAR queries to stored-value embeddings, dense expansion of mixed block projections, continuous clockwork writes with skipped output projections, reset behavior, parameter/state budgets, common protocols, and fullgraph compilation of every follow-up path using the eager backend. Each of the 20 MQAR configurations also completes a two-update tiny CPU pipeline with checkpoint selection and held-out metrics. Two CLI checks additionally use the actual GRNN H156 and clockwork-output H180 cores with vocabulary 8192 and the tiny smoke data. These checks validate implementation and do not measure benchmark quality or GPU throughput.

State accounting distinguishes initial and steady-state allocations and counts views once. For normalized Grid-LRU H180, initial allocations are 4,320 floats per sequence and steady-state allocations are 5,040; the dense carried information is 3,600. For Grid-GRU H156, the cached top output is a view of the hidden state, so both actual allocation and carried information are 1,248 floats. Inspect `config.json` / `metrics.json` for each run's recorded counts.
