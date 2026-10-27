# Five LRU memory and access experiments

Implementation: [grnn_lru_experimental.py](../../knitwork/models/grnn_lru_experimental.py), registered as `grnn_lru_experimental`. Every variant predicts exclusively from column 0 and trains with the task cross-entropy. There are no auxiliary losses, oracle query/write labels, or growing token-history caches. These configurations prepare experiments; local correctness checks do not establish a quality improvement.

## Common base and configurations

The primary comparison holds L2C4, hidden width H180, horizons `[1, 1000]`, feedback RMS normalization, star connectivity and dynamic hub query/key width 8 fixed. The hub query starts at zero, as in the existing `dynamic_hub_zero` experiment. Each variant adds one mechanism. Construction of extra modules preserves the CPU RNG, so common retained LRU weights, embedding and output head retain their same-seed initialization. The `dynamic_control.yaml` in each directory supplies the shared baseline.

There are five variant files plus `dynamic_control.yaml` for each task:

- [Text8 mid configurations](../../knitwork/exps/text/config/lru_architecture_mid): existing 1B-token protocol, 512 streams, TBPTT64, RMSprop, unchanged learning-rate/reset schedules, no communication loss. Logging defaults to the `knitwork-aistat` Comet project.
- [MQAR mid configurations](../../knitwork/exps/mqar/config/lru_architecture_mid): unchanged vocabulary 8192, fixed independent train/validation/test data, Adam 0.001, batch 32, 32 epochs, full-sequence BPTT and query-only loss. Local logging and checkpoint output are the defaults. [MQAR protocol](mqar.md) describes data and evaluation.
- [Aliased Cubes mid configurations](../../knitwork/exps/aliased_cubes/config/lru_architecture_mid): unchanged observation/action adapter, fixed map and separate random walks, 25,000 iterations, sequence length 400 and TBPTT64. This uses the project's provisional cube surface, **not an exact reproduction of the ICML-2024 benchmark**; see [environment provenance](aliased_cubes.md).

| File / mechanism | Core weights | Text8 weights, V27 | MQAR weights, V8192 | Cube adapter weights | Carried floats / sequence |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dynamic_control.yaml` | 1,045,476 | 1,055,223 | 4,002,788 | 1,050,528 | 3,960 |
| `write_gate.yaml` | 1,047,642 | 1,057,389 | 4,004,954 | 1,052,694 | 3,960 |
| `query_read.yaml` | 1,103,076 | 1,112,823 | 4,060,388 | 1,108,128 | 3,960 |
| `associative.yaml` | 820,574 | 830,321 | 3,777,886 | 825,626 | 4,264 |
| `rotations.yaml` | 1,045,500 | 1,055,247 | 4,002,812 | 1,050,552 | 3,960 |
| `two_reads.yaml` | 1,045,476 | 1,055,223 | 4,002,788 | 1,050,528 | 3,960 |

Widths are fixed; these are not all matched in parameter budget. In particular, the associative variant has fewer weights and more persistent state. Query reading adds 57,600 weights, writing gates add 2,166, rotations add 24, and two reads share the existing weights but perform extra computation. Report core/full weights, state, measured time and peak memory alongside task quality. To measure Text8 budgets locally:

```bash
uv run python -m knitwork.common.state_size knitwork/exps/text/config/lru_architecture_mid/associative.yaml
```

## Exact mechanisms

### 1. `write_gate`: hub-controlled peripheral updates

Each layer generates C−1 scalar gates from the concatenation of the previous hub output and fresh input (the lower-layer hub in upper layers). Each peripheral state becomes `lerp(previous_h, ordinary_lru_update, sigmoid(gate))`; the hub always uses gate 1. Both real and imaginary components share the column gate. Initial gate weights are small and bias 2 gives approximately 0.881. Setting a gate to zero preserves the hidden memory exactly, including its phase and decay; gating only the input write would not do this.

Cached messages are recomputed every token, even when hidden memory is preserved, because the cell output includes the residual input. All projections still execute: this is selective memory updating, without a claimed sparse-execution speedup. Capture includes `write_gate`, and the Text8 runner logs layer means. Useful controls are `dynamic_control` and `--grnn_lru_experimental_write_gate.write_gate_bias=4.0` for an initially more open gate.

### 2. `query_read`: query-conditioned narrow reads from private states

For each peripheral column, packed complex memory `[2H]` is projected to `read_rank: 16` channels. A hub-context projection supplies the multiplicative gate `2 * sigmoid(query)`, shared across the peripheral reads, and a column-specific projection expands the gated channels to H. These reconstructed values replace peripheral public messages **only when the hub reads them**. Peripheral routing continues to receive the original public messages; token and hub sources remain unchanged. The hub's existing dynamic routing still selects column weights.

This allows a query to change the contents of a read, as well as which column contributes. Reads use the previous state of the corresponding layer. Parameter cost is `L * (3 * (C−1) * H * rank + H * rank)`; a narrow read does not reduce the private state. Test `--grnn_lru_experimental_query_read.read_rank=8` or `=32` separately from the rank-16 configuration.

### 3. `associative`: replace a peripheral LRU with delta-rule memory

Column 3 in each layer becomes a fixed-size matrix `M` of shape `[16, 32]`. Its input generates a unit-normalized key `k`, bounded value `v = tanh(value_projection)` and scalar write rate `beta = sigmoid(rate_projection)`. The update is `M_next = M + beta * outer(k, v − k @ M)`. Write-rate bias −2 initially gives approximately 0.119. A normalized query from the hub context retrieves `query @ M`, followed by a projection to H. The hub reads the old matrix before that token's write; the column emits a read of the updated matrix for downstream layers and next-step feedback. Everything is learned from the task inputs and ordinary task loss.

This replaces a column rather than adding memory beside an unused LRU. The bank stores only the retained three LRU columns. Hidden state shape is `[L, C−1, B, 2H]`, with physical column mapping in `lru_columns`; matrix state is `[L, B, 16, 32]`. Matrices start at zero, reset per example and detach at the experiment's BPTT boundaries. Other peripheral columns remain ordinary LRUs. Matrix capacity is fixed as sequence length grows, and no exact arbitrary-dictionary capacity is claimed.

Steady allocated state is 5,344 floats per sequence, versus 5,040 for the other configurations. Initial allocations are 4,624 versus 4,320. Carried information counts exclude redundant output storage. Text8's CKA and column-similarity diagnostics use all four public column messages for this heterogeneous variant, rather than comparing an absent fourth LRU state; those curves are not directly equivalent to the hidden-state diagnostics of homogeneous LRUs. Capture also reports `memory_write_rate`. Controls can vary `memory_key_size`, `memory_value_size`, `memory_column` or `memory_write_bias` without changing the data protocol.

### 4. `rotations`: sparse direct exchange of recurrent states

Before each layer's LRU update, column 0 and each peripheral column undergo a Givens rotation: `(hub, peripheral) -> (cos(a)*hub + sin(a)*peripheral, −sin(a)*hub + cos(a)*peripheral)`. Rotations apply sequentially in ascending peripheral-column order. There are four angle groups over hidden channels; real and imaginary coordinates share angles. Groups partition H using integer channel indices, including when H is not divisible by four.

Angles start at zero, making the initial forward pass equal to the shared control. The rotations preserve the state norm by construction and exchange memory without a dense cross-column projection. The later LRU decay, input writes and nonlinear feedback remain present; norm preservation of this exchange alone does not establish stability of the whole network. Change `rotation_groups` to vary expressiveness. The primary configuration adds only 24 trainable angles.

### 5. `two_reads`: refine the hub with fresh peripheral outputs

The first pass routes messages and computes ordinary candidate states/outputs for every column. A second hub-only read uses the provisional hub as query and the freshly computed column outputs as sources, retaining the external token source in layer 0. The same hub LRU weights then recompute its final candidate **from the original previous hub state**, rather than decaying memory a second time. Peripheral candidates from the first pass become final unchanged.

Peripherals therefore write once per token and the hub performs two candidate computations with shared parameters. Higher layers receive the final lower-layer output. Capture contains both `attn_weights` and `second_attn_weights`; the existing Text8 attention visualization displays the first pass. Cell projection arithmetic increases by 1/C, or 25% at C4, with additional hub routing and temporary activations; this is not a measured wall-clock overhead. The variant introduces no adaptive halting or computation penalty.

## Launching

Run commands from the repository root. `uv run python` can be replaced by `.venv/bin/python` in an existing environment. Full configurations choose the normal available device; set `--device=cuda:0` or `--device=cpu` explicitly when needed. The following commands launch full training and were not run during implementation.

```bash
# Text8: common control and one variant.
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/lru_architecture_mid/dynamic_control.yaml
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/lru_architecture_mid/write_gate.yaml --gens.text8.path=/absolute/path/to/text8.txt

# MQAR: no downloaded dataset is required.
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_architecture_mid/dynamic_control.yaml
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_architecture_mid/query_read.yaml

# Provisional Aliased Cubes, same environment/data across variants.
uv run python -m knitwork.exps.aliased_cubes.run knitwork/exps/aliased_cubes/config/lru_architecture_mid/rotations.yaml
```

Replace the filename with `write_gate`, `query_read`, `associative`, `rotations` or `two_reads` to choose the mechanism. To launch all five MQAR experiments sequentially:

```bash
for variant in write_gate query_read associative rotations two_reads; do
  uv run python -m knitwork.exps.mqar.run "knitwork/exps/mqar/config/lru_architecture_mid/${variant}.yaml"
done
```

Use `--seed=1337` and `--seed=31337` for model/order repeats, keeping the split seeds fixed. MQAR and cube runners save a unique directory under their configured `output_dir`, including resolved `config.json`, `best.pt`, `last.pt` and `metrics.json`. MQAR records core/full parameter counts and carried/allocated state sizes. Checkpoints are selected by validation loss; test metrics are computed after selection. To enable Comet on these locally logged tasks, add `--log.logger=comet --log.project=knitwork-aistat`. Text8 already uses that project.

Existing dense Grid-LRU and Grid-GRU MQAR controls remain in [the earlier suite](../../knitwork/exps/mqar/config/lru_mid); use `grnn_lru.yaml` and `grnn.yaml` as additional architectural baselines. The new dynamic control isolates each change relative to the common routing base.

## Short local checks

[smoke.yaml](../../knitwork/exps/mqar/config/lru_architecture_mid/smoke.yaml) contains all five mechanisms plus the dynamic control. It uses CPU, H12, vocabulary 64, tiny independent data splits and two optimizer updates. Its larger associative dimensions are intentionally retained to exercise the actual mechanism; this tiny configuration is not a matched-budget benchmark.

```bash
for variant in write_gate query_read associative rotations two_reads; do
  uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_architecture_mid/smoke.yaml \
    "--model=grnn_lru_experimental.${variant}"
done

# Actual mid width and full MQAR vocabulary, while retaining tiny smoke data.
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_architecture_mid/smoke.yaml \
  --model=grnn_lru_experimental.associative \
  --grnn_lru_experimental_associative.hidden_size=180 --data.vocab_size=8192

uv run python -m unittest discover -s tests -p test_lru_architecture.py -v
```

The new tests check gate preservation/control limits, query-dependent private reads, targeted delta-rule overwrites, gradients through stored matrices and late MQAR queries, norm-preserving rotations, zero-angle control equivalence, a single peripheral write in two-read execution, original-state hub refinement, RNG matching, partial resets, detach, exact budgets and common task protocols. All five mechanisms also run through the Text8 training/inspection path on synthetic text, and two-update MQAR/cube pipelines with held-out evaluation and saved checkpoints. Fullgraph compilation is checked against eager results; local compilation and smoke tests do not measure GPU throughput or benchmark quality.

Validation on 2026-10-02: all 53 repository tests passed, including 13 new architecture tests. All five cores additionally passed CPU Inductor fullgraph forward/backward checks at H12. All five completed actual CLI runs at H180 and vocabulary 8192 using the tiny MQAR smoke data, saving checkpoints and finite held-out metrics under `/tmp/knitwork-lru-architecture-mid-checks`. Full-budget training and GPU performance measurements were not launched.
