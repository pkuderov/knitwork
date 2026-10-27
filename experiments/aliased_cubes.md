# Aliased Cubes: provisional LRU experiment

Prepared on 2026-10-02. Source task: [Dedieu et al., ICML 2024](https://proceedings.mlr.press/v235/dedieu24a.html), *Learning Cognitive Maps from Transformer Representations for Efficient Planning in Partially Observed Environments*, Section 4.2, Figure 3 and Appendix I.1, Table 7. The paper describes aliased cubes of edge size 6 with 12 observation symbols and a next-observation prediction task using observation/action histories. Its navigation results additionally use discrete latent maps and an external planner.

An official public environment implementation or transition table was not located in the paper, the author's publication page, or the searched repositories as of 2026-10-02. The exact cube transition semantics are not specified sufficiently to claim an exact reproduction. The new local environment is explicitly **paper-inspired**; its metrics must not be presented as reproducing Table 7. This experiment currently evaluates next-observation prediction, not planning or cognitive-map recovery.

## Environment

`knitwork/env/aliased_cube.py` implements a Gymnasium environment with categorical observations, four actions, zero rewards and deterministic transitions. The local design is six square faces of an edge-6 cube: 216 hidden cells, each with a face-local east/west/north/south transition. Crossing a seam folds the coordinates onto the adjacent face and changes the local coordinate frame. There are no boundary self-loops. This surface topology and its action conventions are explicit local choices, not verified author settings.

Twelve symbols are balanced across the 216 cells and shuffled using `environment.map_seed`; each symbol occurs 18 times. The learner receives the symbol, never the cell index, face, coordinates, map, or seed. Gym `info` is empty. Random-walk sampling returns only `observations[N,T]` and `actions[N,T-1]`. Map seed and train/validation/test walk seeds are separate, and a SHA-256 fingerprint of the transition/emission tables is recorded with results.

An external graph can replace the local topology with `--environment.graph_path=/path/to/graph.npz`. The NPZ must contain integer arrays `transitions[n_states,n_actions]` and `observations[n_states]`, with zero-based IDs. The runner derives the action and observation alphabets from these arrays. Importing a table does not automatically establish it as an official benchmark asset; retain its provenance alongside the recorded fingerprint.

## Minimal model adaptation

`ActionObservationModel` computes `embedding(o_t) + embedding(a_t)`, passes that vector to the existing recurrent core, and predicts `o_{t+1}`. This preserves the core architecture and its first-column output bottleneck. There is one cross-entropy objective, with no auxiliary losses, latent-state supervision, vector quantization, planning objective, or RL algorithm.

The default is the current baseline LRU mid configuration: L2C4/H180, horizons `[1,1000]`, and shared feedback normalization. It has 1,044,768 parameters with this observation/action adapter. `--model=rnn.L2` selects a two-layer GRU/H294 control with 1,049,004 parameters, about 0.4% larger. Both use exactly the same maps, walk splits, adapter, optimization and evaluation settings. A training-only smoothed table `P(o_next | o_current, action)` provides a reactive comparison without memory.

The first experiment tests whether the column architecture disambiguates repeated observations from history at an acceptable parameter/state/compute cost. Prediction accuracy alone does not establish a useful cognitive map or a planner: the source paper distinguishes those outcomes. A later experiment could probe the first-column representation or add a common post-hoc map construction protocol for all architectures, but that is outside the implemented objective.

## Protocol

The default YAML creates 2,048 train, 256 validation and 2,048 test walks of 400 observations each on one fixed cube. Train/validation/test walk seeds are 100/101/102. Optimization uses Adam, learning rate 0.001, batch size 32, gradient clipping 1 and 25,000 optimizer updates. These defaults are a provisional research protocol, not a claim that every author detail has been matched.

Each walk contains 399 predictions. TBPTT defaults to 64 transitions, but gradients from all chunks are accumulated before one optimizer update for the entire sampled batch; state is carried across chunks and reset at each new walk. The default full budget is 319.2M training transitions. `--training.rollout_len=399` enables full-walk BPTT. No dropout is added, which differs from the paper's regularization setting.

Validation runs every 250 updates and at the last update. The checkpoint with the lowest full validation CE is selected; test is evaluated once afterward. Evaluation is deterministic at a fixed batch size and preserves the training RNG stream. Reports include CE, overall accuracy, accuracy after 30 context transitions, and accuracy by transition index. The initial uncertain context remains in the training loss and the overall metric. Reactive validation/test metrics are also reported.

Each run saves `config.json`, `best.pt`, `last.pt` and `metrics.json` in a unique directory under `output_dir`, retaining earlier runs. Checkpoints are saved for evaluation, not optimizer-state resume. Logger defaults to local output; Comet can be selected explicitly.

## Launch

From the repository root:

```bash
uv run python -m knitwork.exps.aliased_cubes.run knitwork/exps/aliased_cubes/config/mid.yaml

uv run python -m knitwork.exps.aliased_cubes.run knitwork/exps/aliased_cubes/config/mid.yaml \
  --model=rnn.L2 --environment.map_seed=0 --seed=42
```

A short CPU plumbing check, with no external logger or saved run artifacts:

```bash
uv run python -m knitwork.exps.aliased_cubes.run knitwork/exps/aliased_cubes/config/mid.yaml \
  --device=cpu --output_dir=None \
  --training.iterations=2 --training.batch_size=2 --training.rollout_len=4 \
  --data.sequence_length=9 --data.train_sequences=4 \
  --data.val_sequences=2 --data.test_sequences=2 \
  --eval.schedule=1 --eval.batch_size=2 --eval.report_after=3
```

For a preliminary comparison, use an equal smaller iteration budget for LRU and GRU, then repeat over the same map seeds and model seeds. Separate variation over maps from variation over optimization seeds. A possible later suite uses map seeds 0–9 and model seeds 42/43/44; no such suite has been launched.

## Verification

```bash
uv run python -m unittest discover -s tests -p test_aliased_cubes.py -v
```

Checks cover connected cube topology, reversible seam adjacency, Gymnasium API, seed reproducibility, separation of latent states from learner inputs, external graph import and validation, the reactive model, LRU/GRU forward/backward, LRU Dynamo full-graph tracing, deterministic evaluation, and the complete mid training/validation/test/checkpoint path. Only short local correctness checks have been run; no benchmark quality results are available.

Local verification on 2026-10-02 passed all 8 new tests and all 22 tests in the combined suite. The documented CLI check also completed at the full mid model width with 2 optimizer updates and 32 training transitions, including validation and test.
