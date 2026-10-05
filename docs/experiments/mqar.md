# Multi-query associative recall (MQAR)

MQAR tests whether the recurrent state can retain several key/value associations and retrieve them after distraction. The current implementation uses the existing token embedding, recurrent core and vocabulary head. The LRU prediction still comes from column 0; there are no auxiliary objectives or additional memory modules.

## Task protocol and provenance

The local [generator](../../knitwork/gens/mqar.py) follows single-pass `multiquery_ar` from [Zoology at commit `1ad20d1`](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/data/multiquery_ar.py). Keys are sampled without replacement from tokens `1 .. V//2-1`, values independently from `V//2 .. V-1`. A prefix contains alternating keys and values. Each key is then queried once at a distinct even suffix slot, sampled with weights proportional to `(slot+1) ** (power_a-1)`. With `random_non_queries: true`, filler is uniform over the full vocabulary, including zero. A filler coinciding with a stored key is not a designated scored query.

The target is the stored value **at the same position as the query key**. All other labels are `-100`; the model never receives the target value as an inserted query answer. With zero filler, a sample is:

```text
input:   k1 v1 k2 v2  0 0 k2  0 0 0 k1  0
label:    -  -  -  -  - - v2  - - - v1  -
```

Local NumPy and Torch generators isolate data generation from model initialization and training randomness. Golden-array tests compare inputs and labels against the unmodified upstream generator, with both upstream RNGs explicitly seeded. Split seeds and the seed of each segment are written to the run configuration. JRT-style repeated passes are not part of this implementation.

## Prepared configurations

[mid.yaml](../../knitwork/exps/mqar/config/mid.yaml) uses the vocabulary, training segment sizes and test grid from [Zoology's Based Figure 2 configuration](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/experiments/paper_configs/arxiv24_based_figure2/configs.py). The training set is fixed and shuffled once per epoch.

| Split | Sequence length / stored pairs | Examples per segment |
| --- | --- | --- |
| Train | 64/4, 128/8, 256/16, 256/32, 256/64 | 100,000 for 64/4; 20,000 for each other segment |
| Validation | Same length/pair grid as train, independent samples | 1,000 |
| Test | 64/4, 64/8, 64/16, 128/32, 256/64, 512/128, 1024/256 | 1,000 |

The vocabulary is 8,192, `power_a=0.01`, random filler is enabled. Validation is an additional local split used to select checkpoints; the test is evaluated once after selection. Larger test lengths and capacities include extrapolation beyond the training grid.

| Model | Configuration | Core parameters | Total parameters | Required carried state / sequence |
| --- | --- | --- | --- | --- |
| Column LRU | L2C4, H180, feedback normalization | 1,039,716 | 3,997,028 | 3,600 floats |
| GRU | L2, H211 | 536,784 | 4,002,000 | 422 floats |

The LRU core is the existing mid architecture. The larger vocabulary makes the complete model approximately 4M parameters, rather than Text8's approximately 1M. The GRU total is within 0.13% of the LRU total; this does not match core parameters or state capacity. The current normalized LRU implementation allocates 5,040 state floats including intermediate outputs. Actual resource measurements are recorded separately.

The same configuration also provides the sparse LRU variants, follow-up variants and GRNN controls. The [standalone mid suite](../../knitwork/exps/mqar/config/lru_mid) contains 20 configurations; their mechanisms, parameter budgets and run examples are documented in [LRU follow-up configurations](lru_followup.md). GRNN H156 approximately matches the complete Grid-LRU parameter budget; GRNN H136 retains the original Text8 mid core. MQAR block-budget comparisons use dense H161 and block H204 because the large embedding/head changes the appropriate widths. Each run records actual counts. `dynamic_hub` explicitly uses the star topology, matching the Text8 variant.

## Training and evaluation

The [runner](../../knitwork/exps/mqar/run.py) uses Adam, gradient clipping and **full-sequence BPTT**. State is freshly initialized per batch of independent sequences and is never detached between the store prefix and later queries. Loss is summed over scored queries and divided by their count before one optimizer update. The vocabulary head runs only on scored rows/steps; recurrent state still advances through every input token.

Batches contain one segment and do not pad shorter examples. The last partial batch is retained, and batches from different segments are shuffled together each epoch. This batching, the learning rate, batch size 32, local validation split and checkpoint rule differ from Zoology's complete training harness: this is a matching task generator/grid with a Knitwork training protocol, not a reproduction of published scores. The default budget is 32 epochs; `training.max_updates` can cap it for a pilot.

Report query accuracy, query cross-entropy and sequence exact match, both globally and for every `T..._K...` segment. Global CE/accuracy are weighted by the number of scored queries, so larger pair counts contribute more; exact match is weighted by sequences. Select the lowest validation query CE, then run the held-out test with that checkpoint. Evaluation fixes the random state initialization and restores the training RNG afterward; keep evaluation batch size fixed across runs.

Each run saves `config.json`, `best.pt`, `last.pt` and `metrics.json` in a unique timestamp/model/seed directory. Checkpoints contain model weights, not optimizer state for resuming training. Metadata includes the pinned generator source, split seeds, parameter/state counts, device/GPU, Torch/CUDA versions, training tokens, scored queries and optimizer updates. `initial_allocated_state_floats` counts unique initial-state allocations; `allocated_state_floats` counts steady-state allocations after a core step, without double-counting views. This inspection preserves the training RNG. `training_seconds` and tokens/second measure the forward/backward/update sections with CUDA synchronization, excluding data generation, batch transfer, evaluation and logging. `peak_cuda_allocated_bytes` includes training and evaluation and measures Torch allocation, not total GPU/process memory; it is null on CPU. Logging is local by default.

## Run examples

From the repository root:

```bash
# Tiny model, small vocabulary, two updates: pipeline check only.
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/smoke.yaml

# Prepared mid LRU and matched-total-parameter GRU protocols.
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/mid.yaml
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/mid.yaml --model=rnn.L2

# Sparse routing experiment with the same MQAR data and training protocol.
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/mid.yaml --model=grnn_lru_sparse.star

# Keep data and evaluation seeds fixed while varying initialization/order.
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/mid.yaml --seed=43
```

`smoke.yaml` is not a benchmark result. A short check with the actual mid core and 8,192-token adapter can use that tiny dataset with `--grnn_lru_L2C4.hidden_size=180 --data.vocab_size=8192`. For matched comparisons, keep the data grid, training exposure, optimizer, evaluation batch size and checkpoint rule identical, and report the resource/state differences rather than assuming equal total parameters imply equal memory or compute.
