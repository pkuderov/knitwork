# Store-Distract-Query (SDQ)

## Task formulation

Store-Distract-Query is a synthetic benchmark for isolated evaluation of associative memory in recurrent models. The task tests whether the model can store an arbitrary key → value mapping across an arbitrary time gap filled with noise tokens.

Each episode is a sequence of tokens of one of three types:

| Token type | Description | Training signal |
|---|---|---|
| **Store** `(k, v)` | "Remember: key k has value v" | none (write only) |
| **Distract** `d` | Random noise token (value from vocabulary V) | none |
| **Query** `k` | "What is the value of key k?" | answer `v` — target token |

Episode length is geometrically distributed with mean T. At each step a token type is chosen randomly: Store with probability p_store, Query with probability p_query, otherwise Distract.

## Generator parameters

| Parameter | Easy | Hard |
|---|---|---|
| `n_keys` | 5 | 5 |
| `n_vals` | 10 | 10 |
| `T` (mean length) | 10 steps | 10 steps |
| `p_store` | 0.35 | 0.35 |
| `p_query` | 0.35 | 0.35 |
| `count_stored` | false | **true** |
| `count_queried` | false | **true** |

Easy vs Hard difference: in `hard` mode the target signal encodes how many times the key has already been written and queried (`count_stored`, `count_queried`). This makes the task harder: the model must not only associate a key with a value, but also count accesses to it.

## Token vocabulary

Input vocabulary size: `n_keys × n_vals + n_vals + n_keys` = `5×10 + 10 + 5 = 65` tokens.

- Indices `[0, n_keys×n_vals)` — Store tokens `(k,v)`, encoded as `k × n_vals + v`
- Indices `[n_keys×n_vals, …+n_vals)` — Distract tokens
- Indices `[…, …+n_keys)` — Query tokens

Output vocabulary: `n_vals = 10` tokens (possible values).

## Metrics

| Metric | Description |
|---|---|
| **Acc** | Accuracy on Query tokens (fraction of correct predictions) |
| **Acc++** | Accuracy only on "hard" queries: key was overwritten or queried more than once |
| **Loss** | Cross-entropy only on Query steps (ignore\_index on Store and Distract) |
| **sq\_gap** | Mean Store–Query time gap for answered queries |

## What counts as a good result

- **Acc > 0.85** on hard — strong result; means reliable storage and updating of associations through distractors
- **Acc++ > 0.65** — especially significant; baseline GRU on Acc++ ~ 0.1–0.2 (near random)
- **Acc < 0.5** on easy — poor result; model does not solve the basic associative memory task

Baseline references from project observations:

| Model | Acc | Acc++ |
|---|---|---|
| GRU baseline | ~0.50 | ~0.15 |
| GridRNN | ~0.65 | ~0.30 |
| GridLRU (wide) | **~0.85** | **~0.69** |

## Running

```sh
uv run knitwork/exps/sdq/run_sdq.py knitwork/exps/sdq/config/extend_config.yaml \
    --model=grnn_lru --gen=hard
```

## Logging

AIM project: `grid-rnn-sdq`. Main metrics: `Acc`, `Acc++`, `Loss`, `sq_gap`.
