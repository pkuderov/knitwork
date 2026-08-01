# Text Modeling

## Task formulation

The experiment evaluates recurrent models on character-level language modeling. The model receives characters one by one and must predict the next character; quality is measured in bits-per-character (BPC). The test checks how well the model captures long-term dependencies in natural language: syntactic constructions, recurring phrases, character names.

The default dataset is **Shakespeare** (~1.1M characters, 65 unique characters). The config can be redirected to **text8** by changing the path (`path: ~/data/text/text8`).

## Sequence generator

The `TextGenerator` class slices the training text into random segments. Each "environment" (`n_envs`) is an independent sliding window over the text. For diversity, random position resets are applied with probability `reset_prob`, which gradually decreases during training (curriculum: from 0.01 → 1e-4), giving the model increasingly longer coherent contexts.

| Parameter | Value |
|---|---|
| Dataset | shakespeare.txt (~1.1M characters) |
| Vocabulary size | 65 characters (shakespeare) / 27 characters (text8) |
| `n_envs` | 128 parallel reading streams |
| `rollout_len` | 8 steps per iteration |
| `reset_prob` initial | 0.01 |
| `reset_prob` final | 1e-4 (curriculum over ~1000 updates) |

## Metrics

| Metric | Description |
|---|---|
| **Acc** | Next character prediction accuracy (top-1) |
| **BPC** | Bits per character = Loss / log(2); main metric |
| **PPL** | Perplexity = 2^BPC; alternative scale |
| **Loss** | Cross-entropy loss over characters |

## What counts as a good result

On **shakespeare**:

| Level | BPC | Comment |
|---|---|---|
| Random | ~6.0 | uniform distribution over 65 characters |
| GRU baseline | ~2.0–2.2 | baseline recurrent memory level |
| Good result | **< 1.85** | model has captured language structure |
| Excellent result | **< 1.75** | stable long-term dependencies |

On **text8** (27 characters, more regular):

| Level | BPC |
|---|---|
| GRU baseline | ~1.45–1.55 |
| Good result | **< 1.35** |

Project observations:

| Model | Dataset | BPC |
|---|---|---|
| GRU baseline | shakespeare | ~2.00 |
| GridRNN 4/3 | shakespeare | **~1.72** |
| GridLRU 3/3 H=106 | shakespeare | ~1.83 |

## Running

```sh
uv run knitwork/exps/text/run_text.py knitwork/exps/text/config/extend_config.yaml \
    --model=grnn --name="grnn shakespeare"
```

## Logging

AIM project: `grid-rnn-text`. Main metrics: `Acc`, `BPC`, `PPL`, `Loss`.
