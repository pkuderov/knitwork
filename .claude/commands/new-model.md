---
description: Implement a new GridRNN variant from a method description — creates model file, registers in all run scripts, adds configs, creates docs, and smoke-tests on all 3 experiments.
argument-hint: "<model_name>: <detailed method description>"
allowed-tools: [Read, Write, Edit, Bash]
---

Implement a new GridRNN model and wire it into the full project.

## Input format

Arguments: `<model_name>: <detailed method description>`

- `model_name`: snake_case, e.g. `grnn_delta` — used for file names, registry keys, config keys
- description: mechanism, state structure, key hyperparameters, how it differs from existing models

## Key paths (hardcoded — do not search)

**Model file:** `knitwork/models/<model_name>.py`

**Run scripts (registry `_REGISTRY`):**
- `knitwork/exps/sdq/run_sdq.py`
- `knitwork/exps/text/run_text.py`
- `knitwork/exps/treasure/run_treasure_hunt.py`

**Config files (`models:` section):**
- `knitwork/exps/sdq/config/extend_config.yaml`
- `knitwork/exps/text/config/extend_config.yaml`
- `knitwork/exps/treasure/config_treasure_hunt.yaml`

**Docs:** `docs/methods/<model_name>.md`

**Docs sidebar:** `docs/_sidebar.md`

## Step 1 — Read base reference

Read `knitwork/models/grnn.py` for the base GridRNN pattern and `knitwork/models/grnn_fw.py` for the fast-weight memory pattern (if the new model uses a fast-weight/memory mechanism).

## Step 2 — Implement the model

Create `knitwork/models/<model_name>.py`:

- Class name: derive from model_name in PascalCase, e.g. `grnn_prec_delta` → `GridRnnPrecDelta`
- Follow `grnn_fw.py` structure exactly: same `__init__` keyword-only signature, same `forward` / `_grid_step` / `reset_state` / `detach_state` / `_init_*` / `_cell_input_dim` / `_prepare_grid_input` pattern
- Keep `n_attn_heads` in `__init__` but mark it unused (config compat)
- State: tuple, e.g. `(h, A)` or `(h, A, extra)` — all tensors, detachable
- Use `dtype=torch.float64` for GRUCell (matches existing models)
- Print model name/config at init; print param count via `format_readable_num`
- No multi-line docstrings; one-line max

## Step 3 — Register in all 3 run scripts

In each run script, find the `_REGISTRY` dict and add **before** the closing `}` or before the `# config aliases` block:

```python
'<model_name>': ('knitwork.models.<model_name>', '<ClassName>'),
```

For `run_sdq.py` and `run_text.py` add before `'grnn_fusion': None,  # factory`.
For `run_treasure_hunt.py` add before the closing `}`.

## Step 4 — Add config entries

In each of the 3 config files, add a `<model_name>:` block under `models:`.

Base template (adjust hyperparams to match the model's `__init__`):

```yaml
  <model_name>:
    embedding_size: 64
    hidden_size: 128
    n_layers: 2
    n_columns: 3
    n_attn_heads: 4
    messaging: post
    col_identities: true
    # model-specific hyperparams here
```

- SDQ config: insert before `grnn_lru:` entry (or at end of `models:` section if not present)
- Text config: insert before the `lr:` key
- Treasure config: insert before the `# LR schedule` comment

## Step 5 — Create docs

Create `docs/methods/<model_name>.md` in Russian following the template:

```markdown
# <ClassName>

<One paragraph: problem solved and core idea.>

## Ключевой механизм

<One sentence, then code snippet (5–15 lines).>

```python
# short English comment  [shapes]
<snippet>
```

<1–2 sentences explaining the snippet.>

## Важные детали реализации

<1–3 more snippets, each with explanation.>

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `param` | non-obvious description |
```

Then add a line to `docs/_sidebar.md` under the appropriate category.

## Step 6 — Smoke test on all 3 experiments

Run each experiment for ~200 steps on CPU with logging disabled:

```bash
# SDQ
uv run knitwork/exps/sdq/run_sdq.py knitwork/exps/sdq/config/extend_config.yaml \
  --model=<model_name> --device=cpu --n_steps=200 --n_envs=4 --log.enabled=false

# Text
uv run knitwork/exps/text/run_text.py knitwork/exps/text/config/extend_config.yaml \
  --model=<model_name> --device=cpu --n_steps=200 --n_envs=4 --log.enabled=false

# TreasureHunt
uv run knitwork/exps/treasure/run_treasure_hunt.py knitwork/exps/treasure/config_treasure_hunt.yaml \
  --model=<model_name> --device=cpu --n_steps=200 --n_envs=4 --log.enabled=false
```

If a test fails: read the traceback, fix the model or config, rerun until all 3 pass.

## Rules

- Write code comments in **English**; docs in **Russian**
- No emojis anywhere
- Do not modify `knitwork/models/grnn.py`, `grnn_err.py`, `gru.py`, or `knitwork/config/base.yaml`
- Do not commit anything
- Report final status: which tests passed, param count, any issues
