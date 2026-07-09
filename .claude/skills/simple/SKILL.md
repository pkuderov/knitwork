---
name: simple
description: Use this skill when the user types "/simple <file_path>" or asks to simplify, refactor, or clean up a model file. Triggers on requests like "упрости", "simplify", "отрефактори", "почисти код" for files in knitwork/models/ or knitwork/exps/. Spawns a dedicated sub-agent that reads the target file alongside base architecture files and rewrites it to be shorter, cleaner, and consistent with the codebase style.
version: 1.0.0
---

# Simple Skill

Spawns a sub-agent to simplify a target file by leveraging existing base classes, shared utilities, and project style rules.

## Steps

1. Resolve the file path from the argument (support both absolute and relative paths from project root)
2. Spawn a **foreground** sub-agent with the prompt below, substituting `{FILE_PATH}` with the resolved absolute path
3. Report to the user what the agent changed

## Sub-agent prompt template

```
You are a code simplification agent working on the knitwork research project.

## Your task
Simplify the file `{FILE_PATH}` so it is shorter, cleaner, and fully consistent with the project style. Do NOT change the model's behavior or mathematical logic — only the code structure.

## Base files to read first (read all of them before touching the target)

Always read these before editing. Use them to eliminate duplication.

**Architecture base (protected — never modify these):**
- `knitwork/models/grnn.py` — `GridRnn` base class, `MessagePassingLayer`; inherit from it instead of `nn.Module` where possible; import `MessagePassingLayer` from here instead of copy-pasting

**Shared model modules (use these to replace duplicated code):**
- `knitwork/models/lru.py` — `LRUCell`, `LRUBlock`; import from here, never redefine
- `knitwork/models/engram.py` — `EngramMemory`, `EngramState`; import from here
- `knitwork/models/diversity.py` — `ColumnDiversityLoss`, `DiversityLossConfig`, `ColumnDiversityAnalyzer`, `ColumnSpecializationLoss`
- `knitwork/models/fusion_cells.py` — `HGRUCell`, `BatchedHGRUColumns`, `BatchedReservoirColumns`

**Shared utilities (`knitwork/common/utils.py`):**
- `format_readable_num(x)` — human-readable param count string
- `to_torch(x, device)` — numpy→tensor conversion
- `isnone(x, default)` — None-coalescing
- `safe_div(num, denom, default)` — division without NaN
- `get_device(device)`, `get_dtype(dtype)`

## Simplification checklist (apply every rule that fits)

1. **Inherit, don't copy** — if the class re-implements `reset_state`, `detach_state`, `init_state`, `_prepare_grid_input`, `_cell_input_dim`, or the constructor boilerplate already present in `GridRnn`, delete the copy and call `super()` instead. Only keep overrides that actually differ.

2. **Use shared utilities** — replace inline logic that duplicates `isnone`, `safe_div`, `to_torch`, `convert_hidden_size`, `format_readable_num` with the imported helper.

3. **No unnecessary abstractions** — remove wrapper methods, trivial one-liners, or helper functions whose body is a single expression. Inline them.

4. **No dead code** — remove unused imports, unreachable branches, commented-out blocks, and variables assigned but never read.

5. **Style rules from CLAUDE.md**:
   - Keyword-only arguments (`*,`) in constructors with many params
   - No multi-line docstrings; one short line max if needed
   - Comments only where non-obvious; always in English; annotate tensor shapes `# [B, T, H]`
   - No emojis anywhere
   - Short and direct — no wrapper layers

modificate big comments to small and delete --- lines in comments

6. **Keep**: all public API methods (`forward`, `reset_state`, `detach_state`, `init_state`), their signatures, and the model's mathematical logic exactly as-is.

## Output
Edit the file in-place using the Edit tool. Do not create a new file.
After editing, print a short bullet list of what you removed or simplified and why.
```

## Notes

- Always spawn the agent in **foreground** (default) so the result is available before reporting back
- If the file is in the protected list (`grnn.py`, `grnn_err.py`, `gru.py`), refuse and tell the user these files must not be modified
- If the argument is missing, ask the user for the file path
