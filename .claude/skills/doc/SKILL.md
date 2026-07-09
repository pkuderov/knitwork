---
name: doc
description: Use this skill when the user types "/doc <file_path>" or asks to document a model file, create a methods doc, write documentation for a model, or generate a methods/*.md file. Triggers on any request like "задокументируй", "создай doc для", "напиши описание метода", "doc knitwork/models/...".
version: 1.0.0
---

# Doc Skill

Reads a model source file and generates a corresponding `docs/methods/<name>.md` documentation file in Russian.

## Steps

1. Read the file passed as argument (e.g. `knitwork/models/grnn_lru.py`)
2. Identify the core idea: what problem the method solves, the key architectural decision
3. Extract 2-4 code snippets that best characterize the method (key equations, unique forward pass logic, unusual init patterns)
4. Write and create `docs/methods/<basename>.md` following the template below

## Output template

```markdown
# <ModelClassName>

<Один абзац: какую проблему решает метод и в чём его суть.>

## Ключевой механизм

<Одно предложение о главной идее, затем фрагмент кода:>

```python
# краткий комментарий на английском о том, что делает этот фрагмент  [shapes if relevant]
<code snippet>
```

<Одно-два предложения объясняющих фрагмент.>

## Важные детали реализации

<Ещё 1-3 фрагмента кода, каждый с коротким описанием после него. Включай только нетривиальные части.>

```python
# comment  [B, T, H]
<snippet>
```

<Пояснение.>

## Гиперпараметры

| Параметр | Описание |
|---|---|
| `param` | что делает, почему нетривиален |
```

## Rules

- Text (descriptions, explanations) is in **Russian**; code comments stay in **English**
- Code snippets: prefer short self-contained excerpts (5-20 lines); cut irrelevant boilerplate
- Omit trivial init assignments, imports, and logging — show only the logic that defines the method
- Hyperparameters table: only non-obvious ones (skip `hidden_size`, `dropout` unless they have an unusual role)
- Do not add a trailing summary paragraph; the doc ends after the last section that has content
- If `docs/methods/` directory doesn't exist yet, create it first
- After writing the file, print its path and a one-line confirmation
