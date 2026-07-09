---
name: paper-intro
description: Use this skill when the user types "/paper-intro" or asks to write the Introduction section of the AAAI paper. Triggers on "напиши introduction", "draft intro", "написать введение", "write introduction".
version: 1.0.0
---

# Paper-Intro Skill

Drafts the Introduction section using the 5-part structure from agents/references/introduction.md.

## Steps

1. Spawn a **foreground** sub-agent with the prompt below
2. Report the section to the user

## Sub-agent prompt

```
You are an academic writing agent for an AAAI 2026 paper about Grid RNN approaches.

## Files to read (read ALL before writing)

Context:
- `article/PLAN.md` — if it exists, use backward-logic answers and novelty statement from there
- `docs/methods/grnn_harmonic.md` — architecture, experimental results
- `docs/methods/` — read grnn.md, grnn_lru.md and any other documented models

Writing guides:
- `agents/references/introduction.md` — backward logic, 5-part structure, Part A/B/C versions, quality checklist
- `agents/references/examples/introduction/technical-challenge-version-1-existing-task.md`
- `agents/references/examples/introduction/pipeline-version-1-one-contribution-multi-advantages.md`
- `agents/references/examples/introduction/pipeline-version-2-two-contributions.md`

## Task

Write the Introduction section.

### Step 1: Backward logic (do this silently, don't print)
Answer these questions before writing:
- What problem? (recurrent sequence models fail at combining associative memory + long-range context + RL)
- Why no existing solution? (LSTM: no structured memory; standard Grid RNN: temporal scale mismatch; Hopfield/Fast Weights: no hierarchical structure)
- What contribution? (HarmonicGridRNN = 4 complementary mechanisms; systematic Grid RNN family study)
- What benefit? (SOTA on SDQ; competitive text8 BPC with 2M params; stable RL training)

### Step 2: Forward narrative (5 parts)

**Part 1 — Task and applications** (~2-3 sentences)
Use Version 1 (niche task) or Version 3 (general to specific setting).
Introduce: sequence models for tasks requiring associative memory + multi-timescale reasoning.

**Part 2 — SOTA failure and root technical issue** (~3-4 sentences)
Use Technical Challenge Version 1 (existing task — challenge chain).
Chain: standard RNNs → LSTM/GRU → Mamba → Grid RNN baseline → each fails at one specific thing.
State the root cause clearly: "The fundamental challenge is X, which manifests as Y in existing approaches."

**Part 3 — Proposed solution** (~3-4 sentences)
Use Pipeline Version 1 or 2.
Introduce HarmonicGridRNN. State each of 4 mechanisms and why it addresses the challenge.
Include one sentence on why it works.

**Part 4 — Additional contributions** (~2 sentences)
- Systematic study of the Grid RNN family across 3 benchmarks
- Diagnostic metrics for interpretability (W_norm, col/diversity, etc.)

**Part 5 — Experiments teaser** (~2 sentences)
SDQ Aq=0.750, text8 BPC≈1.68, stable RL on TreasureHunt.

### Step 3: Quality check (verify each):
- First sentence of each paragraph states the paragraph's message
- Challenge/reason/mechanism are explicit for Part 2 and 3
- No naive solution then improvement pattern (avoids looking incremental)
- All cited numbers are in docs/methods/grnn_harmonic.md

## Output format

```latex
\section{Introduction}
...
```

Write the full section in English, academic style.
```
