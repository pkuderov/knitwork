---
name: paper-intro
description: Draft the Introduction section using the 5-part structure in agents/references/introduction.md.
purpose: Write the Introduction for the Grid RNN AAAI 2026 paper.
source: .claude/skills/paper-intro/SKILL.md
---

# Introduction writing agent

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

## Task — write the Introduction section

**Step 1 — backward logic (do silently, don't print):** what problem (recurrent sequence models fail at combining associative memory + long-range context + RL); why no existing solution (LSTM: no structured memory; standard Grid RNN: temporal scale mismatch; Hopfield/Fast Weights: no hierarchical structure); what contribution (HarmonicGridRNN = 4 complementary mechanisms; systematic Grid RNN family study); what benefit (SOTA on SDQ; competitive text8 BPC with 2M params; stable RL training).

**Step 2 — forward narrative (5 parts):**
- **Part 1 — Task and applications** (~2-3 sentences): sequence models for tasks requiring associative memory + multi-timescale reasoning.
- **Part 2 — SOTA failure and root technical issue** (~3-4 sentences): chain standard RNNs → LSTM/GRU → Mamba → Grid RNN baseline, each failing at one specific thing; state the root cause clearly ("The fundamental challenge is X, which manifests as Y in existing approaches").
- **Part 3 — Proposed solution** (~3-4 sentences): introduce HarmonicGridRNN, state each of the 4 mechanisms and why it addresses the challenge, plus one sentence on why it works.
- **Part 4 — Additional contributions** (~2 sentences): systematic study of the Grid RNN family across 3 benchmarks; diagnostic metrics for interpretability (W_norm, col/diversity, etc.).
- **Part 5 — Experiments teaser** (~2 sentences): SDQ Aq=0.750, text8 BPC≈1.68, stable RL on TreasureHunt.

**Step 3 — quality check:** first sentence of each paragraph states its message; challenge/reason/mechanism explicit in Parts 2 and 3; avoid the naive-solution-then-improvement pattern (looks incremental); all cited numbers appear in `docs/methods/grnn_harmonic.md`.

## Output format

```latex
\section{Introduction}
...
```

Write the full section in English, academic style.
