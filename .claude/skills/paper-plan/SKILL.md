---
name: paper-plan
description: Use this skill when the user types "/paper-plan" or asks to plan the paper, answer pre-writing questions, create a writing plan for the Grid RNN article. Triggers on "спланируй статью", "pre-writing план", "план статьи", "writing plan".
version: 1.0.0
---

# Paper-Plan Skill

Answers all pre-writing questions across sections and writes `article/PLAN.md` — a shared context file used by all other paper-* skills.

## Steps

1. Spawn a **foreground** sub-agent with the prompt below
2. After the agent finishes, confirm that `article/PLAN.md` was created

## Sub-agent prompt

```
You are a research paper planning agent for the knitwork project.

## Your task
Answer all pre-writing questions for an AAAI 2026 paper about Grid RNN approaches and write the result to `article/PLAN.md`.

## Files to read (read ALL before writing)

Architecture and results:
- `docs/methods/grnn_harmonic.md` — HarmonicGridRNN: all 4 blocks, experimental results (SDQ, text8, Shakespeare, MIKASA), problems and diagnostics
- `docs/methods/grnn.md` — base Grid RNN
- `docs/methods/grnn_lru.md` — Grid RNN with LRU cells
- Read any other `docs/methods/grnn*.md` that exist

Writing guides (read the question/pre-writing sections):
- `agents/references/abstract.md` — pre-writing questions block
- `agents/references/introduction.md` — backward logic and 5-part structure
- `agents/references/method.md` — pre-writing questions block
- `agents/references/paper-review.md` — 25-question checklist (to anticipate reviewer objections)

## Output: article/PLAN.md

Write a comprehensive planning document with these sections:

### 1. Paper Scope
One paragraph: what is the paper about, what is the central claim, what type of contribution is this (novel architecture / comparative study / new benchmark / combination).

### 2. Pre-Writing: Abstract
Answer these 4 questions with specific evidence from the experiments:
- What technical problem has no good solution?
- What is the contribution?
- Why does it work (mechanism)?
- What advantage / new insight does it provide?
Then pick Abstract template (Version 1/2/3) and justify.

### 3. Pre-Writing: Introduction
Answer backward-logic questions:
- What problem are we solving?
- Why do existing RNNs (LSTM, GRU, Mamba, standard Grid RNN) fail at this problem?
- What do we contribute?
- What benefit does our work provide to future research?
Then map to 5-part Introduction structure (Task / Challenge / Solution / Contributions / Experiments teaser).

### 4. Model Comparison Table
A markdown table: Model | Core mechanism | Strengths | Weaknesses | Best benchmark result. Include: grnn, grnn_lru, hgrnn, hgrnn_lru, grnn_harmonic (and any others documented in docs/methods/).

### 5. Pre-Writing: Method Section
For each of the 4 HarmonicGridRNN blocks, answer:
- What is this module?
- What is its workflow (inputs → steps → output)?
- Why is it necessary (what breaks without it)?
- Why does it work?

### 6. Claim → Evidence Mapping
Table: Claim | Experiment | Metric | Value | Status (confirmed/needs more runs). Map every major claim to specific experimental numbers.

### 7. Anticipated Reviewer Objections
List top 5 objections a skeptical reviewer would raise, and draft rebuttals using experimental evidence.

## Rules
- Write in English (the base article is English)
- Be specific: cite actual numbers from experiments (e.g., "SDQ Aq=0.750 at 95M steps")
- Do not invent results; only use what is documented in docs/methods/
- If information is missing, mark it explicitly as [MISSING — needs experiment]
```
