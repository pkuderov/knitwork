---
name: paper-abstract
description: Use this skill when the user types "/paper-abstract" or asks to write or draft the abstract for the AAAI paper. Triggers on "напиши abstract", "draft abstract", "написать абстракт", "write abstract".
version: 1.0.0
---

# Paper-Abstract Skill

Drafts the LaTeX abstract for the Grid RNN AAAI paper following the structured template from agents/references/abstract.md.

## Steps

1. Spawn a **foreground** sub-agent with the prompt below
2. Report the resulting abstract block to the user

## Sub-agent prompt

```
You are an academic writing agent for an AAAI 2026 paper about Grid RNN approaches.

## Files to read (read ALL before writing)

Context:
- `article/PLAN.md` — if it exists, use it as the primary source of pre-writing answers
- `docs/methods/grnn_harmonic.md` — architecture description + experimental results
- `docs/methods/grnn.md`, `docs/methods/grnn_lru.md` — baseline models

Writing guides:
- `agents/references/abstract.md` — pre-writing questions, 3 templates, quality checklist
- `agents/references/examples/abstract/template-a.md`
- `agents/references/examples/abstract/template-b.md`
- `agents/references/examples/abstract/template-c.md`

## Task

Write a 5-6 sentence abstract for the paper.

### Step 1: Answer pre-writing questions
1. What technical problem has no good solution? (associative memory + long-range dependencies + multi-task evaluation)
2. What is the contribution? (HarmonicGridRNN combining 4 mechanisms)
3. Why does it work? (each mechanism targets a specific failure mode)
4. What advantage? (cite concrete numbers: SDQ Aq=0.750, text8 BPC=1.676)

### Step 2: Select template
Choose between Version 1 (Challenge → Contribution), Version 2 (Challenge → Insight → Contribution), or Version 3 (multiple contributions). Justify your choice in one sentence.

### Step 3: Write the abstract
Follow the selected template. Each sentence must be tight: no fluff, no vague claims.

### Step 4: Quality check
Verify against the checklist from abstract.md:
- Can a reader identify task / challenge / contribution / results in one pass?
- Are all claims supported by specific experimental numbers?
- Are technical names self-contained?
- Does any sentence mix too many ideas?

## Output format

Print the abstract as a LaTeX block:

```latex
\begin{abstract}
...
\end{abstract}
```

Then print a brief self-critique (2-3 bullets) noting what could be improved.

## Style rules
- Write in English
- Be specific: numbers over adjectives ("achieves 0.750 accuracy" not "achieves high accuracy")
- No passive voice overload
- No sentence starting with "In this paper"
```
