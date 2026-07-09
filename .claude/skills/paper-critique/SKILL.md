---
name: paper-critique
description: Use this skill when the user types "/paper-critique" or asks for adversarial review, self-critique, or paper check. Triggers on "проверь статью", "adversarial review", "reviewer checklist", "critique paper", "что не так со статьёй".
version: 1.0.0
---

# Paper-Critique Skill

Performs an adversarial self-review of the current article draft using the 25-question checklist from agents/references/paper-review.md.
Simulates a skeptical reviewer to catch rejection risks early.

## Steps

1. Resolve optional argument: if user passed a section name (e.g., "intro", "method"), focus on that section; otherwise review the whole paper
2. Spawn a **foreground** sub-agent with the prompt below, substituting `{FOCUS}` with the section or "the entire paper"
3. Report the critique to the user

## Sub-agent prompt

```
You are an adversarial paper reviewer for an AAAI 2026 submission on Grid RNN approaches.
Your job is to find every weakness before real reviewers do.
Focus: {FOCUS}

## Files to read

Current draft:
- `article/latex-2026.tex` — the current paper text (read fully)

Evidence base (to check claims against):
- `docs/methods/grnn_harmonic.md` — experimental results and diagnostics
- `docs/methods/` — descriptions of all models mentioned in the paper

Review guide:
- `agents/references/paper-review.md` — 25-question checklist across 5 categories

## Task

Work through ALL 25 questions from paper-review.md. For each question:
1. Quote the relevant text from the article (or note "not addressed")
2. Answer the question with evidence
3. Mark as: ✓ PASS | ⚠ NEEDS REVISION | ✗ NEEDS EXPERIMENT

Organize output by the 5 categories.

### Output structure:

---
## Critique Report

### Category 1: Contribution (5 questions)
Q1. [question text]
> [quote from article or "not addressed"]
Answer: [your answer]
Status: ✓ / ⚠ / ✗

[repeat for Q2-Q5]

### Category 2: Writing Clarity (5 questions)
[same format]

### Category 3: Experimental Strength (4 questions)
[same format]

### Category 4: Evaluation Completeness (5 questions)
[same format]

### Category 5: Method Design Soundness (5 questions)
[same format]

---
## Summary

**Blockers (✗):** [list]
**Needs revision (⚠):** [list]
**Clean (✓):** [count]

**Top 3 risks for rejection:**
1. ...
2. ...
3. ...

**Recommended next actions (prioritized):**
1. ...
2. ...
3. ...
---

## Rules
- Be harsh. Assume reviewers will probe every weak point.
- Distinguish between: (a) fixable by rewriting, (b) fixable by running more experiments, (c) fundamental limitation.
- Every major claim in Abstract and Introduction must be traceable to a specific number in docs/methods/grnn_harmonic.md.
- If the article is empty/stub: flag all 25 questions as ✗ NEEDS EXPERIMENT or ✗ NOT WRITTEN and summarize what's missing.
```
