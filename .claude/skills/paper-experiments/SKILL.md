---
name: paper-experiments
description: Use this skill when the user types "/paper-experiments" or asks to write the Experiments section of the AAAI paper. Triggers on "напиши experiments", "draft experiments", "написать эксперименты", "write experiments section".
version: 1.0.0
---

# Paper-Experiments Skill

Drafts the Experiments section using all result tables from docs/methods/grnn_harmonic.md.
Follows the 3-question structure (better than baselines? / which modules matter? / how far does it generalize?).

## Steps

1. Spawn a **foreground** sub-agent with the prompt below
2. Report the section to the user

## Sub-agent prompt

```
You are an academic writing agent for an AAAI 2026 paper about Grid RNN approaches.

## Files to read (read ALL before writing)

Results:
- `docs/methods/grnn_harmonic.md` — ALL result tables: SDQ v2 (#043), SDQ v3.1 (#055), text8 v2/v3, Shakespeare v2, MIKASA (#058), diagnostic metrics
- `docs/methods/` — check for result tables in grnn.md, grnn_lru.md, other grnn*.md files

Writing guide:
- `agents/references/experiments.md` — 3-question structure, table formatting rules, figure/caption rules

## Task

Write the Experiments section.

### Structure:

**4.1 Experimental Setup**
Three subsections (short paragraphs):

*Benchmarks:*
- SDQ (Store-Distract-Query): synthetic associative memory; metrics: Acc/query (Aq), Acc/distract; 1B steps budget
- Text8 / Shakespeare: character-level language modeling; metric: BPC (bits per character); 520M steps
- TreasureHunt (MIKASA RepeatFirstEasy): RL with PPO; metric: episode return EpRet; 200M steps

*Baselines:*
- grnn: base Grid RNN (message passing, no LRU/memory)
- grnn_lru: Grid RNN with LRU cells
- hgrnn_lru: Hierarchical Grid RNN + LRU
- GRU/LSTM: standard baselines at ~2M params
- Note any baselines with known results from docs/methods/

*Implementation:*
- All models: ~2.1M parameters, CUDA, Adam optimizer
- HarmonicGridRNN config: 3 layers × 4 columns, dk=32, dv=128, n_attn_heads=4
- For text: n_reservoir_cols=4; for SDQ/RL: n_reservoir_cols=0

**4.2 Main Results**

Table 1: SDQ benchmark
- Columns: Model | Params | Acc/query ↑ | Acc/distract ↑ | Steps
- Use booktabs style: \toprule, \midrule, \bottomrule
- No vertical lines
- Highlight best result
- Include grnn_harmonic v2 peak (Aq=0.750 at 95M) and v3.1 (Aq=0.715 at 110M)
- If baseline results are available in docs/methods/, include them; otherwise mark [TBD]

Table 2: Language Modeling
- Columns: Model | Params | text8 BPC ↓ | Shakespeare BPC ↓
- Include: grnn_harmonic (text8 BPC=1.676 at 270M, Shakespeare BPC=0.788 at 340M)
- LSTM ~2M reference: ~1.6 BPC on text8

Short paragraph summarizing main takeaways from both tables.

**4.3 Ablation Studies**

*v2 vs v3.1 on SDQ* — table with Aq at 10M/30M/50M/75M/95M/110M steps for both versions.
Discuss: v3.1 converges slower (−0.035 Aq at 75M), but multi_col_head design for LM/SDQ justified by column collapse in v3.1.

*Memory diagnostics* — a table or figure description:
| Metric | Layer 0 | Layer 1 | Layer 2 | Interpretation |
For SDQ v2 at 300M: W_norm (1.06/1.46/1.88), alpha (0.85/0.86/0.87), surprise, fullness, error.
Discuss monotonic W_norm growth (slower layers accumulate more), fullness=1-3% (memory not saturated).

**4.4 Analysis: Column Specialization**

Column diversity and norm diagnostics for SDQ v3.1:
- col/diversity: L0=2.34, L1=2.80, L2=7.79 — severe specialization collapse at top layer
- col0_norm: L2 rises to 325 vs others at 15-25 — single column dominates

Compare with text8 (v3): diversity L0=1.92/L1=1.80/L2=2.16, col norms stable.
Hypothesis: Hopfield attention with high learned β enters sharp attractor mode in SDQ; less problematic in text because reservoir columns dilute the attention.

### Table rules (from experiments.md):
- Caption above the table: \caption{...}
- booktabs: \toprule / \midrule / \bottomrule only, no \hline, no vertical bars
- Metric direction in header: BPC ↓, Acc ↑
- Consistent decimal places per column
- Mark [TBD] for results not yet available

## Output format

```latex
\section{Experiments}

\subsection{Experimental Setup}
...

\subsection{Main Results}

\begin{table}[t]
\caption{SDQ benchmark results.}
...
\end{table}

...

\subsection{Ablation Studies}
...

\subsection{Analysis: Column Specialization}
...
```
```
