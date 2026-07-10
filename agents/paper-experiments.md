---
name: paper-experiments
description: Draft the Experiments section from the result tables in docs/methods/grnn_harmonic.md, using the 3-question structure (beats baselines? / which modules matter? / how far does it generalize?).
purpose: Write the Experiments section for the Grid RNN AAAI 2026 paper.
source: .claude/skills/paper-experiments/SKILL.md
---

# Experiments writing agent

You are an academic writing agent for an AAAI 2026 paper about Grid RNN approaches.

## Files to read (read ALL before writing)

Results:
- `docs/methods/grnn_harmonic.md` — ALL result tables: SDQ v2 (#043), SDQ v3.1 (#055), text8 v2/v3, Shakespeare v2, MIKASA (#058), diagnostic metrics
- `docs/methods/` — check for result tables in grnn.md, grnn_lru.md, other grnn*.md files

Writing guide:
- `agents/references/experiments.md` — 3-question structure, table formatting rules, figure/caption rules

## Task — write the Experiments section

**4.1 Experimental Setup** (three short paragraphs):
- *Benchmarks:* SDQ (Store-Distract-Query; synthetic associative memory; metrics Acc/query (Aq), Acc/distract; 1B steps); Text8 / Shakespeare (char-level LM; BPC; 520M steps); TreasureHunt (MIKASA RepeatFirstEasy; RL with PPO; episode return EpRet; 200M steps).
- *Baselines:* grnn (base, no LRU/memory); grnn_lru; hgrnn_lru; GRU/LSTM at ~2M params; note any baselines with known results from docs/methods/.
- *Implementation:* all models ~2.1M params, CUDA, Adam; HarmonicGridRNN config 3 layers × 4 columns, dk=32, dv=128, n_attn_heads=4; text n_reservoir_cols=4, SDQ/RL n_reservoir_cols=0.

**4.2 Main Results:**
- Table 1 (SDQ): Model | Params | Acc/query ↑ | Acc/distract ↑ | Steps. booktabs style; no vertical lines; highlight best; include grnn_harmonic v2 peak (Aq=0.750 at 95M) and v3.1 (Aq=0.715 at 110M); include available baselines else mark [TBD].
- Table 2 (Language Modeling): Model | Params | text8 BPC ↓ | Shakespeare BPC ↓. Include grnn_harmonic (text8 BPC=1.676 at 270M, Shakespeare BPC=0.788 at 340M); LSTM ~2M reference ≈1.6 BPC on text8.
- Short paragraph summarizing takeaways from both tables.

**4.3 Ablation Studies:**
- *v2 vs v3.1 on SDQ* — table with Aq at 10M/30M/50M/75M/95M/110M for both; discuss v3.1 converges slower (−0.035 Aq at 75M) but multi_col_head design justified by column collapse in v3.1.
- *Memory diagnostics* — table/figure: | Metric | Layer 0 | Layer 1 | Layer 2 | Interpretation |; for SDQ v2 at 300M: W_norm (1.06/1.46/1.88), alpha (0.85/0.86/0.87), surprise, fullness, error; discuss monotonic W_norm growth and fullness=1-3% (not saturated).

**4.4 Analysis: Column Specialization:**
- Column diversity/norm for SDQ v3.1: col/diversity L0=2.34, L1=2.80, L2=7.79 (severe specialization collapse at top layer); col0_norm L2 rises to 325 vs 15-25 (single column dominates). Compare with text8 (v3): diversity L0=1.92/L1=1.80/L2=2.16, stable col norms. Hypothesis: Hopfield attention with high learned β enters sharp attractor mode in SDQ; less problematic in text because reservoir columns dilute attention.

## Table rules (from experiments.md)
- Caption above the table (`\caption{...}`).
- booktabs only (`\toprule` / `\midrule` / `\bottomrule`), no `\hline`, no vertical bars.
- Metric direction in header (BPC ↓, Acc ↑).
- Consistent decimal places per column. Mark `[TBD]` for missing results.

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
