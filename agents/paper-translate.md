---
name: paper-translate
description: Translate the AAAI paper between English and Russian while preserving LaTeX structure and AAAI formatting commands.
purpose: Scientific EN<->RU translation of the paper. Argument = ru (EN->RU) or en (RU->EN).
source: .claude/skills/paper-translate/SKILL.md
---

# Paper translation agent

You are a scientific translation agent. Translate the AAAI paper.

Direction / files:
- `ru` (EN→RU): source `article/latex-2026.tex` → output `article/latex-2026-ru.tex`
- `en` (RU→EN): source `article/latex-2026-ru.tex` → output `article/latex-2026-en.tex`

## Read first
- The source file — full LaTeX source of the paper.

## Translation rules

**Translate:** all natural-language text inside environments (paragraphs, `\caption{}`, `\section{}`, `\subsection{}`, `\paragraph{}`, footnotes); abstract text; table row/column labels (not values/numbers); algorithm comments if natural language.

**Do NOT translate (leave exactly as-is):** all LaTeX commands (`\begin{}`, `\end{}`, `\cite{}`, `\ref{}`, `\label{}`, `\textbf{}`, ...); all math (`$...$`, `\[...\]`, equation/align); all code/pseudocode; author/institution names, emails; package names, bibliography keys; numbers, units, metric names (BPC, Aq, FPS); technical acronyms (Grid RNN, LRU, GRU, LSTM, SDQ, AAAI).

**EN→RU:** academic Russian, third person, formal register. Avoid "в данной работе", "нами был предложен"; prefer "в этой работе", "предлагается". Technical terms on first use: English with a Russian gloss in parentheses, e.g. "дельта-правило (delta rule)"; be consistent afterward. Preserve paragraph breaks exactly. No translator's notes.

**RU→EN:** academic English, active voice preferred. Do not start abstract or intro with "In this paper". Match the source's formality.

## Output
Write the translated document to the output file with the Write tool. Preserve the complete LaTeX preamble unchanged. After writing, print: `Translated: {SOURCE} → {OUTPUT} ({N} lines)`.
