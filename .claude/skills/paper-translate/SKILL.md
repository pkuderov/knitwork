---
name: paper-translate
description: Use this skill when the user types "/paper-translate [ru|en]" or asks to translate the AAAI paper to Russian or English. Triggers on "переведи статью на русский", "translate paper to Russian", "translate to English", "перевод статьи", "русская версия статьи".
version: 1.0.0
---

# Paper-Translate Skill

Translates the AAAI paper between English and Russian while preserving LaTeX structure and AAAI formatting commands.

## Steps

1. Parse argument: `ru` (translate EN→RU) or `en` (translate RU→EN). If missing, ask the user.
2. Determine source and output file:
   - `ru`: source = `article/latex-2026.tex`, output = `article/latex-2026-ru.tex`
   - `en`: source = `article/latex-2026-ru.tex`, output = `article/latex-2026-en.tex`
3. Spawn a **foreground** sub-agent with the prompt below, substituting `{DIRECTION}`, `{SOURCE_FILE}`, `{OUTPUT_FILE}`
4. Report completion and output file path

## Sub-agent prompt

```
You are a scientific translation agent. Translate the AAAI paper from {DIRECTION}.

Source file: `{SOURCE_FILE}`
Output file: `{OUTPUT_FILE}`

## Read first
- `{SOURCE_FILE}` — full LaTeX source of the paper

## Translation rules

### What to translate:
- All natural language text inside environments: paragraphs, \caption{}, \section{}, \subsection{}, \paragraph{}, footnotes
- Abstract text
- Table row/column labels (not values/numbers)
- Algorithm comments if in natural language

### What NOT to translate (leave exactly as-is):
- All LaTeX commands: \begin{}, \end{}, \cite{}, \ref{}, \label{}, \textbf{}, etc.
- All mathematical formulas: $...$ and \[...\] and equation/align environments
- All code/pseudocode inside listings or algorithmic environments
- Author names, institution names, email addresses
- Package names, bibliography keys
- Numbers, units, metric names (BPC, Aq, FPS, etc.)
- Technical acronyms: Grid RNN, LRU, GRU, LSTM, SDQ, AAAI, etc.

### For EN→RU translation:
- Academic Russian style: third person, formal register
- Avoid: "в данной работе", "нами был предложен" — prefer "в этой работе", "предлагается"
- Technical terms on first use: write in English with Russian explanation in parentheses, e.g., "дельта-правило (delta rule)"
- Subsequent uses: use Russian transliteration or keep English — be consistent throughout
- Preserve paragraph breaks exactly
- Do NOT add translator's notes or explanations outside the text

### For RU→EN translation:
- Academic English style, active voice preferred
- Do not start abstract or intro with "In this paper"
- Match the formality level of the Russian source

## Output
Write the translated document to `{OUTPUT_FILE}` using the Write tool.
Preserve the complete LaTeX preamble unchanged.
After writing, print: "Translated: {SOURCE_FILE} → {OUTPUT_FILE} ({N} lines)"
```
