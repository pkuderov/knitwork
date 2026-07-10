---
name: paper-typst
description: Convert the AAAI paper between LaTeX and Typst (both directions), handling math, tables, figures, algorithms, bibliography, and AAAI-specific macros.
purpose: Bidirectional LaTeX<->Typst conversion. Argument = to-typst (LaTeX->Typst) or to-latex (Typst->LaTeX).
source: .claude/skills/paper-typst/SKILL.md
---

# LaTeX <-> Typst conversion agent

You are a document conversion agent. Convert the source and write to the output.

Direction / files:
- `to-typst` (LaTeX → Typst): source `article/latex-2026.tex` → output `article/paper.typ`
- `to-latex` (Typst → LaTeX): source `article/paper.typ` → output `article/paper-from-typst.tex`

## Read first
- The source document, in full.

## Conversion rules

### LaTeX → Typst

**Document structure:** remove the LaTeX preamble (`\documentclass`, `\usepackage`, ...). Add a Typst preamble:
```typst
#import "@preview/scholarly-taffy:0.2.0": *   // or basic typst
#set text(font: "New Computer Modern", size: 10pt)
#set math.equation(numbering: "(1)")
```
`\maketitle` → `#maketitle` or a manual title block; `\begin{abstract}...\end{abstract}` → `#abstract[...]`.

**Sections:** `\section{X}` → `= X`; `\subsection{X}` → `== X`; `\subsubsection{X}` → `=== X`; `\paragraph{X}` → `*X*`.

**Text:** `\textbf{X}` → `*X*`; `\emph{X}`/`\textit{X}` → `_X_`; `\texttt{X}` → `` `X` ``; `\footnote{X}` → `#footnote[X]`; `~` → `#h(0pt, weak: true)` or a space.

**Math:** inline `$...$` → `$...$`; display `\[...\]` / `\begin{equation}...\end{equation}` → `$ ... $` on its own line; `\begin{align}...\end{align}` → `$ ... $` with `&` alignment; `\mathbf` → `bold()`, `\mathcal` → `cal()`, `\hat` → `hat()`, `\tilde` → `tilde()`; `\text{...}` in math → `"..."`.

**Tables:** `\begin{table}[t]\caption{X}\begin{tabular}{lcc}...` →
```typst
#figure(
  caption: [X],
  table(
    columns: (auto, auto, auto),
    table.header([Col1], [Col2], [Col3]),
    [r1c1], [r1c2], [r1c3],
  )
)
```
`\toprule`/`\midrule`/`\bottomrule` → `table.hline()`; `\multicolumn{N}{c}{X}` → `table.cell(colspan: N)[X]`.

**Figures:** `\begin{figure}[t]\includegraphics[width=...]{file}\caption{X}\label{Y}\end{figure}` →
```typst
#figure(
  image("file.pdf", width: 80%),
  caption: [X],
) <Y>
```

**References/citations:** `\label{X}` → `<X>` (after the element); `\ref{X}` → `@X`; `\cite{X}` → `@X`; `\bibliography{...}` → `#bibliography("refs.bib")`.

**Algorithms:** `\begin{algorithm}...\end{algorithm}` → `#import "@preview/lovelace:0.2.0": *` pseudocode block, or a numbered list with `#block` styling.

**AAAI-specific:** `\pdfinfo{...}` → omit; `\frenchspacing` → omit; column layout macros → `#columns(2)[...]`.

### Typst → LaTeX

Reverse all rules. Key mappings: `= X` → `\section{X}`; `== X` → `\subsection{X}`; `*X*` → `\textbf{X}`; `_X_` → `\emph{X}`; display `$ ... $` → `\begin{equation}...\end{equation}`; `@X` (citation) → `\cite{X}`; `@X` (ref) → `\ref{X}`; `#figure(image(...), caption: [...])` → `\begin{figure}...\includegraphics...\caption{...}\end{figure}`; `table(...)` → `\begin{tabular}...\end{tabular}` with booktabs.

Add the AAAI 2026 LaTeX preamble:
```latex
\documentclass[letterpaper]{article}
\usepackage[submission]{aaai2026}
\usepackage{times,helvet,courier}
\usepackage[hyphens]{url}
\usepackage{graphicx,natbib,caption}
\usepackage{algorithm,algorithmic}
\usepackage{booktabs}
```

## Important notes
- Preserve all content exactly — do NOT paraphrase or summarize.
- If a construct has no clean equivalent, add a comment `// TODO: manual check`.
- Math must compile — double-check operator names (sin, cos, etc. need `\` in LaTeX but not in Typst).
- Write the full converted document to the output with the Write tool.
- After writing, print a conversion summary: how many sections, tables, figures, and equations were processed.
