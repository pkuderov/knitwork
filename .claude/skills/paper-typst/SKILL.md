---
name: paper-typst
description: Use this skill when the user types "/paper-typst [to-typst|to-latex]" or asks to convert the paper between LaTeX and Typst formats. Triggers on "конвертируй в typst", "convert to typst", "из latex в typst", "from typst to latex", "переведи в typst".
version: 1.0.0
---

# Paper-Typst Skill

Converts the AAAI paper between LaTeX (article/) and Typst format.
Handles math, tables, figures, algorithms, bibliography, and AAAI-specific macros.

## Steps

1. Parse argument:
   - `to-typst`: LaTeX → Typst. Source: `article/latex-2026.tex`, output: `article/paper.typ`
   - `to-latex`: Typst → LaTeX. Source: `article/paper.typ`, output: `article/paper-from-typst.tex`
   - If missing: ask the user
2. Spawn a **foreground** sub-agent with the prompt below, substituting `{DIRECTION}`, `{SOURCE}`, `{OUTPUT}`
3. Report completion

## Sub-agent prompt

```
You are a document conversion agent. Convert `{SOURCE}` from {DIRECTION} and write to `{OUTPUT}`.

## Read first
- `{SOURCE}` — full source document

## Conversion rules

### LaTeX → Typst

**Document structure:**
- Remove LaTeX preamble (\documentclass, \usepackage, etc.)
- Add Typst preamble at top:
  ```typst
  #import "@preview/scholarly-taffy:0.2.0": *   // or use basic typst
  #set text(font: "New Computer Modern", size: 10pt)
  #set math.equation(numbering: "(1)")
  ```
- `\maketitle` → `#maketitle` or manual title block
- `\begin{abstract}...\end{abstract}` → `#abstract[...]`

**Sections:**
- `\section{X}` → `= X`
- `\subsection{X}` → `== X`
- `\subsubsection{X}` → `=== X`
- `\paragraph{X}` → `*X*` (bold inline)

**Text formatting:**
- `\textbf{X}` → `*X*`
- `\emph{X}` or `\textit{X}` → `_X_`
- `\texttt{X}` → `` `X` ``
- `\footnote{X}` → `#footnote[X]`
- `~` (non-breaking space) → `#h(0pt, weak: true)` or just space

**Math:**
- Inline `$...$` → `$...$` (same in Typst)
- Display `\[...\]` or `\begin{equation}...\end{equation}` → `$ ... $` on its own line
- `\begin{align}...\end{align}` → `$ ... $` with `&` alignment
- Common LaTeX math: `\mathbf` → `bold()`, `\mathcal` → `cal()`, `\hat` → `hat()`, `\tilde` → `tilde()`
- `\text{...}` inside math → `"..."` in Typst math mode

**Tables:**
- `\begin{table}[t]\caption{X}\begin{tabular}{lcc}...\end{tabular}\end{table}` →
  ```typst
  #figure(
    caption: [X],
    table(
      columns: (auto, auto, auto),
      table.header([Col1], [Col2], [Col3]),
      [r1c1], [r1c2], [r1c3],
      ...
    )
  )
  ```
- `\toprule` / `\midrule` / `\bottomrule` → use table.hline() in Typst
- `\multicolumn{N}{c}{X}` → `table.cell(colspan: N)[X]`

**Figures:**
- `\begin{figure}[t]\includegraphics[width=...]{file}\caption{X}\label{Y}\end{figure}` →
  ```typst
  #figure(
    image("file.pdf", width: 80%),
    caption: [X],
  ) <Y>
  ```

**References and citations:**
- `\label{X}` → `<X>` (after the element)
- `\ref{X}` → `@X`
- `\cite{X}` → `@X`
- `\bibliography{...}` → use Typst's `#bibliography("refs.bib")`

**Algorithms:**
- `\begin{algorithm}...\end{algorithm}` → use `#import "@preview/lovelace:0.2.0": *` and pseudocode block
- Or convert to a numbered list with `#block` styling

**AAAI-specific:**
- `\pdfinfo{...}` → omit (Typst handles PDF metadata differently)
- `\frenchspacing` → omit
- Column layout macros → Typst uses `#columns(2)[...]` for two-column layout

---

### Typst → LaTeX

Reverse all rules above. Key mappings:
- `= X` → `\section{X}`
- `== X` → `\subsection{X}`
- `*X*` → `\textbf{X}`
- `_X_` → `\emph{X}`
- `$ ... $` display → `\begin{equation}...\end{equation}`
- `@X` (citation) → `\cite{X}`
- `@X` (figure/table ref) → `\ref{X}`
- `#figure(image(...), caption: [...])` → `\begin{figure}...\includegraphics...\caption{...}\end{figure}`
- `table(...)` → `\begin{tabular}...\end{tabular}` with booktabs

Add standard LaTeX preamble for AAAI 2026:
```latex
\documentclass[letterpaper]{article}
\usepackage[submission]{aaai2026}
\usepackage{times,helvet,courier}
\usepackage[hyphens]{url}
\usepackage{graphicx,natbib,caption}
\usepackage{algorithm,algorithmic}
\usepackage{booktabs}
```

---

## Important notes
- Preserve all content exactly — do NOT paraphrase or summarize
- If a LaTeX construct has no clean Typst equivalent, add a comment: `// TODO: manual check`
- Math should compile correctly — double-check operator names (sin, cos, etc. need `\` in LaTeX but not in Typst)
- Write the full converted document to `{OUTPUT}` using the Write tool
- After writing, print a conversion summary: how many sections, tables, figures, equations were processed
```
