---
name: latex-article
description: Create and compile a LaTeX document (article, thesis, report) from a template or from scratch, compiling to PDF and fixing errors, documenting the fixes.
purpose: Author + compile a LaTeX document. Argument = topic or template path [--template <path>] [--engine lualatex|pdflatex].
source: ~/.claude/commands/latex-article.md (translated from Russian)
---

# LaTeX authoring agent

Create and compile a LaTeX document from the arguments: **$ARGUMENTS**.

## Environment (fixed)
- TeX Live 2023, paths `/usr/bin/pdflatex`, `/usr/bin/lualatex`.
- **`texlive-lang-cyrillic` is NOT installed** — T2A encoding and `\usepackage[russian]{babel}` with pdflatex fail with `t2aenc.def not found`.
- Available: `fontspec`, `polyglossia`, DejaVu fonts (support Cyrillic). CMU (Computer Modern Unicode) fonts are absent.

## Rule #1: Russian text — lualatex only
**Never** use for Russian text:
```latex
\usepackage[cp1251]{inputenc}   % error in a UTF-8 environment
\usepackage[T2A]{fontenc}        % error: t2aenc.def not found
\usepackage[russian]{babel}      % error: russian.ldf not found (no lang-cyrillic)
```
Correct preamble for Russian:
```latex
\usepackage{fontspec}
\usepackage{polyglossia}
\setmainlanguage{russian}
\setotherlanguage{english}
\setmainfont{Linux Libertine O}   % better than DejaVu: no Bold artifacts
\setsansfont{Linux Libertine O}
\setmonofont{DejaVu Sans Mono}
\usepackage{microtype}            % improves justification, removes overfull
```
Compile with:
```bash
lualatex -interaction=nonstopmode document.tex
lualatex -interaction=nonstopmode document.tex   # second pass for \ref and \cite
```

## Rule #2: Figures
| Engine | Format | Command |
|---|---|---|
| lualatex | PNG, JPG, PDF | `\includegraphics[width=0.9\linewidth]{img.png}` |
| pdflatex | PNG, JPG, PDF | same |
| latex | EPS | `\includegraphics{fig.eps}` |

EPS → PDF: `epstopdf fig.eps` (then `\includegraphics{fig.pdf}`). Filenames with spaces or non-ASCII work with lualatex but are best avoided.

## Preamble template (Russian article)
```latex
\documentclass[12pt]{article}
\usepackage{fontspec}
\usepackage{polyglossia}
\setmainlanguage{russian}
\setotherlanguage{english}
\setmainfont{Linux Libertine O}
\setsansfont{Linux Libertine O}
\setmonofont{DejaVu Sans Mono}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{cite}
\usepackage{caption}
```

## Sirius template preamble (follow the template strictly)
The Sirius template (`sirius_template.tex`) uses custom commands — **do not redefine**:
```latex
\def\title#1{...}
\def\author#1{...}
\def\affiliation#1{...}
```
Replace only the `inputenc`/`fontenc`/`babel` block with fontspec+polyglossia (Rule #1). Leave unchanged: `\setlength` margins, `\pagestyle`, `\def\figurename`, `caption2`.

Document structure:
```latex
\begin{center}
\title{Title}
\author{\underline{Surname I.\,O.}$^{1}$}
\affiliation{${}^{1}$Organization, City, Country}
\end{center}

% ... text, formulas, figures ...

\noindent\textit{Supervisor:} Surname I.\,O., Dr. Sci., professor, Organization.

\begin{thebibliography}{99}
\bibitem{key} Author. \textit{Title} // Journal, year, vol., pp.
\end{thebibliography}
```

## Workflow
1. Read the template (if any) and source material.
2. Write the `.tex` file.
3. First compile: `lualatex -interaction=nonstopmode file.tex 2>&1 | grep -E "^!|Error|Output"`.
4. Fix errors (see table below).
5. Second compile: confirm `Output written on ... (N pages)`.
6. Check the PDF via Read — assess it visually.
7. If over the page budget: shrink `[width=0.85\linewidth]` on figures, tighten the text.

## Known errors and fixes
| Error | Cause | Fix |
|---|---|---|
| `t2aenc.def not found` | texlive-lang-cyrillic missing | drop `\usepackage[T2A]{fontenc}`, switch to lualatex+fontspec |
| `Unknown option 'russian'` (babel) | texlive-lang-cyrillic missing | drop `\usepackage[russian]{babel}`, use polyglossia |
| `Unicode character X not set up` | pdflatex + UTF-8 without T2A | switch to lualatex |
| `Language russian not found in language.dat.lua` | polyglossia without hyphenation patterns | **non-critical** — Russian hyphenation off, text still renders |
| Undefined references (`\ref`, `\cite`) | single compile pass | run compilation twice |
| Over the page budget | figures take too much room | reduce `width=` on `\includegraphics`, tighten text |
| `caption2.sty` warnings | deprecated package | acceptable; replace with `caption` if it interferes |
| Figures drift to end of document | `[!ht]` without `\usepackage{float}` | add `\usepackage{float}` and use `[H]` |
| `fontspec requires XeTeX or LuaTeX` | compiled with pdflatex | use `lualatex`, not `pdflatex` |
| Page number runs past the bottom margin | lualatex defaults to Letter (792pt); template assumes A4 (842pt) | add `a4paper`: `\documentclass[12pt,a4paper]{article}` |
| Wide spaces / "rivers" | no Russian hyphenation (`\sloppy` too aggressive) | `\tolerance=1500 \emergencystretch=15pt \hfuzz=8pt` + `microtype` |
| Dancing letters / wide spaces in **Bold** | DejaVu Serif Bold poor without Russian hyphenation | switch font to `Linux Libertine O` (correct Cyrillic Bold) |
| Overfull hbox on a long Russian word | no hyphenation patterns | insert `\-` in long words |
| `\noindent` line stretches | justified + short line without hyphenation | wrap in `{\raggedright\noindent...\par}` |
| Minipage caption wraps | narrow column, long caption | shorten caption, use `~` for spaces: `(mean~$\pm$~std)` |

## Side-by-side figures (two per row)
```latex
\usepackage{float}   % preamble

\begin{figure}[H]
\centering
\begin{minipage}[t]{0.48\linewidth}
  \centering
  \includegraphics[width=\linewidth]{fig1.png}
  \caption{\small Caption fig.~1}
  \label{fig:1}
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\linewidth}
  \centering
  \includegraphics[width=\linewidth]{fig2.png}
  \caption{\small Caption fig.~2}
  \label{fig:2}
\end{minipage}
\end{figure}
```
Key points: `[H]` forces placement (needs `float`); `[t]` aligns minipages at the top (figures of different heights); `0.48\linewidth` × 2 + `\hfill` fills the width without overflow; captions inside the minipage number correctly.

## Useful commands
```bash
lualatex -interaction=nonstopmode file.tex
lualatex -interaction=nonstopmode file.tex 2>&1 | grep -E "^!|Output|pages"
lualatex file.tex > compile.log 2>&1
epstopdf figure.eps
pdftotext output.pdf -
fc-list | grep -i "DejaVu\|Liberation\|FreeSerif"
```
