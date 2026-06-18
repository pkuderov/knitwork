// AAAI 2026 Typst Template
// Reflects all structural elements from latex-2026.tex:
//   page, typography, headings (1–3 + paragraph), lists, math, footnotes,
//   quotes, figures (single/wide), tables (single/wide), algorithms,
//   listings, links block, equal-contribution, appendix.
// Compile: typst compile aaai2026.typ

// ─────────────────────────────────────────────────────────────────────────────
// PAGE  (8.5 × 11 in, no numbers, no header/footer)
// ─────────────────────────────────────────────────────────────────────────────

#set page(
  paper: "us-letter",
  // AAAI margins: top 1.25in (p.1) / 0.75in (rest), l/r 0.75in, bottom 1.25in.
  // Typst has no per-page top margin; 0.75in is used and an extra 0.5in v-pad
  // is added inside the title block to approximate the first-page requirement.
  margin: (top: 0.75in, bottom: 1.25in, left: 0.75in, right: 0.75in),
  numbering: none,
  header: none,
  footer: none,
)

// ─────────────────────────────────────────────────────────────────────────────
// TYPOGRAPHY
// ─────────────────────────────────────────────────────────────────────────────

// Linux Libertine O ≈ Times New Roman (freely embeddable by Typst).
// Swap to "Times New Roman" if the font is available on your system.
// Sans: "Linux Biolinum O" ≈ Helvetica. Mono: system monospace for raw blocks.
#set text(font: "Linux Libertine O", size: 10pt, lang: "en")

// 12pt leading = 10pt font + 2pt extra. Frenchspacing (no extra space after ".")
// is Typst's default — nothing to configure.
#set par(leading: 2pt, justify: true, first-line-indent: 10pt)

// ─────────────────────────────────────────────────────────────────────────────
// HEADINGS
// ─────────────────────────────────────────────────────────────────────────────

// Unnumbered by default (\setcounter{secnumdepth}{0} equivalent).
// To enable section numbers: set heading(numbering: "1.") and remove next line.
#set heading(numbering: none)

// Section (\section)
#show heading.where(level: 1): it => block(above: 1em, below: 0.5em)[
  #set text(size: 10pt, weight: "bold")
  #it.body
]
// Subsection (\subsection)
#show heading.where(level: 2): it => block(above: 0.75em, below: 0.3em)[
  #set text(size: 10pt, weight: "bold")
  #it.body
]
// Subsubsection (\subsubsection)
#show heading.where(level: 3): it => block(above: 0.6em, below: 0.2em)[
  #set text(size: 10pt, weight: "bold", style: "italic")
  #it.body
]

// \paragraph{Title.} — inline bold run-in heading, text continues on same line.
// Usage: #par-heading[Heading Text] rest of paragraph here.
#let par-heading(title) = strong[#title.#h(0.5em)]

// Unnumbered section when numbering is globally on (\section* equivalent):
//   #heading(level: 1, numbering: none)[Acknowledgments]

// ─────────────────────────────────────────────────────────────────────────────
// LISTS
// ─────────────────────────────────────────────────────────────────────────────

#set list(indent: 1em, body-indent: 0.5em)   // \itemize
#set enum(indent: 1em, body-indent: 0.5em)   // \enumerate

// ─────────────────────────────────────────────────────────────────────────────
// MATH
// ─────────────────────────────────────────────────────────────────────────────

// Inline:  $x in RR^d$
// Display: $ bold(h)_t = f(bold(W) bold(x)_t + bold(b)) $
//
// Typst uses New Computer Modern Math by default.
// AAAI permits Computer Modern for math only (not body text) — this is fine.
// Minimum math font size: 6.5pt (reduce with #text(size: 6.5pt)[$...$]).

// ─────────────────────────────────────────────────────────────────────────────
// FIGURES AND CAPTIONS
// ─────────────────────────────────────────────────────────────────────────────

// AAAI: 10pt roman captions placed BELOW figures and tables.
#show figure.caption: set text(size: 10pt)

// \begin{figure*} equivalent — spans both columns (requires Typst 0.11+).
#let wide-figure(caption: [], body) = figure(
  body,
  caption: caption,
  placement: top,
  scope: "parent",
)

// ─────────────────────────────────────────────────────────────────────────────
// TABLES
// ─────────────────────────────────────────────────────────────────────────────

// AAAI: 10pt roman type; caption BELOW. Compress with column-gutter if needed.
// \begin{table*} equivalent — spans both columns.
#let wide-table(caption: [], body) = figure(
  body,
  caption: caption,
  placement: top,
  scope: "parent",
  kind: table,
)

// ─────────────────────────────────────────────────────────────────────────────
// ALGORITHMS  (\usepackage{algorithm,algorithmic} equivalent)
// ─────────────────────────────────────────────────────────────────────────────

// Caption rendered at TOP inside horizontal rules (AAAI algorithm style).
#show figure.where(kind: "algorithm"): it => block(
  width: 100%,
  breakable: false,
)[
  #set align(left)
  #line(length: 100%, stroke: 0.5pt)
  #pad(x: 0.5em, y: 0.3em)[
    *#it.supplement #context it.counter.display(it.numbering):*
    #if it.caption != none { it.caption.body }
  ]
  #line(length: 100%, stroke: 0.5pt)
  #pad(x: 1.5em, y: 0.4em)[
    #set par(first-line-indent: 0pt, spacing: 0.3em)
    #set text(size: 10pt)
    #it.body
  ]
  #line(length: 100%, stroke: 0.5pt)
]

// Algorithmic command set  (\STATE, \WHILE, \IF, \FOR, \RETURN, etc.)
#let algo-state(body) = block(spacing: 0.2em)[#body]
#let algo-require(body)  = algo-state[*Input:* #body]
#let algo-ensure(body)   = algo-state[*Output:* #body]
#let algo-param(body)    = algo-state[*Parameter:* #body]
#let algo-return(body)   = algo-state[*return* #body]
#let algo-comment(body)  = text(style: "italic")[\// #body]

#let algo-while(cond, body) = {
  algo-state[*while* #cond *do*]
  pad(left: 1.5em)[#body]
  algo-state[*end while*]
}
#let algo-for(cond, body) = {
  algo-state[*for* #cond *do*]
  pad(left: 1.5em)[#body]
  algo-state[*end for*]
}
#let algo-if(cond, body) = {
  algo-state[*if* #cond *then*]
  pad(left: 1.5em)[#body]
  algo-state[*end if*]
}
#let algo-if-else(cond, then-body, else-body) = {
  algo-state[*if* #cond *then*]
  pad(left: 1.5em)[#then-body]
  algo-state[*else*]
  pad(left: 1.5em)[#else-body]
  algo-state[*end if*]
}

// Float container — mirrors \begin{algorithm}[tb].
// Usage: #algorithm(caption: [Title])[ #algo-require[...] ... ]
#let algorithm(caption: [], body) = figure(
  body,
  kind: "algorithm",
  supplement: [Algorithm],
  caption: caption,
  placement: top,
)

// ─────────────────────────────────────────────────────────────────────────────
// LISTINGS  (\usepackage{newfloat,listings} equivalent)
// ─────────────────────────────────────────────────────────────────────────────

// Caption at TOP, horizontal rules, no background, 9pt monospace body.
#show figure.where(kind: "listing"): it => block(
  width: 100%,
  breakable: false,
)[
  #set align(left)
  #line(length: 100%, stroke: 0.5pt)
  #pad(x: 0.5em, y: 0.3em)[
    *#it.supplement #context it.counter.display(it.numbering):*
    #if it.caption != none { it.caption.body }
  ]
  #line(length: 100%, stroke: 0.5pt)
  #pad(x: 1em, y: 0.3em)[
    #set text(size: 9pt)
    #set par(first-line-indent: 0pt)
    #it.body
  ]
  #line(length: 100%, stroke: 0.5pt)
]

// Float container — mirrors \begin{listing}[tb].
// Body must be a raw block: ```lang\ncode\n```
// Usage: #code-listing(caption: [title `file.hs`])[ ```haskell\n...\n``` ]
#let code-listing(caption: [], body) = figure(
  body,
  kind: "listing",
  supplement: [Listing],
  caption: caption,
  placement: top,
)

// ─────────────────────────────────────────────────────────────────────────────
// QUOTE / EXTRACT  (\begin{quote} equivalent)
// ─────────────────────────────────────────────────────────────────────────────

// AAAI: long quotations indented 10pt from left and right.
#let extract(body) = pad(x: 10pt)[
  #set par(first-line-indent: 0pt)
  #body
]

// ─────────────────────────────────────────────────────────────────────────────
// LINKS BLOCK  (\begin{links} equivalent, after abstract, before body)
// ─────────────────────────────────────────────────────────────────────────────

// Usage: #links-block[*Code:* #link("url")[url] \ *Datasets:* ...]
// Do not de-anonymize yourself with these links in blind submissions.
#let links-block(body) = pad(x: 0.5in)[
  #set par(first-line-indent: 0pt)
  #body
]

// ─────────────────────────────────────────────────────────────────────────────
// EQUAL CONTRIBUTION  (\equalcontrib equivalent)
// ─────────────────────────────────────────────────────────────────────────────

// Usage: Author Name#equalcontrib#super[1]
// Adds a * superscript; add a matching footnote or note in the affiliations.
#let equalcontrib = super[\*]

// ─────────────────────────────────────────────────────────────────────────────
// TITLE BLOCK
// ─────────────────────────────────────────────────────────────────────────────

// Full-width block above two-column body.
// \thanks equivalent: use #footnote[...] inline within the authors argument.
#let aaai-title(title: [], authors: [], affiliations: []) = {
  v(0.5in)  // extra padding → total ~1.25in from physical top of page
  align(center)[
    #text(size: 16pt, weight: "bold")[#title]
    #v(6pt)
    #text(size: 12pt)[#authors]
    #v(3pt)
    #text(size: 9pt)[#affiliations]
  ]
  v(1em)
  // AAAI copyright slug appears here in camera-ready (hardcoded in aaai2026.sty).
}

// ─────────────────────────────────────────────────────────────────────────────
// ABSTRACT
// ─────────────────────────────────────────────────────────────────────────────

// Full-width, indented block above two-column body. No references inside.
#let aaai-abstract(body) = pad(x: 0.5in)[
  #align(center)[#text(weight: "bold")[Abstract]]
  #set par(first-line-indent: 0pt)
  #body
]

// ═════════════════════════════════════════════════════════════════════════════
// PAPER CONTENT — replace placeholder text with your own.
// ═════════════════════════════════════════════════════════════════════════════

#aaai-title(
  title: [Paper Title in Title Case],
  authors: [
    // #equalcontrib adds * for equal contribution.
    // #footnote[...] is the \thanks equivalent.
    // #super[n] creates superscript affiliation indices.
    Author One#equalcontrib#super[1]#footnote[Equal contribution.],
    Author Two#equalcontrib#super[1,2],
    Author Three#super[3]
  ],
  affiliations: [
    #super[1]Institution One, City, Country \
    #super[2]Institution Two, City, Country \
    #super[3]Institution Three, City, Country \
    author1\@example.com, author2\@example.com
  ],
)

#aaai-abstract[
  Your abstract here. Do not include references in the abstract.
  Summarize the problem, method, and key results in one concise paragraph.
]

// Optional links block (after abstract, before body).
// #v(0.5em)
// #links-block[
//   *Code:* https://example.com/code \
//   *Datasets:* https://example.com/data \
//   *Extended version:* https://arxiv.org/abs/xxxx.yyyy
// ]

#v(1em)

// Two-column body: 3.3in each column, 0.375in gutter (AAAI specification).
#columns(2, gutter: 0.375in)[

= Introduction

Body text is 10pt Times with 12pt leading, justified, 10pt first-line indent.

Inline math: $x in RR^d$. Display math:

$ bold(h)_t = sigma(bold(W) bold(x)_t + bold(U) bold(h)_{t-1} + bold(b)) $

Inline code / verbatim: #raw("x = f(h)").
Italic: _text_. Bold: *bold*. Typewriter: #raw("code").
URL: #link("https://example.com")[example.com].
Footnote.#footnote[Footnotes appear at column bottom, separated by a thin rule.]

= Related Work

Long quotation / extract (10pt indent on each side, \begin{quote} equivalent):

#extract[
  This is an example of an extract or quotation. Quotation marks are not
  necessary when the text is offset in a block like this, and the source
  is cited in the text.
]

Itemize list (\begin{itemize}):
- First item
- Second item
- Third item

Enumerate list (\begin{enumerate}):
+ First item
+ Second item
+ Third item

= Method

== Subsection Heading

Text under a subsection. Reference a figure: @fig-single.

=== Subsubsection Heading.

#par-heading[Paragraph Heading.] Text continues on the same line after
the bold run-in heading, just like \paragraph{} in LaTeX.

= Experiments

// ── Single-column figure (\begin{figure}[t])
#figure(
  // Replace with: image("figure1.pdf", width: 90%) or .png/.jpg
  rect(width: 90%, height: 4cm, fill: luma(230))[#align(center + horizon)[figure1]],
  caption: [Single-column figure. Caption is 10pt roman, placed below.
    Crop figures in a graphics program before including them.],
  placement: top,
) <fig-single>

As shown in @fig-single, single-column figures float to the top of the
column. Reference tables with @tab-results.

// ── Double-column figure (\begin{figure*}[t])
#wide-figure(caption: [
  Full-width figure spanning both columns (#raw("\\begin{figure*}") equivalent).
  Use `scope: "parent"` in Typst to escape the columns block.
])[
  // Replace with: image("figure2.pdf", width: 80%) or .png/.jpg
  #rect(width: 80%, height: 4cm, fill: luma(230))[#align(center + horizon)[figure2]]
]

// ── Single-column table (\begin{table}[t])
#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    [*Method*], [*Acc*],  [*F1*],   [*AUC*],
    table.hline(),
    [Baseline], [82.3],   [81.1],   [0.89],
    [Ours],     [*85.7*], [*84.2*], [*0.92*],
    table.hline(),
  ),
  caption: [Single-column table. Caption is 10pt roman, placed below the table.
    Use `column-gutter: 1mm` to compress columns if too wide.],
  placement: top,
  kind: table,
) <tab-results>

// ── Double-column table (\begin{table*}[t])
#wide-table(caption: [Wide table spanning both columns.])[
  #table(
    columns: (auto,) * 6,
    stroke: none,
    table.hline(),
    [*Col A*], [*Col B*], [*Col C*], [*Col D*], [*Col E*], [*Col F*],
    table.hline(),
    [val], [val], [val], [val], [val], [val],
    table.hline(),
  )
]

// ── Algorithm (\begin{algorithm}[tb] + \begin{algorithmic}[1])
// Caption appears at top inside horizontal rules (AAAI requirement).
#algorithm(caption: [Example algorithm])[
  #algo-require[Algorithm's input]
  #algo-param[Optional list of parameters]
  #algo-ensure[Algorithm's output]
  Let $t = 0$.
  #algo-while[$t < T$][
    #algo-state[Do some action. #algo-comment[inline comment]]
    #algo-if-else[$"condition"$][
      #algo-state[Perform task A.]
    ][
      #algo-state[Perform task B.]
    ]
    #algo-state[$t <- t + 1$]
  ]
  #algo-return[solution]
] <alg-example>

See @alg-example. For a for-loop: `#algo-for[each $x in cal(D)$][...]`.

// ── Code listing (\begin{listing}[tb] + \begin{lstlisting}[language=...])
// Caption appears at top inside horizontal rules; no background color.
#code-listing(caption: [Example listing #raw("quicksort.hs")])[
  ```haskell
  quicksort :: Ord a => [a] -> [a]
  quicksort []     = []
  quicksort (p:xs) = quicksort lesser ++ [p] ++ quicksort greater
    where
      lesser  = filter (< p) xs
      greater = filter (>= p) xs
  ```
] <lst-quicksort>

See @lst-quicksort.

= Conclusion

Your conclusion here.

// Ethical Statement — optional, unnumbered (\section* equivalent).
// In Typst the default is unnumbered, so just use:
// = Ethical Statement

// Acknowledgments — optional, unnumbered, right before References.
// = Acknowledgments

// References.
// AAAI author-year citations: "(Author Year)" / "(Author Year; ...)".
// @key         → full citation, e.g. "(Smith 2020)"
// @key[year]   → year only
// Closest built-in CSL: "american-psychological-association".
// Supply a custom aaai.csl file for exact formatting.
// Uncomment after providing your .bib file:
// #bibliography("aaai2026.bib",
//   style: "american-psychological-association",
//   title: "References")

] // end columns


// ─────────────────────────────────────────────────────────────────────────────
// APPENDIX  (\appendix + \section{...} equivalent)
// ─────────────────────────────────────────────────────────────────────────────
// Appendices follow main content. AAAI requires letter-numbered sections
// when section numbering is enabled. To activate:
//
//   #counter(heading).update(0)
//   #set heading(numbering: "A.")  // → "A. Title", "B. Title", ...
//
//   #columns(2, gutter: 0.375in)[
//   = Reference Examples
//   <label: sec-reference-examples>
//
//   #par-heading[Book.] @book entry type. \
//   #par-heading[Article.] @article entry type. \
//   #par-heading[InProceedings.] @inproceedings entry type. \
//   #par-heading[TechReport.] @techreport entry type. \
//   #par-heading[PhdThesis.] @phdthesis entry type. \
//   #par-heading[Misc/ArXiv.] @misc with eprint and archivePrefix fields.
//
//   #heading(level: 1, numbering: none)[Acknowledgments]
//   Text of acknowledgments.
//
//   // #bibliography(...)  // if not already placed above
//   ] // end appendix columns
