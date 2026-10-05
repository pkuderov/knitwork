// MoSAIC architecture figure — standalone document (not part of paper_en.typ).
//
// Source of truth: knitwork/models/grnn_core.py, docs/methods/grnn_core.md
// Compile:  typst compile fig_architecture.typ fig_architecture.pdf
//
// AAAI-26 figure compliance:
//   - PDF output (no .eps/.gif), fonts embedded, no Type 3, vector line art
//   - all in-figure text >= 9pt, Times-metric family (TeX Gyre Termes)
//   - line widths 0.5-1.0pt, no hairlines
//   - CMYK base tints, dark ink text (contrast >= 4.5:1); every block also
//     carries a text label and role-specific shape, so the figure stays
//     decipherable when the archival version prints in grayscale
//   - natural size 6.54in x 5.17in, under the 7.00in AAAI text width => include
//     at 100% scale in a two-column-spanning figure*; never rescale, or the 9pt
//     floor is violated

#import "@preview/fletcher:0.5.7" as fletcher: diagram, node, edge
#import fletcher.shapes: house, hexagon

#set page(width: auto, height: auto, margin: 4pt, fill: white)
#set text(font: "TeX Gyre Termes", size: 9pt)
#show math.equation: set text(font: "TeX Gyre Termes Math", size: 9pt)

// ─────────────────────────────────────────────────────────────────────────────
// PALETTE (CMYK base tints; blobs use lighten/darken of the same hue)
// ─────────────────────────────────────────────────────────────────────────────

#let ink     = cmyk(0%, 0%, 0%, 88%)
#let t-state = cmyk(78%, 30%, 0%, 6%)    // recurrent column states
#let t-route = cmyk(0%, 42%, 88%, 0%)    // attention router
#let t-cell  = cmyk(62%, 0%, 62%, 10%)   // independent GRU cells
#let t-in    = cmyk(0%, 68%, 55%, 0%)    // external input
#let t-out   = cmyk(12%, 0%, 82%, 6%)    // readout

// ─────────────────────────────────────────────────────────────────────────────
// BUILDING BLOCKS
// ─────────────────────────────────────────────────────────────────────────────

#let blob(pos, label, tint: t-state, ..args) = node(
  pos, align(center, label),
  fill: tint.lighten(76%),
  stroke: 0.8pt + tint.darken(18%),
  corner-radius: 4pt,
  inset: 5pt,
  ..args,
)

// One column slot, tinted a shade darker than its containing blob.
#let sq(tint) = box(
  width: 7pt, height: 7pt, radius: 1pt,
  stroke: 0.6pt + tint.darken(18%),
  fill: tint.lighten(42%),
)

// A row of C column slots; `extra` appends a differently-tinted slot (e.g. e_t),
// set off by a wider gap so the external input reads as a separate participant.
// `first` re-tints column 0, the hardwired readout column.
#let slots(n, tint, first: none, extra: ()) = grid(
  columns: (..(auto,) * n, ..extra.map(_ => auto)),
  column-gutter: (..(1.8pt,) * (n - 1), ..(5pt,) * extra.len()),
  align: horizon,
  ..range(n).map(i => sq(if i == 0 and first != none { first } else { tint })),
  ..extra.map(g => sq(g)),
)

// Label line above a row of column slots.
#let blk(lbl, n, tint, first: none, extra: ()) = stack(
  dir: ttb, spacing: 4pt,
  align(center, lbl),
  align(center, slots(n, tint, first: first, extra: extra)),
)

// Routing matrix pi^l_t: C queries (rows) x C+I messages (columns).
// Shading is illustrative — a sharpened, mostly-diagonal distribution.
#let pi-weights = (
  (0.62, 0.08, 0.05, 0.03, 0.22),
  (0.14, 0.55, 0.19, 0.08, 0.04),
  (0.06, 0.11, 0.60, 0.20, 0.03),
  (0.09, 0.05, 0.16, 0.66, 0.04),
)

#let pi-matrix = grid(
  columns: 5, column-gutter: 1.2pt, row-gutter: 1.2pt,
  ..pi-weights.flatten().map(w => box(
    width: 6.5pt, height: 6.5pt,
    stroke: 0.5pt + t-route.darken(18%),
    fill: t-route.darken(8%).lighten(calc.round(88 - 74 * w) * 1%),
  )),
)


// ─────────────────────────────────────────────────────────────────────────────
// PANEL (c) — inside the router: the routing distribution is built explicitly
// ─────────────────────────────────────────────────────────────────────────────

#let step(pos, label, name) = blob(
  pos, label, tint: t-route, name: name, inset: 4pt,
)

#let panel-c = diagram(
  spacing: (7mm, 7mm),
  node-shape: rect,
  edge-stroke: 0.8pt + ink,
  edge-corner-radius: 4pt,
  mark-scale: 70%,
  label-size: 9pt,
  label-sep: 2.5pt,

  // sources
  blob((0, 0), $bold(H)^l_(t-1)$, name: <src-q>),
  blob((0, 2), $bold(M)^l_t$, name: <src-kv>),

  // identity-augmented SiLU projections
  step((1, 0), $"SiLU" bold(W)^Q (dot + bold(a)^Q)$, <pq>),
  step((1, 1), $"SiLU" bold(W)^K (dot + bold(a)^K)$, <pk>),
  step((1, 2), $"SiLU" bold(W)^V (dot)$, <pv>),

  // scores
  step((2, 0.5), $bold(q) bold(k)^top \/ sqrt(H\/R)$, <scores>),

  // train-only routing noise
  blob((2, -0.7), $+ epsilon ~ cal(N)(0, sigma^2)$, tint: t-in,
       stroke: (dash: "dashed", thickness: 0.8pt, paint: t-in.darken(18%)),
       name: <noise>),

  // learnable per-query-column inverse temperature
  step((3, 0.5), stack(
    dir: ttb, spacing: 4pt,
    align(center)[$beta_c$-softmax$""_j$],
    align(center, pi-matrix),
  ), <soft>),

  // aggregation and output projection
  step((4, 1.25), $sum_j pi_(c j) bold(v)_j$, <agg>),
  step((5, 1.25), $bold(W)^O$, <wo>),
  step((6, 1.25), $"LN"$, <ln>),
  blob((7, 1.25), $bold(x)^l_t$, tint: t-cell, name: <xout>),

  // all edges routed orthogonally: sources fan out, then merge into the scores
  edge(<src-q>, <pq>, "-|>", label: $bold(Q)$, label-side: left),
  edge(<src-kv>, (0, 1), <pk>, "-|>", label: $bold(K)$, label-pos: 0.88,
       label-side: left),
  edge(<src-kv>, <pv>, "-|>", label: $bold(V)$, label-side: right),
  edge(<pq>, (1.62, 0), (1.62, 0.5), <scores>, "-|>", label: $bold(q)$,
       label-pos: 0.25, label-side: left),
  edge(<pk>, (1.62, 1), (1.62, 0.5), <scores>, "-|>", label: $bold(k)$,
       label-pos: 0.25, label-side: right),
  edge(<noise>, <scores>, "--|>", label: [train only], label-pos: 0.42,
       label-side: left),
  edge(<scores>, <soft>, "-|>"),
  edge(<soft>, (3.9, 0.5), <agg>, "-|>", label: $pi^l_t$, label-pos: 0.3,
       label-side: left),
  edge(<pv>, (3.9, 2), <agg>, "-|>", label: $bold(v)$, label-pos: 0.3,
       label-side: right),
  edge(<agg>, <wo>, "-|>"),
  edge(<wo>, <ln>, "-|>"),
  edge(<ln>, <xout>, "-|>"),
)

// ─────────────────────────────────────────────────────────────────────────────
// COMPOSITION
// ─────────────────────────────────────────────────────────────────────────────

#stack(
  dir: ttb, spacing: 9pt,
  grid(
    columns: (auto, auto),
    column-gutter: 12mm,
    row-gutter: 6pt,
    align: (horizon + center, horizon + center),

  ),
  align(center, panel-c),
  align(center)[(c) inside the router of panel (b)],
)
