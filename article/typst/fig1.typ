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
// BUILDING BLOCKS (full‑size versions for (a) and (b), same as original)
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
// PANEL (a) — depth is feed-forward within a step; recurrence closes via the top
// (widened even more)
// ─────────────────────────────────────────────────────────────────────────────

#let panel-a = diagram(
  spacing: (35mm, 7.5mm),               // increased horizontal spacing
  node-shape: rect,
  edge-stroke: 0.8pt + ink,
  edge-corner-radius: 4pt,
  mark-scale: 70%,
  label-size: 9pt,
  label-sep: 3pt,

  // readout: column 0 of the top layer only
  blob((0, 0), $bold(o)_(t-1)$, tint: t-out, shape: hexagon, name: <o0>),
  blob((1, 0), $bold(o)_t$, tint: t-out, shape: hexagon, name: <o1>),

  // top layer l = L-1; column 0 is the hardwired readout column
  blob((0, 1), blk($bold(H)^(L-1)_(t-1)$, 4, t-state, first: t-out), name: <a0>),
  blob((1, 1), blk($bold(H)^(L-1)_t$, 4, t-state, first: t-out), name: <a1>),

  // bottom layer l = 0
  blob((0, 2), blk($bold(H)^0_(t-1)$, 4, t-state), name: <b0>),
  blob((1, 2), blk($bold(H)^0_t$, 4, t-state), name: <b1>),

  // external input
  blob((0, 3), $bold(e)_(t-1)$, tint: t-in, shape: house.with(angle: 25deg),
       name: <e0>),
  blob((1, 3), $bold(e)_t$, tint: t-in, shape: house.with(angle: 25deg),
       name: <e1>),

  // input -> bottom layer
  edge(<e0>, <b0>, "-|>"),
  edge(<e1>, <b1>, "-|>"),

  // bottom-up within one timestep: M^l_t = H^(l-1)_t
  edge(<b0>, <a0>, "-|>"),
  edge(<b1>, <a1>, "-|>", label: $bold(M)^l_t$, label-side: right),

  // readout from column 0 of the top layer
  edge(<a0>, <o0>, "-|>"),
  edge(<a1>, <o1>, "-|>", label: $bold(h)^(L-1,0)_t$, label-side: right),

  // per-column GRU state carried across time (upper lane)
  edge(<a0>, <a1>, "-|>", shift: 7pt, label: $bold(h)^(l,c)_(t-1)$),
  edge(<b0>, <b1>, "-|>", shift: 7pt),

  // delayed complete top layer joins the next step's first-layer bank
  edge(<a0>, (0.5, 1), (0.5, 2), <b1>, "--|>",
       label: $z^(-1)$, label-pos: 0.55, label-side: left),

  // continuation: the same delayed route arrives from t-2 and leaves to t+1
  node((-0.42, 2), $dots.h$, stroke: none, name: <past>),
  node((1.42, 2), $dots.h$, stroke: none, name: <future>),
  edge(<past>, <b0>, "--|>"),
  edge(<a1>, (1.42, 1), <future>, "--|>"),
)

// ─────────────────────────────────────────────────────────────────────────────
// PANEL (b) — inside one layer: one shared router, C independent GRU cells
// (widened even more)
// ─────────────────────────────────────────────────────────────────────────────

#let panel-b = diagram(
  spacing: (35mm, 8mm),
  node-shape: rect,
  edge-stroke: 0.8pt + ink,
  edge-corner-radius: 4pt,
  mark-scale: 70%,
  label-size: 9pt,
  label-sep: 3pt,

  // updated states
  blob((1, 0), blk($bold(H)^l_t$, 4, t-state), name: <new>),

  // C independently parameterized GRU cells
  blob((1, 1), blk($"GRU"_(l,c)$, 4, t-cell), tint: t-cell, name: <gru>),

  // router shared by all columns; expanded in panel (c)
  blob((1, 2), stack(
    dir: ttb, spacing: 4pt,
    align(center)[routing $pi^l_t$ #text(size: 9pt)[(c)]],
    align(center, pi-matrix),
  ), tint: t-route, name: <router>),

  // message bank: keys and values
  blob((1, 3), blk($bold(M)^l_t$, 4, t-state, extra: (t-in,)), name: <bank>),

  // previous states: queries and per-column recurrent states
  blob((0, 2), blk($bold(H)^l_(t-1)$, 4, t-state), name: <prev>),

  edge(<bank>, <router>, "-|>", label: $bold(K),bold(V)$, label-side: right),
  edge(<prev>, <router>, "-|>", label: $bold(Q)$),
  edge(<router>, <gru>, "-|>", label: $bold(x)^l_t$, label-side: right),
  edge(<prev>, (0, 1), <gru>, "-|>", label: $bold(h)^(l,c)_(t-1)$,
       label-pos: 0.62, label-side: left),
  edge(<gru>, <new>, "-|>"),
)

#stack(
  dir: ttb, spacing: 12pt,
  align(center)[
    #panel-a
    #align(center)[(a) grid over time and depth]
  ],
  align(center)[
    #panel-b
    #align(center)[(b) one layer at timestep $t$]
  ]

)