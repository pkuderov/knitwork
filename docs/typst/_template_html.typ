// HTML-compatible template (no page config)
#let template(title: "", body) = {
  set text(size: 11pt, lang: "en")
  set heading(numbering: none)
  set par(justify: false, leading: 0.65em)

  show raw.where(block: true): it => block(
    fill: luma(240),
    inset: (x: 10pt, y: 8pt),
    radius: 4pt,
    width: 100%,
    text(size: 9pt, it),
  )

  show raw.where(block: false): it => box(
    fill: luma(240),
    inset: (x: 3pt, y: 1pt),
    radius: 2pt,
    text(size: 9pt, it),
  )

  if title != "" {
    text(size: 16pt, weight: "bold")[#title]
    linebreak()
    linebreak()
  }

  body
}
