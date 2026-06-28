// Shared template for knitwork method docs
#let template(title: "", body) = {
  set document(title: title)
  set page(
    paper: "a4",
    margin: (x: 2.5cm, y: 2.5cm),
    numbering: "1",
  )
  set text(font: "New Computer Modern", size: 11pt, lang: "en")
  set heading(numbering: none)
  set par(justify: true, leading: 0.65em)

  // Code blocks
  show raw.where(block: true): it => block(
    fill: luma(240),
    inset: (x: 10pt, y: 8pt),
    radius: 4pt,
    width: 100%,
    text(size: 9pt, it),
  )

  // Inline code
  show raw.where(block: false): it => box(
    fill: luma(240),
    inset: (x: 3pt, y: 1pt),
    radius: 2pt,
    text(size: 9pt, it),
  )

  // Title
  if title != "" {
    align(center)[
      #text(size: 16pt, weight: "bold")[#title]
      #v(0.5em)
      #line(length: 100%, stroke: 0.5pt)
      #v(1em)
    ]
  }

  body
}
