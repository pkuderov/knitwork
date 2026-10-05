// Подключаем fletcher 0.5.7 и фигуры
#import "@preview/fletcher:0.5.7" as fletcher: diagram, node, edge
#import fletcher.shapes: rect, diamond, pill, ellipse

#set page(width: auto, height: auto, margin: 4pt, fill: white)
#set text(font: "TeX Gyre Termes", size: 8pt)
#show math.equation: set text(font: "TeX Gyre Termes Math", size: 8pt)

// ============================================================
// Experiment 7: MEG → Whisper Decoder (Embedding Injection)
// ============================================================
#block(breakable: false)[
  #align(center)[
    #text(size: 10pt, weight: "bold", fill: navy)[
      Experiment 7: MEG-to-Speech Decoding via Embedding Injection
    ]
    #v(4pt)
    #diagram(
      spacing: (0.9em, 0.8em),
      node-stroke: 0.6pt,
      node-fill: none,
      node-inset: 3pt,

      // ──── 1. MEG Signal (слева) ────
      node((-4.6, 0), [MEG Signal\ #text(size: 6.5pt, fill: gray)[30-s segments]],
        name: <meg>, shape: ellipse, fill: blue.lighten(90%), stroke: blue,
        width: 2.1cm, height: 0.9cm),

      // ──── 2. MEG Encoder ────
      node((-2.8, 0), [MEG Encoder\ #text(size: 6.5pt, fill: gray)[predicts L4 embeddings]],
        name: <encoder>,
        fill: gradient.linear(blue.lighten(60%), blue.lighten(30%)),
        stroke: blue.darken(20%), width: 2.1cm, height: 1.0cm),

      // ──── 3. Whisper Encoder ────
      node((-0.4, 0),
        [
          #align(center)[
            #text(size: 7pt, fill: white, weight: "bold")[Whisper-medium.en Encoder]
            #v(1pt)

            #box(
              fill: red.lighten(70%),
              stroke: red,
              inset: 2pt,
              radius: 1.5pt,
              width: 98%,
            )[
              #align(center)[
                #text(size: 6pt, fill: red.darken(30%))[Layers 1–3: *skipped*]
              ]
            ]

            #v(0.5pt)



            #v(0.5pt)

            #box(
              fill: gray.lighten(70%),
              stroke: gray,
              inset: 2pt,
              radius: 1.5pt,
              width: 98%,
            )[
              #align(center)[
                #text(size: 6pt)[Layers 5–24: frozen]
              ]
            ]
          ]
        ],
        name: <whisper_enc>,
        fill: gradient.linear(gray.lighten(50%), gray.lighten(20%)),
        stroke: (paint: gray.darken(20%), thickness: 0.8pt, dash: (2.5pt, 1.5pt)),
        width: 2.4cm,
        height: 1.9cm
      ),

      // ──── 4. Whisper Decoder ────
      node((1.8, 0),
        align(center)[
          #text(fill: white, weight: "bold", size: 7.5pt)[Whisper Decoder]
          #v(1pt)

        ],
        name: <decoder>,
        fill: gradient.linear(purple.lighten(60%), purple.lighten(30%)),
        stroke: purple.darken(20%),
        width: 2.1cm, height: 1.05cm),

      // ──── 5. Output Text (справа) ────
      node((3.8, 0),
        [
          #align(center)[
            #text(weight: "bold", size: 7.5pt)[Generated Text]
            #v(1.5pt)
            #box(
              fill: yellow.lighten(90%),
              stroke: yellow.darken(20%),
              inset: 2.5pt,
              radius: 2pt,
              width: 98%
            )[
              #align(left)[
                #text(size: 5.5pt, font: "Courier New", fill: black, tracking: -0.1pt)[
                  the The the The The The. The. the\ 
                  But But the The. The. the the the\ 
                  the. The the. the the But The...
                ]
              ]
            ]
            #v(1pt)
            #text(size: 6pt, fill: red)[WER: 0.987–0.994]
            #text(size: 5.5pt, fill: gray)[47–67 w. / 2,620 ref.]
          ]
        ],
        name: <output>,
        shape: rect, fill: yellow.lighten(95%), stroke: yellow.darken(10%),
        width: 2.6cm, height: 1.85cm),

      // ──── Стрелки (горизонтальный поток) ────
      edge(<meg>, <encoder>, "->", stroke: blue + 0.8pt),

      edge(<encoder>, <whisper_enc>, "->", stroke: orange + 1.2pt,
        label: text(size: 6pt, fill: orange, weight: "bold")[inject at L4]),

      edge(<whisper_enc>, <decoder>, "->", stroke: gray + 0.8pt),

      edge(<decoder>, <output>, "->", stroke: purple + 0.8pt,
        label: text(size: 6pt, fill: purple)[decode]),

      // ──── Аннотации статуса ────
      node((-2.8, 0.95), [#text(fill: white, size: 6pt)[trainable]],
        fill: blue, inset: 2pt, shape: pill, stroke: none),

      node((-0.4, -0.95), [#text(fill: white, size: 6pt)[frozen]],
        fill: gray, inset: 2pt, shape: pill, stroke: none),

      node((1.8, -0.95), [#text(fill: white, size: 6pt)[frozen]],
        fill: purple.darken(20%), inset: 2pt, shape: pill, stroke: none),

      // ──── Боковая аннотация: No Audio ────
      node((1.8, 0.95),
        align(center)[
          #text(size: 6.5pt, fill: red, weight: "bold")[⊗ No Audio]
          #text(size: 5.5pt, fill: gray)[MEG-derived only]
        ],
        fill: red.lighten(95%), stroke: red, inset: 2pt, shape: pill),
    )
  ]
]