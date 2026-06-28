#import "@preview/shiroa:0.2.0": *

#show: book

#book-meta(
  title: "knitwork — Grid RNN methods",
  description: "Research documentation for Grid RNN experiments",
  repository: "https://github.com/pkuderov/knitwork",
  authors: ("Vladimir",),
  language: "en",
  summary: [
    #prefix-chapter("README.typ")[Overview]

    = Experiments
    - #chapter("experiments/sdq.typ")[SDQ]
    - #chapter("experiments/text.typ")[Text Modeling]
    - #chapter("experiments/treasure.typ")[TreasureHunt]
    - #chapter("experiments/mikasa.typ")[MIKASA / POPGym]

    = Basics
    - #chapter("methods/gru.typ")[GRU baseline]
    - #chapter("methods/grnn.typ")[GridRNN]
    - #chapter("methods/grnn_err.typ")[GridRNN-err]

    = Cell Modifications
    - #chapter("methods/grnn2.typ")[GridRNN2 + VAE]
    - #chapter("methods/hgrnn.typ")[HGRNN]
    - #chapter("methods/hgrn_grnn.typ")[HGRN-GridRNN]
    - #chapter("methods/lru.typ")[LRU]
    - #chapter("methods/grnn_lru.typ")[GridLRU]
    - #chapter("methods/hgrnn_lru.typ")[HopfieldGridLRU]

    = Memory and Associations
    - #chapter("methods/grnn_delta.typ")[DeltaGrid]
    - #chapter("methods/engram_grnn.typ")[Engram]
    - #chapter("methods/grnn_fw.typ")[Fast Weights]
    - #chapter("methods/grnn_reservoir.typ")[Reservoir]
    - #chapter("methods/grnn_prec_delta.typ")[PrecDelta]
    - #chapter("methods/grnn_ema_mem.typ")[EmaMem]

    = Iterative Methods
    - #chapter("methods/grnn_eq.typ")[Equilibrium]
    - #chapter("methods/grnn_eq1.typ")[Equilibrium v2]

    = Regularization and Loss
    - #chapter("methods/diversity.typ")[DiversityLoss]
    - #chapter("methods/grnn_loss.typ")[GridRNN-Loss]
    - #chapter("methods/grnn_adv_loss.typ")[GridRNN-AdvLoss]
    - #chapter("methods/grnn_disc.typ")[GridRNN-Disc]

    = Harmonic
    - #chapter("methods/grnn_harmonic.typ")[HarmonicGridRNN]

    = Fusion
    - #chapter("methods/fusion_cells.typ")[Fusion cells]
    - #chapter("methods/grnn_fusion.typ")[GridRNN-Fusion]

    = Infrastructure
    - #chapter("methods/curriculum_scheduling.typ")[Curriculum Scheduling]
    - #chapter("methods/mikasa_exp.typ")[MIKASA experiment]
  ]
)

// re-export page template
#import "/templates/page.typ": project
#let book-page = project
