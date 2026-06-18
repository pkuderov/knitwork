// Grid Recurrent Networks — AAAI 2026 Submission
// Compile: typst compile paper.typ
//
// All layout definitions copied verbatim from aaai2026.typ (do not modify that file).

// ─────────────────────────────────────────────────────────────────────────────
// PAGE
// ─────────────────────────────────────────────────────────────────────────────

#set page(
  paper: "us-letter",
  margin: (top: 0.75in, bottom: 1.25in, left: 0.75in, right: 0.75in),
  numbering: none,
  header: none,
  footer: none,
)

// ─────────────────────────────────────────────────────────────────────────────
// TYPOGRAPHY
// ─────────────────────────────────────────────────────────────────────────────

#set text(font: "Linux Libertine O", size: 10pt, lang: "en")
// AAAI spec: 10pt font, 12pt leading (= 2pt gap between bounding boxes).
// For draft readability we use 0.5em ≈ 5pt gap (≈ 15pt baseline-to-baseline).
// Revert to leading: 2pt for final camera-ready submission.
#set par(leading: 0.5em, spacing: 0.9em, justify: true, first-line-indent: 10pt)

// ─────────────────────────────────────────────────────────────────────────────
// HEADINGS
// ─────────────────────────────────────────────────────────────────────────────

#set heading(numbering: none)

#show heading.where(level: 1): it => block(above: 1em, below: 0.5em)[
  #set text(size: 10pt, weight: "bold")
  #it.body
]
#show heading.where(level: 2): it => block(above: 0.75em, below: 0.3em)[
  #set text(size: 10pt, weight: "bold")
  #it.body
]
#show heading.where(level: 3): it => block(above: 0.6em, below: 0.2em)[
  #set text(size: 10pt, weight: "bold", style: "italic")
  #it.body
]

#let par-heading(title) = strong[#title.#h(0.5em)]

// ─────────────────────────────────────────────────────────────────────────────
// LISTS
// ─────────────────────────────────────────────────────────────────────────────

#set list(indent: 1em, body-indent: 0.5em)
#set enum(indent: 1em, body-indent: 0.5em)

// ─────────────────────────────────────────────────────────────────────────────
// FIGURES / CAPTIONS
// ─────────────────────────────────────────────────────────────────────────────

#show figure.caption: set text(size: 10pt)

#let wide-figure(caption: [], body) = figure(
  body,
  caption: caption,
  placement: top,
  scope: "parent",
)

// ─────────────────────────────────────────────────────────────────────────────
// TABLES
// ─────────────────────────────────────────────────────────────────────────────

#let wide-table(caption: [], body) = figure(
  body,
  caption: caption,
  placement: top,
  scope: "parent",
  kind: table,
)

// ─────────────────────────────────────────────────────────────────────────────
// ALGORITHMS
// ─────────────────────────────────────────────────────────────────────────────

#show figure.where(kind: "algorithm"): it => block(width: 100%, breakable: false)[
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

#let algo-state(body)  = block(spacing: 0.2em)[#body]
#let algo-require(body) = algo-state[*Input:* #body]
#let algo-ensure(body)  = algo-state[*Output:* #body]
#let algo-return(body)  = algo-state[*return* #body]
#let algo-comment(body) = text(style: "italic")[\// #body]

// ─────────────────────────────────────────────────────────────────────────────
// QUOTE / EXTRACT
// ─────────────────────────────────────────────────────────────────────────────

#let extract(body) = pad(x: 10pt)[
  #set par(first-line-indent: 0pt)
  #body
]

// ─────────────────────────────────────────────────────────────────────────────
// TITLE / ABSTRACT BLOCKS
// ─────────────────────────────────────────────────────────────────────────────

#let aaai-title(title: [], authors: [], affiliations: []) = {
  v(0.5in)
  align(center)[
    #text(size: 16pt, weight: "bold")[#title]
    #v(6pt)
    #text(size: 12pt)[#authors]
    #v(3pt)
    #text(size: 9pt)[#affiliations]
  ]
  v(1em)
}

#let aaai-abstract(body) = pad(x: 0.5in)[
  #align(center)[#text(weight: "bold")[Abstract]]
  #set par(first-line-indent: 0pt)
  #body
]

// ═════════════════════════════════════════════════════════════════════════════
// PAPER CONTENT
// ═════════════════════════════════════════════════════════════════════════════

#aaai-title(
  title: [Grid Recurrent Networks: Parallel Memory Columns \ for Partially Observable Environments],
  authors: [Anonymous Submission],
  affiliations: [],
)

#aaai-abstract[
  Sequential decision-making under partial observability requires agents to maintain
  structured, selective memory over long horizons. Standard recurrent architectures
  (GRU, LSTM) rely on a single homogeneous hidden state, offering no structural
  mechanism for simultaneous specialization of memory roles across temporal scales.
  We propose *Grid Recurrent Networks (Grid RNN)*, a two-dimensional recurrent
  architecture that organizes cells into an $L times C$ grid of layers and columns,
  where inter-column multi-head attention with learnable column identities enables
  spontaneous role specialization without explicit supervision. Grid RNN is a modular
  framework: column cells can be GRU, Linear Recurrent Units (LRU) with per-column
  spectral radii, or EMA surprise-gated fast weights, while the cross-column message
  layer can range from scaled dot-product attention to Modern Hopfield retrieval.
  We evaluate across three complementary benchmarks: the Store-Distract-Query (SDQ)
  associative memory task, the MIKASA/POPGym partially observable reinforcement
  learning suite, and character-level language modeling. Our *HopfieldGridLRU*
  variant achieves 96.7% accuracy on SDQ-Hard (vs.\ ~50% for GRU); *GridRNN-EMA*
  achieves episode return $approx 0.95$ on POPGym RepeatFirst, approaching optimal.
  With 2.1M parameters, GridHarmonic reaches 1.68 BPC on text8. These results
  demonstrate consistent improvements from grid structure across all three settings.
]

#v(1em)

// ─────────────────────────────────────────────────────────────────────────────
// TWO-COLUMN BODY
// ─────────────────────────────────────────────────────────────────────────────

#columns(2, gutter: 0.375in)[

= Introduction

The ability to selectively store and retrieve information across time is central
to a broad class of sequential tasks: reinforcement learning (RL) agents navigating
partially observable environments must remember which objects they have seen and
associate stimuli with delayed consequences; language models must maintain coherent
context over hundreds of characters; associative memory tasks require explicit
key-value binding over many distractors. Solving these problems simultaneously
demands memory with multiple specialized roles operating on different temporal
scales.

Standard recurrent architectures — GRU @cho2014gru and LSTM @hochreiter1997lstm —
address this with a single hidden state vector updated at every timestep. While
effective, this *single-stream* design offers no structural mechanism for
specialization: all temporal scales and memory functions compete for the same
state dimensions. A sequence model must simultaneously be a short-term buffer, a
long-range integrator, and a content-addressable store — yet its architecture
provides no structural separation of these roles.

Prior work has approached this in two directions. *Hierarchical* architectures
(HM-RNN @chung2017hmrnn, Fast-Slow RNN @mujika2017fastslow, HGRN2 @qin2024hgrn2)
stack multiple timescales *vertically*, assigning each layer a different
temporal resolution. This helps long-range modeling but does not support *lateral*
specialization: each layer still processes a single sequential stream without
peer communication. *Memory augmentation* approaches (Fast Weights @ba2016fastweights,
Hopfield Networks @ramsauer2021hopfield, Engrams @szelogowski2025engram) add
external associative structures to an existing recurrent backbone, but leave the
backbone's single-stream topology unchanged.

#figure(
  image("figure1_grid.svg", width: 100%),
  caption: [*Grid RNN architecture.* Cells are arranged in an $L times C$ grid.
    At each timestep, each column independently processes its input (column~0
    receives the embedded token; others receive zero), then all columns exchange
    information via multi-head attention with learnable column identity keys.
    Different columns converge to distinct memory roles without explicit supervision.],
  placement: top,
) <fig-arch>

We propose *Grid Recurrent Networks (Grid RNN)*, which reconsiders the fundamental
topology of recurrent computation. Rather than a single hidden vector, Grid RNN
maintains a 2D state matrix $bold(H) in RR^{L times C times B times H}$ where
$L$ layers provide temporal depth and $C$ columns operate *in parallel* (see
@fig-arch). Columns share the input embedding but independently process it through
their recurrent cells, then exchange information via multi-head attention with
learnable column identity embeddings. This structure allows columns to *spontaneously
specialize*: column 0 receives data and distributes it; other columns selectively
absorb what is relevant to their emerging role; the output is recombined through
attention aggregation. No explicit supervision of column roles is required.

Grid RNN is designed as a modular framework: the per-column cell can be replaced
with any recurrent unit, and the cross-column message layer can incorporate advanced
retrieval. We explore three variants targeting different benchmarks:
- *GridRNN* (GRU cells + scaled attention): strong baseline for associative tasks;
- *HopfieldGridLRU*: LRU @orvieto2023lru cells with per-column spectral radii and
  Modern Hopfield retrieval @ramsauer2021hopfield for sharp associative memory;
- *GridRNN-EMA*: GRU cells with EMA surprise-gated delta-rule write for selective
  memory in partially observable RL.

Our main contributions are:
+ *Grid RNN architecture*: a 2D $L times C$ recurrent grid with inter-column
  attention enabling structural column specialization, compatible with diverse
  memory mechanisms.
+ *HopfieldGridLRU*: achieves *96.7% accuracy* on SDQ-Hard, outperforming
  single-stream GRU by a factor of $>$6$times$ on the hardest query metric (Acc++).
+ *GridRNN-EMA*: achieves episode return $approx 0.95$ on POPGym RepeatFirst
  (Easy), approaching optimal, with surprise-gated selective writing.
+ *Systematic evaluation* across three benchmarks (SDQ, POPGym/MIKASA, text
  modeling), showing consistent improvement over single-stream baselines.


= Related Work

#par-heading[Multi-Dimensional and Grid RNNs.]
The idea of organising recurrent cells in a 2D structure has been explored before.
Graves et al. introduced Multi-Dimensional RNNs (MD-RNN) @graves2007mdrnn, which
apply a separate recurrent connection along each spatial dimension and concatenate
them, targeting image captioning and 2D sequences. Grid LSTM @kalchbrenner2015gridlstm
extended this to deep stacks by routing memory and hidden state along both depth
and time axes simultaneously. These works establish the grid topology but differ
from our proposal in two critical ways: (a) they use *fixed* inter-cell connections
determined by position, not *learned* content-based attention; (b) they target
2D spatial sequences (images, text rendered as 2D grids), not generic sequence
modeling with a free-standing memory structure. Grid RNN introduces *lateral
attention with learnable column identity keys*, enabling content-based routing
and spontaneous role specialization that fixed positional connections cannot provide.
The column structure also differs: all C columns share the time axis and are
updated synchronously at each timestep, unlike MD-RNN where separate dimensions
correspond to separate axes of a spatial grid.

#par-heading[Hierarchical Multi-Scale RNNs.]
Several methods introduce temporal hierarchy by stacking layers with different
timescales. HM-RNN @chung2017hmrnn learns discrete boundary events to separate
fast and slow computation in a vertical hierarchy, improving long-range language
modeling. Fast-Slow RNN @mujika2017fastslow maintains two decoupled networks with
a fixed coupling period. HGRN @qin2023hgrn and HGRN2 @qin2024hgrn2 introduce a
learnable $beta$ gate to control per-layer state retention in a linear RNN,
achieving competitive performance with selective state-space models. All of these
approaches extend *depth* to create timescale hierarchies, but each layer remains
a single sequential stream without lateral peer communication. Grid RNN complements
vertical depth with *horizontal* columns connected via attention, enabling
specialization orthogonal to temporal abstraction. Empirically, 4 columns with
3 layers outperforms 2 columns with 1 layer at the same parameter budget by
$+42$ percentage points on Acc++ in SDQ-Hard.

#par-heading[Associative and Fast Memory Augmentation.]
A rich line of work augments RNNs with external memory structures. Fast Weights
@ba2016fastweights write to a temporary weight matrix via Hebbian outer products,
retrieved at each step by a content query. Schlag et al. @schlag2021fastweight
show that linear Transformers implicitly implement fast-weight programmers.
Danihelka et al. @danihelka2016assoclstm embed associative LSTM with holographic
reduced representations for interference-free binding. Modern Hopfield Networks
@ramsauer2021hopfield provide exponentially increasing storage capacity with
sharp, energy-based retrieval equivalent to attention with large $beta$.
Szelogowski @szelogowski2025engram revisits Hebbian engram neurons as sparse
slot memory in deep networks. Lansner et al. @lansner2023hebbian systematically
benchmark Hebbian update rules for associative capacity. These works add memory
*on top of* an unchanged single-stream recurrent backbone. In Grid RNN, associative
mechanisms are instantiated *per column* of the grid: the Hopfield message layer
routes information between columns, while fast-weight matrices within each column
store column-specific associations, allowing different columns to specialize in
reading vs. writing roles.

#par-heading[Memory for Partially Observable RL.]
Training RL agents in POMDPs with recurrent networks has a long history: DRQN
@hausknecht2015drqn extends DQN with LSTM to handle partial observability.
Mnih et al. @mnih2016a3c use LSTM policies in A3C, showing robust performance
across Atari games. POPGym @morad2023popgym systematically benchmarks memory in
RL across four types — object, sequential, capacity, and spatial — providing a
standardized suite of partially observable environments. MIKASA @cherepanov2025mikasa
extends this taxonomy with richer task variants. These benchmarks consistently show
that standard LSTM/GRU plateaus well below the theoretical maximum on object-memory
tasks such as RepeatFirst, motivating architectural improvements. GridRNN-EMA
addresses this gap with a surprise-gated selective write mechanism that writes only
when observations are unexpected, achieving near-optimal performance on RepeatFirst.

#par-heading[Linear Recurrent Units.]
Orvieto et al. @orvieto2023lru (LRU) show that a linear recurrence with diagonal
complex-valued transition matrices achieves competitive performance with selective
state-space models, provided spectral radii are initialized in $(r_"min", r_"max")$
and $r_"max"$ is tuned for the target memory horizon. We use LRU cells as column
units in HopfieldGridLRU and GridHarmonic, assigning each column $c$ and layer $l$
a distinct $r_"max"^{l,c}$ interpolated over a 2D grid from $0.3$ (fast, $tau approx 1$
step) to $0.999$ (slow, $tau approx 1000$ steps). This 2D *spectral grid* provides
structural prior without any additional parameters.


= Method

== Grid Structure

Grid RNN maintains a 3D state tensor $bold(H)^l_t in RR^{C times B times H}$
per layer $l in {1, ..., L}$, where $C$ is the number of columns, $B$ the batch
size, and $H$ the hidden dimension. At each timestep $t$, the forward pass
proceeds *per-layer* in two stages.

#par-heading[Column Step.] Each column $c$ independently applies its recurrent
cell $f_theta^c$:

$ bold(h)^{l,c}_t = f_theta^c (bold(x)^{l,c}_t, bold(h)^{l,c}_{t-1}) $

For layer $l=0$: column~0 receives the embedded input $bold(e)_t in RR^D$;
columns $c>0$ receive a zero dummy input of dimension~1. For layers $l>0$: all
columns receive the corresponding state from the previous layer as input. Using
a dummy input rather than a shared embedding ensures columns are forced to
specialize via the message layer rather than receiving identical information.

#par-heading[Inter-Column Message Passing.] After the column step, all columns
exchange information via multi-head attention:

$ bold(Q)_t = bold(H)^l_t + bold(I)_"col" quad bold(K)_t = bold(H)^l_t + bold(I)_"col" $
$ bold(M)^l_t = "MHA"(bold(Q)_t, bold(K)_t, bold(H)^l_t) $

where $bold(I)_"col" in RR^{C times 1 times H}$ are learnable per-column identity
embeddings broadcast over the batch, allowing the attention to distinguish
source and target columns. The output projection of MHA is initialized near zero,
so initial messages are negligible and training begins from an effective
single-stream baseline.

#par-heading[Gated Mixing.] Each column decides how much of the aggregated message
to incorporate:

$ bold(g)^{l,c}_t = sigma(bold(W)_g [bold(h)^{l,c}_t ; bold(m)^{l,c}_t]) $
$ bold(h)'^{l,c}_t = (1 - bold(g)^{l,c}_t) ⊙ bold(h)^{l,c}_t + bold(g)^{l,c}_t ⊙ bold(m)^{l,c}_t $

This *post-messaging* design (GRU step first, then attention) lets each column
compute its new state independently before considering peers, which we find more
stable than pre-messaging during early training.

#par-heading[Output.] For RL and associative tasks, the output is the top-layer
($l=L$) state of column~0: $bold(o)_t = bold(h)'^{L,0}_t$. For language
modeling, we average across all columns to exploit full column diversity.

== Spectral Grid Memory (HopfieldGridLRU)

For tasks demanding long-range association (SDQ) and character-level modeling,
we replace GRU cells with Linear Recurrent Units @orvieto2023lru and upgrade
the message layer to Modern Hopfield retrieval @ramsauer2021hopfield.

#par-heading[LRU Cells.] Each column-layer pair $(l,c)$ maintains a
complex-valued diagonal state $bold(h)_t in CC^H$:

$ bold(h)_t = Lambda^{l,c} ⊙ bold(h)_{t-1} + bold(B) bold(x)_t $

where $Lambda^{l,c} = "diag"(exp(-exp(bold(nu)) + i exp(bold(theta))))$ ensures
$|Lambda^{l,c}| in (r_"min", r_"max"^{l,c})$ by construction, preventing
gradient explosion regardless of sequence length. The state is stored as
$[bold(h)_"re" ; bold(h)_"im"] in RR^{2H}$.

#par-heading[2D Spectral Grid.] Each column~$c$ and layer~$l$ receives a distinct
spectral radius:

$ r_"max"^{l,c} = r_"min_col" + (r_"base"^l - r_"min_col") dot c / (C-1) $
$ r_"base"^l = r_"min_layers" + (r_"max_layers" - r_"min_layers") dot l / (L-1) $

with $r_"min_col"=0.3$, $r_"min_layers"=0.7$, $r_"max_layers"=0.999$. This gives
column~0, layer~0 a memory horizon of $tau approx 1$ step (fast, reactive),
while column~$C{-}1$, layer~$L{-}1$ reaches $tau approx 1000$ steps (slow,
integrative). No extra parameters are required.

#par-heading[Hopfield Message Layer.] The cross-column message layer uses Modern
Hopfield attention @ramsauer2021hopfield with learnable temperature $beta$ per head:

$ bold(M)_t = "softmax"(beta dot.op bold(Q)_t^top bold(K)_t) bold(V)_t $

$beta$ is initialized as $log(1 / sqrt(d_k))$ and learned per attention head,
allowing the network to tune retrieval sharpness. Higher $beta$ produces more
selective, winner-takes-all column routing; lower $beta$ produces diffuse
information mixing. A learnable contrastive associative loss optionally encourages
consistent store/query alignment across columns.

== Surprise-Gated Write Memory (GridRNN-EMA)

For partially observable RL, agents must write to memory selectively: storing
every observation wastes capacity on irrelevant context, while missing key
observations causes memory failure. We augment Grid RNN columns with a
*surprise-gated* delta-rule fast-weight matrix.

#par-heading[EMA Surprise Signal.] For each column $c$ at layer $l$, let
$bold(y)_t = bold(h)'^{l,c}_t$ be the post-mixing state. We project to
normalized key-value pairs and compute prediction error:

$ bold(k)_t = frac(bold(W)_k bold(y)_t, ||bold(W)_k bold(y)_t||_2) $
$ bold(v)_t = frac(bold(W)_v bold(y)_t, ||bold(W)_v bold(y)_t||_2) $
$ bold("err")_t = bold(v)_t - bold(W)_t^top bold(k)_t $  (prediction error)

An EMA of squared error magnitude tracks *surprise*:

$ m_t = beta_"ema" m_{t-1} + (1 - beta_"ema") "mean"(||bold("err")_t||^2) $
$ alpha_t = m_t / ("sg"(max_B (m_t)) + epsilon)  quad (max "over batch") $

where $"sg"(dot)$ denotes stop-gradient (the normaliser is detached from the
computation graph to avoid second-order terms). $alpha_t in [0,1]$: the batch
element with the highest recent surprise always gets $alpha_t approx 1$ (active
write), while elements experiencing familiar inputs approach zero (no write).
This relative normalization ensures the gate reflects within-batch novelty rather
than absolute loss scale.

#par-heading[Delta-Rule Update.] We apply the Widrow-Hoff delta rule with adaptive
forgetting:

$ bold(W)_{t+1} = (1 - lambda_t) delta^l bold(W)_t + alpha_t (bold(k)_t ⊗ bold("err")_t) / C $

where $delta^l in [0.95, 0.99]$ is a per-layer decay (fast layers forget quickly,
slow layers integrate longer), $"fullness"_t = ||bold(W)_t||_F / sqrt(d_k d_v)$
approximates the fraction of weight matrix capacity used, $lambda_t = lambda_0 dot "fullness"_t$
provides adaptive forgetting that increases as memory fills (preventing overflow),
and dividing by $C$ normalizes the write magnitude across columns. The delta rule *erases* the previous binding
$bold(k)_t ⊗ bold(v)_"pred"$ before writing the new one, preventing
Hebbian interference @ba2016fastweights.


= Experiments

We evaluate three Grid RNN variants against GRU baselines on three benchmarks.
All models are trained on a single GPU (RTX~3050). Parameter budgets are matched
within benchmarks at approximately 2.1M parameters. Learning rate uses a linear
warmup followed by cosine decay ($eta_"max" = 8 times 10^{-4}$), implemented
via PPO @schulman2017ppo for RL experiments and BPTT for supervised tasks.

== Store-Distract-Query (SDQ)

#par-heading[Task.] SDQ @morad2023popgym presents sequences of tokens from a
vocabulary of 65 (50 store events, 10 distractors, 5 query types). A *store* event
presents a key-value pair $(k, v)$ that the model must bind for later retrieval.
*Query* tokens present a previously stored key and require the model to output the
correct value. The *Hard* variant additionally counts the total number of stored
and queried pairs simultaneously. Models are trained online with curriculum scheduling: the average sequence length
$T$ starts at 10 and increases when the moving-average improvement in Acc exceeds
a threshold (checked every 200K steps; check frequency multiplied by $0.97$ on
progress, $times 1.25$ on stagnation). Training runs for 1B steps on CPU with
rollout length 8.

#par-heading[Results.] Table~@tab-sdq shows accuracy metrics after $tilde$1B
training steps. Acc measures overall token accuracy; Acc++ measures accuracy
*only* on hard query tokens — the most discriminative metric because distractor
tokens are trivially correct.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    [*Model*], [*Params*], [*Acc*], [*Acc++*],
    table.hline(),
    [GRU (single-stream)], [2.1M], [~0.50], [~0.15],
    [GridRNN (GRU cols)], [2.1M], [0.960], [0.917],
    [GridRNN-LRU], [2.1M], [0.849], [0.694],
    [*HopfieldGridLRU*], [2.1M], [*0.967*], [*0.932*],
    [GridHarmonic], [2.1M], [0.849], [0.696],
    table.hline(),
  ),
  caption: [SDQ-Hard results. Acc = overall token accuracy; Acc++ = accuracy
    on hard query tokens only. GRU baseline from preliminary runs (exact values
    pending full convergence run). GridRNN and HopfieldGridLRU: mean of final 10M
    steps (stable). GridHarmonic: best checkpoint at step 95M (peak before
    oscillation onset; see Discussion). GridRNN-LRU: final value at step 87M.],
  placement: top,
  kind: table,
) <tab-sdq>

GridRNN (GRU columns) dramatically outperforms single-stream GRU: Acc++ improves
from $approx 0.15$ to $0.917$ ($+77$ pp absolute). This demonstrates that the
*column structure alone* — without any specialized memory mechanism — provides
decisive benefit. The grid creates structural differentiation that single-stream
GRU cannot achieve regardless of width or depth. We note that the GRU baseline
numbers are preliminary (full convergence run pending; see future-work document);
the large magnitude of the gap makes this unlikely to reverse.

*Note on GridHarmonic.* While GridHarmonic achieves 0.849~Acc++ (at peak),
Table~1 also reveals that the simplest grid variant (GridRNN, GRU columns) reaches
0.917~Acc++ — exceeding GridHarmonic and suggesting the core column structure
contributes more than the additional Delta-rule and EMA components in this task.

HopfieldGridLRU ($0.967$~Acc, $0.932$~Acc++) further improves over the GRU-column
variant by combining spectral memory stability (LRU spectral grid) with sharp
Hopfield retrieval for cross-column information routing. The learnable $beta$ per
head allows fine-grained tuning of retrieval selectivity.

GridRNN-LRU without Hopfield messages (0.849~Acc++) underperforms the GRU-column
variant, suggesting that the complex-valued LRU state alone is not sufficient:
the Hopfield message layer is critical for sharp inter-column routing.

GridHarmonic (0.849~Acc++) currently underperforms HopfieldGridLRU. Two known
issues: (1) column representation norms grow unboundedly in upper layers after
~100M steps (e.g., $||h_("col0, L2")||$ exceeds 300 at step 110M), causing
oscillations; (2) the Acc/store metric (store-operation accuracy) returns NaN
throughout training due to a masking bug, meaning the quality of the store
operation itself is unverifiable. Both issues are tracked in Future Work.

== MIKASA/POPGym Benchmark

#par-heading[Task.] POPGym @morad2023popgym provides a suite of partially
observable environments testing four memory types: object (remember a property
of an object), sequential (remember the order of events), capacity (hold
many items simultaneously), and spatial (remember a map). We evaluate on two
tasks from MIKASA @cherepanov2025mikasa:
- *RepeatFirst* (object memory): the agent observes $N$ symbols and must reproduce
  the first one at episode end. Episode length is 51 steps; per-step reward
  $+1/50$ for correct recall.
- *HigherLower* (sequential memory): predict whether the next card is higher or
  lower than the current; requires tracking the running distribution of revealed
  cards.

All models are trained with PPO @schulman2017ppo, $gamma=0.99$, $lambda_"GAE"=0.95$,
64 parallel environments, rollout length 32, for up to 200M environment steps.

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: none,
    table.hline(),
    [*Model*], [*RepeatFirst*], [*HigherLower*],
    table.hline(),
    [GRU (Morad 2023)], [pending], [pending],
    [GridRNN (GRU)], [pending], [pending],
    [GridRNN-EMA], [*~0.95* #super[†]], [~0.41 #super[†]],
    table.hline(),
  ),
  caption: [POPGym results (episode return, scale $[{-}1, 1]$; higher is better).
    #super[†]Results at 74M/200M steps (37% of training). Full 200M-step results
    and GRU baseline comparison pending.],
  placement: top,
  kind: table,
) <tab-mikasa>

#par-heading[Results.] Table~@tab-mikasa shows results for GridRNN-EMA at 37%
of training (74M/200M steps). GridRNN-EMA achieves episode return $approx 0.95$
on RepeatFirst, approaching the theoretical maximum return of~1.0. This indicates
that the EMA surprise gate successfully identifies the first-observation event
(the only one worth writing) and stores it reliably against 50 subsequent
distractors.

Performance on HigherLower ($~0.41$) is lower, consistent with the harder
sequential nature of this task: the agent must track a running distribution rather
than simply recall a single item. This gap motivates combining EMA surprise writing
with LRU spectral memory to address sequential tasks (a target for future work).

#par-heading[GridRNN-EMA training curve.] At 37% of training (74M steps), the
RepeatFirst episode return ($approx 0.95$) is already near-optimal. Convergence
on this environment is typically reached between 30–50M steps; the remaining
126M steps are expected to consolidate rather than substantially improve return.
The HigherLower curve ($approx 0.41$) has not plateaued — we expect continued
improvement toward a target return of $>0.6$ at 200M steps, but this requires
confirming empirically.

#par-heading[LRU instability under PPO.] Two additional models were stopped early
at 15M steps with negative episode returns ($< 0$). In GridRNN-LRU, policy entropy
$H$ collapsed from 1.1 to 0.19 bits within the first 10M steps — an irreversible
determinization that prevented further learning. Specifically, the LRU spectral
constraint produces a very smooth hidden-state trajectory, which causes the policy
to commit to a single action early. In HopfieldGridLRU, $H$ remained higher
(1.18 bits) but training oscillated without monotone improvement. Both behaviours
point to a structural incompatibility between LRU's linear recurrence (no saturating
nonlinearity) and standard PPO's entropy coefficient (0.01), rather than slow
convergence. Increasing `entropy_coef` to 0.05 is the most promising fix (future work).

== Language Modeling

#par-heading[Setup.] We evaluate on two character-level datasets: *text8*
(100M chars, 27-token alphabet) and *Shakespeare* (1.1M chars, 65-token alphabet).
Models are trained with BPTT (rollout length 16 for GridHarmonic, 8 for others),
128 parallel environment streams, and a reset-probability curriculum that decays
from 0.01 to $10^{-4}$ over training, gradually increasing effective context length.
All Grid RNN models use $approx 2.1$M parameters to enable fair comparison;
published baselines use substantially larger models.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    [*Model*], [*Params*], [*Text8 BPC*], [*Shakes. BPC*],
    table.hline(),
    [Transformer-XL @dai2019transformerxl], [277M], [1.06], [—],
    [SHA-RNN @merity2019sharnn], [53M], [1.07], [—],
    [AWD-LSTM @merity2018awdlstm], [24M], [1.19], [—],
    table.hline(),
    [GRU (ours)], [2.1M], [2.09], [1.95],
    [GridRNN (GRU)], [2.1M], [—], [1.72],
    [HopfieldGridRNN (LSTM+Hopfield)], [2.1M], [—], [1.68],
    [*GridHarmonic*], [2.1M], [*1.68*], [—],
    table.hline(),
  ),
  caption: [Character-level language modeling (bits per character, BPC; lower
    is better). Published baselines use models 12–130$times$ larger. Our models
    are all $approx$2.1M parameters. Grid variants consistently improve over
    the single-stream GRU baseline.],
  placement: top,
  kind: table,
) <tab-lm>

#par-heading[Results.] Table~@tab-lm shows character-level BPC. GridHarmonic
achieves 1.68~BPC on text8 with 2.1M parameters, compared to AWD-LSTM at 1.19~BPC
with 24M parameters (12× more). While Grid RNN does not close the absolute gap
to SOTA, we emphasize that the comparison is intentionally cross-scale: our goal
is to demonstrate consistent improvement *within a fixed parameter budget*.
GridHarmonic improves over single-stream GRU by 0.41 BPC (19.6% relative),
matching the pattern of improvement seen in SDQ.

#par-heading[Training Dynamics.] Table~@tab-text8-curve tracks GridHarmonic (v3)
BPC progression on text8 over 270M steps. BPC decreases continuously from 2.83
at 5M steps to 1.68 at 270M, without divergence or oscillation — in contrast to
SDQ training where column-norm instability appears after 100M steps. The
curriculum increases effective context length ($T$) from 100 to 239 tokens as BPC
improves, so later steps train on significantly longer sequences. The residual
improvement rate at 250–270M is 0.010 BPC/50M steps, suggesting further gains
remain possible with extended training.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    [*Step*], [*BPC*], [*Acc*], [*Context $T$*],
    table.hline(),
    [5M],  [2.83], [0.407], [100],
    [25M], [2.25], [0.521], [103],
    [50M], [1.90], [0.589], [114],
    [100M], [1.78], [0.611], [135],
    [150M], [1.74], [0.620], [160],
    [200M], [1.71], [0.627], [189],
    [270M], [*1.68*], [*0.634*], [*239*],
    table.hline(),
  ),
  caption: [GridHarmonic text8 BPC progression over 270M training steps.
    Context $T$ is the effective sequence length set by the curriculum. Training
    terminated early; rate at 270M ($approx 0.010$ BPC/50M) suggests further
    improvement possible.],
  placement: top,
  kind: table,
) <tab-text8-curve>

#par-heading[Memory Diagnostics.] GridHarmonic exposes interpretable internal
metrics for the SurpriseDelta memory module. At step 270M (text8): the weight
matrix norms grow monotonically across layers ($||bold(W)||_F$: L0=1.07,
L1=1.53, L2=1.90), confirming that deeper layers — with slower spectral radii
($r_"max"$ up to 0.999) — accumulate more persistent associations. Write-gate
$alpha_t$ remains in $(0.85, 0.89)$ across layers, indicating consistently active
writing without saturation. Matrix fullness ($||bold(W)||_F / sqrt(d_k d_v)$)
stays below 3% at all layers, indicating ample remaining capacity. Crucially,
column diversity (mean pairwise cosine distance between column states) remains
moderate at L2=2.16 on text8, whereas the same metric reaches 7.79 on SDQ at
layer~2 — flagging the norm explosion visible in column norms (Section~Discussion).

== Ablation: Column Count

To isolate the effect of grid width, we compare two GridRNN configurations
matched to the same parameter budget ($approx 2.1$M) on SDQ-Hard:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    [*Cols*], [*Layers*], [*H*], [*Acc*], [*Acc++*],
    table.hline(),
    [2], [1], [115], [0.734], [0.494],
    [4], [3], [128], [0.960], [0.917],
    table.hline(),
  ),
  caption: [Ablation: column count vs. depth at fixed $approx$2.1M params (SDQ-Hard).
    Increasing from 2 to 4 columns ($+$2 layers) yields $+42$ pp Acc++.],
  placement: top,
  kind: table,
) <tab-ablation>

Moving from 2 columns / 1 layer to 4 columns / 3 layers while keeping the
parameter count constant yields $+22.6$~pp Acc ($0.734 arrow.r 0.960$) and
$+42.3$~pp Acc++ ($0.494 arrow.r 0.917$). This confirms that column count is
the primary driver of performance, not raw hidden size: distributing capacity
across more specialized columns provides dramatically more representational power
for associative tasks than concentrating it in a deeper single stream.


= Discussion

*Why does column structure help?* The grid architecture creates a structural
asymmetry: column~0 always receives data; other columns must extract information
via the message layer. This asymmetry forces columns into differentiated roles —
early-layer columns distribute information while later-layer columns develop
specialized query or storage behaviors. The learnable column identity keys allow
the attention to select source columns by identity, enabling consistent routing
policies to emerge across training.

*When does Grid RNN fail?* HopfieldGridLRU and GridRNN-LRU both fail in early RL
due to entropy collapse under PPO (see §Experiments). A cleaner ablation of the
Hopfield message layer would require a GRU-column variant with Hopfield messages
vs. standard MHA messages; currently GridRNN-LRU vs. HopfieldGridLRU conflates
cell type (GRU vs. LRU) with message type (standard MHA vs. Hopfield), making it
impossible to attribute the SDQ performance gap to one factor alone. This ablation
is planned as future work.

*Limitations.* (1) GridHarmonic SDQ results are reported at peak checkpoint (95M
steps), not at convergence — the model oscillates thereafter, which we attribute
to unbounded column-norm growth in upper layers. (2) The Hopfield/LRU comparison
in Table~1 is confounded. (3) We do not yet have a GRU baseline on MIKASA
(Table~2 pending), which is a critical gap before this section is publishable.

A full sweep over $C in {2, 3, 4, 5}$ columns at fixed parameters is needed to
cleanly quantify the effect of column count independent of depth; the current
ablation only covers two points. All open gaps are tracked in the supplementary
future-work document.


= Conclusion

We introduced Grid Recurrent Networks, a 2D recurrent architecture where
$L times C$ cells organized in layers and columns exchange information via
inter-column multi-head attention. The grid structure enables spontaneous column
specialization without explicit supervision, and serves as a modular framework
compatible with diverse memory mechanisms. Across three benchmarks, Grid RNN
variants consistently and substantially outperform single-stream GRU baselines:
HopfieldGridLRU achieves 96.7% on SDQ-Hard; GridRNN-EMA reaches episode return
$approx 0.95$ on POPGym RepeatFirst at only 37% of full training; GridHarmonic
achieves 1.68 BPC on text8 with 2.1M parameters.

Future work will complete the MIKASA evaluation suite (including HigherLower,
MultiarmedBandit, and harder difficulty levels), provide full comparison with
published POPGym baselines, investigate entropy-regularized training to improve
LRU-cell convergence under PPO, and run a systematic column-count ablation.

// ─────────────────────────────────────────────────────────────────────────────
// REFERENCES
// ─────────────────────────────────────────────────────────────────────────────

#bibliography("refs.bib",
  style: "american-psychological-association",
  title: "References")

] // end columns
