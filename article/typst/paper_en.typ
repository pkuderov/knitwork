// MoSAIC — AAAI 2027 Submission
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
  title: [MoSAIC: Modular Self-Attentive Interacting Columns \ for Recurrent Memory],
  authors: [Anonymous Submission],
  affiliations: [],
)

#aaai-abstract[
Recurrent sequence models compress history into a fixed-size state. Increasing their width adds capacity, but leaves the organization and exchange of information within that state implicit. We introduce MoSAIC (Modular Self-Attentive Interacting Columns), a recurrent architecture that distributes a parameter budget across a grid of independently parameterized GRU columns. At every timestep, the columns use learned attention to read from a small message bank. The first layer reads both the current input and the previous top-layer column states, while each subsequent layer reads the newly updated states below it. Attention therefore determines what each column receives before its recurrent update, and the complete top layer is carried forward as a compact recurrent communication workspace.

We study whether this explicit modular routing improves sequence modeling under matched parameter budgets. At approximately 10M parameters on text8, MoSAIC-L2C4 reaches $1.437 plus.minus 0.003$ held-out BPC, compared with $1.484 plus.minus 0.008$ for the strongest stacked-GRU baseline, a 3.2% relative reduction. Every evaluated MoSAIC grid shape outperforms every matched GRU, while the widest grid is not the strongest despite retaining more than twice as many recurrent-state coordinates as L2C4. MoSAIC also reaches lower BPC in fewer optimizer updates than the evaluated HGRN2, mLSTM, and DeltaNet implementations.
]

#v(1em)

// ─────────────────────────────────────────────────────────────────────────────
// TWO-COLUMN BODY
// ─────────────────────────────────────────────────────────────────────────────

#columns(2, gutter: 0.375in)[

= Introduction

Many sequence problems require a model to retain information while continually incorporating new observations. Character-level language modeling tests this ability on natural streams, while controlled associative-recall tasks isolate storage, interference, and retrieval over known delays. Recurrent neural networks are a natural fit for these settings because their inference cost and state size do not grow with the processed sequence length.

Standard GRUs @cho2014gru and LSTMs @hochreiter1997lstm represent the past with one hidden vector per layer. Widening or stacking these vectors increases capacity, but does not explicitly organize how distinct stateful computations exchange information. Modern linear recurrences instead improve parallel training and long-range propagation @qin2024hgrn2 @dao2024mamba2 @yang2024deltanet, yet usually retain a single recurrent stream per layer. This motivates a complementary question: *how should a fixed recurrent parameter budget be organized when several persistent stateful modules are allowed to communicate?*

Modular recurrent architectures provide an important precedent. Recurrent Independent Mechanisms (RIMs) @goyal2019rims maintain separate recurrent modules, select a sparse subset for each input, and allow the active modules to communicate through attention. BRIMs @mittal2020brims introduce hierarchical top-down and bottom-up communication, while global-workspace models @goyal2021workspace impose a shared communication bottleneck. These methods establish that modular recurrence can be useful, but couple modularity to sparse activation or a particular workspace mechanism. We instead study a simpler dense setting in which every module updates at every timestep and learned routing determines its recurrent input.

#figure(
  image("figure1_grid.svg", width: 100%),
  caption: [*MoSAIC at timestep $t$.* The first routing layer is queried by the previous states $bold(H)_(t-1)^0$ and reads a bank containing the delayed top-layer states $bold(H)_(t-1)^{L-1}$ and current embedding $bold(e)_t$. Its $C$ messages are processed by independent GRU cells to produce $bold(H)_t^0$. Each later layer repeats the operation using the newly updated layer below as its message bank. The readout uses column~0 of the top layer, while all top-layer columns are carried to timestep $t+1$.],
  placement: top,
) <fig-arch>

We propose *MoSAIC (Modular Self-Attentive Interacting Columns)*, illustrated in @fig-arch. MoSAIC replaces each recurrent layer with $C$ independently parameterized GRU cells. Before a cell updates, its previous state queries a shared message bank through multi-head attention. At the bottom layer this bank contains the current input together with the *delayed complete top layer* from the preceding timestep; at higher layers it contains the current states of the layer below. Consequently, the model retains column-local recurrent states while learning both bottom-up and temporally delayed communication routes.

The architecture deliberately separates its defining computation from optional training regularizers. Learnable query/key identities make communication participants distinguishable. In the configuration used in our main experiments, training-time logit noise, a diagonal-route cost, and an attention-entropy bonus regularize routing, but they do not change the inference-time recurrent graph. We ablate these choices separately.

Our main contributions are:
+ *A recurrent modular architecture* in which independently parameterized GRU columns learn their inputs through attention over bottom-up and delayed top-down message banks.
+ *A controlled fixed-budget evaluation* against monolithic GRUs and modern recurrent and attention-based sequence models, reporting predictive quality together with parameter count and recurrent-state size. At approximately 10M parameters, MoSAIC-L2C4 reduces held-out text8 BPC by 3.2% relative to the strongest stacked GRU.
+ *An analysis of modular routing* through width/depth comparisons and ablations of delayed feedback, participant identities, and routing regularization. The completed grid-shape comparison shows that increasing width from four to sixteen columns does not improve BPC, despite more than doubling the recurrent-state size.


= Related Work

#par-heading[Multi-Dimensional and Grid RNNs.]
Multi-Dimensional RNNs @graves2007mdrnn recurrently propagate information along multiple axes of structured data. Grid LSTM @kalchbrenner2015gridlstm extends the idea by arranging LSTM blocks in a multidimensional grid whose dimensions exchange hidden and memory vectors through fixed connections. MoSAIC uses a grid in a different sense: all columns share one temporal axis, retain separate recurrent states, and exchange information through content-dependent routing rather than through edges fixed by the geometry of an input.

#par-heading[Modular Networks and Attention-Based Communication.]
RIMs @goyal2019rims are the closest conceptual predecessor: they preserve independent recurrent mechanisms, sparsely select mechanisms for each input, and allow active mechanisms to communicate through attention. BRIMs @mittal2020brims organize such mechanisms hierarchically, and shared global workspaces @goyal2021workspace restrict communication through a bandwidth-limited latent. Relational recurrent networks @santoro2018relational instead update a set of interacting memory slots through self-attention. MoSAIC shares the separation of persistent states and learned communication, but uses dense synchronous updates: every column queries a layer-specific message bank and then performs its own recurrent update. Its delayed top-layer bank also makes the communication graph recurrent across both layers and time.

#par-heading[Modern Recurrent Sequence Models.]
Recent work has substantially strengthened recurrent alternatives to Transformers. LRUs @orvieto2023lru stabilize long linear recurrences; HGRN2 @qin2024hgrn2 expands the state of a gated linear RNN; Mamba-2 @dao2024mamba2 connects selective state-space models and structured attention; and DeltaNet @yang2024deltanet uses a delta-rule matrix state to improve associative recall. xLSTM @beck2024xlstm extends LSTM gating and introduces scalar- and matrix-memory variants, including the mLSTM baseline used in our evaluation. These models primarily redesign the recurrent update or its state representation. MoSAIC instead keeps a conventional nonlinear GRU update and changes how a fixed parameter budget is distributed and routed.

#par-heading[Recall Evaluation.]
Associative-recall studies show that aggregate language-modeling quality can obscure substantial differences in information retrieval. Zoology @arora2023zoology uses multi-query associative recall to connect synthetic retrieval behavior with language-modeling performance. POPGym @morad2023popgym provides controlled partially observable tasks spanning object, sequential, capacity, and spatial memory. We use character-level language modeling as the principal natural-sequence evaluation and store--distract--query tasks to measure behavior as the retention interval and interference increase.


= Preliminaries and Problem Setting

Let $bold(e)_t in RR^{B times H}$ be an input embedding at timestep $t$, where $B$ is batch size and $H$ is hidden width. A stacked GRU with $L$ layers maintains $bold(S)_t in RR^{L times B times H}$ and computes one recurrent stream per layer. We compare models under a matched trainable-parameter budget, using the same token embedding, readout, data order, optimizer, and number of training tokens.

Parameter matching does not equate every resource. A stacked GRU stores $L B H$ recurrent scalars, whereas an $L times C$ MoSAIC grid stores $L C B H$. We therefore report the number of recurrent scalars per example, measured throughput, and peak memory in addition to parameters. This separates predictive gains under a fixed weight budget from the activation-memory and compute costs of exposing multiple column states.


= Method

== Recurrent Grid

MoSAIC maintains $bold(H)_t in RR^{L times C times B times H}$, where $bold(h)_t^{l,c}$ is the state of column $c in {0,...,C-1}$ at layer $l in {0,...,L-1}$. Each layer--column pair has independent GRU parameters. The cells of a layer are evaluated in one batched operation, but do not share recurrent weights.

The first layer reads from a message bank containing the delayed top layer and the current external inputs:

$ bold(M)_t^0 = [bold(H)_{t-1}^{L-1}; bold(e)_t^1; ...; bold(e)_t^I]
  in RR^{(C+I) times B times H}. $

Here $I$ is the number of external input streams; $I=1$ for language modeling. For every subsequent layer, the message bank is the newly updated layer below:

$ bold(M)_t^l = bold(H)_t^{l-1}, quad l > 0. $

Thus information moves bottom-up within a timestep, whereas the complete top layer supplies delayed feedback across timesteps. Unlike a conventional stacked GRU, a column does not receive the same-index lower-layer state by a fixed edge: it learns a distribution over the available messages.

== Learned Routing

At layer $l$, the $C$ previous column states act as queries. Learnable participant identities $bold(a)^Q_c$ and $bold(a)^K_j$ distinguish query columns and message sources. For attention head $r$,

$ bold(q)_{t,c,r}^l =
    "SiLU"(bold(W)^Q_r (bold(h)_{t-1}^{l,c}+bold(a)^Q_c)+bold(b)^Q_r), $
$ bold(k)_{t,j,r}^l =
    "SiLU"(bold(W)^K_r (bold(m)_{t,j}^{l}+bold(a)^K_j)+bold(b)^K_r), $
$ bold(v)_{t,j,r}^l =
    "SiLU"(bold(W)^V_r bold(m)_{t,j}^{l}+bold(b)^V_r). $

Each query column has a learned positive inverse temperature $beta_c="softplus"(rho_c)$. The routing probabilities are

$ pi_{t,c,j,r}^l =
  "softmax"_j (beta_c (
    frac(bold(q)_{t,c,r}^l dot bold(k)_{t,j,r}^l, sqrt(H/R))
    + epsilon_{t,c,j,r}^l)), $

where $R$ is the number of heads. The perturbation $epsilon$ is zero at inference; during training we optionally sample independent zero-mean Gaussian noise. Concatenated head outputs are projected and normalized:

$ bold(x)_{t,c}^l =
  "LN"(bold(W)^O [sum_j pi_{t,c,j,1}^l bold(v)_{t,j,1}^l;
                   ...;
                   sum_j pi_{t,c,j,R}^l bold(v)_{t,j,R}^l]). $

This attention operation produces the input to the recurrent cell; it does not replace or post-process the cell state.

== Independent Recurrent Updates and Readout

After routing, every column performs an ordinary GRU update with its own parameters:

$ bold(h)_t^{l,c} =
  "GRU"_{l,c}(bold(x)_{t,c}^l, bold(h)_{t-1}^{l,c}). $

The prediction head reads column~0 of the final layer, $bold(o)_t=bold(h)_t^{L-1,0}$. The recurrent state passed to the next timestep contains every column, and $bold(H)_t^{L-1}$ becomes part of the next first-layer message bank. Restricting the readout to one column prevents a pooled output from bypassing the learned routes, while leaving all columns available as recurrent memory.

== Routing Regularization

The recurrent grid, message banks, and learned routing distribution define MoSAIC. Our main training configuration additionally uses three routing regularizers: Gaussian logit noise, a diagonal-route cost that biases columns toward retaining a low-cost self route, and an entropy bonus that discourages prematurely deterministic routing. At the first layer the cost also encourages the designated readout column to receive the external input while discouraging direct input access for later columns. For task loss $L_"task"$,

$ L = L_"task" + lambda_"comm" L_"comm"
                    - lambda_"ent" H(pi). $

These terms are applied only during training and add no recurrent state or inference-time operation. We report them as ablations rather than treating them as architectural contributions.

== Complexity and Resource Accounting

Ignoring embeddings and biases, independent GRUs contribute approximately $6 L C H^2$ parameters and the routing projections contribute approximately $4 L H^2$. The recurrent state contains $L C H$ scalars per example. Routing also requires $O(L C (C+I) H)$ score computation per timestep in addition to the dense projections and GRU updates. Because a parameter-matched MoSAIC can therefore retain more state coordinates than a monolithic GRU, our evaluation reports parameters, recurrent-state size, throughput, and peak memory together.


= Experiments

Our experiments ask four questions: (Q1) does distributing a fixed parameter budget across routed recurrent columns improve language modeling over a monolithic GRU; (Q2) how should that budget be divided between depth, column count, and within-column width; (Q3) which routing and feedback components are necessary; and (Q4) what activation-memory and computational costs accompany any quality gain?

== Character-Level Language Modeling

#par-heading[Data and evaluation.]
We use text8, a 100-million-character normalized English corpus with a 27-character alphabet. The first 90M characters are used for training and the final 10M form a contiguous held-out evaluation split. We used held-out bits per character (BPC) for limited manual configuration tuning rather than a large hyperparameter search; we therefore call this quantity *held-out BPC* rather than an untouched test estimate. All models process the corpus in the same contiguous order and use the same tokenization and reset schedule.

#par-heading[Training.]
Models are optimized with RMSprop and truncated backpropagation through time using 512 parallel streams and a truncation length of 64. The per-token random state-reset probability decays during training, increasing the expected uninterrupted context from approximately 320 to 64,000 characters. Gradients are clipped to norm~1.0. The learning rate warms up from near zero, reaches $5 times 10^{-4}$, and decays to $5 times 10^{-5}$. We train each configuration for approximately one billion observed characters. Reported central values aggregate completed independent runs of each frozen configuration.

#par-heading[Models and resource matching.]
@tab-lm-current lists the currently tested configurations. GRU depth and MoSAIC grid shape are varied while keeping trainable parameters close to 10M. Because parameter matching yields different recurrent-state sizes, the table also reports the number of state scalars per example. We additionally compare against parameter-matched mLSTM @beck2024xlstm, HGRN2 @qin2024hgrn2, DeltaNet @yang2024deltanet using the same data and objective. These models exceed available accelerator memory at the MoSAIC/GRU batch geometry. They therefore use four- to eight-times fewer parallel streams, with their character budgets reduced correspondingly. We compare their optimization progress using the logged number of parameter updates.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    stroke: none,
    table.hline(),
    [*Model*], [*Shape*], [*Params*], [*State*], [*Held-out BPC ↓*],
    table.hline(),
    [GRU], [L1], [10.16M], [1,296], [$1.561 plus.minus 0.003$],
    [GRU], [L2], [10.04M], [1,824], [1.502],
    [GRU], [L3], [10.23M], [2,256], [$1.484 plus.minus 0.008$],
    table.hline(),
    [MoSAIC], [L1C8], [10.11M], [3,520], [1.473],
    [MoSAIC], [L2C4], [10.11M], [3,392], [$1.437 plus.minus 0.003$],
    [MoSAIC], [L2C8], [10.17M], [4,992], [1.443],
    [MoSAIC], [L2C16], [10.09M], [7,168], [1.456],
    [MoSAIC], [L3C4], [9.99M], [4,128], [*1.433*],
    table.hline(),
  ),
  caption: [Parameter-matched text8 evaluation. State counts exclude optimizer state and report persistent recurrent scalars per example. Uncertainty is shown where multiple completed runs are currently available.],
  placement: top,
  kind: table,
) <tab-lm-current>

#par-heading[Results.]
MoSAIC improves consistently over the parameter-matched GRU family. Its replicated L2C4 configuration reaches $1.437 plus.minus 0.003$ BPC, compared with $1.484 plus.minus 0.008$ for the strongest GRU, an absolute reduction of 0.047 BPC and a 3.2% relative reduction. The result is not driven by one favorable grid shape: even the weakest evaluated MoSAIC configuration (L1C8, 1.473 BPC) outperforms the strongest GRU. Depth benefits both families, but distributing a two-layer MoSAIC across more columns is not monotonically helpful: L2C4, L2C8, and L2C16 obtain 1.437, 1.443, and 1.456 BPC, respectively. In particular, L2C16 retains 7,168 state scalars—more than twice the 3,392 of L2C4—yet performs worse. Thus the gain cannot be explained by state size alone; the allocation of capacity between column width, grid width, and depth matters.

The memory-limited modern baselines process fewer characters per update, so their raw character horizons are not directly matched to the 1B-character MoSAIC/GRU runs. The update-indexed comparison is nevertheless decisive within the tested implementations: HGRN2, mLSTM, and DeltaNet finish near 1.673, 1.686, and 1.844 BPC after approximately 50k optimizer updates, whereas MoSAIC-L2C4 reaches 1.437 BPC in approximately 30k updates. This establishes stronger optimization-update efficiency under the common objective; it does not by itself imply equal wall-clock, FLOP, or accelerator-memory efficiency.

== Architectural Ablations

The completed grid-shape comparison varies columns at fixed depth and depth at comparable column count. It identifies a moderate grid as the strongest allocation: L2C4 improves on L2C8 and L2C16, while L3C4 gives the best central BPC in @tab-lm-current. The remaining component ablations remove: (i) participant identities, (ii) delayed top-layer feedback from the first-layer bank, (iii) nonlinear query/key/value projections, (iv) training-time routing noise, (v) the diagonal communication cost, and (vi) the routing-entropy bonus. These experiments are required to distinguish the recurrent modular topology from its training-time routing regularizers.

== Store--Distract--Query

We use a controlled online task to evaluate retention under interference. With five keys and ten values, the input vocabulary contains 50 key--value store tokens, ten distractor tokens, and five key-specific query tokens. Store events overwrite the selected key. Distractors update a running sum modulo ten. At a query, the target combines the most recently stored value, the distractor sum, and, in the Hard variant, per-key store and query counts. Loss and accuracy are computed only on query events.

Episode lengths are geometrically distributed with a curriculum-controlled mean. We report query accuracy by store--query gap, including failures on keys that have been overwritten, rather than aggregate token accuracy. All architectures use the same online sequence distribution, curriculum decisions, number of training tokens, and parameter budget.
// TODO(port): insert results only after SDQ uses grnn_core and the current baseline suite.

== Routing and Efficiency Analysis

@tab-lm-current reports parameter count and recurrent-state scalars for every completed text8 configuration. The final resource comparison will add training throughput, inference throughput at batch size one, and peak accelerator memory. Learned routes will be summarized by per-layer entropy and average routing matrices. To test whether a route or column is functionally important rather than merely correlated with the output, we will mask it at evaluation and measure the increase in BPC or query error on held-out data.


= Discussion

MoSAIC changes the allocation of recurrent capacity: the same number of trainable weights is distributed across several narrower persistent states connected by a learned routing graph. The text8 results show that this allocation is useful: MoSAIC-L2C4 improves over the strongest parameter-matched GRU by 0.047 BPC, and every tested grid shape improves over every GRU depth. The within-family comparison also argues against a simple state-volume explanation, because L2C16 has more than twice the recurrent state of L2C4 but worse BPC. The improvement is not free, however: increasing column count enlarges persistent activation state and introduces quadratic routing cost in $C$. Throughput- and memory-aware measurements remain necessary to characterize that trade-off.

The architecture also does not guarantee functional specialization. Distinct parameters and participant identities make differentiated behavior possible, but attention visualizations alone cannot establish it. We use held-out route and column interventions to support only those specialization claims that produce repeatable causal effects.

#par-heading[Limitations.]
MoSAIC remains sequential over time and therefore does not inherit the sequence-parallel training algorithms of modern linear recurrent models. The present natural-language evaluation is character-level and small-scale, and SDQ is synthetic. The grid exposes more recurrent state coordinates than some parameter-matched GRUs, so gains must be interpreted together with activation memory and throughput. The 10M-character held-out split was also used for limited manual configuration tuning, so it is not a strictly untouched test set. Finally, routing regularizers introduce additional hyperparameters; their robustness across tasks remains an empirical question.


= Conclusion

We introduced MoSAIC, a recurrent architecture that distributes a fixed parameter budget across independently parameterized GRU columns and learns their inputs through attention over bottom-up and delayed top-down message banks. This design separates persistent column states from an explicit recurrent communication graph while retaining constant state size with respect to sequence length. Our parameter-matched text8 evaluation shows a consistent advantage over stacked GRUs: MoSAIC-L2C4 reaches $1.437 plus.minus 0.003$ held-out BPC versus $1.484 plus.minus 0.008$ for the strongest GRU, and every evaluated MoSAIC shape outperforms every GRU depth. Increasing the grid from four to sixteen columns does not improve BPC despite more than doubling recurrent-state size, showing that the organization—not merely the volume—of persistent state matters. MoSAIC also reaches lower BPC in fewer optimizer updates than the evaluated HGRN2, mLSTM, and DeltaNet implementations. Future evaluation must complement these predictive gains with wall-clock, accelerator-memory, and component-ablation measurements.

/*
LEGACY EXPERIMENT DRAFT (pre-grnn_core). Kept temporarily for provenance while current runs finish; none of the following content is rendered.

=============== BEGIN LEGACY ===============

We evaluate three Grid RNN variants against GRU baselines on three benchmarks. All models are trained on a single GPU (RTX~3050). Parameter budgets are matched within benchmarks at approximately 2.1M parameters. Learning rate uses a linear warmup followed by cosine decay ($eta_"max" = 8 times 10^{-4}$), implemented via PPO @schulman2017ppo for RL experiments and BPTT for supervised tasks.

== Store-Distract-Query (SDQ)

#par-heading[Task.] SDQ @morad2023popgym presents sequences of tokens from a vocabulary of 65 (50 store events, 10 distractors, 5 query types). A *store* event presents a key-value pair $(k, v)$ that the model must bind for later retrieval. *Query* tokens present a previously stored key and require the model to output the correct value. The *Hard* variant additionally counts the total number of stored and queried pairs simultaneously. Models are trained online with curriculum scheduling: the average sequence length $T$ starts at 10 and increases when the moving-average improvement in Acc exceeds a threshold (checked every 200K steps; check frequency multiplied by $0.97$ on progress, $times 1.25$ on stagnation). Training runs for 1B steps on CPU with rollout length 8.

#par-heading[Results.] Table~@tab-sdq shows accuracy metrics after $tilde$1B training steps. Acc measures overall token accuracy; Acc++ measures accuracy *only* on hard query tokens — the most discriminative metric because distractor tokens are trivially correct.

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
  caption: [SDQ-Hard results. Acc = overall token accuracy; Acc++ = accuracy on hard query tokens only. GRU baseline from preliminary runs (exact values pending full convergence run). GridRNN and HopfieldGridLRU: mean of final 10M steps (stable). GridHarmonic: best checkpoint at step 95M (peak before oscillation onset; see Discussion). GridRNN-LRU: final value at step 87M.],
  placement: top,
  kind: table,
) <tab-sdq>

GridRNN (GRU columns) dramatically outperforms single-stream GRU: Acc++ improves from $approx 0.15$ to $0.917$ ($+77$ pp absolute). This demonstrates that the *column structure alone* — without any specialized memory mechanism — provides decisive benefit. The grid creates structural differentiation that single-stream GRU cannot achieve regardless of width or depth. We note that the GRU baseline numbers are preliminary (full convergence run pending; see future-work document); the large magnitude of the gap makes this unlikely to reverse.

*Note on GridHarmonic.* While GridHarmonic achieves 0.849~Acc++ (at peak), Table~1 also reveals that the simplest grid variant (GridRNN, GRU columns) reaches 0.917~Acc++ — exceeding GridHarmonic and suggesting the core column structure contributes more than the additional Delta-rule and EMA components in this task.

HopfieldGridLRU ($0.967$~Acc, $0.932$~Acc++) further improves over the GRU-column variant by combining spectral memory stability (LRU spectral grid) with sharp Hopfield retrieval for cross-column information routing. The learnable $beta$ per head allows fine-grained tuning of retrieval selectivity.

GridRNN-LRU without Hopfield messages (0.849~Acc++) underperforms the GRU-column variant, suggesting that the complex-valued LRU state alone is not sufficient: the Hopfield message layer is critical for sharp inter-column routing.

GridHarmonic (0.849~Acc++) currently underperforms HopfieldGridLRU. Two known issues: (1) column representation norms grow unboundedly in upper layers after ~100M steps (e.g., $||h_("col0, L2")||$ exceeds 300 at step 110M), causing oscillations; (2) the Acc/store metric (store-operation accuracy) returns NaN throughout training due to a masking bug, meaning the quality of the store operation itself is unverifiable. Both issues are tracked in Future Work.

== MIKASA/POPGym Benchmark

#par-heading[Task.] POPGym @morad2023popgym provides a suite of partially observable environments testing four memory types: object (remember a property of an object), sequential (remember the order of events), capacity (hold many items simultaneously), and spatial (remember a map). We evaluate on two tasks from MIKASA @cherepanov2025mikasa:
- *RepeatFirst* (object memory): the agent observes $N$ symbols and must reproduce the first one at episode end. Episode length is 51 steps; per-step reward $+1/50$ for correct recall.
- *HigherLower* (sequential memory): predict whether the next card is higher or lower than the current; requires tracking the running distribution of revealed cards.

All models are trained with PPO @schulman2017ppo, $gamma=0.99$, $lambda_"GAE"=0.95$, 64 parallel environments, rollout length 32, for up to 200M environment steps.

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
  caption: [POPGym results (episode return, scale $[{-}1, 1]$; higher is better). #super[†]Results at 74M/200M steps (37% of training). Full 200M-step results and GRU baseline comparison pending.],
  placement: top,
  kind: table,
) <tab-mikasa>

#par-heading[Results.] Table~@tab-mikasa shows results for GridRNN-EMA at 37% of training (74M/200M steps). GridRNN-EMA achieves episode return $approx 0.95$ on RepeatFirst, approaching the theoretical maximum return of~1.0. This indicates that the EMA surprise gate successfully identifies the first-observation event (the only one worth writing) and stores it reliably against 50 subsequent distractors.

Performance on HigherLower ($~0.41$) is lower, consistent with the harder sequential nature of this task: the agent must track a running distribution rather than simply recall a single item. This gap motivates combining EMA surprise writing with LRU spectral memory to address sequential tasks (a target for future work).

#par-heading[GridRNN-EMA training curve.] At 37% of training (74M steps), the RepeatFirst episode return ($approx 0.95$) is already near-optimal. Convergence on this environment is typically reached between 30–50M steps; the remaining 126M steps are expected to consolidate rather than substantially improve return. The HigherLower curve ($approx 0.41$) has not plateaued — we expect continued improvement toward a target return of $>0.6$ at 200M steps, but this requires confirming empirically.

#par-heading[LRU instability under PPO.] Two additional models were stopped early at 15M steps with negative episode returns ($< 0$). In GridRNN-LRU, policy entropy $H$ collapsed from 1.1 to 0.19 bits within the first 10M steps — an irreversible determinization that prevented further learning. Specifically, the LRU spectral constraint produces a very smooth hidden-state trajectory, which causes the policy to commit to a single action early. In HopfieldGridLRU, $H$ remained higher (1.18 bits) but training oscillated without monotone improvement. Both behaviours point to a structural incompatibility between LRU's linear recurrence (no saturating nonlinearity) and standard PPO's entropy coefficient (0.01), rather than slow convergence. Increasing `entropy_coef` to 0.05 is the most promising fix (future work).

== Language Modeling

#par-heading[Setup.] We evaluate on two character-level datasets: *text8* (100M chars, 27-token alphabet) and *Shakespeare* (1.1M chars, 65-token alphabet). Models are trained with BPTT (rollout length 16 for GridHarmonic, 8 for others), 128 parallel environment streams, and a reset-probability curriculum that decays from 0.01 to $10^{-4}$ over training, gradually increasing effective context length. All Grid RNN models use $approx 2.1$M parameters to enable fair comparison; published baselines use substantially larger models.

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
  caption: [Character-level language modeling (bits per character, BPC; lower is better). Published baselines use models 12–130$times$ larger. Our models are all $approx$2.1M parameters. Grid variants consistently improve over the single-stream GRU baseline.],
  placement: top,
  kind: table,
) <tab-lm>

#par-heading[Results.] Table~@tab-lm shows character-level BPC. GridHarmonic achieves 1.68~BPC on text8 with 2.1M parameters, compared to AWD-LSTM at 1.19~BPC with 24M parameters (12× more). While Grid RNN does not close the absolute gap to SOTA, we emphasize that the comparison is intentionally cross-scale: our goal is to demonstrate consistent improvement *within a fixed parameter budget*. GridHarmonic improves over single-stream GRU by 0.41 BPC (19.6% relative), matching the pattern of improvement seen in SDQ.

#par-heading[Training Dynamics.] Table~@tab-text8-curve tracks GridHarmonic (v3) BPC progression on text8 over 270M steps. BPC decreases continuously from 2.83 at 5M steps to 1.68 at 270M, without divergence or oscillation — in contrast to SDQ training where column-norm instability appears after 100M steps. The curriculum increases effective context length ($T$) from 100 to 239 tokens as BPC improves, so later steps train on significantly longer sequences. The residual improvement rate at 250–270M is 0.010 BPC/50M steps, suggesting further gains remain possible with extended training.

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
  caption: [GridHarmonic text8 BPC progression over 270M training steps. Context $T$ is the effective sequence length set by the curriculum. Training terminated early; rate at 270M ($approx 0.010$ BPC/50M) suggests further improvement possible.],
  placement: top,
  kind: table,
) <tab-text8-curve>

#par-heading[Memory Diagnostics.] GridHarmonic exposes interpretable internal metrics for the SurpriseDelta memory module. At step 270M (text8): the weight matrix norms grow monotonically across layers ($||bold(W)||_F$: L0=1.07, L1=1.53, L2=1.90), confirming that deeper layers — with slower spectral radii ($r_"max"$ up to 0.999) — accumulate more persistent associations. Write-gate $alpha_t$ remains in $(0.85, 0.89)$ across layers, indicating consistently active writing without saturation. Matrix fullness ($||bold(W)||_F / sqrt(d_k d_v)$) stays below 3% at all layers, indicating ample remaining capacity. Crucially, column diversity (mean pairwise cosine distance between column states) remains moderate at L2=2.16 on text8, whereas the same metric reaches 7.79 on SDQ at layer~2 — flagging the norm explosion visible in column norms (Section~Discussion).

== Ablation: Column Count

To isolate the effect of grid width, we compare two GridRNN configurations matched to the same parameter budget ($approx 2.1$M) on SDQ-Hard:

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
  caption: [Ablation: column count vs. depth at fixed $approx$2.1M params (SDQ-Hard). Increasing from 2 to 4 columns ($+$2 layers) yields $+42$ pp Acc++.],
  placement: top,
  kind: table,
) <tab-ablation>

Moving from 2 columns / 1 layer to 4 columns / 3 layers while keeping the parameter count constant yields $+22.6$~pp Acc ($0.734 arrow.r 0.960$) and $+42.3$~pp Acc++ ($0.494 arrow.r 0.917$). This confirms that column count is the primary driver of performance, not raw hidden size: distributing capacity across more specialized columns provides dramatically more representational power for associative tasks than concentrating it in a deeper single stream.


= Discussion

*Why does column structure help?* The grid architecture creates a structural asymmetry: column~0 always receives data; other columns must extract information via the message layer. This asymmetry forces columns into differentiated roles — early-layer columns distribute information while later-layer columns develop specialized query or storage behaviors. The learnable column identity keys allow the attention to select source columns by identity, enabling consistent routing policies to emerge across training.

*When does Grid RNN fail?* HopfieldGridLRU and GridRNN-LRU both fail in early RL due to entropy collapse under PPO (see §Experiments). A cleaner ablation of the Hopfield message layer would require a GRU-column variant with Hopfield messages vs. standard MHA messages; currently GridRNN-LRU vs. HopfieldGridLRU conflates cell type (GRU vs. LRU) with message type (standard MHA vs. Hopfield), making it impossible to attribute the SDQ performance gap to one factor alone. This ablation is planned as future work.

*Limitations.* (1) GridHarmonic SDQ results are reported at peak checkpoint (95M steps), not at convergence — the model oscillates thereafter, which we attribute to unbounded column-norm growth in upper layers. (2) The Hopfield/LRU comparison in Table~1 is confounded. (3) We do not yet have a GRU baseline on MIKASA (Table~2 pending), which is a critical gap before this section is publishable.

A full sweep over $C in {2, 3, 4, 5}$ columns at fixed parameters is needed to cleanly quantify the effect of column count independent of depth; the current ablation only covers two points. All open gaps are tracked in the supplementary future-work document.


= Conclusion

We introduced Grid Recurrent Networks, a 2D recurrent architecture where $L times C$ cells organized in layers and columns exchange information via inter-column multi-head attention. The grid structure enables spontaneous column specialization without explicit supervision, and serves as a modular framework compatible with diverse memory mechanisms. Across three benchmarks, Grid RNN variants consistently and substantially outperform single-stream GRU baselines: HopfieldGridLRU achieves 96.7% on SDQ-Hard; GridRNN-EMA reaches episode return $approx 0.95$ on POPGym RepeatFirst at only 37% of full training; GridHarmonic achieves 1.68 BPC on text8 with 2.1M parameters.

Future work will complete the MIKASA evaluation suite (including HigherLower, MultiarmedBandit, and harder difficulty levels), provide full comparison with published POPGym baselines, investigate entropy-regularized training to improve LRU-cell convergence under PPO, and run a systematic column-count ablation.

=============== END LEGACY ===============
*/

// ─────────────────────────────────────────────────────────────────────────────
// REFERENCES
// ─────────────────────────────────────────────────────────────────────────────

#bibliography("refs.bib",
  style: "american-psychological-association",
  title: "References")

] // end columns
