// MoSAIC — Anonymous AAAI-27 submission
// Compile from this directory with:
//   typst compile paper_en.typ paper_en.pdf
//
// The layout follows the repository's AAAI Typst approximation. Keep the
// editable authority in this file and preserve anonymous-submission metadata.

// ─────────────────────────────────────────────────────────────────────────────
// PAGE AND TYPOGRAPHY
// ─────────────────────────────────────────────────────────────────────────────

#set page(
  paper: "us-letter",
  margin: (top: 0.75in, bottom: 1.25in, left: 0.75in, right: 0.75in),
  numbering: none,
  header: none,
  footer: none,
)

#set text(font: "Times New Roman", size: 10pt, lang: "en")
#set par(leading: 2pt, justify: true, first-line-indent: 10pt)

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
#set list(indent: 1em, body-indent: 0.5em)
#set enum(indent: 1em, body-indent: 0.5em)

#show figure.caption: set text(size: 10pt)
#let wide-figure(caption: [], body) = figure(
  body,
  caption: caption,
  placement: top,
  scope: "parent",
)
#let wide-table(caption: [], body) = figure(
  body,
  caption: caption,
  placement: top,
  scope: "parent",
  kind: table,
)

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

// ─────────────────────────────────────────────────────────────────────────────
// PAPER
// ─────────────────────────────────────────────────────────────────────────────

#aaai-title(
  title: [MoSAIC: Modular Self-Attentive Interacting Columns \ for Recurrent Memory],
  authors: [Anonymous Submission],
  affiliations: [],
)

#aaai-abstract[
Recurrent models preserve a fixed-size state, but conventionally organize that state as one monolithic stream per layer. We introduce MoSAIC (Modular Self-Attentive Interacting Columns), which factorizes recurrent state into persistent GRU columns and uses attention-mediated routing to determine the input received by each column. Under approximately parameter-matched, one-billion-token training, this structural bias consistently improves both associative memory and character-level prediction. On Store--Distract--Query, MoSAIC-L2C4 obtains $0.843 plus.minus 0.006$ final-window accuracy on long-gap queries versus $0.532 plus.minus 0.103$ for a two-layer GRU; the best observed configuration, L3C4, obtains $0.920 plus.minus 0.013$. On text8, L2C4 obtains $1.437 plus.minus 0.003$ validation bits per character (BPC) and L3C4 obtains $1.435 plus.minus 0.002$, compared with $1.483 plus.minus 0.006$ for the strongest GRU and $1.449 plus.minus 0.012$ for a similarly sized finite-context Transformer. These results support modular state organization as a useful inductive bias for recurrent computation at this scale.
]

#v(1em)

#columns(2, gutter: 0.375in)[

= Introduction

Recurrent neural networks compress a sequence history into a fixed-size state and update it incrementally. This property makes them natural models for streaming prediction and memory tasks. A conventional GRU @cho2014gru or LSTM @hochreiter1997lstm, however, represents each layer as a single hidden-state stream. Increasing its width or depth adds capacity, but leaves the internal organization of state and the exchange of information within that state implicit.

We study a complementary architectural question: *can a fixed recurrent parameter budget be organized into persistent modules whose communication is learned?* The motivation is not primarily to extend nominal context length. It is to give recurrent computation an explicit topology. Separate state-bearing components can maintain local dynamics, while content-dependent routing determines how information moves among them.

We introduce *MoSAIC (Modular Self-Attentive Interacting Columns)*. MoSAIC distributes each recurrent layer across independently parameterized GRU columns. Before every recurrent update, the previous state of each column queries a small message bank through multi-head attention. The bottom layer reads the current input together with the delayed top-layer states from the preceding timestep; higher layers read the newly updated states below them. All columns update at every timestep, and the complete top layer persists to the next timestep.

This design uses modular state organization as an inductive bias rather than sparse expert selection. The number of columns and layers provides explicit axes along which recurrent state and computation can be organized. Attention is applied only over a fixed collection of columns and current inputs, so the recurrent state remains fixed in size as the processed sequence grows.

We evaluate this thesis under controlled training at approximately ten million parameters. Store--Distract--Query (SDQ) isolates binding and retrieval under interference; text8 tests transfer to natural character-level prediction. The principal comparisons use a common one-billion-token horizon and multiple completed runs. Text8 additionally includes similarly sized finite-context Transformers. L2C4 serves as a compact reference shape, while L3C4 is the best observed configuration; we do not treat this distinction as a resolved choice of a universal primary model.

Our contributions are:

+ *Architecture:* a layered grid of persistent recurrent columns with attention-mediated inter-column communication.
+ *Controlled topology:* column count and depth provide explicit axes for organizing recurrent state and computation at roughly fixed parameter scale.
+ *Empirical evidence:* matched GRU/MoSAIC comparisons show consistent improvements on SDQ and text8, and the text8 comparison includes a similarly sized finite-context Transformer.

= Related Work

#par-heading[Modular recurrence and communication]
Recurrent Independent Mechanisms (RIMs) @goyal2019rims maintain separate recurrent modules, sparsely activate a subset for each input, and allow active modules to communicate through attention. BRIMs @mittal2020brims add hierarchical bottom-up and top-down communication, while shared-workspace models @goyal2021workspace impose a communication bottleneck. Relational Memory Core @santoro2018relational instead updates interacting memory slots through attention. These works establish modular recurrence and learned communication as important design principles. MoSAIC differs in using a regular layered topology with dense synchronous updates: every persistent column updates at every timestep after reading from a layer-specific message bank. We therefore describe the mechanism as learned routing, not sparse or conditional computation.

#par-heading[Grid and multidimensional recurrence]
Multi-Dimensional RNNs @graves2007mdrnn propagate information recurrently along multiple axes of structured data. Grid LSTM @kalchbrenner2015gridlstm arranges LSTM blocks in a multidimensional grid whose dimensions exchange hidden and memory vectors through fixed connections. MoSAIC uses a grid as an organization of modules rather than as an additional axis of the input. All columns share one temporal axis, retain separate recurrent states, and communicate through content-dependent routing rather than edges fixed by input geometry.

#par-heading[Modern recurrent and attention models]
Recent architectures strengthen recurrent alternatives to full self-attention by redesigning the recurrent update or its memory representation. LRUs @orvieto2023lru stabilize long linear recurrences; HGRN2 @qin2024hgrn2 expands the state of a gated linear RNN; Mamba-2 @dao2024mamba2 connects selective state-space models with structured attention; DeltaNet @yang2024deltanet uses delta-rule matrix state; and xLSTM @beck2024xlstm includes scalar- and matrix-memory variants. Transformers @vaswani2017attention and segment-recurrent variants such as Transformer-XL @dai2019transformerxl instead retain a finite or growing bank of token representations for attention. MoSAIC keeps a conventional nonlinear GRU update and changes how recurrent state is divided and routed. Our experiments compare against GRUs and finite-context Transformers under the same token budget; memory-limited implementations of HGRN2, DeltaNet, and mLSTM are reported only under their separate reduced-token protocols.

= Method

#figure(
  image("fig_architecture.pdf", width: 100%),
  caption: [*MoSAIC mechanism.* At timestep $t$, each persistent column state queries a layer-specific message bank. The first layer reads the current input together with the previous top-layer column states; later layers read the newly updated states below them. The routed message is the input to an independent recurrent update. All top-layer column states are carried to timestep $t+1$.],
  placement: top,
) <fig-architecture>

== Persistent Columns and Layered Topology

Let $bold(e)_t in RR^{B times H}$ be an embedded input for a batch of size $B$. A conventional $L$-layer GRU maintains $bold(S)_t in RR^{L times B times H}$. MoSAIC instead maintains

$ bold(H)_t in RR^{L times C times B times H}, $

where $bold(h)_t^{l,c}$ denotes column $c$ at layer $l$. Each layer--column pair has independent GRU parameters. Columns in a layer are evaluated together for efficiency, but do not share recurrent weights.

The message bank for the first layer contains the delayed top layer and the current external inputs:

$ bold(M)_t^0 =
  [bold(H)_{t-1}^{L-1}; bold(e)_t^1; ...; bold(e)_t^I]
  in RR^{(C+I) times B times H}. $

Here $I$ is the number of input streams and $I=1$ in our experiments. At subsequent layers,

$ bold(M)_t^l = bold(H)_t^{l-1}, quad l > 0. $

Information therefore moves bottom-up within a timestep, while the complete top layer provides a delayed feedback path across timesteps. @fig-architecture summarizes this computation.

== Attention-Mediated Routing

At layer $l$, the previous states of its $C$ columns form the queries. Learnable identities distinguish query columns and message sources. For head $r$,

$ bold(q)_{t,c,r}^l =
    "SiLU"(bold(W)^Q_r (bold(h)_{t-1}^{l,c}+bold(a)^Q_c)+bold(b)^Q_r), $
$ bold(k)_{t,j,r}^l =
    "SiLU"(bold(W)^K_r (bold(m)_{t,j}^l+bold(a)^K_j)+bold(b)^K_r), $
$ bold(v)_{t,j,r}^l =
    "SiLU"(bold(W)^V_r bold(m)_{t,j}^l+bold(b)^V_r). $

Each query column has a learned positive inverse temperature
$beta_c = "softplus"(rho_c)$. The routing probabilities are

$ pi_{t,c,j,r}^l =
  "softmax"_j (beta_c (
    frac(bold(q)_{t,c,r}^l dot bold(k)_{t,j,r}^l, sqrt(H/R))
    + epsilon_{t,c,j,r}^l)), $

where $R$ is the number of heads. The perturbation $epsilon$ is independent zero-mean Gaussian noise during training and zero at evaluation. Concatenated head outputs are linearly projected and layer-normalized:

$ bold(x)_{t,c}^l =
  "LN"(bold(W)^O [
    sum_j pi_{t,c,j,1}^l bold(v)_{t,j,1}^l;
    ...;
    sum_j pi_{t,c,j,R}^l bold(v)_{t,j,R}^l
  ]). $

Routing determines the input to each recurrent cell; it does not replace or post-process its persistent state.

== Recurrent Update and Readout

Every column performs an ordinary independent GRU update,

$ bold(h)_t^{l,c} =
  "GRU"_{l,c}(bold(x)_{t,c}^l, bold(h)_{t-1}^{l,c}). $

The prediction head reads column 0 of the top layer. All top-layer columns remain in the recurrent state and form part of the next bottom-layer message bank. This makes routing necessary for information stored outside the readout column to affect predictions.

The main configurations additionally use three training-only routing terms: Gaussian logit noise, a cost favoring inexpensive diagonal routes, and an entropy bonus that discourages early route collapse. For task loss $cal(L)_"task"$,

$ cal(L) = cal(L)_"task"
  + lambda_"comm" cal(L)_"comm"
  - lambda_"ent" cal(H)(pi). $

These terms add no inference-time state and are not treated as separate empirical contributions in the absence of completed component ablations.

== Fixed-State and Resource Accounting

Attention is applied over $C+I$ messages in the bottom layer and $C$ messages above it, rather than over token history. For fixed $L$, $C$, and $H$, MoSAIC therefore carries a fixed-size recurrent state and supports incremental processing as sequence length grows. This is an architectural property, not an empirical claim of superior long-context scaling.

Ignoring embeddings and biases, the independent GRUs contribute approximately $6 L C H^2$ parameters and routing projections approximately $4 L H^2$. Persistent state contains $L C H$ scalars per example, compared with $L H$ for a stacked GRU. Parameter matching consequently does not match activation state or computation. We report recurrent-state size to make this distinction explicit, but do not claim resource efficiency.

= Experiments

== Shared Protocol and Reporting

All principal GRU and MoSAIC configurations are approximately parameter matched at ten million trainable parameters. The principal horizon is one billion processed tokens. Runs with a final logged evaluation point slightly below one billion are retained only under the documented logging-loss convention; longer runs are truncated at the one-billion-token horizon. Results report the mean and standard deviation across completed replicates. We make no statistical-significance claim.

L2C4 is the compact reference configuration used for direct two-layer comparisons. L3C4 is the best observed configuration in both tasks. The shape sweep is reported in full rather than using rhetoric to designate either as a universal primary model.

== Store--Distract--Query

#par-heading[Task]
SDQ is an online synthetic task designed to isolate associative storage and retrieval under interference. With five keys and ten values, inputs comprise 50 key--value store tokens, ten distractor tokens, and five query tokens. A store overwrites the value associated with its key. Distractor values contribute to a running sum modulo ten. At a query, the target combines the current stored value, the distractor sum, and the configured per-key store and query counts. Cross-entropy is applied only at query events.

Episode lengths are geometrically distributed. During training, their mean increases from 10 toward 500, while store and query probabilities decrease from 0.35 toward 0.10 and 0.25. Models use 512 parallel streams, a truncation length of 32, RMSprop, gradient clipping at norm 1, and the same learning-rate schedule and online generator.

#par-heading[Metric]
The code logs `Acc++` as query accuracy on examples whose valid store--query gaps are above the current batch mean. This emphasizes the longer-gap half of non-missing queries. For each completed replicate, we average the final five logged `Acc++` values through the one-billion-token horizon and then aggregate those replicate-level averages.

#wide-table(
  caption: [Matched SDQ results at the one-billion-token protocol. Values are mean $plus.minus$ standard deviation of replicate-level final-five `Acc++` averages. Runs slightly short of the horizon follow the logging-loss convention.],
)[
  #table(
    columns: (1.5fr, 0.8fr, 0.85fr, 0.65fr, 1.15fr, 1.3fr),
    stroke: none,
    inset: (x: 5pt, y: 2pt),
    align: (left, center, right, center, center, right),
    table.hline(),
    [*Family*], [*Shape*], [*Params*], [*$n$*], [*Protocol*], [*Final-five Acc++ ↑*],
    table.hline(),
    [GRU], [L1], [10.18M], [3], [1B tokens], [$0.6177 plus.minus 0.0052$],
    [GRU], [L2], [10.06M], [2], [1B tokens], [$0.5322 plus.minus 0.1027$],
    [GRU], [L3], [10.25M], [2], [1B tokens], [$0.2865 plus.minus 0.0798$],
    table.hline(),
    [MoSAIC], [L1C8], [10.12M], [2], [1B tokens], [$0.7116 plus.minus 0.0167$],
    [MoSAIC], [L2C4], [10.12M], [3], [1B tokens], [$0.8433 plus.minus 0.0055$],
    [MoSAIC], [L2C8], [10.18M], [1], [1B tokens], [$0.8678$],
    [MoSAIC], [L2C16], [10.09M], [1], [1B tokens], [$0.8901$],
    [MoSAIC], [L3C4], [9.99M], [3], [1B tokens], [*$0.9204 plus.minus 0.0128$*],
    table.hline(),
  )
] <tab-sdq>

#par-heading[Results]
@tab-sdq shows a consistent family-level advantage for MoSAIC: every evaluated shape has a higher central `Acc++` value than every GRU depth. In the replicated two-layer comparison, L2C4 reaches $0.8433 plus.minus 0.0055$, while GRU-L2 reaches $0.5322 plus.minus 0.1027$. L3C4 is the best observed configuration at $0.9204 plus.minus 0.0128$. The single-replicate L2C8 and L2C16 entries describe observed runs but provide less evidence about run-to-run variability.

== Character-Level Modeling

#par-heading[Data and training]
Text8 contains 100 million normalized English characters with a 27-character alphabet. We use the first 90 million characters for training and the final 10 million as a contiguous validation split. The validation split was used for limited manual configuration tuning, so we call the metric validation BPC rather than an untouched test estimate.

GRU and MoSAIC models process 512 parallel contiguous streams with truncated backpropagation through time over 64 steps. Random state resets decay from probability $0.2/64$ toward $0.001/64$ per token, increasing the expected uninterrupted context from about 320 to 64,000 characters. Models use RMSprop, a learning rate warming to $5 times 10^{-4}$ and decaying toward $5 times 10^{-5}$, and gradient clipping at norm 1. Validation runs disable random resets and evaluate contiguous streams.

The Transformer uses the same tokenization, data split, reset schedule, optimizer, token batch per update, and one-billion-token budget. It processes each 64-token training segment in parallel and carries a rolling valid key/value cache between segments. We evaluate cache lengths of 256 and 64 tokens as distinct configurations.

#wide-table(
  caption: [Fixed-horizon text8 results. BPC is taken at the final logged validation point through the one-billion-token protocol; lower is better. The Transformer cache variants are distinct configurations, not replicates.],
)[
  #table(
    columns: (1.35fr, 0.9fr, 0.85fr, 0.65fr, 1.15fr, 1.25fr),
    stroke: none,
    inset: (x: 5pt, y: 2pt),
    align: (left, center, right, center, center, right),
    table.hline(),
    [*Family*], [*Shape / cache*], [*Params*], [*$n$*], [*Protocol*], [*Validation BPC ↓*],
    table.hline(),
    [GRU], [L1], [10.16M], [3], [1B tokens], [$1.5626 plus.minus 0.0028$],
    [GRU], [L2], [10.04M], [3], [1B tokens], [$1.5004 plus.minus 0.0119$],
    [GRU], [L3], [10.23M], [3], [1B tokens], [$1.4828 plus.minus 0.0062$],
    table.hline(),
    [MoSAIC], [L1C8], [10.11M], [2], [1B tokens], [$1.4709 plus.minus 0.0033$],
    [MoSAIC], [L2C4], [10.11M], [3], [1B tokens], [$1.4367 plus.minus 0.0026$],
    [MoSAIC], [L2C8], [10.17M], [2], [1B tokens], [$1.4444 plus.minus 0.0017$],
    [MoSAIC], [L2C16], [10.09M], [2], [1B tokens], [$1.4548 plus.minus 0.0023$],
    [MoSAIC], [L3C4], [9.99M], [3], [1B tokens], [*$1.4345 plus.minus 0.0017$*],
    table.hline(),
    [Transformer], [cache 256], [10.07M], [3], [1B tokens], [$1.4492 plus.minus 0.0120$],
    [Transformer], [cache 64], [10.07M], [3], [1B tokens], [$1.4826 plus.minus 0.0057$],
    table.hline(),
  )
] <tab-text8>

#par-heading[Results]
As shown in @tab-text8, every evaluated MoSAIC shape has a lower central BPC than every matched GRU. The compact L2C4 reference reaches $1.4367 plus.minus 0.0026$, compared with $1.5004 plus.minus 0.0119$ for GRU-L2. L3C4 is the best observed shape at $1.4345 plus.minus 0.0017$; the strongest GRU, L3, reaches $1.4828 plus.minus 0.0062$. The cache-256 Transformer reaches $1.4492 plus.minus 0.0120$ under the same token budget. Thus the observed MoSAIC means are lower while retaining recurrent fixed-state operation; this comparison does not establish general Transformer superiority or long-context scaling.

== Topology and State Allocation

Parameter matching changes hidden width and persistent-state size. @tab-topology isolates those differences for the recurrent families. At two layers, increasing MoSAIC from four to eight and sixteen columns increases state size while worsening text8 BPC relative to L2C4. L2C16 stores more than twice as many recurrent scalars as L2C4 but obtains a higher BPC. On SDQ, the central values increase across these shapes, but L2C8 and L2C16 each have only one completed run. Depth also behaves differently across families: L3C4 improves over L2C4 on both tasks, whereas deeper GRUs improve on text8 but not SDQ. These observations show that state volume alone does not explain the results and that the allocation across depth, column count, and within-column width matters.

#figure(
  table(
    columns: (1.1fr, 0.75fr, 0.85fr, 1.05fr),
    stroke: none,
    inset: (x: 3pt, y: 2pt),
    align: (left, right, right, right),
    table.hline(),
    [*Shape*], [*$H$*], [*State*], [*Params*],
    table.hline(),
    [GRU-L1], [1,296], [1,296], [10.16M],
    [GRU-L2], [912], [1,824], [10.04M],
    [GRU-L3], [752], [2,256], [10.23M],
    table.hline(),
    [L1C8], [440], [3,520], [10.11M],
    [L2C4], [424], [3,392], [10.11M],
    [L2C8], [312], [4,992], [10.17M],
    [L2C16], [224], [7,168], [10.09M],
    [L3C4], [344], [4,128], [9.99M],
    table.hline(),
  ),
  caption: [Recurrent topology and persistent-state scalars per example for text8 configurations. SDQ uses the same hidden widths and shapes, with small parameter-count differences from its input and output vocabularies.],
  placement: top,
  kind: table,
) <tab-topology>

== Reduced-Token Baselines

HGRN2, DeltaNet, and mLSTM exceed the available accelerator memory at the standard 512-stream geometry. They were therefore trained with fewer streams, smaller token batches per update, reduced token budgets, and more optimizer updates than the one-billion-token runs. @tab-reduced reports these runs as a separate implementation-level reference. They are not equal-token comparisons and do not support claims about relative resource or optimization efficiency.

#figure(
  table(
    columns: (0.95fr, 0.65fr, 0.45fr, 0.75fr, 1.35fr),
    stroke: none,
    inset: (x: 3pt, y: 2pt),
    align: (left, right, right, right, right),
    table.hline(),
    [*Model*], [*Tokens*], [*$n$*], [*Updates*], [*Final BPC ↓*],
    table.hline(),
    [HGRN2], [100M], [2], [48.8k], [$1.6675 plus.minus 0.0076$],
    [DeltaNet], [200M], [2], [48.8k], [$1.8280 plus.minus 0.0231$],
    [mLSTM], [200M], [1], [48.8k], [$1.6856$],
    table.hline(),
  ),
  caption: [Reduced-token text8 baselines. HGRN2 uses 2,048 tokens/update; DeltaNet and mLSTM use 4,096. Standard GRU, MoSAIC, and Transformer runs use 32,768 tokens/update and approximately 30.5k updates.],
  placement: top,
  kind: table,
) <tab-reduced>

= Discussion and Limitations

Across the controlled comparisons, MoSAIC's columnar topology is associated with better associative-memory and character-modeling results than monolithic recurrent baselines at approximately fixed parameter scale. The pattern is consistent across the evaluated shapes rather than depending only on the best run family member. The text8 result relative to the finite-context Transformer provides additional context: MoSAIC retains recurrent fixed-state operation and obtains a lower mean BPC under the present matched token budget.

These results support a narrow conclusion. They do not show that columns acquire identifiable functional roles, that dense routing behaves as sparse expert selection, or that MoSAIC replaces Transformers generally. The architecture permits incremental inference with state size independent of processed sequence length, but the experiments do not establish superior long-context scaling or long-horizon retention against all alternatives.

Parameter matching also leaves important resources unmatched. MoSAIC exposes more persistent state coordinates than the GRUs and performs routing computation at every layer and timestep. We have not completed controlled throughput, peak-memory, or inference-latency measurements, so the results should be interpreted as evidence about an architectural inductive bias rather than resource efficiency. Likewise, the reduced-token recurrent baselines use different token and update budgets and are not direct quality comparisons.

The empirical scope is limited. Text8 is a small character-level corpus, its validation split received limited manual tuning, and the study operates at roughly ten million parameters. SDQ is synthetic and uses a curriculum-controlled online generator. Some topology entries have only one or two completed runs, and we do not make statistical-significance claims. Training-only routing regularizers add hyperparameters whose individual effects have not been isolated. Finally, the present paper omits reinforcement-learning results because the task-matched evidence is incomplete and variable.

= Conclusion

MoSAIC factorizes a monolithic recurrent state into persistent columns and learns how information is routed among them before independent recurrent updates. Under controlled, approximately parameter-matched one-billion-token training, this structural bias consistently improves associative memory and character-level modeling over monolithic GRUs. A similarly sized finite-context Transformer provides a matched Text8 reference, while the separate reduced-token results are reported only as non-comparable implementation context. The evidence supports modular state organization as a useful inductive bias for recurrent computation at this scale; broader claims about specialization, long-context scaling, and resource efficiency remain open.

#bibliography(
  "refs.bib",
  style: "american-psychological-association",
  title: "References",
)

] // end columns
