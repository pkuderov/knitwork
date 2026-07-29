# Working context

Last refreshed: 2026-07-29, with about **3 hours remaining**.

## AAAI-27 deadline

- Target: **AAAI-27**. Repository and manuscript references to “AAAI 2026” are
  a known year-labeling error.
- Hard deadline: **2026-07-29 15:00 Europe/Moscow**.
- The paper and submission mechanics are now the critical path. Do not start
  new experiment branches or spend user attention optimizing seed coverage.
- Multimodal SDQ is deferred.

## Paper state

- Editable authority: `article/typst/paper_en.typ`.
- Read `article/AGENTS.md` before paper work and preserve anonymity and
  submission formatting.
- A coherent first manuscript iteration now exists and compiles to
  `article/typst/paper_en.pdf`: five rendered pages including references under
  the temporary custom Typst layout. The central argument is conservative and
  aligned with the positioning brief; Mikasa is correctly omitted for now.
- The paper is not yet submission-ready. The source manually approximates AAAI
  formatting rather than using the official AAAI-27 author kit, and pagination
  may change during template migration.
- `fig_architecture.pdf` is included as a mechanism diagram, but it is currently
  placed at single-column width and its detailed router labels are too small.
  Make it span both columns or simplify/crop it.
- The dirty draft may contain reusable material, especially related work, but
  every claim and citation must be checked. Do not inherit obsolete
  Grid-RNN/Harmonic-GRNN framing or quantitative claims.
- Results must improve a paper that remains submittable without unfinished RL
  runs. Mikasa is optional if it cannot be made credible and integrated safely.

## Positioning brief for writing agents

### High-level motivation

The motivating problem is **how recurrent computation is organized**, not
primarily how to extend its nominal context length. A conventional RNN
compresses memory and computation into a single monolithic hidden-state stream.
Increasing its width or depth adds capacity but does not give the model an
explicit topology in which different state-bearing components can interact.

MoSAIC introduces **modularity as an architectural inductive bias**. It
factorizes hidden state into persistent recurrent columns and lets information
move among them through learned attention-based routing. The core intuition is
“divide state and computation, then learn how the parts communicate,” rather
than forcing every role through one undifferentiated recurrent vector.

The original research vision was broader and agent-centric: bound and free
columns, multimodal inputs and outputs, emergent functional specialization,
perception/planning/control subsystems, auxiliary objectives, feedback paths,
and lifelong or multitask agents. This vision can motivate the direction or
appear briefly in discussion, but **it is not what the present paper
demonstrates**.

### Paper-scoped thesis

Use the following as the central thesis:

> MoSAIC treats modular state organization as an inductive bias for recurrent
> computation. It factorizes a monolithic hidden state into persistent columns
> and learns how information is routed among them. Under controlled,
> approximately parameter-matched training, this structural bias consistently
> improves associative memory and character-level modeling over monolithic
> recurrent baselines.

The Text8 Transformer result is supporting context, not the definition of the
paper:

> MoSAIC retains recurrent fixed-state operation while achieving lower mean
> Text8 BPC than a similarly sized finite-context Transformer under the current
> matched token budget.

Phrase this as the observed result, not as proof that MoSAIC generally replaces
Transformers or scales better to long contexts.

### Supported contribution structure

1. **Architecture:** a layered grid of persistent recurrent columns with
   content-dependent, attention-mediated inter-column communication.
2. **Controlled topology:** column count and depth provide explicit axes for
   organizing recurrent state and computation at roughly fixed parameter
   scale.
3. **Empirical evidence:** matched RNN/GRNN comparisons on SDQ and Text8 show
   consistent benefits from the modular topology; Text8 additionally includes
   the finite-context Transformer comparison.
4. **Breadth, if defensible:** Mikasa may show that the architecture functions
   in online RL, but it is preliminary supporting evidence and not required for
   the main thesis.

The new `article/typst/fig_architecture.pdf` is a **mechanism diagram**. It
explains the time/depth grid, per-column recurrent update, and routing
calculation. It is not behavioral evidence that columns specialize.

### Architectural property versus demonstrated advantage

Attention is applied among a fixed collection of recurrent columns rather than
over an ever-growing token history. Consequently, the model carries a
fixed-size recurrent state as sequence length grows and is compatible with
incremental/streaming inference. State this as an architectural property or
secondary motivation. The current experiments do **not** establish superior
long-context scaling or long-horizon retention relative to all alternatives.

### Explicit non-claims

Do not claim:

- emergent column specialization or identifiable functional roles;
- bound/free-column behavior or multimodal fusion;
- sparse, conditional, or mixture-of-experts computation;
- state of the art or general Transformer replacement;
- superiority over reduced-budget external baselines under equal token
  budgets;
- robust RL superiority from the current Mikasa evidence;
- empirically demonstrated long-context scaling.

Use “attention-mediated communication” or “learned routing” for the implemented
dense mechanism. Do not imply sparse expert selection. Treat hypotheses from
`_supp/` as motivation/future work, never as established results.

### Conceptual relation to prior work

Do not claim to invent modular recurrence in general. The closest conceptual
neighbors include Recurrent Independent Mechanisms, Relational Memory Core,
Grid LSTM/multidimensional recurrent models, and parallel-cell RNNs. The
intended distinction is the regular layered topology of persistent recurrent
columns with learned inter-column communication. Verify exact comparisons and
citations before writing them.

### Recommended narrative

1. Monolithic recurrence provides persistent state but no explicit modular
   organization.
2. MoSAIC supplies a columnar state topology plus learned communication.
3. SDQ tests whether this bias helps structured associative memory.
4. Text8 tests whether the benefit transfers to natural sequential prediction,
   with RNN and Transformer references.
5. Analysis/topology results show how depth and column organization matter.
6. RL, if retained, is preliminary evidence of online applicability.
7. Discussion returns to the broader modular-agent vision while clearly
   labeling it future work.

### Immediate work order

1. Apply the locked positioning above; freeze the outline and
   evidence/figure/table set.
2. Inventory the current Typst source against AAAI page, anonymity, references,
   and submission requirements.
3. Draft or repair the minimum complete paper using only verified evidence.
4. Integrate results and the new architecture visualization; compile and
   inspect the PDF.
5. Reserve final time for page-limit reduction, citation/anonymity checks,
   upload, and submission validation.

Experiments continue asynchronously. Review them only at bounded checkpoints;
do not wait for them before writing.

### First-iteration review priorities

1. Migrate/check against the official AAAI-27 author kit; font replacement
   alone is not a complete compliance check.
2. Make the architecture mechanism figure legible at final rendered size.
3. Specify the implemented communication loss, normalized entropy term,
   task-specific coefficients, and routing-noise values sufficiently for
   reproduction. The current generic objective hides meaningful training
   details.
4. Add a clean paper-specific learning-curve figure if time permits. Do not
   insert `aaai_comet_snapshot.png` unchanged because it contains interim runs
   and operational legends inconsistent with the paper aggregation rules.
5. Add the Text8 data citation and a compact Transformer architecture
   description; change the SDQ wording from “inputs comprise” to “the
   vocabulary comprises” when listing token types.

### Default main-paper result presentation

Use a clean matched-budget story in the main paper; do not place every tracked
model on the same plot.

1. Add one required full-width, two-panel learning-curve figure. Generate a
   vector PDF at `article/latex/fig_learning_curves.pdf`, approximately
   7.0 inches wide by 2.8--3.0 inches high. The exact figure title is
   **“Learning dynamics under the standard 1B-token protocol.”**

   **Panel (a):**
   - Panel title: **“text8 character modeling”**.
   - Series: `GRU-L2`, `GRU-L3`, `MoSAIC-L2C4`, `MoSAIC-L3C4`, and
     `Transformer-256`.
   - X axis label: **“Processed tokens (billions)”**; linear range
     `[0.0, 1.0]`, ticks at `0.0, 0.2, ..., 1.0`.
   - Y axis label: **“Validation BPC ↓”**; linear range approximately
     `[1.40, 2.65]`, using readable 0.2 increments. Do not use a log scale.

   **Panel (b):**
   - Panel title: **“Store–Distract–Query”**.
   - Series: `GRU-L1`, `GRU-L2`, `MoSAIC-L2C4`, and `MoSAIC-L3C4`.
   - X axis label and range: the same as panel (a).
   - Y axis label: **“Long-gap query accuracy (Acc++ ↑)”**; linear range
     `[0.0, 1.0]`, ticks at `0.0, 0.2, ..., 1.0`.

   **Data and aggregation:**
   - Use standard-protocol runs only and cap every curve at 1B processed
     tokens. Exclude reduced-budget HGRN2, DeltaNet, and mLSTM runs.
   - Use completed replicates only. Do not include interim runs or a
     single-replicate topology merely to add another line.
   - Align replicate curves on their common scheduled logging grid. Do not
     extrapolate through a missing tail or past the last logged point.
   - Plot the replicate mean and a shaded **±1 sample standard deviation**
     band at each shared point. State replicate count in the legend.
   - Text8 uses logged validation BPC, not training BPC or a best-checkpoint
     envelope. SDQ uses the logged `Acc++` trajectory; the final table, not the
     curve, applies the final-five aggregation rule.
   - Do not apply additional smoothing beyond the metric already stored by the
     experiment logger.

   **Names and style:**
   - All user-facing labels must say **MoSAIC**, never `GRNN`, `grnn`, or raw
     tracker names. Likewise use `GRU`, not `rnn`.
   - Use family colors consistently in both panels: GRU `#4C78A8`, MoSAIC
     `#F58518`, Transformer `#54A24B`.
   - Use line styles to distinguish topology: L1 dotted, L2/L2C4 dashed,
     L3/L3C4 solid, and Transformer-256 dash-dot. Use line width about 2 pt.
   - Draw uncertainty bands in the corresponding family color with alpha about
     `0.15`, no visible band edge. Use a light unobtrusive grid.
   - Use one shared legend centered below the two panels, in at most two rows.
     Labels should be, for example, `MoSAIC-L2C4 (n=3)`.
   - Use compact publication typography (roughly 8 pt labels/ticks and
     7--8 pt legend) compatible with the AAAI LaTeX figure width. Avoid large
     plotting-library default titles and excessive whitespace.

   Suggested LaTeX caption:
   **“Learning dynamics under the standard 1B-token protocol. Lines show the
   mean across completed replicates and shading shows ±1 standard deviation.
   (a) Validation BPC on text8. (b) Online long-gap query accuracy on SDQ.
   Final SDQ values in the table use the mean of each replicate's last five
   logged values.”**

   `Transformer-64` remains in the final-results table as a context variant but
   is omitted from the curve to reduce clutter.
2. Keep the current two task-specific full-width final-results tables for the
   standard protocol. They are readable and show the full topology sweep.
   Consolidate them only if the official template creates severe page
   pressure; a combined table would otherwise become unnecessarily dense.
   In every table and caption, rename the tracker/config family `grnn` to
   **MoSAIC** and `rnn` to **GRU**; raw internal names and Comet IDs must not
   appear.
3. Keep the compact topology/state-allocation table because it addresses the
   major parameter-versus-state confound.
4. Do not mix reduced-token HGRN2, DeltaNet, or mLSTM curves with the standard
   1B curves. Put their exact token/update/batch table and an update-indexed
   Text8 diagnostic plot in supplementary material if time permits. That
   optional supplementary plot should use optimizer updates (thousands) on the
   x axis, validation BPC on the y axis, and clearly label every model's token
   budget and batch size in the caption. In the main paper, mention these
   models only as separately budgeted implementation references.
5. Do not use `docs/experiments/figures/aaai_comet_snapshot.png` unchanged. It
   contains operational legends, interim runs, and too many series.

## Evidence source

- **All ordinary agents must use `docs/experiments/results_aaai.md` as the
  shared quantitative evidence snapshot.** Do not query Comet or substitute
  remembered numbers.
- Only the explicitly designated Comet-results task may access Comet and
  refresh the snapshot at the user’s request.
- Current snapshot retrieval time: **2026-07-29 07:32 UTC**.
- Comet can transiently label healthy jobs `finished` when server connectivity
  fails. Direct user-confirmed `tmux` observation overrides Comet only for live
  process state; the snapshot remains the authority for logged evidence.
- Code/configs are authoritative for implementation and protocol details.

### Authoritative implementation and configuration map

Do not use legacy `knitwork/models/grnn.py` or older experiment recipes to
describe the submitted model. The unified experiment registry is
`knitwork/models/utils.py`. Its active core implementations are:

- MoSAIC/GRNN: `knitwork/models/grnn_core.py` — `GridRnn`.
- Monolithic GRU: `knitwork/models/gru.py` — `GruCore`.
- DeltaNet: `knitwork/models/baseline/delta_net.py` — `DeltaNetCore`.
- HGRN2: `knitwork/models/baseline/hgrn2.py` — `HGRN2Core`.
- mLSTM: `knitwork/models/baseline/mlstm.py` — `mLSTMCore`.
- Transformer: `knitwork/models/baseline/transformer.py` —
  `TransformerCore`.

The submitted experiment scale/protocol is defined by each experiment's
`large.yaml`, not `base.yaml` or older standalone configs:

- Text8: `knitwork/exps/text/config/large.yaml`, run through
  `knitwork/exps/text/run.py`; Transformer uses
  `knitwork/exps/text/run_offline.py` with the same large config.
- SDQ: `knitwork/exps/sdq/config/large.yaml`, run through
  `knitwork/exps/sdq/run.py`.
- Mikasa: `knitwork/exps/mikasa/config/large.yaml`, run through
  `knitwork/exps/mikasa/run.py`.

## Current experiment evidence

### Text8

- The main 1B-token RNN/GRNN/Transformer comparison is complete enough for the
  paper, with three replicates for the principal configurations.
- Decision-relevant snapshot values:
  - GRNN-L2C4: `1.4367 ± 0.0026` validation BPC.
  - GRNN-L3C4: `1.4345 ± 0.0017`.
  - RNN-L2: `1.5004 ± 0.0119`.
  - RNN-L3: `1.4828 ± 0.0062`.
  - Transformer (`mem_len=256`): `1.4492 ± 0.0120`.
  - Transformer (`mem_len=64`): `1.4826 ± 0.0057`.
- The `mem_len=64` Transformer is a separate context-length variant, not a
  replicate of the `mem_len=256` model.
- DeltaNet, HGRN2, and mLSTM use a separate reduced-token/increased-update
  protocol. No third replicates are planned for these slow baselines.

### Store–Distract–Query

- SDQ is complete enough for the paper and strongly favors GRNN.
- Decision-relevant matched results:
  - GRNN-L2C4: `0.8433 ± 0.0055` final-five `Acc++`.
  - RNN-L2: `0.5322 ± 0.1027`.
  - GRNN-L3C4: `0.9204 ± 0.0128`.
- Reduced-budget DeltaNet, HGRN2, and mLSTM results are reported separately and
  are not directly comparable to the standard 1B protocol.

### Mikasa RL

- Core tasks: `RepeatFirstEasy`, `HigherLowerMedium`, and optionally
  `RepeatFirstMedium`.
- Core models: parameter-matched `rnn.L2` and `grnn.L2C4`.
- RL remains status-only until task-matched cells complete and a reporting
  convention is fixed.
- At the current snapshot:
  - `HigherLowerMedium`: GRNN has one completed run at current return `0.429`;
    RNN has one completed run at `0.250` and one running around `0.462`.
    Evidence is incomplete and variable.
  - `RepeatFirstEasy`: one RNN run completed stably near `1.0`. One GRNN run
    reached about `0.997` but later collapsed and ended early at 25.9M/30M with
    current return about `0.079`; another GRNN run was running near `0.995`.
  - No completed `RepeatFirstMedium` comparison is in the snapshot.
- Treat the GRNN collapse as genuine instability. Do not hide it by selecting
  peak return post hoc. Do not claim RL superiority from the present snapshot.
- HGRN2 RL is no longer deadline-critical; it requires batch/protocol changes.

## Result-reporting conventions

- RNN/GRNN primary horizon: **1B processed tokens/steps**. Truncate longer runs
  at 1B.
- Runs slightly short of 1B may count as the 1B protocol when the missing tail
  is known server/network logging loss; show the final logged point.
- Keep reduced-token/increased-update baselines separate and state tokens,
  updates, and batch accounting from configs.
- Text8 fixed-horizon and best-validation views must be labeled separately.
  Best checkpoints are searched only through the 1B horizon; unfinished runs
  are not silently included.
- SDQ uses an online generator, so no held-out validation split is required.
  For each completed replicate, average the final five logged `Acc++` values at
  the reporting horizon, then compute mean and standard deviation across those
  replicate-level averages. Do not use peak `Acc++` as the primary aggregate.

## Open decisions

- Which GRNN configuration is the primary model versus an architectural sweep.
- Whether and how Mikasa enters the main paper.
- Final architecture/results figures and tables.
- What material from the dirty draft is safe to retain.
- Page budget, supplementary-material scope, and submission checklist.

## Project pointer

Knitwork studies MoSAIC (Modular Self-Attentive Interacting Columns for
Recurrent Memory): recurrent columns maintain local state and communicate via
learned attention routing. Use code/configs for exact behavior,
`docs/methods/` for model summaries, `docs/experiments/` for experiment context,
and `_supp/` only as exploratory material.
