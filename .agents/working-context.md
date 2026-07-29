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
- The manuscript has not yet been brought into submission shape. No main-text
  revision or submission/compliance work is complete.
- A new GRNN visualization scheme has been prepared; its final paper use and
  caption still need to be decided.
- The dirty draft may contain reusable material, especially related work, but
  every claim and citation must be checked. Do not inherit obsolete
  Grid-RNN/Harmonic-GRNN framing or quantitative claims.
- Results must improve a paper that remains submittable without unfinished RL
  runs. Mikasa is optional if it cannot be made credible and integrated safely.

### Immediate work order

1. Freeze the paper’s central claim, outline, and evidence/figure/table set.
2. Inventory the current Typst source against AAAI page, anonymity, references,
   and submission requirements.
3. Draft or repair the minimum complete paper using only verified evidence.
4. Integrate results and the new architecture visualization; compile and
   inspect the PDF.
5. Reserve final time for page-limit reduction, citation/anonymity checks,
   upload, and submission validation.

Experiments continue asynchronously. Review them only at bounded checkpoints;
do not wait for them before writing.

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

- Exact central claim and positioning of MoSAIC.
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
