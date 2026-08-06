# Working context

Last refreshed: 2026-08-06. This branch is dedicated to the AAAI-27 paper, rebuttal preparation, paper-facing experiments, and subsequent submission revisions.

## Branch scope and current stage

- The AAAI-27 main paper, reproducibility checklist, and source-code supplement were submitted. Repository references to “AAAI 2026” are a known stale year-labeling error; the target is AAAI-27.
- Commit `86990dd` is the repository's recorded “final submitted version.” Treat the uploaded artifacts as the historical submission baseline and make later paper changes explicitly rather than silently rewriting that record.
- Use this branch for rebuttal analysis, paper corrections, improved presentation, additional experiments that could materially support the paper, and later AAAI revisions.
- Large architectural departures and open-ended conceptual work around weak neuroanatomical biases belong on `main`, not here, unless the user explicitly decides that a specific result is needed for the paper.

## Paper artifacts and submitted scope

- Editable paper authority: `article/latex/paper.tex`. The Typst draft is historical and should not be treated as the current manuscript.
- The paper uses the official AAAI-27 LaTeX author kit and compiles to `article/latex/paper.pdf`. The reproducibility checklist is `article/latex/ReproducibilityChecklist.tex` and compiles separately to `article/latex/ReproducibilityChecklist.pdf`.
- `article/latex/fig_architecture.pdf` is the mechanism diagram used in the paper. It explains the time/depth grid, per-column recurrent update, and router; it is not evidence of functional specialization.
- `article/latex/fig_learning_curves.pdf` remains a useful revision/supplement artifact but was removed from the submitted main paper because its text did not satisfy AAAI's 9-point minimum at final rendered scale. Before reuse, regenerate it from the current evidence snapshot with every label, tick, title, and legend entry at least 9 pt after LaTeX scaling, then render and inspect the paper PDF.
- The submitted empirical scope is SDQ and Text8 under the standard protocol, with GRU and finite-context Transformer references. HGRN2, DeltaNet, and mLSTM were described only as memory-constrained, non-comparable implementation references. Mikasa RL results were omitted because completed runs showed unstable final performance.

## Positioning brief

### High-level motivation

The motivating problem is how recurrent computation is organized, not primarily how to extend nominal context length. A conventional RNN compresses memory and computation into a monolithic hidden-state stream. Increasing width or depth adds capacity but does not create an explicit topology in which persistent state-bearing components interact.

MoSAIC introduces modularity as an architectural inductive bias. It factorizes hidden state into persistent recurrent columns and lets information move among them through learned attention-mediated routing: divide state and computation, then learn how the parts communicate.

MoSAIC arose within a broader agent-centric research motivation involving multimodality, specialization, perception/planning/control subsystems, and lifelong or multitask agents. On this branch, that background is only a claim boundary: the submitted paper does not demonstrate those capabilities, and the broader architecture program is not an active AAAI workstream.

### Paper-scoped thesis

> MoSAIC treats modular state organization as an inductive bias for recurrent computation. It factorizes a monolithic hidden state into persistent columns and learns how information is routed among them. Under controlled, approximately parameter-matched training, this structural bias consistently improves associative memory and character-level modeling over monolithic recurrent baselines.

The Text8 Transformer result is supporting context: MoSAIC retains recurrent fixed-state operation while achieving lower mean Text8 BPC than a similarly sized finite-context Transformer under the matched token budget. This observation does not establish general Transformer replacement or superior long-context scaling.

### Supported contributions

1. A layered grid of persistent recurrent columns with content-dependent, attention-mediated inter-column communication.
2. Column count and depth as explicit axes for organizing recurrent state and computation at roughly fixed parameter scale.
3. Matched GRU/MoSAIC evidence on SDQ and Text8, plus a matched-token Text8 Transformer reference.

### Architectural property versus demonstrated advantage

Attention operates over a fixed collection of recurrent columns rather than an ever-growing token history. The model therefore carries a fixed-size recurrent state as sequence length grows and supports incremental inference. This is an architectural property. The current experiments do not establish superior long-context scaling or long-horizon retention against all alternatives.

### Explicit non-claims

Do not claim emergent column specialization, bound/free-column behavior, multimodal fusion, sparse or mixture-of-experts computation, state of the art, general Transformer replacement, resource efficiency, robust RL superiority, superiority over the reduced-budget external baselines under equal budgets, or empirically demonstrated long-context scaling. Use “attention-mediated communication” or “learned routing” for the implemented dense mechanism. Treat `_supp/` hypotheses as exploratory motivation or future work, not evidence.

### Relation to prior work

Do not claim to invent modular recurrence in general. Relevant neighbors include Recurrent Independent Mechanisms, BRIMs/shared workspaces, Relational Memory Core, Grid LSTM and multidimensional recurrence, and parallel-cell RNNs. MoSAIC's intended distinction is a regular layered topology of persistent recurrent columns with learned inter-column communication. Verify exact comparisons and publication metadata before future revisions.

## Evidence authority and reporting conventions

- `docs/experiments/results_aaai.md` is the shared quantitative evidence snapshot for the submission experiments. Its current retrieval time is 2026-07-29 11:45 UTC. Ordinary agents should use it rather than querying Comet or copying remembered values.
- Only a task explicitly designated by the user may query Comet and refresh the snapshot. Code/configuration is authoritative for implementation and protocol details.
- Standard GRU/MoSAIC comparisons use a 1B-token horizon. Truncate longer runs at 1B. Runs slightly short of 1B may count under the documented network-logging-loss convention, using their final logged point.
- Text8 fixed-horizon values and best-validation checkpoints are distinct views and must be labeled separately.
- SDQ is generated online and has no held-out validation split. For each run, average the final five logged `Acc++` values at the reporting horizon, then compute the mean and sample standard deviation across run-level averages. Do not use peak `Acc++` as the primary result.
- Mikasa RL uses the analogous final-five `env/EpRet` aggregation. Peak return is diagnostic only because several policies later collapsed.
- Reduced-budget HGRN2, DeltaNet, and mLSTM runs use different token, batch, and update budgets. Keep them separate and state all three quantities; do not make direct quality or efficiency claims from them.
- In public-facing paper text and plots, use “MoSAIC” rather than internal `GRNN/grnn` names and “GRU” rather than `rnn`.

## Authoritative implementation and protocol map

- Unified model registry: `knitwork/models/utils.py`.
- MoSAIC: `knitwork/models/grnn_core.py`, class `GridRnn`. Do not use legacy `knitwork/models/grnn.py` to describe the submitted architecture.
- GRU: `knitwork/models/gru.py`, class `GruCore`.
- DeltaNet: `knitwork/models/baseline/delta_net.py`, class `DeltaNetCore`.
- HGRN2: `knitwork/models/baseline/hgrn2.py`, class `HGRN2Core`.
- mLSTM: `knitwork/models/baseline/mlstm.py`, class `mLSTMCore`.
- Transformer: `knitwork/models/baseline/transformer.py`, class `TransformerCore`.
- Submitted Text8 protocol: `knitwork/exps/text/config/large.yaml` with `knitwork/exps/text/run.py`; Transformer uses `knitwork/exps/text/run_offline.py` with the same large configuration.
- Submitted SDQ protocol: `knitwork/exps/sdq/config/large.yaml` with `knitwork/exps/sdq/run.py`.
- Exploratory Mikasa protocol: `knitwork/exps/mikasa/config/large.yaml` with `knitwork/exps/mikasa/run.py`.

## Final experiment snapshot

### Text8

- MoSAIC-L2C4: `1.4367 ± 0.0026` validation BPC, `n=3`.
- MoSAIC-L3C4: `1.4345 ± 0.0017`, `n=3`.
- GRU-L2: `1.5004 ± 0.0119`, `n=3`.
- GRU-L3: `1.4828 ± 0.0062`, `n=3`.
- Transformer cache 256: `1.4492 ± 0.0120`, `n=3`.
- Transformer cache 64: `1.4826 ± 0.0057`, `n=3`; this is a separate context variant, not a replicate of cache 256.
- MoSAIC-L2C16 has `n=2`; the other principal recurrent topology groups have three runs. See the snapshot for the full sweep.

### Store–Distract–Query

- MoSAIC-L2C4: `0.8433 ± 0.0055` final-five `Acc++`, `n=3`.
- MoSAIC-L3C4: `0.9204 ± 0.0128`, `n=3`.
- MoSAIC-L2C16: `0.8911 ± 0.0049`, `n=3`.
- GRU-L2: `0.5465 ± 0.0768`, `n=3`.
- GRU-L3: `0.3241 ± 0.0862`, `n=3`.
- All standard SDQ topology groups in the final snapshot have three completed runs.

### Reduced-budget baselines

- Text8: DeltaNet `1.8280 ± 0.0231` at 200M tokens, HGRN2 `1.6675 ± 0.0076` at 100M, and mLSTM `1.6797 ± 0.0084` at 200M; each has `n=2` and 48.8k planned updates.
- SDQ: DeltaNet `0.1529 ± 0.0317` at 250M tokens, HGRN2 `0.1095 ± 0.0004` at 125M, and mLSTM `0.1239 ± 0.0037` at 250M; each has `n=2` and 61.0k planned updates.
- These are not directly comparable with the standard 1B-token runs because batch sizes, token budgets, and update counts differ.

### Mikasa RL

- HigherLowerMedium, three 30M-step runs per family: MoSAIC-L2C4 `0.3577 ± 0.0718`, GRU-L2 `0.2968 ± 0.0363`. The difference is uncertain; GRU reached the higher diagnostic peak.
- RepeatFirstEasy: MoSAIC-L2C4 `0.6921 ± 0.2428`, `n=4`; GRU-L2 `0.9962 ± 0.0037`, `n=3`. Every MoSAIC run reached a diagnostic peak near `0.996`, but several later collapsed, indicating training instability.
- No completed RepeatFirstMedium comparison is in the snapshot. Do not claim RL superiority from this evidence.

## Possible paper improvements and supporting experiments

These are candidates for rebuttal preparation or later paper revisions, not a committed work queue. Prioritize them only after considering likely reviewer questions, evidential value, runtime, and risk.

- Regenerate the learning-curve figure with final three-run groups and AAAI-compliant text size before reintroducing it.
- Decide whether the reduced-budget table and an update-indexed diagnostic plot belong in supplementary material; retain explicit non-comparability language.
- Improve reproducibility coverage for random seeds, hardware/software environment, hyperparameter search history, final configuration tables, and clean standalone checklist compilation.
- Update published-venue metadata for references still listed as arXiv preprints, including Mamba-2, DeltaNet, and xLSTM.
- In future paper revisions, describe the Mikasa runs as completed but unstable rather than incomplete or merely preliminary.
- Investigate MoSAIC's RL collapse and complete stronger task-matched comparisons before presenting RL as supporting breadth.
- Consider multimodal SDQ, routing/regularizer component ablations, measurable column-specialization analysis, throughput and memory measurements, controlled long-context tests, and better-matched modern recurrent baselines as possible additions to the paper's evidence.
- Do not use this branch for the larger neuroanatomical-bias architecture program or unrelated frontier redesigns; those belong on `main` unless explicitly pulled into the paper's scope.

## Project pointer

Knitwork studies MoSAIC (Modular Self-Attentive Interacting Columns for Recurrent Memory): recurrent columns maintain local state and communicate through learned attention routing. Use code/configuration for exact behavior, `docs/methods/` for model summaries, `docs/experiments/` for experiment context, and `_supp/` only as exploratory research material.
