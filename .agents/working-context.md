# Working context

Last refreshed: 2026-08-01. This file describes frontier work on `main`; AAAI-27 rebuttal and submission follow-up belong on the `aaai-27` branch.

## Branch ownership and current stage

- The current branch is `main`, based on commit `047c7d2`. The working tree was clean before this context refresh.
- Use `main` for further architecture, research, and infrastructure development that is not driven by the AAAI submission. Use `aaai-27` for rebuttal, paper revisions, supplementary follow-up, and other submission-specific changes. Do not make rebuttal edits on `main` merely because the paper files remain present here.
- Repository references to “AAAI 2026” are a known stale year-labeling error; the submitted target is AAAI-27.

## Frontier research direction

Knitwork studies modular recurrent architectures, especially MoSAIC/Grid RNN variants, and how explicit organization of persistent state and communication can change learning and computation. The frontier agenda is broader than the submitted paper and may include heterogeneous modules, structured recurrent loops, multimodal systems, reinforcement learning, adaptive routing, different state geometries and timescales, local learning signals, and agent-oriented architectures.

The newest conceptual starting point is `_supp/neuro_bias.typ`, “Weak Neuroanatomical Biases for Modular Recurrent Systems.” It is an early design-space document, not a selected architecture, implementation plan, or experimental program. Follow its interpretive rules: biological names are handles rather than contracts; mention is not commitment; functions may arise from multi-area loops; mechanisms should remain multiply realizable; and ambiguity should be explored before forcing implementation decisions.

Other relevant starting points are `_supp/high_level_directions.typ`, `_supp/rnn_attn.typ`, and `_supp/fund_agents_proposal.typ`. `.agents/research-context.md` is opt-in and currently contains pointers rather than curated conclusions; update it only when the user explicitly asks to preserve research discussion.

No single frontier implementation or experiment is currently selected as the next priority. Ask the user before converting the conceptual design space into an architecture, implementation sequence, or expensive experiment program.

## Established architecture baseline

- Unified model registry: `knitwork/models/utils.py`.
- Current MoSAIC implementation: `knitwork/models/grnn_core.py`, class `GridRnn`. Legacy `knitwork/models/grnn.py` is not authoritative for the submitted or current core architecture.
- Monolithic GRU: `knitwork/models/gru.py`, class `GruCore`.
- Implemented comparison cores: `knitwork/models/baseline/delta_net.py`, `knitwork/models/baseline/hgrn2.py`, `knitwork/models/baseline/mlstm.py`, and `knitwork/models/baseline/transformer.py`.
- Established experiment protocols use `knitwork/exps/text/config/large.yaml`, `knitwork/exps/sdq/config/large.yaml`, and `knitwork/exps/mikasa/config/large.yaml` with their adjacent runners. Code and configuration are authoritative for exact behavior.

The established MoSAIC mechanism factorizes recurrent state into persistent columns and uses dense attention-mediated routing among a fixed set of messages before independent recurrent updates. Attention is not applied over the growing token history, so the carried recurrent state remains fixed in size as a sequence grows. This is an architectural property, not established evidence of superior long-context scaling.

## Historical AAAI evidence

`docs/experiments/results_aaai.md` is the frozen shared snapshot for the AAAI experiments, retrieved from Comet on 2026-07-29 at 11:45 UTC. Use it when discussing or reproducing those experiments; do not query Comet or reinterpret tracker state without explicit user authorization. New frontier experiments should get their own purpose, protocol, and evidence records rather than silently extending the AAAI snapshot.

The AAAI evidence supports a narrow historical conclusion: under approximately parameter-matched 1B-token training, the tested MoSAIC topologies improved SDQ associative-memory accuracy and Text8 BPC over monolithic GRUs, while the best observed MoSAIC means were also lower than the matched finite-context Text8 Transformer mean. The evidence does not establish state of the art, specialization, sparse/MoE behavior, resource efficiency, general Transformer replacement, long-context superiority, or robust RL superiority.

Key historical values:

- Text8: MoSAIC-L2C4 `1.4367 ± 0.0026`, MoSAIC-L3C4 `1.4345 ± 0.0017`, GRU-L2 `1.5004 ± 0.0119`, GRU-L3 `1.4828 ± 0.0062`, and Transformer cache-256 `1.4492 ± 0.0120` validation BPC.
- SDQ: MoSAIC-L2C4 `0.8433 ± 0.0055`, MoSAIC-L3C4 `0.9204 ± 0.0128`, GRU-L2 `0.5465 ± 0.0768`, and GRU-L3 `0.3241 ± 0.0862` final-five `Acc++`; all listed groups have three completed runs.
- Reduced-budget HGRN2, DeltaNet, and mLSTM runs used different token, batch, and update budgets and are not direct quality or efficiency comparisons.
- Completed Mikasa runs were mixed: HigherLowerMedium showed a small uncertain final-window difference, while RepeatFirstEasy exposed MoSAIC policy collapse after near-perfect peaks. Treat RL stability as an open problem, not supporting evidence of superiority.

For the historical reporting conventions, aggregation rules, per-run identifiers, reduced-budget accounting, and full topology sweep, use `docs/experiments/results_aaai.md` rather than expanding this short context.

## Frontier research principles and open directions

- Treat the AAAI paper's contribution boundaries as evidence boundaries, not prohibitions on future research. Specialization, multimodal fusion, sparse communication, long-context behavior, efficiency, and agent-oriented modularity are open hypotheses that require new evidence.
- Preserve the distinction between architectural organization and semantic assignment. A weak bias may constrain connectivity, state geometry, timescale, update dynamics, learning signals, or control without preassigning a module's learned meaning.
- Consider loops and channels, not only named modules, as possible units of organization. Content, routing, prediction error, value, salience, modulation, and write permission may require distinct signal types and timescales.
- Avoid collapsing rapid adaptation into fast weights by default; persistent activation, writable state, external memory, temporary plasticity, ordinary parameters, and hybrid mechanisms remain alternatives.
- Future empirical candidates include component ablations, column specialization analysis, multimodal SDQ, controlled long-context tests, throughput/memory/latency measurement, RL-stability investigation, and broader agent tasks. These are candidates, not an approved queue.

## Project pointer

Use code/configuration for exact behavior, `docs/methods/` for model summaries, `docs/experiments/` for experiment context, `_supp/` for exploratory design-space material, and `.agents/research-context.md` only for research discussion the user explicitly asks to preserve.
