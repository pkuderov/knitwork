# Working context

Last refreshed: 2026-07-29.

## Active deadline: AAAI-27

The current target is **AAAI-27**, not AAAI 2026. Older references to
"AAAI 2026" throughout the repository are a known year-labeling mistake.

- Hard deadline: **2026-07-29 15:00 Europe/Moscow**.
- All submission work remains, including finishing the paper and the
  submission/compliance mechanics.
- The current paper is dirty, but may contain reusable material, especially
  related work; claims and citations still need rechecking.
- No agents are actively working at this snapshot. Existing Codex tasks retain
  context for Comet/result analysis, paper updates, Transformer rework, and RL
  pipeline rework.

### Minimum viable paper and experiment priority

The intended minimum paper has:

1. SDQ results.
2. Text8 results from the currently available recurrent baselines.
3. A small, credible Mikasa RL demonstration if it can be validated in time.

A Transformer result on Text8 is important, but the current judgment is that a
paper with an RL demonstration and no Transformer is more acceptable than the
reverse. Multimodal SDQ is deferred unless the critical path changes.

New results must improve a paper that remains submittable using already
available evidence; the submission should not depend on a risky unfinished run.

### Current experiment state

- SDQ, Text8, and Mikasa RL pipelines exist for RNN, GRNN/MoSAIC, HGRN2,
  DeltaNet, and mLSTM, but RL has not yet been checked for correctness.
- A Transformer Text8 pipeline is close: the expected fix is roughly 20--30
  minutes, followed by conversion to a faster batched offline experiment mode.
  With offline mode, training is expected to be fast for the intended small
  roughly 10M-parameter Transformer at context length 256 or 512, using either
  the 1B-token/data budget or the approximately 30k-update budget used for the
  recurrent models.
- As of the latest review, the Transformer implementation is ready for a test
  but has not been extensively reviewed by the user. The intended full run is
  1B tokens with rollout length 64, 512 environments, and `mem_len=256`,
  matching the recurrent runs in update count and total processed steps.
- RL is the highest-risk direction. A useful run is expected to take roughly
  1--4 hours on one GPU. One seed is accepted under the deadline constraint;
  one validated Mikasa task would already be useful, while two or three are
  aspirational.
- Most Text8 models have at least one completed run except Transformer.
  Additional known seeds: RNN L1 (2), RNN L2 (3), GRNN L2C4 (2), and GRNN
  L3C4 (2 completed plus 1 running at the initial report).
- Compute availability remains a risk, but the lab cloud is clearing and there
  is now a reasonable chance of enough machines. Three slower dedicated
  servers should become free within roughly 1--2 hours. Some lower-priority
  SDQ/Text8 seed-adding runs can be stopped if necessary.
- Additional SDQ/Text8 runs have just been submitted to obtain second or third
  seeds where possible.
- Live experiment truth is in Comet ML. `inference/comet_aaai_snapshot.py` is
  an AI-generated inspection script; the existing Comet-analysis task can be
  asked to download and refresh results.
- The refreshed snapshot is `docs/experiments/results_aaai.md`, retrieved
  2026-07-28 23:37 UTC. Its aggregate tables are exploratory, not directly
  paper-ready: their shared-horizon aggregation includes incomplete seeds, so
  a newly running seed can move the comparison far earlier in training and
  make the reported aggregate misleading.

### Open decisions

- Exact hour-by-hour critical path and ownership.
- Whether Transformer and RL validation can start concurrently given compute.
- Smallest defensible RL comparison: task(s), baselines, evaluation protocol,
  and what can honestly be claimed from one seed.
- Which existing SDQ/Text8 results are paper-ready after a fresh Comet
  snapshot.
- Paper positioning, result inclusion/exclusion, and submission checklist.

### Default result-reporting conventions

Use these conventions unless the user explicitly revises them:

- For RNN and GRNN/MoSAIC runs, the primary comparison horizon is **1B
  processed tokens/steps**. Truncate runs that continue beyond 1B.
- Treat runs that stop slightly short of 1B as completed 1B runs when the
  missing tail is attributable to known server/network logging loss. Do not
  interpret the absent last few logged steps as a smaller intended budget.
- Do not mix HGRN2, DeltaNet, and mLSTM into the main 1B-token RNN/GRNN table.
  Report them in a separate table using their **reduced token budget but
  increased update budget** protocol. Verify and state the precise token,
  update, and batching accounting from the configs when preparing the paper.
- For Text8 analysis, produce both (a) results at the declared comparable
  horizon and (b) per-seed best validation-checkpoint results, clearly labeled
  and without silently treating unfinished runs as completed seeds. The final
  paper presentation will be chosen later.
- SDQ uses an online generator, so a separate held-out validation dataset or
  validation metric is not required. Prefer learning curves and a
  fixed-horizon or final-window statistic over an unlabeled peak training
  accuracy.

Knitwork is a living research project on modular recurrent sequence models. Its main architectural direction is a grid of recurrent columns that keep local state and exchange information through learned attention-based routing. The current main framing is **MoSAIC** (Modular Self-Attentive Interacting Columns for Recurrent Memory).

The codebase contains the base Grid RNN plus many research variants exploring memory mechanisms, temporal dynamics, feedback, hierarchical and LRU cells, reservoirs, losses, and multimodal routing. Use `docs/methods/` for a quick model-level map; read the relevant implementation and YAML config for exact behavior.

The principal experiment families are Store-Distract-Query (associative memory), character-level text modeling, TreasureHunt and related RL work, plus multimodal and analysis experiments. `docs/experiments/` explains what each experiment tests; scripts and configs under `knitwork/exps/` define the current runnable setup.

The active paper material is under `article/`. The current English Typst draft is `article/typst/paper_en.typ`; it presents MoSAIC as an AAAI 2026 submission. A paper is a curated snapshot, not the authority for the whole evolving repository.

For the broader research agenda and exploratory ideas, consult `_supp/` when relevant. For curated notes intentionally left for future research discussions, use `.agents/research-context.md`.
