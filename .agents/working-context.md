# Working context

Last refreshed: 2026-07-25.

Knitwork is a living research project on modular recurrent sequence models. Its main architectural direction is a grid of recurrent columns that keep local state and exchange information through learned attention-based routing. The current main framing is **MoSAIC** (Modular Self-Attentive Interacting Columns for Recurrent Memory).

The codebase contains the base Grid RNN plus many research variants exploring memory mechanisms, temporal dynamics, feedback, hierarchical and LRU cells, reservoirs, losses, and multimodal routing. Use `docs/methods/` for a quick model-level map; read the relevant implementation and YAML config for exact behavior.

The principal experiment families are Store-Distract-Query (associative memory), character-level text modeling, TreasureHunt and related RL work, plus multimodal and analysis experiments. `docs/experiments/` explains what each experiment tests; scripts and configs under `knitwork/exps/` define the current runnable setup.

The active paper material is under `article/`. The current English Typst draft is `article/typst/paper_en.typ`; it presents MoSAIC as an AAAI 2026 submission. A paper is a curated snapshot, not the authority for the whole evolving repository.

For the broader research agenda and exploratory ideas, consult `_supp/` when relevant. For curated notes intentionally left for future research discussions, use `.agents/research-context.md`.
