# Agent guide

Knitwork is a private research codebase for modular recurrent architectures, especially Grid RNN / MoSAIC variants, and their evaluation on memory, language-modeling, multimodal, and reinforcement-learning tasks.

## Start here

Read `.agents/working-context.md` at the start of any task. It is the short current snapshot of the project.

Then read only the context relevant to the task:

- `docs/methods/` for high-level model summaries and `docs/experiments/` for experiment purpose, setup, and run examples.
- Code and YAML configs for exact current behavior.
- `article/AGENTS.md` for paper work.
- `agents/README.md` and `agents/references/` for paper-writing methods. Some `agents/paper-*.md` recipes describe earlier work; verify their paths, assumptions, and claims before using them.
- `.agents/research-context.md` and relevant `_supp/` files for explicit research discussion, hypotheses, or future-facing design work.

## Collaboration mode

Default to **guided** work: ask before taking a materially different valid direction or making consequential decisions. Work is **trusted** only when the user explicitly grants that authority for the task; then make ordinary in-scope decisions and report them.

Ask when the desired autonomy or scope is genuinely ambiguous. Do not launch long experiments, use remote infrastructure, query external tracking systems, or make publication-affecting claims without explicit task authorization.

## Working principles

- Keep changes scoped; preserve unrelated work and existing artifacts.
- Treat code/config as the authority for exact implementation behavior. Treat tracker data or explicitly identified run artifacts as the authority for current quantitative evidence.
- Documentation is a useful explanation layer, but it can lag behind the code. Update directly affected, high-value context when confident; otherwise flag the discrepancy and suggest a follow-up.
- Treat `_supp/` as exploratory research material, not a complete or necessarily current specification.
- Do not add to `.agents/research-context.md` automatically. Update it only when asked to preserve or summarize research discussion.
