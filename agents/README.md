# agents/

Mirrored, English-language versions of the paper-writing and Typst↔LaTeX tooling.
These are convenient reference copies of the canonical Claude Code skills/commands;
the originals remain authoritative and are **not** removed.

| Agent file | Canonical source | What it does |
|---|---|---|
| `paper-plan.md` | `.claude/skills/paper-plan/` | Answer pre-writing questions → `article/PLAN.md` (shared context). |
| `paper-abstract.md` | `.claude/skills/paper-abstract/` | Draft the LaTeX abstract. |
| `paper-intro.md` | `.claude/skills/paper-intro/` | Draft the Introduction (5-part structure). |
| `paper-method.md` | `.claude/skills/paper-method/` | Draft the Method section (module triad). |
| `paper-experiments.md` | `.claude/skills/paper-experiments/` | Draft the Experiments section from result tables. |
| `paper-critique.md` | `.claude/skills/paper-critique/` | Adversarial 25-question reviewer critique. |
| `paper-translate.md` | `.claude/skills/paper-translate/` | Translate the paper EN↔RU, preserving LaTeX. |
| `paper-typst.md` | `.claude/skills/paper-typst/` | Convert the paper LaTeX↔Typst (both directions). |
| `math-typst.md` | `~/.claude/commands/math-typst.md` | Generate a Typst math study guide (translated from Russian). |
| `latex-article.md` | `~/.claude/commands/latex-article.md` | Author + compile a LaTeX document (translated from Russian). |

The `references/` subfolder holds the writing-guide reference docs consumed by the
paper-* agents; `agents/openai.yaml` is a separate Codex-style interface manifest.

To actually invoke the tooling, use the canonical skills/commands (e.g. `/paper-typst to-latex`).
These files are English mirrors for reading and for non-Claude agents.
