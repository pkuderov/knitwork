# Paper work guide

`article/` holds paper artifacts, not the complete current research record. Treat a paper as a deliberate, time-bounded argument: every technical description and quantitative claim must be checked against the relevant current implementation, configuration, and evidence before it is added or revised.

## Current paper

The current English Typst draft is `typst/paper_en.typ`, an anonymous AAAI 2026 submission presenting MoSAIC (Modular Self-Attentive Interacting Columns for Recurrent Memory). Related Typst sources, bibliography, figures, and generated PDF are in `typst/`; LaTeX material is in `latex/`.

Use the paper source, rather than a generated PDF, as the editable authority. Keep submission-specific formatting and anonymity constraints intact unless the task explicitly changes them.

## Context and writing support

- Use `../agents/README.md` to discover the mirrored paper-writing toolkit and `../agents/references/` for reusable planning, section-writing, and review guidance.
- The named `../agents/paper-*.md` workflows contain useful patterns but several are tied to earlier Grid RNN/HarmonicGridRNN drafts. Adapt them only after checking the current paper, paths, and evidence.

For paper changes, distinguish supported results from hypotheses or missing evidence. Escalate uncertain claims, interpretation changes, and decisions that alter the paper's scope or contribution.
