# Paper work guide

`article/` holds paper artifacts, not the complete current research record. Treat a paper as a deliberate, time-bounded argument: every technical description and quantitative claim must be checked against the relevant current implementation, configuration, and evidence before it is added or revised.

## Current paper

The current AISTATS 2027 revision is `latex/paper.tex`, an anonymous submission presenting MoSAIC (Modular Self-Attentive Interacting Columns for Recurrent Memory), using the official `latex/aistats2027.sty`. Build instructions and remaining evidence gaps are in `latex/README.md`. The earlier AAAI-27 Typst draft remains in `typst/paper_en.typ`; its numbers and formatting are not synchronized with the AISTATS revision.

Use the paper source, rather than a generated PDF, as the editable authority. Keep submission-specific formatting and anonymity constraints intact unless the task explicitly changes them.

## Context and writing support

- Use `../agents/README.md` to discover the mirrored paper-writing toolkit and `../agents/references/` for reusable planning, section-writing, and review guidance.
- The named `../agents/paper-*.md` workflows contain useful patterns but several are tied to earlier Grid RNN/HarmonicGridRNN drafts. Adapt them only after checking the current paper, paths, and evidence.

For paper changes, distinguish supported results from hypotheses or missing evidence. Escalate uncertain claims, interpretation changes, and decisions that alter the paper's scope or contribution.
