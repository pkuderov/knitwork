# MoSAIC — AISTATS 2027 manuscript

The active anonymous source is `paper.tex`, with the separate evidence-focused edition in `paper_evidence_only.tex`. Both share the evidence tables, bibliography, style files, and figure assets. The workspace branch is `aistat-knitwork`. Earlier Typst/AAAI drafts are separate artifacts.

## Build

From the repository root:

```bash
make -C article/latex paper.pdf paper_evidence_only.pdf
```

Regenerate tables, figures, and provenance offline, without training or tracker requests:

```bash
.venv/bin/python inference/aistats_revision_evidence.py
.venv/bin/python inference/aistats_resource_accounting.py
make -C article/latex paper.pdf paper_evidence_only.pdf
```

The existing research environment supplies PyTorch, Matplotlib, SciPy, NumPy, and YAML dependencies. Local parameter/state accounting uses logged model configurations. The generator also reads the retained test CSV export, distinct from the weekly snapshot's metric-presence inventory.

## Current evidence and scope

The paper compares GRU-column MoSAIC with stacked GRU and selected local recurrent, modular, and attention references. The larger-scale SDQ/Text8 study uses approximately 10M parameters; the smaller-scale Text8 table contains 17 configurations and 39 runs near 1M parameters. Main training budgets are one billion tokens. Text8 primary values summarize final-checkpoint continuous-state validation; SDQ averages the final five online diagnostic measurements per run. Means and sample SD describe between-run variation.

Eighteen available test evaluations cover 12 of the 17 smaller-scale groups. They use checkpoints selected by reset-1024 validation loss, evaluated with continuous state on the final 5M characters. Missing test scores remain explicit. This test segment overlaps validation used during earlier larger-scale development. Four exploratory Welch comparisons have intervals spanning zero; additional seeds and aligned segment losses are unavailable.

The local Bayesian section is an interpretive Gaussian surrogate, with five theorems, four lemmas, and complete proofs. It is separated from architectural novelty and supplies no learned-router or BPC guarantee. The separate edition continues to omit earlier specialization and gradient hypotheses.

The [audit response](../2026-10-07-aistats-audit-response.md) maps every major and minor item to its correction and remaining evidence gap. Its current validation record is [aistats2027_audit_response_validation.json](../../docs/experiments/aistats2027_audit_response_validation.json). Earlier dated validation records describe earlier PDFs and are retained as historical artifacts.

## Anonymous review code

The supplementary artifact is [mosaic_review_code.zip](../supplementary/mosaic_review_code.zip), with [run instructions](../supplementary/README.md). Regenerate it locally:

```bash
.venv/bin/python inference/aistats_review_archive.py
```

It includes the models, 1M Text8 configuration, training/evaluation runner, dependencies, and checksum manifest. Tracking is disabled and the dataset path is made portable. It contains no tracker exports or checkpoints. It has been checked from an extracted independent directory, including the main-table parameter/state counts and finite CPU forward outputs. Preparing the archive does not publish it or run training. Public code release is planned after acceptance. The author must attach this archive when submitting the manuscript for its supplementary-code availability statement to apply.

## Format

`aistats2027.sty` and `fancyhdr.sty` remain the unmodified official [AISTATS 2027 author kit](https://aistats.org/aistats2027/AISTATS2027PaperPack.zip). Both editions use anonymous submission mode, US Letter, author–year citations, and single-column appendices. The main text remains within eight pages. AI Use Statement, references, checklist, and appendices follow outside that limit. The AI statement discloses literature analysis, coding assistance, and other assistance used in preparing the work. PDF author and build dates are blank.

Appendix tables use fixed placement and ragged page bottoms to avoid artificial vertical gaps. Resource tables remain paired by task on separate pages. The architecture diagram uses the same `sqrt(d)` denominator as the equations. There are no unresolved citations or cross-references; the official title-block warning is reported in the validation record.

Both rebuilt PDFs contain 25 pages: main text 1–8, AI statement and references 9–10, checklist 11, and appendices 12–25. Checklist answer explanations were shortened without changing its questions or font size, eliminating the almost-empty continuation page. The checklist page and representative result tables were visually inspected after rebuilding.
