# MoSAIC — AISTATS 2027 revision

The active anonymous source is `paper.tex`; `paper.pdf` is generated from it. The current workspace branch is `aistat-knitwork`. Earlier Typst/AAAI artifacts are historical and are not synchronized automatically.

## Build and regenerate evidence

From the repository root:

```bash
make -C article/latex
```

The build runs pdfLaTeX, BibTeX, and three subsequent pdfLaTeX passes. To regenerate the new tables, figures, and numeric provenance **offline** from retained evidence:

```bash
.venv/bin/python inference/aistats_revision_evidence.py
make -C article/latex
```

`make -C article/latex evidence` performs the same offline evidence generation. It does not launch training or call Comet. Python requires the existing research environment and Matplotlib; local model accounting uses the logged widths/configurations with current model definitions. To regenerate the historical resource tables offline, run `.venv/bin/python inference/aistats_resource_accounting.py`.

## Separate manuscript without the earlier hypotheses

`paper_evidence_only.tex` and `paper_evidence_only.pdf` are a separate edition of the article. They retain the architecture, all empirical tables/figures, uncertainty, reproducibility limitations, and AI disclosure, while removing discussion of earlier specialization, column-collapse, and gradient-mechanism hypotheses and speculative explanations of the results. The AI disclosure describes assistance with architectural design. This edition shares the retained table files, bibliography, style files, and figure assets with `paper.tex`; it is edited separately and is not regenerated from the main manuscript automatically.

```bash
make -C article/latex paper_evidence_only.pdf
```

The separate PDF has 20 pages: main text 1–8, AI statement/references 9–10, checklist 11, appendices 12–20. Its validation record is `docs/experiments/aistats2027_evidence_only_validation.json`. Both manuscripts load the standard `float` package so Appendix J resource tables use `[H]` placement and appear consecutively instead of spreading across a float page. The resource-table generator preserves this placement when run again; measurements and official style files are unchanged.

`aistats2027.sty` and `fancyhdr.sty` remain the unmodified official [AISTATS 2027 author kit](https://aistats.org/aistats2027/AISTATS2027PaperPack.zip), downloaded on 2026-10-01. The paper uses anonymous submission mode, US Letter, author–year citations, and single-column appendices. The [current call for papers](https://virtual.aistats.org/Conferences/2027/CallForPapers), checked on October 5, limits the main text to eight pages and imposes no page limit on references, AI Use Statement, checklist, or appendices. Even twenty appendix pages would be allowed. The AI statement immediately precedes references and explicitly discloses literature analysis and assistance in writing/debugging code. The September 30 policy update requires this statement in the paper only, not in the OpenReview form. The title remains unchanged.

The final October 5 PDF has 21 pages: main text 1–8; AI Use Statement and references 9–10; checklist 11; appendices 12–21 (ten appendix pages). Compilation has no unresolved citations or cross-references. PDF metadata is anonymous and fonts are Type 0/Type 1, without Type 3. Both style files were checked byte for byte against a fresh official-kit download. The only remaining overfull-box warning is the official title block's 5.12pt warning. Key pages were rendered and inspected: no clipping of tables, equations, figures, or text. The retained [validation record](../../docs/experiments/aistats2027_revision_validation.json) also records the recheck of 124 pilot scalars against raw logs.

## Evidence in the October 5 revision

The revision keeps the original GRU-column architecture as its central contribution. It separates the historical approximately 10M series from the new approximately 1M Text8 series, and separates full-budget comparisons from short or single-seed exploratory probes. The full reviewer-response map and remaining gaps are in [the revision report](../../2026-10-05-aistats-paper-revision.md).

**Historical quality was reaggregated, rather than copied at different endpoints.** A read-only retrieval on October 5 recovered quality curves for the fixed 65-run inventory from `docs/experiments/results_aaai.md`; the July report and October resource snapshot remain unchanged. The new [historical quality snapshot](../../docs/experiments/aistats2027_historical_quality.json) supplies 29 main Text8 launches and 24 main SDQ launches. The quality table uses:

- Text8: 960,036,864 logged tokens. All recurrent rows are observed exactly; Transformer rows are linearly interpolated between 942,538,752 and 962,592,768, without extrapolation.
- SDQ: the same five counters for every launch, 950,036,480 through 970,037,248. Acc++ remains an online curriculum diagnostic rather than an independent memory test.

These reporting rules change the printed historical means/SDs. L2C4 now has Text8 `1.4389 ± 0.0033` and SDQ `0.8389 ± 0.0062`; GRU-L2 has `1.5005 ± 0.0121` and `0.5409 ± 0.0735`. Abstract, body, tables, and curves use the revised rules consistently. The historical figure `fig_learning_curves.pdf` is preserved; the paper uses the regenerated `fig_aistats_historical_curves.pdf`.

**New evidence** is frozen in the [weekly snapshot](../../docs/experiments/weekly_2026-09-28_2026-10-04_snapshot.json), collected through October 4 at 23:53 Moscow time. Table 2 is in the main text and reports all 18 available late-budget baseline/topology groups: its upper panel contains 16 groups and 36 launches at exactly 980,037,632 tokens; its lower panel adds Grid-LRU L2C4 (three launches at 960,036,864) and Transformer-256 (two at 982,646,784). The different horizons are explicitly labeled, without interpolation. Baselines are local implementations using a shared recipe, not official optimized checkpoints or an exhaustive tuning comparison. Mamba is the original selective-SSM family, not Mamba-2, and is stronger than MoSAIC in the new Text8 series.

The main text adds top-1/top-2 routing and the five-condition regularizer pilot. The LRU mechanism table also reports already logged periodic resets every 1,024 valid tokens, with the reset penalty relative to continuous evaluation; this is a single-seed robustness diagnostic, not a longer-memory claim. Appendices report all five LRU mechanism variants, seven sparse-LRU variants with paired controls, eight BPTT-length pilots, six MQAR quality pilots, and ten POPGym RL pilots. Their negative results are retained. MQAR does not have completed strong-baseline comparisons, and the RL tasks are POPGym tasks accessed through the project runner rather than a separate MIKASA benchmark. Addressable-slot smoke checks and runs without quality metrics are not promoted to benchmark results. Numeric test values were not used for selection or reported comparison.

Generated TeX tables live in `aistats_*.tex`; the offline evidence builder writes [full numeric provenance](../../docs/experiments/aistats2027_revision_evidence.json), including exact horizons, launch IDs, seeds, raw replicate summaries, interpolation flags, and state/parameter counts. Launch IDs and private tracker links are not included in the anonymous PDF.

## Resources and reproducibility

The historical [resource report](../../docs/experiments/aistats2027_resources.md) preserves all 65 launches and their original device labels: 48 H100 80GB HBM3, eight TITAN RTX, five RTX 3080 Ti, and four V100-SXM3-32GB. Paper Table 4 and Appendix J now restrict resource reporting to the 48 verified H100 launches, with actual subset counts. The other 17 launches remain in the historical quality cohort; the manuscript explicitly states that this cohort used heterogeneous hardware. No GPU labels or measurements are reassigned to H100. Loop time is the integral of logged token increments divided by interval fps; it includes overhead/contention and is neither exclusive GPU-hours nor FLOPs. Resource prefixes run through the last log at or below 1B and differ from the common quality horizons.

New device labels were retrieved read-only for the fixed 87-run `knitwork-aistat` weekly inventory and preserved in [a sanitized snapshot](../../docs/experiments/aistats2027_new_devices.json). All identify H100 80GB HBM3. Only GPU model labels were saved; environment variables, hostnames, usernames, IP addresses, and credentials were omitted. Shared hardware labels do not establish controlled workload conditions.

The paper reports carried-state storage separately from peak training memory and runtime. It does not claim efficient sparse execution or learned semantic specialization. The code/configuration release is planned after acceptance; no anonymous code archive accompanies this draft. Historical seeds, the complete tuning record, and a complete asset-license inventory remain unavailable. The official checklist retains truthful No answers for those omissions. The author kit requests it after references, permits omission at initial submission without desk rejection, and requires its later inclusion at author response/camera-ready. It is retained here as a completed submission section.

## Remaining evidence gaps

- Full-budget state-matched and compute-matched controls. The found state-matched GRU run is only a technical check, not a quality comparison.
- Independent SDQ evaluation with fixed gaps and held-out sequences.
- Full-budget, repeated-seed regulator/projection/readout ablations, particularly on SDQ.
- Strong-baseline and repeated-seed MQAR/RL comparisons; broader tasks and scales.
- Controlled throughput, inference latency, peak training memory, and consistently estimated FLOPs if making an efficiency claim.
- An untouched test evaluation: the newly configured Text8 test segment overlaps historical development validation. The current paper reports internal validation throughout.

The [final audit](../../2026-10-05-aistats-final-audit.md) checks conference requirements and all 24 questions in the repository's paper-review guide. The [reviewer follow-up](../../2026-10-05-aistats-reviewer-followup.md) additionally maps all retained reviewer claims, both critical issues, eleven major and fourteen minor audit items, and the AI architecture analysis. This pass clarified BRIMs overlap, the approximate 0.973M–1.069M parameter range, readout-width confounding, unverifiable historical seed independence, and routing-diagnostic limits; all three figures now have prose references. No training or external submission was performed during revision. The local abstract was updated; the exact abstract previously entered in OpenReview is unavailable here. The author should compare it against this revision before upload, because the current call for papers flags major title/abstract changes after the abstract deadline. OpenReview author/profile/quota/reviewer fields and concurrent-submission status were not accessible for verification.
