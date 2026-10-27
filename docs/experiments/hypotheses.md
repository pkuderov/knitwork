# Hypothesis registry

Every hypothesis tested in the AISTATS 2027 cycle and the October RL study, with its verdict and the evidence. Status
values:

- **confirmed**: the effect is clear at the stated scale. Multi-seed where noted, one seed otherwise.
- **rejected**: the change hurt or did nothing at the stated scale.
- **inconclusive**: mixed, single-seed within noise, or the run did not finish.

Details: [AISTATS results](experiments/aistats2027_results.md), [Ablations](experiments/ablations.md), [RL experiments](experiments/rl_2026-10.md).

## Grid-GRU (MoSAIC), language modeling and SDQ

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| G1 | A grid of narrow GRU columns with routing beats a parameter-matched GRU on Text8 | **confirmed** (10M, 3 seeds) | 1.4345 vs 1.4830, p = 0.003 |
| G2 | ...and on long-range memory (SDQ) | **confirmed** (10M, 3 seeds) | Acc++ 0.920 vs 0.618 |
| G3 | The advantage also holds at 1M | inconclusive | -0.029 BPC, p = 0.12 (2-3 seeds) |
| G4 | Grid-GRU is competitive with a Transformer of the same size | **confirmed** (not better) | 10M: -0.015, p = 0.165; 268x less state |
| G5 | Grid-GRU beats modular RNN baselines (RIMs, BRIMs) | **confirmed** (1M) | 1.5918 vs 1.8741 / 1.8853 |
| G6 | Grid-GRU beats linear-attention / matrix-memory models at 1M | **confirmed** | HGRN2 1.6695, mLSTM 1.7163, DeltaNet 1.7844 |
| G7 | Grid-GRU beats Mamba | **rejected** | Mamba 1.5651 vs 1.5918 |
| G8 | Depth matters more than column count | **confirmed** | L3C4 / L2C4 best, L1C8 worst at both scales |
| G9 | More columns help SDQ at fixed depth | **confirmed** (10M) | L2C4 0.843, L2C8 0.866, L2C16 0.891 |
| G10 | The three routing regularizers are needed | **rejected** (pilot) | each |delta| < 0.005 BPC |
| G11 | Content-dependent routing beats a learned static table | **rejected** (1 seed) | Text8 equal; SDQ static 85.5% vs 45.7% |
| G12 | Columns must communicate | **confirmed** (1 seed) | none_mean 1.6452 vs static_mean 1.6144 |
| G13 | Column-0 readout bottleneck is better than a mean readout | inconclusive | better on Text8 (-0.02), worse for dynamic on SDQ |
| G14 | Hard sparse routing (top-1/top-2) is enough | **rejected** | top-1 +0.29, top-2 +0.04 BPC |
| G15 | Longer BPTT (512 vs 64) helps | **rejected** (pilot) | +0.020 for GRU and Grid-GRU |
| G16 | Dynamic routing with column-0 readout is stable on SDQ | **rejected** | peak 66.0% at 840M, final 45.7% |
| G17 | One wide column is as good as four communicating ones | inconclusive | L2C1 1.6920 vs static_mean 1.6921 at 320M (paused) |

## Grid-LRU, language modeling

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| L1 | Replacing the GRU cell with an LRU keeps quality | **rejected** | every shape 0.07-0.11 BPC worse at 1M |
| L2 | `fb_norm` is needed for long rollouts | **confirmed** | overflow after ~1,200 steps without it |
| L3 | Two read heads per column | **confirmed** (1 seed) | -0.0185 BPC |
| L4 | Gated writes | inconclusive | -0.0023 |
| L5 | Rotations in the state | inconclusive | -0.0023 |
| L6 | Query-based reads | **rejected** | +0.0199; collapses on MQAR |
| L7 | Associative matrix memory | **rejected** | +0.0265 |
| L8 | Dynamic hub in a sparse topology | **confirmed** (1 seed) | -0.0102 |
| L9 | Star / ring / entmax topologies are as good as dense | inconclusive | +0.006 to +0.010 |
| L10 | Clockwork columns | **rejected** | +0.0323 |
| L11 | Block-sparse projections (half the core) | **rejected** | +0.0877 |
| L12 | Any variant solves MQAR | **rejected** | best 2.7% overall, about 25% only on the shortest bin |

## Grid-LRU and alternatives, RL (POPGym)

| # | Hypothesis | Status | Evidence (AutoencodeEasy eval return unless stated) |
|---|---|---|---|
| R1 | The paper RL failure was task difficulty plus input dilution | **confirmed** | oracle grid .47 vs LRU .996; with `x_add` .964 |
| R2 | `x_add` | **confirmed** | dynamic .821 vs .101 without |
| R3 | Dynamic routing | **confirmed** | .821 vs static .175, table .102 |
| R4 | Column horizon bands | **confirmed** (3 seeds) | .875 / .847 / .867 vs .821 |
| R5 | One layer instead of two | **confirmed** (1 seed) | .918 vs .875; oracle .513 vs .461 |
| R6 | Recurrent feedback is needed | **confirmed** | -.162 without |
| R7 | Sparse top-k writes | **rejected** | oracle .28 / .74 vs .975 (router collapse) |
| R8 | Selective decay | **rejected** | .102 |
| R9 | Extra communication rounds | **rejected** | 2 rounds .298, 3 rounds .067 |
| R10 | Top-2 routes | inconclusive | .788 / .652 (2 seeds) vs .821 |
| R11 | Read attention / input to all layers | inconclusive | .760 / .768 |
| R12 | Aux loss on past observations | inconclusive | .874 vs .875 |
| R13 | Routing entropy term | **rejected** | .681 / .551 vs .875 |
| R14 | Bank of independent LRUs + attention read | **rejected** | .238 |
| R15 | Bank + soft content-addressed write | **rejected** | .586 best of the family; oracle .386 |
| R16 | Feed-forward-in-time routing | **rejected** | .374 |
| R17 | Soft write allocation in the grid | **rejected** | .198 |
| R18 | Routing queries from the LRU state | **rejected** | .795 |
| R19 | Clockwork layer strides | **rejected** | -.083 |
| R20 | Snapshot memory slots | **rejected** | .285 |
| R21 | LRU backbone + grid | **rejected** | .726 |
| R22 | Shared column projections | inconclusive | .819 vs .875 |
| R23 | Readout over all columns | **rejected** in L1 (.623), neutral in L2 (.831) | |
| R24 | One raw LRU stream column helps transfer | inconclusive | .376 in L1C4, .052 in L1C5 |
| R25 | Linear cell output | **rejected** for transfer | Easy .821, Medium -.086 |
| R26 | 16 cheap shared-weight columns | **rejected** | best .259 at 0.31M |
| R27 | Grid transfers to longer episodes as well as LRU | **rejected** | best .376 vs .612 |
| R28 | Grid beats GRU/LRU on other POPGym tasks | **rejected** | RepeatFirst, MineSweeper, Concentration, CountRecall |
| R29 | CountRecall fails because of memory capacity | **rejected** | oracle LRU learns it (.678): an RL-signal problem |
| R30 | Curriculum (mixing or growing deck length) from scratch | inconclusive | adaptive growth .03 grid / .06 LRU; still far below Easy |

## Data validity notes

- **Text8 file mixup.** From 2026-06-03 the text8 generator pointed to `shakespeare.txt`. June "text8" runs are not
  valid Text8 results. All tables on these pages come from runs after the fix (the AISTATS evidence scripts check data
  hashes).
- **Seeds.** 10M Text8 rows have 3 seeds (L2C16: 2), 1M rows 2-3 seeds. Every pilot, the communication study, all
  Grid-LRU mechanism variants and the RL rounds use one seed, except where the table says otherwise.
- **Checkpoints.** Final-checkpoint scores are reported, not the best point of the curve. This matters for SDQ dynamic
  routing (best 66.0%, final 45.7%).
- **Tests.** Test BPC is reported only for validation-selected checkpoints and never used for selection; the 10M test
  split overlaps with development validation.

## What to run next

1. Seeds 1-2 for L1C4 and L2C4 horizon bands on AutoencodeEasy, and the transfer protocol for L1C4 vs raw column.
2. SDQ none_mean (paused at 477M tokens): the missing no-communication control on SDQ.
3. A 1B-token, multi-seed regularizer ablation if the paper keeps the regularizer claim.
4. Grid-GRU with the RL fixes (`x_add`, dynamic routing exists already) on the RL protocol: the RL study only redesigned
   the LRU grid.
