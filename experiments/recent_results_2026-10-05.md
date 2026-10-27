# Recent experiments: results and logger tables (2026-09-29 to 2026-10-05)

Summary of the runs of the last six days, exported from Comet (workspace `team-rl-exp`, project `knitwork-aistat`) on 2026-10-05. Full per-run metrics are in [recent_metrics_2026-10-05.csv](recent_metrics_2026-10-05.csv). All values are last logged values from the logger; single seed 42 unless a seed is given. Differences below about 0.01 BPC are within seed spread (see the stability table) and should not be read as rankings.

## Short summary

- **Text8, about 1M parameters.** Mamba is best (val BPC 1.58). Grid GRU (`grnn.L2C4`, `L3C4`) is next at about 1.60, then GRU `L2`/`L3` at 1.63-1.64 and the transformer at 1.65-1.66. HGRN2 and Grid LRU are near 1.68-1.69, mLSTM 1.72, DeltaNet 1.79; RIMs and BRIMs are worst (1.8-1.96) with seed spread above 0.08 BPC.
- **Seeds.** Seeds 42, 1337 and 31337 agree within 0.015 BPC for all models except RIMs and BRIMs.
- **LRU variants (mid).** In the connectivity series the best is `dynamic_hub` (1.677 against control 1.688); in the memory/access series `two_reads` is best (1.661 against `dynamic_control` 1.679). These are single-seed gaps of 0.01-0.02 BPC.
- **Rollout pilots (67M tokens).** A 512-step rollout was not better than 64 for any model in these pilots (r512 rows are logged only to 63M tokens, so the comparison is approximate). For Grid GRU, dropping noise or the entropy term gives the same val BPC as the full model (1.868 and 1.861 against 1.862); the `none` and `no_comm` rows stop at 63M and are not comparable.
- **MQAR (8 epochs).** No variant, including the control, generalizes: train accuracy about 0.20 (only the easiest segment is learned), validation accuracy 0.015-0.027. `query_read` did not train.
- **MIKASA shortlist.** Only the LRU baseline moved off the random level (-0.35 on RepeatPreviousHard, -0.47 on AutoencodeEasy); every other model stays near the random return of -0.50 at the 8.39M-slot budget.
- **Slot memory.** No results: the 8-epoch slot MQAR runs were stopped, and the redesigned slot jobs were queued but the queue was stopped before they started.

## Text8, mid protocol (about 1M parameters, 1e9 tokens, 90M/5M/5M split)

`val` is the best-checkpoint window-1024 validation; `test` is the one-time test of the best-val checkpoint (BPC, lower is better). A dash in the test columns means the test metric is absent from the Comet run (it was not logged or the upload failed); all runs have the validation value, so rows are sorted by it. Rows labeled partial stopped logging a few steps before the budget but finished training.

| model | seed | params | val BPC (w1024) | test BPC | test BPC (w1024) | tokens | status |
|---|---|---|---|---|---|---|---|
| mamba | 1337 | 1.00M | 1.579 | 1.633 | 1.645 | 1.00B | done |
| mamba | 42 | 1.00M | 1.583 | 1.636 | 1.652 | 1.00B | done |
| grnn.L3C4 | 1337 | 1.07M | 1.601 | 1.659 | 1.668 | 1.00B | done |
| grnn.L2C4 | 42 | 1.05M | 1.602 | - | - | 995M | done |
| grnn.L3C4 | 31337 | 1.07M | 1.602 | - | - | 995M | done |
| grnn.L2C4 | 1337 | 1.05M | 1.603 | 1.660 | 1.668 | 1.00B | done |
| grnn.L3C4 | 42 | 1.07M | 1.603 | - | - | 995M | done |
| grnn.L2C4 | 31337 | 1.05M | 1.611 | 1.668 | 1.675 | 1.00B | done |
| grnn.L2C8 | 1337 | 972.72K | 1.617 | 1.673 | 1.681 | 1.00B | done |
| grnn.L2C8 | 42 | 972.72K | 1.618 | - | - | 995M | done |
| rnn.L3 | 1337 | 985.56K | 1.624 | 1.677 | 1.685 | 1.00B | done |
| rnn.L3 | 42 | 985.56K | 1.634 | 1.686 | 1.693 | 1.00B | done |
| rnn.L2 | 42 | 1.01M | 1.643 | 1.696 | 1.703 | 1.00B | done |
| rnn.L2 | 1337 | 1.01M | 1.643 | - | - | 995M | done |
| rnn.L2 | 31337 | 1.01M | 1.644 | 1.697 | 1.703 | 1.00B | done |
| grnn.L1C8 | 1337 | 1.03M | 1.645 | - | - | 995M | done |
| grnn.L2C4_topk2 | 42 | 1.05M | 1.645 | - | - | 995M | done |
| grnn.L1C8 | 42 | 1.03M | 1.647 | 1.703 | 1.712 | 1.00B | done |
| transformer | 1337 | 1.00M | 1.647 | - | - | 998M | done |
| transformer | 42 | 1.00M | 1.661 | - | - | 998M | done |
| hgrn2 | 1337 | 996.21K | 1.678 | - | - | 995M | done |
| hgrn2 | 42 | 996.21K | 1.680 | - | - | 995M | done |
| grnn_lru.L3C4 | 1337 | 1.06M | 1.681 | 1.730 | 1.743 | 1.00B | done |
| grnn_lru.L3C4 | 42 | 1.06M | 1.684 | 1.730 | 1.742 | 1.00B | done |
| grnn_lru.L2C4 | 31337 | 1.05M | 1.685 | - | - | 995M | done |
| grnn_lru.L2C4 | 42 | 1.05M | 1.686 | - | - | 995M | done |
| grnn_lru.L2C4 | 1337 | 1.05M | 1.686 | - | - | 975M | partial |
| grnn_lru.L3C4 | 31337 | 1.06M | 1.686 | - | - | 995M | done |
| rnn.L1 | 42 | 1.00M | 1.688 | - | - | 995M | done |
| rnn.L1 | 1337 | 1.00M | 1.688 | 1.741 | 1.748 | 1.00B | done |
| mlstm | 42 | 995.58K | 1.721 | - | - | 995M | done |
| grnn_lru.L2C8 | 1337 | 1.06M | 1.725 | - | - | 995M | done |
| mlstm | 1337 | 995.58K | 1.726 | - | - | 995M | done |
| grnn_lru.L2C8 | 42 | 1.06M | 1.731 | - | - | 990M | partial |
| grnn_lru.L1C8 | 42 | 1.00M | 1.761 | 1.812 | 1.824 | 1.00B | done |
| grnn_lru.L1C8 | 1337 | 1.00M | 1.761 | - | - | 995M | done |
| delta_net | 42 | 1.00M | 1.792 | - | - | 995M | done |
| delta_net | 1337 | 1.00M | 1.796 | 1.840 | 1.851 | 1.00B | done |
| rims | 1337 | 994.86K | 1.813 | 1.863 | 1.870 | 1.00B | done |
| brims | 42 | 1.00M | 1.880 | 1.926 | 1.932 | 1.00B | done |
| grnn.L2C4_topk1 | 42 | 1.05M | 1.895 | - | - | 995M | done |
| brims | 1337 | 1.00M | 1.963 | 1.952 | 2.017 | 1.00B | done |
| rims | 42 | 994.86K | 1.966 | 2.003 | 2.018 | 1.00B | done |

### Seed stability (val BPC, w1024, models with more than one seed)

| model | n seeds | val BPC (w1024) per seed | mean | range |
|---|---|---|---|---|
| mamba | 2 | 1.579, 1.583 | 1.581 | 0.004 |
| grnn.L3C4 | 3 | 1.601, 1.602, 1.603 | 1.602 | 0.002 |
| grnn.L2C4 | 3 | 1.602, 1.603, 1.611 | 1.605 | 0.009 |
| grnn.L2C8 | 2 | 1.617, 1.618 | 1.618 | 0.001 |
| rnn.L3 | 2 | 1.624, 1.634 | 1.629 | 0.010 |
| rnn.L2 | 3 | 1.643, 1.643, 1.644 | 1.643 | 0.001 |
| grnn.L1C8 | 2 | 1.645, 1.647 | 1.646 | 0.002 |
| transformer | 2 | 1.647, 1.661 | 1.654 | 0.014 |
| hgrn2 | 2 | 1.678, 1.680 | 1.679 | 0.002 |
| grnn_lru.L3C4 | 3 | 1.681, 1.684, 1.686 | 1.684 | 0.005 |
| grnn_lru.L2C4 | 3 | 1.685, 1.686, 1.686 | 1.686 | 0.001 |
| rnn.L1 | 2 | 1.688, 1.688 | 1.688 | 0.000 |
| mlstm | 2 | 1.721, 1.726 | 1.724 | 0.005 |
| grnn_lru.L2C8 | 2 | 1.725, 1.731 | 1.728 | 0.006 |
| grnn_lru.L1C8 | 2 | 1.761, 1.761 | 1.761 | 0.000 |
| delta_net | 2 | 1.792, 1.796 | 1.794 | 0.004 |
| rims | 2 | 1.813, 1.966 | 1.889 | 0.153 |
| brims | 2 | 1.880, 1.963 | 1.921 | 0.083 |

## Sparse-connectivity Grid LRU variants (Text8, mid)

| variant | params | val BPC (w1024) | test BPC | tokens | status |
|---|---|---|---|---|---|
| control | 1.05M | 1.688 | 1.739 | 1.00B | done |
| self_heavy | 1.05M | 1.681 | - | 995M | done |
| star | 1.05M | 1.692 | 1.741 | 1.00B | done |
| star_ring | 1.05M | 1.694 | - | 995M | done |
| clockwork | 1.05M | 1.724 | - | 995M | done |
| block | 531.06K | 1.776 | - | 995M | done |
| entmax | 1.05M | 1.698 | 1.748 | 1.00B | done |
| dynamic_hub | 1.06M | 1.677 | 1.730 | 1.00B | done |

## LRU memory and access variants (Text8, mid)

| variant | params | val BPC (w1024) | test BPC | tokens | status |
|---|---|---|---|---|---|
| dynamic_control | 1.06M | 1.679 | 1.731 | 1.00B | done |
| write_gate | 1.06M | 1.677 | 1.728 | 1.00B | done |
| query_read | 1.11M | 1.701 | - | 995M | done |
| associative | 830.32K | 1.712 | - | 995M | done |
| rotations | 1.06M | 1.677 | 1.729 | 1.00B | done |
| two_reads | 1.06M | 1.661 | 1.713 | 1.00B | done |

## Rollout-length pilots and Grid-GRU ablations (Text8, 67M tokens, single seed)

Short pilots on a smaller validation set: compare only with each other. Several runs stop logging at 63M of the 67M budget, so rows are not token-matched.

| run | val BPC | last logged tokens |
|---|---|---|
| rollout_mid/gru/r64 | 1.912 | 67M |
| rollout_mid/gru/r512 | 2.003 | 63M |
| rollout_mid/grnn_lru/r64 | 2.300 | 67M |
| rollout_mid/grnn_gru/r64 | 1.862 | 67M |
| rollout_mid/grnn_lru/r512 | 2.487 | 63M |
| rollout_mid/grnn_gru/r512 | 1.946 | 63M |
| grnn_ablation_mid/none | 1.928 | 63M |
| rollout_mid/lru/r64 | 2.157 | 63M |
| rollout_mid/lru/r512 | 2.155 | 63M |
| grnn_ablation_mid/no_noise | 1.868 | 67M |
| grnn_ablation_mid/no_comm | 1.922 | 63M |
| grnn_ablation_mid/no_entropy | 1.861 | 67M |

## MQAR (8 epochs, full-sequence BPTT, vocab 8192)

Validation accuracy over query tokens (last evaluation). `skipped` is the smoothed skipped-step metric logged by the old runner (the exact counter was added later). Runs marked stopped were terminated by us.

| run | train Acc | val Acc | val T64_K4 | val T128_K8 | val T256_K16 | val T256_K32 | val T256_K64 | skipped | epoch |
|---|---|---|---|---|---|---|---|---|---|
| lru_architecture_mid/dynamic_control | 0.199 | 0.0269 | 0.254 | 0.113 | 0.045 | 0.019 | 0.001 | 7.0 | 8.0 |
| lru_architecture_mid/write_gate | 0.201 | 0.0232 | 0.247 | 0.107 | 0.034 | 0.009 | 0.003 | 0.0 | 8.0 |
| lru_architecture_mid/query_read | 0.000 | 0.0002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 171.0 | 8.0 |
| lru_architecture_mid/associative | 0.204 | 0.0152 | 0.183 | 0.081 | 0.025 | 0.003 | 0.000 | 11.0 | 8.0 |
| lru_architecture_mid/rotations | 0.210 | 0.0170 | 0.233 | 0.076 | 0.019 | 0.005 | 0.002 | 0.0 | 8.0 |
| lru_architecture_mid/two_reads | 0.186 | 0.0180 | 0.225 | 0.079 | 0.022 | 0.006 | 0.002 | 8.0 | 8.0 |
| lru_slots_mid/slot_content | - | stopped | - | - | - | - | - | - | - |
| lru_slots_mid/slot_alloc | - | stopped | - | - | - | - | - | - | - |

## MIKASA / POPGym shortlist (PPO, 8.39M slots budget, seed 42)

Return range is -1 to +1, random policy is about -0.50, perfect is +1. `train EpRet` is the last logged training return, `eval EpRet` the last independent evaluation.

| run | train EpRet | eval EpRet | entropy | slots | Skipped |
|---|---|---|---|---|---|
| 01_RepeatPreviousHard_gru | -0.504 | -0.518 | 0.970 | 8M | 0 |
| 02_RepeatPreviousHard_lru | -0.346 | -0.353 | 0.084 | 8M | 0 |
| 03_RepeatPreviousHard_grnn_gru | -0.498 | -0.515 | 1.278 | 8M | 0 |
| 04_RepeatPreviousHard_grnn_lru | -0.498 | -0.505 | 1.306 | 8M | 0 |
| 05_RepeatPreviousHard_dynamic_hub | -0.499 | -0.495 | 1.302 | 8M | 0 |
| 06_RepeatPreviousHard_two_reads | -0.501 | -0.501 | 1.237 | 8M | 0 |
| 07_AutoencodeEasy_gru | -0.494 | -0.498 | 0.821 | 8M | 0 |
| 08_AutoencodeEasy_lru | -0.473 | -0.469 | 0.073 | 8M | 0 |
| 09_AutoencodeEasy_dynamic_hub | -0.499 | -0.511 | 1.272 | 8M | 0 |
| 10_AutoencodeEasy_two_reads | -0.499 | -0.526 | 1.352 | 8M | 0 |
