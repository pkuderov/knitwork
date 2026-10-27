# Experiment Results

Source: Comet workspace `team-rl-exp`, project `knitwork`. Regenerate with `uv run server/query_comet.py --format markdown --out docs/experiments/results.md`.

Metrics are best-so-far (Loss = min, Acc = max); `Steps` is the training step at which the best value was reached. Run names encode model, size and benchmark.

| Run | Loss ↓ | Acc ↑ | Acc/query ↑ | Acc/distract ↑ | Steps | Created |
| --- | --- | --- | --- | --- | --- | --- |
| knitwork_grnn_fix 200K SDQ_v4.2 2L8C | 0.2100 | 0.9157 | 0.8556 | 0.9930 | 70M | 2026-07-10 09:22 |
| knitwork_grnn_fix 200K SDQ_v4.2 2L6C | 0.1376 | 0.9461 | 0.9088 | 0.9967 | 80M | 2026-07-10 09:22 |
| knitwork_grnn_multimodal v4 200k mdsum | 1.4220 | 0.5336 | - | - | 89M | 2026-07-09 20:19 |
| knitwork_grnn fix v4 400k n_columns=8 sdq | 0.1176 | 0.9549 | 0.9290 | 0.9972 | 117M | 2026-07-09 15:39 |
| knitwork_grnn fix v4 200k sdq | 0.2938 | 0.8802 | 0.7958 | 0.9915 | 75M | 2026-07-09 10:44 |
| knitwork_grnn fix v4 400k sdq | 0.2251 | 0.9145 | 0.8554 | 0.9951 | 82M | 2026-07-09 10:44 |
| knitwork_grnn_fix 200K SDQ_v5.1 hybrid | 0.2173 | 0.9118 | 0.8631 | 0.9911 | 121M | 2026-07-08 15:23 |
| knitwork_grnn_fix 200K text8_v4.1 r32 (lr x2) | 1.3514 | 0.5780 | - | - | 211M | 2026-07-08 15:13 |
| knitwork_grnn_fix 200K text8_v5 | 1.5459 | 0.5281 | - | - | 48M | 2026-07-07 17:40 |
| knitwork_grnn_fix 200K SDQ_v5 1L6C | 0.9724 | 0.6397 | 0.4052 | 0.9283 | 69M | 2026-07-07 16:09 |
| knitwork_grnn_fix 200K SDQ_v5 | 0.8107 | 0.6958 | 0.4949 | 0.9515 | 74M | 2026-07-07 15:26 |
| knitwork_grnn_fix 200K SDQ_v4.1 1L5C | 0.1662 | 0.9334 | 0.8969 | 0.9971 | 134M | 2026-07-07 13:13 |
| knitwork_grnn_fix 200K text8_v4.1 r32 | 1.3628 | 0.5756 | - | - | 190M | 2026-07-07 13:12 |
| knitwork_gru H=222 ~200K text8 | 1.2550 | 0.6081 | - | - | 272M | 2026-07-07 10:20 |
| knitwork_grnn_fix 200K SDQ_v4 1L5C | 1.0699 | 0.6096 | 0.3799 | 0.7642 | 12M | 2026-07-07 09:54 |
| knitwork_grnn_fix 200K text8_v4 | 1.2847 | 0.5972 | - | - | 117M | 2026-07-07 09:42 |
| knitwork_hgrnn_fix 200K SDQ_v4 | 0.2082 | 0.9173 | 0.8691 | 0.9926 | 113M | 2026-07-07 09:42 |
| knitwork_grnn_fix 200K SDQ_v4 | 0.1766 | 0.9288 | 0.8868 | 0.9914 | 92M | 2026-07-06 21:25 |
| knitwork_grnn_fix 200K SDQ_v3 | 0.1754 | 0.9292 | 0.8933 | 0.9932 | 139M | 2026-07-06 19:41 |
| knitwork_grnn_fix 200K text8_v2 | 1.2819 | 0.5979 | - | - | 118M | 2026-07-06 18:26 |
| knitwork_grnn_fix 200K SDQ_v2 | 0.3285 | 0.8664 | 0.8013 | 0.9890 | 162M | 2026-07-06 17:30 |
| knitwork_grnn_harmonic H=40 L=3 C=4 ~224K text8 +eval | 1.3820 | 0.5819 | - | - | 61M | 2026-07-05 09:48 |
| knitwork_grnn_harmonic H=40 L=3 C=4 ~224K SDQ | 0.7718 | 0.7037 | 0.5030 | 0.9302 | 56M | 2026-07-05 09:46 |
| knitwork_hgrnn H=46 L=3 C=4 text8 +eval | 1.2591 | 0.6133 | - | - | 153M | 2026-07-04 17:40 |
| knitwork_transformer ~2.1M text8 +eval | 2.5101 | 0.2969 | - | - | 25M | 2026-07-04 16:00 |
| knitwork_hgrnn H=128 L=3 C=4 SDQ | 0.1063 | 0.9576 | 0.9320 | 0.9958 | 95M | 2026-07-04 12:40 |
| knitwork_hgrnn H=46 L=3 C=4 ~215K SDQ | 0.1640 | 0.9343 | 0.8953 | 0.9926 | 94M | 2026-07-04 12:40 |
| knitwork_hgrnn H=128 1L2C text8 +eval | 1.2154 | 0.6242 | - | - | 217M | 2026-07-04 12:39 |
| knitwork_hgrnn H=128 L=3 C=4 SDQ | 0.5605 | 0.7793 | 0.6055 | 0.9180 | 19M | 2026-07-04 09:28 |
| knitwork_transformer ~540K text8 +eval | 2.4823 | 0.3006 | - | - | 37M | 2026-07-04 09:06 |
| knitwork_grnn H=128 1L2C text8 +eval | 1.3992 | 0.5795 | - | - | 72M | 2026-07-04 08:17 |
| knitwork_grnn_base ~193K text8 +eval | 1.2782 | 0.6096 | - | - | 84M | 2026-07-04 08:15 |
| knitwork_grnn_base 200K SDQ | 0.2426 | 0.9037 | 0.8620 | 0.9906 | 209M | 2026-07-02 14:19 |
| knitwork_hgrnn_lru 200K SDQ | 0.8740 | 0.6677 | 0.4472 | 0.9486 | 74M | 2026-07-02 09:11 |
| knitwork_grnn_ema_mem 200K SDQ comet | 0.6480 | 0.7457 | 0.5682 | 0.9555 | 73M | 2026-07-01 18:13 |
| knitwork_grnn 200K SDQ comet | 0.6408 | 0.7469 | 0.5596 | 0.9769 | 68M | 2026-07-01 16:30 |
| knitwork_baseline_mlstm_200k | 0.8279 | 0.6931 | 0.4689 | 0.9396 | 56M | 2026-06-27 07:25 |
| knitwork_baseline_hgrn2_200k | 1.1547 | 0.5712 | 0.3329 | 0.7976 | 34M | 2026-06-27 07:25 |
| knitwork_baseline_delta_net_200k | 0.9253 | 0.6571 | 0.4306 | 0.9004 | 55M | 2026-06-27 07:24 |
