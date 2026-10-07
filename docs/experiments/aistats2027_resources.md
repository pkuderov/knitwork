# AISTATS 2027: historical resource accounting

Retrieved read-only at 2026-10-01T11:38:35.115077+00:00. The cohort is fixed by run IDs in `results_aaai.md`; this does not update its quality metrics or recruit newer runs.

For each run, the loop time is Σ Δtokens / perf/fps through the last positive logged point at or below 1B tokens. This follows `Logger.flush` and `Timer.fps`: fps is the token increment divided by elapsed loop time since the preceding flush. It includes compilation, training, validation, inspection, logging overhead between flushes, and any resource contention. It is elapsed wall time, not exclusive GPU-hours or FLOPs. Finalization after the last flush is excluded. Endpoints are not extrapolated.

Effective rate is accounted tokens divided by reconstructed loop time, not the arithmetic mean of logged rates. Updates are floor(tokens / (n_envs × rollout_len)); they count reached update boundaries, not confirmed successful optimizer steps. Experiment elapsed time is metadata durationMillis and covers the full recorded run, which can exceed the scored 1B prefix. GPU names are inventory labels; no contention control or peak-memory measurement is implied.

Aggregates are mean ± sample standard deviation across available launches. Tokens and hours are per launch, not sums. Mixed-device group aggregates are inventory summaries, not performance comparisons. Two runs have metadata durations shorter than their integrated fps time; those durations are inconsistent and are not used as paper cost estimates.

| Task | Config | Available / cohort n | Accounted tokens (M) | Update boundaries (k) | Loop elapsed (h) | Effective rate (k tokens/s) | Full experiment elapsed (h) | GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SDQ | delta_net / delta_net | 2/2 | 250.0 ± 0.0 | 61.03 ± 0.00 | 3.261 ± 0.042 | 21.3 ± 0.3 | 3.264 ± 0.041 | NVIDIA H100 80GB HBM3 |
| SDQ | grnn / grnn_L1C8 | 3/3 | 1000.0 ± 0.0 | 61.03 ± 0.00 | 1.126 ± 0.095 | 248.0 ± 22.1 | 1.128 ± 0.095 | NVIDIA H100 80GB HBM3 |
| SDQ | grnn / grnn_L2C16 | 3/3 | 1000.0 ± 0.0 | 61.03 ± 0.00 | 1.639 ± 0.144 | 170.4 ± 15.7 | 1.641 ± 0.144 | NVIDIA H100 80GB HBM3 |
| SDQ | grnn / grnn_L2C4 | 3/3 | 995.0 ± 8.6 | 60.73 ± 0.53 | 2.390 ± 1.142 | 136.1 ± 67.2 | 2.412 ± 1.140 | NVIDIA GeForce RTX 3080 Ti, NVIDIA H100 80GB HBM3, Tesla V100-SXM3-32GB |
| SDQ | grnn / grnn_L2C8 | 3/3 | 1000.0 ± 0.0 | 61.03 ± 0.00 | 1.554 ± 0.138 | 179.7 ± 16.8 | 1.557 ± 0.138 | NVIDIA H100 80GB HBM3 |
| SDQ | grnn / grnn_L3C4 | 3/3 | 1000.0 ± 0.0 | 61.03 ± 0.00 | 1.946 ± 0.311 | 145.4 ± 25.1 | 1.948 ± 0.311 | NVIDIA H100 80GB HBM3 |
| SDQ | hgrn2 / hgrn2 | 2/2 | 125.0 ± 0.0 | 61.03 ± 0.00 | 2.305 ± 0.017 | 15.1 ± 0.1 | 2.307 ± 0.017 | NVIDIA H100 80GB HBM3 |
| SDQ | mlstm / mlstm | 2/2 | 250.0 ± 0.0 | 61.03 ± 0.00 | 3.065 ± 0.026 | 22.7 ± 0.2 | 3.067 ± 0.026 | NVIDIA H100 80GB HBM3 |
| SDQ | rnn / rnn_L1 | 3/3 | 990.0 ± 17.3 | 60.43 ± 1.06 | 1.749 ± 0.689 | 177.8 ± 79.8 | 1.754 ± 0.700 | NVIDIA GeForce RTX 3080 Ti, NVIDIA TITAN RTX, Tesla V100-SXM3-32GB |
| SDQ | rnn / rnn_L2 | 3/3 | 990.0 ± 10.0 | 60.43 ± 0.61 | 1.463 ± 0.755 | 218.5 ± 89.1 | 1.475 ± 0.756 | NVIDIA GeForce RTX 3080 Ti, NVIDIA H100 80GB HBM3, NVIDIA TITAN RTX |
| SDQ | rnn / rnn_L3 | 3/3 | 996.7 ± 5.8 | 60.83 ± 0.35 | 1.871 ± 0.704 | 166.8 ± 75.6 | 1.510 ± 1.322 | NVIDIA GeForce RTX 3080 Ti, NVIDIA TITAN RTX, Tesla V100-SXM3-32GB |
| text8 | delta_net / delta_net | 2/2 | 200.0 ± 0.0 | 48.83 ± 0.00 | 3.403 ± 0.011 | 16.3 ± 0.1 | 3.406 ± 0.011 | NVIDIA H100 80GB HBM3 |
| text8 | grnn / grnn_L1C8 | 3/3 | 1000.0 ± 0.0 | 30.52 ± 0.00 | 1.213 ± 0.097 | 229.9 ± 17.6 | 1.217 ± 0.096 | NVIDIA H100 80GB HBM3 |
| text8 | grnn / grnn_L2C16 | 2/2 | 1000.0 ± 0.0 | 30.52 ± 0.00 | 1.825 ± 0.123 | 152.6 ± 10.3 | 1.828 ± 0.124 | NVIDIA H100 80GB HBM3 |
| text8 | grnn / grnn_L2C4 | 3/3 | 998.3 ± 2.9 | 30.47 ± 0.09 | 3.011 ± 2.403 | 131.0 ± 74.2 | 4.011 ± 4.129 | NVIDIA H100 80GB HBM3, NVIDIA TITAN RTX |
| text8 | grnn / grnn_L2C8 | 3/3 | 1000.0 ± 0.0 | 30.52 ± 0.00 | 1.633 ± 0.146 | 171.0 ± 14.7 | 1.635 ± 0.146 | NVIDIA H100 80GB HBM3 |
| text8 | grnn / grnn_L3C4 | 3/3 | 1000.0 ± 0.0 | 30.52 ± 0.00 | 2.264 ± 0.033 | 122.7 ± 1.8 | 2.267 ± 0.034 | NVIDIA H100 80GB HBM3 |
| text8 | hgrn2 / hgrn2 | 2/2 | 100.0 ± 0.0 | 48.83 ± 0.00 | 2.431 ± 0.006 | 11.4 ± 0.0 | 2.433 ± 0.006 | NVIDIA H100 80GB HBM3 |
| text8 | mlstm / mlstm | 2/2 | 200.0 ± 0.0 | 48.83 ± 0.00 | 3.248 ± 0.027 | 17.1 ± 0.1 | 3.250 ± 0.027 | NVIDIA H100 80GB HBM3 |
| text8 | rnn / rnn_L1 | 3/3 | 998.3 ± 2.9 | 30.47 ± 0.09 | 1.677 ± 1.304 | 230.4 ± 126.5 | 2.148 ± 2.115 | NVIDIA H100 80GB HBM3, NVIDIA TITAN RTX |
| text8 | rnn / rnn_L2 | 3/3 | 1000.0 ± 0.0 | 30.52 ± 0.00 | 2.011 ± 1.582 | 192.9 ± 104.4 | 2.016 ± 1.587 | NVIDIA H100 80GB HBM3, NVIDIA TITAN RTX |
| text8 | rnn / rnn_L3 | 3/3 | 998.3 ± 2.9 | 30.47 ± 0.09 | 2.400 ± 2.008 | 168.8 ± 95.5 | 2.413 ± 2.028 | NVIDIA H100 80GB HBM3, NVIDIA TITAN RTX |
| text8 | transformer / transformer | 3/3 | 1000.0 ± 0.0 | 30.52 ± 0.00 | 2.331 ± 1.658 | 222.2 ± 229.2 | 1.046 ± 1.340 | NVIDIA H100 80GB HBM3, NVIDIA TITAN RTX, Tesla V100-SXM3-32GB |
| text8 | transformer / transformer_64 | 3/3 | 992.5 ± 12.9 | 30.29 ± 0.39 | 0.668 ± 0.525 | 579.0 ± 316.8 | 0.677 ± 0.538 | NVIDIA GeForce RTX 3080 Ti, NVIDIA H100 80GB HBM3 |

## Per-run provenance

| Task | Config | Run prefix | Accounted tokens (M) | Loop elapsed (h) | Effective rate (k tokens/s) | GPU | Duration audit | Status |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| SDQ | delta_net / delta_net | `e4036e50` | 250.000 | 3.2907 | 21.10 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | delta_net / delta_net | `e4583757` | 250.000 | 3.2320 | 21.49 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L1C8 | `4d99ced1` | 1000.000 | 1.0158 | 273.46 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L1C8 | `916dc7cb` | 1000.000 | 1.1816 | 235.09 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L1C8 | `f5fcd7c7` | 1000.000 | 1.1804 | 235.32 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L2C16 | `445fe938` | 1000.000 | 1.6991 | 163.48 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L2C16 | `d529d07a` | 1000.000 | 1.7427 | 159.40 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L2C16 | `e913443d` | 1000.000 | 1.4746 | 188.37 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L2C4 | `78aebd89` | 1000.000 | 1.3271 | 209.32 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L2C4 | `a270cafe` | 1000.000 | 3.5971 | 77.22 | Tesla V100-SXM3-32GB | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L2C4 | `e96f713f` | 985.038 | 2.2450 | 121.88 | NVIDIA GeForce RTX 3080 Ti | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L2C8 | `295cb9db` | 1000.000 | 1.6333 | 170.07 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L2C8 | `a9e72f09` | 1000.000 | 1.6351 | 169.88 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L2C8 | `df1a63f2` | 1000.000 | 1.3951 | 199.12 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L3C4 | `014edb68` | 1000.000 | 2.1948 | 126.56 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L3C4 | `4f55585a` | 1000.000 | 1.5973 | 173.91 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | grnn / grnn_L3C4 | `bd180492` | 1000.000 | 2.0446 | 135.86 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | hgrn2 / hgrn2 | `e677376d` | 125.000 | 2.3164 | 14.99 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | hgrn2 / hgrn2 | `e8fa8d5c` | 125.000 | 2.2930 | 15.14 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | mlstm / mlstm | `2e5ffc55` | 250.000 | 3.0465 | 22.79 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | mlstm / mlstm | `fd556d5f` | 250.000 | 3.0828 | 22.53 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | rnn / rnn_L1 | `45901ecb` | 970.037 | 1.0044 | 268.26 | NVIDIA GeForce RTX 3080 Ti | no shorter-duration anomaly | collected |
| SDQ | rnn / rnn_L1 | `6ce37058` | 1000.000 | 2.3643 | 117.49 | NVIDIA TITAN RTX | no shorter-duration anomaly | collected |
| SDQ | rnn / rnn_L1 | `91ca2493` | 1000.000 | 1.8797 | 147.78 | Tesla V100-SXM3-32GB | no shorter-duration anomaly | collected |
| SDQ | rnn / rnn_L2 | `442850b9` | 1000.000 | 0.9843 | 282.21 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| SDQ | rnn / rnn_L2 | `69d27694` | 980.038 | 2.3342 | 116.63 | NVIDIA TITAN RTX | no shorter-duration anomaly | collected |
| SDQ | rnn / rnn_L2 | `6dfc8619` | 990.038 | 1.0719 | 256.57 | NVIDIA GeForce RTX 3080 Ti | no shorter-duration anomaly | collected |
| SDQ | rnn / rnn_L3 | `150aa3b6` | 990.038 | 2.4784 | 110.97 | NVIDIA TITAN RTX | no shorter-duration anomaly | collected |
| SDQ | rnn / rnn_L3 | `8db8399c` | 1000.000 | 2.0342 | 136.55 | Tesla V100-SXM3-32GB | no shorter-duration anomaly | collected |
| SDQ | rnn / rnn_L3 | `f90b4373` | 1000.000 | 1.0990 | 252.76 | NVIDIA GeForce RTX 3080 Ti | inconsistent | collected |
| text8 | delta_net / delta_net | `09e90471` | 200.000 | 3.3956 | 16.36 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | delta_net / delta_net | `27cc5c7e` | 200.000 | 3.4109 | 16.29 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L1C8 | `4d8fa3da` | 1000.000 | 1.1409 | 243.48 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L1C8 | `55c0a476` | 1000.000 | 1.3229 | 209.98 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L1C8 | `e8976d83` | 1000.000 | 1.1757 | 236.26 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L2C16 | `6b99c6b4` | 1000.000 | 1.9123 | 145.26 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L2C16 | `e1c8f4a8` | 1000.000 | 1.7377 | 159.85 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L2C4 | `250f9d15` | 1000.000 | 1.4600 | 190.26 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L2C4 | `6f5b2321` | 1000.000 | 1.7941 | 154.83 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L2C4 | `e302a67f` | 995.038 | 5.7798 | 47.82 | NVIDIA TITAN RTX | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L2C8 | `2096ea92` | 1000.000 | 1.5207 | 182.66 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L2C8 | `5ea4642f` | 1000.000 | 1.7986 | 154.44 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L2C8 | `c67c3ce1` | 1000.000 | 1.5795 | 175.87 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L3C4 | `1642ea03` | 1000.000 | 2.2542 | 123.23 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L3C4 | `3c4487e1` | 1000.000 | 2.3012 | 120.71 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | grnn / grnn_L3C4 | `fe0228bf` | 1000.000 | 2.2367 | 124.19 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | hgrn2 / hgrn2 | `075ff085` | 100.000 | 2.4268 | 11.45 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | hgrn2 / hgrn2 | `0e139079` | 100.000 | 2.4356 | 11.40 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | mlstm / mlstm | `61dbc44b` | 200.000 | 3.2290 | 17.21 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | mlstm / mlstm | `af86a53a` | 200.000 | 3.2672 | 17.00 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | rnn / rnn_L1 | `539f02e3` | 1000.000 | 0.8524 | 325.89 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | rnn / rnn_L1 | `bd167357` | 995.038 | 3.1804 | 86.91 | NVIDIA TITAN RTX | no shorter-duration anomaly | collected |
| text8 | rnn / rnn_L1 | `c159bbe1` | 1000.000 | 0.9979 | 278.36 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | rnn / rnn_L2 | `28f302f7` | 1000.000 | 3.8373 | 72.39 | NVIDIA TITAN RTX | no shorter-duration anomaly | collected |
| text8 | rnn / rnn_L2 | `376d8795` | 1000.000 | 1.0863 | 255.71 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | rnn / rnn_L2 | `eb741ad3` | 1000.000 | 1.1086 | 250.57 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | rnn / rnn_L3 | `06eefd39` | 1000.000 | 1.2508 | 222.08 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | rnn / rnn_L3 | `8771e623` | 995.038 | 4.7183 | 58.58 | NVIDIA TITAN RTX | no shorter-duration anomaly | collected |
| text8 | rnn / rnn_L3 | `981f62ec` | 1000.000 | 1.2301 | 225.81 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | transformer / transformer | `26744915` | 999.981 | 3.8643 | 71.88 | NVIDIA TITAN RTX | inconsistent | collected |
| text8 | transformer / transformer | `385fd659` | 999.981 | 2.5558 | 108.68 | Tesla V100-SXM3-32GB | no shorter-duration anomaly | collected |
| text8 | transformer / transformer | `e32cd50f` | 999.981 | 0.5715 | 486.07 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | transformer / transformer_64 | `36f19488` | 999.981 | 0.3624 | 766.46 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | transformer / transformer_64 | `95c2b701` | 999.981 | 0.3668 | 757.36 | NVIDIA H100 80GB HBM3 | no shorter-duration anomaly | collected |
| text8 | transformer / transformer_64 | `cc0279fb` | 977.633 | 1.2735 | 213.25 | NVIDIA GeForce RTX 3080 Ti | no shorter-duration anomaly | collected |
