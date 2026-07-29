# AAAI Comet Snapshot

Retrieved read-only from Comet workspace `team-rl-exp` at 2026-07-29 02:42 UTC.
This is an exploratory tracker snapshot, not a selected paper result set.
Runs are grouped by the Comet `model` and `model_cfg` parameters, with the user-confirmed legacy `rnn_2L` alias merged into `rnn / rnn_L2`.

## Launch priority: configurations below three replicates

This is an operational coverage ranking, not a performance ranking. Completed and running runs both count toward the three-replicate target. Standard-protocol configurations rank ahead of nonstandard or reduced-budget configurations; within each tier, fewer required launches rank first and SDQ precedes Text8 under the current critical path.

| Rank | Priority | Experiment | Model config | Completed | Running | Counted | New launches to reach 3 | Protocol |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | P1: complete 3-replicate standard group | SDQ | grnn / grnn_L2C4 | 2 | 0 | 2 | 1 | standard |
| 2 | P1: complete 3-replicate standard group | SDQ | rnn / rnn_L2 | 2 | 0 | 2 | 1 | standard |
| 3 | P1: complete 3-replicate standard group | SDQ | rnn / rnn_L3 | 2 | 0 | 2 | 1 | standard |
| 4 | P1: complete 3-replicate standard group | text8 | grnn / grnn_L1C8 | 1 | 1 | 2 | 1 | standard |
| 5 | P1: complete 3-replicate standard group | text8 | grnn / grnn_L2C8 | 2 | 0 | 2 | 1 | standard |
| 6 | P1: complete 3-replicate standard group | text8 | rnn / rnn_L1 | 2 | 0 | 2 | 1 | standard |
| 7 | P1: complete 3-replicate standard group | text8 | transformer / transformer | 1 | 1 | 2 | 1 | standard |
| 8 | P2: standard group needs multiple launches | SDQ | grnn / grnn_L1C8 | 1 | 0 | 1 | 2 | standard |
| 9 | P2: standard group needs multiple launches | SDQ | grnn / grnn_L2C16 | 1 | 0 | 1 | 2 | standard |
| 10 | P2: standard group needs multiple launches | SDQ | grnn / grnn_L2C8 | 1 | 0 | 1 | 2 | standard |
| 11 | P2: standard group needs multiple launches | SDQ | grnn / grnn_L3C4 | 1 | 0 | 1 | 2 | standard |
| 12 | P2: standard group needs multiple launches | text8 | grnn / grnn_L2C16 | 1 | 0 | 1 | 2 | standard |
| 13 | P2: standard group needs multiple launches | text8 | transformer / transformer_64 | 0 | 1 | 1 | 2 | standard |
| 14 | P3: nonstandard or reduced-budget group | SDQ | hgrn2 / hgrn2 | 0 | 2 | 2 | 1 | nonstandard/reduced |
| 15 | P3: nonstandard or reduced-budget group | text8 | delta_net / delta_net | 2 | 0 | 2 | 1 | nonstandard/reduced |
| 16 | P3: nonstandard or reduced-budget group | text8 | hgrn2 / hgrn2 | 1 | 1 | 2 | 1 | nonstandard/reduced |
| 17 | P3: nonstandard or reduced-budget group | SDQ | delta_net / delta_net | 1 | 0 | 1 | 2 | nonstandard/reduced |
| 18 | P3: nonstandard or reduced-budget group | SDQ | mlstm / mlstm | 1 | 0 | 1 | 2 | nonstandard/reduced |
| 19 | P3: nonstandard or reduced-budget group | text8 | mlstm / mlstm | 1 | 0 | 1 | 2 | nonstandard/reduced |

## Per-seed status

`same model/model_cfg` is verified from Comet. `replicate N` is an analysis label: intentional null seeds mean the Comet ID is the stable run identifier.

| Experiment | Model config | Seed | State | Progress | Metrics | Logged budget | Configuration comparability | Obvious anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text8 | delta_net_10.10M (delta_net / delta_net) | replicate 1 (`09e90471`) | finished | 200.0M / 200.0M | val/BPC: best 1.8292; current 1.8443 | 64 envs × 200.0M | same model/model_cfg (Comet); reduced budget | current BPC is 0.015 above best |
| text8 | delta_net_10.10M (delta_net / delta_net) | replicate 2 (`27cc5c7e`) | finished | 200.0M / 200.0M | val/BPC: best 1.7909; current 1.8116 | 64 envs × 200.0M | same model/model_cfg (Comet); reduced budget | current BPC is 0.021 above best |
| text8 | grnn.L1C8_10.11M (grnn / grnn_L1C8) | replicate 1 (`4d8fa3da`) | running | 425.0M / 1000.0M | val/BPC: best 1.5155; current 1.5155 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | state: running |
| text8 | grnn.L1C8_10.11M (grnn / grnn_L1C8) | replicate 2 (`55c0a476`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4689; current 1.4733 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C16_10.09M (grnn / grnn_L2C16) | replicate 1 (`6b99c6b4`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4506; current 1.4564 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C4_10.11M (grnn / grnn_L2C4) | replicate 1 (`250f9d15`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4352; current 1.4391 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C4_10.11M (grnn / grnn_L2C4) | replicate 2 (`6f5b2321`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4397; current 1.4450 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C4_10.11M (grnn / grnn_L2C4) | replicate 3 (`e302a67f`) | finished | 1495.1M / 1500.0M | val/BPC: best 1.4319; current 1.4327 | 512 envs × 1500.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C8_10.17M (grnn / grnn_L2C8) | replicate 1 (`2096ea92`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4387; current 1.4456 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C8_10.17M (grnn / grnn_L2C8) | replicate 2 (`5ea4642f`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4404; current 1.4431 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 1 (`1642ea03`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4286; current 1.4330 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 2 (`3c4487e1`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4322; current 1.4341 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 3 (`fe0228bf`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4327; current 1.4364 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | hgrn2_10.13M (hgrn2 / hgrn2) | replicate 1 (`075ff085`) | running | 3.0M / 100.0M | val/BPC: best —; current — | 32 envs × 100.0M | same model/model_cfg (Comet); reduced budget | state: running; missing val/BPC |
| text8 | hgrn2_10.13M (hgrn2 / hgrn2) | replicate 2 (`0e139079`) | finished | 100.0M / 100.0M | val/BPC: best 1.6728; current 1.6728 | 32 envs × 100.0M | same model/model_cfg (Comet); reduced budget | — |
| text8 | mlstm_10.11M (mlstm / mlstm) | replicate 1 (`61dbc44b`) | finished | 200.0M / 200.0M | val/BPC: best 1.6624; current 1.6856 | 64 envs × 200.0M | same model/model_cfg (Comet); reduced budget | current BPC is 0.023 above best |
| text8 | rnn.L1_10.16M (rnn / rnn_L1) | replicate 1 (`bd167357`) | finished | 1495.1M / 1500.0M | val/BPC: best 1.5522; current 1.5525 | 512 envs × 1500.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L1_10.16M (rnn / rnn_L1) | replicate 2 (`c159bbe1`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.5594; current 1.5606 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L2_10.04M (rnn_2L / —) | replicate 1 (`28f302f7`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4869; current 1.4875 | 512 envs × 1000.0M | legacy alias to rnn / rnn_L2 (user-confirmed); standard budget | — |
| text8 | rnn.L2_10.04M (rnn / rnn_L2) | replicate 2 (`376d8795`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.5098; current 1.5112 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L2_10.04M (rnn / rnn_L2) | replicate 3 (`eb741ad3`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.5017; current 1.5024 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L3_10.23M (rnn / rnn_L3) | replicate 1 (`06eefd39`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4786; current 1.4787 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L3_10.23M (rnn / rnn_L3) | replicate 2 (`8771e623`) | finished | 995.0M / 1000.0M | val/BPC: best 1.4899; current 1.4899 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L3_10.23M (rnn / rnn_L3) | replicate 3 (`981f62ec`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4799; current 1.4805 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | transformer_10.07M (transformer / transformer) | replicate 1 (`385fd659`) | running | 170.5M / 1000.0M | val/BPC: best 1.6809; current 1.6809 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | state: running |
| text8 | transformer_10.07M (transformer / transformer) | replicate 2 (`91402d05`) | finished | 75.2M / 1000.0M | val/BPC: best 1.9231; current 2.5351 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current BPC is 0.612 above best |
| text8 | transformer.64_10.07M (transformer / transformer_64) | replicate 1 (`cc0279fb`) | running | 95.3M / 1000.0M | val/BPC: best 1.8216; current 1.8216 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | state: running |
| SDQ | delta_net_10.12M (delta_net / delta_net) | replicate 1 (`e4583757`) | finished | 250.0M / 250.0M | Acc++: best 0.2892; current 0.1725 | 128 envs × 250.0M | same model/model_cfg (Comet); nonstandard budget | current Acc++ is 0.117 below best |
| SDQ | grnn.L1C8_10.12M (grnn / grnn_L1C8) | replicate 1 (`f5fcd7c7`) | finished | 1000.0M / 1000.0M | Acc++: best 0.8006; current 0.7050 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.096 below best |
| SDQ | grnn.L2C16_10.09M (grnn / grnn_L2C16) | replicate 1 (`d529d07a`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9411; current 0.8866 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.055 below best |
| SDQ | grnn.L2C4_10.12M (grnn / grnn_L2C4) | replicate 1 (`a270cafe`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9243; current 0.8416 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.083 below best |
| SDQ | grnn.L2C4_10.12M (grnn / grnn_L2C4) | replicate 2 (`e96f713f`) | finished | 985.0M / 1000.0M | Acc++: best 0.9061; current 0.8471 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.059 below best |
| SDQ | grnn.L2C8_10.18M (grnn / grnn_L2C8) | replicate 1 (`295cb9db`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9017; current 0.8716 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 1 (`bd180492`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9359; current 0.9311 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | hgrn2_10.14M (hgrn2 / hgrn2) | replicate 1 (`e677376d`) | running | 19.5M / 125.0M | Acc++: best 0.1677; current 0.1677 | 64 envs × 125.0M | same model/model_cfg (Comet); nonstandard budget | state: running |
| SDQ | hgrn2_10.14M (hgrn2 / hgrn2) | replicate 2 (`e8fa8d5c`) | running | 64.5M / 125.0M | Acc++: best 0.1446; current 0.1059 | 64 envs × 125.0M | same model/model_cfg (Comet); nonstandard budget | state: running |
| SDQ | mlstm_10.12M (mlstm / mlstm) | replicate 1 (`fd556d5f`) | finished | 250.0M / 250.0M | Acc++: best 0.1809; current 0.1262 | 128 envs × 250.0M | same model/model_cfg (Comet); nonstandard budget | current Acc++ is 0.055 below best |
| SDQ | rnn.L1_10.18M (rnn / rnn_L1) | replicate 1 (`45901ecb`) | finished | 970.0M / 1000.0M | Acc++: best 0.7067; current 0.6094 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.097 below best |
| SDQ | rnn.L1_10.18M (rnn / rnn_L1) | replicate 2 (`6ce37058`) | finished | 1000.0M / 1000.0M | Acc++: best 0.7104; current 0.6264 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.084 below best |
| SDQ | rnn.L1_10.18M (rnn / rnn_L1) | replicate 3 (`91ca2493`) | finished | 1000.0M / 1000.0M | Acc++: best 0.7627; current 0.6126 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.150 below best |
| SDQ | rnn.L2_10.06M (rnn / rnn_L2) | replicate 1 (`69d27694`) | finished | 980.0M / 1000.0M | Acc++: best 0.4643; current 0.4643 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | rnn.L2_10.06M (rnn / rnn_L2) | replicate 2 (`6dfc8619`) | finished | 990.0M / 1000.0M | Acc++: best 0.6105; current 0.6105 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | rnn.L3_10.25M (rnn / rnn_L3) | replicate 1 (`150aa3b6`) | finished | 990.0M / 1000.0M | Acc++: best 0.3454; current 0.3445 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | rnn.L3_10.25M (rnn / rnn_L3) | replicate 2 (`8db8399c`) | finished | 1000.0M / 1000.0M | Acc++: best 0.3003; current 0.2284 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.072 below best |

## Text8: completed-seed results at the comparable horizon

RNN and GRNN/MoSAIC only. Each run is truncated at 1B tokens; runs with a final logged validation point slightly below 1B are retained under the logging-loss convention. Unfinished runs and reduced-budget baselines are excluded.

| Model config | Completed replicates | Protocol | Final logged point | Val BPC ↓ | Comet IDs |
| --- | ---: | --- | ---: | ---: | --- |
| grnn / grnn_L1C8 | 1 | 1B tokens | 1000.0M | 1.4733 ± 0.0000 | `55c0a476` |
| grnn / grnn_L2C16 | 1 | 1B tokens | 1000.0M | 1.4564 ± 0.0000 | `6b99c6b4` |
| grnn / grnn_L2C4 | 3 | 1B tokens | 980.0M | 1.4367 ± 0.0026 | `250f9d15`, `6f5b2321`, `e302a67f` |
| grnn / grnn_L2C8 | 2 | 1B tokens | 1000.0M | 1.4444 ± 0.0017 | `2096ea92`, `5ea4642f` |
| grnn / grnn_L3C4 | 3 | 1B tokens | 1000.0M | 1.4345 ± 0.0017 | `1642ea03`, `3c4487e1`, `fe0228bf` |
| rnn / rnn_L1 | 2 | 1B tokens | 980.0M | 1.5614 ± 0.0029 | `bd167357`, `c159bbe1` |
| rnn / rnn_L2 | 3 | 1B tokens | 1000.0M | 1.5004 ± 0.0119 | `28f302f7`, `376d8795`, `eb741ad3` |
| rnn / rnn_L3 | 3 | 1B tokens | 980.0M | 1.4828 ± 0.0062 | `06eefd39`, `8771e623`, `981f62ec` |

## Text8: completed-seed best validation checkpoints

This is a separate checkpoint-selection view, not a fixed-horizon comparison. It includes only completed RNN and GRNN/MoSAIC runs, and searches each validation curve only through the 1B-token protocol horizon.

| Model config | Completed replicates | Best val BPC ↓ | Comet IDs |
| --- | ---: | ---: | --- |
| grnn / grnn_L1C8 | 1 | 1.4689 ± 0.0000 | `55c0a476` |
| grnn / grnn_L2C16 | 1 | 1.4506 ± 0.0000 | `6b99c6b4` |
| grnn / grnn_L2C4 | 3 | 1.4367 ± 0.0026 | `250f9d15`, `6f5b2321`, `e302a67f` |
| grnn / grnn_L2C8 | 2 | 1.4395 ± 0.0012 | `2096ea92`, `5ea4642f` |
| grnn / grnn_L3C4 | 3 | 1.4312 ± 0.0022 | `1642ea03`, `3c4487e1`, `fe0228bf` |
| rnn / rnn_L1 | 2 | 1.5614 ± 0.0029 | `bd167357`, `c159bbe1` |
| rnn / rnn_L2 | 3 | 1.4995 ± 0.0116 | `28f302f7`, `376d8795`, `eb741ad3` |
| rnn / rnn_L3 | 3 | 1.4828 ± 0.0062 | `06eefd39`, `8771e623`, `981f62ec` |

## Text8: reduced-token, increased-update baselines

This separate table is not part of the 1B-token RNN/GRNN comparison. Updates equal `n_steps / (n_envs × rollout_len)`; the rollout length is 64 for these runs. The standard 1B-token RNN/GRNN protocol has 30.5k planned updates. Final and best BPC are both shown because this is a completed-run status view, not a fixed-horizon comparison.

| Model config | Tokens | Batch tokens/update | Planned updates | Final val BPC ↓ | Best val BPC ↓ | Comet IDs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| delta_net / delta_net | 200.0M | 4,096 | 48.8k | 1.8280 ± 0.0231 | 1.8101 ± 0.0271 | `09e90471`, `27cc5c7e` |
| hgrn2 / hgrn2 | 100.0M | 2,048 | 48.8k | 1.6728 ± 0.0000 | 1.6728 ± 0.0000 | `0e139079` |
| mlstm / mlstm | 200.0M | 4,096 | 48.8k | 1.6856 ± 0.0000 | 1.6624 ± 0.0000 | `61dbc44b` |

## Store--Distract--Query: completed-replicate final-window results

`Acc++` is the logged online-generator evaluation metric. For each completed standard-protocol replicate, the statistic is the mean of its final five logged `Acc++` values through the 1B-token protocol horizon; the table then reports mean ± standard deviation across replicates. Unfinished and reduced-budget runs are excluded.

| Model config | Completed replicates | Protocol | Final logged point | Final-five Acc++ ↑ | Comet IDs |
| --- | ---: | --- | ---: | ---: | --- |
| grnn / grnn_L1C8 | 1 | 1B tokens | 1000.0M | 0.6998 ± 0.0000 | `f5fcd7c7` |
| grnn / grnn_L2C16 | 1 | 1B tokens | 1000.0M | 0.8901 ± 0.0000 | `d529d07a` |
| grnn / grnn_L2C4 | 2 | 1B tokens | 985.0M | 0.8458 ± 0.0046 | `a270cafe`, `e96f713f` |
| grnn / grnn_L2C8 | 1 | 1B tokens | 1000.0M | 0.8678 ± 0.0000 | `295cb9db` |
| grnn / grnn_L3C4 | 1 | 1B tokens | 1000.0M | 0.9276 ± 0.0000 | `bd180492` |
| rnn / rnn_L1 | 3 | 1B tokens | 970.0M | 0.6177 ± 0.0052 | `45901ecb`, `6ce37058`, `91ca2493` |
| rnn / rnn_L2 | 2 | 1B tokens | 980.0M | 0.5322 ± 0.1027 | `69d27694`, `6dfc8619` |
| rnn / rnn_L3 | 2 | 1B tokens | 990.0M | 0.2865 ± 0.0798 | `150aa3b6`, `8db8399c` |

## Exploratory reading

- The three non-RNN baselines (DeltaNet, HGRN2, and mLSTM) use reduced `n_envs` and `n_steps` because of their memory requirements. The update-indexed text8 panel is the appropriate relative-efficiency view for those baselines.
- SDQ's online generator supplies the reported evaluation metrics, so no separate validation split is expected. Its aggregate uses completed standard-protocol runs only.

The companion figure is `figures/aaai_comet_snapshot.png`.
