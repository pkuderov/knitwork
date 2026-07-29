# AAAI Comet Snapshot

Retrieved read-only from Comet workspace `team-rl-exp` at 2026-07-29 10:06 UTC.
This is an exploratory tracker snapshot, not a selected paper result set.
Runs are grouped by the Comet `model` and `model_cfg` parameters, with the user-confirmed legacy `rnn_2L` alias merged into `rnn / rnn_L2`. Mikasa RL groups additionally require the same Comet `env`.

## Exploratory reading

- The three non-RNN baselines (DeltaNet, HGRN2, and mLSTM) use reduced `n_envs` and `n_steps` because of their memory requirements. Their Text8 results belong in the update-indexed view; their SDQ results are shown separately below and are not directly comparable to the standard 1B-token protocol.
- SDQ's online generator supplies the reported evaluation metrics, so no separate validation split is expected. Each aggregate is the mean of each completed run's final five logged `Acc++` values, followed by mean ± standard deviation across runs.
- Mikasa RL is status-only until task-matched runs complete. Do not compare `env/EpRet` across different POPGym tasks.

The companion figure is `figures/aaai_comet_snapshot.png`; it covers Text8 and SDQ only, not Mikasa RL.

## Text8: completed-seed results at the comparable horizon

RNN, GRNN/MoSAIC, and Transformer only. Each run is truncated at 1B tokens; runs with a final logged validation point slightly below 1B are retained under the logging-loss convention. Runs ending before 950M tokens, unfinished runs, and reduced-budget baselines are excluded.

| Model config | Completed replicates | Protocol | Final logged point | Val BPC ↓ | Comet IDs |
| --- | ---: | --- | ---: | ---: | --- |
| grnn / grnn_L1C8 | 3 | 1B tokens | 1000.0M | 1.4687 ± 0.0046 | `4d8fa3da`, `55c0a476`, `e8976d83` |
| grnn / grnn_L2C16 | 2 | 1B tokens | 1000.0M | 1.4548 ± 0.0023 | `6b99c6b4`, `e1c8f4a8` |
| grnn / grnn_L2C4 | 3 | 1B tokens | 980.0M | 1.4367 ± 0.0026 | `250f9d15`, `6f5b2321`, `e302a67f` |
| grnn / grnn_L2C8 | 3 | 1B tokens | 1000.0M | 1.4452 ± 0.0019 | `2096ea92`, `5ea4642f`, `c67c3ce1` |
| grnn / grnn_L3C4 | 3 | 1B tokens | 1000.0M | 1.4345 ± 0.0017 | `1642ea03`, `3c4487e1`, `fe0228bf` |
| rnn / rnn_L1 | 3 | 1B tokens | 980.0M | 1.5626 ± 0.0028 | `539f02e3`, `bd167357`, `c159bbe1` |
| rnn / rnn_L2 | 3 | 1B tokens | 1000.0M | 1.5004 ± 0.0119 | `28f302f7`, `376d8795`, `eb741ad3` |
| rnn / rnn_L3 | 3 | 1B tokens | 980.0M | 1.4828 ± 0.0062 | `06eefd39`, `8771e623`, `981f62ec` |
| transformer / transformer | 3 | 1B tokens | 1000.0M | 1.4492 ± 0.0120 | `26744915`, `385fd659`, `e32cd50f` |
| transformer / transformer_64 | 3 | 1B tokens | 962.6M | 1.4826 ± 0.0057 | `36f19488`, `95c2b701`, `cc0279fb` |

## Text8: completed-seed best validation checkpoints

This is a separate checkpoint-selection view, not a fixed-horizon comparison. It includes only completed RNN, GRNN/MoSAIC, and Transformer runs meeting the 1B-protocol completion rule, and searches each validation curve only through the 1B-token protocol horizon.

| Model config | Completed replicates | Best val BPC ↓ | Comet IDs |
| --- | ---: | ---: | --- |
| grnn / grnn_L1C8 | 3 | 1.4665 ± 0.0024 | `4d8fa3da`, `55c0a476`, `e8976d83` |
| grnn / grnn_L2C16 | 2 | 1.4485 ± 0.0028 | `6b99c6b4`, `e1c8f4a8` |
| grnn / grnn_L2C4 | 3 | 1.4367 ± 0.0026 | `250f9d15`, `6f5b2321`, `e302a67f` |
| grnn / grnn_L2C8 | 3 | 1.4399 ± 0.0010 | `2096ea92`, `5ea4642f`, `c67c3ce1` |
| grnn / grnn_L3C4 | 3 | 1.4312 ± 0.0022 | `1642ea03`, `3c4487e1`, `fe0228bf` |
| rnn / rnn_L1 | 3 | 1.5626 ± 0.0028 | `539f02e3`, `bd167357`, `c159bbe1` |
| rnn / rnn_L2 | 3 | 1.4995 ± 0.0116 | `28f302f7`, `376d8795`, `eb741ad3` |
| rnn / rnn_L3 | 3 | 1.4828 ± 0.0062 | `06eefd39`, `8771e623`, `981f62ec` |
| transformer / transformer | 3 | 1.4489 ± 0.0114 | `26744915`, `385fd659`, `e32cd50f` |
| transformer / transformer_64 | 3 | 1.4816 ± 0.0049 | `36f19488`, `95c2b701`, `cc0279fb` |

## Text8: reduced-token, increased-update baselines

This separate table is not part of the 1B-token RNN/GRNN comparison. Updates equal `n_steps / (n_envs × rollout_len)`; the rollout length is 64 for these runs. The standard 1B-token RNN/GRNN protocol has 30.5k planned updates. Final and best BPC are both shown because this is a completed-run status view, not a fixed-horizon comparison.

| Model config | Tokens | Batch tokens/update | Planned updates | Final val BPC ↓ | Best val BPC ↓ | Comet IDs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| delta_net / delta_net | 200.0M | 4,096 | 48.8k | 1.8280 ± 0.0231 | 1.8101 ± 0.0271 | `09e90471`, `27cc5c7e` |
| hgrn2 / hgrn2 | 100.0M | 2,048 | 48.8k | 1.6675 ± 0.0076 | 1.6675 ± 0.0076 | `075ff085`, `0e139079` |
| mlstm / mlstm | 200.0M | 4,096 | 48.8k | 1.6797 ± 0.0084 | 1.6553 ± 0.0101 | `61dbc44b`, `af86a53a` |

## Store--Distract--Query: completed-replicate final-window results

`Acc++` is the logged online-generator evaluation metric. For each completed standard-protocol replicate, the statistic is the mean of its final five logged `Acc++` values through the 1B-token protocol horizon; the table then reports mean ± standard deviation across replicates. Unfinished and reduced-budget runs are excluded.

| Model config | Completed replicates | Protocol | Final logged point | Final-five Acc++ ↑ | Comet IDs |
| --- | ---: | --- | ---: | ---: | --- |
| grnn / grnn_L1C8 | 3 | 1B tokens | 1000.0M | 0.7051 ± 0.0163 | `4d99ced1`, `916dc7cb`, `f5fcd7c7` |
| grnn / grnn_L2C16 | 1 | 1B tokens | 1000.0M | 0.8901 ± 0.0000 | `d529d07a` |
| grnn / grnn_L2C4 | 3 | 1B tokens | 985.0M | 0.8433 ± 0.0055 | `78aebd89`, `a270cafe`, `e96f713f` |
| grnn / grnn_L2C8 | 3 | 1B tokens | 1000.0M | 0.8664 ± 0.0099 | `295cb9db`, `a9e72f09`, `df1a63f2` |
| grnn / grnn_L3C4 | 3 | 1B tokens | 1000.0M | 0.9204 ± 0.0128 | `014edb68`, `4f55585a`, `bd180492` |
| rnn / rnn_L1 | 3 | 1B tokens | 970.0M | 0.6177 ± 0.0052 | `45901ecb`, `6ce37058`, `91ca2493` |
| rnn / rnn_L2 | 2 | 1B tokens | 980.0M | 0.5322 ± 0.1027 | `69d27694`, `6dfc8619` |
| rnn / rnn_L3 | 2 | 1B tokens | 990.0M | 0.2865 ± 0.0798 | `150aa3b6`, `8db8399c` |

## Store--Distract--Query: reduced-budget baseline results

Completed DeltaNet, HGRN2, and mLSTM runs only. These use reduced token budgets and different batch/update accounting, so they are reported separately from the standard 1B-token SDQ table.

| Model config | Completed replicates | Tokens | Batch tokens/update | Planned updates | Final logged point | Final-five Acc++ ↑ | Comet IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| delta_net / delta_net | 2 | 250.0M | 4,096 | 61.0k | 250.0M | 0.1529 ± 0.0317 | `e4036e50`, `e4583757` |
| hgrn2 / hgrn2 | 2 | 125.0M | 2,048 | 61.0k | 125.0M | 0.1095 ± 0.0004 | `e677376d`, `e8fa8d5c` |
| mlstm / mlstm | 2 | 250.0M | 4,096 | 61.0k | 250.0M | 0.1239 ± 0.0037 | `2e5ffc55`, `fd556d5f` |

## Launch priority: configurations below three replicates

This is an operational coverage ranking, not a performance ranking. Completed and running runs both count toward the three-replicate target. Standard-protocol configurations rank ahead of nonstandard or reduced-budget configurations; within each tier, fewer required launches rank first and SDQ precedes Text8 under the current critical path. Mikasa RL is intentionally excluded because its task-matched two-replicate matrix has a different coverage target.

| Rank | Priority | Experiment | Model config | Completed | Running | Counted | New launches to reach 3 | Protocol |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | P1: complete 3-replicate standard group | SDQ | rnn / rnn_L2 | 2 | 0 | 2 | 1 | standard |
| 2 | P1: complete 3-replicate standard group | SDQ | rnn / rnn_L3 | 2 | 0 | 2 | 1 | standard |
| 3 | P1: complete 3-replicate standard group | text8 | grnn / grnn_L2C16 | 2 | 0 | 2 | 1 | standard |
| 4 | P3: nonstandard or reduced-budget group | SDQ | delta_net / delta_net | 2 | 0 | 2 | 1 | nonstandard/reduced |
| 5 | P3: nonstandard or reduced-budget group | SDQ | hgrn2 / hgrn2 | 2 | 0 | 2 | 1 | nonstandard/reduced |
| 6 | P3: nonstandard or reduced-budget group | SDQ | mlstm / mlstm | 2 | 0 | 2 | 1 | nonstandard/reduced |
| 7 | P3: nonstandard or reduced-budget group | text8 | delta_net / delta_net | 2 | 0 | 2 | 1 | nonstandard/reduced |
| 8 | P3: nonstandard or reduced-budget group | text8 | hgrn2 / hgrn2 | 2 | 0 | 2 | 1 | nonstandard/reduced |
| 9 | P3: nonstandard or reduced-budget group | text8 | mlstm / mlstm | 2 | 0 | 2 | 1 | nonstandard/reduced |

## Per-seed status

`same model/model_cfg` is verified from Comet. `replicate N` is an analysis label: intentional null seeds mean the Comet ID is the stable run identifier.

| Experiment | Model config | Seed | State | Progress | Metrics | Logged budget | Configuration comparability | Obvious anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text8 | delta_net_10.10M (delta_net / delta_net) | replicate 1 (`09e90471`) | finished | 200.0M / 200.0M | val/BPC: best 1.8292; current 1.8443 | 64 envs × 200.0M | same model/model_cfg (Comet); reduced budget | current BPC is 0.015 above best |
| text8 | delta_net_10.10M (delta_net / delta_net) | replicate 2 (`27cc5c7e`) | finished | 200.0M / 200.0M | val/BPC: best 1.7909; current 1.8116 | 64 envs × 200.0M | same model/model_cfg (Comet); reduced budget | current BPC is 0.021 above best |
| text8 | grnn.L1C8_10.11M (grnn / grnn_L1C8) | replicate 1 (`4d8fa3da`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4666; current 1.4686 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L1C8_10.11M (grnn / grnn_L1C8) | replicate 2 (`55c0a476`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4689; current 1.4733 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L1C8_10.11M (grnn / grnn_L1C8) | replicate 3 (`e8976d83`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4641; current 1.4642 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C16_10.09M (grnn / grnn_L2C16) | replicate 1 (`6b99c6b4`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4506; current 1.4564 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C16_10.09M (grnn / grnn_L2C16) | replicate 2 (`e1c8f4a8`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4465; current 1.4531 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C4_10.11M (grnn / grnn_L2C4) | replicate 1 (`250f9d15`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4352; current 1.4391 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C4_10.11M (grnn / grnn_L2C4) | replicate 2 (`6f5b2321`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4397; current 1.4450 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C4_10.11M (grnn / grnn_L2C4) | replicate 3 (`e302a67f`) | finished | 1495.1M / 1500.0M | val/BPC: best 1.4319; current 1.4327 | 512 envs × 1500.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C8_10.17M (grnn / grnn_L2C8) | replicate 1 (`2096ea92`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4387; current 1.4456 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C8_10.17M (grnn / grnn_L2C8) | replicate 2 (`5ea4642f`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4404; current 1.4431 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C8_10.17M (grnn / grnn_L2C8) | replicate 3 (`c67c3ce1`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4405; current 1.4469 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 1 (`1642ea03`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4286; current 1.4330 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 2 (`3c4487e1`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4322; current 1.4341 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 3 (`fe0228bf`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4327; current 1.4364 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | hgrn2_10.13M (hgrn2 / hgrn2) | replicate 1 (`075ff085`) | finished | 100.0M / 100.0M | val/BPC: best 1.6621; current 1.6621 | 32 envs × 100.0M | same model/model_cfg (Comet); reduced budget | — |
| text8 | hgrn2_10.13M (hgrn2 / hgrn2) | replicate 2 (`0e139079`) | finished | 100.0M / 100.0M | val/BPC: best 1.6728; current 1.6728 | 32 envs × 100.0M | same model/model_cfg (Comet); reduced budget | — |
| text8 | mlstm_10.11M (mlstm / mlstm) | replicate 1 (`61dbc44b`) | finished | 200.0M / 200.0M | val/BPC: best 1.6624; current 1.6856 | 64 envs × 200.0M | same model/model_cfg (Comet); reduced budget | current BPC is 0.023 above best |
| text8 | mlstm_10.11M (mlstm / mlstm) | replicate 2 (`af86a53a`) | finished | 200.0M / 200.0M | val/BPC: best 1.6482; current 1.6738 | 64 envs × 200.0M | same model/model_cfg (Comet); reduced budget | current BPC is 0.026 above best |
| text8 | rnn.L1_10.16M (rnn / rnn_L1) | replicate 1 (`539f02e3`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.5648; current 1.5656 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L1_10.16M (rnn / rnn_L1) | replicate 2 (`bd167357`) | finished | 1495.1M / 1500.0M | val/BPC: best 1.5522; current 1.5525 | 512 envs × 1500.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L1_10.16M (rnn / rnn_L1) | replicate 3 (`c159bbe1`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.5594; current 1.5606 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L2_10.04M (rnn_2L / —) | replicate 1 (`28f302f7`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4869; current 1.4875 | 512 envs × 1000.0M | legacy alias to rnn / rnn_L2 (user-confirmed); standard budget | — |
| text8 | rnn.L2_10.04M (rnn / rnn_L2) | replicate 2 (`376d8795`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.5098; current 1.5112 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L2_10.04M (rnn / rnn_L2) | replicate 3 (`eb741ad3`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.5017; current 1.5024 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L3_10.23M (rnn / rnn_L3) | replicate 1 (`06eefd39`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4786; current 1.4787 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L3_10.23M (rnn / rnn_L3) | replicate 2 (`8771e623`) | finished | 995.0M / 1000.0M | val/BPC: best 1.4899; current 1.4899 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L3_10.23M (rnn / rnn_L3) | replicate 3 (`981f62ec`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4799; current 1.4805 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | transformer_10.07M (transformer / transformer) | replicate 1 (`26744915`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4416; current 1.4416 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | transformer_10.07M (transformer / transformer) | replicate 2 (`385fd659`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4620; current 1.4631 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | transformer_10.07M (transformer / transformer) | replicate 3 (`e32cd50f`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4430; current 1.4430 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | transformer.64_10.07M (transformer / transformer_64) | replicate 1 (`36f19488`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4805; current 1.4805 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | transformer.64_10.07M (transformer / transformer_64) | replicate 2 (`95c2b701`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4869; current 1.4869 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | transformer.64_10.07M (transformer / transformer_64) | replicate 3 (`cc0279fb`) | finished | 977.6M / 1000.0M | val/BPC: best 1.4774; current 1.4774 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | delta_net_10.12M (delta_net / delta_net) | replicate 1 (`e4036e50`) | finished | 250.0M / 250.0M | Acc++: best 0.2156; current 0.1304 | 128 envs × 250.0M | same model/model_cfg (Comet); nonstandard budget | current Acc++ is 0.085 below best |
| SDQ | delta_net_10.12M (delta_net / delta_net) | replicate 2 (`e4583757`) | finished | 250.0M / 250.0M | Acc++: best 0.2892; current 0.1725 | 128 envs × 250.0M | same model/model_cfg (Comet); nonstandard budget | current Acc++ is 0.117 below best |
| SDQ | grnn.L1C8_10.12M (grnn / grnn_L1C8) | replicate 1 (`4d99ced1`) | finished | 1000.0M / 1000.0M | Acc++: best 0.7994; current 0.6968 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.103 below best |
| SDQ | grnn.L1C8_10.12M (grnn / grnn_L1C8) | replicate 2 (`916dc7cb`) | finished | 1000.0M / 1000.0M | Acc++: best 0.7994; current 0.7316 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.068 below best |
| SDQ | grnn.L1C8_10.12M (grnn / grnn_L1C8) | replicate 3 (`f5fcd7c7`) | finished | 1000.0M / 1000.0M | Acc++: best 0.8006; current 0.7050 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.096 below best |
| SDQ | grnn.L2C16_10.09M (grnn / grnn_L2C16) | replicate 1 (`445fe938`) | running | 980.0M / 1000.0M | Acc++: best 0.9404; current 0.8857 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | state: running; current Acc++ is 0.055 below best |
| SDQ | grnn.L2C16_10.09M (grnn / grnn_L2C16) | replicate 2 (`d529d07a`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9411; current 0.8866 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.055 below best |
| SDQ | grnn.L2C16_10.09M (grnn / grnn_L2C16) | replicate 3 (`e913443d`) | running | 915.0M / 1000.0M | Acc++: best 0.9212; current 0.8854 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | state: running |
| SDQ | grnn.L2C4_10.12M (grnn / grnn_L2C4) | replicate 1 (`78aebd89`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9056; current 0.8330 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.073 below best |
| SDQ | grnn.L2C4_10.12M (grnn / grnn_L2C4) | replicate 2 (`a270cafe`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9243; current 0.8416 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.083 below best |
| SDQ | grnn.L2C4_10.12M (grnn / grnn_L2C4) | replicate 3 (`e96f713f`) | finished | 985.0M / 1000.0M | Acc++: best 0.9061; current 0.8471 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.059 below best |
| SDQ | grnn.L2C8_10.18M (grnn / grnn_L2C8) | replicate 1 (`295cb9db`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9017; current 0.8716 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | grnn.L2C8_10.18M (grnn / grnn_L2C8) | replicate 2 (`a9e72f09`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9001; current 0.8618 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | grnn.L2C8_10.18M (grnn / grnn_L2C8) | replicate 3 (`df1a63f2`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9081; current 0.8765 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 1 (`014edb68`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9265; current 0.9124 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 2 (`4f55585a`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9324; current 0.9230 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 3 (`bd180492`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9359; current 0.9311 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | hgrn2_10.14M (hgrn2 / hgrn2) | replicate 1 (`e677376d`) | finished | 125.0M / 125.0M | Acc++: best 0.1677; current 0.1091 | 64 envs × 125.0M | same model/model_cfg (Comet); nonstandard budget | current Acc++ is 0.059 below best |
| SDQ | hgrn2_10.14M (hgrn2 / hgrn2) | replicate 2 (`e8fa8d5c`) | finished | 125.0M / 125.0M | Acc++: best 0.1446; current 0.1104 | 64 envs × 125.0M | same model/model_cfg (Comet); nonstandard budget | — |
| SDQ | mlstm_10.12M (mlstm / mlstm) | replicate 1 (`2e5ffc55`) | finished | 250.0M / 250.0M | Acc++: best 0.1946; current 0.1241 | 128 envs × 250.0M | same model/model_cfg (Comet); nonstandard budget | current Acc++ is 0.071 below best |
| SDQ | mlstm_10.12M (mlstm / mlstm) | replicate 2 (`fd556d5f`) | finished | 250.0M / 250.0M | Acc++: best 0.1809; current 0.1262 | 128 envs × 250.0M | same model/model_cfg (Comet); nonstandard budget | current Acc++ is 0.055 below best |
| SDQ | rnn.L1_10.18M (rnn / rnn_L1) | replicate 1 (`45901ecb`) | finished | 970.0M / 1000.0M | Acc++: best 0.7067; current 0.6094 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.097 below best |
| SDQ | rnn.L1_10.18M (rnn / rnn_L1) | replicate 2 (`6ce37058`) | finished | 1000.0M / 1000.0M | Acc++: best 0.7104; current 0.6264 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.084 below best |
| SDQ | rnn.L1_10.18M (rnn / rnn_L1) | replicate 3 (`91ca2493`) | finished | 1000.0M / 1000.0M | Acc++: best 0.7627; current 0.6126 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.150 below best |
| SDQ | rnn.L2_10.06M (rnn / rnn_L2) | replicate 1 (`69d27694`) | finished | 980.0M / 1000.0M | Acc++: best 0.4643; current 0.4643 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | rnn.L2_10.06M (rnn / rnn_L2) | replicate 2 (`6dfc8619`) | finished | 990.0M / 1000.0M | Acc++: best 0.6105; current 0.6105 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | rnn.L3_10.25M (rnn / rnn_L3) | replicate 1 (`150aa3b6`) | finished | 990.0M / 1000.0M | Acc++: best 0.3454; current 0.3445 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | rnn.L3_10.25M (rnn / rnn_L3) | replicate 2 (`8db8399c`) | finished | 1000.0M / 1000.0M | Acc++: best 0.3003; current 0.2284 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.072 below best |

## Mikasa RL: live per-seed status

This is a tracker-status view only: the new runs are not treated as a completed comparison or paper result. `env/EpRet` is the online mean completed-episode return logged by the runner. The relevant core matrix is task-matched `rnn / rnn_L2` versus `grnn / grnn_L2C4`.

| Task | Model config | Seed | State | Progress | Metrics | Logged budget | Configuration comparability | Obvious anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HigherLowerMedium | grnn.L2C4_HigherLowerMedium_10.10M (grnn / grnn_L2C4) | replicate 1 (`5a6de20e`) | finished | 24.9M / 30.0M | env/EpRet: best 0.3686; current 0.3537 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | finished below 95% of configured steps |
| HigherLowerMedium | grnn.L2C4_HigherLowerMedium_10.10M (grnn / grnn_L2C4) | replicate 2 (`bceadf8c`) | finished | 30.0M / 30.0M | env/EpRet: best 0.4292; current 0.4292 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | — |
| HigherLowerMedium | grnn.L2C4_HigherLowerMedium_10.10M (grnn / grnn_L2C4) | replicate 3 (`c746f325`) | finished | 30.0M / 30.0M | env/EpRet: best 0.3929; current 0.3675 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | — |
| HigherLowerMedium | rnn.L2_HigherLowerMedium_10.01M (rnn / rnn_L2) | replicate 1 (`0ecc98e5`) | finished | 30.0M / 30.0M | env/EpRet: best 0.4918; current 0.2955 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | — |
| HigherLowerMedium | rnn.L2_HigherLowerMedium_10.01M (rnn / rnn_L2) | replicate 2 (`571cbdb9`) | running | 21.8M / 50.0M | env/EpRet: best 0.4904; current 0.3818 | 512 envs × 50.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | state: running |
| HigherLowerMedium | rnn.L2_HigherLowerMedium_10.01M (rnn / rnn_L2) | replicate 3 (`a25445fd`) | finished | 30.0M / 30.0M | env/EpRet: best 0.4846; current 0.2502 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | — |
| HigherLowerMedium | rnn.L2_HigherLowerMedium_10.01M (rnn / rnn_L2) | replicate 4 (`a808fa96`) | finished | 30.0M / 30.0M | env/EpRet: best 0.4929; current 0.3131 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | — |
| RepeatFirstEasy | grnn.L2C4_RepeatFirstEasy_10.10M (grnn / grnn_L2C4) | replicate 1 (`222d9042`) | running | 18.8M / 30.0M | env/EpRet: best 0.9964; current 0.9964 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | state: running |
| RepeatFirstEasy | grnn.L2C4_RepeatFirstEasy_10.10M (grnn / grnn_L2C4) | replicate 2 (`329f0e24`) | finished | 30.0M / 30.0M | env/EpRet: best 0.9968; current 0.5601 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | — |
| RepeatFirstEasy | grnn.L2C4_RepeatFirstEasy_10.10M (grnn / grnn_L2C4) | replicate 3 (`32ea146a`) | finished | 30.0M / 30.0M | env/EpRet: best 0.9967; current 0.2297 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | — |
| RepeatFirstEasy | grnn.L2C4_RepeatFirstEasy_10.10M (grnn / grnn_L2C4) | replicate 4 (`ecde89d7`) | finished | 30.0M / 30.0M | env/EpRet: best 0.9950; current 0.9255 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | — |
| RepeatFirstEasy | rnn.L2_RepeatFirstEasy_10.00M (rnn / rnn_L2) | replicate 1 (`55935152`) | finished | 30.0M / 30.0M | env/EpRet: best 1.0000; current 0.9983 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | — |
| RepeatFirstEasy | rnn.L2_RepeatFirstEasy_10.00M (rnn / rnn_L2) | replicate 2 (`627a3c9e`) | finished | 30.0M / 30.0M | env/EpRet: best 0.9999; current 0.9977 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | — |
| RepeatFirstEasy | rnn.L2_RepeatFirstEasy_10.00M (rnn / rnn_L2) | replicate 3 (`f8600b20`) | finished | 30.0M / 30.0M | env/EpRet: best 0.9993; current 0.9913 | 512 envs × 30.0M | same model/model_cfg (Comet); RL task/budget recorded; compare within task and matching batch settings | — |
