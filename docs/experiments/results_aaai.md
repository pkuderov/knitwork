# AAAI Comet Snapshot

Retrieved read-only from Comet workspace `team-rl-exp` at 2026-07-29 00:30 UTC.
This is an exploratory tracker snapshot, not a selected paper result set.
Runs are grouped by the Comet `model` and `model_cfg` parameters, with the user-confirmed legacy `rnn_2L` alias merged into `rnn / rnn_L2`.

## Per-seed status

`same model/model_cfg` is verified from Comet. `replicate N` is an analysis label: intentional null seeds mean the Comet ID is the stable run identifier.

| Experiment | Model config | Seed | State | Progress | Metrics | Logged budget | Configuration comparability | Obvious anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text8 | delta_net_10.10M (delta_net / delta_net) | replicate 1 (`09e90471`) | finished | 200.0M / 200.0M | val/BPC: best 1.8292; current 1.8443 | 64 envs × 200.0M | same model/model_cfg (Comet); reduced budget | current BPC is 0.015 above best |
| text8 | delta_net_10.10M (delta_net / delta_net) | replicate 2 (`27cc5c7e`) | running | 75.0M / 200.0M | val/BPC: best 2.1031; current 2.1031 | 64 envs × 200.0M | same model/model_cfg (Comet); reduced budget | state: running |
| text8 | grnn.L1C8_10.11M (grnn / grnn_L1C8) | replicate 1 (`55c0a476`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4689; current 1.4733 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C16_10.09M (grnn / grnn_L2C16) | replicate 1 (`6b99c6b4`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4506; current 1.4564 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C4_10.11M (grnn / grnn_L2C4) | replicate 1 (`250f9d15`) | running | 875.0M / 1000.0M | val/BPC: best 1.4410; current 1.4410 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | state: running |
| text8 | grnn.L2C4_10.11M (grnn / grnn_L2C4) | replicate 2 (`6f5b2321`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4397; current 1.4450 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C4_10.11M (grnn / grnn_L2C4) | replicate 3 (`e302a67f`) | finished | 1495.1M / 1500.0M | val/BPC: best 1.4319; current 1.4327 | 512 envs × 1500.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L2C8_10.17M (grnn / grnn_L2C8) | replicate 1 (`5ea4642f`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4404; current 1.4431 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 1 (`1642ea03`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4286; current 1.4330 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 2 (`3c4487e1`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4322; current 1.4341 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 3 (`fe0228bf`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4327; current 1.4364 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | hgrn2_10.13M (hgrn2 / hgrn2) | replicate 1 (`0e139079`) | finished | 100.0M / 100.0M | val/BPC: best 1.6728; current 1.6728 | 32 envs × 100.0M | same model/model_cfg (Comet); reduced budget | — |
| text8 | mlstm_10.11M (mlstm / mlstm) | replicate 1 (`61dbc44b`) | finished | 200.0M / 200.0M | val/BPC: best 1.6624; current 1.6856 | 64 envs × 200.0M | same model/model_cfg (Comet); reduced budget | current BPC is 0.023 above best |
| text8 | rnn.L1_10.16M (rnn / rnn_L1) | replicate 1 (`bd167357`) | finished | 1495.1M / 1500.0M | val/BPC: best 1.5522; current 1.5525 | 512 envs × 1500.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L1_10.16M (rnn / rnn_L1) | replicate 2 (`c159bbe1`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.5594; current 1.5606 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L2_10.04M (rnn / rnn_L2) | replicate 1 (`376d8795`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.5098; current 1.5112 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L2_10.04M (rnn / rnn_L2) | replicate 2 (`eb741ad3`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.5017; current 1.5024 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L3_10.23M (rnn / rnn_L3) | replicate 1 (`06eefd39`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4786; current 1.4787 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L3_10.23M (rnn / rnn_L3) | replicate 2 (`8771e623`) | finished | 995.0M / 1000.0M | val/BPC: best 1.4899; current 1.4899 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| text8 | rnn.L3_10.23M (rnn / rnn_L3) | replicate 3 (`981f62ec`) | running | 575.0M / 1000.0M | val/BPC: best 1.4993; current 1.4997 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | state: running |
| text8 | rnn.L2_10.04M (rnn_2L / —) | replicate 1 (`28f302f7`) | finished | 1000.0M / 1000.0M | val/BPC: best 1.4869; current 1.4875 | 512 envs × 1000.0M | legacy alias to rnn / rnn_L2 (user-confirmed); standard budget | — |
| SDQ | delta_net_10.12M (delta_net / delta_net) | replicate 1 (`e4583757`) | running | 171.3M / 250.0M | Acc++: best 0.2892; current 0.1604 | 128 envs × 250.0M | same model/model_cfg (Comet); nonstandard budget | state: running; current Acc++ is 0.129 below best |
| SDQ | grnn.L1C8_10.12M (grnn / grnn_L1C8) | replicate 1 (`f5fcd7c7`) | finished | 1000.0M / 1000.0M | Acc++: best 0.8006; current 0.7050 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.096 below best |
| SDQ | grnn.L2C16_10.09M (grnn / grnn_L2C16) | replicate 1 (`d529d07a`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9411; current 0.8866 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.055 below best |
| SDQ | grnn.L2C4_10.12M (grnn / grnn_L2C4) | replicate 1 (`a270cafe`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9243; current 0.8416 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.083 below best |
| SDQ | grnn.L2C4_10.12M (grnn / grnn_L2C4) | replicate 2 (`e96f713f`) | finished | 985.0M / 1000.0M | Acc++: best 0.9061; current 0.8471 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.059 below best |
| SDQ | grnn.L2C8_10.18M (grnn / grnn_L2C8) | replicate 1 (`295cb9db`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9017; current 0.8716 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | grnn.L3C4_9.99M (grnn / grnn_L3C4) | replicate 1 (`bd180492`) | finished | 1000.0M / 1000.0M | Acc++: best 0.9359; current 0.9311 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | mlstm_10.12M (mlstm / mlstm) | replicate 1 (`fd556d5f`) | running | 101.3M / 250.0M | Acc++: best 0.1809; current 0.1134 | 128 envs × 250.0M | same model/model_cfg (Comet); nonstandard budget | state: running; current Acc++ is 0.067 below best |
| SDQ | rnn.L1_10.18M (rnn / rnn_L1) | replicate 1 (`45901ecb`) | finished | 970.0M / 1000.0M | Acc++: best 0.7067; current 0.6094 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.097 below best |
| SDQ | rnn.L1_10.18M (rnn / rnn_L1) | replicate 2 (`6ce37058`) | finished | 1000.0M / 1000.0M | Acc++: best 0.7104; current 0.6264 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.084 below best |
| SDQ | rnn.L1_10.18M (rnn / rnn_L1) | replicate 3 (`91ca2493`) | finished | 1000.0M / 1000.0M | Acc++: best 0.7627; current 0.6126 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | current Acc++ is 0.150 below best |
| SDQ | rnn.L2_10.06M (rnn / rnn_L2) | replicate 1 (`69d27694`) | finished | 980.0M / 1000.0M | Acc++: best 0.4643; current 0.4643 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | rnn.L2_10.06M (rnn / rnn_L2) | replicate 2 (`6dfc8619`) | finished | 990.0M / 1000.0M | Acc++: best 0.6105; current 0.6105 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | rnn.L3_10.25M (rnn / rnn_L3) | replicate 1 (`150aa3b6`) | finished | 990.0M / 1000.0M | Acc++: best 0.3454; current 0.3445 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | — |
| SDQ | rnn.L3_10.25M (rnn / rnn_L3) | replicate 2 (`8db8399c`) | running | 885.0M / 1000.0M | Acc++: best 0.3003; current 0.2171 | 512 envs × 1000.0M | same model/model_cfg (Comet); standard budget | state: running; current Acc++ is 0.083 below best |

## Text8: completed-replicate results at the comparable horizon

RNN and GRNN/MoSAIC only. Each run is truncated at 1B tokens; runs with a final logged validation point slightly below 1B are retained under the logging-loss convention. Unfinished runs and reduced-budget baselines are excluded.

| Model config | Completed replicates | Common logged horizon | Val BPC ↓ | Comet IDs |
| --- | ---: | ---: | ---: | --- |
| grnn / grnn_L1C8 | 1 | 1000.0M | 1.4733 ± 0.0000 | `55c0a476` |
| grnn / grnn_L2C16 | 1 | 1000.0M | 1.4564 ± 0.0000 | `6b99c6b4` |
| grnn / grnn_L2C4 | 2 | 980.0M | 1.4374 ± 0.0033 | `6f5b2321`, `e302a67f` |
| grnn / grnn_L2C8 | 1 | 1000.0M | 1.4431 ± 0.0000 | `5ea4642f` |
| grnn / grnn_L3C4 | 3 | 1000.0M | 1.4345 ± 0.0017 | `1642ea03`, `3c4487e1`, `fe0228bf` |
| rnn / rnn_L1 | 2 | 980.0M | 1.5614 ± 0.0029 | `bd167357`, `c159bbe1` |
| rnn / rnn_L2 | 3 | 1000.0M | 1.5004 ± 0.0119 | `28f302f7`, `376d8795`, `eb741ad3` |
| rnn / rnn_L3 | 2 | 980.0M | 1.4843 ± 0.0080 | `06eefd39`, `8771e623` |

## Text8: completed-replicate best validation checkpoints

This is a separate checkpoint-selection view, not a fixed-horizon comparison. It includes only completed RNN and GRNN/MoSAIC runs; best checkpoints may occur before or after 1B tokens.

| Model config | Completed replicates | Best val BPC ↓ | Comet IDs |
| --- | ---: | ---: | --- |
| grnn / grnn_L1C8 | 1 | 1.4689 ± 0.0000 | `55c0a476` |
| grnn / grnn_L2C16 | 1 | 1.4506 ± 0.0000 | `6b99c6b4` |
| grnn / grnn_L2C4 | 2 | 1.4358 ± 0.0055 | `6f5b2321`, `e302a67f` |
| grnn / grnn_L2C8 | 1 | 1.4404 ± 0.0000 | `5ea4642f` |
| grnn / grnn_L3C4 | 3 | 1.4312 ± 0.0022 | `1642ea03`, `3c4487e1`, `fe0228bf` |
| rnn / rnn_L1 | 2 | 1.5558 ± 0.0051 | `bd167357`, `c159bbe1` |
| rnn / rnn_L2 | 3 | 1.4995 ± 0.0116 | `28f302f7`, `376d8795`, `eb741ad3` |
| rnn / rnn_L3 | 2 | 1.4843 ± 0.0080 | `06eefd39`, `8771e623` |

## Text8: reduced-token, increased-update baselines

This separate table is not part of the 1B-token RNN/GRNN comparison. Updates equal `n_steps / (n_envs × rollout_len)`; the rollout length is 64 for these runs. The standard 1B-token RNN/GRNN protocol has 30.5k planned updates. Final and best BPC are both shown because this is a completed-run status view, not a fixed-horizon comparison.

| Model config | Tokens | Batch tokens/update | Planned updates | Final val BPC ↓ | Best val BPC ↓ | Comet IDs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| delta_net / delta_net | 200.0M | 4,096 | 48.8k | 1.8443 ± 0.0000 | 1.8292 ± 0.0000 | `09e90471` |
| hgrn2 / hgrn2 | 100.0M | 2,048 | 48.8k | 1.6728 ± 0.0000 | 1.6728 ± 0.0000 | `0e139079` |
| mlstm / mlstm | 200.0M | 4,096 | 48.8k | 1.6856 ± 0.0000 | 1.6624 ± 0.0000 | `61dbc44b` |

## Store--Distract--Query (`knitwork-sdq`)

`Acc++` is the logged online-generator evaluation metric. Curves aggregate runs with the same `model` and `model_cfg`; the table uses the peak of the group mean curve.

| Model config | Seeds | Shared horizon | Peak mean Acc++ ↑ | Seed IDs |
| --- | ---: | ---: | ---: | --- |
| delta_net / delta_net | 1 | 172.5M | 0.2890 ± 0.0000 | `e4583757` |
| grnn / grnn_L1C8 | 1 | 1000.0M | 0.8001 ± 0.0000 | `f5fcd7c7` |
| grnn / grnn_L2C16 | 1 | 1000.0M | 0.9404 ± 0.0000 | `d529d07a` |
| grnn / grnn_L2C4 | 2 | 985.0M | 0.9128 ± 0.0112 | `a270cafe`, `e96f713f` |
| grnn / grnn_L2C8 | 1 | 1000.0M | 0.9012 ± 0.0000 | `295cb9db` |
| grnn / grnn_L3C4 | 1 | 1000.0M | 0.9311 ± 0.0000 | `bd180492` |
| mlstm / mlstm | 1 | 101.3M | 0.1809 ± 0.0000 | `fd556d5f` |
| rnn / rnn_L1 | 3 | 970.0M | 0.7243 ± 0.0305 | `45901ecb`, `6ce37058`, `91ca2493` |
| rnn / rnn_L2 | 2 | 980.0M | 0.5327 ± 0.1058 | `69d27694`, `6dfc8619` |
| rnn / rnn_L3 | 2 | 885.0M | 0.2902 ± 0.0134 | `150aa3b6`, `8db8399c` |

## Exploratory reading

- The three non-RNN baselines (DeltaNet, HGRN2, and mLSTM) use reduced `n_envs` and `n_steps` because of their memory requirements. The update-indexed text8 panel is the appropriate relative-efficiency view for those baselines.
- SDQ completion ranges from 140M to 1B steps. Its online generator supplies the reported evaluation metrics, so no separate validation split is expected.

The companion figure is `figures/aaai_comet_snapshot.png`.
