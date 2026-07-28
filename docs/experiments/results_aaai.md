# AAAI Comet Snapshot

Retrieved read-only from Comet workspace `team-rl-exp` at 2026-07-28 14:23 UTC.
This is an exploratory tracker snapshot, not a selected paper result set.
Runs with the same displayed name are treated as seeds of a frozen configuration.

## text8 (`knitwork-text`)

Text8 curves are truncated at 1B training steps. Means and standard deviations are computed over seeds at each shared curve position; a one-seed group has zero plotted uncertainty.

| Model config | Seeds | Shared horizon | Final mean val BPC ↓ | Seed IDs |
| --- | ---: | ---: | ---: | --- |
| delta_net_10.10M | 1 | 200.0M | 1.8443 ± 0.0000 | `09e90471` |
| grnn.L1C8_10.11M | 1 | 1000.0M | 1.4733 ± 0.0000 | `55c0a476` |
| grnn.L2C16_10.09M | 1 | 1000.0M | 1.4564 ± 0.0000 | `6b99c6b4` |
| grnn.L2C4_10.11M | 2 | 980.0M | 1.4374 ± 0.0033 | `6f5b2321`, `e302a67f` |
| grnn.L2C8_10.17M | 1 | 1000.0M | 1.4431 ± 0.0000 | `5ea4642f` |
| grnn.L3C4_9.99M | 1 | 1000.0M | 1.4330 ± 0.0000 | `1642ea03` |
| hgrn2_10.13M | 1 | 100.0M | 1.6728 ± 0.0000 | `0e139079` |
| mlstm_10.11M | 1 | 200.0M | 1.6856 ± 0.0000 | `61dbc44b` |
| rnn.L1_10.16M | 2 | 980.0M | 1.5614 ± 0.0029 | `bd167357`, `c159bbe1` |
| rnn.L2_10.04M | 1 | 1000.0M | 1.5024 ± 0.0000 | `eb741ad3` |
| rnn.L3_10.23M | 2 | 980.0M | 1.4843 ± 0.0080 | `06eefd39`, `8771e623` |
| rnn_2L_10.13M | 1 | 1000.0M | 1.4875 ± 0.0000 | `28f302f7` |

## Store--Distract--Query (`knitwork-sdq`)

`Acc++` is reported exactly as logged by the tracker. Curves aggregate seeds with the same model name; the table uses the peak of the group mean curve.

| Model config | Seeds | Shared horizon | Peak mean Acc++ ↑ | Seed IDs |
| --- | ---: | ---: | ---: | --- |
| grnn.L2C4_10.12M | 2 | 585.0M | 0.9128 ± 0.0127 | `a270cafe`, `e96f713f` |
| grnn.L2C8_10.18M | 1 | 440.0M | 0.9013 ± 0.0000 | `295cb9db` |
| rnn.L1_10.18M | 1 | 1000.0M | 0.7565 ± 0.0000 | `084e65b6` |
| rnn.L2_10.06M | 1 | 860.0M | 0.4353 ± 0.0000 | `69d27694` |

## Exploratory reading

- The three non-RNN baselines (DeltaNet, HGRN2, and mLSTM) use reduced `n_envs` and `n_steps` because of their memory requirements. The update-indexed text8 panel is the appropriate relative-efficiency view for those baselines.
- SDQ completion ranges from 140M to 1B steps. These runs expose training metrics but no separately named validation metrics in Comet.

The companion figure is `figures/aaai_comet_snapshot.png`.
