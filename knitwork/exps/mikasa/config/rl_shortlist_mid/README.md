# Active RL shortlist: exactly ten runs

The active queue has six models on RepeatPreviousHard (GRU, plain LRU, Grid-GRU, dense Grid-LRU, dynamic hub, two reads) and four on AutoencodeEasy (GRU, plain LRU, dynamic hub, two reads). Every numbered YAML is a standalone configuration. Historical results, selection rationale and commands are in the [report](../../../../../2026-10-03-mikasa-lru-experiment-plan.md).

Shared settings: seed42, approximately1M parameters, 32 envs × rollout128 =4096 vector slots/update, PPO2, gamma0.995, GAE lambda0.99, normalized readout, KL stop and state refresh. No auxiliary losses or recurrent noise. The first-column output bottleneck remains in Grid models. The default budget is8,388,608 slots per run,83,886,080 total, including reset-only slots; valid transitions are logged separately.

From the repository root, choose one command before training:

```bash
# Ten runs at the standard budget.
bash knitwork/exps/mikasa/run_rl_shortlist.sh
```

Or:

```bash
# The same ten runs at a minimal screening budget instead.
KNITWORK_RL_STEPS=2097152 bash knitwork/exps/mikasa/run_rl_shortlist.sh
```

The script checks that the numbered queue contains exactly ten files and stops on failure. Select another device with `KNITWORK_RL_DEVICE=cuda:1`. The root MLspace YAML uses the same script. There is no automatic seed sweep, separate sanity training or stability sweep. The older `lru_mid` folder is a configuration library outside this queue.

One training seed supports screening rather than significance claims; a2M budget may miss delayed learning on the harder task. The runner has no checkpoint/resume: running both commands starts twenty fresh runs. Parameter counts and CPU forward checks are recorded in [JSON](../../../../../docs/experiments/mikasa_rl_shortlist_sizes.json).
