# Grid-LRU RL configuration library

**The active recommendation is now exactly ten runs in [rl_shortlist_mid](../rl_shortlist_mid/README.md). This folder preserves previously prepared options; its configs are not additional queued experiments.**

Current runner: `uv run python -m knitwork.exps.mikasa.run`. Execute from the repository root, using the corrected core API; `run_mikasa.py` and the `aicenter3` copy are legacy paths.

See the [experiment report and commands](../../../../../2026-10-03-mikasa-lru-experiment-plan.md) for historical results, baseline priorities, task selection and stability controls.

Twelve model configs share PPO v2, seed42, 4096 vector slots per rollout, a 2,097,152-slot screening budget, deterministic recurrent transitions, small actor-head initialization, normalized readout, KL stopping and rollout-state refresh. Categorical actions remain stochastic. Observation encoding provides one-hot discrete components, previous action and reward; discrete action conversion supports POPGym Battleship. Actual valid transitions are logged separately from vector slots.

`rollout/` provides matched-batch 256/512 controls for Grid-GRU, Grid-LRU, GRU and plain LRU. `stability/` changes one factor at a time; `long_gae_r512` compares with the rollout512 control, and `no_fb_norm` is only a short diagnostic. `obs_only` changes the information supplied to the policy and needs corresponding baseline controls.

Run a technical check without external logging before submitting a job:

```bash
uv run python -m knitwork.exps.mikasa.run knitwork/exps/mikasa/config/lru_mid/grnn_lru.yaml \
  --device=cpu --compile=false --n_envs=2 --rollout_len=16 --n_steps=128 \
  --eval.enabled=false --log.logger=None
```

A 2M-slot pilot tests early progress and stability; it is insufficient to reject a model on difficult tasks. Confirm selected comparisons at larger equal budgets and multiple seeds. New protocols are not directly comparable to historical large-model online returns or external-paper results.
