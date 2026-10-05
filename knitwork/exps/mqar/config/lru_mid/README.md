# Standalone MQAR mid suite

Run from the repository root:

```bash
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_mid/grnn.yaml
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_mid/grnn_lru.yaml
uv run python -m knitwork.exps.mqar.run knitwork/exps/mqar/config/lru_mid/dynamic_hub_top.yaml
```

There are 20 configurations with identical data/training/evaluation protocols. Controls are `grnn.yaml`, `grnn_core_mid.yaml`, `grnn_lru.yaml`, `gru.yaml` and sparse-class `control.yaml`. The other files select routing, projection and clockwork variants. Logging is local; add `--log.logger=comet` when launching to an external tracker.

See [mechanisms, parameter budgets and the complete suite](../../../../../docs/experiments/lru_followup.md). For a tiny local check use `knitwork/exps/mqar/config/smoke.yaml --model=grnn.L2C4` or another model selector from the suite. The standalone mid files launch full training.
