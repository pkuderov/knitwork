# LRU architecture experiments, mid

The five files are `write_gate.yaml`, `query_read.yaml`, `associative.yaml`, `rotations.yaml` and `two_reads.yaml`. They share the L2C4/H180 star-connected dynamic-hub base; `dynamic_control.yaml` supplies the common comparison. Predictions use only column 0 and training uses the task loss.

```bash
uv run python -m knitwork.exps.aliased_cubes.run knitwork/exps/aliased_cubes/config/lru_architecture_mid/write_gate.yaml
```

This command launches full training. See [mechanisms, budgets, controls, and launch/smoke instructions](../../../../../docs/experiments/lru_architecture_variants.md).

The configured cube surface is provisional and does not reproduce the original ICML-2024 graph.
