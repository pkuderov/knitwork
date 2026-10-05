# Text8 mid LRU configurations

Run a configuration from the repository root:

```bash
uv run python -m knitwork.exps.text.run knitwork/exps/text/config/lru_mid/dynamic_hub_top.yaml
```

The eight follow-up files are `dynamic_hub_top.yaml`, `dynamic_hub_zero.yaml`, `dynamic_hub_self_heavy.yaml`, `dense_small.yaml`, `block_wide.yaml`, `block_dense_hub.yaml`, `clockwork_light.yaml` and `clockwork_output.yaml`. They use the existing Text8 mid training protocol and the `knitwork-aistat` Comet project. The older eight configurations remain available as controls and first-series experiments.

See [mechanisms, parameter budgets and MQAR equivalents](../../../../../docs/experiments/lru_followup.md). Use `--seed=1337` for a repeat and `--gens.text8.path=/absolute/path/to/text8.txt` for a dataset override. These commands launch full training.
