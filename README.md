# knitwork

Research project for Grid RNN experiments — associative memory benchmarks and language modeling.

## Setup

```sh
uv sync
```

## Running experiments

All scripts use `uv run <script> <config> [overrides]`. Overrides use dot-notation: `key=value` or `--key=value`.

### SDQ (Store-Distract-Query associative memory)

```sh
uv run knitwork/exps/sdq/run_sdq.py knitwork/exps/sdq/config/extend_config.yaml --model=grnn
uv run knitwork/exps/sdq/run_sdq.py knitwork/exps/sdq/config/extend_config.yaml --model=grnn --name="my run"
uv run knitwork/exps/sdq/run_sdq.py knitwork/exps/sdq/config/extend_config.yaml --model=grnn_fusion --log.enabled=false
```

Available models: `rnn`, `grnn`, `grnn_err`, `hgrnn`, `grnn_fw`, `grnn_reservoir`, `grnn_hgrn`, `grnn2`, `grnn_engram`, `grnn_loss`, `grnn_eq`, `grnn_eq1`, `grnn_disc`, `grnn_adv_loss`, `grnn_fusion`, `grnn_lru`, `grnn_lru_wide`, `grnn_lru_hop`

### Text (shakespeare / text8 language modeling)

```sh
uv run knitwork/exps/text/run_text.py knitwork/exps/text/config/extend_config.yaml --model=grnn
uv run knitwork/exps/text/run_text.py knitwork/exps/text/config/extend_config.yaml --model=hgrnn --name="hgrnn text8 ~78K"
```

Available models: `rnn`, `grnn`, `grnn_err`, `hgrnn`, `grnn2`, `grnn_loss`, `grnn_res`, `grnn_engram`

### TreasureHunt (RL with PPO)

```sh
uv run knitwork/exps/treasure/run_treasure_hunt.py knitwork/exps/treasure/config_treasure_hunt.yaml --model=grnn
uv run knitwork/exps/treasure/run_treasure_hunt.py knitwork/exps/treasure/config_treasure_hunt.yaml --model=grnn_lru
```

Available models: `rnn`, `grnn`, `grnn_lru`, `grnn_lru_wide`

## Common overrides

```sh
--model=<name>          # select model
--name="<run name>"     # AIM run name
--device=cuda|cpu       # default: cuda
--n_steps=1e9
--n_envs=128
--seed=42
--log.enabled=false     # disable AIM logging
--visualize=false       # disable visualizations
```

## Count model parameters

```sh
uv run python -m knitwork.common.count_params --model grnn --input_size 27 --output_size 27
```

## Methods documentation

Docs are in `docs/methods/` and served via GitHub Pages (Docsify).  
To view locally: open `docs/index.html` in a browser or run `npx serve docs`.

## Project structure

```
knitwork/
  common/         # shared utilities: config, logging, scheduler, tracker, entrypoint
  gens/           # data generators: sdq.py, text.py, periodic.py
  env/            # RL environments: treasure_hunt.py
  models/         # model implementations
  exps/
    sdq/          # Store-Distract-Query: run_sdq.py + _viz.py
    text/         # text8/shakespeare: run_text.py
    treasure/     # TreasureHunt PPO RL: run_treasure_hunt.py
  visualization/  # CKA, attention flow
docs/
  methods/        # model documentation (.md per model)
  index.html      # Docsify GitHub Pages site
  _sidebar.md     # navigation
```
