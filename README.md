# knitwork

Experiments are in `knitwork/exps/`. Run:

```sh
# SDQ experiment
uv run sdq/run.py sdq/config/base.yaml --model=rnn
uv run sdq/run.py sdq/config/base.yaml --model=grnn

# Text experiment
uv run text/run.py text/config/base.yaml --model=rnn
uv run text/run.py text/config/base.yaml --model=grnn
```
