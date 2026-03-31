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


For add name for aim run::
```sh
uv run knitwork/exps/sdq/run.py knitwork/exps/sdq/config/extend_config.yaml     --model=hgrnn  --name="hgrnn baseline"
```

For get count of parameters of model:
```sh
uv run python -m knitwork.common.count_params --model rnn --input_size 27 --output_size 27
```

Compare 3 models om experiments(78k parameters):
```sh
uv run knitwork/exps/text/run1.py knitwork/exps/text/config/extend_config.yaml     --model=grnn --name="grnn text8 ~78K"

uv run knitwork/exps/text/run1.py knitwork/exps/text/config/extend_config.yaml     --model=grnn_err --name="grnn_err text8 ~78K"

uv run knitwork/exps/text/run1.py knitwork/exps/text/config/extend_config.yaml     --model=hgrnn --name="hgrnn text8 ~78K"   
```