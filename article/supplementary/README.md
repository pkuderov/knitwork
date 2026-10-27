# MoSAIC review code

This archive contains the local model implementations, the Text8 training/evaluation runner, shared utilities, and the approximately one-million-parameter configurations used for the main comparison. The code is supplied for reviewing and rerunning this comparison. It contains no credentials, experiment-tracker exports, model checkpoints, or private repository history. Configuration differences from the research workspace are limited to disabling external tracking and replacing the dataset location with `text8.txt`.

Use Python 3.12 or later. From the extracted archive directory, create an environment and install the listed dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`environment.txt` records the installed versions used to check this archive. It is not a record of the software environment of earlier training runs. For CUDA, install the appropriate PyTorch build for the available GPU from the official PyTorch distribution. External tracking packages are optional because the supplied configuration disables them.

Download and extract `text8.zip` from the [dataset provider](https://www.mattmahoney.net/dc/textdata.html), then place the extracted corpus at `text8.txt` or override `gens.text8.path`. The runner takes the first 90M characters for training, the next 5M for validation, and the last 5M for testing, with vocabulary constructed from training data. The larger-scale models were developed using the last 10M characters as validation; therefore the 1M test segment overlaps earlier development data.

Run the accounting check without training:

```bash
python -m knitwork.common.state_size knitwork/exps/text/config/mid.yaml
```

The following are full-budget training commands; they are not quick smoke checks. Replace the model argument to run the other configurations, and repeat with the seed inventory below:

```bash
python -m knitwork.exps.text.run knitwork/exps/text/config/mid.yaml --model=grnn.L2C4 --seed=42 --gens.text8.path=text8.txt --log.logger=None
python -m knitwork.exps.text.run knitwork/exps/text/config/mid.yaml --model=rnn.L3 --seed=42 --gens.text8.path=text8.txt --log.logger=None
```

| Model argument | Seeds reported in the validation comparison |
|---|---|
| `grnn.L3C4`, `grnn.L2C4`, `rnn.L2`, `grnn_lru.L3C4`, `grnn_lru.L2C4` | 42, 1337, 31337 |
| `grnn.L2C8`, `grnn.L1C8`, `rnn.L1`, `rnn.L3`, `grnn_lru.L1C8`, `grnn_lru.L2C8` | 42, 1337 |
| `transformer`, `hgrn2`, `mlstm`, `delta_net`, `rims`, `brims` | 42, 1337 |

The recipe uses 512 training streams, BPTT length 64, float32, RMSprop, norm clipping at 1, and a one-billion-token budget. Model sizes and states differ. The implementations share a recipe rather than reproducing each model family's published optimum.

Final-checkpoint validation BPC carries recurrent state continuously. An additional reset-every-1024-token validation loss selects the checkpoint used for `test/BPC`; its test evaluation also carries continuous state. These checkpoints can differ. Test values are not used for model selection. The manuscript reports available test scores and missing evaluations separately.

The supplementary archive does not contain earlier checkpoints, aligned segment losses, or the missing third seeds. It cannot provide exact replay of the larger-scale results because their seeds were not retained. Anonymous review access and public release are separate: public release of the research code and configurations is planned after acceptance.

The model papers and public datasets are cited in the accompanying manuscript. This archive contains local implementations, not redistributed third-party checkpoints or the Text8 dataset. `MANIFEST.json` lists SHA-256 checksums for the packaged files. No complete license inventory is claimed here.
