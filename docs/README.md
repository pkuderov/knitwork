# knitwork — methods

Documentation for methods and architectures from `knitwork/models/`.

Select a model in the sidebar on the left.

## Start here (AISTATS 2027)

| Page | Contents |
|---|---|
| [Grid-GRU (MoSAIC)](methods/grid_gru.md) | main architecture: grid of GRU columns with an attention router |
| [Grid-LRU](methods/grid_lru.md) | LRU-cell grid, language-model and RL versions |
| [AISTATS results](experiments/aistats2027_results.md) | all tables at 1M and 10M parameters, communication study, MQAR, costs |
| [Ablations](experiments/ablations.md) | every controlled ablation in one place |
| [RL experiments](experiments/rl_2026-10.md) | POPGym study, 181 runs: what worked and what did not |
| [Hypothesis registry](experiments/hypotheses.md) | confirmed, rejected and open hypotheses |

## Categories

| Category | Models |
|---|---|
| Basics | GRU, GridRNN, GridRNN-err |
| Cell Modifications | GridRNN2+VAE, HGRNN, HGRN-GridRNN, LRU, GridLRU, HopfieldGridLRU |
| Memory | Engram, Fast Weights, Reservoir |
| Iterative | Equilibrium, Equilibrium v2 |
| Regularization | DiversityLoss, GridRNN-Loss, GridRNN-AdvLoss, GridRNN-Disc |
| Fusion | Fusion cells, GridRNN-Fusion |
