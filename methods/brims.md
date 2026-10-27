# BRIMs (Bidirectional RIMs)

Hierarchical extension of [RIMs](rims.md) (Mittal et al., ICML 2020, arXiv 2006.16981). Several RIMs layers are stacked, and the
modules of each layer attend over the modules of the layer below (bottom-up, current step) **and** the modules of the layer
above (top-down, previous step), plus a null input. Implemented as `BrimsCore` in `baseline/rims.py` (`top_down = True`);
the layer code is shared with RIMs.

## Key mechanism

```python
srcs = [below]                                  # bottom-up: x for layer 0, modules of layer l-1 otherwise
if li < n_layers - 1:
    srcs.append(h_old[li + 1])                  # top-down: previous-step modules of the layer above
h, c = layer(srcs, h_old[li], c_old[li])        # same top-k active-module update as RIMs
```

State per sequence: `2 * L * n_modules * module_size` (h and c). The model is structurally close to the grid RNN
(layers x columns with bottom-up and delayed top-down signals), but modules are LSTM cells with top-k conditional updates instead of
dense routing between GRU columns.

## Hyperparameters

`n_modules=6`, `n_active=3`, `n_layers=2`; `module_size` set by the parameter budget.
