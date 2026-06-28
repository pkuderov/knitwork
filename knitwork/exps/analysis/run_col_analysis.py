"""Column activation analysis for GridRNN variants.

Usage:
    uv run knitwork/exps/analysis/run_col_analysis.py \
        knitwork/exps/sdq/config/extend_config.yaml \
        --model=grnn --n_steps=500 --device=cpu --out=runs/col_analysis
"""
from __future__ import annotations

import importlib
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from knitwork.common.entrypoint import run_experiment
from knitwork.common.config import load_config
from knitwork.common.utils import get_device
from knitwork.visualization.column_analysis import (
    ColumnProbe,
    col_cka_matrix,
    col_token_correlation,
    mean_activation_norm,
    plot_activation_norms,
    plot_cka_matrix,
    plot_token_correlation,
    top_activating_contexts,
)

_REGISTRY: dict[str, tuple[str, str]] = {
    'grnn':     ('knitwork.models.grnn',     'GridRnn'),
    'grnn_lru': ('knitwork.models.grnn_lru', 'GridLRU'),
    'hgrnn':    ('knitwork.models.hgrnn',    'HierarchicalGridRNN'),
}


def _build_model(cfg: dict):
    name = cfg.get('model', 'grnn')
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Supported: {list(_REGISTRY)}")
    module_path, cls_name = _REGISTRY[name]
    cls = getattr(importlib.import_module(module_path), cls_name)
    model_cfg = cfg['models'][name]
    return cls(**model_cfg)


def _run_sdq_dataset(model, cfg: dict, device, n_steps: int):
    """Run SDQ generator through model and collect (acts, tokens) pairs."""
    from knitwork.gens.sdq import StoreDistractQueryGenerator

    gen_cfg = cfg.get('generator', {})
    gen = StoreDistractQueryGenerator(
        n_envs=cfg.get('n_envs', 16),
        T=gen_cfg.get('T', 10),
        p_store=gen_cfg.get('p_store', 0.2),
        p_query=gen_cfg.get('p_query', 0.5),
        n_symbols=gen_cfg.get('n_symbols', 20),
        device=device,
    )

    model.eval()
    h = None
    all_acts, all_tokens = [], []

    with torch.no_grad(), ColumnProbe(model) as probe:
        for step in range(n_steps):
            tokens, _ = gen.step()
            tokens = tokens.unsqueeze(-1)  # [batch, 1]
            y, h = model(tokens, h)
            h = model.detach_state(h)

            if (step + 1) % 50 == 0:
                acts_batch = probe.get_tensor()  # [T_so_far, layers, cols, batch, H]
                # store the most recent 50 steps worth
                tok_tensor = torch.stack([
                    gen._last_tokens if hasattr(gen, '_last_tokens') else tokens.squeeze(-1)
                ], dim=0)
                probe.clear()

    return probe.get_tensor()


def runner(config: dict):
    out_dir = Path(config.get('out', 'runs/col_analysis'))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(config.get('device', 'cpu'))
    n_steps = int(config.get('n_steps', 200))

    model = _build_model(config).to(device)
    vocab_size = config['models'][config.get('model', 'grnn')].get('input_size', 27)

    print(f"Running {n_steps} steps on {device}...")
    from knitwork.gens.sdq import StoreDistractQueryGenerator

    gen_cfg = config.get('generator', {})
    n_envs = config.get('n_envs', 16)
    gen = StoreDistractQueryGenerator(
        n_envs=n_envs,
        T=gen_cfg.get('T', 10),
        p_store=gen_cfg.get('p_store', 0.2),
        p_query=gen_cfg.get('p_query', 0.5),
        n_symbols=gen_cfg.get('n_symbols', 20),
        device=device,
    )

    model.eval()
    h = None
    token_log: list[torch.Tensor] = []

    with torch.no_grad(), ColumnProbe(model) as probe:
        for _ in range(n_steps):
            tokens, _ = gen.step()
            token_log.append(tokens.cpu())     # [batch]
            tokens_in = tokens.unsqueeze(-1)   # [batch, 1]
            _, h = model(tokens_in, h)
            h = model.detach_state(h)

    acts = probe.get_tensor()          # [T, layers, cols, batch, H]
    tokens_t = torch.stack(token_log)  # [T, batch]
    print(f"Collected activations: {tuple(acts.shape)}")

    # Analysis
    norms = mean_activation_norm(acts)
    print("Column activation norms:", norms.tolist())

    cka = col_cka_matrix(acts, layer=-1)
    corr = col_token_correlation(acts, tokens_t, vocab_size=vocab_size, layer=-1)
    top_ctx = top_activating_contexts(acts, tokens_t, k=5, n=2, layer=-1)

    # Save figures
    fig = plot_activation_norms(norms)
    fig.savefig(out_dir / "activation_norms.png", dpi=120)
    plt.close(fig)

    fig = plot_cka_matrix(cka, layer=acts.shape[1] - 1)
    fig.savefig(out_dir / "cka_matrix.png", dpi=120)
    plt.close(fig)

    fig = plot_token_correlation(corr)
    fig.savefig(out_dir / "token_correlation.png", dpi=120)
    plt.close(fig)

    # Save top contexts as text
    with open(out_dir / "top_contexts.txt", "w") as f:
        for c, ctxs in top_ctx.items():
            f.write(f"\n=== Column {c} ===\n")
            for ngram, score in ctxs:
                f.write(f"  {ngram}  score={score:.4f}\n")

    # Save raw data
    torch.save({"acts": acts, "tokens": tokens_t, "norms": norms}, out_dir / "data.pt")
    np.save(out_dir / "cka.npy", cka)

    print(f"Results saved to {out_dir}/")
    return 0


import matplotlib.pyplot as plt  # noqa: E402 — after matplotlib.use in column_analysis

if __name__ == "__main__":
    run_experiment(runner=runner)
