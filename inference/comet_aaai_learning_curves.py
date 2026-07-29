"""Generate the paper learning-curve figure from completed Comet replicates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from comet_ml.api import API

from comet_aaai_snapshot import (
    PROJECTS,
    TEXT_STEP_LIMIT,
    collect_project,
    curve,
    group_runs,
    is_completed_protocol_run,
    is_standard_protocol,
)


OUTPUT = Path("article/latex/fig_learning_curves.pdf")
COLORS = {
    "GRU": "#4C78A8",
    "MoSAIC": "#F58518",
    "Transformer": "#54A24B",
}
TEXT_SERIES = (
    ("rnn", "rnn_L2", "GRU-L2", "GRU", "--"),
    ("rnn", "rnn_L3", "GRU-L3", "GRU", "-"),
    ("grnn", "grnn_L2C4", "MoSAIC-L2C4", "MoSAIC", "--"),
    ("grnn", "grnn_L3C4", "MoSAIC-L3C4", "MoSAIC", "-"),
    ("transformer", "transformer", "Transformer-256", "Transformer", "-."),
)
SDQ_SERIES = (
    ("rnn", "rnn_L1", "GRU-L1", "GRU", ":"),
    ("rnn", "rnn_L2", "GRU-L2", "GRU", "--"),
    ("grnn", "grnn_L2C4", "MoSAIC-L2C4", "MoSAIC", "--"),
    ("grnn", "grnn_L3C4", "MoSAIC-L3C4", "MoSAIC", "-"),
)


def selected_runs(groups, model, model_cfg):
    for group in groups:
        runs = [
            run for run in group["runs"]
            if run["group_model"] == model
            and run["group_model_cfg"] == model_cfg
            and is_completed_protocol_run(run)
            and is_standard_protocol(run)
        ]
        if runs:
            return runs
    raise ValueError(f"No completed standard-protocol runs for {model}/{model_cfg}")


def common_grid_summary(runs, metric_name):
    traces = []
    for run in runs:
        points = {
            int(round(step)): value
            for step, value in curve(run, metric_name)
            if step <= TEXT_STEP_LIMIT
        }
        if not points:
            raise ValueError(f"No {metric_name} curve for {run['id']}")
        traces.append(points)

    common_steps = sorted(set.intersection(*(set(trace) for trace in traces)))
    if not common_steps:
        raise ValueError(f"No shared logging grid for {metric_name}")
    values = np.asarray([
        [trace[step] for step in common_steps]
        for trace in traces
    ])
    return {
        "x": np.asarray(common_steps) / 1e9,
        "mean": values.mean(axis=0),
        "std": values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(len(common_steps)),
    }


def plot_series(axis, groups, spec, metric_name):
    model, model_cfg, label, family, linestyle = spec
    runs = selected_runs(groups, model, model_cfg)
    result = common_grid_summary(runs, metric_name)
    color = COLORS[family]
    handle, = axis.plot(
        result["x"],
        result["mean"],
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=f"{label} (n={len(runs)})",
    )
    axis.fill_between(
        result["x"],
        result["mean"] - result["std"],
        result["mean"] + result["std"],
        color=color,
        alpha=0.15,
        linewidth=0,
    )
    return handle


def configure_axis(axis, title, ylabel, ylimits, yticks):
    axis.set(
        title=title,
        xlabel="Processed tokens (billions)",
        ylabel=ylabel,
        xlim=(0.0, 1.0),
        ylim=ylimits,
        xticks=np.arange(0.0, 1.01, 0.2),
        yticks=yticks,
    )
    axis.grid(color="#B8B8B8", alpha=0.35, linewidth=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.title.set_fontsize(9)
    axis.xaxis.label.set_size(8)
    axis.yaxis.label.set_size(8)


def main():
    api = API()
    text_groups = group_runs(collect_project(api, PROJECTS[0]))
    sdq_groups = group_runs(collect_project(api, PROJECTS[1]))

    plt.rcParams.update({
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.95))
    text_handles = [
        plot_series(axes[0], text_groups, spec, "val/BPC")
        for spec in TEXT_SERIES
    ]
    sdq_handles = [
        plot_series(axes[1], sdq_groups, spec, "Acc++")
        for spec in SDQ_SERIES
    ]
    configure_axis(
        axes[0],
        "(a) text8 character modeling",
        "Validation BPC ↓",
        (1.40, 2.65),
        np.arange(1.4, 2.61, 0.2),
    )
    configure_axis(
        axes[1],
        "(b) Store–Distract–Query",
        "Long-gap query accuracy (Acc++ ↑)",
        (0.0, 1.0),
        np.arange(0.0, 1.01, 0.2),
    )
    figure.suptitle(
        "Learning dynamics under the standard 1B-token protocol.",
        fontsize=9,
        y=0.99,
    )
    handles = text_handles + sdq_handles
    figure.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        ncol=5,
        fontsize=7.2,
        frameon=False,
        columnspacing=1.25,
        handlelength=2.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.992,
        top=0.83,
        bottom=0.28,
        wspace=0.28,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, format="pdf")
    plt.close(figure)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
