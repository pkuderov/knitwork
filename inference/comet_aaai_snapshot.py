"""Create a read-only snapshot of the AAAI Comet projects.

The report is intentionally exploratory: it preserves each run's tracker
summary and plots its logged training curve without choosing a paper cohort.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from comet_ml.api import API


WORKSPACE = "team-rl-exp"
PROJECTS = ("knitwork-text", "knitwork-sdq")
TEXT_STEP_LIMIT = 1e9
REDUCED_BUDGET_BASELINES = {
    "delta_net_10.10M",
    "hgrn2_10.13M",
    "mlstm_10.11M",
}


def summary_map(experiment):
    return {
        item["name"]: item
        for item in experiment.get_metrics_summary()
    }


def number(value):
    if value is None:
        return None
    return float(value)


def format_number(value, digits=4):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def format_step(value):
    if value is None:
        return "—"
    return f"{value / 1e6:.1f}M"


def metric(summary, name, field):
    item = summary.get(name)
    if item is None:
        return None
    return number(item.get(field))


def collect_project(api, project):
    runs = []
    for experiment in api.get_experiments(
        WORKSPACE,
        project,
        page_size=1000,
    ):
        runs.append({
            "experiment": experiment,
            "id": experiment.id,
            "name": experiment.get_name().strip(),
            "summary": summary_map(experiment),
        })
    return sorted(runs, key=lambda run: (run["name"], run["id"]))


def group_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["name"]].append(run)
    return [
        {
            "name": name,
            "runs": sorted(group, key=lambda run: run["id"]),
        }
        for name, group in sorted(grouped.items())
    ]


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0


def format_mean_std(mean, std):
    if mean is None:
        return "—"
    return f"{mean:.4f} ± {std:.4f}"


def write_report(path, text_groups, sdq_groups):
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# AAAI Comet Snapshot",
        "",
        f"Retrieved read-only from Comet workspace `{WORKSPACE}` at {retrieved_at}.",
        "This is an exploratory tracker snapshot, not a selected paper result set.",
        "Runs with the same displayed name are treated as seeds of a frozen configuration.",
        "",
        "## text8 (`knitwork-text`)",
        "",
        "Text8 curves are truncated at 1B training steps. Means and standard deviations are computed over seeds at each shared curve position; a one-seed group has zero plotted uncertainty.",
        "",
        "| Model config | Seeds | Shared horizon | Final mean val BPC ↓ | Seed IDs |",
        "| --- | ---: | ---: | ---: | --- |",
        *(
            "| {name} | {seeds} | {horizon} | {bpc} | {ids} |".format(
                name=group["name"],
                seeds=len(group["runs"]),
                horizon=format_step(group["curve"]["x"][-1]),
                bpc=format_mean_std(
                    group["curve"]["mean"][-1],
                    group["curve"]["std"][-1],
                ),
                ids=", ".join(f"`{run['id'][:8]}`" for run in group["runs"]),
            )
            for group in text_groups
        ),
        "",
        "## Store--Distract--Query (`knitwork-sdq`)",
        "",
        "`Acc++` is reported exactly as logged by the tracker. Curves aggregate seeds with the same model name; the table uses the peak of the group mean curve.",
        "",
        "| Model config | Seeds | Shared horizon | Peak mean Acc++ ↑ | Seed IDs |",
        "| --- | ---: | ---: | ---: | --- |",
        *(
            "| {name} | {seeds} | {horizon} | {acc} | {ids} |".format(
                name=group["name"],
                seeds=len(group["runs"]),
                horizon=format_step(group["curve"]["x"][-1]),
                acc=format_mean_std(
                    group["curve"]["mean"].max(),
                    group["curve"]["std"][group["curve"]["mean"].argmax()],
                ),
                ids=", ".join(f"`{run['id'][:8]}`" for run in group["runs"]),
            )
            for group in sdq_groups
        ),
        "",
        "## Exploratory reading",
        "",
        "- The three non-RNN baselines (DeltaNet, HGRN2, and mLSTM) use reduced `n_envs` and `n_steps` because of their memory requirements. The update-indexed text8 panel is the appropriate relative-efficiency view for those baselines.",
        "- SDQ completion ranges from 140M to 1B steps. These runs expose training metrics but no separately named validation metrics in Comet.",
        "",
        "The companion figure is `figures/aaai_comet_snapshot.png`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def curve(run, metric_name):
    cache = run.setdefault("metric_cache", {})
    if metric_name in cache:
        return cache[metric_name]
    rows = run["experiment"].get_metrics(metric_name)
    cache[metric_name] = sorted([
        (number(row.get("step")), number(row.get("metricValue")))
        for row in rows
        if row.get("step") is not None and row.get("metricValue") is not None
    ])
    return cache[metric_name]


def aggregate_curve(group, metric_name, x_metric_name=None, step_limit=None):
    traces = []
    for run in group["runs"]:
        metric_values = curve(run, metric_name)
        if step_limit is not None:
            metric_values = [
                value for value in metric_values
                if value[0] <= step_limit
            ]
        if not metric_values:
            continue
        if x_metric_name is None:
            traces.append(metric_values)
            continue
        x_values = curve(run, x_metric_name)
        if not x_values:
            continue
        steps, updates = zip(*x_values)
        values = [
            (float(np.interp(step, steps, updates)), metric)
            for step, metric in metric_values
            if steps[0] <= step <= steps[-1]
        ]
        if values:
            traces.append(values)

    if not traces:
        raise ValueError(f"No {metric_name} curves for {group['name']}")
    horizon = min(trace[-1][0] for trace in traces)
    start = max(trace[0][0] for trace in traces)
    grid = np.linspace(start, horizon, 121)
    values = np.stack([
        np.interp(grid, *zip(*trace))
        for trace in traces
    ])
    return {
        "x": grid,
        "mean": values.mean(axis=0),
        "std": values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(len(grid)),
    }


def plot_group(
    axis,
    group,
    metric_name,
    x_metric_name=None,
    step_limit=None,
    color=None,
    x_divisor=1e9,
):
    result = aggregate_curve(group, metric_name, x_metric_name, step_limit)
    x_values = result["x"] / x_divisor
    label = f"{group['name']} (n={len(group['runs'])})"
    axis.plot(x_values, result["mean"], color=color, linewidth=1.8, label=label)
    if len(group["runs"]) > 1:
        axis.fill_between(
            x_values,
            result["mean"] - result["std"],
            result["mean"] + result["std"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    return result


def plot_curves(path, text_groups, sdq_groups):
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    colors = plt.get_cmap("tab20").colors

    for index, group in enumerate(text_groups):
        color = colors[index % len(colors)]
        group["curve"] = plot_group(
            axes[0],
            group,
            "val/BPC",
            step_limit=TEXT_STEP_LIMIT,
            color=color,
        )
        plot_group(
            axes[1],
            group,
            "val/BPC",
            x_metric_name="Upd",
            step_limit=TEXT_STEP_LIMIT,
            color=color,
            x_divisor=1e3,
        )

    for index, group in enumerate(sdq_groups):
        group["curve"] = plot_group(
            axes[2],
            group,
            "Acc++",
            color=colors[index % len(colors)],
        )

    axes[0].set(
        title="text8: validation BPC by training steps",
        xlabel="Training steps (billions; capped at 1B)",
        ylabel="Validation BPC (lower is better)",
    )
    axes[1].set(
        title="text8: validation BPC by updates",
        xlabel="Updates (thousands; runs capped at 1B steps)",
        ylabel="Validation BPC (lower is better)",
    )
    axes[2].set(
        title="SDQ: logged Acc++ by training steps",
        xlabel="Training steps (billions)",
        ylabel="Acc++ (higher is better)",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=5.8, frameon=False, loc="best")
    axes[1].text(
        0.02,
        0.03,
        "DeltaNet, HGRN2, and mLSTM use reduced n_envs / n_steps.",
        transform=axes[1].transAxes,
        fontsize=7,
    )

    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/experiments/results_aaai.md"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/experiments/figures/aaai_comet_snapshot.png"),
    )
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    api = API()
    text_groups = group_runs(collect_project(api, PROJECTS[0]))
    sdq_groups = group_runs(collect_project(api, PROJECTS[1]))
    plot_curves(args.figure, text_groups, sdq_groups)
    write_report(args.report, text_groups, sdq_groups)
    print(f"Wrote {args.report} and {args.figure}")


if __name__ == "__main__":
    main()
