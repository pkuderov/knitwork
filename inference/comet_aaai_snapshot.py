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
PROJECT_LABELS = {
    "knitwork-text": "text8",
    "knitwork-sdq": "SDQ",
}
TEXT_STEP_LIMIT = 1e9
REDUCED_BUDGET_BASELINES = {
    "delta_net_10.10M",
    "hgrn2_10.13M",
    "mlstm_10.11M",
}
MODEL_CFG_ALIASES = {
    ("rnn_2L", None): ("rnn", "rnn_L2"),
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


def format_int(value):
    if value is None:
        return "—"
    return f"{int(float(value)):,}"


def metric(summary, name, field):
    item = summary.get(name)
    if item is None:
        return None
    return number(item.get(field))


def parameter_map(experiment):
    return {
        item["name"]: item.get("valueCurrent")
        for item in experiment.get_parameters_summary()
    }


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
            "project": project,
            "state": experiment.get_state(),
            "summary": summary_map(experiment),
            "parameters": parameter_map(experiment),
        })
        runs[-1]["model"] = runs[-1]["parameters"].get("model")
        runs[-1]["model_cfg"] = runs[-1]["parameters"].get("model_cfg")
        runs[-1]["group_model"], runs[-1]["group_model_cfg"] = MODEL_CFG_ALIASES.get(
            (runs[-1]["model"], runs[-1]["model_cfg"]),
            (runs[-1]["model"], runs[-1]["model_cfg"]),
        )
    return sorted(runs, key=lambda run: (run["name"], run["id"]))


def group_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["group_model"], run["group_model_cfg"])].append(run)
    return [
        {
            "name": model_label(model, model_cfg),
            "runs": sorted(group, key=lambda run: run["id"]),
        }
        for (model, model_cfg), group in sorted(grouped.items())
    ]


def model_label(model, model_cfg):
    return f"{model or '—'} / {model_cfg or '—'}"


def main_text_groups(groups):
    result = []
    for group in groups:
        runs = [
            run for run in group["runs"]
            if run["state"] == "finished" and run["group_model"] in {"rnn", "grnn"}
        ]
        if runs:
            result.append({"name": group["name"], "runs": runs})
    return result


def fixed_horizon_result(group):
    return aggregate_curve(
        group,
        "val/BPC",
        step_limit=TEXT_STEP_LIMIT,
    )


def best_checkpoint_result(group):
    values = [
        metric(run["summary"], "val/BPC", "valueMin")
        for run in group["runs"]
    ]
    mean, std = mean_std(values)
    return {"mean": mean, "std": std}


def reduced_text_groups(groups):
    result = []
    for group in groups:
        runs = [
            run for run in group["runs"]
            if run["state"] == "finished" and run["name"] in REDUCED_BUDGET_BASELINES
        ]
        if runs:
            result.append({"name": group["name"], "runs": runs})
    return result


def planned_updates(run):
    parameters = run["parameters"]
    n_envs = number(parameters.get("n_envs"))
    rollout_len = number(parameters.get("rollout_len"))
    n_steps = number(parameters.get("n_steps"))
    if None in (n_envs, rollout_len, n_steps):
        return None
    return n_steps / (n_envs * rollout_len)


def format_updates(value):
    if value is None:
        return "—"
    return f"{value / 1e3:.1f}k"


def reduced_budget_result(group, field):
    values = [
        metric(run["summary"], "val/BPC", field)
        for run in group["runs"]
    ]
    return mean_std(values)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0


def format_mean_std(mean, std):
    if mean is None:
        return "—"
    return f"{mean:.4f} ± {std:.4f}"


def status_metric(run):
    if run["project"] == "knitwork-text":
        metric_name = "val/BPC"
        best = metric(run["summary"], metric_name, "valueMin")
        current = metric(run["summary"], metric_name, "valueCurrent")
        return metric_name, best, current
    metric_name = "Acc++"
    best = metric(run["summary"], metric_name, "valueMax")
    current = metric(run["summary"], metric_name, "valueCurrent")
    return metric_name, best, current


def budget_label(run):
    parameters = run["parameters"]
    return "{envs} envs × {steps}".format(
        envs=format_int(parameters.get("n_envs")),
        steps=format_step(number(parameters.get("n_steps"))),
    )


def comparability_label(run):
    parameters = run["parameters"]
    n_envs = number(parameters.get("n_envs"))
    n_steps = number(parameters.get("n_steps"))
    if run["name"] in REDUCED_BUDGET_BASELINES:
        budget = "reduced budget"
    elif n_envs == 512 and n_steps >= TEXT_STEP_LIMIT:
        budget = "standard budget"
    else:
        budget = "nonstandard budget"
    original = (run["model"], run["model_cfg"])
    grouped = (run["group_model"], run["group_model_cfg"])
    if original != grouped:
        config = f"legacy alias to {model_label(*grouped)} (user-confirmed)"
    else:
        config = "same model/model_cfg (Comet)"
    return f"{config}; {budget}"


def anomaly_label(run):
    metric_name, best, current = status_metric(run)
    notes = []
    if run["state"] != "finished":
        notes.append(f"state: {run['state']}")
    if best is None or current is None:
        notes.append(f"missing {metric_name}")
    elif metric_name == "val/BPC" and current - best > 0.01:
        notes.append(f"current BPC is {current - best:.3f} above best")
    elif metric_name == "Acc++" and best - current > 0.05:
        notes.append(f"current Acc++ is {best - current:.3f} below best")
    return "; ".join(notes) if notes else "—"


def status_row(run, seed_index):
    metric_name, best, current = status_metric(run)
    final_step = metric(run["summary"], "global_step", "valueCurrent")
    target_step = number(run["parameters"].get("n_steps"))
    return "| {experiment} | {model} | {seed} | {state} | {progress} | {metrics} | {budget} | {comparability} | {anomaly} |".format(
        experiment=PROJECT_LABELS[run["project"]],
        model=f"{run['name']} ({model_label(run['model'], run['model_cfg'])})",
        seed=f"replicate {seed_index} (`{run['id'][:8]}`)",
        state=run["state"],
        progress=f"{format_step(final_step)} / {format_step(target_step)}",
        metrics="{name}: best {best}; current {current}".format(
            name=metric_name,
            best=format_number(best),
            current=format_number(current),
        ),
        budget=budget_label(run),
        comparability=comparability_label(run),
        anomaly=anomaly_label(run),
    )


def write_report(path, text_groups, sdq_groups):
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# AAAI Comet Snapshot",
        "",
        f"Retrieved read-only from Comet workspace `{WORKSPACE}` at {retrieved_at}.",
        "This is an exploratory tracker snapshot, not a selected paper result set.",
        "Runs are grouped by the Comet `model` and `model_cfg` parameters, with the user-confirmed legacy `rnn_2L` alias merged into `rnn / rnn_L2`.",
        "",
        "## Per-seed status",
        "",
        "`same model/model_cfg` is verified from Comet. `replicate N` is an analysis label: intentional null seeds mean the Comet ID is the stable run identifier.",
        "",
        "| Experiment | Model config | Seed | State | Progress | Metrics | Logged budget | Configuration comparability | Obvious anomaly |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *(
            status_row(run, seed_index)
            for groups in (text_groups, sdq_groups)
            for group in groups
            for seed_index, run in enumerate(group["runs"], start=1)
        ),
        "",
        "## Text8: completed-seed results at the comparable horizon",
        "",
        "RNN and GRNN/MoSAIC only. Each run is truncated at 1B tokens; runs with a final logged validation point slightly below 1B are retained under the logging-loss convention. Unfinished runs and reduced-budget baselines are excluded.",
        "",
        "| Model config | Completed replicates | Common logged horizon | Val BPC ↓ | Comet IDs |",
        "| --- | ---: | ---: | ---: | --- |",
        *(
            "| {name} | {seeds} | {horizon} | {bpc} | {ids} |".format(
                name=group["name"],
                seeds=len(group["runs"]),
                horizon=format_step(fixed_horizon_result(group)["x"][-1]),
                bpc=format_mean_std(
                    fixed_horizon_result(group)["mean"][-1],
                    fixed_horizon_result(group)["std"][-1],
                ),
                ids=", ".join(f"`{run['id'][:8]}`" for run in group["runs"]),
            )
            for group in main_text_groups(text_groups)
        ),
        "",
        "## Text8: completed-seed best validation checkpoints",
        "",
        "This is a separate checkpoint-selection view, not a fixed-horizon comparison. It includes only completed RNN and GRNN/MoSAIC runs; best checkpoints may occur before or after 1B tokens.",
        "",
        "| Model config | Completed replicates | Best val BPC ↓ | Comet IDs |",
        "| --- | ---: | ---: | --- |",
        *(
            "| {name} | {seeds} | {bpc} | {ids} |".format(
                name=group["name"],
                seeds=len(group["runs"]),
                bpc=format_mean_std(
                    best_checkpoint_result(group)["mean"],
                    best_checkpoint_result(group)["std"],
                ),
                ids=", ".join(f"`{run['id'][:8]}`" for run in group["runs"]),
            )
            for group in main_text_groups(text_groups)
        ),
        "",
        "## Text8: reduced-token, increased-update baselines",
        "",
        "This separate table is not part of the 1B-token RNN/GRNN comparison. Updates equal `n_steps / (n_envs × rollout_len)`; the rollout length is 64 for these runs. The standard 1B-token RNN/GRNN protocol has 30.5k planned updates. Final and best BPC are both shown because this is a completed-run status view, not a fixed-horizon comparison.",
        "",
        "| Model config | Tokens | Batch tokens/update | Planned updates | Final val BPC ↓ | Best val BPC ↓ | Comet IDs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        *(
            "| {name} | {tokens} | {batch} | {updates} | {final_bpc} | {best_bpc} | {ids} |".format(
                name=group["name"],
                tokens=format_step(number(group["runs"][0]["parameters"].get("n_steps"))),
                batch=format_int(
                    number(group["runs"][0]["parameters"].get("n_envs"))
                    * number(group["runs"][0]["parameters"].get("rollout_len")),
                ),
                updates=format_updates(planned_updates(group["runs"][0])),
                final_bpc=format_mean_std(*reduced_budget_result(group, "valueCurrent")),
                best_bpc=format_mean_std(*reduced_budget_result(group, "valueMin")),
                ids=", ".join(f"`{run['id'][:8]}`" for run in group["runs"]),
            )
            for group in reduced_text_groups(text_groups)
        ),
        "",
        "## Store--Distract--Query (`knitwork-sdq`)",
        "",
        "`Acc++` is the logged online-generator evaluation metric. Curves aggregate runs with the same `model` and `model_cfg`; the table uses the peak of the group mean curve.",
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
        "- SDQ completion ranges from 140M to 1B steps. Its online generator supplies the reported evaluation metrics, so no separate validation split is expected.",
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
