import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from bias.plotting import PALETTE, WIDE, format_log_axis, use_style

THRESHOLDS = (0.1, 0.05, 0.02)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def target_time(arm, threshold):
    return next(
        (
            record["step"]
            for record in arm["records"]
            if record["mse"] <= threshold * 0.5
        ),
        None,
    )


def main():
    use_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    arms = data["arms"]
    levels = sorted({arm["level"] for arm in arms})
    seeds = sorted({arm["seed"] for arm in arms})
    steps = sorted({record["step"] for arm in arms for record in arm["records"]})
    colors = PALETTE

    fig, axes = plt.subplots(1, 3, figsize=WIDE, constrained_layout=True)
    for color, level in zip(colors, levels, strict=True):
        curves = []
        for seed in seeds:
            arm = next(a for a in arms if a["level"] == level and a["seed"] == seed)
            curve = [record["mse"] / 0.5 for record in arm["records"]]
            curves.append(curve)
            sns.lineplot(
                x=steps,
                y=curve,
                ax=axes[0],
                color=color,
                lw=0.45,
                alpha=0.18,
                legend=False,
            )
        sns.lineplot(
            x=steps,
            y=[mean(curve[i] for curve in curves) for i in range(len(steps))],
            ax=axes[0],
            color=color,
            lw=1.15,
            label=f"m={level}",
            legend=False,
        )

    for color, level in zip(colors, levels, strict=True):
        final_nmse = []
        for seed in seeds:
            arm = next(a for a in arms if a["level"] == level and a["seed"] == seed)
            sns.scatterplot(
                x=[level],
                y=[arm["records"][-1]["mse"] / 0.5],
                ax=axes[2],
                color=color,
                s=16,
                alpha=0.5,
                linewidth=0,
                legend=False,
            )
            final_nmse.append(arm["records"][-1]["mse"] / 0.5)
        sns.scatterplot(
            x=[level],
            y=[mean(final_nmse)],
            ax=axes[2],
            color="black",
            marker="_",
            s=90,
            zorder=3,
            legend=False,
        )

        for threshold, offset, marker in zip(
            THRESHOLDS, (-0.16, 0.0, 0.16), ("o", "s", "^"), strict=True
        ):
            times = [target_time(a, threshold) for a in arms if a["level"] == level]
            times = [time for time in times if time is not None]
            if times:
                sns.scatterplot(
                    x=[level + offset] * len(times),
                    y=times,
                    ax=axes[1],
                    color=color,
                    marker=marker,
                    s=16,
                    alpha=0.5,
                    linewidth=0,
                    legend=False,
                )
                sns.scatterplot(
                    x=[level + offset],
                    y=[mean(times)],
                    ax=axes[1],
                    color="black",
                    marker="_",
                    s=60,
                    zorder=3,
                    legend=False,
                )

    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Training step",
        ylabel="Normalized MSE",
        title="(a) Held-out error",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].set(
        yscale="log",
        xlabel="Harmonic order m",
        ylabel="Steps to normalized MSE",
        title="(b) Time to thresholds",
    )
    for threshold, marker in zip(THRESHOLDS, ("o", "s", "^"), strict=True):
        axes[1].scatter(
            [], [], color="0.3", marker=marker, s=20, label=f"≤{threshold:g}"
        )
    threshold_handles, threshold_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles + threshold_handles,
        labels + threshold_labels,
        loc="outside lower center",
        ncol=7,
    )
    axes[2].set(
        yscale="log",
        xlabel="Harmonic order m",
        ylabel="Final normalized MSE",
        title="(c) Final error",
    )
    for ax in axes:
        format_log_axis(ax)
    format_log_axis(axes[2], subs=(1.0, 2.0, 5.0))
    fig.align_labels(axes)
    fig.savefig(args.output)

    for level in levels:
        selected = [a for a in arms if a["level"] == level]
        final = [a["records"][-1]["var_explained"] for a in selected]
        at4k = [
            next(r["var_explained"] for r in a["records"] if r["step"] == 4096)
            for a in selected
        ]
        print(
            f"m={level} var@4096={mean(at4k):.4f} final={mean(final):.4f} "
            f"range=({min(final):.4f},{max(final):.4f})"
        )
        for threshold in THRESHOLDS:
            times = [target_time(a, threshold) for a in selected]
            reached = [time for time in times if time is not None]
            print(
                f"  threshold={threshold:g} reached={len(reached)}/3 "
                f"mean_t={mean(reached) if reached else None} times={times}"
            )


if __name__ == "__main__":
    main()
