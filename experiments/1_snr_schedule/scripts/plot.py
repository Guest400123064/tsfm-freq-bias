import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from bias.plotting import (
    BLUE,
    OCHRE,
    PALETTE,
    VERMILION,
    WIDE,
    format_log_axis,
    use_style,
)

THRESHOLDS = (0.5, 0.2, 0.1)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def final_rows(arms):
    final = max(record["step"] for arm in arms for record in arm["records"])
    return final, [
        next(r for r in arm["records"] if r["step"] == final) for arm in arms
    ]


def curve(arms, key, step):
    values = [
        mean(record[key])
        for arm in arms
        for record in arm["records"]
        if record["step"] == step
    ]
    return mean(values)


def times_to_threshold(arms, threshold):
    times = []
    for arm in arms:
        for index in range(len(arm["frequencies"])):
            hit = next(
                (
                    record["step"]
                    for record in arm["records"]
                    if record["clean_dk"][index] <= threshold
                ),
                None,
            )
            if hit is not None:
                times.append(hit)
    return times


def plot_snr(data, output):
    use_style()
    arms = data["arms"]
    sigmas = sorted({arm["sigma"] for arm in arms})
    grouped = {
        sigma: [arm for arm in arms if arm["sigma"] == sigma] for sigma in sigmas
    }
    steps = sorted({record["step"] for arm in arms for record in arm["records"]})
    colors = PALETTE

    fig, axes = plt.subplots(1, 3, figsize=WIDE, constrained_layout=True)
    for color, sigma in zip(colors, sigmas, strict=True):
        clean = [curve(grouped[sigma], "clean_dk", step) for step in steps]
        noisy = [curve(grouped[sigma], "noisy_dk", step) for step in steps]
        label = rf"$\sigma_\mathrm{{rel}}={sigma / math.sqrt(0.5):g}$"
        sns.lineplot(
            x=steps,
            y=clean,
            ax=axes[0],
            color=color,
            lw=1.1,
            label=label,
            legend=False,
        )
        sns.lineplot(
            x=steps,
            y=noisy,
            ax=axes[1],
            color=color,
            lw=1.1,
            legend=False,
        )

    threshold_colors = (BLUE, VERMILION, OCHRE)
    positions = list(range(len(sigmas)))
    for threshold, color in zip(THRESHOLDS, threshold_colors, strict=True):
        xs, ys = [], []
        for position, sigma in zip(positions, sigmas, strict=True):
            times = times_to_threshold(grouped[sigma], threshold)
            if times:
                xs.append(position)
                ys.append(mean(times))
        sns.lineplot(
            x=xs,
            y=ys,
            ax=axes[2],
            color=color,
            marker="o",
            markersize=3.5,
            lw=1.0,
            label=f"Δk≤{threshold:g}",
            legend=False,
        )

    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Training step",
        ylabel="Clean-probe Δk",
        title="(a) Clean probe",
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Training step",
        ylabel="Noisy-context Δk",
        title="(b) Noisy context",
    )
    axes[2].set(
        yscale="log",
        xticks=positions,
        xticklabels=[f"{sigma / math.sqrt(0.5):g}" for sigma in sigmas],
        xlabel="Noise / signal RMS",
        ylabel="Steps to threshold",
        title="(c) Time to clean thresholds",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    threshold_handles, threshold_labels = axes[2].get_legend_handles_labels()
    fig.legend(
        handles + threshold_handles,
        labels + threshold_labels,
        loc="outside lower center",
        ncol=7,
    )
    for ax in axes:
        format_log_axis(ax)
    fig.align_labels(axes)
    fig.savefig(output)

    final, rows = final_rows(arms)
    print(f"final step {final}")
    for sigma in sigmas:
        selected = [
            row for arm, row in zip(arms, rows, strict=True) if arm["sigma"] == sigma
        ]
        print(
            f"sigma_rel={sigma / math.sqrt(0.5):g} "
            f"clean_r={mean(mean(row['clean_r']) for row in selected):.3f} "
            f"clean_dk={mean(mean(row['clean_dk']) for row in selected):.3f} "
            f"noisy_dk={mean(mean(row['noisy_dk']) for row in selected):.3f}"
        )
        for threshold in THRESHOLDS:
            times = times_to_threshold(grouped[sigma], threshold)
            reached = len(times)
            average = mean(times) if times else None
            print(f"  threshold={threshold:g} reached={reached} mean_t={average}")


def grouped_by_level(arms):
    levels = {}
    for arm in arms:
        final = max(record["step"] for record in arm["records"])
        row = next(record for record in arm["records"] if record["step"] == final)
        order = sorted(range(len(arm["powers"])), key=arm["powers"].__getitem__)
        for level in range(8):
            indices = order[level * 4 : (level + 1) * 4]
            for index in indices:
                hit = next(
                    (
                        record["step"]
                        for record in arm["records"]
                        if record["clean_dk"][index] <= 0.5
                    ),
                    None,
                )
                levels.setdefault(level, {}).setdefault(arm["arm"], []).append(
                    (row["clean_dk"][index], hit)
                )
    return levels


def plot_assignment(data, output):
    use_style()
    levels = grouped_by_level(data["arms"])
    styles = {
        "red": (VERMILION, "o", "Red: loud low bands"),
        "anti_red": (BLUE, "s", "Anti-red: loud high bands"),
    }

    fig, axes = plt.subplots(1, 2, figsize=WIDE, constrained_layout=True)
    for arm_name, (color, marker, label) in styles.items():
        xs = list(range(8))
        dk = [mean(value for value, _ in levels[level][arm_name]) for level in xs]
        sns.lineplot(
            x=xs,
            y=dk,
            ax=axes[0],
            color=color,
            marker=marker,
            markersize=3.5,
            lw=1.0,
            label=label,
            legend=False,
        )
        for level in xs:
            sns.scatterplot(
                x=[level] * len(levels[level][arm_name]),
                y=[value for value, _ in levels[level][arm_name]],
                ax=axes[0],
                color=color,
                marker=marker,
                s=12,
                alpha=0.2,
                linewidth=0,
                legend=False,
            )

        time_x, time_y = [], []
        for level in xs:
            reached = [hit for _, hit in levels[level][arm_name] if hit is not None]
            if reached:
                time_x.append(level)
                time_y.append(mean(reached))
        sns.lineplot(
            x=time_x,
            y=time_y,
            ax=axes[1],
            color=color,
            marker=marker,
            markersize=3.5,
            lw=1.0,
            legend=False,
        )

    axes[0].set(
        yscale="log",
        xlabel="Power level (weak → loud)",
        ylabel="Final clean Δk",
        title="(a) Final error",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    axes[1].set(
        yscale="log",
        xlabel="Power level (weak → loud)",
        ylabel="Steps to clean Δk ≤ 0.5",
        title="(b) Time to threshold",
    )
    for ax in axes:
        format_log_axis(ax)
    format_log_axis(axes[0], subs=(1.0, 2.0, 5.0))
    fig.align_labels(axes)
    fig.savefig(output)

    for level in range(1, 8):
        red = mean(value for value, _ in levels[level]["red"])
        anti = mean(value for value, _ in levels[level]["anti_red"])
        diff = abs(red - anti)
        print(f"level={level} red_dk={red:.3f} anti_dk={anti:.3f} diff={diff:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("snr", "assignment"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if args.mode == "snr":
        plot_snr(data, args.output)
    else:
        plot_assignment(data, args.output)


if __name__ == "__main__":
    main()
