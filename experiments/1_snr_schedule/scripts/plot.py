import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "figure.facecolor": "white",
    }
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
    arms = data["arms"]
    sigmas = sorted({arm["sigma"] for arm in arms})
    grouped = {
        sigma: [arm for arm in arms if arm["sigma"] == sigma] for sigma in sigmas
    }
    steps = sorted({record["step"] for arm in arms for record in arm["records"]})
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (len(sigmas) - 1)) for i in range(len(sigmas))]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.55), constrained_layout=True)
    for color, sigma in zip(colors, sigmas, strict=True):
        clean = [curve(grouped[sigma], "clean_dk", step) for step in steps]
        noisy = [curve(grouped[sigma], "noisy_dk", step) for step in steps]
        label = f"σ_rel={sigma / math.sqrt(0.5):g}"
        axes[0].plot(steps, clean, color=color, lw=1.4, label=label)
        axes[1].plot(steps, noisy, color=color, lw=1.4, label=label)

    threshold_colors = ("#0072B2", "#D55E00", "#009E73")
    positions = list(range(len(sigmas)))
    for threshold, color in zip(THRESHOLDS, threshold_colors, strict=True):
        xs, ys = [], []
        for position, sigma in zip(positions, sigmas, strict=True):
            times = times_to_threshold(grouped[sigma], threshold)
            if times:
                xs.append(position)
                ys.append(mean(times))
        axes[2].plot(
            xs,
            ys,
            color=color,
            marker="o",
            ms=4,
            lw=1.2,
            label=f"Δk≤{threshold:g}",
        )

    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Training step",
        ylabel="Clean-probe Δk",
        title="(a) Clean signal learning",
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Training step",
        ylabel="Noisy-context Δk",
        title="(b) Noisy-context prediction",
    )
    axes[0].legend(frameon=False, fontsize=6)
    axes[2].set(
        yscale="log",
        xticks=positions,
        xticklabels=[f"{sigma / math.sqrt(0.5):g}" for sigma in sigmas],
        xlabel="Noise / signal RMS",
        ylabel="Steps to threshold",
        title="(c) Time to clean thresholds",
    )
    axes[2].legend(frameon=False, fontsize=6)
    fig.savefig(output, dpi=300)

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
    levels = grouped_by_level(data["arms"])
    styles = {
        "red": ("#0072B2", "o", "red: loud low freq"),
        "anti_red": ("#D55E00", "s", "anti-red: loud high freq"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.55), constrained_layout=True)
    for arm_name, (color, marker, label) in styles.items():
        xs = list(range(8))
        dk = [mean(value for value, _ in levels[level][arm_name]) for level in xs]
        axes[0].plot(xs, dk, color=color, marker=marker, ms=4, lw=1.3, label=label)
        for level in xs:
            axes[0].scatter(
                [level] * len(levels[level][arm_name]),
                [value for value, _ in levels[level][arm_name]],
                color=color,
                marker=marker,
                s=9,
                alpha=0.35,
            )

        time_x, time_y = [], []
        for level in xs:
            reached = [hit for _, hit in levels[level][arm_name] if hit is not None]
            if reached:
                time_x.append(level)
                time_y.append(mean(reached))
        axes[1].plot(
            time_x,
            time_y,
            color=color,
            marker=marker,
            ms=4,
            lw=1.3,
            label=label,
        )

    axes[0].set(
        yscale="log",
        xlabel="Power level (weak → loud)",
        ylabel="Final clean Δk",
        title="(a) Final error by assigned power",
    )
    axes[0].legend(frameon=False, fontsize=6)
    axes[1].set(
        yscale="log",
        xlabel="Power level (weak → loud)",
        ylabel="Steps to clean Δk ≤ 0.5",
        title="(b) Time to threshold by assigned power",
    )
    fig.savefig(output, dpi=300)

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
