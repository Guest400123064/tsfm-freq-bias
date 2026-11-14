import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

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
PRIMARY_THRESHOLD = 0.5
MARKERS = ("o", "s", "^")


def mean(values):
    return sum(values) / len(values)


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2
        for index in order[i:j]:
            result[index] = rank
        i = j
    return result


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    covariance = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return covariance / math.sqrt(vx * vy)


def threshold_times(records, threshold, final_step):
    times = []
    for seed in sorted({row["seed"] for row in records}):
        seed_rows = sorted(
            (row for row in records if row["seed"] == seed),
            key=lambda row: row["step"],
        )
        first = seed_rows[0]
        for index, frequency in enumerate(first["frequencies"]):
            hit = next(
                (row["step"] for row in seed_rows if row["dk"][index] <= threshold),
                None,
            )
            times.append(
                {
                    "seed": seed,
                    "power": first["powers"][index],
                    "frequency": frequency,
                    "step": hit if hit is not None else final_step,
                    "reached": hit is not None,
                }
            )
    return times


def fit_loglog(times):
    reached = [point for point in times if point["reached"]]
    by_power = {}
    for point in reached:
        by_power.setdefault(point["power"], []).append(point["step"])
    points = sorted((power, mean(steps)) for power, steps in by_power.items())
    if len(points) < 2:
        return None

    xs = [math.log(power) for power, _ in points]
    ys = [math.log(step) for _, step in points]
    mx, my = mean(xs), mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum(
        (x - mx) ** 2 for x in xs
    )
    intercept = my - slope * mx
    predicted = [intercept + slope * x for x in xs]
    residual = sum((y - yhat) ** 2 for y, yhat in zip(ys, predicted, strict=True))
    total = sum((y - my) ** 2 for y in ys)
    return {
        "levels": len(points),
        "gamma": -slope,
        "intercept": intercept,
        "r2": 1 - residual / total,
        "min_power": points[0][0],
        "max_power": points[-1][0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    records = data["records"]
    steps = sorted({row["step"] for row in records})
    final_step = max(steps)
    seeds = sorted({row["seed"] for row in records})
    marker_of = dict(zip(seeds, MARKERS, strict=True))
    unique_powers = sorted({power for row in records for power in row["powers"]})
    norm = LogNorm(vmin=min(unique_powers), vmax=max(unique_powers))
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.55), constrained_layout=True)

    for level, power in enumerate(unique_powers):
        curves = []
        for step in steps:
            values = [
                row["dk"][index]
                for row in records
                if row["step"] == step
                for index, assigned in enumerate(row["levels"])
                if assigned == level
            ]
            curves.append(mean(values))
        axes[0].plot(
            steps,
            curves,
            color=cmap(norm(power)),
            lw=1.6,
            label=f"{power / min(unique_powers):.0f}x",
        )

    times = threshold_times(records, PRIMARY_THRESHOLD, final_step)
    for point in times:
        axes[1].scatter(
            point["power"],
            point["step"],
            color=cmap(norm(point["power"])),
            marker=marker_of[point["seed"]] if point["reached"] else "x",
            s=13,
            alpha=0.75,
        )
    fit = fit_loglog(times)
    if fit:
        grid = [
            fit["min_power"] * (fit["max_power"] / fit["min_power"]) ** (i / 40)
            for i in range(41)
        ]
        axes[1].plot(
            grid,
            [math.exp(fit["intercept"]) * p ** (-fit["gamma"]) for p in grid],
            color="black",
            lw=1.2,
            label=f"fit: γ={fit['gamma']:.2f}",
        )
        axes[1].legend(frameon=False, fontsize=6, loc="upper right")

    final_rows = [row for row in records if row["step"] == final_step]
    for row in final_rows:
        for frequency, power, dk in zip(
            row["frequencies"], row["powers"], row["dk"], strict=True
        ):
            axes[2].scatter(
                frequency,
                dk,
                color=cmap(norm(power)),
                marker=marker_of[row["seed"]],
                s=13,
                alpha=0.75,
            )

    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Training step",
        ylabel="Band Δk",
        title="(a) Learning curves by power level",
    )
    axes[0].legend(frameon=False, fontsize=6, ncol=2, title="Relative power")
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Assigned band power",
        ylabel=f"Steps to Δk ≤ {PRIMARY_THRESHOLD:g}",
        title=f"(b) Time to Δk ≤ {PRIMARY_THRESHOLD:g}",
    )
    axes[1].text(
        0.98,
        0.03,
        "× = censored",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="0.35",
    )
    axes[2].set(
        yscale="log",
        xlabel="Band b (cycles/patch)",
        ylabel=f"Δk at step {final_step}",
        title="(c) Final error vs frequency",
    )
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        location="bottom",
        shrink=0.45,
        label="Band power (log scale)",
    )
    fig.savefig(args.output, dpi=300)

    observations = [
        (power, frequency, dk)
        for row in final_rows
        for frequency, power, dk in zip(
            row["frequencies"], row["powers"], row["dk"], strict=True
        )
    ]
    power_rank = spearman(
        [math.log(power) for power, _, _ in observations],
        [dk for _, _, dk in observations],
    )
    frequency_rank = spearman(
        [frequency for _, frequency, _ in observations],
        [dk for _, _, dk in observations],
    )
    print(f"final dk: spearman(log power, dk)={power_rank:.3f}")
    print(f"final dk: spearman(frequency, dk)={frequency_rank:.3f}")
    for threshold in THRESHOLDS:
        points = threshold_times(records, threshold, final_step)
        fit = fit_loglog(points)
        reached = sum(point["reached"] for point in points)
        if fit:
            print(
                f"target {threshold:g}: reached={reached}/{len(points)}, "
                f"levels={fit['levels']}, gamma={fit['gamma']:.3f}, "
                f"r2={fit['r2']:.3f}"
            )
        else:
            print(f"target {threshold:g}: reached={reached}/{len(points)}, no fit")
    print(
        f"variance explained: min={min(row['var_explained'] for row in final_rows):.3f}"
    )


if __name__ == "__main__":
    main()
