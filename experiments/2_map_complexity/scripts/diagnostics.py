import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from bias.data import make_phase_map
from bias.plotting import PALETTE, WIDE, use_style

K = 16
CARRIER = 1.0
BETA = 0.1
LEVELS = (0, 1, 2, 3)
SEED = 2027
N_SPEC = 16_384


def harmonic_phases():
    gen = torch.Generator().manual_seed(SEED)
    return 2 * math.pi * torch.rand((1, len(LEVELS) - 1), generator=gen)[0]


def phase_map_diagnostics(level, phases):
    if level == 0:
        return {"min_derivative": 1.0, "phase_rms": 0.0}
    theta = torch.linspace(0, 2 * math.pi, 2048)
    gamma = BETA * math.sqrt(2 / level)
    nonlinear = sum(
        gamma * torch.sin(k * theta + phases[k - 1]) for k in range(1, level + 1)
    )
    derivative = 1 + sum(
        k * gamma * torch.cos(k * theta + phases[k - 1]) for k in range(1, level + 1)
    )
    return {
        "min_derivative": float(derivative.min()),
        "phase_rms": float(nonlinear.square().mean().sqrt()),
    }


def spectral_diagnostics(x):
    spectrum = torch.fft.rfft(x)
    power = spectrum.abs().square()
    probability = power / power.sum()
    entropy = float(
        -(probability[probability > 0] * probability[probability > 0].log()).sum()
        / math.log(len(probability))
    )
    frequency = torch.fft.rfftfreq(len(x), d=1 / K)
    distance = (frequency - CARRIER).abs()
    order = torch.argsort(distance)
    cumulative = torch.cumsum(power[order], dim=0) / power.sum()
    index = int(torch.searchsorted(cumulative, torch.tensor(0.95)))
    return {
        "spectral_entropy": entropy,
        "occupied_bandwidth_95": float(2 * distance[order[index]]),
        "power_outside_carrier_bin": float(
            1 - power[(frequency - CARRIER).abs() <= 2 * K / len(x)].sum() / power.sum()
        ),
        "frequency": frequency,
        "power": power,
    }


def main():
    use_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fig", type=Path, required=True)
    args = parser.parse_args()
    phases = harmonic_phases()
    rows = []
    colors = PALETTE

    fig, axes = plt.subplots(1, 3, figsize=WIDE, constrained_layout=True)
    for color, level in zip(colors, LEVELS, strict=True):
        x = make_phase_map([CARRIER], [level], BETA, 1, N_SPEC, K, SEED)[0]
        spec = spectral_diagnostics(x)
        diagnostics = {
            "level": level,
            **phase_map_diagnostics(level, phases),
            "spectral_entropy": spec["spectral_entropy"],
            "occupied_bandwidth_95": spec["occupied_bandwidth_95"],
            "power_outside_carrier_bin": spec["power_outside_carrier_bin"],
            "rms": float(x.square().mean().sqrt()),
        }
        rows.append(diagnostics)

        time = torch.arange(256)
        sns.lineplot(
            x=time,
            y=x[:256] + (len(LEVELS) - 1 - level) * 2.1,
            ax=axes[0],
            color=color,
            lw=0.75,
            legend=False,
        )

        theta = torch.linspace(0, 2 * math.pi, 512)
        derivative = torch.ones_like(theta)
        for k in range(1, level + 1):
            gamma = BETA * math.sqrt(2 / level)
            derivative += k * gamma * torch.cos(k * theta + phases[k - 1])
        sns.lineplot(
            x=theta,
            y=derivative,
            ax=axes[1],
            color=color,
            lw=1.0,
            label=f"m={level}",
            legend=False,
        )

        mask = spec["frequency"] <= 4
        sns.lineplot(
            x=spec["frequency"][mask],
            y=10 * torch.log10(spec["power"][mask] / spec["power"].max()),
            ax=axes[2],
            color=color,
            lw=0.85,
            legend=False,
        )

    axes[0].set(
        xlabel="Time (samples)",
        ylabel="",
        yticks=[(len(LEVELS) - 1 - level) * 2.1 for level in LEVELS],
        yticklabels=[f"m={level}" for level in LEVELS],
        title="(a) Waveforms",
        xlim=(0, 255),
    )
    axes[0].grid(False)
    axes[1].set(
        xlabel="Phase θ",
        ylabel="Phase-map derivative",
        title="(b) Phase maps",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    axes[2].set(
        xlabel="Frequency (cycles/patch)",
        ylabel="Power (dB)",
        title="(c) Spectra",
        xlim=(0, 4),
        ylim=(-160, 5),
    )
    fig.align_labels(axes)
    fig.savefig(args.fig)

    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
