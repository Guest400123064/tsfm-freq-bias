r"""Figures for P14 (learning dynamics) and the setup they run in.

Reads the stored run JSONs -- no training, no re-computation of anything that was
not already recorded.

  experiments/7_mixed_dynamics/runs/mixed_dynamics_<tag>.json   dynamics
  experiments/6_free_shape/runs/free_shape_s10k.json            free shapes

Writes PNGs to ``experiments/7_mixed_dynamics/figures/``.

Run from the repo root:
``.venv/bin/python experiments/7_mixed_dynamics/scripts/make_figures.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
P14_RUNS = ROOT / "experiments/7_mixed_dynamics/runs"
P13_RUNS = ROOT / "experiments/6_free_shape/runs"
OUT = ROOT / "experiments/7_mixed_dynamics/figures"
NYQUIST = 16.0


def LABEL_BANDS(b):
    r"""Label only a few curve ends: 15 labels collide where the curves converge."""
    return b % 4 == 1 or b == 15


def key(prefix, b):
    return f"{prefix}@{float(b)}"


def rec_with(recs, field):
    r"""First run that actually recorded ``field`` (the dk rerun, for instance)."""
    for r in recs:
        _, _, _, curves = series(r, field)
        if not all(math.isnan(v) for c in curves.values() for v in c):
            return r
    return recs[0]


def load(p14_tag="s20k", p13_tag="s10k"):
    runs = [p for p in sorted(P14_RUNS.glob(f"mixed_dynamics_{p14_tag}*.json"))]
    p14 = []
    for p in runs:
        rec = json.loads(p.read_text())
        rec["config"]["run_tag"] = p.stem.replace("mixed_dynamics_", "")
        p14.append(rec)
    p13_path = P13_RUNS / f"free_shape_{p13_tag}.json"
    p13 = json.loads(p13_path.read_text()) if p13_path.exists() else None
    return p14, p13


def series(rec, field="r"):
    r"""``(steps, bands, shares, {band: [values per step]})`` averaged over seeds."""
    bands = [b for b in rec["config"]["bands"] if b != NYQUIST]
    steps = [m["step"] for m in rec["seeds"][0]["marks"]]
    shares = {b: rec["generator"]["shares"][str(b)] for b in bands}
    curves = {}
    for b in bands:
        vals = []
        for k in range(len(steps)):
            got = [s["marks"][k].get(key(field, b)) for s in rec["seeds"]]
            got = [g for g in got if g is not None]
            vals.append(sum(got) / len(got) if got else float("nan"))
        curves[b] = vals
    return steps, bands, shares, curves


def fig_setup_bands(rec, p13):
    bands = [b for b in rec["config"]["bands"] if b != NYQUIST]
    freqs = {b: f for b, f in zip(rec["config"]["bands"], rec["config"]["freqs"])}
    shares = {b: rec["generator"]["shares"][str(b)] for b in bands}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bs = list(range(1, 17))
    ax[0].bar(bs, [freqs[float(b)] for b in bs], color=sns.color_palette("crest", 16))
    ax[0].set_xlabel("band index $b$ (cycles per patch)")
    ax[0].set_ylabel("frequency (cycles per 512-sample window)")
    ax[0].set_title("the band grid: 16 carriers filling the $k=32$ patch")
    ax[0].axvline(15.5, color="crimson", ls="--", lw=1)
    ax[0].annotate(
        "b16 = Nyquist\n(excluded)",
        xy=(16, freqs[16.0]),
        xytext=(11.5, 210),
        color="crimson",
        fontsize=9,
    )
    ax[0].set_xticks(bs)
    p14_shares = [shares[float(b)] for b in bs[:15]]
    colors = ["0.6"] * 15 + ["crimson"]
    vals = p14_shares + [rec["generator"]["marginal"][str(16.0)][0] / 0.5]
    ax[1].bar(bs, vals, color=colors)
    for b, v in zip(bs[:15], p14_shares):
        ax[1].text(b, v + 0.005, f"{v * 100:.1f}", ha="center", fontsize=7)
    ax[1].set_xlabel("band index $b$")
    ax[1].set_ylabel("share of window power")
    ax[1].set_title("P14 corpus marginal power per band (measured)")
    ax[1].set_xticks(bs)
    fig.tight_layout()
    fig.savefig(OUT / "setup_bands.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_setup_shapes(rec, p13):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    bands = np.arange(1, 17)
    if p13 is not None:
        for name, prof in p13["config"]["profiles"].items():
            ax[0].plot(
                bands, np.array(prof) / 0.5, marker="o", ms=3, lw=1.2, label=name
            )
        ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_title("P13 amplitude spectra: 10 free shapes")
    ax[0].set_ylabel("share of window power")
    mean = np.array(rec["generator"]["mean_profile"])
    sigma = rec["config"]["sigma"]
    ax[1].plot(bands, mean / 0.5, "k-", lw=2, label="mean spectrum")
    ax[1].fill_between(
        bands,
        mean / 0.5 * np.exp(-sigma),
        mean / 0.5 * np.exp(sigma),
        color="0.7",
        alpha=0.5,
        label=r"$\pm1\sigma$ per-window jitter",
    )
    ax[1].set_yscale("log")
    ax[0].set_yscale("log")
    shape = rec["config"].get("shape", "red")
    ax[1].set_title(f"corpus: per-window shapes ({shape} mean, sigma={sigma})")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("band index $b$")
        a.set_xticks(bands[::1])
        a.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "setup_shapes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_setup_phase(rec):
    seed = rec["config"]["seeds"][0]
    gen = np.random.default_rng(seed * 1000 + 71)
    n_win, n_band = rec["config"]["n_per"], len(rec["config"]["bands"])
    phase = 2 * math.pi * gen.random((n_win, n_band))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].hist(phase.ravel(), bins=40, color="steelblue")
    ax[0].set_title("phases: uniform, per window per band")
    ax[0].set_xlabel("phase (rad)")
    ax[1].plot(np.arange(40), phase[:40, 0], "o-", ms=3, label="band 1")
    ax[1].plot(np.arange(40), phase[:40, 7], "s-", ms=3, label="band 8")
    ax[1].set_xlabel("window index")
    ax[1].set_ylabel("phase (rad)")
    ax[1].set_title("phases differ window to window\n(no constant-tone shortcut)")
    ax[1].legend(fontsize=8)
    ax[2].scatter(phase[:, 0], phase[:, 7], s=6, alpha=0.6)
    ax[2].set_xlabel("band 1 phase (rad)")
    ax[2].set_ylabel("band 8 phase (rad)")
    ax[2].set_title("bands independent within a window")
    fig.tight_layout()
    fig.savefig(OUT / "setup_phase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def matrix(rec, field):
    steps, bands, shares, curves = series(rec, field)
    return np.array([curves[b] for b in bands]), bands, steps, shares


def fig_heatmap(rec, field, fname, label, cmap):
    r"""One panel: the colour limits hug the data so the structure is visible."""
    m, bands, steps = (
        matrix(rec, field)[0],
        matrix(rec, field)[1],
        matrix(rec, field)[2],
    )
    if np.isnan(m).all():
        print(f"  skip {fname}: no {field} recorded in any run")
        return
    stride = max(1, len(steps) // 10)  # a fine checkpoint grid would repeat "0k"
    labels = [f"{s / 1000:g}k" if k % stride == 0 else "" for k, s in enumerate(steps)]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        m,
        ax=ax,
        cmap=cmap,
        vmin=float(np.nanmin(m)),
        vmax=float(np.nanmax(m)),
        xticklabels=labels,
        yticklabels=[f"b{int(b)}" for b in bands],
        cbar_kws={"label": label},
    )
    ax.set_xlabel("SGD step")
    ax.set_title(f"{label} by band -- run {rec['config']['run_tag']}", fontsize=10)
    ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_curves(rec, field="r", fname="p14_curves.png"):
    steps, bands, shares, curves = series(rec, field)
    if all(math.isnan(v) for c in curves.values() for v in c):
        print(f"  skip {fname}: no {field} recorded in this run")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("viridis")
    lo, hi = math.log(min(shares.values())), math.log(max(shares.values()))
    for b in bands:
        c = cmap((math.log(shares[b]) - lo) / (hi - lo))
        ax.plot(steps, curves[b], color=c, lw=1.6)
        if LABEL_BANDS(int(b)):
            ax.annotate(
                f"b{int(b)}",
                (steps[-1], curves[b][-1]),
                fontsize=7,
                color=c,
                xytext=(3, -2),
                textcoords="offset points",
            )
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(min(shares.values()), max(shares.values()))
    )
    fig.colorbar(sm, ax=ax, label="band's share of the loss")
    ax.set_xlabel("SGD step")
    ax.set_ylabel(r"retention $r$" if field == "r" else r"$\Delta_k$")
    ax.set_title(
        f"per-band learning curves: {ax.get_ylabel()}, coloured by power share"
    )
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_collapse(rec):
    steps, bands, shares, curves = series(rec, "r")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = plt.get_cmap("viridis")
    lo, hi = math.log(min(shares.values())), math.log(max(shares.values()))
    pts = []
    for b in bands:
        u = [s * shares[b] for s in steps]
        c = cmap((math.log(shares[b]) - lo) / (hi - lo))
        ax[0].plot(u, curves[b], color=c, lw=1.5)
        if LABEL_BANDS(int(b)):
            ax[0].annotate(
                f"b{int(b)}",
                (u[-1], curves[b][-1]),
                fontsize=7,
                color=c,
                xytext=(3, -2),
                textcoords="offset points",
            )
        pts += [(uu, v, b) for uu, v in zip(u, curves[b]) if v < 0.9]
    ax[0].set_xlabel(r"$u = \mathrm{step} \times \mathrm{share}_b$")
    ax[0].set_ylabel(r"retention $r$")
    ax[0].set_title("time collapse: rescaling by share puts the bands on one curve")
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(min(shares.values()), max(shares.values()))
    )
    fig.colorbar(sm, ax=ax[0], label="share")

    pts.sort()
    nb = 8
    xs, means, spreads = [], [], []
    for k in range(nb):
        chunk = pts[k * len(pts) // nb : (k + 1) * len(pts) // nb]
        if not chunk:
            continue
        rs = [c[1] for c in chunk]
        xs.append(sum(c[0] for c in chunk) / len(chunk))
        means.append(sum(rs) / len(rs))
        spreads.append(max(rs) - min(rs))
    ax[1].errorbar(xs, means, yerr=np.array(spreads) / 2, fmt="o-", capsize=4)
    ax[1].set_xlabel(r"$u$ (bin centre)")
    ax[1].set_ylabel(r"$r$ (bin mean $\pm$ half-spread)")
    ax[1].set_title(f"binned collapse: worst spread {max(spreads):.3f} (tol 0.15)")
    fig.tight_layout()
    fig.savefig(OUT / "p14_collapse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_t80(rec):
    d = rec["decision"]
    shares = d["shares"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    bands = sorted(shares, key=lambda b: shares[b])
    ax[0].plot([shares[b] for b in bands], [d["t80"][b] for b in bands], "o")
    for b in bands:
        ax[0].annotate(
            f"b{int(float(b))}",
            (shares[b], d["t80"][b]),
            fontsize=7,
            xytext=(3, 2),
            textcoords="offset points",
        )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("band's share of the loss")
    ax[0].set_ylabel(r"$t_{80}$ (steps to reach $r=0.8$)")
    ax[0].set_title(f"ordering: Spearman = {d['spearman_share_vs_t80']:+.3f}")
    for b in bands:
        ax[1].plot([shares[b]], [d["t80"][b]], "o", color="k")
    xs = np.array([shares[b] for b in bands])
    ys = np.array([d["t80"][b] for b in bands])
    ax[1].plot(
        xs,
        ys.mean() / xs * xs.min(),
        "--",
        color="crimson",
        lw=1,
        label=r"$t \propto 1/\mathrm{share}$ through the first point",
    )
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("share")
    ax[1].set_ylabel(r"$t_{80}$")
    ax[1].set_title("against the $1/\\mathrm{share}$ prediction")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "p14_t80_share.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_early(rec):
    steps, bands, shares, curves = series(rec, "r")
    marks = [i for i, s in enumerate(steps) if s <= 4000]
    fig, ax = plt.subplots(1, len(marks), figsize=(4.1 * len(marks), 4.0), sharey=True)
    cmap = plt.get_cmap("turbo")
    for a, k in zip(np.atleast_1d(ax), marks):
        for b in sorted(bands, key=lambda x: -shares[x]):
            a.scatter(shares[b], curves[b][k], color=cmap((b - 1) / 15), s=45, zorder=3)
        a.set_xscale("log")
        a.set_xlabel("band's share of the loss")
        a.set_title(f"step {steps[k] // 1000}k", fontsize=10)
    np.atleast_1d(ax)[0].set_ylabel(r"retention $r$")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 16))
    fig.colorbar(sm, ax=list(np.atleast_1d(ax)), label="band index $b$ (frequency)")
    fig.suptitle(
        "early rise is ordered by power, at every frequency "
        "(colour = which band: no high-frequency-only pattern)"
    )
    fig.savefig(OUT / "p14_early_share_vs_r.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_oracle(rec):
    steps, bands, shares, curves = series(rec, "r_oracle")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cmap = plt.get_cmap("viridis")
    lo, hi = math.log(min(shares.values())), math.log(max(shares.values()))
    for b in bands:
        c = cmap((math.log(shares[b]) - lo) / (hi - lo))
        ax.plot(steps, curves[b], color=c, lw=1.4)
    ax.axhline(0.9, color="crimson", ls="--", lw=1, label="pre-registered oracle floor")
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(min(shares.values()), max(shares.values()))
    )
    fig.colorbar(sm, ax=ax, label="band's share of the loss")
    ax.set_xlabel("SGD step")
    ax.set_ylabel("oracle retention")
    ax.set_title("the representation learns the weak bands too (frozen-latent oracle)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "p14_oracle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_exponent(rec):
    r"""``t80 ~ share**(-beta)``: fitted beta, plus the two reference values."""
    d = rec["decision"]
    xs = [math.log(d["shares"][b]) for b in d["shares"]]
    ys = [math.log(d["t80"][b]) for b in d["shares"]]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    return -slope


def collapse_worst(rec, beta, n_bins=8):
    r"""Worst within-bin spread of ``r`` under ``u = step * share**beta``."""
    bands = [b for b in rec["config"]["bands"] if b != NYQUIST]
    shares = {b: rec["generator"]["shares"][str(b)] for b in bands}
    pts = []
    for s in rec["seeds"]:
        for m in s["marks"]:
            for b in bands:
                v = m[key("r", b)]
                if v < 0.9:
                    pts.append((m["step"] * shares[b] ** beta, v, b))
    pts.sort()
    worst = 0.0
    for k in range(n_bins):
        chunk = pts[k * len(pts) // n_bins : (k + 1) * len(pts) // n_bins]
        rs = [c[1] for c in chunk]
        if len({c[2] for c in chunk}) >= 2:
            worst = max(worst, max(rs) - min(rs))
    return worst


def fig_scaling_law(rec):
    d = rec["decision"]
    beta = fit_exponent(rec)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    xs = [d["shares"][b] for b in d["shares"]]
    ys = [d["t80"][b] for b in d["shares"]]
    ax[0].plot(xs, ys, "o", label=r"$t_{80}$ per band")
    gm = math.exp(sum(math.log(v) for v in xs) / len(xs))
    tm = math.exp(sum(math.log(v) for v in ys) / len(ys))
    xr = np.array([min(xs) * 0.95, max(xs) * 1.05])
    for gamma, style, col, lab in (
        (0.5, "--", "seagreen", r"$1/\mathrm{amplitude}$, slope $-0.5$"),
        (1.0, ":", "crimson", r"$1/\mathrm{power}$, slope $-1$"),
    ):
        ax[0].plot(xr, tm * (xr / gm) ** (-gamma), style, color=col, label=lab)
    ax[0].set_ylim(min(ys) * 0.75, max(ys) * 1.35)
    floor = min(d["t80"].values())
    floored = [b for b in d["shares"] if d["t80"][b] <= floor]
    ax[0].annotate(
        f"$t_{{80}}$ floor = {floor / 1000:.0f}k: "
        f"{len(floored)} of {len(d['shares'])} bands censored",
        xy=(max(xs), min(ys)),
        xytext=(-6, 10),
        textcoords="offset points",
        fontsize=8,
        ha="right",
        color="0.35",
    )
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("band's share of the loss")
    ax[0].set_ylabel(r"$t_{80}$ (steps)")
    ax[0].set_title(rf"fitted $t_{{80}} \propto \mathrm{{share}}^{{-{beta:.2f}}}$")
    ax[0].legend(fontsize=8)

    betas = np.linspace(0.1, 1.1, 11)
    worst = [collapse_worst(rec, b) for b in betas]
    ax[1].plot(betas, worst, "o-")
    ax[1].axhline(0.15, color="crimson", ls="--", label="pre-registered tolerance")
    ax[1].axvline(1.0, color="0.5", ls=":", label=r"pre-registered $\beta=1$")
    ax[1].axvline(beta, color="seagreen", ls=":", label=f"fitted {beta:.2f}")
    ax[1].set_xlabel(r"$\beta$ in $u = \mathrm{step} \times \mathrm{share}^\beta$")
    ax[1].set_ylabel("worst within-bin spread of $r$")
    ax[1].set_title("no exponent collapses the family")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "p14_scaling_law.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_p13(p13):
    if p13 is None:
        return
    bands = [b for b in p13["config"]["bands"] if b != NYQUIST]
    cmap = plt.get_cmap("turbo")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs, ys = [], []
    for arm, v in p13["arms"].items():
        prof = p13["config"]["profiles"][arm]
        for j, b in enumerate(bands):
            share = prof[j] / 0.5
            r = v["mean_std"][key("r", b)][0]
            ax.scatter(share, r, color=cmap((b - 1) / 15), s=32, zorder=3)
            xs.append(math.log(share))
            ys.append(r)
    order = np.argsort(xs)
    ax.plot(np.exp(np.array(xs)[order]), np.array(ys)[order], "k-", lw=0.8, alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("band's share of the loss")
    ax.set_ylabel(r"retention $r$")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 16))
    fig.colorbar(sm, ax=ax, label="band index $b$ (frequency)")
    ax.set_title(
        "P13: 10 free shapes, 150 points\n"
        "colour (frequency) mixes along one curve -- "
        f"share-only $R^2$ = {p13['decision']['share_only_r2']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(OUT / "p13_share_vs_r.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--p14-tag", default="s20k")
    ap.add_argument("--p13-tag", default="s10k")
    ap.add_argument("--out", default="figures", help="output subdirectory")
    args = ap.parse_args(argv)
    global OUT
    OUT = ROOT / "experiments/7_mixed_dynamics" / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    recs, p13 = load(args.p14_tag, args.p13_tag)
    print(f"loaded {len(recs)} P14 run(s), P13 {'yes' if p13 else 'no'}")

    primary = recs[0]
    fig_setup_bands(primary, p13)
    fig_setup_shapes(primary, p13)
    fig_setup_phase(primary)
    fig_heatmap(
        rec_with(recs, "r"), "r", "p14_heatmap_r.png", "retention $r$", "viridis"
    )
    fig_heatmap(
        rec_with(recs, "dk"),
        "dk",
        "p14_heatmap_dk.png",
        r"Fredformer $\Delta_k$",
        "magma_r",
    )
    fig_curves(primary, "r", "p14_curves.png")
    fig_curves(rec_with(recs, "dk"), "dk", "p14_curves_dk.png")
    fig_collapse(primary)
    fig_t80(primary)
    fig_early(primary)
    fig_oracle(primary)
    fig_scaling_law(primary)
    fig_p13(p13)
    for p in sorted(OUT.glob("*.png")):
        print(f"  wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
