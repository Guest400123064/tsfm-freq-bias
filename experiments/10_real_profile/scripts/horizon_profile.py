r"""T5 step 5: does the forecast horizon create a predictability gradient?

Everything in the project so far forecasts **one patch** ahead -- 32 samples, 16 h
at k = 32. That is short. The TSFM literature forecasts 96-720 samples, and the
mechanisms supposed to make high frequencies intrinsically harder (turbulence's
scale-local cascade, autocorrelation decay, microstructure noise in markets) are
all statements about how predictability *decays with lead time*. They predict that
the per-band predictability gradient is **small at short leads and grows with the
lead**.

This script measures that directly and model-free. For each horizon ``h`` in
patches the baseline's band-only ridge -- context's own coefficients at a band to
the target patch's coefficient at that band, fitted on training windows, scored on
held-out ones -- is refitted with the target moved ``h`` patches past the context.
Nothing is trained.

Three corpora, so the real reading comes with a benchmark and a floor:

  real   the Monash electricity corpus, window starts identical across horizons;
  pm     synthetic phase-diffusing tones carrying the real profile and the real
         per-band ``P(b)``. The phase advances by an independent ``beta * eps``
         per patch boundary, so over ``h`` patches the cumulative step is
         ``N(0, h beta^2)`` and the predictable variance fraction is exactly
         ``P_h(b) = P_1(b)**h`` -- an **analytic** benchmark for a corpus whose
         predictability is memoryless per patch, and a self-consistency check on
         the estimator;
  null   white noise through the identical pipeline: the estimator's floor.

Reading. ``P_h(b)`` falls with ``h`` for every band, so the level is not the
question. The discrimination is in the **gradient across bands**:
``P_h(b) / P_1(b)**h`` says which bands lose predictability faster than the
memoryless benchmark, and ``Spearman(P_h(b), b)`` says whether a gradient appears
where the one-patch reading had none.

Pre-registered, before the run:

  1. the pipeline is sound -- the null reads |P| <= 0.05 at every horizon and
     band, and the PM corpus reproduces ``P_1(b)**h`` to within 0.03 absolute;
  2. if the real corpus tracks ``P_1(b)**h``, its long-horizon gradient is
     *inherited* from the one-patch gradient and nothing new appears -- the
     mechanisms above would then need a band-specific decorrelation to show up,
     not just a long horizon;
  3. the mechanisms' signature is ``P_h(b) / P_1(b)**h`` falling with ``b``, and
     ``Spearman(P_h(b), b)`` growing more negative with ``h``. Its absence is the
     answer, and it is the answer to "why does the bias sit at high frequency".

Run from the repo root:
``.venv/bin/python experiments/10_real_profile/scripts/horizon_profile.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from fbias.data import make_pm_profile  # noqa: E402

BASE_PATH = Path(__file__).resolve().parent / "baseline.py"
RUNS = ROOT / "experiments/10_real_profile/runs"

K, CTX = 32, 512
BANDS = tuple(float(b) for b in range(1, 17))
WELL = BANDS[:-1]  # b16 is Nyquist: its matched filter is its own mirror
FREQS = torch.tensor([b * CTX / K for b in BANDS])
HORIZONS = (1, 2, 4, 8, 16)
MAX_H = max(HORIZONS)
STRIDE = CTX + MAX_H * K  # one stride for every h: identical window starts
PM_SEED = 909
NULL_TOL, PM_TOL = 0.05, 0.03


def load_baseline():
    spec = importlib.util.spec_from_file_location("bl", BASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bl"] = mod
    spec.loader.exec_module(mod)
    return mod


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / math.sqrt((rx**2).sum() * (ry**2).sum()))


def norm(patches):
    r"""Per-window normalization from the context: the standing T5 convention."""
    n, n_patch, k = patches.shape
    x = patches.reshape(n, -1)
    mu = x[:, :CTX].mean(axis=1, keepdims=True)
    sd = x[:, :CTX].std(axis=1, keepdims=True)
    return ((x - mu) / sd).reshape(n, n_patch, k)


def real_windows(series, h):
    win = CTX + h * K
    out = []
    for v in series:
        n = len(v) // STRIDE
        x = v[np.arange(n)[:, None] * STRIDE + np.arange(win)[None, :]]
        out.append(x.reshape(n, -1, K))
    return norm(np.concatenate(out))


def null_windows(series, h, rng):
    win = CTX + h * K
    out = [
        rng.standard_normal((len(v) // STRIDE, win)).reshape(-1, win // K, K)
        for v in series
    ]
    return norm(np.concatenate(out))


def pm_windows(power, betas, n, h):
    r"""PM windows of ``16 + h`` patches, with the phase prefix shared across ``h``.

    ``make_pm_profile`` builds ``ctx + k`` samples with patches of ``k``; asking
    for ``ctx = 32 (15 + h)`` and ``k = 32`` gives a ``16 + h`` patch window, and
    scaling ``freqs`` by ``15 + h`` keeps band ``b`` at ``b`` cycles per patch.
    The RNG draws the phase steps for all patches at once, so the first sixteen
    patches are the same prefix at every ``h`` -- the corpora are nested.
    """
    freqs = torch.tensor([b * (15 + h) for b in BANDS])
    x, _ = make_pm_profile(freqs, power, betas, n, K * (15 + h), K, PM_SEED)
    return norm(x.reshape(n, 16 + h, K))


def p_by_band(bl, patches, h):
    r"""``P_h(b)`` for every band: the baseline ridge, target ``h`` patches out."""
    x = torch.from_numpy(np.ascontiguousarray(patches, dtype=np.float32))
    n = len(patches)
    tr, ev = np.arange(n // 2), np.arange(n // 2, n)
    out = {}
    for b in BANDS:
        c = bl._matched_np(x, b, K)  # (n, 16 + h)
        f = c[:, : CTX // K].reshape(n, -1)  # the context
        z = c[:, CTX // K + h - 1]  # the target, h patches past it
        lam = bl._pick_lambda(f, z, tr)
        out[b] = bl._p_band(f[tr], z[tr], f[ev], z[ev], lam)
    return out


def span(P, name, h):
    r"""``P_h(b1) / P_h(b15)``, or NaN where the high band has no measurable signal."""
    lo = P[name][str(h)]["15.0"]
    return float("nan") if lo < 0.05 else P[name][str(h)]["1.0"] / lo


def ratio(P, name, h, b):
    r"""``P_h(b)`` over the memoryless benchmark ``P_1(b)**h``."""
    return P[name][str(h)][str(b)] / max(P[name]["1"][str(b)] ** h, 1e-9)


def mae_vs_memoryless(P, name, h):
    return float(
        np.mean(
            [abs(P[name][str(h)][str(b)] - P[name]["1"][str(b)] ** h) for b in WELL]
        )
    )


def table(head, rows):
    line = " ".join(head)
    print(line)
    print("  " + "-" * (len(line) - 2))
    for r in rows:
        print(" ".join(r))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    bl = load_baseline()
    prof = bl.load_profile()
    pred = prof.load_pred()
    _, raw = pred.load_monash(pred.DATASET)
    series = [v for _, _, v in raw]
    n_win = sum(len(v) // STRIDE for v in series)

    base = bl.build(K, CTX, series, "random")
    b0 = bl.band_bound(base[0], BANDS, base[3], base[4])
    power = bl.band_power(base[0], BANDS, K)
    betas = torch.tensor([math.sqrt(max(-math.log(b0[str(b)]), 0.0)) for b in BANDS])
    print(
        f"corpus: {len(series)} series, {sum(len(v) for v in series)} samples, "
        f"stride {STRIDE} -> {n_win} windows at every horizon"
    )
    print(
        "  the target is the h-th patch after the context, so the lead to its first "
        f"sample is {K // 2 * (min(HORIZONS) - 1)}-{K // 2 * (max(HORIZONS) - 1)} h "
        f"and to its last {K // 2 * min(HORIZONS)}-{K // 2 * max(HORIZONS)} h"
    )
    print(
        "  PM betas from the baseline's P_1(b): "
        + " ".join(f"{float(betas[i]):.3f}" for i in range(8))
        + " ..."
    )

    rng = np.random.default_rng(0)
    corpora = {
        "real": {h: real_windows(series, h) for h in HORIZONS},
        "null": {h: null_windows(series, h, rng) for h in HORIZONS},
        "pm": {h: pm_windows(power, betas, n_win, h) for h in HORIZONS},
    }
    out = {"config": vars(args), "horizons": list(HORIZONS), "n_windows": n_win}
    out["corpora"] = {
        name: {
            str(h): {str(b): v for b, v in p_by_band(bl, w[h], h).items()}
            for h in HORIZONS
        }
        for name, w in corpora.items()
    }
    P = out["corpora"]

    print("\n== P_h(b) ==")
    for name in P:
        print(f"\n  corpus {name}")
        table(
            [
                f"    {'b':>4}",
                f"{'period_h':>9}",
                *[f"{f'h={h}':>8}" for h in HORIZONS],
            ],
            [
                [
                    f"    {int(b):>4}",
                    f"{K / b * 0.5:>9.2f}",
                    *[f"{P[name][str(h)][str(b)]:>8.3f}" for h in HORIZONS],
                ]
                for b in WELL
            ],
        )

    print("\n== the gradient across bands, and the memoryless benchmark ==")
    table(
        [
            f"  {'corpus':<6}",
            f"{'h':>3}",
            f"{'P(b1)':>7}",
            f"{'P(b15)':>7}",
            f"{'span':>6}",
            f"{'Spearman':>9}",
            f"{'vs P_1^h MAE':>13}",
        ],
        [
            [
                f"  {name:<6}",
                f"{h:>3}",
                f"{P[name][str(h)]['1.0']:>7.3f}",
                f"{P[name][str(h)]['15.0']:>7.3f}",
                f"{span(P, name, h):>6.2f}",
                f"{spearman(WELL, [P[name][str(h)][str(b)] for b in WELL]):>+9.3f}",
                f"{mae_vs_memoryless(P, name, h):>13.3f}",
            ]
            for name in P
            for h in HORIZONS
        ],
    )

    print("\n== pre-registered checks ==")
    null_worst = max(abs(P["null"][str(h)][str(b)]) for h in HORIZONS for b in WELL)
    # only where the benchmark is above the estimator's own floor
    cells = [
        (abs(P["pm"][str(h)][str(b)] - P["pm"]["1"][str(b)] ** h), b, h)
        for h in HORIZONS
        for b in WELL
        if P["pm"]["1"][str(b)] ** h >= 0.10
    ]
    pm_worst, pm_b, pm_h = max(cells)
    pm_mean = float(np.mean([c[0] for c in cells]))
    print(
        f"  1. null worst |P| = {null_worst:.4f} (tol {NULL_TOL});  "
        f"PM vs P_1**h: mean |diff| {pm_mean:.4f}, worst {pm_worst:.4f} at "
        f"b{int(pm_b)} h={pm_h}, over {len(cells)} cells with P_1**h >= 0.10 "
        f"(tol {PM_TOL}, as written)"
    )
    print("\n  2/3. real corpus over its own P_1(b)**h  (1.00 = memoryless per patch)")
    table(
        [f"    {'b':>4}", f"{'period_h':>9}", *[f"{f'h={h}':>9}" for h in HORIZONS]],
        [
            [
                f"    {int(b):>4}",
                f"{K / b * 0.5:>9.2f}",
                *[f"{ratio(P, 'real', h, b):>9.3f}" for h in HORIZONS],
            ]
            for b in WELL
        ],
    )
    print("\n  the same for the PM benchmark (should be 1.00) and the null, over")
    print("  the bands whose benchmark is still above the estimator's floor")
    for name in ("pm", "null"):
        parts = []
        for h in HORIZONS:
            ok = [b for b in WELL if P[name]["1"][str(b)] ** h >= 0.10]
            parts.append(
                f"h={h} {np.mean([ratio(P, name, h, b) for b in ok]):.2f}"
                if ok
                else f"h={h} n/a"
            )
        print(f"    {name}: " + " ".join(parts))

    RUNS.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = RUNS / f"horizon_profile{tag}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
