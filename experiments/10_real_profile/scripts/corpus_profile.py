r"""T5 step 1: profile the real corpus before intervening on it.

Why. The intervention we want to run -- adding synthetic windows to balance the
corpus's *marginal* band power, never touching the real data's own spectrum --
needs three things first: what the corpus looks like in the band scale, which
bands are measurable at all (the matched filter leaks), and which bands are
*capturable* (conditionally predictable). This script produces that profile and
nothing else: no training.

Normalization (the setting the model will see). Per window, statistics from the
**context only**:

    mu_i = mean(context_i),  sigma_i = std(context_i)     (never the target)
    x_norm = (x - mu_i) / sigma_i      over the whole (ctx + k)-sample window

The model will only ever see normalized data, the loss will be computed in that
space, and nothing is denormalized. Taking the statistics from the context alone
-- not from ctx + k -- is what keeps the target's own amplitude out of its
normalization; using the whole window would mechanically pull retention toward 1.
Note what this does *not* do: a single per-window scalar rescaling cannot
equalize bands, so the corpus's spectral *shape* survives untouched, which is the
quantity under study.

Deliverables (the numbering follows the profiling plan):

  1. splits and window counts;
  3. per-band power and `in-band` (leakage share), on the normalized corpus and,
     for comparison, on the T1 corpus (per-series norm + per-patch demean);
  4. per-band conditional predictability ``P(b)`` with its null control, computed
     through the *same* normalization so the control is meaningful;
  8. split-half reproducibility of the band profiles;
 10. the power floor (bands near it are not measurable and cannot be intervened
     on).

Items 2, 5, 6 and 7 need a trained model and are the next script.

Run from the repo root:
``.venv/bin/python experiments/10_real_profile/scripts/profile.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PRED_PATH = ROOT / "experiments/1_realdata/scripts/predictability.py"
RUNS = ROOT / "experiments/10_real_profile/runs"


def load_pred():
    r"""Import T1's module so its estimator is reused verbatim, not reimplemented."""
    spec = importlib.util.spec_from_file_location("pred", PRED_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pred"] = mod  # dataclasses needs the module registered
    spec.loader.exec_module(mod)
    return mod


def norm_patches(series, cfg):
    r"""Per-context-window normalization, ``(n_windows, n_patch, k)``.

    Context-only statistics, applied to the whole window. Non-overlapping
    windows, as in T1, so a window split cannot share samples.
    """
    out = []
    for v in series:
        n = len(v) // cfg.win
        x = v[: n * cfg.win].reshape(n, cfg.win)
        mu = x[:, : cfg.ctx].mean(axis=1, keepdims=True)
        sd = x[:, : cfg.ctx].std(axis=1, keepdims=True)
        out.append(((x - mu) / sd).reshape(n, cfg.n_patch, cfg.k))
    return np.concatenate(out, axis=0)


def coeffs_of(patches, bands, pred):
    return np.stack([pred.matched(patches, b) for b in bands], axis=-1)


def in_band(patches, bands, cfg, half=0.5):
    r"""Share of each band's reading that survives its own band-limiting.

    A ``k``-sample matched filter has a mainlobe of about ``+-1`` cycle/patch, so
    keeping ``|f - b| <= half`` cycles/patch is the most generous reading of what
    band ``b``'s own content is; everything else in the coefficient is leakage
    from other frequencies. Computed per window, then averaged -- the same
    definition T1 used, applied to the normalized windows.
    """
    total = np.zeros(len(bands))
    inside = np.zeros(len(bands))
    flat = patches.reshape(-1, cfg.k)
    freqs = np.fft.rfftfreq(cfg.k) * cfg.k  # cycles per patch
    for j, b in enumerate(bands):
        z = np.fft.rfft(flat, axis=-1)
        keep = np.abs(freqs - b) <= half
        banded = np.fft.irfft(z * keep, n=cfg.k, axis=-1).reshape(patches.shape)
        total[j] = np.mean(np.abs(_matched(patches, b)) ** 2)
        inside[j] = np.mean(np.abs(_matched(banded, b)) ** 2)
    return inside / total


_MATCH = None


def _matched(patches, b):
    return _MATCH(patches, b)


def profile(cfg, series, pred, noise_series=None):
    r"""Per-band power, share, leakage, predictability and split-half stability."""
    t0 = time.time()
    patches = norm_patches(series, cfg)
    coeffs = coeffs_of(patches, cfg.bands, pred)
    n_win = len(patches)

    power = pred.band_power(coeffs)
    share = power / power.sum()
    leak = in_band(patches, cfg.bands, cfg)

    half = n_win // 2
    p_a = pred.band_power(coeffs[:half])
    p_b = pred.band_power(coeffs[half:])
    leak_a = in_band(patches[:half], cfg.bands, cfg)
    leak_b = in_band(patches[half:], cfg.bands, cfg)

    res, n_tr, n_ev = pred.run(cfg, coeffs, ("own",), seed=pred.SPLIT_SEED)
    p_own = np.array(res["own"]["p"])
    null = None
    if noise_series is not None:
        ncoeffs = coeffs_of(norm_patches(noise_series, cfg), cfg.bands, pred)
        nres, _, _ = pred.run(cfg, ncoeffs, ("own",), seed=pred.SPLIT_SEED)
        null = np.array(nres["own"]["p"])

    return {
        "k": cfg.k,
        "ctx": cfg.ctx,
        "n_windows": int(n_win),
        "n_train": int(n_tr),
        "n_eval": int(n_ev),
        "seconds": round(time.time() - t0, 1),
        "bands": [float(b) for b in cfg.bands],
        "period_h": [float(v) for v in cfg.periods_h],
        "power": [float(v) for v in power],
        "share": [float(v) for v in share],
        "in_band": [float(v) for v in leak],
        "power_half_a": [float(v) for v in p_a],
        "power_half_b": [float(v) for v in p_b],
        "in_band_half_a": [float(v) for v in leak_a],
        "in_band_half_b": [float(v) for v in leak_b],
        "p_own": [float(v) for v in p_own],
        "p_own_null": None if null is None else [float(v) for v in null],
    }


def rank_corr(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den else 0.0


def summarise(rec):
    p = np.array(rec["power"])
    leak = np.array(rec["in_band"])
    pa, pb = np.array(rec["power_half_a"]), np.array(rec["power_half_b"])
    ja, jb = np.array(rec["in_band_half_a"]), np.array(rec["in_band_half_b"])
    floor = float(np.percentile(p, 10))
    return {
        "power_range": float(p.max() / p.min()),
        "power_floor_p10": floor,
        "bands_at_floor": [int(i) for i in np.where(p <= 3 * floor)[0]],
        "in_band_min": float(leak.min()),
        "in_band_median": float(np.median(leak)),
        "split_half_power_ratio": float(pa.max() / pa.min() and 1.0),
        "split_half_rank_corr_power": rank_corr(pa, pb),
        "split_half_rank_corr_in_band": rank_corr(ja, jb),
        "split_half_median_rel_diff_power": float(
            np.median(np.abs(pa - pb) / ((pa + pb) / 2))
        ),
    }


def table(head, rows):
    line = " ".join(head)
    print(line)
    print("  " + "-" * (len(line) - 2))
    for r in rows:
        print(" ".join(r))


def print_band_table(rec):
    n = len(rec["bands"])
    print(f"\nk = {rec['k']}, ctx = {rec['ctx']}, windows = {rec['n_windows']}")
    table(
        [
            f"  {'b':>5}",
            f"{'period_h':>9}",
            f"{'share':>8}",
            f"{'in-band':>8}",
            f"{'P_own':>7}",
            f"{'P_null':>7}",
            f"{'power A/B':>10}",
        ],
        [
            [
                f"  {rec['bands'][j]:>5.1f}",
                f"{rec['period_h'][j]:>9.2f}",
                f"{rec['share'][j]:>8.5f}",
                f"{rec['in_band'][j]:>8.3f}",
                f"{rec['p_own'][j]:>7.3f}",
                (f"{rec['p_own_null'][j]:>7.3f}" if rec["p_own_null"] else f"{'-':>7}"),
                f"{rec['power_half_a'][j] / max(rec['power_half_b'][j], 1e-12):>10.3f}",
            ]
            for j in range(n)
        ],
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ks", nargs="+", type=int, default=[32, 64])
    ap.add_argument("--ctx-patches", type=int, default=16)
    ap.add_argument("--grid", default="t1", choices=("t1", "phys"), help="band grid")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    pred = load_pred()
    global _MATCH
    _MATCH = pred.matched

    _, raw = pred.load_monash(pred.DATASET)
    series = [v for _, _, v in raw]
    print(
        f"dataset {pred.DATASET}: {len(series)} series, "
        f"{sum(len(v) for v in series)} samples, {pred.STEP_MIN:.0f}-min steps"
    )
    noise = pred.white_noise(length=len(series[0]) * len(series), n=len(series))

    out = {"dataset": pred.DATASET, "configs": {}, "t1_corpus": {}}
    for k in args.ks:
        cfg = pred.config(k, args.ctx_patches * k, args.grid)
        print(f"\nprofiling k = {k} (ctx = {cfg.ctx}, win = {cfg.win}) ...", flush=True)
        rec = profile(cfg, series, pred, noise_series=noise)
        rec["summary"] = summarise(rec)
        out["configs"][str(k)] = rec
        print_band_table(rec)
        s = rec["summary"]
        print(
            f"  power range {s['power_range']:.1f}x   power floor (p10) "
            f"{s['power_floor_p10']:.2e}   bands within 3x of the floor: "
            f"{s['bands_at_floor']}"
        )
        print(
            f"  in-band min {s['in_band_min']:.3f} / median {s['in_band_median']:.3f}"
        )
        print(
            f"  split-half: median |rel diff| of power "
            f"{s['split_half_median_rel_diff_power']:.3f}, rank corr "
            f"{s['split_half_rank_corr_power']:+.3f}; in-band rank corr "
            f"{s['split_half_rank_corr_in_band']:+.3f}"
        )

        t1 = pred.corpus(cfg)
        tp = pred.band_power(t1)
        out["t1_corpus"][str(k)] = {
            "power": [float(v) for v in tp],
            "share": [float(v) for v in tp / tp.sum()],
        }
        print(
            f"  T1 corpus (per-series norm + per-patch demean): power range "
            f"{tp.max() / tp.min():.1f}x"
        )

    tag = f"_{args.tag}" if args.tag else ""
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"profile{tag}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
