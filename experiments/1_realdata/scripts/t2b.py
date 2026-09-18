r"""T2b: the shape-hypothesis control -- does flattening the corpus remove T2's decline?

T2 (``scripts/t2.py``) found that a model trained on the real corpus attenuates
a **clean, flat-spectrum** probe by 7.9x (``k32``) / 5.8x (``k64``) from ``b = 1``
to ``b = 12``, while the corpus's own per-band predictability ``P(b)`` spans only
1.31x. Two accounts fit that number:

(a) the model carries a band-wise prior calibrated to the corpus's per-band
    *predictability* (``PLAN.md`` §0), and a corpus with a flat ``P(b)`` should
    have produced a flat probe curve;
(b) the model responds to the corpus's spectral **shape** / dynamic range.
    Contract A9 matches the probe's *total* RMS to the corpus, which on a 617x-red
    corpus leaves the probe at 0.29x the corpus's lowest band and **180x** its
    highest -- spectrally out of distribution in shape.

This run separates them with two arms that differ in exactly one thing:

===========  ============================  ==========================
arm          training corpus               probe
===========  ============================  ==========================
``A``        the corpus as T2 used it      T2's flat, clean probe
``B``        the same corpus, spectrally   the *identical* probe
             flattened
===========  ============================  ==========================

``C`` is a dose check: the same flattening at half strength.

**The flattening.** One linear filter, applied to each whole series:

``g(f) = S(f) ** (-alpha / 2)``, ``alpha in {1 (B), 0.5 (C), 0 (A)}``

where ``S`` is the corpus's mean power spectrum -- the mean over the five series
of their whole-record periodograms, interpolated onto a common frequency grid --
smoothed with a boxcar of half-width ``HALF_B = 1.0`` b-units at
``K_REF = 64``, i.e. ``1 / 64`` cycles/sample. The matched filter's mainlobe is
``+-1`` b-units wide at *every* ``k`` (nulls at integer ``d b``, contract A4), so
a ``+-1``-b boxcar is the narrowest smoothing that does not try to equalise
structure *inside* one aperture's passband, and ``k = 64`` is the finer of the
two grids this experiment reads. No floor and no ceiling are applied: the gain
hands the record's own trend and clock lines the same treatment as everything
else, and the series is re-standardised to unit variance afterwards, so the
filter's overall scale is irrelevant. ``alpha = 0`` is the identity, by
construction -- ``np.array_equal`` against ``t2.raw_windows`` asserts it -- so
arm A *is* T2's run and not a re-implementation of it.

The filtering is linear, so the conditional structure is transformed coherently
rather than scrambled, and ``P(b)`` is re-measured on each arm's own corpus
(below) rather than assumed to carry over.

**Everything else is T2's**, imported rather than re-implemented: 2 122 / 1 061
non-overlapping windows, per-series centre and unit variance (A9), per-patch
demeaning before the matched filter (T1's convention), the same split
(``predictability.SPLIT_SEED``), the same model (hidden 32 / 2 layers / 4 heads,
SGD 1e-2, batch 64), the same 20 000 steps, the same 3 seeds, the same band grid,
the same readouts, the same fit gate with persistence and mean-predictor
baselines. ``t2.check_aperture`` asserts band-by-band that each arm's training
windows reproduce its own corpus through T1's estimator (``atol = 1e-9``).

Run one (variant, arm, seed) per process (they are CPU-bound):

``.venv/bin/python experiments/1_realdata/scripts/t2b.py --variant B --arm k32
--seed 0``

then read the tables back with ``--summarize``. ``--report`` prints the filter's
own diagnostics (corpus flatness before and after, ``P(b)`` per arm) without
training anything.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import predictability as pred
import t2
import torch

from fbias.realdata import load_monash

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = RUNS / "t2b"

# The one thing that changes between arms: the whitening exponent.
ALPHA = {"A": 0.0, "B": 1.0, "C": 0.5}
LABEL = {
    "A": "A: T2's corpus",
    "B": "B: flattened, alpha = 1",
    "C": "C: dose, alpha = 0.5",
}
# Flattening parameters, fixed before the run. See the module docstring.
K_REF = 64
HALF_B = 1.0
GAIN_AT = (1e-4, 1e-3, 1e-2, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


def corpus_series():
    """T1's five series: per-series centre and unit variance (contract A9)."""
    _, raw = load_monash(pred.DATASET)
    return [(v - v.mean()) / v.std() for _, _, v in raw]


def mean_spectrum(ser, n_grid):
    """Corpus mean power spectrum on a common frequency grid.

    Each series' whole-record periodogram is interpolated onto the grid of the
    longest record, so the five spectra can be averaged. Below the shortest
    record's own resolution (``1 / N``) the interpolation clamps, which is the
    honest reading -- the record cannot resolve those frequencies at all.
    """
    f = np.arange(n_grid // 2 + 1) / n_grid
    acc = np.zeros_like(f)
    for x in ser:
        n = len(x)
        acc += np.interp(
            f, np.fft.rfftfreq(n), np.abs(np.fft.rfft(x - x.mean())) ** 2 / n
        )
    return f, acc / len(ser)


def smoothed(y, df, half):
    """Local mean of ``y`` over a truncated window of half-width ``half`` in ``df``.

    The window is truncated at the ends of the record rather than padded with
    the edge value: the periodogram's extremes are single bins -- the record's
    coherent components land in one bin each -- so replicating an edge bin would
    import a spike into every window that touches it. Two of the five series
    have a Nyquist bin ~5 000x their local continuum (a component whose power
    share is 7e-5, so it matters to no band power, but it is 20 000x the local
    *density*), and replicating it inflated the top of the estimate by 6 700x.
    Averaging only over the frequencies the record actually resolves leaves the
    local mean unbiased.
    """
    half = max(int(round(half / df)), 1)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    i = np.arange(len(y))
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, len(y))
    return (cs[hi] - cs[lo]) / (hi - lo)


def equaliser(ser, alpha):
    """The flattening gain on the common frequency grid, ``S ** (-alpha / 2)``."""
    f, s = mean_spectrum(ser, max(len(x) for x in ser))
    return f, smoothed(s, f[1] - f[0], HALF_B / K_REF) ** (-alpha / 2.0)


def filtered(ser, f, g):
    """Apply the gain to each whole series and re-standardise it.

    Real, non-negative gain on the half-spectrum of a real series, so this is a
    single zero-phase linear filter (non-causal by construction: it is a whole
    record operation, not something a forecaster could do). The output is
    re-centred and scaled to unit variance over the whole series, which is T1's
    per-series normalisation; the filter's own scale therefore cancels.
    """
    out = []
    for x in ser:
        n = len(x)
        y = np.fft.irfft(
            np.fft.rfft(x - x.mean()) * np.interp(np.fft.rfftfreq(n), f, g), n=n
        )
        out.append((y - y.mean()) / y.std())
    return out


def arm_series(variant):
    """One arm's corpus as five series. ``alpha = 0`` is the identity."""
    ser = corpus_series()
    if ALPHA[variant] == 0.0:
        return ser
    f, g = equaliser(ser, ALPHA[variant])
    return filtered(ser, f, g)


def windows_from(ser, cfg):
    """T1's windows from a given set of series -- ``t2.raw_windows``' body.

    ``t2.raw_windows`` loads and normalises the corpus itself; this is the same
    construction applied to an arm's series. Arm A is asserted equal to it.
    """
    out = []
    for x in ser:
        n = len(x) // cfg.win
        out.append(x[: n * cfg.win].reshape(n, cfg.win))
    return np.concatenate(out)


def grid_flatness(power, at):
    """Max/min band power over the 32-band grid and over the probe bands."""
    p = np.asarray(power)
    return {
        "grid_max_min": float(p.max() / p.min()),
        "probe_max_min": float(p[at].max() / p[at].min()),
        "probe_max_min_no_nyquist": float(p[at[:-1]].max() / p[at[:-1]].min()),
    }


def run_arm(variant, arm, seed, steps, hidden, layers, write=True):
    """Everything one (variant, arm, seed) run reports -- ``t2.run_arm``'s shape.

    Identical to ``t2.run_arm`` except that the corpus, and therefore ``P(b)``,
    the power table, the leakage column and the A9 probe scale, come from this
    arm's own series.
    """
    out = OUT / f"{variant}_{arm}_s{seed}.json"
    if write and out.exists():
        print(f"{out.name} exists, skipping")
        return None
    cfg = t2.arm_cfg(arm)
    t0 = time.time()
    t_start = time.time()
    ser = arm_series(variant)
    windows = windows_from(ser, cfg)
    if variant == "A":
        assert np.array_equal(windows, t2.raw_windows(cfg)), (
            "arm A must be T2's corpus byte for byte; the identity filter is not "
            "the identity"
        )
    print(
        f"[{variant} {arm} seed {seed}] {len(windows)} windows of {cfg.win} "
        f"samples, ctx {cfg.ctx} / k {cfg.k}, horizon "
        f"{cfg.k * pred.STEP_MIN / 60:.0f} h, alpha {ALPHA[variant]}"
    )

    coeffs = pred.corpus(cfg, series=ser)
    t2.check_aperture(cfg, windows, coeffs)
    res, n_tr, n_ev = pred.run(cfg, coeffs, ("last", "own", "full"))
    leak = pred.in_band_fraction(cfg, series=ser)
    power = pred.band_power(coeffs)
    power_raw = pred.band_power(pred.corpus(cfg))
    leak_raw = pred.in_band_fraction(cfg)
    print(f"  T1's estimator: {n_tr}/{n_ev} windows, {time.time() - t0:.1f}s")

    n = len(windows)
    idx = np.random.default_rng(pred.SPLIT_SEED).permutation(n)
    held = windows[idx[n // 2 :]]
    rms = math.sqrt(float(np.mean(windows[idx[: n // 2]] ** 2)))
    raw = t2.raw_windows(cfg)
    rms_raw = math.sqrt(float(np.mean(raw[idx[: n // 2]] ** 2)))
    probe = t2.probe_windows(cfg, rms)
    at = [int(np.argmin(np.abs(cfg.bands - b))) for b in t2.BANDS]
    assert all(abs(cfg.bands[j] - b) < 1e-9 for j, b in zip(at, t2.BANDS, strict=True))
    evals = sorted({s for s in t2.EVAL_STEPS if s < steps} | {steps})

    t1 = time.time()
    model, curve, loss, n_train, _ = t2.train_one(
        seed,
        windows,
        cfg,
        torch.from_numpy(probe.astype(np.float32)),
        torch.from_numpy(held.astype(np.float32)),
        steps,
        hidden,
        layers,
        evals,
    )
    print(f"  {steps} steps in {time.time() - t1:.1f}s")

    record = {
        "tag": f"t2b-{variant}",
        "variant": variant,
        "alpha": ALPHA[variant],
        "arm": arm,
        "k": cfg.k,
        "ctx": cfg.ctx,
        "P": cfg.n_ctx,
        "window": cfg.win,
        "horizon_hours": cfg.k * pred.STEP_MIN / 60,
        "band_hours": [float(cfg.k / b * pred.STEP_MIN / 60) for b in t2.BANDS],
        "seed": seed,
        "steps": steps,
        "model": {"hidden": hidden, "layers": layers, "heads": t2.HEADS},
        "optim": {"name": "SGD", "lr": t2.LR, "batch": t2.BATCH, "threads": t2.THREADS},
        "n_windows": n,
        "n_train": n_train,
        "n_eval": n_ev,
        "split_seed": pred.SPLIT_SEED,
        "bands": list(t2.BANDS),
        "grid_index": at,
        "power": [float(power[j]) for j in at],
        "power_frac": [float(power[j] / power.sum()) for j in at],
        "power_raw": [float(power_raw[j]) for j in at],
        "power_frac_raw": [float(power_raw[j] / power_raw.sum()) for j in at],
        "grid_power": power.tolist(),
        "grid_power_raw": power_raw.tolist(),
        "P_last": [float(res["last"]["p"][j]) for j in at],
        "P_own": [float(res["own"]["p"][j]) for j in at],
        "P_full": [float(res["full"]["p"][j]) for j in at],
        "P_own_train": [float(res["own"]["p_train"][j]) for j in at],
        "in_band": [float(leak[j]) for j in at],
        "in_band_raw": [float(leak_raw[j]) for j in at],
        "corpus_rms": rms,
        "corpus_rms_raw": rms_raw,
        "probe_rms": float(np.sqrt(np.mean(probe**2))),
        "probe_power": t2.probe_band_power(cfg, probe),
        "filter": {"k_ref": K_REF, "half_b": HALF_B, "alpha": ALPHA[variant]},
        "flat_after": grid_flatness(power, at),
        "flat_before": grid_flatness(power_raw, at),
        "curve": curve,
        "wall": time.time() - t0,
        "wall_data": time.time() - t_start,
    }
    record.update(curve[-1])  # the final checkpoint's metrics, at the top level
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"  wrote {out} ({record['wall']:.1f}s total)")
    return record


def report():
    """The filter's own diagnostics: flatness and ``P(b)`` per arm, no training."""
    t0 = time.time()
    ser = corpus_series()
    rec = {
        "dataset": pred.DATASET,
        "method": (
            "g(f) = S(f)^(-alpha/2), S = the corpus's mean power spectrum "
            "(mean over the five whole-record periodograms on a common grid), "
            f"boxcar-smoothed to a half-width of {HALF_B} b-units at k_ref = "
            f"{K_REF} ({HALF_B / K_REF} cycles/sample, the matched filter's own "
            "mainlobe half-width). No floor and no ceiling."
        ),
        "k_ref": K_REF,
        "half_b": HALF_B,
        "alpha": ALPHA,
        "arms": {},
    }
    for variant in ALPHA:
        s = arm_series(variant)
        f, g = equaliser(ser, ALPHA[variant]) if ALPHA[variant] else (None, None)
        rec["arms"][variant] = {
            "alpha": ALPHA[variant],
            "gain": None
            if g is None
            else {f"{x:g}": float(g[int(np.argmin(np.abs(f - x)))]) for x in GAIN_AT},
            "series_std": [float(x.std()) for x in s],
            "rms": [float(np.sqrt(np.mean(x**2))) for x in s],
        }
        for arm in t2.ARMS:
            cfg = t2.arm_cfg(arm)
            windows = windows_from(s, cfg)
            if variant == "A":
                assert np.array_equal(windows, t2.raw_windows(cfg))
            coeffs = pred.corpus(cfg, series=s)
            power = pred.band_power(coeffs)
            leak = pred.in_band_fraction(cfg, series=s)
            res, n_tr, n_ev = pred.run(cfg, coeffs, ("last", "own", "full"))
            at = [int(np.argmin(np.abs(cfg.bands - b))) for b in t2.BANDS]
            idx = np.random.default_rng(pred.SPLIT_SEED).permutation(len(windows))
            rms = math.sqrt(float(np.mean(windows[idx[: len(windows) // 2]] ** 2)))
            rec["arms"][variant][arm] = {
                "n_windows": int(len(windows)),
                "window_rms": rms,
                "power": [float(power[j]) for j in at],
                "in_band": [float(leak[j]) for j in at],
                "band_hours": [float(cfg.k / b * pred.STEP_MIN / 60) for b in t2.BANDS],
                "P_own": [float(res["own"]["p"][j]) for j in at],
                "P_full": [float(res["full"]["p"][j]) for j in at],
                "P_last": [float(res["last"]["p"][j]) for j in at],
                **grid_flatness(power, at),
            }
            r = rec["arms"][variant][arm]
            print(
                f"[{variant}/{arm}] power max/min {r['probe_max_min']:.2f} "
                f"(grid {r['grid_max_min']:.2f}), in-band "
                f"[{min(r['in_band']):.3f}, {max(r['in_band']):.3f}], "
                f"P_own [{min(r['P_own']):.3f}, {max(r['P_own']):.3f}], "
                f"window rms {rms:.5f}  ({time.time() - t0:.0f}s)"
            )
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "flatten.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"\nwrote {out} ({time.time() - t0:.0f}s)")
    return rec


def load_runs():
    """Every ``<variant>_<arm>_s<seed>.json`` on disk, grouped by (variant, arm)."""
    groups = {}
    for path in sorted(OUT.glob("*_s*.json")):
        rec = json.loads(path.read_text())
        if "variant" not in rec or "seed" not in rec:
            continue
        groups.setdefault((rec["variant"], rec["arm"]), []).append(rec)
    for rows in groups.values():
        rows.sort(key=lambda r: r["seed"])
    return groups


def print_flatness(tag, rows):
    """Item 6: the corpus's spectral flatness before and after the filter."""
    d = rows[0]
    print(
        f"\n-- {tag}: corpus spectral flatness (max/min band power, T1's units)"
        f"\n   probe bands  before {d['flat_before']['probe_max_min']:8.2f}"
        f"  -> after {d['flat_after']['probe_max_min']:8.2f}"
        f"   (before, excluding the Nyquist band "
        f"{d['flat_before']['probe_max_min_no_nyquist']:.2f})"
        f"\n   32-band grid before {d['flat_before']['grid_max_min']:8.2f}"
        f"  -> after {d['flat_after']['grid_max_min']:8.2f}"
        f"\n   window rms   before {d['corpus_rms_raw']:.5f}"
        f"  -> after {d['corpus_rms']:.5f}   probe rms {d['probe_rms']:.5f} "
        f"(A9 ratio {d['probe_rms'] / d['corpus_rms']:.4f})"
    )
    return {
        "flat_before": d["flat_before"],
        "flat_after": d["flat_after"],
        "corpus_rms": d["corpus_rms"],
        "corpus_rms_raw": d["corpus_rms_raw"],
        "probe_rms": d["probe_rms"],
    }


def decision_table(record):
    """The one question: does flattening remove the decline?"""
    print(
        "\n=== the decision: r_probe's shape, per arm\n"
        "    (decline = r(b=1) / r(b=12), T2's headline statistic; max/min over "
        "the 8 probe bands and over the 7 non-Nyquist bands)"
    )
    print(
        f"   {'variant':>7} {'k':>4} {'decline':>8} {'max/min':>8} {'max/min-ny':>11} "
        f"{'spread_b':>9} {'seed sd':>8} {'spread/sd':>10} {'corr(log2b)':>12} "
        f"{'P max/min':>10} {'corpus max/min':>15}"
    )
    for key, d in sorted(record["arms"].items()):
        variant, arm = key.split("/")
        m = np.array(d["probe_mean"])
        p = np.array(d["P_own"])
        print(
            f"   {variant:>7} {arm:>4} {m[0] / m[6]:8.2f} {m.max() / m.min():8.2f} "
            f"{m[:-1].max() / m[:-1].min():11.2f} {d['shape']['spread_b']:9.4f} "
            f"{d['shape']['seed_sd']:8.4f} {d['shape']['spread_over_seed_sd']:10.2f} "
            f"{d['shape']['corr_log2b']:12.2f} {p.max() / p.min():10.2f} "
            f"{d['flat_after']['probe_max_min']:15.2f}"
        )
    print(
        "   (decline is r at b = 1 over r at b = 12, T2's headline statistic; "
        "'max/min-ny' drops b = 16,\n    the k32 sampling Nyquist and the probe's "
        "own degenerate bin; 'P max/min' and 'corpus max/min'\n    are over the 8 "
        "probe bands of that arm.)"
    )


def check_arm_a(record):
    """Arm A must reproduce T2's own numbers, field by field."""
    ref = json.loads((RUNS / "t2_summary.json").read_text())
    print("\n=== arm A against T2 (runs/t2_summary.json)")
    fields = (
        ("P_own", np.array),
        ("P_full", np.array),
        ("in_band", np.array),
        ("power", np.array),
        ("probe_mean", np.array),
        ("probe_sd", np.array),
        ("corpus_mean", np.array),
        ("oracle_probe_mean", np.array),
        ("band_var_explained", np.array),
        ("ve_last", np.array),
    )
    scalars = (
        ("gate", "ve_last"),
        ("gate", "persistence"),
        ("gate", "mean"),
        ("shape", "spread_b"),
        ("shape", "seed_sd"),
        ("shape", "corr_log2b"),
        ("relation", "mean_ratio_over_sqrtP"),
    )
    out, worst = {}, 0.0
    for arm in t2.ARMS:
        got, want = record["arms"][f"A/{arm}"], ref["arms"][arm]
        diffs = {}
        for key, cast in fields:
            diffs[key] = float(np.max(np.abs(cast(got[key]) - cast(want[key]))))
        for a, b in scalars:
            diffs[f"{a}.{b}"] = float(abs(got[a][b] - want[a][b]))
        out[arm] = diffs
        worst = max(worst, max(diffs.values()))
        print(
            f"   {arm}: max |arm A - T2| over "
            + ", ".join(
                f"{k} {v:.2e}"
                for k, v in sorted(diffs.items(), key=lambda x: -x[1])[:4]
            )
        )
    tol = 1e-9
    print(
        f"   worst difference {worst:.3e} -> "
        + ("REPRODUCES T2" if worst <= tol else "DOES NOT REPRODUCE T2 (stop)")
    )
    out["worst"] = worst
    out["reproduces"] = bool(worst <= tol)
    return out


def summarize():
    """Print the per-arm tables and pool them to ``runs/t2b/summary.json``."""
    groups = load_runs()
    record = {"arms": {}, "wall": {}}
    for (variant, arm), rows in sorted(groups.items()):
        tag = f"{LABEL[variant]} / {arm}"
        d = rows[0]
        print(
            f"\n=== {tag} | {len(rows)} seeds | k {d['k']} ctx {d['ctx']} "
            f"window {d['window']} | horizon {d['horizon_hours']:.0f} h | "
            f"steps {d['steps']} | hidden {d['model']['hidden']} / "
            f"{d['model']['layers']}L | {d['n_windows']} windows "
            f"({d['n_train']}/{d['n_eval']})"
        )
        gate = t2.print_gate(variant, arm, rows)
        tab = t2.print_bands(variant, arm, rows)
        shape = t2.print_shape(variant, arm, rows, tab)
        rel = t2.print_relation(variant, arm, rows, tab)
        flat = print_flatness(tag, rows)
        record["arms"][f"{variant}/{arm}"] = {
            "variant": variant,
            "alpha": d["alpha"],
            "arm": arm,
            "seeds": [r["seed"] for r in rows],
            "k": d["k"],
            "ctx": d["ctx"],
            "window": d["window"],
            "horizon_hours": d["horizon_hours"],
            "steps": d["steps"],
            "model": d["model"],
            "n_windows": d["n_windows"],
            "n_train": d["n_train"],
            "n_eval": d["n_eval"],
            "bands": d["bands"],
            "band_hours": d["band_hours"],
            "power": d["power"],
            "power_raw": d["power_raw"],
            "probe_power": d["probe_power"],
            "P_own": d["P_own"],
            "P_full": d["P_full"],
            "P_last": d["P_last"],
            "in_band": d["in_band"],
            "in_band_raw": d["in_band_raw"],
            "flat_before": d["flat_before"],
            "flat_after": d["flat_after"],
            "gate": gate,
            "shape": shape,
            "relation": rel,
            "probe_mean": tab["probe"].mean(axis=0).tolist(),
            "probe_sd": tab["probe"].std(axis=0).tolist(),
            "corpus_mean": tab["corpus"].mean(axis=0).tolist(),
            "oracle_probe_mean": tab["oracle_probe"].mean(axis=0).tolist(),
            "oracle_corpus_mean": tab["oracle_corpus"].mean(axis=0).tolist(),
            "band_var_explained": tab["band_ve"].mean(axis=0).tolist(),
            "ve_last": [r["var_explained_last"] for r in rows],
            **flat,
            "wall": [r["wall"] for r in rows],
            "wall_data": [r["wall_data"] for r in rows],
        }
        record["wall"][f"{variant}/{arm}"] = float(np.mean([r["wall"] for r in rows]))
    if record["arms"]:
        if any(k.startswith("A/") for k in record["arms"]):
            record["reproduce"] = check_arm_a(record)
        decision_table(record)
    record["wall_max"] = float(max(record["wall"].values()))
    record["wall_total"] = float(sum(record["wall"].values()))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "summary.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out}")
    return record


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", choices=tuple(ALPHA), default="B")
    p.add_argument("--arm", choices=tuple(t2.ARMS), default="k32")
    p.add_argument("--seed", type=int, nargs="+", default=list(t2.SEEDS))
    p.add_argument("--steps", type=int, default=t2.STEPS)
    p.add_argument("--hidden", type=int, default=t2.HIDDEN)
    p.add_argument("--layers", type=int, default=t2.LAYERS)
    p.add_argument("--threads", type=int, default=t2.THREADS)
    p.add_argument("--report", action="store_true", help="filter diagnostics only")
    p.add_argument("--summarize", action="store_true", help="read the tables back")
    p.add_argument("--no-write", action="store_true", help="smoke test, no artifact")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.report:
        report()
        return 0
    if args.summarize:
        summarize()
        return 0
    for seed in args.seed:
        run_arm(
            args.variant,
            args.arm,
            seed,
            args.steps,
            args.hidden,
            args.layers,
            write=not args.no_write,
        )
    return 0


if __name__ == "__main__":
    main()
