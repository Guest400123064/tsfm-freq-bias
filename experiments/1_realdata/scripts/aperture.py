r"""T1 follow-up: is the near-flatness of ``P(b)`` an aperture artifact?

T1 measured ``P(b)`` with a ``k = 32`` matched filter and found it weakly
varying and floor-dominated: ``P_own`` falls ~1.25x across ``b in [0.5, 16]``
while the band power falls 463x. The reported cause is the aperture: a 32-sample
patch has a mainlobe one ``b``-unit wide, so at ``b = 8`` only 10% of the
measured "band power" is in the band and the rest is leakage from the clock
lines and the red continuum. If the flatness is that leakage, a longer patch --
which integrates more cycles of a given physical band -- should resolve it away.

This driver re-runs T1's estimator **unchanged** (imported from
``predictability.py``; same corpus, same construction, same ridge, same seeds)
and varies only ``k``:

  ``k32``   ctx  512  k  32   P = 16   (T1, reproduction)
  ``k64``   ctx 1024  k  64   P = 16
  ``k128``  ctx 1024  k 128   P =  8

The matched filter's response depends only on ``b - b'``, so everything the
estimator reads is a function of the pair ``(k, b)`` -- the grid only decides
which pairs are looked at. A physical band of period ``T`` hours sits at
``b = k / (2 T)``, so holding ``b`` fixed while growing ``k`` moves the physical
band, and only growing ``b`` with ``k`` holds the band. Both readings are
reported:

``grid = "t1"``    T1's ``b`` values, so the arms line up column by column in
                   ``b`` -- the band is not held (at ``b = 8``, ``k = 32`` reads
                   2 h and ``k = 128`` reads 8 h). The literal arms of the brief.
``grid = "phys"``  ``b`` scaled by ``k / 32``, so the arms line up in physical
                   period (T1's 1 h .. 32 h). This is the aperture test: the same
                   band sits at a larger ``b``, hence with more cycles per patch.

The two grids coincide at ``k = 32``, so that arm is run once.

Run from the repo root:
``.venv/bin/python experiments/1_realdata/scripts/aperture.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import predictability as pred

RUNS = Path(__file__).resolve().parents[1] / "runs"

ARMS = (
    (32, 512, "t1"),
    (64, 1024, "t1"),
    (128, 1024, "t1"),
    (64, 1024, "phys"),
    (128, 1024, "phys"),
)
# The two readings: the same b at every k, and the same physical period.
FIXB = ("k32/t1", "k64/t1", "k128/t1")
CANON = ("k32/t1", "k64/phys", "k128/phys")

# The bands the brief asks about, in T1's units (k = 32): 4 h, 2 h and 1 h.
SOND_B = (4.0, 8.0, 16.0)
# T1's 32 physical bands, in hours, one per row of its table.
T1_PERIODS = tuple(16.0 / b for b in pred.BANDS)


def measure(cfg):
    """T1's readouts for one arm, all through the same estimator."""
    name = f"k{cfg.k}/{cfg.grid}"
    t0 = time.time()
    real = pred.corpus(cfg)
    real_p, n_tr, n_ev = pred.run(cfg, real, ("last", "own", "full"))
    rep = pred.run(cfg, real, ("own",), seed=pred.SPLIT_SEED + 1)[0]
    surro = pred.run(
        cfg, pred.corpus(cfg, surrogate_seed=pred.SURROGATE_SEED), ("own",)
    )[0]
    declk = pred.run(cfg, pred.corpus(cfg, declocked=True), ("own",))[0]
    noise = pred.run(cfg, pred.corpus(cfg, series=pred.white_noise()), ("own",))[0]
    over = pred.run(
        cfg, pred.corpus(cfg, series=pred.white_noise(), prefilter=0.25), ("own",)
    )[0]
    leak = pred.in_band_fraction(cfg)
    leak_wn = pred.in_band_fraction(cfg, pred.white_noise())
    return {
        "name": name,
        "k": cfg.k,
        "ctx": cfg.ctx,
        "grid": cfg.grid,
        "P": cfg.n_ctx,
        "window": cfg.win,
        "bands": cfg.bands,
        "period_hours": cfg.periods_h,
        "n_windows": int(len(real)),
        "n_train": n_tr,
        "n_eval": n_ev,
        "power": pred.band_power(real),
        "P_last": np.array(real_p["last"]["p"]),
        "P_own": np.array(real_p["own"]["p"]),
        "P_full": np.array(real_p["full"]["p"]),
        "P_own_train": np.array(real_p["own"]["p_train"]),
        "P_own_split2": np.array(rep["own"]["p"]),
        "S_own": np.array(surro["own"]["p"]),
        "D_own": np.array(declk["own"]["p"]),
        "null_own": np.array(noise["own"]["p"]),
        "over_own": np.array(over["own"]["p"]),
        "in_band": leak,
        "in_band_white": leak_wn,
        "wall": time.time() - t0,
    }


def at_b(a, b):
    """Index of band ``b`` -- exact whenever ``b`` is in the grid."""
    return int(np.argmin(np.abs(a["bands"] - b)))


def at_period(a, hours):
    """Index of the band whose physical period is closest to ``hours``."""
    return int(np.argmin(np.abs(a["period_hours"] - hours)))


def matched_idx(a):
    """Indices reproducing T1's 32 physical periods, or ``None``.

    Only an arm whose grid reaches the sampling Nyquist can: the ``t1`` grid
    stops at ``b = 16``, which at ``k = 128`` is the 4 h band, so that arm has
    no 1 h .. 4 h rows.
    """
    idx = [at_period(a, t) for t in T1_PERIODS]
    exact = all(
        abs(a["period_hours"][j] - t) < 0.01 * t for j, t in zip(idx, T1_PERIODS)
    )
    return idx if exact else None


def span(v):
    """Max-min and max/min of an array."""
    return float(v.max() - v.min()), float(v.max() / v.min())


def row_table(a):
    """Per-band table for one arm: everything the report reads."""
    head = (
        f"{'b':>6} {'T(h)':>6} {'power':>10} {'P_last':>7} {'P_own':>7} "
        f"{'P_full':>7} {'S_own':>7} {'P-S':>7} {'D_own':>7} {'in-band':>8} {'wn':>6}"
    )
    print(
        f"\narm {a['name']} ctx={a['ctx']} (P={a['P']}, {a['n_windows']} windows, "
        f"{a['n_train']}/{a['n_eval']} train/eval, {a['wall']:.0f}s)\n  " + head
    )
    print("  " + "-" * len(head))
    for j, b in enumerate(a["bands"]):
        print(
            f"  {b:6.1f} {a['period_hours'][j]:6.2f} {a['power'][j]:10.3e} "
            f"{a['P_last'][j]:7.3f} {a['P_own'][j]:7.3f} {a['P_full'][j]:7.3f} "
            f"{a['S_own'][j]:7.3f} {a['P_own'][j] - a['S_own'][j]:7.3f} "
            f"{a['D_own'][j]:7.3f} {a['in_band'][j]:8.4f} "
            f"{a['in_band_white'][j]:6.3f}"
        )
    print(
        f"  null {a['null_own'].mean():+.3f} (range "
        f"[{a['null_own'].min():+.3f}, {a['null_own'].max():+.3f}])  "
        f"oversampled null {a['over_own'].mean():+.3f}  "
        f"in-band mean {a['in_band'].mean():.3f} "
        f"(white {a['in_band_white'].mean():.3f})"
    )


def summary(a):
    """The four items the brief asks for, as plain numbers."""
    own = a["P_own"]
    jb = [at_b(a, b) for b in SOND_B]
    rng, ratio = span(own)
    out = {
        "name": a["name"],
        "k": a["k"],
        "ctx": a["ctx"],
        "grid": a["grid"],
        "P": a["P"],
        "window": a["window"],
        "n_windows": a["n_windows"],
        "n_train": a["n_train"],
        "n_eval": a["n_eval"],
        "power_ratio": float(a["power"].max() / a["power"].min()),
        "P_own_min": float(own.min()),
        "P_own_max": float(own.max()),
        "P_own_range": rng,
        "P_own_ratio": ratio,
        "P_own_mean": float(own.mean()),
        "P_full_mean": float(a["P_full"].mean()),
        "P_own_train_mean": float(a["P_own_train"].mean()),
        "P_own_repl_rms": float(np.sqrt(np.mean((own - a["P_own_split2"]) ** 2))),
        "P_own_repl_max": float(np.max(np.abs(own - a["P_own_split2"]))),
        "P_minus_S_mean": float((own - a["S_own"]).mean()),
        "in_band_mean": float(a["in_band"].mean()),
        "in_band_white_mean": float(a["in_band_white"].mean()),
        "null_mean": float(a["null_own"].mean()),
        "null_min": float(a["null_own"].min()),
        "null_max": float(a["null_own"].max()),
        "oversampled_null_mean": float(a["over_own"].mean()),
        "period_at_b": [float(a["period_hours"][j]) for j in jb],
        "in_band_at_b": [float(a["in_band"][j]) for j in jb],
        "P_own_at_b": [float(own[j]) for j in jb],
        "P_minus_S_at_b": [float((own - a["S_own"])[j]) for j in jb],
    }
    m = matched_idx(a)
    if m is not None:
        rng, ratio = span(own[m])
        out["matched_P_own_range"] = rng
        out["matched_P_own_ratio"] = ratio
        out["matched_P_own_mean"] = float(own[m].mean())
        out["matched_in_band_mean"] = float(a["in_band"][m].mean())
        out["matched_in_band"] = a["in_band"][m].tolist()
        out["matched_P_own"] = own[m].tolist()
        out["matched_P_minus_S"] = (own[m] - a["S_own"][m]).tolist()
        out["matched_P_last"] = a["P_last"][m].tolist()
        out["matched_S_own"] = a["S_own"][m].tolist()
        out["matched_D_own"] = a["D_own"][m].tolist()
        out["matched_power"] = a["power"][m].tolist()
        out["matched_bands"] = a["bands"][m].tolist()
    return out


def main():
    t0 = time.time()
    arms = {}
    for k, ctx, grid in ARMS:
        a = measure(pred.config(k, ctx, grid))
        arms[a["name"]] = a
        row_table(a)

    s = {nm: summary(a) for nm, a in arms.items()}
    head = (
        f"{'arm':>8} {'n_win':>6} {'null':>7} {'ovsamp':>7} {'in-band':>8} "
        f"{'range':>7} {'max/min':>8} {'matched range':>14} {'ratio':>6} "
        f"{'in-band':>8} {'repl rms':>9} {'repl max':>9} {'P-S mean':>9}"
    )
    print(
        "\n\nper arm. range/max-min over the arm's own grid; 'matched' over T1's "
        f"32 physical bands; repl = split-seed replication\n{head}"
    )
    print("-" * len(head))
    for r in s.values():
        if "matched_P_own_range" in r:
            mat = (
                f"{r['matched_P_own_range']:14.3f} {r['matched_P_own_ratio']:6.3f} "
                f"{r['matched_in_band_mean']:8.3f}"
            )
        else:
            mat = " " * 30
        print(
            f"  {r['name']:>6} {r['n_windows']:6d} {r['null_mean']:+7.3f} "
            f"{r['oversampled_null_mean']:7.3f} {r['in_band_mean']:8.3f} "
            f"{r['P_own_range']:7.3f} {r['P_own_ratio']:8.3f} {mat} "
            f"{r['P_own_repl_rms']:9.3f} {r['P_own_repl_max']:9.3f} "
            f"{r['P_minus_S_mean']:+9.3f}"
        )

    for title, key in (
        ("in-band, by b (the brief's bands)", "in_band_at_b"),
        ("P_own, by b", "P_own_at_b"),
        ("P_own - S_own, by b", "P_minus_S_at_b"),
    ):
        print(
            f"\n{title}\n{'arm':>8} "
            + " ".join(f"{f'b={b:g}':>9}" for b in SOND_B)
            + "   physical period, same order"
        )
        for nm in FIXB:
            r = s[nm]
            print(
                f"  {nm:>6} "
                + " ".join(f"{v:9.4f}" for v in r[key])
                + "   "
                + "/".join(f"{p:g}h" for p in r["period_at_b"])
            )

    def matched_block(title, key):
        print(f"\n{title}\n  {'T(h)':>6} " + " ".join(f"{nm:>10}" for nm in CANON))
        print("  " + "-" * (7 + 11 * len(CANON)))
        for i, t in enumerate(T1_PERIODS):
            print(f"  {t:6.2f} " + " ".join(f"{s[nm][key][i]:10.4f}" for nm in CANON))

    matched_block(
        "in-band, at T1's physical periods (the aperture test)", "matched_in_band"
    )
    matched_block("P_own, at T1's physical periods", "matched_P_own")
    matched_block("P_own - S_own, at T1's physical periods", "matched_P_minus_S")
    matched_block("D_own (de-clocked), at T1's physical periods", "matched_D_own")

    record = {
        "dataset": pred.DATASET,
        "resolution_minutes": pred.STEP_MIN,
        "bands_asked": list(SOND_B),
        "periods_asked_hours": [16.0 / b for b in SOND_B],
        "t1_periods_hours": list(T1_PERIODS),
        "arms": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
            for a in arms.values()
        ],
        "summary": s,
        "wall": time.time() - t0,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "aperture.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out}\ntotal wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
