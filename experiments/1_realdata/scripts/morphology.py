r"""T1 step 3: four morphology diagnostics on the chosen corpus.

PLAN.md §6.3 expects the corpus's periodic content to be **externally clocked**
(phase known, non-diffusing), which is the opposite of an autonomous oscillator
with phase noise and also unlike a stationary narrowband Gaussian process. These
four cheap diagnostics settle which of the three the corpus is closer to, and
they are run on the phase-randomized surrogate as well -- the surrogate has the
same amplitude spectrum but by construction no clock and no amplitude modulation,
so it is the right null for each statistic.

Bands, in cycles per patch (``b = k / period_samples``, k = 32, half-hourly):
``CLOCK`` b in [0.542, 0.792] (period 20-30 h, contains the diurnal at b = 2/3),
``MID`` b in [2.875, 3.125], ``HIGH`` b in [7.875, 8.125]. The two upper bands
are a quarter of a cycle/patch wide at their centres, the clock band is a third
wider so that day-to-day modulation sidebands are inside it.

1. **Narrowband envelope CV** -- bandpass, Hilbert, envelope, ``std/mean``.
   Oscillator/PM ~ 0, stationary Rayleigh ~ 0.52, but the band width sets the
   null, so the surrogate column is the reference.
2. **Phase diffusion vs phase slips** (*decisive*) -- does the unwrapped phase
   variance grow linearly in lag, ``D(tau) ~ tau**gamma``, and are the large
   phase jumps the ones where the envelope fades? Stationary-Gaussian "diffusion"
   is nothing but fades, so its jumps sit at envelope minima; a phase-noise
   oscillator jumps without fading.
3. **Linewidth vs record length** -- the clock line's apparent width in records
   of length ``N/16 ... N``. A phase-locked line is 1/L wide at every L (it keeps
   narrowing); a stationary narrowband component converges to its own width.
4. **Is the periodic component phase locked?** -- fold on 24 h and on 7 d,
   report the variance explained and the residual's spectrum in the same band
   units.

Run from the repo root:
``.venv/bin/python experiments/1_realdata/scripts/morphology.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from fbias.realdata import load_monash

DATASET = "australian_electricity_demand_dataset"
K = 32  # patch, samples
STEP_MIN = 30.0
DAY = 48  # samples per 24 h
WEEK = 7 * DAY
BANDS = np.round(0.5 * np.arange(1, 33), 6)  # same grid as predictability.py

DIAG = {
    "clock": (0.542, 0.792, 2 / 3),  # period 20-30 h, carrier at the diurnal
    "mid": (2.875, 3.125, 3.0),
    "high": (7.875, 8.125, 8.0),
}

SURROGATE_SEED = 1
RUNS = Path(__file__).resolve().parents[1] / "runs"


def analytic_band(x, b_lo, b_hi):
    """Complex analytic signal of ``x`` restricted to ``b in [b_lo, b_hi]``."""
    spec = np.fft.fft(x - x.mean())
    bb = np.fft.fftfreq(len(x)) * K
    return 2 * np.fft.ifft(spec * ((bb >= b_lo) & (bb <= b_hi)))


def phase_randomize(x, rng):
    """Theiler-style surrogate: keep the amplitude spectrum, randomize the phases."""
    spec = np.fft.rfft(x - x.mean())
    phase = np.zeros(len(spec))
    phase[1:] = rng.uniform(0.0, 2 * np.pi, len(spec) - 1)
    if len(x) % 2 == 0:
        phase[-1] = 0.0
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phase), n=len(x))


def load_series(surrogate=False):
    """The corpus as a list of unit-variance float64 series."""
    _, raw = load_monash(DATASET)
    rng = np.random.default_rng(SURROGATE_SEED)
    out = []
    for _, _, v in raw:
        x = (v - v.mean()) / v.std()
        out.append(phase_randomize(x, rng) if surrogate else x)
    return out


def envelope_cv(x, b_lo, b_hi):
    """``std/mean`` of the band's envelope. Rayleigh (stationary) is 0.523."""
    env = np.abs(analytic_band(x, b_lo, b_hi))
    return float(env.std() / env.mean())


def phase_stats(x, b_lo, b_hi, b_c, lags=(1, 4, 16, 64, 256, 1024, 4096)):
    """Structure function, its long-lag exponent, and the slip/fade ratio.

    ``D(tau)`` is the mean squared unwrapped phase change over ``tau`` samples,
    measured on the carrier-demodulated band-limited phase. A diffusing phase
    has ``D ~ tau`` (exponent 1); a phase that is locked or stationary saturates
    (exponent 0). The exponent is fitted on the last three lags, which is where
    diffusion would still be growing.
    """
    t = np.arange(len(x))
    a = analytic_band(x, b_lo, b_hi)
    phase = np.unwrap(np.angle(a * np.exp(-2j * np.pi * b_c * t / K)))
    d = [float(np.mean((phase[tau:] - phase[:-tau]) ** 2)) for tau in lags]
    gamma = float(
        (np.log(d[-1]) - np.log(d[-3])) / (np.log(lags[-1]) - np.log(lags[-3]))
    )

    step = np.diff(phase)
    thr = np.quantile(np.abs(step), 0.99)
    jumps = np.flatnonzero(np.abs(step) > thr) + 1
    env = np.abs(a)
    fade = float(env[jumps].mean() / env.mean())
    return {
        "lags": list(lags),
        "D": d,
        "gamma_4_64": gamma,
        "slip_envelope_ratio": fade,
        "phase_std": float(phase.std()),
    }


def linewidth(x, b_c, fractions=(16, 8, 4, 2, 1)):
    """Apparent clock-line width (in cycles/patch) vs record length.

    The second moment of the mean periodogram inside ``+-0.05`` cycles/patch of
    the carrier, computed on ``len(x)/fraction`` sample segments. A phase-locked
    line is one resolution element wide at every length, so its width tracks
    ``1/L``; a stationary narrowband component converges to its own width.
    """
    out = {}
    for f in fractions:
        L = len(x) // f
        if L < 4096:
            continue
        seg = x[: L * (len(x) // L)].reshape(-1, L)
        spec = np.abs(np.fft.rfft(seg * np.hanning(L), axis=1)) ** 2
        bb = np.fft.rfftfreq(L) * K
        near = np.abs(bb - b_c) < 0.05
        w = spec[:, near].mean(axis=0)
        b = bb[near]
        w = w - w.min()
        out[f] = float(np.sum(w * (b - b_c) ** 2) / np.sum(w))
    return out


def fold_r2(x, period):
    """Variance explained by the mean profile folded over ``period`` samples."""
    n = len(x) // period
    prof = x[: n * period].reshape(n, period).mean(axis=0)
    return float(1 - np.var(x - np.resize(prof, len(x))) / np.var(x))


def residual_spectrum(x, period=WEEK):
    """Band powers of the clock-folded residual, in the same band units."""
    n = len(x) // period
    prof = x[: n * period].reshape(n, period).mean(axis=0)
    r = x - np.resize(prof, len(x))
    spec = np.abs(np.fft.rfft(r)) ** 2
    bb = np.fft.rfftfreq(len(r)) * K
    return np.array(
        [spec[(bb >= b - 0.25) & (bb < b + 0.25)].sum() / spec.sum() for b in BANDS]
    )


def main():
    t0 = time.time()
    real = load_series()
    surro = load_series(surrogate=True)
    print(f"{DATASET}: {len(real)} series loaded in {time.time() - t0:.1f}s")

    record = {"dataset": DATASET, "step_minutes": STEP_MIN, "bands": DIAG}

    print("\n=== 1. narrowband envelope CV (Rayleigh reference 0.523) ===")
    print(f"  {'band':>6} {'real':>18} {'surrogate':>18}")
    cvs = {}
    for name, (lo, hi, _) in DIAG.items():
        r = np.array([envelope_cv(x, lo, hi) for x in real])
        s = np.array([envelope_cv(x, lo, hi) for x in surro])
        cvs[name] = {"real": r.tolist(), "surrogate": s.tolist()}
        print(
            f"  {name:>6} {np.median(r):8.3f} [{r.min():.3f},{r.max():.3f}]"
            f" {np.median(s):8.3f} [{s.min():.3f},{s.max():.3f}]"
        )

    print("\n=== 2. phase diffusion vs phase slips ===")
    print("  D(tau) = mean squared unwrapped phase change; gamma = log-log slope")
    print("  on lags 256..4096: 1 = diffusing, 0 = saturated")
    print("  slip_envelope_ratio ~1 => jumps without fades (phase noise); <<1 => fades")
    print(f"  {'band':>6} {'':>8} {'gamma long':>11} {'phase sd':>9} {'slip/env':>9}")
    phase = {}
    for name, (lo, hi, b_c) in DIAG.items():
        for label, series in (("real", real), ("surr", surro)):
            st = [phase_stats(x, lo, hi, b_c) for x in series]
            phase[f"{name}_{label}"] = st
            g = np.median([s["gamma_4_64"] for s in st])
            p = np.median([s["phase_std"] for s in st])
            f = np.median([s["slip_envelope_ratio"] for s in st])
            print(f"  {name:>6} {label:>8} {g:12.2f} {p:9.2f} {f:9.3f}")
    d_lags = phase["clock_real"][0]["lags"]
    d_real = np.median([s["D"] for s in phase["clock_real"]], axis=0)
    d_surr = np.median([s["D"] for s in phase["clock_surr"]], axis=0)
    print(f"  clock D(tau): lags {d_lags}")
    print(f"    real {np.round(d_real, 2)}")
    print(f"    surr {np.round(d_surr, 2)}")

    print("\n=== 3. clock line width vs record length (b units) ===")
    lw_real = [linewidth(x, 2 / 3) for x in real]
    lw_surr = [linewidth(x, 2 / 3) for x in surro]
    keys = sorted(lw_real[0], key=int, reverse=True)
    print(f"  {'N':>8} {'real':>10} {'surrogate':>10} {'real xN':>9} {'surr xN':>9}")
    for k in keys:
        L = 230736 // k
        r = np.median([d[k] for d in lw_real])
        s = np.median([d[k] for d in lw_surr])
        print(f"  {L:>8} {r:10.2e} {s:10.2e} {r * L:9.3f} {s * L:9.3f}")

    print("\n=== 4. is the periodic component phase locked? ===")
    print(f"  {'fold':>8} {'real var expl':>22} {'surrogate':>22}")
    fold = {}
    for label, period in (("24 h", DAY), ("7 d", WEEK)):
        r = np.array([fold_r2(x, period) for x in real])
        s = np.array([fold_r2(x, period) for x in surro])
        fold[label] = {"real": r.tolist(), "surrogate": s.tolist()}
        print(
            f"  {label:>8} {np.median(r):9.4f} [{r.min():.3f},{r.max():.3f}]"
            f" {np.median(s):9.4f} [{s.min():.3f},{s.max():.3f}]"
        )
    print(
        "\n  note: the 24 h fold's variance explained is invariant under phase\n"
        "  randomization (it depends only on the magnitudes of the 24 h harmonics),\n"
        "  so the real/surrogate columns must agree there and it cannot separate\n"
        "  clock-locked from stationary-line content. The 7 d fold's can, and does."
    )
    resid = np.median([residual_spectrum(x) for x in real], axis=0)
    resid_s = np.median([residual_spectrum(x) for x in surro], axis=0)
    print("\n  mean share of residual power per band (7 d fold removed), real | surr")
    for b, r, s in zip(BANDS, resid, resid_s):
        print(f"    b={b:5.1f} T={K / b * STEP_MIN / 60:5.2f}h  {r:8.5f} | {s:8.5f}")

    record.update(
        {
            "envelope_cv": cvs,
            "phase": phase,
            "linewidth": {"real": lw_real, "surrogate": lw_surr},
            "fold_r2": fold,
            "residual_band_frac": {
                "real": resid.tolist(),
                "surrogate": resid_s.tolist(),
            },
            "wall": time.time() - t0,
        }
    )
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "morphology.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out}\ntotal wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
