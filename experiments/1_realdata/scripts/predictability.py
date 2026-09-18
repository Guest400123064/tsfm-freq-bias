r"""T1 step 2: the per-band conditional predictability spectrum ``P(b)``.

The explaining variable of the whole claim (PLAN.md §6.1): for a given corpus,
how much of the *next* patch's band-``b`` content is determined by the observed
context? A next-patch model is predicted to retain ``sqrt(P(b))`` of the
amplitude of a clean probe at band ``b``, so this curve estimates the claim's
independent variable without training anything.

Corpus: ``australian_electricity_demand_dataset`` -- see ``../README.md`` for the
survey of the four candidates and the choice. Half-hourly, so ``b = k /
period_samples``: at T1's ``k = 32`` the 24 h clock sits at ``b = 32/48 = 0.667``,
the 7 d clock at ``b = 32/336 = 0.095``.

Estimator, per band ``b``:

1. window the corpus into ``ctx + k``-sample windows (``ctx / k`` context
   patches plus the target patch);
2. matched-filter every patch at band ``b``, with the patch demeaned first, ->
   one complex coefficient per (window, patch, band);
3. ridge-regress the target patch's coefficient on the context patches';
   fitted on a train half, scored on a held-out half, lambda chosen inside the
   train half;
4. ``P(b) = 1 - residual_var / var(target band)``.

Three feature sets, run on the real corpus, a phase-randomized surrogate of it
(:func:`phase_randomize`, Theiler 1992 style), the de-clocked real corpus and a
de-clocked surrogate:

  ``last``  the newest context patch only -- the patch-local reference
  ``own``   all context patches at band ``b``
  ``full``  all context patches at all bands

``P_real - P_surrogate`` is the part that is genuinely conditional structure
rather than what the linear spectrum already implies.

Three checks are printed with the tables because they decide how the numbers may
be read:

- a **white-noise control** (the same pipeline on 5 distinct iid series) must
  read ``P ~ 0`` at every band -- it does, so the ridge is not doing anything
  degenerate;
- the **in-band fraction** of each band's patch power (patch DFT power vs the
  power of the same series ideally band-limited to ``|b' - b| < 0.5``) -- on this
  corpus the high bands are mostly leakage from the clock lines, and this is the
  column that says which bands are quotable;
- an **oversampling control**: if the series is pre-filtered to 0.5-wide bands
  before the patch coefficients are taken, the patch coefficients in a window
  span only ~``ctx / k + 1`` complex degrees of freedom, so the regression
  predicts the target from the context *by construction* and white noise reads
  ``P = 1``. That is why the bands here are taken from the raw patches and the
  leakage is reported rather than filtered away.

``--k`` (with ``--ctx`` and ``--grid``) re-runs the identical estimator on a
different patch size, which is the aperture lever of T1's follow-up: the matched
filter's response depends only on ``b - b'``, so at a fixed ``b`` grid a longer
patch changes nothing about the aperture and only rescales the corpus's physical
frequency axis, while rescaling the grid to T1's physical periods (``grid =
"phys"``) is what actually narrows the relative passband around a given band.
``scripts/aperture.py`` drives those arms.

Run from the repo root:
``.venv/bin/python experiments/1_realdata/scripts/predictability.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fbias.realdata import load_monash

DATASET = "australian_electricity_demand_dataset"
STEP_MIN = 30.0  # half-hourly
WEEK = 7 * 24 * 60 // int(STEP_MIN)  # 336 samples

# T1's arm: 32 bands, 0.5 cycles/patch apart, up to the per-patch Nyquist
# k / 2 = 16. 2b is an integer at every centre, so the matched filter is exact
# there (probes.py).
K, CTX = 32, 512
BANDS = np.round(0.5 * np.arange(1, 33), 6)
HALF = 0.5  # half-width of a patch's own frequency resolution, in b units

SPLIT_SEED = 0
SURROGATE_SEED = 1
LAMBDAS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

RUNS = Path(__file__).resolve().parents[1] / "runs"


@dataclass(frozen=True)
class Cfg:
    """One aperture arm: patch size, context length, and the band grid they imply."""

    k: int
    ctx: int
    bands: np.ndarray
    grid: str

    @property
    def n_ctx(self):
        """Context patches per window."""
        return self.ctx // self.k

    @property
    def n_patch(self):
        """Patches per window, the last one being the target."""
        return self.n_ctx + 1

    @property
    def win(self):
        """Window length in samples."""
        return self.ctx + self.k

    @property
    def periods_h(self):
        """Period of each band in hours, ``T = k / b * 0.5 h``."""
        return self.k / self.bands * STEP_MIN / 60


def config(k=K, ctx=None, grid="t1"):
    """Build one arm's :class:`Cfg`, with ``ctx`` defaulting to 16 patches."""
    bands = np.round(BANDS * k / K, 6) if grid == "phys" else BANDS
    return Cfg(k=k, ctx=ctx if ctx is not None else 16 * k, bands=bands, grid=grid)


def matched(patches, b):
    """Matched filter of ``(..., k)`` patches at band ``b`` -> complex ``(...)``.

    ``b`` may be an array, in which case the result gains a trailing band axis.
    Equals the patch's DFT bin scaled by ``2/k`` when ``b`` is an integer.
    """
    k = patches.shape[-1]
    t = np.arange(k)
    w = (2 / k) * np.exp(-2j * np.pi * np.outer(t, np.atleast_1d(b)) / k)
    out = patches @ w
    return out if np.ndim(b) else out[..., 0]


def windows_of(x, cfg):
    """Non-overlapping ``ctx + k`` windows of ``x`` as ``(n, n_patch, k)``, demeaned."""
    n = len(x) // cfg.win
    patches = x[: n * cfg.win].reshape(n, cfg.n_patch, cfg.k)
    return patches - patches.mean(axis=-1, keepdims=True)


def band_limit(x, b, k, half=HALF):
    """Ideal whole-record band-pass of ``x`` to ``|b' - b| < half``.

    Used only for the in-band fraction and the oversampling control; it is
    non-causal, an analysis reference rather than anything a model could do.
    """
    spec = np.fft.rfft(x)
    bb = np.fft.rfftfreq(len(x)) * k
    spec[np.abs(bb - b) >= half] = 0.0
    return np.fft.irfft(spec, n=len(x))


def phase_randomize(x, rng):
    """Theiler-style surrogate: keep the amplitude spectrum, randomize the phases."""
    spec = np.fft.rfft(x - x.mean())
    phase = np.zeros(len(spec))
    phase[1:] = rng.uniform(0.0, 2 * np.pi, len(spec) - 1)
    if len(x) % 2 == 0:
        phase[-1] = 0.0
    return np.fft.irfft(np.abs(spec) * np.exp(1j * phase), n=len(x))


def declock(x, period=WEEK):
    """Subtract the per-series mean profile folded over one clock ``period``.

    ``period = 336`` samples removes the 24 h profile, the 7 d profile and their
    interaction in one step (PLAN.md §6.2: "after phase-folding out clock-locked
    components").
    """
    full = x[: len(x) // period * period].reshape(-1, period).mean(axis=0)
    return x - np.resize(full, len(x))


def corpus(cfg, series=None, surrogate_seed=None, declocked=False, prefilter=None):
    """Band coefficients of the pooled corpus, ``(n_windows, n_patch, n_bands)``.

    Each series is centred and scaled to unit variance over its whole length and
    then cut into non-overlapping ``ctx + k`` windows, so every series
    contributes the same number of windows and the pool is not dominated by the
    longest one. ``prefilter`` (a half-width in b units) band-limits each series
    before the patch coefficients are taken -- only the oversampling control
    uses it.
    """
    if series is None:
        _, raw = load_monash(DATASET)
        series = [v for _, _, v in raw]
    rng = np.random.default_rng(surrogate_seed) if surrogate_seed is not None else None
    per_band = [[] for _ in cfg.bands]
    for v in series:
        x = (v - v.mean()) / v.std()
        if declocked:
            x = declock(x)
        if rng is not None:
            x = phase_randomize(x, rng)
        for j, b in enumerate(cfg.bands):
            signal = x if prefilter is None else band_limit(x, b, cfg.k, prefilter)
            per_band[j].append(matched(windows_of(signal, cfg), b))
    return np.stack([np.concatenate(rows) for rows in per_band], axis=-1)


def in_band_fraction(cfg, series=None):
    """Per band, ideally in-band power / power of the raw patch coefficients.

    The matched filter of a ``k``-sample patch has a mainlobe of ``+- 1`` cycles
    per patch, so ``band_limit(x, b, k, 0.5)`` is the most generous reading of
    what band ``b``'s own content is. Anything else in the patch coefficient is
    the aperture's leakage from other frequencies.
    """
    if series is None:
        _, raw = load_monash(DATASET)
        series = [v for _, _, v in raw]
    total = np.zeros(len(cfg.bands))
    inside = np.zeros(len(cfg.bands))
    for v in series:
        x = (v - v.mean()) / v.std()
        patches = windows_of(x, cfg)
        for j, b in enumerate(cfg.bands):
            total[j] += np.mean(np.abs(matched(patches, b)) ** 2)
            inside[j] += np.mean(
                np.abs(matched(windows_of(band_limit(x, b, cfg.k), cfg), b)) ** 2
            )
    return inside / total


def features(coeffs, variant, band, cfg):
    """Context features for one band, ``(n_windows, d)`` complex."""
    ctx = coeffs[:, : cfg.n_ctx]
    if variant == "last":
        return ctx[:, -1:, band]
    if variant == "own":
        return ctx[:, :, band]
    return ctx.reshape(len(coeffs), -1)  # "full"


def p_band(f_tr, z_tr, f_ev, z_ev, lam):
    """Ridge fit on train, ``P = 1 - residual_var / var(target)`` on eval."""
    mu, m = f_tr.mean(axis=0), z_tr.mean()
    fc, zc = f_tr - mu, z_tr - m
    gram = fc.conj().T @ fc
    d = gram.shape[0]
    gram.flat[:: d + 1] += lam * np.trace(gram).real / d
    w = np.linalg.solve(gram, fc.conj().T @ zc)
    pred = f_ev @ w + (m - mu @ w)
    var = np.mean(np.abs(z_ev - z_ev.mean()) ** 2)
    return float(1 - np.mean(np.abs(pred - z_ev) ** 2) / var)


def fit_band(cfg, coeffs, variant, band, tr, ev):
    """``P`` on the held-out half, in-sample ``P`` on train, and the chosen lambda."""
    f = features(coeffs, variant, band, cfg)
    z = coeffs[:, -1, band]
    order = np.random.default_rng(SPLIT_SEED).permutation(len(tr))
    a, b = tr[order[: len(tr) // 2]], tr[order[len(tr) // 2 :]]
    lam = max((p_band(f[a], z[a], f[b], z[b], t), t) for t in LAMBDAS)[1]
    return (
        p_band(f[tr], z[tr], f[ev], z[ev], lam),
        p_band(f[tr], z[tr], f[tr], z[tr], lam),
        lam,
    )


def run(cfg, coeffs, variants, seed=SPLIT_SEED):
    """P(b) per variant. Windows do not overlap, so a random window split cannot
    share a sample between train and eval. ``seed`` moves the split, which is the
    cheap replication check on a thin arm."""
    n = len(coeffs)
    idx = np.random.default_rng(seed).permutation(n)
    tr, ev = idx[: n // 2], idx[n // 2 :]
    n_bands = len(cfg.bands)
    out = {}
    for variant in variants:
        rows = [fit_band(cfg, coeffs, variant, j, tr, ev) for j in range(n_bands)]
        out[variant] = {
            "p": [r[0] for r in rows],
            "p_train": [r[1] for r in rows],
            "lambda": [r[2] for r in rows],
        }
    return out, len(tr), len(ev)


def band_power(coeffs):
    """Mean band power ``|Z|^2 / 2`` over every window and patch."""
    return np.mean(np.abs(coeffs) ** 2, axis=(0, 1)) / 2


def white_noise(length=230736, n=5, seed=3):
    """``n`` distinct iid Gaussian series, the estimator's null corpus."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(length) for _ in range(n)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--k", type=int, default=K, help=f"patch size in samples [{K}]")
    p.add_argument("--ctx", type=int, default=None, help="context in samples [16 * k]")
    p.add_argument(
        "--grid",
        choices=("t1", "phys"),
        default="t1",
        help="band grid: T1's b values, or their image at T1's physical periods",
    )
    p.add_argument("--out", type=Path, default=None, help="output JSON path")
    return p.parse_args(argv)


def out_path(cfg):
    if cfg.k == K and cfg.grid == "t1":
        return RUNS / "predictability.json"
    return RUNS / f"predictability_k{cfg.k}_{cfg.grid}.json"


def main(argv=None):
    args = parse_args(argv)
    cfg = config(args.k, args.ctx, args.grid)
    out = args.out or out_path(cfg)
    t0 = time.time()
    print(
        f"corpus {DATASET}: {STEP_MIN:.0f}-min samples, b = {cfg.k} / period_samples"
        f"  [k = {cfg.k}, ctx = {cfg.ctx}, grid = {cfg.grid}]"
    )

    real = corpus(cfg)
    print(f"corpus {real.shape} in {time.time() - t0:.1f}s")

    t1 = time.time()
    surro = corpus(cfg, surrogate_seed=SURROGATE_SEED)
    declk = corpus(cfg, declocked=True)
    declk_surro = corpus(cfg, declocked=True, surrogate_seed=SURROGATE_SEED + 1)
    noise = corpus(cfg, series=white_noise())
    print(f"surrogates, de-clocked corpus and controls in {time.time() - t1:.1f}s")

    t2 = time.time()
    res_real, n_tr, n_ev = run(cfg, real, ("last", "own", "full"))
    res_surro, _, _ = run(cfg, surro, ("own", "full"))
    res_declk, _, _ = run(cfg, declk, ("last", "own", "full"))
    res_declk_s, _, _ = run(cfg, declk_surro, ("own", "full"))
    res_noise, _, _ = run(cfg, noise, ("own",))
    print(f"ridge fits in {time.time() - t2:.1f}s ({n_tr} train / {n_ev} eval windows)")

    t3 = time.time()
    leak = in_band_fraction(cfg)
    leak_wn = in_band_fraction(cfg, white_noise())
    over = run(cfg, corpus(cfg, series=white_noise(), prefilter=0.25), ("own",))[0]
    print(f"contamination diagnostics in {time.time() - t3:.1f}s")

    power = band_power(real)
    frac = power / power.sum()
    periods = cfg.periods_h

    def table(title, head, row):
        print(f"\n{title}\n  " + head)
        print("  " + "-" * len(head))
        for j, b in enumerate(cfg.bands):
            print(f"  {b:5.1f} {periods[j]:6.2f} " + row(j))

    table(
        "A. P(b) on the real corpus (deliverable)",
        f"{'b':>5} {'T(h)':>6} {'power':>10} {'frac':>7} "
        f"{'P_last':>7} {'P_own':>7} {'P_full':>7} {'in-band':>8}",
        lambda j: (
            f"{power[j]:10.3e} {frac[j]:7.4f} "
            f"{res_real['last']['p'][j]:7.3f} {res_real['own']['p'][j]:7.3f} "
            f"{res_real['full']['p'][j]:7.3f} {leak[j]:8.4f}"
        ),
    )
    table(
        "B. P(b) on the de-clocked residual (clock profiles removed)",
        f"{'b':>5} {'T(h)':>6} {'power':>10} {'frac':>7} "
        f"{'P_last':>7} {'P_own':>7} {'P_full':>7}",
        lambda j: (
            f"{band_power(declk)[j]:10.3e} "
            f"{band_power(declk)[j] / band_power(declk).sum():7.4f} "
            f"{res_declk['last']['p'][j]:7.3f} {res_declk['own']['p'][j]:7.3f} "
            f"{res_declk['full']['p'][j]:7.3f}"
        ),
    )
    table(
        "C. phase-randomized surrogates: S = raw, S' = de-clocked",
        f"{'b':>5} {'T(h)':>6} {'P_own':>7} {'S_own':>7} {'P-S':>7} "
        f"{'D_own':>7} {'S$_own':>7} {'D-S$':>7}",
        lambda j: (
            f"{res_real['own']['p'][j]:7.3f} {res_surro['own']['p'][j]:7.3f} "
            f"{res_real['own']['p'][j] - res_surro['own']['p'][j]:7.3f} "
            f"{res_declk['own']['p'][j]:7.3f} {res_declk_s['own']['p'][j]:7.3f} "
            f"{res_declk['own']['p'][j] - res_declk_s['own']['p'][j]:7.3f}"
        ),
    )

    noise_p = np.array(res_noise["own"]["p"])
    print(
        f"\n  white-noise control        : P_own mean {noise_p.mean():+.3f} "
        f"range [{noise_p.min():+.3f}, {noise_p.max():+.3f}]"
    )
    print(
        f"  in-band fraction, white    : mean {np.mean(leak_wn):.3f} "
        f"range [{np.min(leak_wn):.3f}, {np.max(leak_wn):.3f}]"
    )
    print(
        f"  oversampling control       : 0.5-wide bands, white noise -> P_own mean "
        f"{np.mean(over['own']['p']):.3f} (degenerate)"
    )
    own_tr = np.mean(res_real["own"]["p_train"])
    full_tr = np.mean(res_real["full"]["p_train"])
    print(
        f"  in-sample (train) P on real: own {own_tr:.3f}, full {full_tr:.3f} "
        f"(eval {np.mean(res_real['own']['p']):.3f} / "
        f"{np.mean(res_real['full']['p']):.3f})"
    )

    record = {
        "dataset": DATASET,
        "resolution_minutes": STEP_MIN,
        "band_definition": (
            f"b = k / period_samples, k = {cfg.k:g}, Nyquist b = {cfg.k / 2:g}"
        ),
        "arm": {
            "k": cfg.k,
            "ctx": cfg.ctx,
            "grid": cfg.grid,
            "context_patches": cfg.n_ctx,
            "window": cfg.win,
        },
        "patch": cfg.k,
        "context": cfg.ctx,
        "context_patches": cfg.n_ctx,
        "window": cfg.win,
        "bands": cfg.bands.tolist(),
        "period_hours": periods.tolist(),
        "in_band_half_width": HALF,
        "clock_period_samples": WEEK,
        "n_windows": int(len(real)),
        "n_train": n_tr,
        "n_eval": n_ev,
        "split_seed": SPLIT_SEED,
        "surrogate_seed": SURROGATE_SEED,
        "power": power.tolist(),
        "power_frac": frac.tolist(),
        "in_band_fraction": leak.tolist(),
        "real": res_real,
        "surrogate": res_surro,
        "declocked": res_declk,
        "declocked_surrogate": res_declk_s,
        "controls": {
            "white_noise_P_own": res_noise["own"]["p"],
            "white_noise_in_band_fraction": leak_wn.tolist(),
            "oversampled_white_noise_P_own": over["own"]["p"],
        },
        "wall": time.time() - t0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out}\ntotal wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
