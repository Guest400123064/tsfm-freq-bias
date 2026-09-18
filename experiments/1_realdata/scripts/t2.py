r"""T2: is a model trained on the real corpus flat in per-band retention?

T2 of ``PLAN.md`` §6.2, the real-data analogue of P2b. T1/T1b established the
explaining variable on this corpus: per-band conditional predictability
``P(b)`` is nearly flat across a 32x range of ``b`` (periods 32 h .. 1 h) while
band power falls 463x. If a next-patch model's per-band attenuation is the
Bayes-optimal shrink for its training marginal, this corpus should produce a
**nearly flat per-band attenuation** -- not the "high frequency is worse"
pattern the field reports. That prediction is what this script tests.

Two arms, both requested by the brief. The aperture *is* the horizon on a
patch-based construction (T1b), so they are two different forecasting problems
and are never compared to each other band by band:

===========  =====  ====  ====  ==========  ===================
arm          ctx    k     P     window      next-patch horizon
===========  =====  ====  ====  ==========  ===================
``k32``      512    32    16    544         16 h
``k64``      1024   64    16    1088        32 h
===========  =====  ====  ====  ==========  ===================

Corpus construction is T1's, imported rather than reimplemented: same five
series (``australian_electricity_demand_dataset``, 30 min), same per-series
centre and unit-variance over the whole series (contract 9), same
non-overlapping stride (so the split stays leak-free), same window split
(``predictability.SPLIT_SEED``, half train / half eval), and the same per-patch
demeaning before the matched filter. ``check_aperture`` asserts that this
script's training windows reproduce T1's own coefficients exactly, band by
band, so the demeaning convention -- the binding invariant, not the value of
``k`` -- cannot drift.

Two readouts, both through T1's matched filter:

``corpus``  ``r_b = |Z(pred, b)| / |Z(target, b)|`` at the last trained
            position (``pred[:, -2]`` vs ``pat(x)[:, -1]``) on the held-out
            half. On the corpus shrinking is *correct*, so this cannot by
            itself show a prior.
``probe``   the same reading on a fixed, clean, deterministic 8-tone probe,
            identical in every arm and seed, RMS-matched to the corpus's
            window RMS (contract 9). A deficit that survives here is the prior
            transferring -- P2b's structure.

Each readout is quoted beside contract 5's **oracle**: a ridge readout of the
same frozen latents, fitted on the held-out corpus windows. P3's lesson is that
the head can sit uniformly below its own optimum for optimisation reasons, so
the oracle is what separates "the representation carries the shrink" from "the
head under-fits".

Probe bands are ``b in {1, 2, 4, 6, 8, 10, 12, 16}`` (contract A4 with no
slack: every gap is >= 1 cycle/patch, and the minimum gap is exactly 1), all
integers so ``2b`` is an integer and the matched filter is exact. Physical
periods are ``16 / b`` hours in the ``k32`` arm and ``32 / b`` hours in the
``k64`` arm -- the same ``b``, the corpus's physical spectrum slid by 2x, which
is the relabel T1b describes.

Fit gate (contract 8), stated before the run: an arm is quotable only if its
across-seed mean held-out variance explained beats **both** interpretable
baselines -- persistence (copy the last context patch) and the mean predictor
-- by at least ``GATE_MIN`` = 0.02 absolute *and* at least ``GATE_SD`` = 3x the
across-seed sd. If it does not, the brief's escalation runs one larger config
(``hidden 64 / 4 layers``, same ``ctx``/``k`` and step count) and both are
reported. Nothing is tuned to make the curve flat or sloped.

Run one arm/seed per process (they are CPU-bound and parallelise well):

``.venv/bin/python experiments/1_realdata/scripts/t2.py --arm k32 --seed 0``

then read the tables back with ``--summarize``. ``--tag`` names a variant
(e.g. the escalation's ``big``); ``--steps``/``--hidden``/``--layers`` override
the defaults, which are S0 per ``PLAN.md`` Appendix C.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import predictability as pred
import torch
import torch.nn.functional as F

from fbias import probes
from fbias.data import make_mixture
from fbias.model import SimTFM
from fbias.probes import fit_oracle

ARMS = {"k32": (32, 512), "k64": (64, 1024)}
BANDS = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
P = 16  # ctx / k, patches per context -- the same in both arms
FREQS = [b * P for b in BANDS]  # cycles per ctx window
PROBE_SEED = 123
N_PROBE = 256
SEEDS = (0, 1, 2)
STEPS = 20_000
EVAL_STEPS = (1_000, 2_000, 5_000, 10_000, 15_000, 20_000)
BATCH, LR = 64, 1e-2
HIDDEN, LAYERS, HEADS = 32, 2, 4
THREADS = 2
GATE_MIN, GATE_SD = 0.02, 3.0
RUNS = Path(__file__).resolve().parents[1] / "runs"


def arm_cfg(arm):
    """T1's :class:`predictability.Cfg` for one arm: ctx = 16 k, T1's ``b`` grid."""
    k, _ = ARMS[arm]
    return pred.config(k=k)


def raw_windows(cfg):
    """T1's corpus as raw ``(n, ctx + k)`` windows."""
    _, raw = pred.load_monash(pred.DATASET)
    out = []
    for _, _, v in raw:
        x = (v - v.mean()) / v.std()
        n = len(x) // cfg.win
        out.append(x[: n * cfg.win].reshape(n, cfg.win))
    return np.concatenate(out)


def check_aperture(cfg, windows, coeffs):
    """Assert the training windows are T1's corpus, through T1's estimator.

    ``coeffs`` is ``predictability.corpus(cfg)``, built independently from the
    loader. Matching here means the training windows, the per-patch demeaning
    and the matched filter are all T1's, band by band. ``predictability``'s
    helpers take a 1-D series, so the window matrix is raveled first -- the
    windows are contiguous and non-overlapping, so raveling rescans exactly the
    same cuts.
    """
    patches = pred.windows_of(windows.ravel(), cfg)
    for j, b in enumerate(cfg.bands):
        got = pred.matched(patches, b)
        assert np.allclose(got, coeffs[:, :, j], rtol=0.0, atol=1e-9), (
            f"training corpus differs from T1's at band {b} "
            f"(max |dZ| {np.abs(got - coeffs[:, :, j]).max():.2e})"
        )


def probe_windows(cfg, rms):
    """Fixed clean multi-tone probe, RMS-matched to the corpus (contract 9)."""
    x = make_mixture(FREQS, N_PROBE, cfg.ctx, cfg.k, PROBE_SEED, normalize=True)
    x = x.numpy().astype(np.float64)
    return x * (rms / math.sqrt(float(np.mean(x**2))))


def probe_band_power(cfg, probe):
    """``|Z|^2 / 2`` of the probe at each band, in the corpus's band units."""
    patches = pred.windows_of(np.ascontiguousarray(probe).ravel(), cfg)
    return [float(np.mean(np.abs(pred.matched(patches, b)) ** 2) / 2) for b in BANDS]


def ridge_readout(cfg, coeffs, variant, band, tr, ev):
    """T1's ridge, read as a retention: ``(P, r, r_ms)`` for one band.

    Same features, same split, same lambda search and the same prediction
    formula as ``predictability.p_band``; the extra readout is the amplitude
    ratio ``|Z_pred| / |Z_target|`` on the held-out half, so ``P`` (an R2) can
    be compared against the retention it implies *without training anything*.
    ``r`` is the mean of the per-window ratios (the model readouts' convention)
    and ``r_ms`` the pooled RMS ratio ``sqrt(sum|Z_pred|^2 / sum|Z_ev|^2)``.
    """
    f = pred.features(coeffs, variant, band, cfg)
    z = coeffs[:, -1, band]
    order = np.random.default_rng(pred.SPLIT_SEED).permutation(len(tr))
    a, b = tr[order[: len(tr) // 2]], tr[order[len(tr) // 2 :]]
    lam = max((pred.p_band(f[a], z[a], f[b], z[b], t), t) for t in pred.LAMBDAS)[1]
    mu, m = f[tr].mean(axis=0), z[tr].mean()
    fc, zc = f[tr] - mu, z[tr] - m
    gram = fc.conj().T @ fc
    gram.flat[:: gram.shape[0] + 1] += lam * np.trace(gram).real / gram.shape[0]
    w = np.linalg.solve(gram, fc.conj().T @ zc)
    z_pred, z_ev = f[ev] @ w + (m - mu @ w), z[ev]
    var = np.mean(np.abs(z_ev - z_ev.mean()) ** 2)
    return (
        float(1 - np.mean(np.abs(z_pred - z_ev) ** 2) / var),
        float(np.mean(np.abs(z_pred) / np.abs(z_ev))),
        float(np.sqrt(np.sum(np.abs(z_pred) ** 2) / np.sum(np.abs(z_ev) ** 2))),
    )


def relation(arm):
    """`P(b)` against the retention T1's own estimator achieves, band by band.

    Model-free: the same corpus, aperture and split as ``P(b)``, so this fixes
    which relation holds between the R2 ``P`` and an amplitude ratio ``r`` on
    this data before any model is trained.
    """
    cfg = arm_cfg(arm)
    coeffs = pred.corpus(cfg)
    n = len(coeffs)
    idx = np.random.default_rng(pred.SPLIT_SEED).permutation(n)
    tr, ev = idx[: n // 2], idx[n // 2 :]
    out = {
        "arm": arm,
        "k": cfg.k,
        "ctx": cfg.ctx,
        "window": cfg.win,
        "bands": cfg.bands.tolist(),
        "period_hours": cfg.periods_h.tolist(),
        "n_train": int(len(tr)),
        "n_eval": int(len(ev)),
    }
    for variant in ("own", "full", "last"):
        rows = [
            ridge_readout(cfg, coeffs, variant, j, tr, ev)
            for j in range(len(cfg.bands))
        ]
        out[f"P_{variant}"] = [r[0] for r in rows]
        out[f"r_{variant}"] = [r[1] for r in rows]
        out[f"r_ms_{variant}"] = [r[2] for r in rows]
    return out


def print_relation_data(rec):
    """The model-free half of the P-vs-r question, at the probe bands."""
    bands = np.array(rec["bands"])
    at = [int(np.argmin(np.abs(bands - b))) for b in BANDS]
    print(
        f"\n-- {rec['arm']}: T1's own ridge read as a retention "
        f"({rec['n_train']}/{rec['n_eval']} windows, model-free)"
    )
    print(
        f"   {'b':>5} {'T(h)':>6} {'P_own':>6} {'r_own':>7} {'r_ms':>7} "
        f"{'sqrt(P)':>8} {'r/sqrt(P)':>10} {'r/P':>7} {'P_full':>7} {'r_full':>7} "
        f"{'r_last':>7}"
    )
    for j, b in enumerate(BANDS):
        i = at[j]
        p, r = rec["P_own"][i], rec["r_own"][i]
        print(
            f"   {b:5.1f} {rec['period_hours'][i]:6.2f} {p:6.3f} {r:7.4f} "
            f"{rec['r_ms_own'][i]:7.4f} {np.sqrt(p):8.4f} {r / np.sqrt(p):10.4f} "
            f"{r / p:7.4f} {rec['P_full'][i]:7.3f} {rec['r_full'][i]:7.4f} "
            f"{rec['r_last'][i]:7.4f}"
        )
    over = np.array(rec["P_own"])
    r_all = np.array(rec["r_own"])
    root = np.sqrt(over)
    print(
        f"   over all {len(bands)} bands: P_own in "
        f"[{over.min():.3f}, {over.max():.3f}] "
        f"(max/min {over.max() / over.min():.3f}), "
        f"r_own in [{r_all.min():.3f}, {r_all.max():.3f}] "
        f"(max/min {r_all.max() / r_all.min():.3f})"
        f"\n   mean |r - sqrt(P)| {np.abs(r_all - root).mean():.4f}, "
        f"mean |r - P| {np.abs(r_all - over).mean():.4f}, "
        f"mean r/sqrt(P) {(r_all / root).mean():.4f}"
    )


def matched(patch, b):
    """T1's matched filter, on each patch demeaned first (T1's convention)."""
    x = patch - patch.mean(-1, keepdim=True)
    return probes.matched_amp(x, b, patch.shape[-1])


def band_component(patch, b):
    """Band-``b`` waveform of each patch, ``(..., k)``, patches demeaned first."""
    k = patch.shape[-1]
    z = matched(patch, b)
    t = torch.arange(k, dtype=torch.float32)
    return (z.unsqueeze(-1) * torch.exp(2j * math.pi * b * t / k)).real


@torch.no_grad()
def forward_parts(model, x):
    """``(pred, target)`` aligned at every position, plus the readout position."""
    model.eval()
    pred = model(x)[0]
    patches = model.pat(x)
    return pred[:, :-1], patches[:, 1:], pred[:, -2], patches[:, -1]


@torch.no_grad()
def retention_table(model, x, bands=BANDS):
    """``r_b = |Z(pred, b)| / |Z(target, b)|`` per band, at the last position.

    Mean over windows of the per-window amplitude ratio, the synthetic work's
    convention. Index-aligned with ``BANDS``.
    """
    _, _, f_hat, true = forward_parts(model, x)
    return [
        float((matched(f_hat, b).abs() / matched(true, b).abs()).mean()) for b in bands
    ]


@torch.no_grad()
def fit_gate(model, x, cfg, bands=BANDS):
    """Contract 8: held-out variance explained, baselines, per-band breakdown."""
    pred_all, target_all, f_hat, target = forward_parts(model, x)
    patches = model.pat(x)
    ctx_mean = x[:, : cfg.ctx].mean(-1, keepdim=True)
    var_last = float(target.var(unbiased=False))
    var_all = float(target_all.var(unbiased=False))
    mse_last = float(((f_hat - target) ** 2).mean())
    mse_all = float(((pred_all - target_all) ** 2).mean())
    out = {
        "var_target_last": var_last,
        "var_target_all": var_all,
        "mse_last": mse_last,
        "var_explained_last": 1 - mse_last / var_last,
        "mse_all": mse_all,
        "var_explained_all": 1 - mse_all / var_all,
        "baseline_persistence_last": 1
        - float(((patches[:, -2] - target) ** 2).mean()) / var_last,
        "baseline_mean_last": 1 - float(((ctx_mean - target) ** 2).mean()) / var_last,
        "baseline_persistence_all": 1
        - float(((patches[:, :-1] - target_all) ** 2).mean()) / var_all,
        "baseline_mean_all": 1
        - float(((ctx_mean.unsqueeze(-1) - target_all) ** 2).mean()) / var_all,
        "band_var_explained": [],
    }
    for b in bands:
        p, t = band_component(f_hat, b), band_component(target, b)
        out["band_var_explained"].append(
            float(1 - ((p - t) ** 2).mean() / ((t - t.mean()) ** 2).mean())
        )
    return out


@torch.no_grad()
def oracle_table(model, fit_x, read_x, cfg, bands=BANDS):
    """Contract 5's upper bound: a ridge readout of the frozen latents.

    Fitted on the held-out corpus windows and read at the same last position
    (``z[:, -2:]``, so ``fit_oracle``'s internal one-patch shift lands exactly
    on the window's last patch). It bounds what a *linear readout* of this
    representation can do, which is what separates "the head under-fits
    uniformly" from "the representation itself carries the shrink" (P3's
    resolution). Contract 5's caveat applies: the fit target is the *observed*
    patch, so the absolute level is not a Bayes bound and only the comparison
    against the head is read.
    """
    z_fit, patches_fit = model.map(model.pat(fit_x)), model.pat(fit_x)
    z_read, patches_read = model.map(model.pat(read_x)), model.pat(read_x)
    return [
        float(
            fit_oracle(
                z_fit, patches_fit, z_read[:, -2:], patches_read[:, -2:], b, cfg.k
            )["retention"]
        )
        for b in bands
    ]


@torch.no_grad()
def readout(model, held, probe, cfg):
    """Both readouts, the gate, and the oracle, one checkpoint's worth."""
    row = fit_gate(model, held, cfg)
    row["r_corpus"] = retention_table(model, held)
    row["r_probe"] = retention_table(model, probe)
    row["r_oracle_corpus"] = oracle_table(model, held, held, cfg)
    row["r_oracle_probe"] = oracle_table(model, held, probe, cfg)
    return row


def train_one(seed, windows, cfg, probe, held, steps, hidden, layers, evals):
    """S0's training loop (P1/P2b/P3's, unchanged) on the real corpus.

    Windows do not overlap, so T1's random half-split cannot share a sample
    between train and eval; the split is the same permutation T1's ``P(b)`` is
    scored on, so both numbers describe the same held-out half.
    """
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=cfg.ctx,
        patch_size=cfg.k,
        hidden_size=hidden,
        num_layers=layers,
        num_attn_heads=HEADS,
    )
    n = len(windows)
    idx = np.random.default_rng(pred.SPLIT_SEED).permutation(n)
    train = idx[: n // 2]
    data = torch.from_numpy(windows[train].astype(np.float32))
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    curve, loss = [], torch.tensor(0.0)
    for step in range(1, steps + 1):
        x = data[torch.randint(len(data), (BATCH,), generator=gen)]
        pred_p, _, _ = model(x)
        loss = F.mse_loss(pred_p[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in evals:
            row = readout(model, held, probe, cfg)
            curve.append({"step": step, "loss": float(loss), **row})
            model.train()
            print(
                f"    step {step:6d} loss {float(loss):.6f} "
                f"ve_last {row['var_explained_last']:+.4f} "
                f"ve_all {row['var_explained_all']:+.4f} "
                f"probe r mean {np.mean(row['r_probe']):.4f}"
            )
    return model, curve, float(loss), len(train), len(idx) - len(train)


def run_arm(arm, seed, steps, hidden, layers, tag):
    """Everything one (arm, seed) run reports, as a JSON-ready dict."""
    out = RUNS / f"t2{tag}_{arm}_s{seed}.json"
    if out.exists():
        print(f"{out.name} exists, skipping")
        return None
    cfg = arm_cfg(arm)
    t0 = time.time()
    windows = raw_windows(cfg)
    print(
        f"[{tag or 's0'} {arm} seed {seed}] {len(windows)} windows of {cfg.win} "
        f"samples, ctx {cfg.ctx} / k {cfg.k}, horizon "
        f"{cfg.k * pred.STEP_MIN / 60:.0f} h"
    )

    coeffs = pred.corpus(cfg)
    check_aperture(cfg, windows, coeffs)
    res, n_tr, n_ev = pred.run(cfg, coeffs, ("last", "own", "full"))
    leak = pred.in_band_fraction(cfg)
    power = pred.band_power(coeffs)
    print(f"  T1's estimator: {n_tr}/{n_ev} windows, {time.time() - t0:.1f}s")

    n = len(windows)
    idx = np.random.default_rng(pred.SPLIT_SEED).permutation(n)
    held = windows[idx[n // 2 :]]
    rms = math.sqrt(float(np.mean(windows[idx[: n // 2]] ** 2)))
    probe = probe_windows(cfg, rms)
    at = [int(np.argmin(np.abs(cfg.bands - b))) for b in BANDS]
    assert all(abs(cfg.bands[j] - b) < 1e-9 for j, b in zip(at, BANDS, strict=True))
    evals = sorted({s for s in EVAL_STEPS if s < steps} | {steps})

    t1 = time.time()
    model, curve, loss, n_train, _ = train_one(
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
        "tag": tag,
        "arm": arm,
        "k": cfg.k,
        "ctx": cfg.ctx,
        "P": cfg.n_ctx,
        "window": cfg.win,
        "horizon_hours": cfg.k * pred.STEP_MIN / 60,
        "band_hours": [float(cfg.k / b * pred.STEP_MIN / 60) for b in BANDS],
        "seed": seed,
        "steps": steps,
        "model": {"hidden": hidden, "layers": layers, "heads": HEADS},
        "optim": {"name": "SGD", "lr": LR, "batch": BATCH, "threads": THREADS},
        "n_windows": n,
        "n_train": n_train,
        "n_eval": n_ev,
        "split_seed": pred.SPLIT_SEED,
        "bands": list(BANDS),
        "grid_index": at,
        "power": [float(power[j]) for j in at],
        "power_frac": [float(power[j] / power.sum()) for j in at],
        "P_last": [float(res["last"]["p"][j]) for j in at],
        "P_own": [float(res["own"]["p"][j]) for j in at],
        "P_full": [float(res["full"]["p"][j]) for j in at],
        "P_own_train": [float(res["own"]["p_train"][j]) for j in at],
        "in_band": [float(leak[j]) for j in at],
        "corpus_rms": rms,
        "probe_rms": float(np.sqrt(np.mean(probe**2))),
        "probe_power": probe_band_power(cfg, probe),
        "curve": curve,
        "wall": time.time() - t0,
    }
    record.update(curve[-1])  # the final checkpoint's metrics, at the top level
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"  wrote {out} ({record['wall']:.1f}s total)")
    return record


def load_runs():
    """Every ``t2<tag>_<arm>_s<seed>.json`` on disk, grouped by ``(tag, arm)``.

    ``runs/t2_summary.json`` matches the glob's shape, so anything without an
    ``arm``/``seed`` pair is skipped rather than mistaken for a run.
    """
    groups = {}
    for path in sorted(RUNS.glob("t2*_s*.json")):
        rec = json.loads(path.read_text())
        if "arm" not in rec or "seed" not in rec:
            continue
        groups.setdefault((rec["tag"], rec["arm"]), []).append(rec)
    for rows in groups.values():
        rows.sort(key=lambda r: r["seed"])
    return groups


def collect(rows, key):
    """Across-seed ``(mean, sd)`` for a scalar metric."""
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    return float(vals.mean()), float(vals.std())


def print_gate(tag, arm, rows):
    """Fit gate per seed and pooled, with the pre-stated verdict."""
    print(f"\n-- {tag or 's0'} / {arm}: fit gate (contract 8, held-out windows)")
    print(
        f"   {'seed':>4} {'mse_last':>9} {'ve_last':>8} {'ve_all':>8} "
        f"{'persist':>8} {'mean':>8} {'persist_all':>11} {'mean_all':>8}"
    )
    for r in rows:
        print(
            f"   {r['seed']:>4} {r['mse_last']:9.5f} {r['var_explained_last']:+8.4f} "
            f"{r['var_explained_all']:+8.4f} {r['baseline_persistence_last']:+8.4f} "
            f"{r['baseline_mean_last']:+8.4f} {r['baseline_persistence_all']:+11.4f} "
            f"{r['baseline_mean_all']:+8.4f}"
        )
    ve, ve_sd = collect(rows, "var_explained_last")
    pe, pe_sd = collect(rows, "baseline_persistence_last")
    me, me_sd = collect(rows, "baseline_mean_last")
    margin = min(ve - pe, ve - me)
    need = max(GATE_MIN, GATE_SD * ve_sd)
    print(
        f"   mean±sd ve_last {ve:+.4f}±{ve_sd:.4f} vs persistence "
        f"{pe:+.4f}±{pe_sd:.4f} / mean {me:+.4f}±{me_sd:.4f}: "
        f"margin {margin:+.4f}, needs {need:+.4f} -> "
        f"{'PASS' if margin >= need else 'FAIL'}"
    )
    return {
        "ve_last": ve,
        "ve_last_sd": ve_sd,
        "persistence": pe,
        "persistence_sd": pe_sd,
        "mean": me,
        "mean_sd": me_sd,
        "margin": float(margin),
        "needed": float(need),
        "pass": bool(margin >= need),
    }


def print_bands(tag, arm, rows):
    """The band table: power, predictability, both retentions, oracle, leakage."""
    d = rows[0]
    probe = np.array([r["r_probe"] for r in rows])
    corpus = np.array([r["r_corpus"] for r in rows])
    orc_p = np.array([r["r_oracle_probe"] for r in rows])
    orc_c = np.array([r["r_oracle_corpus"] for r in rows])
    bve = np.array([r["band_var_explained"] for r in rows])
    p_own = np.array(d["P_own"])
    p_full = np.array(d["P_full"])
    root = np.sqrt(np.where(p_own > 0, p_own, np.nan))
    print(
        f"\n-- {tag or 's0'} / {arm}: bands, horizon {d['horizon_hours']:.0f} h, "
        f"{len(rows)} seeds"
    )
    print(
        f"   {'b':>5} {'T(h)':>6} {'power':>10} {'P_own':>6} {'P_full':>6} "
        f"{'sqrt(P)':>7} {'in-band':>7} {'r_probe':>15} {'r_corpus':>15} "
        f"{'or_probe':>8} {'or_corp':>8} {'ve_band':>8}"
    )
    for i, b in enumerate(d["bands"]):
        print(
            f"   {b:5.1f} {d['band_hours'][i]:6.2f} {d['power'][i]:10.3e} "
            f"{p_own[i]:6.3f} {p_full[i]:6.3f} {root[i]:7.3f} "
            f"{d['in_band'][i]:7.4f} "
            f"{probe[:, i].mean():8.4f}±{probe[:, i].std():.4f} "
            f"{corpus[:, i].mean():8.4f}±{corpus[:, i].std():.4f} "
            f"{orc_p[:, i].mean():8.4f} {orc_c[:, i].mean():8.4f} "
            f"{bve[:, i].mean():+8.4f}"
        )
    return {
        "probe": probe,
        "corpus": corpus,
        "oracle_probe": orc_p,
        "oracle_corpus": orc_c,
        "band_ve": bve,
        "P_own": p_own,
        "P_full": p_full,
        "root": root,
    }


def print_shape(tag, arm, rows, tab):
    """Is the clean-probe curve flat? spread_b against the seed spread.

    The top band ``b = k / 2`` is the aperture's own Nyquist for the ``k32``
    arm, where the matched filter's positive and negative frequency images
    coincide and a pure tone of amplitude ``A`` reads ``|Z| = 2A``. Ratios stay
    valid (both sides go through the same filter) but the band is reported
    separately as well.
    """
    probe = tab["probe"]
    m = probe.mean(axis=0)
    seed_sd = float(np.sqrt((probe.std(axis=0) ** 2).mean()))
    spread = float(m.max() - m.min())
    inner = m[:-1]
    spread_inner = float(inner.max() - inner.min())
    logs = np.log2(np.array([float(b) for b in rows[0]["bands"]]))
    ratio = spread / seed_sd if seed_sd else float("nan")
    print(
        f"\n-- {tag or 's0'} / {arm}: is the clean-probe curve flat?"
        f"\n   spread_b (max-min over bands, seed mean) {spread:.4f}"
        f"\n   spread_b excluding b = {rows[0]['bands'][-1]:g} (Nyquist at k32) "
        f"        {spread_inner:.4f}"
        f"\n   pooled per-band seed sd                 {seed_sd:.4f}"
        f"\n   spread_b / seed sd                      {ratio:.2f}"
        f"\n   corr(log2 b, r_probe)                   "
        f"{np.corrcoef(logs, m)[0, 1]:+.2f}"
        f"\n   mean r_probe {m.mean():.4f}, range [{m.min():.4f}, {m.max():.4f}]"
        f"\n   per-band seed sd                        "
        + " ".join(f"{s:.3f}" for s in probe.std(axis=0))
    )
    return {
        "spread_b": spread,
        "spread_b_inner": spread_inner,
        "seed_sd": seed_sd,
        "spread_over_seed_sd": float(ratio),
        "corr_log2b": float(np.corrcoef(logs, m)[0, 1]),
        "mean_r": float(m.mean()),
        "min_r": float(m.min()),
        "max_r": float(m.max()),
    }


def print_relation(tag, arm, rows, tab):
    """Which relation the data shows: r vs P, or r vs sqrt(P), or neither.

    Reported for the head (both readouts) and for the oracle readout of the
    probe, since the head's *level* can be an optimisation gap (P3) while the
    relation itself is a property of the marginal.
    """
    m = tab["probe"].mean(axis=0)
    orc = tab["oracle_probe"].mean(axis=0)
    p, root = tab["P_own"], tab["root"]
    print(
        f"\n-- {tag or 's0'} / {arm}: r_b against P and sqrt(P) "
        f"(clean probe; 'or' is the oracle readout)"
    )
    print(
        f"   {'b':>5} {'r':>7} {'P_own':>7} {'sqrt(P)':>8} {'r-P':>8} "
        f"{'r-sqrt(P)':>10} {'r/sqrt(P)':>10} {'or':>7} {'or/sqrt(P)':>11}"
    )
    for i, b in enumerate(rows[0]["bands"]):
        print(
            f"   {b:5.1f} {m[i]:7.4f} {p[i]:7.4f} {root[i]:8.4f} "
            f"{m[i] - p[i]:+8.4f} {m[i] - root[i]:+10.4f} {m[i] / root[i]:10.4f} "
            f"{orc[i]:7.4f} {orc[i] / root[i]:11.4f}"
        )
    dev_p, dev_root = np.abs(m - p), np.abs(m - root)
    out = {
        "mean_abs_dev_P": float(dev_p.mean()),
        "mean_abs_dev_sqrtP": float(dev_root.mean()),
        "mean_ratio_over_sqrtP": float((m / root).mean()),
        "oracle_mean_ratio_over_sqrtP": float((orc / root).mean()),
        "corr_r_P": float(np.corrcoef(m, p)[0, 1]),
        "corr_r_sqrtP": float(np.corrcoef(m, root)[0, 1]),
        "corr_oracle_sqrtP": float(np.corrcoef(orc, root)[0, 1]),
        "reads_closer_to": "sqrt(P)" if dev_root.mean() < dev_p.mean() else "P",
    }
    print(
        f"   mean |r - P| {dev_p.mean():.4f}, mean |r - sqrt(P)| {dev_root.mean():.4f}"
        f" -> closer to {out['reads_closer_to']};"
        f" oracle mean r/sqrt(P) {out['oracle_mean_ratio_over_sqrtP']:.4f}"
    )
    return out


def summarize(groups=None):
    """Print the report tables over every run on disk, and pool them to JSON."""
    groups = groups if groups is not None else load_runs()
    record = {"arms": {}}
    for (tag, arm), rows in sorted(groups.items()):
        d = rows[0]
        print(
            f"\n=== {'/'.join(x for x in (tag, arm) if x)} | {len(rows)} seeds | "
            f"k {d['k']} ctx {d['ctx']} window {d['window']} | horizon "
            f"{d['horizon_hours']:.0f} h | steps {d['steps']} | hidden "
            f"{d['model']['hidden']} / {d['model']['layers']}L | "
            f"{d['n_windows']} windows ({d['n_train']}/{d['n_eval']})"
        )
        gate = print_gate(tag, arm, rows)
        tab = print_bands(tag, arm, rows)
        shape = print_shape(tag, arm, rows, tab)
        rel = print_relation(tag, arm, rows, tab)
        print(
            f"   corpus rms {d['corpus_rms']:.4f} / probe rms {d['probe_rms']:.4f} "
            f"(contract 9; ratio {d['probe_rms'] / d['corpus_rms']:.4f})"
        )
        record["arms"][f"{tag}/{arm}".strip("/")] = {
            "tag": tag,
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
            "power_frac": d["power_frac"],
            "probe_power": d["probe_power"],
            "P_own": d["P_own"],
            "P_full": d["P_full"],
            "P_last": d["P_last"],
            "in_band": d["in_band"],
            "corpus_rms": d["corpus_rms"],
            "probe_rms": d["probe_rms"],
            "gate": gate,
            "shape": shape,
            "relation": rel,
            "probe_mean": tab["probe"].mean(axis=0).tolist(),
            "probe_sd": tab["probe"].std(axis=0).tolist(),
            "corpus_mean": tab["corpus"].mean(axis=0).tolist(),
            "corpus_sd": tab["corpus"].std(axis=0).tolist(),
            "oracle_probe_mean": tab["oracle_probe"].mean(axis=0).tolist(),
            "oracle_corpus_mean": tab["oracle_corpus"].mean(axis=0).tolist(),
            "band_var_explained": tab["band_ve"].mean(axis=0).tolist(),
            "ve_last": [r["var_explained_last"] for r in rows],
            "wall": [r["wall"] for r in rows],
        }
    out = RUNS / "t2_summary.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out}")
    return record


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=(*ARMS, "both"), default="both")
    p.add_argument("--seed", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--layers", type=int, default=LAYERS)
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--tag", default="", help="variant name, e.g. 'big'")
    p.add_argument("--summarize", action="store_true", help="read the tables back")
    p.add_argument(
        "--relation",
        action="store_true",
        help="model-free: T1's ridge read as a retention (P against r)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.summarize:
        summarize()
        return 0
    arms = list(ARMS) if args.arm == "both" else [args.arm]
    if args.relation:
        rec = {arm: relation(arm) for arm in arms}
        for arm in arms:
            print_relation_data(rec[arm])
        out = RUNS / "t2_relation.json"
        out.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"\nwrote {out}")
        return 0
    for arm in arms:
        for seed in args.seed:
            run_arm(arm, seed, args.steps, args.hidden, args.layers, args.tag)
    return 0


if __name__ == "__main__":
    main()
