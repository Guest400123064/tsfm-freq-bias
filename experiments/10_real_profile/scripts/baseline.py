r"""T5 step 2: the baseline model on the real corpus, and the headroom.

Trains on the *unmodified* corpus under the setting fixed in the README (per-window
normalization from the context only; integer bands; random half-split by default,
first-80%-by-time as a separate practical arm) and records everything the
intervention will be compared against:

  * MASE with its seasonal-naive denominator (m = 48 samples = 24 h), plus a
    persistence baseline;
  * MSE / MAE in normalized space;
  * per-band r, Delta_k and phase error as learning curves -- checkpoints every
    100 steps to 2 000, then every 500 to 10 000;
  * the per-band **frozen-latent oracle** (a ridge on the model's own latents,
    **fitted on training windows and evaluated on held-out ones**, an in-sample
    ridge being degenerate here) and the headroom `(MSE - MSE_oracle) / MSE`.
    On this corpus that oracle is not an upper bound -- the trained nonlinear
    head ends up 40-60 % better than it, so the headroom goes negative and is
    reported only as a description. The reference that replaced it is the
    model-free per-band bound below;
  * the **model-free per-band bound**: a ridge from the context's own
    coefficients at a band to the target patch's coefficient at that band,
    fitted on training windows and scored on held-out ones (T1's `P(b)`). Its
    retention equivalent is `sqrt(P(b))` and the residual it leaves is
    `floor_b = (1 - P(b)) * P_b`. This is a baseline *predictor*, not a strict
    bound -- it is linear and sees one band -- and it is what `res/floor` in the
    intervention reads against.

`r_b` is never compared to 1: on real data the target patch carries content no
function of the context determines, so a deficit is only reported as
`oracle_b - r_b`.

Run from the repo root:
``.venv/bin/python experiments/10_real_profile/scripts/baseline.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from fbias.model import SimTFM  # noqa: E402
from fbias.probes import (  # noqa: E402
    band_component,
    delta_k,
    fit_oracle,
    matched_amp,
    retention,
)

PROF_PATH = ROOT / "experiments/10_real_profile/scripts/corpus_profile.py"
RUNS = ROOT / "experiments/10_real_profile/runs"

K, CTX = 32, 512
BANDS = tuple(float(b) for b in range(1, 17))  # integer bands, periods 16 h .. 1 h
SEASONAL = 48  # 24 h at 30-min steps
HIDDEN, LAYERS, HEADS = 64, 2, 4
STEPS, BATCH, LR = 10000, 64, 1e-2
CKPT, CKPT_FINE, CKPT_FINE_UNTIL = 500, 100, 2000
SEEDS = (0, 1, 2)
ORACLE_FIT = 512  # training windows used to fit the readout
MARK_STEPS = (100, 500, 1000, 2000, 5000, 10000)


def load_profile():
    spec = importlib.util.spec_from_file_location("prof", PROF_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prof"] = mod
    spec.loader.exec_module(mod)
    return mod


def norm_series(v, k, ctx):
    r"""Per-context-window normalized patches of one series, ``(n, n_patch, k)``."""
    win = ctx + k
    n = len(v) // win
    x = v[: n * win].reshape(n, win)
    mu = x[:, :ctx].mean(axis=1, keepdims=True)
    sd = x[:, :ctx].std(axis=1, keepdims=True)
    return ((x - mu) / sd).reshape(n, -1, k)


def build(k, ctx, series, split, rng_seed=0):
    r"""Normalized windows, masked seasonal-naive targets, and a split.

    Normalization is per window from the **context only**. The naive forecast is
    the raw value one day earlier, expressed in the same window's units, so the
    MASE denominator lives in the normalized space the model is trained in.
    """
    win = ctx + k
    patches, naive, ok = [], [], []
    for v in series:
        n = len(v) // win
        x = v[: n * win].reshape(n, win)
        mu = x[:, :ctx].mean(axis=1, keepdims=True)
        sd = x[:, :ctx].std(axis=1, keepdims=True)
        patches.append(((x - mu) / sd).reshape(n, -1, k))
        starts = (np.arange(n) * win + ctx)[:, None] + np.arange(k)[None, :]
        idx = starts - SEASONAL
        valid = idx >= 0
        naive.append((v[np.clip(idx, 0, None)] - mu) / sd)
        ok.append(valid.astype(np.float32))
    patches = np.concatenate(patches)
    naive = np.concatenate(naive).reshape(len(patches), -1)
    ok = np.concatenate(ok).reshape(len(patches), -1)
    n = len(patches)
    if split == "random":
        idx = np.random.default_rng(rng_seed).permutation(n)
        cut = n // 2
    else:  # "blocked": the first 80% of the windows, in time order, is training
        idx = np.arange(n)
        cut = int(0.8 * n)
    return patches, naive, ok, idx[:cut], idx[cut:]


def band_bound(patches, bands, tr, ev, k=K):
    r"""Per-band model-free predictability, and its retention equivalent.

    For each band, a ridge from the *context's own coefficients at that band* to
    the *target patch's coefficient at that band*, fitted on training windows and
    scored on held-out ones -- T1's ``P(b)``. The retention equivalent is
    ``sqrt(P)``: under a pooled-RMS convention the amplitude ratio of the optimal
    linear predictor is exactly the square root of the explained variance
    fraction (T2 measured this relation on this corpus: ``r_ms / sqrt(P)`` =
    0.998 / 0.994).

    It is a **baseline predictor, not a strict upper bound**: it is linear in the
    band's own history and sees no other band. A model below it is leaving money
    on the table; a model above it is using more than that ridge can.
    """
    x = torch.from_numpy(np.ascontiguousarray(patches, dtype=np.float32))
    coeffs = np.stack([_matched_np(x, b, k) for b in bands], axis=0)
    out = {}
    for j, b in enumerate(bands):
        f = coeffs[j][:, :-1].reshape(len(patches), -1)  # the context only
        z = coeffs[j][:, -1]
        lam = _pick_lambda(f, z, tr)
        out[str(b)] = _p_band(f[tr], z[tr], f[ev], z[ev], lam)
    return out


def _pick_lambda(f, z, tr, lambdas=(1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    rng = np.random.default_rng(0)
    order = rng.permutation(len(tr))
    a = tr[order[: len(tr) // 2]]
    b = tr[order[len(tr) // 2 :]]
    return max((_p_band(f[a], z[a], f[b], z[b], t), t) for t in lambdas)[1]


def _p_band(f_tr, z_tr, f_ev, z_ev, lam):
    mu, m = f_tr.mean(axis=0), z_tr.mean()
    fc, zc = f_tr - mu, z_tr - m
    gram = fc.conj().T @ fc
    d = gram.shape[0]
    gram.flat[:: d + 1] += lam * np.trace(gram).real / d
    w = np.linalg.solve(gram, fc.conj().T @ zc)
    pred = f_ev @ w + (m - mu @ w)
    var = np.mean(np.abs(z_ev - z_ev.mean()) ** 2)
    return float(1 - np.mean(np.abs(pred - z_ev) ** 2) / var)


def band_power(patches, b, k):
    x = torch.from_numpy(np.ascontiguousarray(patches, dtype=np.float32))
    z = torch.from_numpy(np.stack([_matched_np(x, bb, k) for bb in b]))
    return (z.abs() ** 2).mean(dim=(1, 2)).numpy() / 2


def _matched_np(x, b, k):
    t = torch.arange(k, dtype=torch.float32)
    w = torch.exp(-2j * np.pi * b * t / k)
    return ((2 / k) * (x.to(torch.complex64) * w).sum(-1)).numpy()


@torch.no_grad()
def oracle_row(model, x_fit, x_ev, bands, k):
    r"""Ridge readout fitted on training windows, read out on held-out ones.

    Returns per-band retention, per-band residual power, and the readout's own
    mean-square error over the whole target patch. The residual is
    ``mean |Z_pred - Z_true|^2 / 2`` per band, which -- unlike ``r`` -- stays
    meaningful when a band is dominated by the emission floor (``r > 1`` happens
    on real data at the weakest bands and carries no information there).
    """
    _, _, z_fit = model(x_fit)
    p_fit = model.pat(x_fit)
    _, _, z_ev = model(x_ev)
    p_ev = model.pat(x_ev)
    true = p_ev[:, 1:]
    fits = {b: fit_oracle(z_fit, p_fit, z_ev, p_ev, b, k) for b in bands}
    pred = fits[bands[0]]["pred"]
    mse = float(((pred - true.reshape(pred.shape)) ** 2).mean())
    res = {}
    for b in bands:
        rb = band_component(fits[b]["pred"], b, k) - band_component(true, b, k)
        res[b] = float((rb**2).mean() / 2)
    return {b: fits[b]["retention"] for b in bands}, res, mse


@torch.no_grad()
def evaluate(model, held, held_naive, held_ok, fit_x, bands):
    model.eval()
    pred, _, _ = model(held)
    patches = model.pat(held)
    target = patches[:, 1:]
    f_hat = pred[:, -2]
    true = target[:, -1]
    mse = float(F.mse_loss(f_hat, true))
    mae = float(F.l1_loss(f_hat, true))
    tn = torch.from_numpy(held_naive)
    w = torch.from_numpy(held_ok)
    naive_mae = float((torch.abs(tn - true) * w).sum() / w.sum())
    model_mae_m = float((torch.abs(f_hat - true) * w).sum() / w.sum())
    row = {
        "mse": mse,
        "mae": mae,
        "mase": model_mae_m / naive_mae if naive_mae else float("nan"),
        "baseline_persistence_mae": float(F.l1_loss(f_hat, patches[:, -2])),
        "baseline_seasonal_mae": naive_mae,
        "var_explained": 1 - mse / float(target.var(unbiased=False)),
    }
    orc, res_orc, mse_orc = oracle_row(model, fit_x, held, bands, K)
    row["mse_oracle"] = mse_orc
    row["headroom"] = (mse - mse_orc) / mse
    for b in bands:
        rb = band_component(f_hat, b, K) - band_component(true, b, K)
        zp, zt = matched_amp(f_hat, b, K), matched_amp(true, b, K)
        row[f"rms@{b}"] = float(
            ((zp.abs() ** 2).mean() / (zt.abs() ** 2).mean()) ** 0.5
        )
        row[f"r@{b}"] = float(retention(f_hat, true, b, K).mean())
        row[f"dk@{b}"] = float(delta_k(f_hat, true, b, K).mean())
        row[f"r_oracle@{b}"] = orc[b]
        row[f"res@{b}"] = float((rb**2).mean() / 2)
        row[f"res_oracle@{b}"] = res_orc[b]
    model.train()
    return row


def train(data, tr, ev, seed, args):
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=K,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_attn_heads=HEADS,
    )
    patches, naive, ok = data[0], data[1], data[2]
    x_tr = torch.from_numpy(patches[tr].reshape(len(tr), -1).astype(np.float32))
    held = torch.from_numpy(patches[ev].reshape(len(ev), -1).astype(np.float32))
    fit_x = torch.from_numpy(
        patches[tr[:ORACLE_FIT]]
        .reshape(min(ORACLE_FIT, len(tr)), -1)
        .astype(np.float32)
    )
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    marks = []
    for step in range(1, args.steps + 1):
        x = x_tr[torch.randint(len(x_tr), (BATCH,), generator=gen)]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        due = (step <= args.ckpt_fine_until and step % args.ckpt_fine == 0) or (
            step % args.ckpt == 0
        )
        if due:
            row = evaluate(model, held, naive[ev], ok[ev], fit_x, BANDS)
            row["step"] = step
            marks.append(row)
            print(
                f"      step {step:>6}  mse {row['mse']:.4f}  mase "
                f"{row['mase']:.4f}  var expl {row['var_explained']:+.4f}",
                flush=True,
            )
    return marks


def headroom(last, power):
    r"""Fraction of the error a perfect readout on the frozen representation removes.

    ``(mse - mse_oracle) / mse`` -- no retention involved, so it stays valid at
    the bands where ``r > 1`` (the emission floor) would corrupt a
    ``(1 - r)^2``-based version. Per-band residuals are reported beside it to
    show where the difference lives, but they are descriptions, not a partition:
    matched filters at integer bands do not tile the spectrum.
    """
    per_band = {
        str(b): float((last[f"res@{b}"] - last[f"res_oracle@{b}"]) / last["mse"])
        for b, p in zip(BANDS, power)
    }
    return {
        "fraction_of_mse": float(last["headroom"]),
        "mse_oracle": float(last["mse_oracle"]),
        "per_band_fraction": per_band,
    }


def mean_marks(per_seed):
    steps = [m["step"] for m in per_seed[0]]
    keys = [k for k in per_seed[0][0] if isinstance(per_seed[0][0][k], float)]
    out = []
    for i, s in enumerate(steps):
        row = {"step": s}
        for k in keys:
            vals = [p[i][k] for p in per_seed]
            row[k] = float(np.mean(vals))
            row[k + "_sd"] = float(np.std(vals))
        out.append(row)
    return out


def table(head, rows):
    line = " ".join(head)
    print(line)
    print("  " + "-" * (len(line) - 2))
    for r in rows:
        print(" ".join(r))


def print_summary(rec):
    marks = rec["marks"]
    power = np.array(rec["power"])
    share = power / power.sum()
    print(f"\n== baseline, split = {rec['split']} ==")
    print(
        f"  windows {rec['n_windows']} ({rec['n_train']} train / {rec['n_eval']} eval)"
    )
    print("  MASE over steps (1.0 = seasonal naive; below 1 means beating it):")
    table(
        [
            f"  {'step':>6}",
            f"{'mse':>9}",
            f"{'mae':>9}",
            f"{'mase':>8}",
            f"{'persist mae':>12}",
            f"{'seasonal mae':>13}",
            f"{'var expl':>9}",
        ],
        [
            [
                f"  {int(m['step']):>6}",
                f"{m['mse']:>9.4f}",
                f"{m['mae']:>9.4f}",
                f"{m['mase']:>8.4f}",
                f"{m['baseline_persistence_mae']:>12.4f}",
                f"{m['baseline_seasonal_mae']:>13.4f}",
                f"{m['var_explained']:>+9.4f}",
            ]
            for m in marks
            if m["step"] in MARK_STEPS
        ],
    )
    last = marks[-1]
    print("\n  per band at the final checkpoint")
    table(
        [
            f"  {'b':>4}",
            f"{'period_h':>8}",
            f"{'share':>9}",
            f"{'r':>8}",
            f"{'r_sd':>7}",
            f"{'oracle':>8}",
            f"{'oracle-r':>9}",
            f"{'dk':>8}",
            f"{'headroom':>10}",
            f"{'h/MSE':>8}",
        ],
        [
            [
                f"  {b:>4.0f}",
                f"{K / b * 0.5:>8.2f}",
                f"{s:>9.5f}",
                f"{last[f'r@{b}']:>8.4f}",
                f"{last[f'r@{b}_sd']:>7.4f}",
                f"{last[f'r_oracle@{b}']:>8.4f}",
                f"{last[f'r_oracle@{b}'] - last[f'r@{b}']:>+9.4f}",
                f"{last[f'dk@{b}']:>8.4f}",
                f"{rec['headroom']['per_band_fraction'][str(b)]:>10.3f}",
                f"{100 * rec['headroom']['per_band_fraction'][str(b)]:>7.2f}%",
            ]
            for b, s in zip(BANDS, share)
        ],
    )
    h = rec["headroom"]
    print(
        f"  headroom = {100 * h['fraction_of_mse']:.2f}% of the final MSE "
        f"({last['mse']:.4f} against the oracle's {h['mse_oracle']:.4f})"
    )


def print_bound(rec):
    r"""The model-free bound against the trained model, both in power units.

    ``res`` is the model's per-band residual ``mean|Z_pred - Z_true|^2 / 2``;
    ``floor`` is what an optimal *linear* predictor from the band's own history
    would leave, ``(1 - P(b)) * P_b``. A band where ``res`` sits well above
    ``floor`` is one where the model is leaving retrievable signal behind.
    """
    last = rec["marks"][-1]
    power = np.array(rec["power"])
    floor = {b: (1 - rec["bound"][str(b)]) * p for b, p in zip(BANDS, power)}
    rms_gap = {b: last[f"rms@{b}"] - rec["bound"][str(b)] ** 0.5 for b in BANDS}
    ratio = {b: last[f"res@{b}"] / max(floor[b], 1e-12) for b in BANDS}
    print("\n  model-free bound vs the trained model (pooled-RMS retention)")
    table(
        [
            f"  {'b':>4}",
            f"{'period_h':>8}",
            f"{'share':>9}",
            f"{'P(b)':>7}",
            f"{'sqrt(P)':>8}",
            f"{'r_ms':>8}",
            f"{'r_ms-P^.5':>10}",
            f"{'res':>10}",
            f"{'floor':>10}",
            f"{'res/floor':>10}",
        ],
        [
            [
                f"  {b:>4.0f}",
                f"{K / b * 0.5:>8.2f}",
                f"{p / power.sum():>9.5f}",
                f"{rec['bound'][str(b)]:>7.3f}",
                f"{rec['bound'][str(b)] ** 0.5:>8.3f}",
                f"{last[f'rms@{b}']:>8.3f}",
                f"{rms_gap[b]:>+10.3f}",
                f"{last[f'res@{b}']:>10.2e}",
                f"{floor[b]:>10.2e}",
                f"{ratio[b]:>10.2f}",
            ]
            for b, p in zip(BANDS, power)
        ],
    )
    print(
        f"  null control (white noise through the same pipeline, median P): "
        f"{rec['bound_null_median']:+.3f}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", nargs="+", default=["random", "blocked"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--ckpt", type=int, default=CKPT)
    ap.add_argument("--ckpt-fine", type=int, default=CKPT_FINE)
    ap.add_argument("--ckpt-fine-until", type=int, default=CKPT_FINE_UNTIL)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    prof = load_profile()
    pred = prof.load_pred()
    _, raw = pred.load_monash(pred.DATASET)
    series = [v for _, _, v in raw]
    print(f"corpus: {len(series)} series, {sum(len(v) for v in series)} samples")

    out = {"config": vars(args), "runs": {}}
    for split in args.splits:
        data = build(K, CTX, series, split)
        patches, naive, ok, tr, ev = data
        power = band_power(patches, BANDS, K)
        bound = band_bound(patches, BANDS, tr, ev)
        noise = np.random.default_rng(0).standard_normal((5, len(series[0])))
        n_patches = np.concatenate([norm_series(v, K, CTX) for v in noise])
        n_half = len(n_patches) // 2
        null = band_bound(
            n_patches,
            BANDS,
            np.arange(n_half),
            np.arange(n_half, len(n_patches)),
        )
        print(
            f"\nsplit {split}: {len(patches)} windows, {len(tr)} train / {len(ev)} eval"
        )
        print(
            "  model-free bound P(b), and sqrt(P) in retention units: "
            + " ".join(
                f"b{b:.0f}:{bound[str(b)]:.3f}/{bound[str(b)] ** 0.5:.3f}"
                for b in BANDS[:6]
            )
        )
        print(
            f"  null control (white noise, same pipeline): median "
            f"{float(np.median(list(null.values()))):+.3f}"
        )
        per_seed = []
        for seed in args.seeds:
            t0 = time.time()
            print(f"    seed {seed}", flush=True)
            per_seed.append(train(data, tr, ev, seed, args))
            print(f"      [{time.time() - t0:.0f}s]", flush=True)
        rec = {
            "split": split,
            "n_windows": int(len(patches)),
            "n_train": int(len(tr)),
            "n_eval": int(len(ev)),
            "power": [float(v) for v in power],
            "bound": bound,
            "bound_null_median": float(np.median(list(null.values()))),
            "marks": mean_marks(per_seed),
        }
        rec["headroom"] = headroom(rec["marks"][-1], power)
        out["runs"][split] = rec
        print_summary(rec)
        print_bound(rec)

    RUNS.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    path = RUNS / f"baseline{tag}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
