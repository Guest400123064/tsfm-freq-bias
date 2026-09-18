r"""P13: free spectral shapes -- is loss-share a sufficient predictor of retention?

Why this run exists. P12's alpha ladder showed one curve r = G(share) describing
five power-law spectra, but it has a built-in limitation: in a power-law family
the share is *perfectly collinear with band identity* (b1 is always the loudest,
b16 always the quietest). Only cross-arm comparisons at matched share can pull
those apart, and they left a residual in the deepest octave.

This run breaks the collinearity directly: **random free-form spectra**, where
each band's share is drawn independently, plus the mirrored (violet) power laws
where the energy sits at high frequency. The question becomes exact:

    at a fixed compute budget, does a band's share of the loss predict its
    retention irrespective of which frequency it is?

Design. Sixteen carriers b = 1..16 fill the k = 32 patch, fully coherent, no
noise, total power 0.5 per window, homogeneous corpora (one profile per arm),
each probed clean with its own profile.

  v1, v2    power laws with alpha = -1, -2 -- violet, high bands loud
  r1..r5    random log-normal profiles: log P_b = sigma * z_b, sigma = 1.2
  bump1..2  a random centre and width, everything else floored (a loud band
            lands in a *random* place, which a power law can never do)
  flat      the alpha = 0 anchor

Every profile is floored at 2e-3 of its own maximum, so no band's probe
amplitude is zero (a zero-amplitude band makes retention a division by zero --
the y4 display artifact from P11).

Pre-registered decision, written before the run into this experiment's
README.md:

  1. **capacity** -- the frozen-latent oracle reads >= 0.9 at every band of
     every arm (P12 showed 16 bands need hidden 64);
  2. **share is sufficient** -- the isotonic (monotone) fit of r on log share
     over all pooled points has **R^2 >= 0.9**;
  3. **frequency adds nothing** -- in equal-count share bins, the correlation
     between r and band index is **|corr| <= 0.5** in every bin with >= 8
     points, and the within-bin spread of r is <= 0.15 (the same tolerance P12
     used for its collapse).

Verdicts: gate/oracle failure -> inconclusive or capacity; (2) failing -> share
is not sufficient; (2) holding with (3) failing -> share plus a residual
frequency dependence; both holding -> **share is the sufficient statistic**.

b16 (Nyquist) is excluded from every statistic, as in P12: a real tone there is
its own negative-frequency image, so the matched filter double-counts and its
phase is a bare sign.

Run from the repo root:
``.venv/bin/python experiments/6_free_shape/scripts/free_shape.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fbias.model import SimTFM
from fbias.probes import delta_k, fit_oracle, matched_amp, retention

CTX, PATCH = 512, 32
HIDDEN, LAYERS, HEADS = 64, 2, 4
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
BANDS = tuple(float(b) for b in range(1, 17))
WELL = BANDS[:-1]  # b16 is Nyquist: excluded from every statistic
FREQS = tuple(b * CTX / PATCH for b in BANDS)
TOTAL_POWER = 0.5
MIN_SHARE = 0.01  # every band carries at least 1% of the window power
# The floor is set by the readout, not by taste: P7 measured an absolute
# emitted-amplitude floor of ~0.015 at a band with no content, so at a band
# whose true amplitude is sqrt(2 * 0.5 * share) the reading is floor-dominated
# below share ~0.25%. A 1% share floor keeps every included band four times
# above that. It also truncates the violet power laws at their bottom end.
PROBE_SEED = 1607
SEED_STRIDE = 1000
GATE = 0.5
ORACLE_TOL = 0.9
R2_TOL = 0.9
CORR_TOL = 0.5
SPREAD_TOL = 0.15
N_BINS = 10
RUNS = Path(__file__).resolve().parents[1] / "runs"


def _norm(w):
    r"""Raise every share to at least ``MIN_SHARE`` and normalise to 0.5."""
    s = sum(w)
    w = [max(x, MIN_SHARE * s) for x in w]
    s2 = sum(w)
    return tuple(TOTAL_POWER * x / s2 for x in w)


def power_law(alpha):
    return _norm([b ** (-alpha) for b in BANDS])


def lognormal(seed, sigma=1.2):
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(len(BANDS), generator=gen)
    return _norm([math.exp(sigma * float(v)) for v in z])


def bump(seed):
    gen = torch.Generator().manual_seed(seed)
    centre = int(torch.randint(1, 17, (1,), generator=gen))
    width = float(torch.randint(2, 5, (1,), generator=gen))
    return _norm([math.exp(-0.5 * ((b - centre) / width) ** 2) for b in BANDS])


PROFILES = {
    "v1": power_law(-1.0),
    "v2": power_law(-2.0),
    "r1": lognormal(11),
    "r2": lognormal(23),
    "r3": lognormal(37),
    "r4": lognormal(53),
    "r5": lognormal(71),
    "bump1": bump(101),
    "bump2": bump(211),
    "flat": power_law(0.0),
}


def synth(allocs, n, gen):
    r"""Windows from per-window allocations, ``(n, ctx + k)``."""
    t = torch.arange(CTX + PATCH, dtype=torch.float32)
    freqs = torch.tensor(FREQS, dtype=torch.float32)
    base = 2 * math.pi * freqs[:, None] * t[None, :] / CTX
    phase = 2 * math.pi * torch.rand((n, len(BANDS)), generator=gen)
    amp = torch.sqrt(2.0 * allocs.to(torch.float32))
    return (amp[:, :, None] * torch.cos(base[None, :, :] + phase[:, :, None])).sum(1)


def fixed_windows(alloc, n, gen):
    return synth(torch.tensor([alloc], dtype=torch.float64).expand(n, -1), n, gen)


def corpus(arm, seed):
    return fixed_windows(
        PROFILES[arm], N_PER, torch.Generator().manual_seed(seed * SEED_STRIDE + 71)
    )


def probe(arm):
    return fixed_windows(
        PROFILES[arm], N_PER, torch.Generator().manual_seed(PROBE_SEED)
    )


def held_out(arm, seed):
    return fixed_windows(
        PROFILES[arm], N_PER, torch.Generator().manual_seed(seed * SEED_STRIDE + 307)
    )


@torch.no_grad()
def band_power(x, b):
    z = matched_amp(x.unfold(1, PATCH, PATCH), b, PATCH)
    return 0.5 * float(z.abs().square().mean())


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


def generator_check():
    out = {}
    for arm in PROFILES:
        prof = PROFILES[arm]
        dev, rms = [], []
        for seed in SEEDS:
            x = corpus(arm, seed)
            rms.append(float(x.square().mean().sqrt()))
            realised = [band_power(x, b) for b in WELL]
            dev.append(max(abs(r - p) for r, p in zip(realised, prof[:-1])))
        prb = probe(arm)
        shares = [p / TOTAL_POWER for p in prof[:-1]]
        out[arm] = {
            "min_share": min(shares),
            "max_share": max(shares),
            "max_band_of_max_share": BANDS[int(max(range(16), key=lambda i: prof[i]))],
            "rms": mean_std(rms),
            "max_marginal_dev": mean_std(dev),
            "probe_rms": float(prb.square().mean().sqrt()),
        }
    return out


def train(seed, arm, steps, hidden=HIDDEN, layers=LAYERS):
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=hidden,
        num_layers=layers,
        num_attn_heads=HEADS,
    )
    data = corpus(arm, seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        x = data[torch.randint(N_PER, (BATCH,), generator=gen)]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, float(loss)


@torch.no_grad()
def evaluate(model, held, prb, train_loss):
    model.eval()
    pred, _, z_held = model(held)
    patches = model.pat(held)
    target = patches[:, 1:]
    mse = float(F.mse_loss(pred[:, :-1], target))
    var = float(target.var(unbiased=False))
    row = {"train_loss": train_loss, "held_var": var, "var_explained": 1 - mse / var}
    for b in BANDS:
        row[f"rm@{b}"] = float(retention(pred[:, -2], target[:, -1], b, PATCH).mean())
    p_hat, _, z = model(prb)
    true = model.pat(prb)
    f_hat = p_hat[:, -2]
    for b in BANDS:
        row[f"r@{b}"] = float(retention(f_hat, true[:, -1], b, PATCH).mean())
        row[f"dk@{b}"] = float(delta_k(f_hat, true[:, -1], b, PATCH).mean())
        row[f"r_oracle@{b}"] = fit_oracle(z_held, patches, z, true, b, PATCH)[
            "retention"
        ]
    return row


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: mean_std([r[k] for r in rows]) for k in keys}


def isotonic(x, y):
    r"""Least-squares monotone (non-decreasing) fit of ``y`` on sorted ``x``.

    Pool-adjacent-violators, implemented with a stack of blocks; returns the
    fitted values in the original (sorted) order.
    """
    order = sorted(range(len(x)), key=lambda i: x[i])
    vals = [y[i] for i in order]
    idx = [1] * len(vals)
    means = vals[:]
    i = 0
    while i < len(means) - 1:
        if means[i] > means[i + 1]:
            w0, w1 = idx[i], idx[i + 1]
            means[i] = (means[i] * w0 + means[i + 1] * w1) / (w0 + w1)
            idx[i] = w0 + w1
            del means[i + 1]
            del idx[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    fitted = []
    for m, c in zip(means, idx):
        fitted.extend([m] * c)
    return order, fitted


def pooled(rec):
    r"""``(share, r, band, arm)`` for every well-conditioned band of every arm."""
    pts = []
    for arm, a in rec["arms"].items():
        prof = PROFILES[arm]
        for j, b in enumerate(WELL):
            pts.append((prof[j] / TOTAL_POWER, a["mean_std"][f"r@{b}"][0], b, arm))
    return pts


def corr(v, w):
    v = torch.tensor(v, dtype=torch.float64)
    w = torch.tensor(w, dtype=torch.float64)
    v = v - v.mean()
    w = w - w.mean()
    if float(v.norm()) == 0.0 or float(w.norm()) == 0.0:
        return 0.0
    return float((v * w).sum() / (v.norm() * w.norm()))


def share_bins(pts, n_bins=N_BINS):
    r"""Equal-count bins over log share: spread, arms, and corr(r, band)."""
    order = sorted(pts, key=lambda p: p[0])
    bins = []
    for k in range(n_bins):
        lo = k * len(order) // n_bins
        hi = (k + 1) * len(order) // n_bins
        chunk = order[lo:hi]
        if not chunk:
            continue
        rs = [c[1] for c in chunk]
        bins.append(
            {
                "share_lo": chunk[0][0],
                "share_hi": chunk[-1][0],
                "n": len(chunk),
                "n_arms": len({c[3] for c in chunk}),
                "share_mean": sum(c[0] for c in chunk) / len(chunk),
                "r_mean": sum(rs) / len(rs),
                "r_spread": max(rs) - min(rs),
                "corr_r_band": corr(rs, [c[2] for c in chunk]),
            }
        )
    return bins


def decide(rec):
    arms = rec["arms"]
    oracle_min = {
        a: min(v["mean_std"][f"r_oracle@{b}"][0] for b in WELL) for a, v in arms.items()
    }
    pts = pooled(rec)
    order, fitted = isotonic([p[0] for p in pts], [p[1] for p in pts])
    y = [pts[i][1] for i in order]
    ybar = sum(y) / len(y)
    ss_res = sum((yi - fi) ** 2 for yi, fi in zip(y, fitted))
    ss_tot = sum((yi - ybar) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot
    bins = share_bins(pts)
    big = [b for b in bins if b["n"] >= 8]
    worst_corr = max((abs(b["corr_r_band"]) for b in big), default=0.0)
    worst_spread = max((b["r_spread"] for b in big), default=0.0)
    unfitted = [a for a in arms if not arms[a]["gate_passed"]]
    bad_oracle = [a for a, o in oracle_min.items() if o < ORACLE_TOL]
    if unfitted:
        verdict = "inconclusive: arm(s) failed the fit gate"
    elif bad_oracle:
        verdict = "capacity: the oracle is short in " + ", ".join(sorted(bad_oracle))
    elif r2 < R2_TOL:
        verdict = "share is not sufficient: a monotone curve in share leaves too much"
    elif worst_corr > CORR_TOL or worst_spread > SPREAD_TOL:
        verdict = "share plus a residual frequency dependence"
    else:
        verdict = (
            "share is the sufficient statistic: one monotone curve in loss-share "
            "predicts retention irrespective of frequency"
        )
    return {
        "oracle_min": oracle_min,
        "share_only_r2": r2,
        "n_points": len(pts),
        "bins": bins,
        "worst_bin_corr_r_band": worst_corr,
        "worst_bin_r_spread": worst_spread,
        "unfitted": unfitted,
        "bad_oracle": bad_oracle,
        "verdict": verdict,
    }


def table(head, rows):
    line = " ".join(head)
    print(line)
    print("  " + "-" * (len(line) - 2))
    for r in rows:
        print(" ".join(r))


def ms(pair, width=10, digits=4):
    return f"{pair[0]:.{digits}f}±{pair[1]:.{digits}f}".rjust(width)


def build(args):
    t0 = time.time()
    arms = {a: PROFILES[a] for a in PROFILES}
    if args.arms:
        missing = set(args.arms) - set(PROFILES)
        assert not missing, f"unknown arms {sorted(missing)}; have {sorted(PROFILES)}"
        arms = {a: v for a, v in arms.items() if a in args.arms}
    seeds = tuple(args.seeds) if args.seeds else SEEDS

    print(
        f"== free_shape (P13): {len(arms)} profiles x {len(seeds)} seeds ==", flush=True
    )
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {args.hidden} / {args.layers}L / "
        f"{HEADS} heads | SGD lr {LR} / batch {BATCH} / {args.steps} steps / cpu",
        flush=True,
    )
    print(
        f"bands b = 1..16 (f = {FREQS[0]:g}..{FREQS[-1]:g}); b16 excluded (Nyquist)",
        flush=True,
    )
    print("\ngenerator check (no training yet)", flush=True)
    gen = generator_check()

    rows_by_arm, walls = {}, []
    for arm in arms:
        rows = []
        for seed in seeds:
            t = time.time()
            model, loss = train(seed, arm, args.steps, args.hidden, args.layers)
            row = evaluate(model, held_out(arm, seed), probe(arm), loss)
            row["seed"] = seed
            row["wall"] = time.time() - t
            walls.append(row["wall"])
            rows.append(row)
            r_lo = row[f"r@{WELL[0]}"]
            r_hi = row[f"r@{WELL[-1]}"]
            print(
                f"  {arm:<6} seed {seed}  loss {loss:.2e}  var expl "
                f"{row['var_explained']:+.5f}  r@1 {r_lo:.3f}  r@15 {r_hi:.3f}  "
                f"[{row['wall']:.0f}s]",
                flush=True,
            )
        rows_by_arm[arm] = rows

    rec = {
        "config": {
            "ctx": CTX,
            "patch": PATCH,
            "hidden": args.hidden,
            "layers": args.layers,
            "heads": HEADS,
            "steps": args.steps,
            "batch": BATCH,
            "lr": LR,
            "n_per": N_PER,
            "seeds": list(seeds),
            "device": "cpu",
            "bands": list(BANDS),
            "freqs": list(FREQS),
            "total_power": TOTAL_POWER,
            "profiles": {a: list(PROFILES[a]) for a in arms},
            "probe_seed": PROBE_SEED,
            "gate_fraction": GATE,
            "oracle_tol": ORACLE_TOL,
            "r2_tol": R2_TOL,
            "corr_tol": CORR_TOL,
            "spread_tol": SPREAD_TOL,
            "n_bins": N_BINS,
            "readout": "pred[:, -2] vs pat(x)[:, -1]",
        },
        "generator": gen,
        "arms": {},
        "wall": 0.0,
    }
    for arm in arms:
        rows = rows_by_arm[arm]
        s = summarize(rows)
        rec["arms"][arm] = {
            "gate_passed": s["var_explained"][0] > GATE,
            "per_seed": rows,
            "mean_std": s,
        }
    rec["wall_train_mean"] = sum(walls) / len(walls) if walls else 0.0
    rec["wall"] = time.time() - t0
    rec["decision"] = decide(rec)
    return rec


def print_generator(rec):
    gen = rec["generator"]
    print("\ngenerator check: profiles, marginals, probe shape")
    table(
        [
            f"  {'arm':<6}",
            f"{'min share':>10}",
            f"{'max share':>10}",
            f"{'b of max':>9}",
            f"{'corpus rms':>11}",
            f"{'max marg dev':>13}",
            f"{'probe rms':>10}",
        ],
        [
            [
                f"  {arm:<6}",
                f"{gen[arm]['min_share']:>10.5f}",
                f"{gen[arm]['max_share']:>10.4f}",
                f"{gen[arm]['max_band_of_max_share']:>9.0f}",
                f"{gen[arm]['rms'][0]:>11.5f}",
                f"{gen[arm]['max_marginal_dev'][0]:>13.5f}",
                f"{gen[arm]['probe_rms']:>10.5f}",
            ]
            for arm in gen
        ],
    )
    print(
        "  every profile is floored at a 1% share, so no probe amplitude is zero;\n"
        "  'b of max' is the band carrying the largest share, and\n"
        "  it is what breaks the alpha ladder's share-vs-band collinearity."
    )


def print_bands(rec):
    print("\nprobe r by band, per profile (share on the left, per arm)")
    table(
        [f"  {'b':>3}", *[f"{'share':>8} {a:>8}" for a in rec["arms"]]],
        [
            [
                f"  {int(b):>3}",
                *[
                    f"{PROFILES[a][j] / TOTAL_POWER:>8.4f} "
                    f"{rec['arms'][a]['mean_std'][f'r@{b}'][0]:>8.3f}"
                    for a in rec["arms"]
                ],
            ]
            for j, b in enumerate(WELL)
        ],
    )


def print_decision(rec):
    d = rec["decision"]
    print("\npre-registered checks")
    print(f"  1. oracle minimum over bands (>= {ORACLE_TOL} required):")
    for a, o in sorted(d["oracle_min"].items(), key=lambda kv: kv[1]):
        flag = "" if o >= ORACLE_TOL else "   <-- SHORT"
        print(f"       {a:<6} {o:.4f}{flag}")
    print(
        f"  2. share-only monotone fit R^2 = {d['share_only_r2']:.4f} "
        f"(>= {R2_TOL} required, {d['n_points']} points)"
    )
    print(
        f"  3. frequency adds nothing: worst |corr(r, band)| within bins = "
        f"{d['worst_bin_corr_r_band']:.3f} (tol {CORR_TOL}), worst within-bin "
        f"spread = {d['worst_bin_r_spread']:.3f} (tol {SPREAD_TOL})"
    )
    print("\n  share bins")
    table(
        [
            f"  {'share range':>20}",
            f"{'n':>4}",
            f"{'arms':>5}",
            f"{'r mean':>8}",
            f"{'r spread':>9}",
            f"{'corr(r,b)':>10}",
        ],
        [
            [
                f"  {b['share_lo']:.5f}-{b['share_hi']:.5f}".rjust(20),
                f"{b['n']:>4}",
                f"{b['n_arms']:>5}",
                f"{b['r_mean']:>8.3f}",
                f"{b['r_spread']:>9.3f}",
                f"{b['corr_r_band']:>10.3f}",
            ]
            for b in d["bins"]
        ],
    )
    print(
        "  arms failing the fit gate: "
        + (", ".join(d["unfitted"]) if d["unfitted"] else "none")
    )
    print(f"  -> {d['verdict']}")


def report(rec):
    print_generator(rec)
    print_bands(rec)
    print_decision(rec)


def out_path(args):
    tag = f"_{args.tag}" if args.tag else ""
    return RUNS / f"free_shape{tag}.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=None, help="subset of profile ids")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="subset of seeds")
    p.add_argument("--steps", type=int, default=STEPS, help="training steps")
    p.add_argument("--hidden", type=int, default=HIDDEN, help="hidden size")
    p.add_argument("--layers", type=int, default=LAYERS, help="transformer layers")
    p.add_argument("--tag", default="", help="suffix for the output JSON")
    p.add_argument("--summarize", action="store_true", help="reprint a stored run")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    path = out_path(args)
    if args.summarize:
        report(json.loads(path.read_text()))
        return 0
    rec = build(args)
    report(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")
    print(
        f"\nwrote {path}\ntotal wall {rec['wall']:.1f}s "
        f"({rec['wall_train_mean']:.0f}s per training)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
