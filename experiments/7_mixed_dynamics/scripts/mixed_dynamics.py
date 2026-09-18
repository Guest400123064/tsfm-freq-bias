r"""P14: learning dynamics on one mixed corpus -- is speed set by the band's power?

Why this run exists. P8, P12 and P13 all read retention *at fixed budgets*. The
schedule claim is a statement about dynamics, so this run makes it dynamic: one
corpus, many bands, and a retention curve per band sampled every 1 000 steps.
The claim becomes a two-part prediction:

  1. **ordering** -- the time a band needs to reach a retention threshold is
     ordered by that band's power share in the corpus (higher power -> earlier);
  2. **time collapse** -- rescaling each band's curve by its share, r_b(t) as a
     function of u = t * share_b, puts every band on ONE curve. That is the
     schedule law in its strongest form: learning time is inversely proportional
     to share, with a single universal curve.

Corpus. Closer to a TSFM setting than any previous run: **one corpus, mixed
spectral shapes**. Each window draws its own spectrum, log P_b = log w_b +
sigma * z_b with w_b proportional to b**(-alpha), so every window has a
different shape while the corpus marginal stays red. Per-window total power is
0.5 and every band's phase is drawn independently per window.

  Why not average several fixed profiles: the marginal of a convex mixture is
  flatter than its parts (ten P13 profiles average to only a 4.4x spread), which
  weakens the ordering test. Jittering around a red mean keeps the marginal red
  (15x spread over b1..b15) while the per-window shapes vary.

  Why alpha = 1 and sigma = 0.8: with alpha = 2 the weakest included band sits
  near the ~0.015 absolute emission floor measured in P7, which inflates its
  early retention and would invert the ordering. alpha = 1 puts the weakest band
  at share ~2%, four times above the floor-dominated regime.

Phases are randomised per window per band, as in every run here: a phase fixed
across windows would let the model emit a standing tone without reading the
context, turning "copy what the context shows" into a degenerate task.

Pre-registered decision, written before the run into this experiment's
README.md, at 20 000 steps with checkpoints every 1 000:

  1. **capacity** -- fit gate passed and the frozen-latent oracle >= 0.9 at the
     final checkpoint (it is expected to start low: the representation itself has
     to learn the weak bands, which is the dynamic version of P12's capacity
     finding -- reported as a curve, not a gate);
  2. **ordering** -- Spearman(log marginal share_b, t80_b) <= -0.8, where t80_b
     is the first checkpoint at which band b reaches r = 0.8 (censored at 20k).
     The sign follows the prediction: a bigger share means an *earlier* t80, so
     the correlation must be strongly negative. (The first version of this file
     wrote `>= 0.8`, which is the wrong direction; the run's stored verdict
     string was produced with the sign bug and the corrected reading is
     Spearman = -0.993, i.e. the strongest possible support.)
  3. **time collapse** -- with u = t * share_b, all points with r < 0.9 fall on
     one curve: within-bin spread of r <= 0.15. The restriction to the rising
     regime is pre-registered because bands that have not converged by 20k would
     leave the common asymptote and the collapse would fail for a reason that is
     not the schedule.

Verdicts: gate/oracle failure -> inconclusive; ordering failing -> speed is not
ordered by power; ordering holding but collapse failing -> power sets the order
but not a single time constant; both holding -> **learning speed is set by power
share through one universal curve**.

b16 (Nyquist) is excluded from every statistic, as in P12/P13.

Run from the repo root:
``.venv/bin/python experiments/7_mixed_dynamics/scripts/mixed_dynamics.py``.
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
STEPS, CKPT, BATCH, LR = 20000, 1000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
BANDS = tuple(float(b) for b in range(1, 17))
WELL = BANDS[:-1]  # b16 is Nyquist: excluded from every statistic
FREQS = tuple(b * CTX / PATCH for b in BANDS)
TOTAL_POWER = 0.5
ALPHA, SIGMA = 1.0, 0.8
PROBE_SEED = 1901
SEED_STRIDE = 1000
GATE = 0.5
ORACLE_TOL = 0.9
THRESHOLD = 0.8
SPEARMAN_TOL = 0.8
COLLAPSE_TOL = 0.15
N_BINS = 10
RUNS = Path(__file__).resolve().parents[1] / "runs"


def mean_profile():
    r"""The red mean spectrum, normalised to ``TOTAL_POWER``."""
    w = [b ** (-ALPHA) for b in BANDS]
    s = sum(w)
    return tuple(TOTAL_POWER * x / s for x in w)


MEAN = mean_profile()


def draw_window_allocs(n, gen):
    r"""Per-window per-band powers: the mean spectrum, lognormal-jittered."""
    z = torch.randn((n, len(BANDS)), generator=gen, dtype=torch.float64)
    p = torch.tensor(MEAN, dtype=torch.float64)[None, :] * torch.exp(SIGMA * z)
    return p * (TOTAL_POWER / p.sum(-1, keepdim=True))


def synth(allocs, gen):
    r"""Windows from per-window allocations, ``(n, ctx + k)``."""
    n = allocs.shape[0]
    t = torch.arange(CTX + PATCH, dtype=torch.float32)
    freqs = torch.tensor(FREQS, dtype=torch.float32)
    base = 2 * math.pi * freqs[:, None] * t[None, :] / CTX
    phase = 2 * math.pi * torch.rand((n, len(BANDS)), generator=gen)
    amp = torch.sqrt(2.0 * allocs.to(torch.float32))
    return (amp[:, :, None] * torch.cos(base[None, :, :] + phase[:, :, None])).sum(1)


def corpus(seed):
    gen_alloc = torch.Generator().manual_seed(seed * SEED_STRIDE + 17)
    gen_phase = torch.Generator().manual_seed(seed * SEED_STRIDE + 71)
    return synth(draw_window_allocs(N_PER, gen_alloc), gen_phase)


def probe():
    r"""The clean probe carries the corpus's *mean* spectrum (shape-matched)."""
    alloc = torch.tensor(MEAN, dtype=torch.float64).expand(N_PER, -1)
    return synth(alloc, torch.Generator().manual_seed(PROBE_SEED))


def held_out(seed):
    alloc = torch.tensor(MEAN, dtype=torch.float64).expand(N_PER, -1)
    return synth(alloc, torch.Generator().manual_seed(seed * SEED_STRIDE + 307))


@torch.no_grad()
def band_power(x, b):
    z = matched_amp(x.unfold(1, PATCH, PATCH), b, PATCH)
    return 0.5 * float(z.abs().square().mean())


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


def generator_check():
    r"""Measured corpus marginal (the x-axis must be measured, not assumed)."""
    acc = {"rms": [], "marginal": []}
    for seed in SEEDS:
        x = corpus(seed)
        acc["rms"].append(float(x.square().mean().sqrt()))
        acc["marginal"].append([band_power(x, b) for b in BANDS])
    marg = {
        str(b): mean_std([m[i] for m in acc["marginal"]]) for i, b in enumerate(BANDS)
    }
    prb = probe()
    return {
        "rms": mean_std(acc["rms"]),
        "marginal": marg,
        "mean_profile": list(MEAN),
        "probe_rms": float(prb.square().mean().sqrt()),
        "probe_marginal": [band_power(prb, b) for b in BANDS],
        "shares": {str(b): marg[str(b)][0] / TOTAL_POWER for b in WELL},
    }


def train(seed, steps, ckpt, hidden, layers):
    r"""Train on the mixed corpus, evaluating the probe every ``ckpt`` steps."""
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=hidden,
        num_layers=layers,
        num_attn_heads=HEADS,
    )
    data = corpus(seed)
    prb, held = probe(), held_out(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss = torch.tensor(0.0)
    marks, walls = [], []
    for step in range(1, steps + 1):
        x = data[torch.randint(N_PER, (BATCH,), generator=gen)]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % ckpt == 0:
            t0 = time.time()
            row = evaluate(model, held, prb)
            row["step"] = step
            row["train_loss"] = float(loss)
            row["eval_wall"] = time.time() - t0
            marks.append(row)
            print(
                f"    step {step:>6}  loss {float(loss):.3e}  var expl "
                f"{row['var_explained']:+.4f}  r@1 {row[f'r@{WELL[0]}']:.3f}  "
                f"r@15 {row[f'r@{WELL[-1]}']:.3f}  [{row['eval_wall']:.0f}s]",
                flush=True,
            )
    return model, marks, walls


@torch.no_grad()
def evaluate(model, held, prb):
    model.eval()
    pred, _, z_held = model(held)
    patches = model.pat(held)
    target = patches[:, 1:]
    mse = float(F.mse_loss(pred[:, :-1], target))
    var = float(target.var(unbiased=False))
    row = {"var_explained": 1 - mse / var}
    p_hat, _, z = model(prb)
    true = model.pat(prb)
    f_hat = p_hat[:, -2]
    for b in BANDS:
        row[f"r@{b}"] = float(retention(f_hat, true[:, -1], b, PATCH).mean())
        row[f"dk@{b}"] = float(delta_k(f_hat, true[:, -1], b, PATCH).mean())
        row[f"r_oracle@{b}"] = fit_oracle(z_held, patches, z, true, b, PATCH)[
            "retention"
        ]
    model.train()
    return row


def spearman(x, y):
    order_x = sorted(range(len(x)), key=lambda i: x[i])
    order_y = sorted(range(len(y)), key=lambda i: y[i])
    rx = [0.0] * len(x)
    ry = [0.0] * len(y)
    for pos, i in enumerate(order_x):
        rx[i] = float(pos)
    for pos, i in enumerate(order_y):
        ry[i] = float(pos)
    a = torch.tensor(rx, dtype=torch.float64)
    b = torch.tensor(ry, dtype=torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def curves(seed_rec):
    r"""``{band: [(step, r), ...]}`` from one seed's checkpoint list."""
    out = {}
    for b in WELL:
        out[b] = [(m["step"], m[f"r@{b}"]) for m in seed_rec]
    return out


def t80(curve, threshold=THRESHOLD, cap=None):
    r"""First checkpoint at which the curve reaches ``threshold``; ``cap`` if never."""
    for step, r in curve:
        if r >= threshold:
            return float(step)
    return float(cap if cap is not None else curve[-1][0] * 2)


def collapse(pts, tol_bins=N_BINS):
    r"""Bin the rescaled abscissa and measure the spread of ``r`` per bin."""
    pts = sorted(pts, key=lambda p: p[0])
    bins = []
    for k in range(tol_bins):
        lo = k * len(pts) // tol_bins
        hi = (k + 1) * len(pts) // tol_bins
        chunk = pts[lo:hi]
        if not chunk:
            continue
        rs = [c[1] for c in chunk]
        bins.append(
            {
                "u_lo": chunk[0][0],
                "u_hi": chunk[-1][0],
                "n": len(chunk),
                "n_bands": len({c[2] for c in chunk}),
                "r_mean": sum(rs) / len(rs),
                "r_spread": max(rs) - min(rs),
            }
        )
    big = [b for b in bins if b["n"] >= 5 and b["n_bands"] >= 2]
    return bins, max((b["r_spread"] for b in big), default=0.0)


def decide(rec):
    seeds = rec["seeds"]
    shares = {b: rec["generator"]["shares"][str(b)] for b in WELL}
    t80_by_seed = {}
    for s in seeds:
        cs = curves(s["marks"])
        t80_by_seed[s["seed"]] = {
            b: t80(cs[b], cap=rec["config"]["steps"] * 2) for b in WELL
        }
    t80_mean = {
        b: sum(t80_by_seed[k][b] for k in t80_by_seed) / len(t80_by_seed) for b in WELL
    }
    sp = spearman([math.log(shares[b]) for b in WELL], [t80_mean[b] for b in WELL])
    final = [s["marks"][-1] for s in seeds]
    oracle_min = min(min(m[f"r_oracle@{b}"] for b in WELL) for m in final)
    spearman_data = {str(b): shares[b] for b in WELL}
    gate_ok = all(m["var_explained"] > GATE for m in final)
    rising = [
        (m["step"] * shares[b], m[f"r@{b}"], b)
        for s in seeds
        for m in s["marks"]
        for b in WELL
        if m[f"r@{b}"] < 0.9
    ]
    bins, worst = collapse(rising)
    if not gate_ok or oracle_min < ORACLE_TOL:
        verdict = "inconclusive: fit gate or oracle short at the final checkpoint"
    elif sp > -SPEARMAN_TOL:
        verdict = "speed is not ordered by the band's power share"
    elif worst > COLLAPSE_TOL:
        verdict = "power sets the order but not one universal time constant"
    else:
        verdict = "learning speed is set by power share through one universal curve"
    return {
        "shares": spearman_data,
        "t80": {str(b): t80_mean[b] for b in WELL},
        "spearman_share_vs_t80": sp,
        "oracle_min_final": oracle_min,
        "gate_ok": gate_ok,
        "collapse_bins": bins,
        "collapse_worst_spread": worst,
        "n_rising_points": len(rising),
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
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    print(
        f"== mixed_dynamics (P14): {len(seeds)} seeds, {args.steps} steps ==",
        flush=True,
    )
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {args.hidden} / {args.layers}L / "
        f"{HEADS} heads | SGD lr {LR} / batch {BATCH} / checkpoint {args.ckpt} / cpu",
        flush=True,
    )
    print(
        f"corpus: per-window lognormal spectra around b**-{ALPHA}, sigma {SIGMA}",
        flush=True,
    )
    print("\ngenerator check: corpus marginal and probe shape", flush=True)
    gen = generator_check()
    shares = gen["shares"]
    spread = max(shares.values()) / min(shares.values())
    print(
        "  shares " + " ".join(f"b{int(b)}:{shares[str(b)]:.4f}" for b in WELL),
        flush=True,
    )
    print(
        f"  corpus rms {gen['rms'][0]:.5f}, probe rms {gen['probe_rms']:.5f}, "
        f"share spread {spread:.1f}x",
        flush=True,
    )

    rec = {
        "config": {
            "ctx": CTX,
            "patch": PATCH,
            "hidden": args.hidden,
            "layers": args.layers,
            "heads": HEADS,
            "steps": args.steps,
            "ckpt": args.ckpt,
            "batch": BATCH,
            "lr": LR,
            "n_per": N_PER,
            "seeds": list(seeds),
            "device": "cpu",
            "bands": list(BANDS),
            "freqs": list(FREQS),
            "alpha": ALPHA,
            "sigma": SIGMA,
            "total_power": TOTAL_POWER,
            "probe_seed": PROBE_SEED,
            "gate_fraction": GATE,
            "oracle_tol": ORACLE_TOL,
            "threshold": THRESHOLD,
            "spearman_tol": SPEARMAN_TOL,
            "collapse_tol": COLLAPSE_TOL,
            "n_bins": N_BINS,
            "readout": "pred[:, -2] vs pat(x)[:, -1]",
        },
        "generator": gen,
        "seeds": [],
        "wall": 0.0,
    }
    for seed in seeds:
        t = time.time()
        print(f"  seed {seed}", flush=True)
        _, marks, _ = train(seed, args.steps, args.ckpt, args.hidden, args.layers)
        rec["seeds"].append({"seed": seed, "marks": marks, "wall": time.time() - t})
    rec["wall"] = time.time() - t0
    rec["decision"] = decide(rec)
    return rec


def print_curves(rec):
    steps = [m["step"] for m in rec["seeds"][0]["marks"]]
    shares = rec["decision"]["shares"]
    n_seeds = len(rec["seeds"])

    def mean_r(k, b):
        return sum(s["marks"][k][f"r@{b}"] for s in rec["seeds"]) / n_seeds

    print("\nretention by band and step (mean over seeds)")
    table(
        [
            f"  {'band':>4}",
            f"{'share':>8}",
            *[f"{str(s // 1000) + 'k':>6}" for s in steps],
        ],
        [
            [
                f"  {int(b):>4}",
                f"{shares[str(b)]:>8.4f}",
                *[f"{mean_r(k, b):>6.3f}" for k in range(len(steps))],
            ]
            for b in WELL
        ],
    )


def print_decision(rec):
    d = rec["decision"]
    print("\npre-registered checks")
    print(
        f"  1. final checkpoint: fit gate {'ok' if d['gate_ok'] else 'FAILED'}, "
        f"oracle minimum {d['oracle_min_final']:.4f} (>= {ORACLE_TOL})"
    )
    print(
        f"  2. ordering: Spearman(log share, t80) = {d['spearman_share_vs_t80']:+.3f} "
        f"(<= -{SPEARMAN_TOL} required)"
    )
    print(
        f"  3. time collapse on the rising regime ({d['n_rising_points']} points): "
        f"worst within-bin spread {d['collapse_worst_spread']:.3f} "
        f"(tol {COLLAPSE_TOL})"
    )
    print("\n  t80 per band (steps, censored at 40k if never reached)")
    table(
        [f"  {'band':>4}", f"{'share':>8}", f"{'t80':>9}"],
        [
            [
                f"  {int(b):>4}",
                f"{d['shares'][str(b)]:>8.4f}",
                f"{d['t80'][str(b)]:>9.0f}",
            ]
            for b in WELL
        ],
    )
    print("\n  collapse bins (u = step x share)")
    table(
        [
            f"  {'u range':>22}",
            f"{'n':>4}",
            f"{'bands':>6}",
            f"{'r mean':>8}",
            f"{'r spread':>9}",
        ],
        [
            [
                f"  {b['u_lo']:.0f}-{b['u_hi']:.0f}".rjust(22),
                f"{b['n']:>4}",
                f"{b['n_bands']:>6}",
                f"{b['r_mean']:>8.3f}",
                f"{b['r_spread']:>9.3f}",
            ]
            for b in d["collapse_bins"]
        ],
    )
    print(f"  -> {d['verdict']}")


def report(rec):
    print_curves(rec)
    print_decision(rec)


def out_path(args):
    tag = f"_{args.tag}" if args.tag else ""
    return RUNS / f"mixed_dynamics{tag}.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="subset of seeds")
    p.add_argument("--steps", type=int, default=STEPS, help="training steps")
    p.add_argument("--ckpt", type=int, default=CKPT, help="checkpoint interval")
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
    print(f"\nwrote {path}\ntotal wall {rec['wall']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
