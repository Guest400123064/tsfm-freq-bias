r"""P12: the 1/f^alpha ladder -- is the apparent bias a function of spectral shape?

Why this run exists. P8 established the schedule mechanism on three bands with a
hand-set power ratio; P11 showed the fixed point tracks the conditional mean
rather than the corpus marginal. This run asks the question the blog needs in
its final form: **does the apparent per-band deficit follow the corpus's
spectral shape, continuously, in a way that transferable to real corpora?**

Design. Sixteen carriers b = 1..16 fill the k = 32 patch (f = 16..256 cycles per
window), fully coherent, no noise, total power 0.5 in every window. Five
homogeneous corpora with profile P_b proportional to b**(-alpha):

    alpha = 0.0   flat, every band 1/16 = 6.25% of the window
    alpha = 0.5   b16 share 2.2%
    alpha = 1.0   b16 share 1.85%
    alpha = 1.5   b16 share 1.1%
    alpha = 2.0   b16 share 0.26%

Each arm is probed clean with its **own** profile (shape-matched, contract 9), so
`r_b` is that arm's response to exactly the spectrum it was trained on.

Pre-registered decision, written before the run into this experiment's
README.md. Three claims, all checkable from one grid:

  1. **Representation is fine**: the frozen-latent oracle reads >= 0.9 at every
     band of every arm. If not, the arm is a capacity story, not a shape story.
  2. **Within an arm, the deficit follows the share**: Spearman corr between
     log share_b and r_b is >= 0.8 in every arm with alpha > 0.
  3. **The collapse**: pooling all (share_b, r_b) points across the five arms,
     every log2(share) octave containing >= 2 different arms has a spread of r
     of at most 0.15 -- i.e. one curve r = G(share) describes every shape, which
     is P8's mechanism read at spectral resolution.

Headline number, reported but not a decision rule: the apparent bias ratio
r(b1) / r(b16) as a function of alpha -- the dose-response curve of "how much
frequency bias a spectrum of this redness buys you at this budget".

Note on the Nyquist band: b16 = k/2 is its own negative-frequency image, so the
matched filter reads **2x** the design power there. Retention is a ratio and
stays consistent (both sides scale alike when the emitted phase matches), but
the b16 *power* columns are to be read with that factor in mind.

Contract 2 binds: 2 000 steps and 10 000 steps are both reported; the collapse
is a fixed-budget statement (G depends on t) and is checked separately at each.

Run from the repo root:
``.venv/bin/python experiments/5_spectrum_shape/scripts/spectrum_shape.py``.
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
HIDDEN, LAYERS, HEADS = 32, 2, 4
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
BANDS = tuple(float(b) for b in range(1, 17))
WELL = BANDS[:-1]  # b16 is Nyquist: degenerate phase, excluded from decisions
FREQS = tuple(b * CTX / PATCH for b in BANDS)  # 16, 32, ..., 256
TOTAL_POWER = 0.5
PROBE_SEED = 1301
SEED_STRIDE = 1000
GATE = 0.5
ORACLE_TOL = 0.9
SPEARMAN_TOL = 0.8
COLLAPSE_TOL = 0.15
RUNS = Path(__file__).resolve().parents[1] / "runs"

ALPHAS = {"a0": 0.0, "a05": 0.5, "a10": 1.0, "a15": 1.5, "a20": 2.0}


def profile(alpha):
    r"""Band powers ``propto b**(-alpha)``, normalised to ``TOTAL_POWER``."""
    w = [b ** (-alpha) for b in BANDS]
    s = sum(w)
    return tuple(TOTAL_POWER * x / s for x in w)


def check_patterns():
    for arm, alpha in ALPHAS.items():
        p = profile(alpha)
        assert abs(sum(p) - TOTAL_POWER) < 1e-9, arm
        share = p[-1] / TOTAL_POWER
        print(
            f"  {arm:<4} alpha={alpha:<4} b16 share = {share:.4f}  b1 share = "
            f"{p[0] / TOTAL_POWER:.4f}"
        )


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
    prof = profile(ALPHAS[arm])
    return fixed_windows(
        prof, N_PER, torch.Generator().manual_seed(seed * SEED_STRIDE + 71)
    )


def probe(arm):
    prof = profile(ALPHAS[arm])
    return fixed_windows(prof, N_PER, torch.Generator().manual_seed(PROBE_SEED))


def held_out(arm, seed):
    prof = profile(ALPHAS[arm])
    return fixed_windows(
        prof, N_PER, torch.Generator().manual_seed(seed * SEED_STRIDE + 307)
    )


@torch.no_grad()
def band_power(x, b):
    z = matched_amp(x.unfold(1, PATCH, PATCH), b, PATCH)
    return 0.5 * float(z.abs().square().mean())


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


def spearman(x, y):
    r"""Rank correlation, no scipy: Pearson on ranks."""

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return torch.tensor(r, dtype=torch.float64)

    a, b = ranks(x), ranks(y)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def generator_check():
    out = {}
    for arm in ALPHAS:
        prof = profile(ALPHAS[arm])
        dev, nyq, rms = [], [], []
        for seed in SEEDS:
            x = corpus(arm, seed)
            rms.append(float(x.square().mean().sqrt()))
            realised = [band_power(x, b) for b in BANDS]
            dev.append(max(abs(r - m) for r, m in zip(realised[:-1], prof[:-1])))
            nyq.append(realised[-1] / prof[-1])
        prb = probe(arm)
        out[arm] = {
            "alpha": ALPHAS[arm],
            "profile_b1": prof[0],
            "profile_b16": prof[-1],
            "rms": mean_std(rms),
            "max_marginal_dev": mean_std(dev),
            "nyquist_ratio": mean_std(nyq),
            "probe_rms": float(prb.square().mean().sqrt()),
            "probe_ratio": [band_power(prb, b) / prof[j] for j, b in enumerate(BANDS)],
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


def collapse(rec):
    r"""Pooled (share, r) points across arms, binned by log2(share) octave."""
    pts = []
    for arm, a in rec["arms"].items():
        prof = profile(ALPHAS[arm])
        for j, b in enumerate(WELL):
            pts.append((prof[j] / TOTAL_POWER, a["mean_std"][f"r@{b}"][0], arm))
    bins = {}
    for share, r, arm in pts:
        bins.setdefault(int(math.floor(math.log2(share))), []).append((r, arm))
    rows = []
    for k in sorted(bins):
        vals = bins[k]
        arms_here = {arm for _, arm in vals}
        rs = [r for r, _ in vals]
        rows.append(
            {
                "octave": k,
                "share_range": (2.0**k, 2.0 ** (k + 1)),
                "n_points": len(vals),
                "n_arms": len(arms_here),
                "r_mean": sum(rs) / len(rs),
                "r_spread": max(rs) - min(rs),
            }
        )
    multi = [x for x in rows if x["n_arms"] >= 2]
    worst = max((x["r_spread"] for x in multi), default=0.0)
    return {"rows": rows, "multi_arm_bins": len(multi), "worst_spread": worst}


def decide(rec):
    arms = rec["arms"]
    oracle_min = {
        a: min(v["mean_std"][f"r_oracle@{b}"][0] for b in WELL) for a, v in arms.items()
    }
    corr = {}
    for a, v in arms.items():
        prof = profile(ALPHAS[a])
        corr[a] = spearman(
            [math.log(prof[j]) for j, _ in enumerate(WELL)],
            [v["mean_std"][f"r@{b}"][0] for b in WELL],
        )
    ratio = {
        a: v["mean_std"][f"r@{BANDS[0]}"][0]
        / max(v["mean_std"][f"r@{WELL[-1]}"][0], 1e-9)
        for a, v in arms.items()
    }
    col = collapse(rec)
    unfitted = [a for a in arms if not arms[a]["gate_passed"]]
    bad_oracle = [a for a, o in oracle_min.items() if o < ORACLE_TOL]
    weak_corr = [a for a, c in corr.items() if ALPHAS[a] > 0 and c < SPEARMAN_TOL]
    if unfitted:
        verdict = "inconclusive: arm(s) failed the fit gate"
    elif bad_oracle:
        verdict = "capacity story: the oracle is short at some band"
    elif weak_corr:
        verdict = "shape does not order the deficit as a share effect"
    elif col["worst_spread"] <= COLLAPSE_TOL:
        verdict = (
            "collapse holds: one curve r = G(share) describes every spectrum, so "
            "the apparent bias is the schedule read at spectral resolution"
        )
    else:
        verdict = "no collapse: the deficit depends on more than the band's share"
    return {
        "oracle_min": oracle_min,
        "spearman_share_vs_r": corr,
        "bias_ratio_b1_over_b16": ratio,
        "collapse": col,
        "unfitted": unfitted,
        "bad_oracle": bad_oracle,
        "weak_corr": weak_corr,
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
    arms = {a: ALPHAS[a] for a in ALPHAS}
    if args.arms:
        missing = set(args.arms) - set(ALPHAS)
        assert not missing, f"unknown arms {sorted(missing)}; have {sorted(ALPHAS)}"
        arms = {a: v for a, v in arms.items() if a in args.arms}
    seeds = tuple(args.seeds) if args.seeds else SEEDS

    print(
        f"== spectrum_shape (P12): {len(arms)} alphas x {len(seeds)} seeds ==",
        flush=True,
    )
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {args.hidden} / {args.layers}L / "
        f"{HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {args.steps} steps / cpu",
        flush=True,
    )
    print(
        f"bands b = 1..16 (f = {FREQS[0]:g}..{FREQS[-1]:g}), profile P_b ~ b^-alpha",
        flush=True,
    )
    print("\nprofiles:", flush=True)
    check_patterns()
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
            print(
                f"  {arm:<4} seed {seed}  loss {loss:.2e}  var expl "
                f"{row['var_explained']:+.5f}  r@1 {row[f'r@{BANDS[0]}']:.3f}  "
                f"r@15 {row[f'r@{WELL[-1]}']:.3f}  [{row['wall']:.0f}s]",
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
            "alphas": {a: ALPHAS[a] for a in arms},
            "profiles": {a: list(profile(ALPHAS[a])) for a in arms},
            "probe_seed": PROBE_SEED,
            "gate_fraction": GATE,
            "oracle_tol": ORACLE_TOL,
            "spearman_tol": SPEARMAN_TOL,
            "collapse_tol": COLLAPSE_TOL,
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
            "alpha": ALPHAS[arm],
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
    print("\ngenerator check, no training: profiles and probe shape match")
    table(
        [
            f"  {'arm':<5}",
            f"{'alpha':>5}",
            f"{'corpus rms':>11}",
            f"{'probe rms':>10}",
            f"{'P(b1)':>9}",
            f"{'P(b16)':>9}",
            f"{'max marg dev':>13}",
            f"{'probe/P min':>12}",
        ],
        [
            [
                f"  {arm:<5}",
                f"{gen[arm]['alpha']:>5.1f}",
                f"{gen[arm]['rms'][0]:>11.5f}",
                f"{gen[arm]['probe_rms']:>10.5f}",
                f"{gen[arm]['profile_b1']:>9.5f}",
                f"{gen[arm]['profile_b16']:>9.5f}",
                f"{gen[arm]['max_marginal_dev'][0]:>13.5f}",
                f"{min(gen[arm]['probe_ratio'][:-1]):>12.4f}",
            ]
            for arm in gen
        ],
    )
    print(
        "  max marg dev excludes the Nyquist band b16, where a real tone is its own\n"
        "  negative-frequency image and the matched filter reads 2x the design power\n"
        "  (retention is a ratio and stays consistent)."
    )


def print_bands(rec):
    print("\nclean probe r by band; the share column is the alpha = 2 profile")
    table(
        [
            f"  {'share a=2':>9}",
            f"{'b':>4}",
            *[f"{'a=' + str(ALPHAS[a]):>14}" for a in rec["arms"]],
        ],
        [
            [
                f"  {profile(2.0)[j] / TOTAL_POWER:>9.4f}",
                f"{int(b):>4}",
                *[ms(rec["arms"][a]["mean_std"][f"r@{b}"], 14) for a in rec["arms"]],
            ]
            for j, b in enumerate(BANDS)
        ],
    )
    print(
        "  each arm is probed with its own profile, so compare down a column; the\n"
        "  share column is the alpha = 2 profile's shape, shown for orientation."
    )


def print_decision(rec):
    d = rec["decision"]
    print("\npre-registered checks")
    print("  1. oracle minimum over bands (>= 0.9 required):")
    for a, o in d["oracle_min"].items():
        flag = "" if o >= ORACLE_TOL else "   <-- SHORT"
        print(f"       {a:<5} {o:.4f}{flag}")
    print("  2. Spearman(log share, r) per arm (>= 0.8 required for alpha > 0):")
    for a, c in d["spearman_share_vs_r"].items():
        print(f"       {a:<5} {c:+.3f}")
    print(
        f"  3. collapse across arms: worst spread {d['collapse']['worst_spread']:.4f} "
        f"over {d['collapse']['multi_arm_bins']} multi-arm octaves "
        f"(tol {COLLAPSE_TOL})"
    )
    print("\n  dose-response, r(b1) / r(b15):")
    for a, v in d["bias_ratio_b1_over_b16"].items():
        print(f"       {a:<5} alpha={ALPHAS[a]:<4} {v:8.3f}")
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
    return RUNS / f"spectrum_shape{tag}.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=None, help="subset of arm ids")
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
