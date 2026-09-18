r"""P11-16: the marginal-matched X/Y discrimination at 16 bands.

The 3-band P11 (see experiments/4_occurrence) closed the occurrence question:
with the corpus marginal fixed, a band that is loud in a small fraction of
windows is copied at convergence (r -> 1), 12-24x above the marginal-prior
prediction. This run generalises that discrimination to a full patch-width
spectrum -- bands b = 1..16 at k = 32 -- where "red" is a 1/b profile and the
target band b16 carries a 1.85% corpus share, inside the slow-schedule regime P8
characterised (share 2% -> r = 0.51 at 2k, 0.96 at 10k).

Design. Every window has total power 0.5 over 16 carriers. Both arms share the
corpus marginal RED16 (alpha = 1 profile) exactly:

  x16: every window is RED16 -- b16 weak (1.85% share) in every window.
  y16: 10% of windows boost b16 to R = 0.0926 (18.5% of the window) with the
       other bands scaled by d = 0.830; 90% of windows drop b16 to zero and
       scale the rest by c = 1.019. Marginals match RED16 by construction:
       0.9 c + 0.1 d = 1 for b < 16, and 0.1 R = RED16[15]. The occurrence-
       weighted schedule share of b16 is 1.85% in BOTH arms.

Probe: x16 with the RED16 profile; y16 with its loud-window profile (b16 at
0.0926). Predictions at 10 000 steps: conditional account r(b16) -> ~1 (as in
the 3-band Y arms); marginal-prior account r(b16) = m16 / R = 0.10.

Pre-registered decision: y16 r(b16) >= 0.9 -> conditional account generalises
to full-spectrum shapes; within a factor of 2 of 0.10 -> marginal prior; else
intermediate. Contract 2: read 2 000 steps (schedule check, x16 should lag
like P8's rho ~ 25 arm) and 10 000 (fixed point).

Run from the repo root:
``.venv/bin/python experiments/4_occurrence/scripts/occurrence16.py``.
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
BANDS = tuple(float(b) for b in range(1, 17))  # 1..16, filling the k=32 patch
FREQS = tuple(b * CTX / PATCH for b in BANDS)  # 16, 32, ..., 256
PROBE_SEED = 1009
SEED_STRIDE = 1000
GATE = 0.5
RUNS = Path(__file__).resolve().parents[1] / "runs"

S16 = sum(1.0 / b for b in range(1, 17))
RED16 = tuple(0.5 * (1.0 / b) / S16 for b in range(1, 17))  # alpha = 1 profile
TGT = 15  # index of the target band; b16 (index 15) is Nyquist and degenerate


def make_arms(tgt):
    r"""Arms aimed at band ``tgt``, marginals pinned to RED16.

    The weak window zeroes the target band and rescales the rest by ``c``; the
    loud window raises it to ``R`` and rescales the rest by ``d``, with
    ``0.9c + 0.1d = 1`` (every other band keeps its RED16 marginal) and
    ``0.1R = RED16[tgt]`` (b16's share, 1.85%). The occurrence-weighted schedule
    share of the target band is therefore identical in both arms.
    """
    rest = 0.5 - RED16[tgt]
    r_loud = RED16[tgt] / 0.10
    c = 0.5 / rest
    d = (0.5 - r_loud) / rest
    weak = tuple(0.0 if j == tgt else c * p for j, p in enumerate(RED16))
    loud = tuple(r_loud if j == tgt else d * p for j, p in enumerate(RED16))
    tag = BANDS[tgt]
    return {
        f"x{tag:g}": {"patterns": [(1.0, RED16)], "probe": RED16},
        f"y{tag:g}": {"patterns": [(0.90, weak), (0.10, loud)], "probe": loud},
    }


ARMS = make_arms(TGT)


def check_patterns():
    r"""Allocations sum to 0.5; y16's marginal equals x16's marginal (RED16)."""
    for arm, spec in ARMS.items():
        for p, a in spec["patterns"]:
            assert abs(sum(a) - 0.5) < 1e-9, (arm, a)
            assert 0.0 <= p <= 1.0
        assert abs(sum(p for p, _ in spec["patterns"]) - 1.0) < 1e-9
        marg = [sum(p * a[j] for p, a in spec["patterns"]) for j in range(16)]
        for j in range(16):
            assert abs(marg[j] - RED16[j]) < 1e-9, (arm, j, marg[j], RED16[j])
    tag = BANDS[TGT]
    loud = ARMS[f"y{tag:g}"]["probe"]
    print(
        f"  x{tag:g}/y{tag:g} share the marginal RED16; "
        f"b{tag:g} share = {RED16[TGT] / 0.5:.4f}"
    )
    print(
        f"  y{tag:g} loud window: b{tag:g} at {loud[TGT]:.4f}, others x "
        f"{loud[0] / RED16[0]:.4f}"
    )


def pattern_table(spec):
    probs = torch.tensor([p for p, _ in spec["patterns"]], dtype=torch.float64)
    allocs = torch.tensor([a for _, a in spec["patterns"]], dtype=torch.float64)
    return probs, allocs


def synth(allocs, n, gen):
    r"""Windows from per-window allocations, ``(n, ctx + k)``; total power per
    window is exactly ``sum(allocs)``."""
    t = torch.arange(CTX + PATCH, dtype=torch.float32)
    freqs = torch.tensor(FREQS, dtype=torch.float32)
    base = 2 * math.pi * freqs[:, None] * t[None, :] / CTX  # (n_bands, T)
    phase = 2 * math.pi * torch.rand((n, len(BANDS)), generator=gen)
    amp = torch.sqrt(2.0 * allocs.to(torch.float32))  # (n, n_bands)
    return (amp[:, :, None] * torch.cos(base[None, :, :] + phase[:, :, None])).sum(1)


def draw_windows(spec, n, gen_pat, gen_ph):
    probs, allocs = pattern_table(spec)
    idx = torch.multinomial(probs.float(), n, replacement=True, generator=gen_pat)
    return synth(allocs[idx], n, gen_ph), idx


def corpus(arm, seed):
    gen_pat = torch.Generator().manual_seed(seed * SEED_STRIDE + 17)
    gen_ph = torch.Generator().manual_seed(seed * SEED_STRIDE + 71)
    return draw_windows(ARMS[arm], N_PER, gen_pat, gen_ph)


def fixed_windows(alloc, n, gen):
    return synth(torch.tensor([alloc], dtype=torch.float64).expand(n, -1), n, gen)


def probe(arm):
    return fixed_windows(
        ARMS[arm]["probe"], N_PER, torch.Generator().manual_seed(PROBE_SEED)
    )


def held_out(arm, seed):
    return fixed_windows(
        ARMS[arm]["probe"],
        N_PER,
        torch.Generator().manual_seed(seed * SEED_STRIDE + 307),
    )


@torch.no_grad()
def band_power(x, b):
    z = matched_amp(x.unfold(1, PATCH, PATCH), b, PATCH)
    return 0.5 * float(z.abs().square().mean())


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


def generator_check():
    r"""Marginals, occurrences, probe shape.

    The Nyquist band b16 is reported separately: for a real signal its
    negative-frequency image coincides with the tone, so the matched filter
    reads 2x the design power there. Retention is a ratio and stays consistent
    (both sides scale alike when the emitted phase matches), but the b16
    *power* columns must be read with that factor of 2 in mind.
    """
    out = {}
    for arm in ARMS:
        acc = {"rms": [], "marg_dev": [], "occ": [], "nyq": []}
        for seed in SEEDS:
            x, idx = corpus(arm, seed)
            acc["rms"].append(float(x.square().mean().sqrt()))
            realised = [band_power(x, b) for b in BANDS]
            acc["marg_dev"].append(
                max(abs(r - m) for r, m in zip(realised[:-1], RED16[:-1]))
            )
            acc["nyq"].append(realised[-1] / RED16[-1])
            acc["occ"].append([(idx == j).double().mean().item() for j in (0, 1)])
        prb = probe(arm)
        out[arm] = {
            "rms": mean_std(acc["rms"]),
            "max_marginal_dev": mean_std(acc["marg_dev"]),
            "nyquist_ratio": mean_std(acc["nyq"]),
            "occ": {str(j): mean_std([o[j] for o in acc["occ"]]) for j in (0, 1)},
            "probe_rms": float(prb.square().mean().sqrt()),
            "probe_b16_power": band_power(prb, BANDS[TGT]),
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
    data, _ = corpus(arm, seed)
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


def decide(rec):
    tag = BANDS[TGT]
    r_y = rec["arms"][f"y{tag:g}"]["mean_std"][f"r@{tag}"]
    r_x = rec["arms"][f"x{tag:g}"]["mean_std"][f"r@{tag}"]
    pred = RED16[TGT] / ARMS[f"y{tag:g}"]["probe"][TGT]
    unfitted = [a for a in rec["arms"] if not rec["arms"][a]["gate_passed"]]
    if unfitted:
        verdict = "inconclusive: arm(s) failed the fit gate"
    elif r_y[0] >= 0.9:
        verdict = (
            "conditional account generalises: full-spectrum "
            "marginal-matched heterogeneity is free of bias"
        )
    elif abs(r_y[0] - pred) < max(pred, 0.05):
        verdict = "marginal prior exists at full-spectrum resolution"
    else:
        verdict = "intermediate"
    return {
        "y_r_target": r_y,
        "x_r_target": r_x,
        "marginal_account_prediction": pred,
        "unfitted": unfitted,
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
    global TGT, ARMS
    if args.target not in BANDS:
        raise SystemExit(f"target {args.target} not in {BANDS}")
    TGT = BANDS.index(args.target)
    ARMS = make_arms(TGT)
    t0 = time.time()
    arms = {a: ARMS[a] for a in ARMS}
    if args.arms:
        missing = set(args.arms) - set(ARMS)
        assert not missing, f"unknown arms {sorted(missing)}; have {sorted(ARMS)}"
        arms = {a: v for a, v in arms.items() if a in args.arms}
    seeds = tuple(args.seeds) if args.seeds else SEEDS

    print(
        f"== occurrence16 (P11-16): {len(arms)} arms x {len(seeds)} seeds ==",
        flush=True,
    )
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {args.hidden} / {args.layers}L / "
        f"{HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {args.steps} steps / cpu",
        flush=True,
    )
    print(
        f"bands b = 1..16 (f = {FREQS[0]:g}..{FREQS[-1]:g}); marginal = 1/b profile",
        flush=True,
    )
    print("\npattern table (marginals must match):", flush=True)
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
            tgt = row[f"r@{BANDS[TGT]}"]
            print(
                f"  {arm:<4} seed {seed}  loss {loss:.2e}  var expl "
                f"{row['var_explained']:+.5f}  r@16 {tgt:.4f}  "
                f"min r {min(row[f'r@{b}'] for b in BANDS):.4f}  [{row['wall']:.0f}s]",
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
            "marginal": list(RED16),
            "loud_target_power": ARMS[f"y{BANDS[TGT]:g}"]["probe"][TGT],
            "probe_seed": PROBE_SEED,
            "gate_fraction": GATE,
            "readout": "pred[:, -2] vs pat(x)[:, -1]",
            "arms": {
                a: {"patterns": ARMS[a]["patterns"], "probe": ARMS[a]["probe"]}
                for a in arms
            },
        },
        "generator": gen,
        "arms": {},
        "wall": 0.0,
    }
    for arm in arms:
        rows = rows_by_arm[arm]
        s = summarize(rows)
        rec["arms"][arm] = {
            "probe": list(ARMS[arm]["probe"]),
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
    print("\ngenerator check, no training: marginals, occurrences, probe shape")
    table(
        [
            f"  {'arm':<5}",
            f"{'corpus rms':>11}",
            f"{'probe rms':>10}",
            f"{'max marg dev':>13}",
            f"{'occ weak/loud':>16}",
            f"{'probe b16 P':>12}",
        ],
        [
            [
                f"  {arm:<5}",
                f"{gen[arm]['rms'][0]:>11.5f}",
                f"{gen[arm]['probe_rms']:>10.5f}",
                f"{gen[arm]['max_marginal_dev'][0]:>13.5f}",
                f"{gen[arm]['occ']['0'][0]:>7.3f}/{gen[arm]['occ']['1'][0]:<7.3f}",
                f"{gen[arm]['probe_b16_power']:>12.5f}",
            ]
            for arm in gen
        ],
    )
    print(
        "  max marg dev is the largest |realised marginal - RED16| over the 16 bands;\n"
        "  both arms target RED16 exactly, per-window total power is 0.5 everywhere."
    )


def print_probe(rec):
    print("\nper arm: probe r at b16 (and its Fredformer dk), min over bands, oracle")
    tgt = BANDS[TGT]
    table(
        [
            f"  {'arm':<5}",
            f"{'r@' + format(BANDS[TGT], 'g'):>16}",
            f"{'dk@' + format(BANDS[TGT], 'g'):>16}",
            f"{'min r':>9}",
            f"{'oracle':>12}",
        ],
        [
            [
                f"  {arm:<5}",
                ms(a["mean_std"][f"r@{tgt}"], 16),
                ms(a["mean_std"][f"dk@{tgt}"], 16),
                f"{min(a['mean_std'][f'r@{b}'][0] for b in BANDS):>9.4f}",
                f"{a['mean_std'][f'r_oracle@{tgt}'][0]:>12.4f}",
            ]
            for arm, a in rec["arms"].items()
        ],
    )


def print_decision(rec):
    d = rec["decision"]
    tag = BANDS[TGT]
    print(f"\npre-registered decision (fixed point, b{tag:g})")
    print(f"  x{tag:g} r (homogeneous reference): {ms(d['x_r_target'])}")
    print(
        f"  y{tag:g} r = {ms(d['y_r_target'])}   "
        f"marginal-account prediction {d['marginal_account_prediction']:.3f}"
    )
    print(
        "  arms failing the fit gate: "
        + (", ".join(d["unfitted"]) if d["unfitted"] else "none")
    )
    print(f"  -> {d['verdict']}")


def report(rec):
    print_generator(rec)
    print_probe(rec)
    print_decision(rec)


def out_path(args):
    tag = f"_{args.tag}" if args.tag else ""
    return RUNS / f"occurrence16{tag}.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=None, help="subset of arm ids")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="subset of seeds")
    p.add_argument("--steps", type=int, default=STEPS, help="training steps")
    p.add_argument("--hidden", type=int, default=HIDDEN, help="hidden size")
    p.add_argument("--layers", type=int, default=LAYERS, help="transformer layers")
    p.add_argument(
        "--target",
        type=float,
        default=BANDS[TGT],
        help="target band in cycles/patch (b16 is Nyquist: degenerate)",
    )
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
