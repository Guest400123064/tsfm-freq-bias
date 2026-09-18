r"""P8: band **power allocation** alone, at fixed total power.

Why this run exists. Fredformer's stated driver of frequency bias is a band's
share of the spectrum's energy (PLAN.md 2.4). Their Case 1 cannot be replicated
-- the paper never states k1/k2/k3, the amplitudes, the sampling rate or the
model config -- so this is the claim itself, tested in a setup we specify
completely. P7 crossed power with predictability; this run has no predictability
axis at all.

Design. Three carriers ``b = 2, 4, 8`` (``f = 32, 64, 128`` at ctx 512 / k 32)
-- all integers, so every matched filter is exact, ``dphi = 0`` and no band can
leak into another. All three are fully coherent and no noise is added, so total
power is ``0.5``, every band is perfectly predictable, the Bayes optimum is
``r* = 1`` at every band, and any shortfall is a failure to fit rather than a
correct shrink.

  Axis. One band carries ``P_w = 0.5 / (2 rho + 1)`` and the other two carry
  ``rho * P_w`` each, so total power is ``0.5`` in every arm. ``rho`` runs over
  ``{5, 25, 100, 1000}``; ``rho = 1`` is the equal-power reference arm.

  Rotation. At a fixed ``rho`` the three arms are **permutations** of one
  another: the power multiset ``{P_w, rho P_w, rho P_w}`` is fixed and only its
  assignment to ``b = 2, 4, 8`` changes. So each band spends one arm of the
  family weak and two loud, and the per-band mean over the family has the power
  effect **cancelled by construction** -- any spread left across bands there is
  frequency, not power. This is the separation P7 could not make with a fixed
  target band.

Pre-registered decision, written before the run into this experiment's
``README.md``. Let ``r`` be a band's clean-probe retention. The energy account
predicts ``r`` tracks the power the band carries; our account predicts ``r = 1``
everywhere, with a deficit that is uniform across bands and shrinking in steps
(contract 2). No knob is tuned after seeing results.

Run from the repo root:
``.venv/bin/python experiments/3_power_alloc/scripts/power_alloc.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fbias.data import make_pm
from fbias.model import SimTFM
from fbias.probes import delta_k, fit_oracle, matched_amp, retention

CTX, PATCH = 512, 32
HIDDEN, LAYERS, HEADS = 32, 2, 4
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
FREQS = (32.0, 64.0, 128.0)
BANDS = tuple(FREQS[j] * PATCH / CTX for j in range(len(FREQS)))  # 2.0, 4.0, 8.0
TOTAL_POWER = 0.5
RHOS = (5, 25, 100, 1000)
PROBE_SEED = 123
SEED_STRIDE = 1000  # one generator seed per band, shared by every arm
GATE = 0.5  # below half the Bayes bound an arm has not learned its marginal
TOL = 0.05  # a retention gap at or below this is not an effect
RUNS = Path(__file__).resolve().parents[1] / "runs"


def arm_id(rho, weak=None):
    r"""``x25_w1``: control/weak power ratio and the band carrying the weak share."""
    return f"x{rho:g}" if weak is None else f"x{rho:g}_w{weak:g}"


ARMS = {arm_id(1.0): {"rho": 1.0, "weak": None}}
for _rho in RHOS:
    for _j in range(len(FREQS)):
        ARMS[arm_id(_rho, _j)] = {"rho": float(_rho), "weak": _j}


def arm_powers(arm):
    r"""Per-band tone powers, summing to ``TOTAL_POWER`` in every arm."""
    rho, weak = ARMS[arm]["rho"], ARMS[arm]["weak"]
    if weak is None:
        return tuple(TOTAL_POWER / len(FREQS) for _ in FREQS)
    p_weak = TOTAL_POWER / (2 * rho + 1)
    return tuple(p_weak if j == weak else rho * p_weak for j in range(len(FREQS)))


def weak_index(arm):
    r"""The band carrying the weak share of an arm, or ``None`` at ``rho = 1``."""
    return ARMS[arm]["weak"]


def walk_seed(seed, j):
    r"""The generator seed band ``j`` uses -- the same in every arm and P7."""
    return seed * SEED_STRIDE + j


def arm_parts(arm, seed):
    r"""Per-band parts of the arm's corpus, each a full ``(2 * N_PER, ctx + k)``.

    ``make_pm`` at ``beta = 0`` with one frequency is an exactly coherent tone
    (P2b's code path, its phase walk switched off), scaled by that band's
    amplitude. The seeds are P7's, so the arm at ``rho = 25`` with band 8 weak
    is P7's ``x25_b0`` corpus -- a free reproduction check.
    """
    powers = arm_powers(arm)
    return [
        math.sqrt(2 * powers[j])
        * make_pm([f], [0.0], 2 * N_PER, CTX, PATCH, walk_seed(seed, j))[0]
        for j, f in enumerate(FREQS)
    ]


def corpus(arm, seed):
    r"""The arm's training marginal: the windows."""
    return torch.stack(arm_parts(arm, seed)).sum(0)


def probe_parts(arm):
    r"""The arm's clean probe, per band: its own powers, no phase walk.

    The same generator seed in every arm, so phases are shared and only the
    amplitudes differ -- the arms of a family are paired.
    """
    gen = torch.Generator().manual_seed(PROBE_SEED)
    phase = 2 * math.pi * torch.rand((N_PER, len(FREQS)), generator=gen)
    t = torch.arange(CTX + PATCH, dtype=torch.float32)
    powers = arm_powers(arm)
    return [
        math.sqrt(2 * powers[j])
        * torch.cos(2 * math.pi * f * t / CTX + phase[:, j, None])
        for j, f in enumerate(FREQS)
    ]


def probe(arm):
    return torch.stack(probe_parts(arm)).sum(0)


@torch.no_grad()
def band_power(x, b):
    r"""``mean_p |Z_p|^2 / 2``: the band's **total** power."""
    return 0.5 * float(
        matched_amp(x.unfold(1, PATCH, PATCH), b, PATCH).abs().square().mean()
    )


@torch.no_grad()
def in_band(mix, parts):
    r"""Share of each band's measured power that lives in its own part (diagonal).

    Integer-spaced carriers are orthogonal over a patch, so this is 1.000 by
    construction -- measured rather than assumed, because leakage is the one
    thing that would let a loud band contaminate a weak band's reading.
    """
    tot = [band_power(mix, b) for b in BANDS]
    out = []
    for i, b in enumerate(BANDS):
        own = band_power(parts[i], b)
        out.append(own / tot[i])
    return out


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


def generator_check():
    r"""No training: total power fixed, bands orthogonal, probe shape-matched."""
    out = {}
    for arm in ARMS:
        acc = {
            "rms": [],
            "probe_rms": [],
            "in_band": [],
            "power": [],
            "probe_power": [],
        }
        for seed in SEEDS:
            parts = arm_parts(arm, seed)
            mix = torch.stack(parts).sum(0)
            prb = probe(arm)
            acc["rms"].append(float(mix.square().mean().sqrt()))
            acc["probe_rms"].append(float(prb.square().mean().sqrt()))
            acc["in_band"].append(in_band(mix, parts))
            acc["power"].append([band_power(mix, b) for b in BANDS])
            acc["probe_power"].append([band_power(prb, b) for b in BANDS])
        out[arm] = {
            "powers": list(arm_powers(arm)),
            "rms": mean_std(acc["rms"]),
            "probe_rms": mean_std(acc["probe_rms"]),
            "in_band": {
                str(b): mean_std([r[i] for r in acc["in_band"]])
                for i, b in enumerate(BANDS)
            },
            "power": {
                str(b): mean_std([p[i] for p in acc["power"]])
                for i, b in enumerate(BANDS)
            },
            "probe_ratio": {
                str(b): mean_std([p[i] for p in acc["probe_power"]])[0]
                / mean_std([p[i] for p in acc["power"]])[0]
                for i, b in enumerate(BANDS)
            },
        }
    return out


def train(seed, arm, steps):
    r"""P7's training loop verbatim; only the corpus differs between arms."""
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        num_attn_heads=HEADS,
    )
    data = corpus(arm, seed)
    train_data = data[:N_PER]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        x = train_data[torch.randint(N_PER, (BATCH,), generator=gen)]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, data[N_PER:], float(loss)


@torch.no_grad()
def evaluate(model, held, prb, train_loss):
    r"""Fit gate, the marginal, then the clean probe in both our units and theirs."""
    model.eval()
    pred, _, z_held = model(held)
    patches = model.pat(held)
    target = patches[:, 1:]
    mse = float(F.mse_loss(pred[:, :-1], target))
    var = float(target.var(unbiased=False))
    row = {
        "train_loss": train_loss,
        "held_var": var,
        "var_explained": 1 - mse / var,
    }
    for b in BANDS:
        row[f"rm@{b}"] = float(retention(pred[:, -2], target[:, -1], b, PATCH).mean())
    p_hat, _, z = model(prb)
    true = model.pat(prb)
    f_hat = p_hat[:, -2]
    for b in BANDS:
        row[f"r@{b}"] = float(retention(f_hat, true[:, -1], b, PATCH).mean())
        row[f"dk@{b}"] = float(delta_k(f_hat, true[:, -1], b, PATCH).mean())
        # out-of-sample ridge (contract 5): held latents -> probe
        row[f"r_oracle@{b}"] = fit_oracle(z_held, patches, z, true, b, PATCH)[
            "retention"
        ]
    return row


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: mean_std([r[k] for r in rows]) for k in keys}


def power_levels(rec):
    r"""``(rho, r at the weak band, r at the loud bands)`` per arm, seed-averaged."""
    rows = []
    for arm, a in rec["arms"].items():
        weak = weak_index(arm)
        if weak is None:
            continue
        s = a["mean_std"]
        loud = [j for j in range(len(BANDS)) if j != weak]
        rows.append(
            {
                "arm": arm,
                "rho": ARMS[arm]["rho"],
                "r_weak": s[f"r@{BANDS[weak]}"][0],
                "r_loud": sum(s[f"r@{BANDS[j]}"][0] for j in loud) / len(loud),
                "dk_weak": s[f"dk@{BANDS[weak]}"][0],
                "mse_share": arm_powers(arm)[weak] / TOTAL_POWER,
            }
        )
    return rows


def decide(rec):
    r"""The pre-registered decision, plus the frequency-versus-power decomposition.

    Power effect: at fixed ``rho``, the weak band's retention against the loud
    bands' (paired within an arm). Frequency effect: each band's retention
    averaged over a ``rho`` family, where every band carries the same mix of
    power levels by construction, so the power effect cancels.
    """
    levels = power_levels(rec)
    by_rho = {}
    for rho in RHOS:
        rows = [r for r in levels if r["rho"] == rho]
        if not rows:
            continue
        by_rho[str(rho)] = {
            "r_weak": mean_std([r["r_weak"] for r in rows]),
            "r_loud": mean_std([r["r_loud"] for r in rows]),
            "gap": mean_std([r["r_loud"] - r["r_weak"] for r in rows]),
        }
    by_band = {}
    for b in BANDS:
        vals = [
            a["mean_std"][f"r@{b}"][0]
            for arm, a in rec["arms"].items()
            if weak_index(arm) is not None
        ]
        if vals:
            by_band[str(b)] = mean_std(vals)
    spreads = [v[0] for v in by_band.values()]
    band_spread = max(spreads) - min(spreads) if spreads else 0.0
    max_gap = max((v["gap"][0] for v in by_rho.values()), default=0.0)
    unfitted = [a for a in rec["arms"] if not rec["arms"][a]["gate_passed"]]
    if unfitted:
        verdict = "inconclusive: arm(s) failed the fit gate"
    elif max_gap <= TOL:
        verdict = "no power effect: the energy account is not supported"
    elif band_spread > TOL:
        verdict = "frequency-locked: the deficit does not follow the power"
    else:
        verdict = "power-locked: the deficit follows the assigned power"
    return {
        "by_rho": by_rho,
        "by_band": by_band,
        "band_spread": band_spread,
        "max_gap": max_gap,
        "unfitted": unfitted,
        "verdict": verdict,
    }


def table(head, rows):
    r"""Print a header row and body rows, all pre-formatted as cell lists."""
    line = " ".join(head)
    print(line)
    print("  " + "-" * (len(line) - 2))
    for r in rows:
        print(" ".join(r))


def ms(pair, width=10, digits=4):
    r"""``mean±sd`` in a fixed-width cell."""
    return f"{pair[0]:.{digits}f}±{pair[1]:.{digits}f}".rjust(width)


def build(args):
    t0 = time.time()
    arms = {a: v for a, v in ARMS.items()}
    if args.arms:
        missing = set(args.arms) - set(ARMS)
        assert not missing, f"unknown arms {sorted(missing)}; have {sorted(ARMS)}"
        arms = {a: v for a, v in arms.items() if a in args.arms}
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    steps = args.steps

    print(f"== power_alloc (P8): {len(arms)} arms x {len(seeds)} seeds ==", flush=True)
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {HIDDEN} / {LAYERS}L / {HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {steps} steps / cpu",
        flush=True,
    )
    print(
        f"bands f = {FREQS} (b = {BANDS}); rho = {list(RHOS)}; "
        f"total power {TOTAL_POWER} in every arm",
        flush=True,
    )
    print("\ngenerator check (no training yet)", flush=True)
    gen = generator_check()

    rows_by_arm, walls = {}, []
    for arm in arms:
        rows = []
        for seed in seeds:
            t = time.time()
            model, held, loss = train(seed, arm, steps)
            row = evaluate(model, held, probe(arm), loss)
            row["seed"] = seed
            row["wall"] = time.time() - t
            walls.append(row["wall"])
            rows.append(row)
            print(
                f"  {arm:<9} seed {seed}  loss {loss:.2e}  "
                f"var expl {row['var_explained']:+.5f}  probe r = "
                + " ".join(f"{row[f'r@{b}']:.4f}" for b in BANDS)
                + "  dk = "
                + " ".join(f"{row[f'dk@{b}']:.4f}" for b in BANDS)
                + f"  [{row['wall']:.0f}s]",
                flush=True,
            )
        rows_by_arm[arm] = rows

    rec = {
        "config": {
            "ctx": CTX,
            "patch": PATCH,
            "hidden": HIDDEN,
            "layers": LAYERS,
            "heads": HEADS,
            "steps": steps,
            "batch": BATCH,
            "lr": LR,
            "n_per": N_PER,
            "seeds": list(seeds),
            "device": "cpu",
            "freqs": list(FREQS),
            "b": list(BANDS),
            "total_power": TOTAL_POWER,
            "rhos": list(RHOS),
            "probe_seed": PROBE_SEED,
            "walk_seed_stride": SEED_STRIDE,
            "gate_fraction": GATE,
            "tol": TOL,
            "readout": "pred[:, -2] vs pat(x)[:, -1]",
            "arms": arms,
        },
        "generator": gen,
        "probe": {
            "seed": PROBE_SEED,
            "shape": "the arm's own per-band powers, no phase walk, no noise",
            "n_per": N_PER,
        },
        "arms": {},
        "wall": 0.0,
    }
    for arm in arms:
        rows = rows_by_arm[arm]
        s = summarize(rows)
        rec["arms"][arm] = {
            "rho": ARMS[arm]["rho"],
            "weak": ARMS[arm]["weak"],
            "powers": list(arm_powers(arm)),
            "gate_passed": s["var_explained"][0] > GATE,
            "per_seed": rows,
            "mean_std": s,
        }
    rec["wall_train_mean"] = sum(walls) / len(walls) if walls else 0.0
    rec["wall"] = time.time() - t0
    rec["decision"] = decide(rec)
    return rec


def print_config(rec):
    c = rec["config"]
    print(
        f"\nconfig: ctx {c['ctx']} / patch {c['patch']} / hidden {c['hidden']} / "
        f"{c['layers']}L / {c['heads']} heads | SGD lr {c['lr']} / batch {c['batch']} "
        f"/ {c['steps']} steps | seeds {c['seeds']}"
    )
    print(
        f"  target-free by design: every arm shares the band set b = {list(c['b'])} "
        f"and total power {c['total_power']},\n  and the rho families differ only in "
        "which band carries the weak share."
    )


def print_generator(rec):
    gen, arms = rec["generator"], rec["config"]["arms"]
    print("\ngenerator check, no training: total power, orthogonality, probe shape")
    table(
        [
            f"  {'arm':<9}",
            f"{'corpus rms':>11}",
            f"{'probe rms':>10}",
            *[f"{'P@' + str(b):>10}" for b in BANDS],
            *[f"{'probe/P@' + str(b):>12}" for b in BANDS],
            *[f"{'in-band@' + str(b):>13}" for b in BANDS],
        ],
        [
            [
                f"  {arm:<9}",
                f"{gen[arm]['rms'][0]:>11.5f}",
                f"{gen[arm]['probe_rms'][0]:>10.5f}",
                *[f"{gen[arm]['power'][str(b)][0]:>10.5f}" for b in BANDS],
                *[f"{gen[arm]['probe_ratio'][str(b)]:>12.4f}" for b in BANDS],
                *[f"{gen[arm]['in_band'][str(b)][0]:>13.4f}" for b in BANDS],
            ]
            for arm in arms
        ],
    )
    print(
        "  every arm's per-band powers are exact by construction and the sum is "
        f"{TOTAL_POWER}: the power axis\n  moves the allocation only. in-band is "
        "the share of a band's measured power that lives in its own\n  part; "
        "integer-spaced carriers are orthogonal over a patch, so 1.0000 means a "
        "loud band cannot\n  leak into a weak one. probe/P is the shape match: "
        "1.0000 at every band by construction."
    )


def print_marginal(rec):
    print("\nper arm: the marginal (its own held-out corpus), and the oracle")
    table(
        [
            f"  {'arm':<9}",
            f"{'var expl':>20}",
            *[f"{'rm@' + str(b):>17}" for b in BANDS],
        ],
        [
            [
                f"  {arm:<9}",
                f"{s['var_explained'][0]:>+9.5f}±{s['var_explained'][1]:<9.5f}",
                *[ms(s[f"rm@{b}"], 17) for b in BANDS],
            ]
            for arm, a in rec["arms"].items()
            for s in (a["mean_std"],)
        ],
    )
    print(
        "  the corpus is deterministic, so the ceiling is 1.000 at every band and "
        "there is no floor to\n  subtract: rm@b below 1.000 is a fitting gap, not "
        "a Bayes limit."
    )

    print("\nper arm: the clean probe, in our units and in Fredformer's")
    table(
        [
            f"  {'arm':<9}",
            *[f"{'r@' + str(b):>16}" for b in BANDS],
            *[f"{'dk@' + str(b):>16}" for b in BANDS],
            *[f"{'oracle@' + str(b):>11}" for b in BANDS],
        ],
        [
            [
                f"  {arm:<9}",
                *[ms(s[f"r@{b}"], 16) for b in BANDS],
                *[ms(s[f"dk@{b}"], 16) for b in BANDS],
                *[f"{s[f'r_oracle@{b}'][0]:>11.4f}" for b in BANDS],
            ]
            for arm, a in rec["arms"].items()
            for s in (a["mean_std"],)
        ],
    )
    print(
        "  r is the amplitude ratio and stays primary; dk = |Z_pred - Z_true| / "
        "|Z_true| is the number\n  Fredformer's figures are in, and on a "
        "phase-locked prediction it is exactly 1 - r. The\n  oracle is a ridge on "
        "the frozen latents: at 1.000 in every cell it says the representation "
        "holds\n  the band and any head shortfall is optimisation (contract 5)."
    )


def print_factorial(rec):
    d = rec["decision"]
    levels = power_levels(rec)
    print("\npower effect: the weak band against the loud bands, paired within an arm")
    table(
        [
            f"  {'rho':>5}",
            f"{'r at weak band':>17}",
            f"{'r at loud bands':>18}",
            f"{'gap':>14}",
            f"{'weak band MSE share':>20}",
        ],
        [
            [
                f"  {rho:>5.0f}",
                ms(d["by_rho"][str(rho)]["r_weak"], 17),
                ms(d["by_rho"][str(rho)]["r_loud"], 18),
                ms(d["by_rho"][str(rho)]["gap"], 14),
                f"{next(r['mse_share'] for r in levels if r['rho'] == rho):>20.5f}",
            ]
            for rho in RHOS
            if str(rho) in d["by_rho"]
        ],
    )
    print(
        f"  a gap above {TOL} is the energy account: the band carrying less power "
        "is retained less. Each\n  row pools the three rotations, so a gap that "
        "comes from 'band 8 is always weak' would have\n  to survive being "
        "rotated onto b = 2 and b = 4 as well."
    )

    print("\nfrequency effect: each band's retention pooled over the rho families")
    table(
        [f"  {'band':>5}", f"{'f':>7}", f"{'r pooled':>18}"],
        [
            [
                f"  {b:>5.1f}",
                f"{FREQS[j]:>7.1f}",
                ms(d["by_band"][str(b)], 18),
            ]
            for j, b in enumerate(BANDS)
        ],
    )
    print(
        "  inside a rho family every band spends one arm weak and two loud, so a "
        "power effect cancels\n  exactly in these means: what is left is "
        f"frequency. Spread {d['band_spread']:.4f} against the {TOL} tolerance."
    )


def print_per_seed(rec):
    print("\nper seed: fit gate, then the marginal and the clean probe")
    table(
        [
            f"  {'arm':<9}",
            f"{'seed':>4}",
            f"{'var expl':>10}",
            *[f"{'rm@' + str(b):>9}" for b in BANDS],
            *[f"{'r@' + str(b):>9}" for b in BANDS],
            *[f"{'dk@' + str(b):>9}" for b in BANDS],
        ],
        [
            [
                f"  {arm:<9}",
                f"{row['seed']:>4}",
                f"{row['var_explained']:>+10.5f}",
                *[f"{row[f'rm@{b}']:>9.5f}" for b in BANDS],
                *[f"{row[f'r@{b}']:>9.5f}" for b in BANDS],
                *[f"{row[f'dk@{b}']:>9.5f}" for b in BANDS],
            ]
            for arm, a in rec["arms"].items()
            for row in a["per_seed"]
        ],
    )


def print_decision(rec):
    d = rec["decision"]
    print("\npre-registered decision")
    print(f"  largest weak-vs-loud gap over rho: {d['max_gap']:.4f} (tolerance {TOL})")
    print(
        f"  spread across bands, power pooled: {d['band_spread']:.4f} (tolerance {TOL})"
    )
    print(
        "  arms failing the fit gate: "
        + (", ".join(d["unfitted"]) if d["unfitted"] else "none")
    )
    print(f"  -> {d['verdict']}")


def report(rec):
    print_config(rec)
    print_generator(rec)
    print_marginal(rec)
    print_factorial(rec)
    print_per_seed(rec)
    print_decision(rec)


def out_path(args):
    tag = f"_{args.tag}" if args.tag else ""
    return RUNS / f"power_alloc{tag}.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=None, help="subset of arm ids")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="subset of seeds")
    p.add_argument("--steps", type=int, default=STEPS, help="training steps")
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
