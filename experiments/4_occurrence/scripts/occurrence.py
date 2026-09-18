r"""P11: marginal-matched homogeneous vs heterogeneous allocation.

Why this run exists. P8 settled the power axis: the deficit is a *schedule*
ordered by a band's share of the loss, never a property of its frequency, and
the fixed point is set by predictability alone. But P8's corpus was
*homogeneous* -- every window carried the same allocation -- which is the one
thing a real corpus's signal generating process decides for you and which data
mixing cannot change. The question P11 asks is the one that matters for data
preparation: if the corpus **marginal** is held fixed and only the
*distribution of allocation across windows* changes, does the bias change?

The two accounts, made falsifiable. Because every window here carries total
power 0.5, a band's corpus marginal is algebraically identical to its
occurrence-weighted share -- so "marginal vs occurrence" cannot be separated by
moving one band's occurrence. What can be separated is whether the model
**conditions on the context-visible allocation**:

- **Conditional account** (ours): the model learns p(future | context);
  allocation is visible in the context, so at convergence any allocation that
  *occurred* in training is copied, whatever its marginal. The marginal enters
  only the schedule, through each pattern's share of the loss.
- **Marginal-prior account** (Fredformer-style energy story at corpus level):
  the model's per-band response scales with the corpus marginal, regardless of
  what the context shows.

Design. Three carriers b = 2, 4, 8 (f = 32, 64, 128 at ctx 512 / k 32), fully
coherent, no noise. Every window has total power 0.5. All discrimination arms
share the corpus marginal m = (0.24, 0.24, 0.02) for (b2, b4, b8) exactly.

  X  (homogeneous): every window is (0.24, 0.24, 0.02) -- b8 moderate in
      every window.
  Yq (heterogeneous): a fraction q of windows is "b8 loud" at power R, the
      rest have b8 absent; q * R = 0.02, so the corpus marginal matches X
      exactly and -- the point of the ladder -- the occurrence-weighted
      *schedule* share of b8 is 0.02 / 0.5 = 4% in every arm, X and Yq alike.
      Only the fixed-point prediction varies:
          q=4%:  loud window (0,    0,    0.5)   marginal account: r = 0.04
          q=6%:  loud window (1/12, 1/12, 1/3)   marginal account: r = 0.06
          q=8%:  loud window (1/8,  1/8,  0.25)  marginal account: r = 0.08
      The conditional account predicts r -> 1 at convergence for all three.
  C  (anchor): every window (0.245, 0.245, 0.01) -- P8's x25_w2 corpus, the
      natural homogeneous red case, run for continuity.
  A  (flat-homogeneous) and B (flat-heterogeneous, uniform orbit of
      (0.40, 0.07, 0.03), marginal exactly flat): controls -- does
      heterogeneity itself cost anything?

Probes. Each arm is probed with a clean window in its own discriminating
allocation (X with (0.24, 0.24, 0.02), Yq with its loud window, B with one of
its orbit patterns), so the probe is shape-matched to the pattern whose
learnability is at stake. Held-out windows for the fit gate and the marginal
are drawn from the same pattern.

Pre-registered decision, written before the run into this experiment's
README.md. At 10 000 steps, let r_q be the retention of b8 on Yq's loud probe.
All three Y arms share X's schedule, so any spread in r_q is a fixed-point
statement:

  min(r_q) >= 0.9                      -> conditional account: with the
                                          marginal matched, heterogeneity is
                                          free of bias; the operative variable
                                          is occurrence x in-pattern power
                                          (loss-share), and data mixing should
                                          target that, not the marginal.
  r_q ~= m8 / R (within a factor of 2) -> a marginal prior exists; flattening
                                          the corpus marginal is necessary.
  anything else                        -> intermediate, report the dose curve.

Contract 2 binds as always: read 2 000 steps for the schedule check (X and Yq
should read alike there) and 10 000 for the fixed point. No knob is tuned
after seeing results.

Run from the repo root:
``.venv/bin/python experiments/4_occurrence/scripts/occurrence.py``.
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
FREQS = (32.0, 64.0, 128.0)
BANDS = tuple(f * PATCH / CTX for f in FREQS)  # 2.0, 4.0, 8.0
MARGINAL = (0.24, 0.24, 0.02)  # the shared corpus marginal, (b2, b4, b8)
PROBE_SEED = 1009
SEED_STRIDE = 1000
GATE = 0.5
TOL = 0.05
RUNS = Path(__file__).resolve().parents[1] / "runs"

# pattern tables: (probability, (P_b2, P_b4, P_b8)); each allocation sums to 0.5
LOUD4 = (0.0, 0.0, 0.5)
LOUD6 = (1.0 / 12.0, 1.0 / 12.0, 1.0 / 3.0)
LOUD8 = (0.125, 0.125, 0.25)
WEAK = (0.25, 0.25, 0.0)
ORBIT = ((0.40, 0.07, 0.03), (0.03, 0.40, 0.07), (0.07, 0.03, 0.40))

ARMS = {
    "x": {"patterns": [(1.0, MARGINAL)], "probe": MARGINAL},
    "y4": {"patterns": [(0.96, WEAK), (0.04, LOUD4)], "probe": LOUD4},
    "y6": {"patterns": [(0.94, WEAK), (0.06, LOUD6)], "probe": LOUD6},
    "y8": {"patterns": [(0.92, WEAK), (0.08, LOUD8)], "probe": LOUD8},
    "c": {"patterns": [(1.0, (0.245, 0.245, 0.01))], "probe": (0.245, 0.245, 0.01)},
    "a": {
        "patterns": [(1.0, (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0))],
        "probe": (1.0 / 6.0,) * 3,
    },
    "b": {"patterns": [(1.0 / 3.0, p) for p in ORBIT], "probe": ORBIT[0]},
}


def check_patterns():
    r"""Every allocation sums to 0.5; the Y marginals match MARGINAL exactly."""
    total = 0.5
    for arm, spec in ARMS.items():
        for p, a in spec["patterns"]:
            assert abs(sum(a) - total) < 1e-9, (arm, a)
            assert 0.0 <= p <= 1.0
        s = sum(p for p, _ in spec["patterns"])
        assert abs(s - 1.0) < 1e-9, (arm, s)
        marg = tuple(sum(p * a[j] for p, a in spec["patterns"]) for j in range(3))
        print(f"  {arm:<4} marginal = {tuple(round(m, 4) for m in marg)}")
    for q, loud in ((0.04, LOUD4), (0.06, LOUD6), (0.08, LOUD8)):
        assert abs(q * loud[2] - MARGINAL[2]) < 1e-9, (q, loud)


def pattern_table(spec):
    r"""``(probs, allocs)`` as tensors: ``probs`` (n_pat,), ``allocs``
    ``(n_pat, n_bands)``.
    """
    probs = torch.tensor([p for p, _ in spec["patterns"]], dtype=torch.float64)
    allocs = torch.tensor([a for _, a in spec["patterns"]], dtype=torch.float64)
    return probs, allocs


def synth(allocs, n, gen):
    r"""Windows from per-window allocations, ``(n, ctx + k)``.

    ``allocs`` (n, n_bands) are the per-window band powers; per-band amplitude
    is ``sqrt(2 P)``, phase uniform, so each window's total power is exactly
    ``sum(allocs)`` and per-window RMS is sqrt of that.
    """
    t = torch.arange(CTX + PATCH, dtype=torch.float32)
    freqs = torch.tensor(FREQS, dtype=torch.float32)
    base = 2 * math.pi * freqs[:, None] * t[None, :] / CTX  # (n_bands, T)
    phase = 2 * math.pi * torch.rand((n, len(BANDS)), generator=gen)
    amp = torch.sqrt(2.0 * allocs.to(torch.float32))  # (n, n_bands)
    return (amp[:, :, None] * torch.cos(base[None, :, :] + phase[:, :, None])).sum(1)


def draw_windows(spec, n, gen_pat, gen_ph):
    r"""Sample pattern indices, then synthesise with paired phases."""
    probs, allocs = pattern_table(spec)
    idx = torch.multinomial(probs.float(), n, replacement=True, generator=gen_pat)
    return synth(allocs[idx], n, gen_ph), idx


def corpus(arm, seed):
    r"""The training marginal: N_PER windows of the arm's pattern mixture."""
    gen_pat = torch.Generator().manual_seed(seed * SEED_STRIDE + 17)
    gen_ph = torch.Generator().manual_seed(seed * SEED_STRIDE + 71)
    x, idx = draw_windows(ARMS[arm], N_PER, gen_pat, gen_ph)
    return x, idx


def probe(arm):
    r"""Clean windows in the arm's discriminating allocation."""
    gen = torch.Generator().manual_seed(PROBE_SEED)
    alloc = torch.tensor([ARMS[arm]["probe"]], dtype=torch.float64).expand(N_PER, -1)
    return synth(alloc, N_PER, gen)


@torch.no_grad()
def band_power(x, b):
    r"""``mean_p |Z_p|^2 / 2``: the band's **total** power."""
    z = matched_amp(x.unfold(1, PATCH, PATCH), b, PATCH)
    return 0.5 * float(z.abs().square().mean())


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


def generator_check():
    r"""No training: marginals exact, occurrences realised, probes matched."""
    out = {}
    for arm, spec in ARMS.items():
        probs, allocs = pattern_table(spec)
        acc = {"rms": [], "marginal": [], "occ": []}
        for seed in SEEDS:
            x, idx = corpus(arm, seed)
            acc["rms"].append(float(x.square().mean().sqrt()))
            acc["marginal"].append([band_power(x, b) for b in BANDS])
            acc["occ"].append(
                [(idx == j).double().mean().item() for j in range(len(probs))]
            )
        prb = probe(arm)
        out[arm] = {
            "rms": mean_std(acc["rms"]),
            "marginal": {
                str(b): mean_std([m[i] for m in acc["marginal"]])
                for i, b in enumerate(BANDS)
            },
            "occ": {
                str(j): mean_std([o[j] for o in acc["occ"]]) for j in range(len(probs))
            },
            "target_marginal": [
                sum(p * a[i] for p, a in spec["patterns"]) for i in range(3)
            ],
            "probe_rms": float(prb.square().mean().sqrt()),
            "probe_power": [band_power(prb, b) for b in BANDS],
        }
    return out


def held_out(arm, seed):
    r"""Held windows in the arm's probe allocation (the pattern at stake)."""
    gen = torch.Generator().manual_seed(seed * SEED_STRIDE + 307)
    alloc = torch.tensor([ARMS[arm]["probe"]], dtype=torch.float64).expand(N_PER, -1)
    return synth(alloc, N_PER, gen)


def train(seed, arm, steps):
    r"""P8's training loop verbatim; only the corpus differs between arms."""
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
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
    r"""Fit gate on the pattern-conditional held set, then the clean probe."""
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
        row[f"r_oracle@{b}"] = fit_oracle(z_held, patches, z, true, b, PATCH)[
            "retention"
        ]
    return row


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: mean_std([r[k] for r in rows]) for k in keys}


def decide(rec):
    r"""The pre-registered fixed-point decision on the Y ladder."""
    y_arms = [a for a in ("y4", "y6", "y8") if a in rec["arms"]]
    x_in = "x" in rec["arms"]
    r = {a: rec["arms"][a]["mean_std"][f"r@{BANDS[2]}"] for a in y_arms}
    pred = {
        "y4": MARGINAL[2] / LOUD4[2],
        "y6": MARGINAL[2] / LOUD6[2],
        "y8": MARGINAL[2] / LOUD8[2],
    }
    unfitted = [a for a in rec["arms"] if not rec["arms"][a]["gate_passed"]]
    r_means = {a: v[0] for a, v in r.items()}
    if unfitted:
        verdict = "inconclusive: arm(s) failed the fit gate"
    elif y_arms and min(r_means.values()) >= 0.9:
        verdict = (
            "conditional account: marginal-matched heterogeneity is free of "
            "bias; the operative variable is occurrence x in-pattern power "
            "(loss-share)"
        )
    elif y_arms and all(abs(r_means[a] - pred[a]) < max(pred[a], 0.05) for a in y_arms):
        verdict = "marginal prior exists: flattening the corpus marginal is necessary"
    else:
        verdict = "intermediate: report the dose curve"
    return {
        "r_b8_on_loud_probe": r,
        "marginal_account_prediction": pred,
        "x_r_b8": rec["arms"]["x"]["mean_std"][f"r@{BANDS[2]}"] if x_in else None,
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
    t0 = time.time()
    arms = {a: ARMS[a] for a in ARMS}
    if args.arms:
        missing = set(args.arms) - set(ARMS)
        assert not missing, f"unknown arms {sorted(missing)}; have {sorted(ARMS)}"
        arms = {a: v for a, v in arms.items() if a in args.arms}
    seeds = tuple(args.seeds) if args.seeds else SEEDS

    print(f"== occurrence (P11): {len(arms)} arms x {len(seeds)} seeds ==", flush=True)
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {HIDDEN} / {LAYERS}L / {HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {args.steps} steps / cpu",
        flush=True,
    )
    print(f"bands f = {FREQS} (b = {BANDS}); shared marginal {MARGINAL}", flush=True)
    print("\npattern table (marginals must match):", flush=True)
    check_patterns()
    print("\ngenerator check (no training yet)", flush=True)
    gen = generator_check()

    rows_by_arm, walls = {}, []
    for arm in arms:
        rows = []
        for seed in seeds:
            t = time.time()
            model, loss = train(seed, arm, args.steps)
            row = evaluate(model, held_out(arm, seed), probe(arm), loss)
            row["seed"] = seed
            row["wall"] = time.time() - t
            walls.append(row["wall"])
            rows.append(row)
            print(
                f"  {arm:<4} seed {seed}  loss {loss:.2e}  var expl "
                f"{row['var_explained']:+.5f}  probe r = "
                + " ".join(f"{row[f'r@{b}']:.4f}" for b in BANDS)
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
            "steps": args.steps,
            "batch": BATCH,
            "lr": LR,
            "n_per": N_PER,
            "seeds": list(seeds),
            "device": "cpu",
            "freqs": list(FREQS),
            "b": list(BANDS),
            "marginal": list(MARGINAL),
            "probe_seed": PROBE_SEED,
            "gate_fraction": GATE,
            "tol": TOL,
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
    gen, arms = rec["generator"], rec["config"]["arms"]
    print("\ngenerator check, no training: realised marginals, occurrences")
    table(
        [
            f"  {'arm':<5}",
            f"{'corpus rms':>11}",
            f"{'probe rms':>10}",
            *[f"{'m@' + str(b):>11}" for b in BANDS],
            *[f"{'tgt@' + str(b):>9}" for b in BANDS],
        ],
        [
            [
                f"  {arm:<5}",
                f"{gen[arm]['rms'][0]:>11.5f}",
                f"{gen[arm]['probe_rms']:>10.5f}",
                *[f"{gen[arm]['marginal'][str(b)][0]:>11.5f}" for b in BANDS],
                *[f"{gen[arm]['target_marginal'][i]:>9.4f}" for i in range(3)],
            ]
            for arm in arms
        ],
    )
    print(
        "  m@b is the realised corpus marginal per band; tgt@b the design value.\n"
        "  All discrimination arms share the marginal (0.24, 0.24, 0.02); per-window\n"
        "  total power is 0.5 everywhere, so RMS reads 0.70711 and every probe is\n"
        "  shape-matched to its own discriminating allocation."
    )


def print_probe(rec):
    print("\nper arm: the clean probe (r, then Fredformer's dk), and the oracle")
    table(
        [
            f"  {'arm':<5}",
            f"{'probe alloc':>26}",
            *[f"{'r@' + str(b):>16}" for b in BANDS],
            *[f"{'dk@' + str(b):>16}" for b in BANDS],
            *[f"{'orc@' + str(b):>9}" for b in BANDS],
        ],
        [
            [
                f"  {arm:<5}",
                f"{str(tuple(round(p, 3) for p in a['probe'])):>26}",
                *[ms(s[f"r@{b}"], 16) for b in BANDS],
                *[ms(s[f"dk@{b}"], 16) for b in BANDS],
                *[f"{s[f'r_oracle@{b}'][0]:>9.4f}" for b in BANDS],
            ]
            for arm, a in rec["arms"].items()
            for s in (a["mean_std"],)
        ],
    )


def print_decision(rec):
    d = rec["decision"]
    print("\npre-registered decision (fixed point, b8 on the Y loud probes)")
    if d["x_r_b8"] is not None:
        print(f"  X  r@8 (homogeneous reference): {ms(d['x_r_b8'])}")
    for arm, r in d["r_b8_on_loud_probe"].items():
        print(
            f"  {arm:<4} r@8 = {ms(r)}   marginal-account prediction "
            f"{d['marginal_account_prediction'][arm]:.3f}"
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
    return RUNS / f"occurrence{tag}.json"


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
