r"""P17: rank -- is the manifold's dimension a cost, and is it proportionate?

Why this run exists. P16 settled position: two carriers with identical share
profiles read identically to |dr| = 0.008. The one axis nothing has touched is
the *dimension of the state the context must resolve* -- the rank of the map
``d y_{T+1} / d y_{t<T}``. PM offers one operation that changes rank and provably
nothing else:

    y(t) = cos(2 pi b_h t / k + beta * cos(2 pi b_m t / k + phi))

The n-th sideband carries power ``J_n(beta)^2 / 2`` -- **independent of phi** --
while its phase carries ``n phi``. So:

    phi fixed            state = (tau,),        rank 1, sideband phases identical
                         in every window
    phi random per window state = (tau, phi),   rank 2, sideband powers bit-identical

Powers, positions, carrier, total power, bandwidth: all unchanged. Only the
relative-phase structure between sidebands varies, and the model must resolve it
from the context.

Why phi and not the carrier's phase: the carrier's phase psi enters linearly and
is absorbed by a time shift (``psi -> psi + 2 pi b_h d`` is ``tau -> tau + d``),
so it is not a degree of freedom at all. The modulator's phase is.

**This is a dynamics experiment.** In this corpus the target is a deterministic
function of the context, the oracle already reads 0.98-1.00 at hidden 32 (P16),
and the corpus is fit to 97-98% of variance by 2 000 steps -- so any rank effect
is a *time* effect, not a representational one. The readout is therefore the
transient, sampled every 100 steps to 2 000, and the 10 000-step state serves
only as the convergence claim.

Phase ladder (K = number of values phi may take; state dimension grows with it):

    phiK1     phi = 0                     rank 1  (the P16 configuration)
    phiK2     phi in {0, pi}              1 + 1 bit
    phiK4     phi in {0, pi/2, pi, 3pi/2} 1 + 2 bits
    phiInf    phi ~ Uniform[0, 2 pi)      rank 2

Two carriers, b_h in {6, 10}, so a rank cost can be tested for proportionate
behaviour across position (P16's question, now crossed with rank). beta = 1.5
(sharess .262 / .311 / .054, every significant line well above the noise floor).

One fairness point, recorded because it is what makes the comparison legitimate:
with phi fixed the corpus **already sweeps its whole one-dimensional state
space** -- window offsets tau are uniform over 1 024 samples and the modulator
runs 32 cycles in that span, so the window-start modulator phase covers [0, 2 pi)
uniformly. In (theta_c, theta_m) coordinates the fixed-phi windows lie on the
curve ``theta_c = 8 (theta_m - phi)``, and the random-phi windows fill the torus.
Both arms therefore span a full family; the random arm adds one orthogonal
dimension rather than "more samples of the same thing".

Pre-registered decision, written before the run into this experiment's README.md:

  A. **design holds** -- the measured per-band shares match ``J_n(beta)^2`` in
     every arm and are equal between arms of the same carrier (the rank
     operation provably did not move the spectrum);
  B. **capacity** -- the frozen-latent oracle >= 0.9 at every significant line of
     every arm at the final checkpoint;
  C. **convergence** -- at 10 000 steps every significant line of every arm reads
     r >= 0.95, i.e. the endpoint is rank-independent;
  D. **transient (the discriminator)** -- t80 (steps to r = 0.8) per line; the
     rank effect is the ratio t80(K) / t80(K=1) at matched lines. Present if any
     line exceeds **1.25x**, reported as a dose-response over K;
  E. **interaction** -- |log2 effect at b6 - log2 effect at b10| against the same
     1.25x threshold; exceeded means the rank cost is **disproportionate** across
     position;
  F. **residual** -- if D is present but C holds, the cost is purely dynamical (a
     time constant); if C fails, a gap survives convergence and is reported as
     such.

Verdicts: B failing -> capacity; C failing -> a gap survives convergence; D absent
-> no rank effect; D present with E exceeded -> disproportionate; D present with E
within tolerance -> a uniform time cost.

Run from the repo root:
``.venv/bin/python experiments/9_pm_rank/scripts/pm_rank.py``.
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
from fbias.probes import delta_k, fit_oracle, matched_amp, phase_error, retention

CTX, PATCH = 512, 32
HIDDEN, LAYERS, HEADS = 32, 2, 4
STEPS, BATCH, LR = 10000, 64, 1e-2
CKPT, CKPT_FINE, CKPT_FINE_UNTIL = 500, 100, 2000
N_PER = 256
SEEDS = (0, 1, 2)
BANDS = tuple(float(b) for b in range(1, 17))
WELL = BANDS[:-1]  # b16 is Nyquist: excluded
MOD_B = 1.0
BETA = 1.5
CARRIERS = (6.0, 10.0)
PHASE_SETS = {
    "phiK1": (0.0,),
    "phiK2": (0.0, math.pi),
    "phiK4": (0.0, math.pi / 2, math.pi, 3 * math.pi / 2),
    "phiInf": None,  # continuous
}
TAU_MAX = 1024.0
PROBE_SEED = 2609
SEED_STRIDE = 1000
GATE = 0.5
ORACLE_TOL = 0.9
CONV_TOL = 0.95
RANK_TOL = 1.25
SIG_SHARE = 0.01
RUNS = Path(__file__).resolve().parents[1] / "runs"


def arm_id(carrier, kname):
    return f"b{carrier:g}_{kname}"


ARMS = {arm_id(c, k): {"carrier": c, "phases": k} for c in CARRIERS for k in PHASE_SETS}


def bessel_j(n, beta, m=20001):
    tau = torch.linspace(0, math.pi, m, dtype=torch.float64)
    return float(
        torch.trapezoid(torch.cos(n * tau - beta * torch.sin(tau)), tau) / math.pi
    )


def sidebands(carrier, beta, thresh=SIG_SHARE):
    r"""``[(n, band, share)]`` -- a function of (carrier, beta) only, never of the
    phase set, which is the point of the design."""
    out = []
    for n in range(-10, 11):
        band = carrier + n * MOD_B
        if band < 1 or band > 15 or band not in WELL:
            continue
        share = bessel_j(abs(n), beta) ** 2
        if share >= thresh:
            out.append((n, band, share))
    return out


def phases(kname, n, gen):
    r"""Per-window modulator phases for one arm."""
    vals = PHASE_SETS[kname]
    if vals is None:
        return 2 * math.pi * torch.rand(n, generator=gen)
    idx = torch.randint(len(vals), (n,), generator=gen)
    return torch.tensor(vals, dtype=torch.float32)[idx]


def pm_windows(carrier, beta, kname, n, seed, tau_max=TAU_MAX):
    r"""Windows of the PM signal, one random modulator phase per window."""
    gen_tau = torch.Generator().manual_seed(seed)
    gen_phi = torch.Generator().manual_seed(seed * 7 + 3)
    tau = tau_max * torch.rand(n, generator=gen_tau)
    phi = phases(kname, n, gen_phi)
    t = torch.arange(CTX + PATCH, dtype=torch.float32)
    big = tau.unsqueeze(1) + t.unsqueeze(0)
    ang = 2 * math.pi * carrier * big / PATCH
    mod = 2 * math.pi * MOD_B * big / PATCH + phi.unsqueeze(1)
    return torch.cos(ang + beta * torch.cos(mod))


def corpus(arm, seed):
    cfg = ARMS[arm]
    return pm_windows(
        cfg["carrier"], BETA, cfg["phases"], N_PER, seed * SEED_STRIDE + 71
    )


def probe(arm):
    cfg = ARMS[arm]
    return pm_windows(cfg["carrier"], BETA, cfg["phases"], N_PER, PROBE_SEED)


def held_out(arm, seed):
    cfg = ARMS[arm]
    return pm_windows(
        cfg["carrier"], BETA, cfg["phases"], N_PER, seed * SEED_STRIDE + 307
    )


@torch.no_grad()
def band_power(x, b):
    z = matched_amp(x.unfold(1, PATCH, PATCH), b, PATCH)
    return 0.5 * float(z.abs().square().mean())


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


def generator_check():
    r"""Check A: measured shares match ``J_n^2`` and are arm-independent."""
    out = {}
    for arm in ARMS:
        cfg = ARMS[arm]
        pred = {str(b): s for _, b, s in sidebands(cfg["carrier"], BETA)}
        acc = {"rms": [], "power": []}
        for seed in SEEDS:
            x = corpus(arm, seed)
            acc["rms"].append(float(x.square().mean().sqrt()))
            acc["power"].append([band_power(x, b) for b in WELL])
        measured = {
            str(b): mean_std([p[i] for p in acc["power"]])[0] / 0.5
            for i, b in enumerate(WELL)
        }
        out[arm] = {
            "rms": mean_std(acc["rms"]),
            "predicted": pred,
            "measured": measured,
            "share_err": max(abs(measured[b] - s) for b, s in pred.items()),
            "probe_rms": float(probe(arm).square().mean().sqrt()),
        }
    # cross-arm equality of the measured shares, per carrier
    cross = {}
    for c in CARRIERS:
        ids = [arm_id(c, k) for k in PHASE_SETS]
        worst = 0.0
        for b in out[ids[0]]["predicted"]:
            vals = [out[a]["measured"][b] for a in ids]
            worst = max(worst, max(vals) - min(vals))
        cross[f"b{c:g}"] = worst
    return {"arms": out, "cross_arm_share_spread": cross}


def train(seed, arm, args):
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_attn_heads=HEADS,
    )
    data = corpus(arm, seed)
    prb, held = probe(arm), held_out(arm, seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss = torch.tensor(0.0)
    marks = []
    for step in range(1, args.steps + 1):
        x = data[torch.randint(N_PER, (BATCH,), generator=gen)]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        due = (
            step <= args.ckpt_fine_until and step % args.ckpt_fine == 0
        ) or step % args.ckpt == 0
        if due:
            row = evaluate(model, held, prb)
            row["step"] = step
            marks.append(row)
    return model, marks


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
        row[f"dphi@{b}"] = float(phase_error(f_hat, true[:, -1], b, PATCH).mean())
        row[f"r_oracle@{b}"] = fit_oracle(z_held, patches, z, true, b, PATCH)[
            "retention"
        ]
    for b in WELL:
        row[f"rm@{b}"] = float(retention(pred[:, -2], target[:, -1], b, PATCH).mean())
    model.train()
    return row


def t80(marks, b, thresh=0.8):
    r"""First checkpoint at which band ``b`` reaches ``thresh``; None if never."""
    for m in marks:
        if m[f"r@{b}"] >= thresh:
            return float(m["step"])
    return None


def decide(rec):
    arms = rec["arms"]
    gen = rec["generator"]
    # B: capacity
    oracle_min = {}
    for a in arms:
        cfg = ARMS[a]
        lines = [b for _, b, _ in sidebands(cfg["carrier"], BETA)]
        oracle_min[a] = min(arms[a]["marks"][-1][f"r_oracle@{b}"] for b in lines)
    bad = [a for a, o in oracle_min.items() if o < ORACLE_TOL]

    # C: convergence at the final checkpoint
    conv = {}
    for a in arms:
        cfg = ARMS[a]
        lines = [b for _, b, _ in sidebands(cfg["carrier"], BETA)]
        conv[a] = min(arms[a]["marks"][-1][f"r@{b}"] for b in lines)
    not_conv = [a for a, v in conv.items() if v < CONV_TOL]

    # D/E: t80 ratios, matched lines within a carrier
    ratio, effect = {}, {}
    for c in CARRIERS:
        base = arm_id(c, "phiK1")
        if base not in arms:
            continue
        for k in PHASE_SETS:
            a = arm_id(c, k)
            if k == "phiK1" or a not in arms:
                continue
            per_line = {}
            for _, b, _ in sidebands(c, BETA):
                t1, tk = t80(arms[base]["marks"], b), t80(arms[a]["marks"], b)
                if t1 and tk:
                    per_line[str(b)] = tk / t1
            ratio[a] = per_line
            effect[a] = max(per_line.values()) if per_line else 1.0
    worst_effect = max(effect.values()) if effect else 1.0

    inter = {}
    for k in PHASE_SETS:
        if k == "phiK1":
            continue
        if arm_id(6.0, k) not in effect or arm_id(10.0, k) not in effect:
            continue
        e6, e10 = effect[arm_id(6.0, k)], effect[arm_id(10.0, k)]
        inter[k] = abs(math.log2(max(e6, 1e-9)) - math.log2(max(e10, 1e-9)))
    worst_inter = max(inter.values()) if inter else 0.0
    inter_thresh = abs(math.log2(RANK_TOL))

    if bad:
        verdict = "capacity: oracle short in " + ", ".join(sorted(bad))
    elif not_conv:
        verdict = "a gap survives convergence in " + ", ".join(sorted(not_conv))
    elif worst_effect <= RANK_TOL:
        verdict = "no rank effect: identical dynamics at rank 1 and rank 2"
    elif worst_inter > inter_thresh:
        verdict = "disproportionate: the rank cost depends on carrier position"
    else:
        verdict = "a uniform time cost: rank slows learning equally at both positions"
    return {
        "oracle_min": oracle_min,
        "final_min_r": conv,
        "bad_oracle": bad,
        "not_converged": not_conv,
        "t80_ratio": ratio,
        "rank_effect": effect,
        "worst_rank_effect": worst_effect,
        "interaction_log2": inter,
        "worst_interaction_log2": worst_inter,
        "interaction_threshold_log2": inter_thresh,
        "cross_arm_share_spread": gen["cross_arm_share_spread"],
        "share_error": {a: gen["arms"][a]["share_err"] for a in arms},
        "verdict": verdict,
    }


def table(head, rows):
    line = " ".join(head)
    print(line)
    print("  " + "-" * (len(line) - 2))
    for r in rows:
        print(" ".join(r))


def build(args):
    t0 = time.time()
    arms = {a: ARMS[a] for a in ARMS}
    if args.arms:
        missing = set(args.arms) - set(ARMS)
        assert not missing, f"unknown arms {sorted(missing)}; have {sorted(ARMS)}"
        arms = {a: v for a, v in arms.items() if a in args.arms}
    seeds = tuple(args.seeds) if args.seeds else SEEDS

    print(f"== pm_rank (P17): {len(arms)} arms x {len(seeds)} seeds ==", flush=True)
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {args.hidden} / {args.layers}L / "
        f"{HEADS} heads | SGD lr {LR} / batch {BATCH} / {args.steps} steps",
        flush=True,
    )
    print(
        f"checkpoints every {args.ckpt_fine} to {args.ckpt_fine_until}, then "
        f"{args.ckpt} | cpu",
        flush=True,
    )
    print(
        f"modulator b_m = {MOD_B:g}, beta = {BETA:g}; carriers {CARRIERS}; "
        f"phase sets {list(PHASE_SETS)}",
        flush=True,
    )
    print("\ngenerator check (no training yet)", flush=True)
    gen = generator_check()
    errs = ", ".join(
        f"{a}:{gen['arms'][a]['share_err']:.6f}" for a in sorted(gen["arms"])
    )
    print(f"  share error vs J_n^2       {errs}", flush=True)
    spreads = ", ".join(
        f"{c}:{v:.6f}" for c, v in gen["cross_arm_share_spread"].items()
    )
    print(f"  cross-arm share spread     {spreads}", flush=True)

    rec = {
        "config": {
            "ctx": CTX,
            "patch": PATCH,
            "hidden": args.hidden,
            "layers": args.layers,
            "heads": HEADS,
            "steps": args.steps,
            "ckpt": args.ckpt,
            "ckpt_fine": args.ckpt_fine,
            "ckpt_fine_until": args.ckpt_fine_until,
            "batch": BATCH,
            "lr": LR,
            "n_per": N_PER,
            "seeds": list(seeds),
            "bands": list(BANDS),
            "mod_b": MOD_B,
            "beta": BETA,
            "carriers": list(CARRIERS),
            "phase_sets": {
                k: (list(v) if v else "continuous") for k, v in PHASE_SETS.items()
            },
            "tau_max": TAU_MAX,
            "probe_seed": PROBE_SEED,
            "gate_fraction": GATE,
            "oracle_tol": ORACLE_TOL,
            "conv_tol": CONV_TOL,
            "rank_tol": RANK_TOL,
            "sig_share": SIG_SHARE,
            "readout": "pred[:, -2] vs pat(x)[:, -1]",
            "arms": {a: ARMS[a] for a in arms},
        },
        "generator": gen,
        "arms": {},
        "wall": 0.0,
    }
    for arm in arms:
        t = time.time()
        print(f"  {arm}", flush=True)
        marks = train_seeds(arm, seeds, args)
        rec["arms"][arm] = {"marks": marks, "wall": time.time() - t}
    rec["wall"] = time.time() - t0
    rec["decision"] = decide(rec)
    return rec


def train_seeds(arm, seeds, args):
    r"""Train one arm per seed; return the seed-mean trajectory."""
    per_seed = []
    for seed in seeds:
        _, marks = train(seed, arm, args)
        per_seed.append(marks)
    steps = [m["step"] for m in per_seed[0]]
    keys = [k for k in per_seed[0][0] if isinstance(per_seed[0][0][k], float)]
    out = []
    for i, s in enumerate(steps):
        row = {"step": s}
        for k in keys:
            vals = [p[i][k] for p in per_seed]
            row[k] = sum(vals) / len(vals)
        out.append(row)
    return out


def print_trajectory(rec):
    print("\nshared trajectory, per arm: r on the significant lines")
    for arm in rec["arms"]:
        cfg = ARMS[arm]
        lines = [(n, b) for n, b, _ in sidebands(cfg["carrier"], BETA)]
        marks = rec["arms"][arm]["marks"]
        shown = [
            m for m in marks if m["step"] in (100, 200, 500, 1000, 2000, 5000, 10000)
        ]
        table(
            [f"  {'step':>6}", *[f"{'r@b' + str(int(b)):>9}" for _, b in lines]],
            [
                [
                    f"  {int(m['step']):>6}",
                    *[f"{m[f'r@{b}']:>9.3f}" for _, b in lines],
                ]
                for m in shown
            ],
        )


def print_decision(rec):
    d = rec["decision"]
    print("\npre-registered checks")
    print(
        f"  A. design: max share error vs J_n^2 = "
        f"{max(d['share_error'].values()):.6f}; "
        f"cross-arm spread = {max(d['cross_arm_share_spread'].values()):.6f}"
    )
    print(f"  B. oracle minimum (>= {ORACLE_TOL}):")
    for a, o in sorted(d["oracle_min"].items(), key=lambda kv: kv[1]):
        flag = "" if o >= ORACLE_TOL else "   <-- SHORT"
        print(f"       {a:<15} {o:.4f}{flag}")
    print(f"  C. convergence (final min r >= {CONV_TOL}):")
    for a, v in sorted(d["final_min_r"].items(), key=lambda kv: kv[1]):
        flag = "" if v >= CONV_TOL else "   <-- NOT CONVERGED"
        print(f"       {a:<15} {v:.4f}{flag}")
    print(f"  D. rank effect (t80 ratio vs phiK1, threshold {RANK_TOL}):")
    for a, v in sorted(d["rank_effect"].items()):
        print(f"       {a:<15} {v:.3f}")
    print(
        f"  E. interaction |log2 effect(b6) - log2 effect(b10)| "
        f"(threshold {d['interaction_threshold_log2']:.3f}):"
    )
    for k, v in sorted(d["interaction_log2"].items()):
        flag = "" if v <= d["interaction_threshold_log2"] else "   <-- DISPROPORTIONATE"
        print(f"       {k:<15} {v:.3f}{flag}")
    print(f"  -> {d['verdict']}")


def report(rec):
    print_trajectory(rec)
    print_decision(rec)


def out_path(args):
    tag = f"_{args.tag}" if args.tag else ""
    return RUNS / f"pm_rank{tag}.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=None, help="subset of arm ids")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="subset of seeds")
    p.add_argument("--steps", type=int, default=STEPS, help="training steps")
    p.add_argument("--ckpt", type=int, default=CKPT, help="checkpoint interval")
    p.add_argument("--ckpt-fine", type=int, default=CKPT_FINE, help="fine interval")
    p.add_argument(
        "--ckpt-fine-until", type=int, default=CKPT_FINE_UNTIL, help="end of fine phase"
    )
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
