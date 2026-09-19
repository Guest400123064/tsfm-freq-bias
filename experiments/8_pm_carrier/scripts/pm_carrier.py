r"""P16: carrier invariance -- one modulator, two carriers, several beta.

Why this run exists. The whole energy branch (P8-P15) says a band's learning
schedule is set by its share of the loss, and P13 found share sufficing to
explain 96% of the variance across ten spectra. What has never been tested is
whether the *band's position* matters once the share is held fixed. The PM
constructor gives an exact way to hold it fixed:

    y(t) = cos(2 pi b_h t / k + beta * cos(2 pi b_m t / k))

with one common modulator (``b_m = 1`` cycle/patch, phase fixed) and one common
power (a cosine has power 1/2 at every beta -- constant envelope). The sidebands
sit at ``b_h +- n b_m`` with powers ``J_n(beta)^2``, which depend only on the
modulation index: **two different carriers at the same beta have bit-identical
share profiles in the carrier frame**. Moving b_h moves only the positions.

So any difference between two carriers at matched n is a frequency-position
effect that no share account can produce -- and, since only the *relative* phase
of the sidebands carries the chirp, the natural readout is Delta_k, not r.

Arms. carriers b_h in {6, 10}, beta in {1.0, 1.5, 2.405} -- six arms. The
modulator is the same function in all of them and the phase is fixed, so the
map's rank (the dimension of the state the context must resolve) is 1
everywhere; this run holds rank fixed and moves position and beta.

Significant sidebands (share >= 1%, all inside b3..b13, clear of Nyquist):

    beta = 1.000    n = 0, 1, 2      shares .586 .194 .013
    beta = 1.500    n = 0, 1, 2      shares .262 .311 .054
    beta = 2.405    n = 1, 2, 3      shares .270 .186 .040   (J_0 = 0)

Probe. A fresh set of **PM windows** (new random offsets, same carrier/beta). It
must not be a per-band amplitude profile with random phases: the sidebands'
relative phases *are* the signal here, and scrambling them would destroy exactly
the structure this run measures. Every window is noiseless; the probe differs
from the corpus only in its draw of time offsets.

Pre-registered decision, written before the run into this experiment's README.md,
at 10 000 steps (2 000-step leg reported for the schedule):

  1. **capacity** -- the frozen-latent oracle >= 0.9 at every significant
     sideband of every arm at the final checkpoint;
  2. **carrier invariance** (primary) -- for every beta and every matched n,
     |r(b_h = 6, n) - r(b_h = 10, n)| <= 0.05 and
     |dk(b_h = 6, n) - dk(b_h = 10, n)| <= 0.08;
  3. **share ordering** -- pooling all arms, Spearman(log share, deficit) is
     <= -0.8, i.e. the beta response is the share response;
  4. reported but not a decision rule: the non-monotone prediction at beta =
     2.405, whose +-1 share (.270) is *lower* than at beta = 2.0 (.333), so its
     +-1 sidebands should be learned more slowly than beta = 2.0's.

Verdicts: capacity failure -> inconclusive; 2 failing -> the band's position
matters beyond its share (a genuinely new effect for this project); 2 and 3
passing -> carrier-invariant and share-ordered.

Run from the repo root:
``.venv/bin/python experiments/8_pm_carrier/scripts/pm_carrier.py``.
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
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
BANDS = tuple(float(b) for b in range(1, 17))
WELL = BANDS[:-1]  # b16 is Nyquist: excluded
MOD_B = 1.0  # modulator frequency, cycles per patch
PHI = 0.0  # fixed modulator phase -> rank 1
CARRIERS = (6.0, 10.0)
BETAS = (1.0, 1.5, 2.405)
TAU_MAX = 1024.0  # window offsets are drawn from [0, TAU_MAX)
PROBE_SEED = 2203
SEED_STRIDE = 1000
GATE = 0.5
ORACLE_TOL = 0.9
R_TOL = 0.05
DK_TOL = 0.08
SPEARMAN_TOL = 0.8
SIG_SHARE = 0.01
RUNS = Path(__file__).resolve().parents[1] / "runs"

ARMS = {
    f"b{bh:g}_beta{b:g}": {"carrier": bh, "beta": b} for bh in CARRIERS for b in BETAS
}


def bessel_j(n, beta, m=20001):
    r"""``J_n(beta)`` by the integral representation, no scipy."""
    tau = torch.linspace(0, math.pi, m, dtype=torch.float64)
    return float(
        torch.trapezoid(torch.cos(n * tau - beta * torch.sin(tau)), tau) / math.pi
    )


def sidebands(carrier, beta, thresh=SIG_SHARE):
    r"""``[(n, band, share)]`` for the sidebands above ``thresh``.

    A PM cosine splits into lines at ``b_h + n b_m`` with power
    ``J_n(beta)^2 / 2`` against a total of 1/2, so the share is ``J_n(beta)^2``
    -- independent of the carrier, which is what makes this design confound-free.
    """
    out = []
    for n in range(-10, 11):
        band = carrier + n * MOD_B
        if band < 1 or band > 15 or band not in WELL:
            continue
        share = bessel_j(abs(n), beta) ** 2
        if share >= thresh:
            out.append((n, band, share))
    return out


def pm_windows(carrier, beta, n, seed, tau_max=TAU_MAX):
    r"""Windows of the PM signal at random offsets, ``(n, ctx + k)``.

    Constant envelope, so every window has power exactly 1/2 and the per-band
    powers are the Bessel shares; only the time offset varies between windows.
    """
    gen = torch.Generator().manual_seed(seed)
    tau = tau_max * torch.rand(n, generator=gen)
    t = torch.arange(CTX + PATCH, dtype=torch.float32)
    big = tau.unsqueeze(1) + t.unsqueeze(0)
    ang = 2 * math.pi * carrier * big / PATCH
    mod = 2 * math.pi * MOD_B * big / PATCH + PHI
    return torch.cos(ang + beta * torch.cos(mod))


def corpus(arm, seed):
    cfg = ARMS[arm]
    return pm_windows(cfg["carrier"], cfg["beta"], N_PER, seed * SEED_STRIDE + 71)


def probe(arm):
    cfg = ARMS[arm]
    return pm_windows(cfg["carrier"], cfg["beta"], N_PER, PROBE_SEED)


def held_out(arm, seed):
    cfg = ARMS[arm]
    return pm_windows(cfg["carrier"], cfg["beta"], N_PER, seed * SEED_STRIDE + 307)


@torch.no_grad()
def band_power(x, b):
    z = matched_amp(x.unfold(1, PATCH, PATCH), b, PATCH)
    return 0.5 * float(z.abs().square().mean())


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


def generator_check():
    r"""Measured band powers against the Bessel prediction, per arm."""
    out = {}
    for arm in ARMS:
        cfg = ARMS[arm]
        pred = {str(b): s for _, b, s in sidebands(cfg["carrier"], cfg["beta"])}
        acc = {"rms": [], "power": [], "off": []}
        for seed in SEEDS:
            x = corpus(arm, seed)
            acc["rms"].append(float(x.square().mean().sqrt()))
            acc["power"].append([band_power(x, b) for b in WELL])
            # power leaking into bands the theory says are empty
            acc["off"].append(
                max(
                    band_power(x, b) / 0.5
                    for b in WELL
                    if str(b) not in pred and abs(b - cfg["carrier"]) <= 6
                )
            )
        out[arm] = {
            "rms": mean_std(acc["rms"]),
            "predicted": pred,
            "measured": {
                str(b): mean_std([p[i] for p in acc["power"]])[0] / 0.5
                for i, b in enumerate(WELL)
            },
            "max_off_band_share": mean_std(acc["off"]),
            "probe_rms": float(probe(arm).square().mean().sqrt()),
        }
    return out


def train(seed, arm, steps, hidden, layers):
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
        row[f"dphi@{b}"] = float(phase_error(f_hat, true[:, -1], b, PATCH).mean())
        row[f"r_oracle@{b}"] = fit_oracle(z_held, patches, z, true, b, PATCH)[
            "retention"
        ]
    return row


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: mean_std([r[k] for r in rows]) for k in keys}


def spearman(x, y):
    rx = {k: i for i, k in enumerate(sorted(range(len(x)), key=lambda i: x[i]))}
    ry = {k: i for i, k in enumerate(sorted(range(len(y)), key=lambda i: y[i]))}
    a = [float(rx[i]) for i in range(len(x))]
    b = [float(ry[i]) for i in range(len(y))]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else 0.0


def profile(rec, arm):
    r"""``{n: (band, share, r, dk, r_oracle)}`` for one arm's significant lines."""
    cfg = ARMS[arm]
    s = rec["arms"][arm]["mean_std"]
    out = {}
    for n, band, share in sidebands(cfg["carrier"], cfg["beta"]):
        out[n] = {
            "band": band,
            "share": share,
            "r": s[f"r@{band}"][0],
            "dk": s[f"dk@{band}"][0],
            "oracle": s[f"r_oracle@{band}"][0],
        }
    return out


def decide(rec):
    arms = rec["arms"]
    prof = {a: profile(rec, a) for a in arms}
    oracle_min = {a: min(v["oracle"] for v in prof[a].values()) for a in arms}
    unfitted = [a for a in arms if not arms[a]["gate_passed"]]
    bad_oracle = [a for a, o in oracle_min.items() if o < ORACLE_TOL]

    inv = {}
    for beta in BETAS:
        a6, a10 = f"b6_beta{beta:g}", f"b10_beta{beta:g}"
        if a6 not in prof or a10 not in prof:
            continue
        ns = sorted(set(prof[a6]) & set(prof[a10]))
        inv[str(beta)] = {
            "n": ns,
            "dr": {str(n): prof[a6][n]["r"] - prof[a10][n]["r"] for n in ns},
            "ddk": {str(n): prof[a6][n]["dk"] - prof[a10][n]["dk"] for n in ns},
        }
    worst_dr = max(
        (abs(v) for b in inv.values() for v in b["dr"].values()), default=0.0
    )
    worst_ddk = max(
        (abs(v) for b in inv.values() for v in b["ddk"].values()), default=0.0
    )

    pts = [(p["share"], 1 - p["r"]) for a in prof for p in prof[a].values()]
    sp = (
        spearman([math.log(s) for s, _ in pts], [d for _, d in pts])
        if len(pts) > 2
        else 0.0
    )

    if unfitted or bad_oracle:
        verdict = "inconclusive: fit gate or oracle short"
    elif worst_dr > R_TOL or worst_ddk > DK_TOL:
        verdict = "the band's position matters beyond its share"
    elif sp <= -SPEARMAN_TOL:
        verdict = "carrier-invariant and share-ordered: no position effect"
    else:
        verdict = "intermediate: carrier-invariant but not share-ordered"
    return {
        "oracle_min": oracle_min,
        "carrier_invariance": inv,
        "worst_dr": worst_dr,
        "worst_ddk": worst_ddk,
        "spearman_share_vs_deficit": sp,
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
    arms = {a: ARMS[a] for a in ARMS}
    if args.arms:
        missing = set(args.arms) - set(ARMS)
        assert not missing, f"unknown arms {sorted(missing)}; have {sorted(ARMS)}"
        arms = {a: v for a, v in arms.items() if a in args.arms}
    seeds = tuple(args.seeds) if args.seeds else SEEDS

    print(f"== pm_carrier (P16): {len(arms)} arms x {len(seeds)} seeds ==", flush=True)
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {args.hidden} / {args.layers}L / "
        f"{HEADS} heads | SGD lr {LR} / batch {BATCH} / {args.steps} steps / cpu",
        flush=True,
    )
    print(
        f"modulator b_m = {MOD_B:g} (fixed phase {PHI:g}); carriers {CARRIERS}; "
        f"beta {BETAS}",
        flush=True,
    )
    print("\nsignificant sidebands, carrier frame (share = J_n(beta)^2):", flush=True)
    for arm, cfg in arms.items():
        lines = " ".join(
            f"n={n:+d}@b{band:g}:{share:.3f}"
            for n, band, share in sidebands(cfg["carrier"], cfg["beta"])
        )
        print(f"  {arm:<14} {lines}", flush=True)
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
            car = ARMS[arm]["carrier"]
            rb = row[f"r@{car}"]
            print(
                f"  {arm:<14} seed {seed}  loss {loss:.2e}  var expl "
                f"{row['var_explained']:+.5f}  r@carrier {rb:.4f}  "
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
            "mod_b": MOD_B,
            "phi": PHI,
            "carriers": list(CARRIERS),
            "betas": list(BETAS),
            "tau_max": TAU_MAX,
            "probe_seed": PROBE_SEED,
            "gate_fraction": GATE,
            "oracle_tol": ORACLE_TOL,
            "r_tol": R_TOL,
            "dk_tol": DK_TOL,
            "spearman_tol": SPEARMAN_TOL,
            "sig_share": SIG_SHARE,
            "readout": "pred[:, -2] vs pat(x)[:, -1]",
            "arms": {a: ARMS[a] for a in arms},
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


def share_err(gen_arm):
    return max(abs(gen_arm["measured"][b] - s) for b, s in gen_arm["predicted"].items())


def print_generator(rec):
    gen = rec["generator"]
    print("\ngenerator check: measured shares against the Bessel prediction")
    table(
        [
            f"  {'arm':<14}",
            f"{'corpus rms':>11}",
            f"{'probe rms':>10}",
            f"{'max |share err|':>17}",
            f"{'off-band share':>15}",
        ],
        [
            [
                f"  {arm:<14}",
                f"{gen[arm]['rms'][0]:>11.5f}",
                f"{gen[arm]['probe_rms']:>10.5f}",
                f"{share_err(gen[arm]):>17.4f}",
                f"{gen[arm]['max_off_band_share'][0]:>15.4f}",
            ]
            for arm in gen
        ],
    )
    print(
        "  max |share err| compares the matched filter's reading with J_n(beta)^2;\n"
        "  off-band share is the largest measured share in a band the theory says is\n"
        "  empty (within +-6 bands of the carrier) -- leakage and discretisation."
    )


def print_profiles(rec):
    for arm in rec["arms"]:
        prof = profile(rec, arm)
        cfg = ARMS[arm]
        print(f"\narm {arm}  (carrier b{cfg['carrier']:g}, beta {cfg['beta']:g})")
        table(
            [
                f"  {'n':>3}",
                f"{'band':>5}",
                f"{'share':>8}",
                f"{'r':>16}",
                f"{'dk':>16}",
                f"{'phase err':>12}",
                f"{'oracle':>9}",
            ],
            [
                [
                    f"  {n:>+3d}",
                    f"{v['band']:>5.0f}",
                    f"{v['share']:>8.4f}",
                    ms(rec["arms"][arm]["mean_std"][f"r@{v['band']}"], 16),
                    ms(rec["arms"][arm]["mean_std"][f"dk@{v['band']}"], 16),
                    f"{rec['arms'][arm]['mean_std'][f'dphi@{v["band"]}'][0]:>12.4f}",
                    f"{v['oracle']:>9.4f}",
                ]
                for n, v in sorted(prof.items(), reverse=True)
            ],
        )


def print_decision(rec):
    d = rec["decision"]
    print("\npre-registered checks")
    print(f"  1. oracle minimum over significant sidebands (>= {ORACLE_TOL}):")
    for a, o in sorted(d["oracle_min"].items(), key=lambda kv: kv[1]):
        flag = "" if o >= ORACLE_TOL else "   <-- SHORT"
        print(f"       {a:<14} {o:.4f}{flag}")
    print(
        f"  2. carrier invariance (primary): worst |dr| = {d['worst_dr']:.4f} "
        f"(tol {R_TOL}), worst |d(dk)| = {d['worst_ddk']:.4f} (tol {DK_TOL})"
    )
    print("     per beta, matched n:")
    for beta, v in d["carrier_invariance"].items():
        dr = " ".join(f"n={int(n):+d}:{x:+.3f}" for n, x in v["dr"].items())
        dk = " ".join(f"n={int(n):+d}:{x:+.3f}" for n, x in v["ddk"].items())
        print(f"       beta {beta:<6} dr[{dr}]")
        print(f"       {'':<12} ddk[{dk}]")
    print(
        f"  3. share ordering, pooled: Spearman(log share, deficit) = "
        f"{d['spearman_share_vs_deficit']:+.3f} (<= -{SPEARMAN_TOL} required)"
    )
    print(
        "  arms failing the fit gate: "
        + (", ".join(d["unfitted"]) if d["unfitted"] else "none")
    )
    print(f"  -> {d['verdict']}")


def report(rec):
    print_generator(rec)
    print_profiles(rec)
    print_decision(rec)


def out_path(args):
    tag = f"_{args.tag}" if args.tag else ""
    return RUNS / f"pm_carrier{tag}.json"


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
