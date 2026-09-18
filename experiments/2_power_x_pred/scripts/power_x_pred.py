r"""P6's amplitude arm: band **relative power** x band **predictability**, 2x2.

Why this run exists. P2b held a band's power fixed (constant envelope) and
varied its predictability; clean-probe retention tracked ``r* = exp(-beta^2/2)``
to within 0.03. T2b then found that on a real corpus, with predictability
nearly flat and band power varying 617x, retention tracked band *power*
(``corr(r_probe, power) = +0.86`` against ``corr(r_probe, sqrt(P)) = -0.08``).
Both can hold only if the model responds to both drivers and whichever varies
dominates. This design puts both under control at once, so every one of the six
cells has a closed-form predicted retention and the two accounts can be told
apart: orthogonal if the model sits at ``r*`` everywhere, an interaction if
``r/r*`` falls in the low-power rows at fixed beta.

Design. Three bands ``b = 2, 4, 8`` (``f = 32, 64, 128``) -- all integers, so
``2b`` is an integer and the matched filter is exact, and any two carriers are
orthogonal over a patch, so no band can leak into another. ``b = 8`` is the
target; ``b = 2, 4`` are control bands held fully coherent. Total power is
``0.5`` in every arm (RMS ``sqrt(0.5)``, the P1/P2b convention): each control
band carries ``rho`` times the target's power, ``rho in {1, 5, 25}``.

  Axis 1, relative power. The axis is the band's **total** power,
  ``mean_p |Z_p|^2 / 2`` -- the mean *square* of the per-patch matched
  amplitude. Under PM the envelope is constant, so that is ``P_t`` at every
  beta: PM only redistributes the band's power between a coherent part
  ``r*^2 P_t`` and an unpredictable part. The coherent quantity
  ``|mean_p Z_p|^2 / 2`` *is* coupled to beta, and using it would collapse the
  2x2 into a diagonal. Both are computed and reported, and the orthogonality is
  verified in the generator check.

  Axis 2, predictability. ``beta in {0, 1.5}`` on the target band only ->
  ``r* = 1.000`` and ``0.3247``. ``make_pm`` is the only generator; it is
  called once per band on that band alone, the parts are scaled by
  ``sqrt(2 P_j)`` and summed, so the phase-walk code is P2b's unmodified and
  the only new thing is the per-band amplitude. Every band draws the same
  generator seed in every arm, so the ``beta`` contrast is paired and the
  ``rho`` contrast holds the realised phases fixed.

Probe. Shape-matched per arm: the arm's own per-band powers, beta = 0, no
noise, seed 123. A flat probe against a 25x-ranged corpus is out of
distribution in shape -- that was T2's confound, and matching total RMS alone
(contract A9) does not fix it. Probe RMS equals corpus RMS exactly
(``sum P_j = 0.5`` in both) and the probe/corpus per-band power ratio is 1.000
at every band by construction; both are verified.

Pre-registered decision, written before the run into this experiment's
``README.md``. Let ``q = r(probe, target) / r*``. All six cells
``|q - 1| <= 0.10`` and the power-axis spread of ``q`` at fixed beta
``<= 0.08`` -> orthogonal, the model at its optimum in each cell. A monotone
fall of ``q`` at fixed beta with a total drop ``>= 0.15`` and ``>= 3x`` its
seed sd -> power modulates the predictability prior. Anything else ->
intermediate, quantify the interaction. No knob is tuned after seeing results.

Run from the repo root:
``.venv/bin/python experiments/2_power_x_pred/scripts/power_x_pred.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fbias.data import b_of, make_pm
from fbias.model import SimTFM
from fbias.probes import band_r2, delta_k, fit_oracle, matched_amp, retention

CTX, PATCH = 512, 32
HIDDEN, LAYERS, HEADS = 32, 2, 4
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
FREQS = (32.0, 64.0, 128.0)
BANDS = tuple(b_of(f, PATCH, CTX) for f in FREQS)  # 2.0, 4.0, 8.0
CONTROLS = tuple(j for j in range(len(BANDS)) if j != 2)
TARGET = 2
NULL_B = 6.0  # no content in corpus or probe: the target reading's noise floor
TOTAL_POWER = 0.5
POWER_RATIOS = (1, 5, 25)
BETAS = (0.0, 1.5)
PROBE_SEED = 123
SEED_STRIDE = 1000  # one generator seed per band, shared by all six arms
GATE = 0.5  # below half the analytic bound an arm has not learned its marginal
Q_TOL, SPREAD_TOL, DROP_TOL = 0.10, 0.08, 0.15
RUNS = Path(__file__).resolve().parents[1] / "runs"


def arm_id(rho, beta):
    r"""``x1_b0``, ``x25_b1.5``: control/target power ratio, target beta."""
    return f"x{rho:g}_b{beta:g}"


ARMS = {
    arm_id(rho, beta): {"rho": float(rho), "beta": beta}
    for rho in POWER_RATIOS
    for beta in BETAS
}


def r_star(beta):
    r"""The conditional mean's amplitude fraction, ``exp(-beta**2 / 2)``."""
    return math.exp(-0.5 * beta * beta)


def arm_powers(arm):
    r"""Per-band tone powers: each control band carries ``rho`` x the target's."""
    rho = ARMS[arm]["rho"]
    ctrl = TOTAL_POWER * rho / (2 * rho + 1)
    return tuple(ctrl / rho if j == TARGET else ctrl for j in range(len(FREQS)))


def arm_betas(arm):
    r"""Per-band betas: only the target band's phase walks."""
    betas = [0.0] * len(FREQS)
    betas[TARGET] = ARMS[arm]["beta"]
    return betas


def walk_seed(seed, j):
    r"""The generator seed band ``j`` uses -- the same in every arm."""
    return seed * SEED_STRIDE + j


def arm_parts(arm, seed):
    r"""Per-band ``(tone, unpredictable)`` parts of the arm's corpus.

    ``make_pm`` on each band alone, scaled by that band's amplitude and summed
    by the caller, so the phase walk is P2b's code path unchanged. Each return
    is a full ``(2 * N_PER, ctx + k)`` window set for that band; the
    unpredictable part is the per-band slice no function of the context can
    know, which is what the fit gate's floor and the realised beta come from.
    """
    powers, betas = arm_powers(arm), arm_betas(arm)
    parts = []
    for j, f in enumerate(FREQS):
        tone, floor = make_pm(
            [f], [betas[j]], 2 * N_PER, CTX, PATCH, walk_seed(seed, j)
        )
        amp = math.sqrt(2 * powers[j])
        parts.append((amp * tone, amp * floor))
    return parts


def corpus(arm, seed):
    r"""The arm's training marginal: the windows, and the per-band floors."""
    parts = arm_parts(arm, seed)
    x = torch.stack([p[0] for p in parts]).sum(0)
    floors = torch.stack([p[1] for p in parts])  # (n_bands, windows, samples)
    return x, floors


def probe_parts(arm):
    r"""The arm's clean probe, per band: its own powers, no phase walk.

    The same generator seed in every arm, so the phases are shared and only the
    amplitudes differ -- the probes across the three power levels are paired.
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
    r"""``mean_p |Z_p|^2 / 2``: the band's **total** power, beta-invariant.

    Every patch holds a whole number of carrier cycles (``b`` is an integer and
    a patch is ``k`` samples), so a patch's ``|Z_p|^2 / 2`` is exactly
    ``A^2 / 2`` whatever the phase walk did to reach it. Averaging ``|Z|^2``
    before the mean over patches is therefore the quantity that does not move
    with beta; averaging ``Z`` first is the coherent one and does.
    """
    patches = x.unfold(1, PATCH, PATCH)
    return 0.5 * float(matched_amp(patches, b, PATCH).abs().square().mean())


@torch.no_grad()
def coherent_power(x, b):
    r"""``mean_w |mean_p Z_p|^2 / 2``: the patch-coherent part PM shrinks."""
    patches = x.unfold(1, PATCH, PATCH)
    return 0.5 * float(matched_amp(patches, b, PATCH).mean(1).abs().square().mean())


@torch.no_grad()
def mainlobe_share(x, b):
    r"""Share of the record's power that lies inside ``+-0.5`` of ``b``.

    A whole-window DFT (``ctx + k`` samples, ``b`` in cycles per patch), so a
    record-level statement. It is reported because PM moves a band's power out
    of the carrier line and into the patch-rate sidebands at ``b +- 1``: the
    per-patch ``band_power`` is blind to that, and this column says how far the
    band's content stays inside a ``+-0.5`` window as the record sees it.
    """
    spec = torch.fft.rfft(x, dim=-1).abs().square()
    bb = torch.fft.rfftfreq(x.shape[-1]) * PATCH  # cycles per patch
    keep = (bb - b).abs() < 0.5
    return float(spec[..., keep].sum(-1).mean() / spec.sum(-1).mean())


@torch.no_grad()
def cross_talk(mix, tones):
    r"""``M[i][j]``: share of band ``i``'s measured power that lives in ``j``.

    The whole signal is built band by band, so the decomposition is exact and
    the diagonal is T1's ``in-band`` column read without a DFT band-pass: near 1
    at every band means the matched filter's reading at that band is that
    band's own content, not leakage from a louder neighbour.
    """
    total = [band_power(mix, b) for b in BANDS]
    return [
        [band_power(tones[j], BANDS[i]) / total[i] for j in range(len(BANDS))]
        for i in range(len(BANDS))
    ]


def realized_betas(floors, powers):
    r"""Per-band beta inverted from the unpredictable part's power (measured).

    ``floor_j = P_j (1 - r*_j^2)``, so ``r*_j`` is recoverable from a mean
    square -- P2b's inversion, and a check that the nominal beta is realised.
    """
    out = []
    for j in range(len(FREQS)):
        r_hat = math.sqrt(max(0.0, 1 - float(floors[j].square().mean()) / powers[j]))
        out.append(0.0 if r_hat >= 1 else math.sqrt(-2 * math.log(r_hat)))
    return out


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


def generator_check():
    r"""No training: the axes must be orthogonal and the probe shape-matched."""
    out = {}
    for arm in ARMS:
        powers, betas = arm_powers(arm), arm_betas(arm)
        acc = {"rms": [], "probe_rms": [], "realized": [], "in_band": []}
        mainlobe_arm = []
        per_band = {
            k: {str(b): [] for b in BANDS}
            for k in ("total", "coherent", "probe", "probe_main")
        }
        for seed in SEEDS:
            parts = arm_parts(arm, seed)
            mix = torch.stack([p[0] for p in parts]).sum(0)
            prb_parts = probe_parts(arm)
            prb = torch.stack(prb_parts).sum(0)
            acc["rms"].append(float(mix.square().mean().sqrt()))
            acc["probe_rms"].append(float(prb.square().mean().sqrt()))
            acc["realized"].append(realized_betas([p[1] for p in parts], powers))
            mix_tot = [band_power(mix, b) for b in BANDS]
            ct = cross_talk(mix, [p[0] for p in parts])
            acc["in_band"].append([ct[i][i] for i in range(len(BANDS))])
            mainlobe_arm.append([mainlobe_share(mix, b) for b in BANDS])
            for i, b in enumerate(BANDS):
                per_band["total"][str(b)].append(mix_tot[i])
                per_band["coherent"][str(b)].append(coherent_power(mix, b))
                per_band["probe"][str(b)].append(band_power(prb, b))
                per_band["probe_main"][str(b)].append(mainlobe_share(prb_parts[i], b))
        out[arm] = {
            "powers": list(powers),
            "betas": betas,
            "rms": mean_std(acc["rms"]),
            "probe_rms": mean_std(acc["probe_rms"]),
            "rms_spread": max(acc["rms"]) - min(acc["rms"]),
            "realized_beta": {
                str(b): mean_std([r[i] for r in acc["realized"]])
                for i, b in enumerate(BANDS)
            },
            "in_band": {
                str(b): mean_std([r[i] for r in acc["in_band"]])
                for i, b in enumerate(BANDS)
            },
            "mainlobe": {
                str(b): mean_std([r[i] for r in mainlobe_arm])
                for i, b in enumerate(BANDS)
            },
            "total_power": {k: mean_std(v) for k, v in per_band["total"].items()},
            "coherent_power": {k: mean_std(v) for k, v in per_band["coherent"].items()},
            "probe_power": {k: mean_std(v) for k, v in per_band["probe"].items()},
            "probe_mainlobe": {
                k: mean_std(v) for k, v in per_band["probe_main"].items()
            },
            "coherent_share": {
                k: per_band["coherent"][k][0] / mean_std(per_band["total"][k])[0]
                for k in per_band["total"]
            },
            "probe_ratio": {
                k: mean_std(per_band["probe"][k])[0] / mean_std(per_band["total"][k])[0]
                for k in per_band["total"]
            },
        }
    return out


def train(seed, arm):
    r"""P2b's training loop, verbatim; only the corpus differs between arms."""
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        num_attn_heads=HEADS,
    )
    data, floors = corpus(arm, seed)
    train_data = data[:N_PER]
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss = torch.tensor(0.0)
    for _ in range(STEPS):
        x = train_data[torch.randint(N_PER, (BATCH,), generator=gen)]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, data[N_PER:], floors.sum(0)[N_PER:], float(loss)


@torch.no_grad()
def evaluate(model, held, held_floor, prb, train_loss):
    r"""Fit gate and its baselines, the marginal, then the clean probe."""
    model.eval()
    pred, _, z_held = model(held)
    patches = model.pat(held)
    target = patches[:, 1:]
    mse = float(F.mse_loss(pred[:, :-1], target))
    var = float(target.var(unbiased=False))
    floor = float(model.pat(held_floor)[:, 1:].square().mean())
    ctx_mean = held[:, :CTX].mean(-1, keepdim=True).unsqueeze(-1)
    row = {
        "train_loss": train_loss,
        "held_var": var,
        "var_explained": 1 - mse / var,
        # what no function of the context can explain: the target's own noise
        "unpredictable_floor": 1 - floor / var,
        # contract 8's baselines: the context mean, and copying the last patch
        "baseline_mean": 1 - float(((ctx_mean - target) ** 2).mean()) / var,
        "baseline_persistence": 1
        - float(((patches[:, :-1] - target) ** 2).mean()) / var,
    }
    for b in BANDS:
        row[f"rm@{b}"] = float(retention(pred[:, -2], target[:, -1], b, PATCH).mean())
        row[f"r2m@{b}"] = band_r2(pred[:, -2], target[:, -1], b, PATCH)
    p_hat, _, z = model(prb)
    true = model.pat(prb)
    f_hat = p_hat[:, -2]
    for b in BANDS:
        row[f"r@{b}"] = float(retention(f_hat, true[:, -1], b, PATCH).mean())
        row[f"dk@{b}"] = float(delta_k(f_hat, true[:, -1], b, PATCH).mean())
        # out-of-sample ridge (contract 5): held latents -> probe, not degenerate
        row[f"r_oracle@{b}"] = fit_oracle(z_held, patches, z, true, b, PATCH)[
            "retention"
        ]
    # amplitude emitted at a band with no content anywhere, in units of the
    # target band's true amplitude -- the target reading's noise floor
    den = matched_amp(true[:, -1], BANDS[TARGET], PATCH).abs()
    row["rnull"] = float((matched_amp(f_hat, NULL_B, PATCH).abs() / den).mean())
    return row


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: mean_std([r[k] for r in rows]) for k in keys}


def q_of(rec, arm, row):
    r"""``q = r(probe, target) / r*`` for one seed of one arm."""
    return row[f"r@{rec['config']['target_b']}"] / rec["arms"][arm]["r_star"]


def q_by_seed(rec, arm):
    r"""``q`` of each seed of one arm, in seed order."""
    return [q_of(rec, arm, row) for row in rec["arms"][arm]["per_seed"]]


def stepwise(rec):
    r"""Where in the power range the two axes stop composing additively.

    ``q = a(rho) + b(beta)`` is the additive account; each row here is one step
    of the power axis, with the drop in ``q`` at each beta and their
    difference-of-differences (the pure two-factor interaction, paired by seed).
    A drop that is the same at both betas is power acting *independently* of
    predictability; a difference between the two columns is the interaction.
    """
    arms = rec["config"]["arms"]
    rhos = sorted({arms[a]["rho"] for a in arms})
    rows = []

    def have(rho, beta):
        return arm_id(rho, beta) in arms

    for i in range(len(rhos) - 1):
        hi, lo = rhos[i], rhos[i + 1]
        if not all(have(r, b) for r in (hi, lo) for b in BETAS):
            continue
        drops = {
            bt: [
                x - y
                for x, y in zip(
                    q_by_seed(rec, arm_id(hi, bt)), q_by_seed(rec, arm_id(lo, bt))
                )
            ]
            for bt in BETAS
        }
        dod = [drops[BETAS[0]][k] - drops[BETAS[-1]][k] for k in range(len(SEEDS))]
        rows.append(
            {
                "from": hi,
                "to": lo,
                "drop": {str(bt): mean_std(drops[bt]) for bt in BETAS},
                "dod": mean_std(dod),
            }
        )
    if all(have(r, b) for r in rhos for b in BETAS):
        allbetas = {
            bt: [
                x - y
                for x, y in zip(
                    q_by_seed(rec, arm_id(rhos[0], bt)),
                    q_by_seed(rec, arm_id(rhos[-1], bt)),
                )
            ]
            for bt in BETAS
        }
        total = {str(bt): mean_std(allbetas[bt]) for bt in BETAS}
        total["dod"] = mean_std(
            [allbetas[BETAS[0]][k] - allbetas[BETAS[-1]][k] for k in range(len(SEEDS))]
        )
    else:
        total = None
    return {"rows": rows, "total": total}


def decide(rec):
    r"""The pre-registered decision, from the six cells' ``q``."""
    arms = rec["config"]["arms"]
    q = {
        arm: mean_std([q_of(rec, arm, row) for row in a["per_seed"]])
        for arm, a in rec["arms"].items()
    }
    spread, drop, mono = {}, {}, {}
    for beta in BETAS:
        ids = sorted(
            (a for a in arms if arms[a]["beta"] == beta), key=lambda a: arms[a]["rho"]
        )
        if len(ids) < len(POWER_RATIOS):
            spread[str(beta)] = None
            drop[str(beta)] = (None, None)
            mono[str(beta)] = None
            continue
        vals = [q[a][0] for a in ids]
        spread[str(beta)] = max(vals) - min(vals)
        paired = [
            q_of(rec, ids[0], lo) - q_of(rec, ids[-1], hi)
            for lo, hi in zip(
                rec["arms"][ids[0]]["per_seed"], rec["arms"][ids[-1]]["per_seed"]
            )
        ]
        drop[str(beta)] = mean_std(paired)
        mono[str(beta)] = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    off = [a for a in arms if abs(q[a][0] - 1) > Q_TOL]
    interacting = [
        b
        for b in (str(x) for x in BETAS)
        if drop[b][0] is not None
        and drop[b][0] >= DROP_TOL
        and drop[b][0] >= 3 * drop[b][1]
        and mono[b]
    ]
    unfitted = [a for a in arms if not rec["arms"][a]["gate_passed"]]
    settled = all(spread[b] is not None for b in spread)
    if unfitted:
        verdict = "inconclusive: arm(s) failed the fit gate"
    elif not settled:
        verdict = "partial run: no verdict"
    elif not off and all(v <= SPREAD_TOL for v in spread.values()):
        verdict = "orthogonal"
    elif interacting:
        verdict = "interaction: power modulates the predictability prior"
    else:
        verdict = "intermediate"
    return {
        "q": q,
        "spread": spread,
        "drop": drop,
        "monotone": mono,
        "cells_off": off,
        "interacting_at": interacting,
        "unfitted": unfitted,
        "stepwise": stepwise(rec),
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
    arms = {a: ARMS[a] for a in ARMS}
    if args.arms:
        missing = set(args.arms) - set(ARMS)
        assert not missing, f"unknown arms {sorted(missing)}; have {sorted(ARMS)}"
        arms = {a: v for a, v in arms.items() if a in args.arms}
    seeds = tuple(args.seeds) if args.seeds else SEEDS

    print(f"== power_x_pred: {len(arms)} arms x {len(seeds)} seeds ==", flush=True)
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {HIDDEN} / {LAYERS}L / {HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {STEPS} steps / cpu",
        flush=True,
    )
    print(
        f"bands f = {FREQS} (b = {BANDS}); target b = {BANDS[TARGET]}, "
        f"null band b = {NULL_B}",
        flush=True,
    )
    print("\ngenerator check (no training yet)", flush=True)
    gen = generator_check()

    rows_by_arm, walls = {}, []
    for arm in arms:
        rows = []
        for seed in seeds:
            t = time.time()
            model, held, held_floor, loss = train(seed, arm)
            row = evaluate(model, held, held_floor, probe(arm), loss)
            row["seed"] = seed
            row["wall"] = time.time() - t
            walls.append(row["wall"])
            rows.append(row)
            print(
                f"  {arm:<8} seed {seed}  loss {loss:.4f}  "
                f"var expl {row['var_explained']:+.3f}  marginal r = "
                + " ".join(f"{row[f'rm@{b}']:.4f}" for b in BANDS)
                + "  probe r = "
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
            "steps": STEPS,
            "batch": BATCH,
            "lr": LR,
            "n_per": N_PER,
            "seeds": list(seeds),
            "device": "cpu",
            "freqs": list(FREQS),
            "b": list(BANDS),
            "target": TARGET,
            "target_b": BANDS[TARGET],
            "target_f": FREQS[TARGET],
            "target_period_samples": PATCH / BANDS[TARGET],
            "null_b": NULL_B,
            "total_power": TOTAL_POWER,
            "power_ratios": list(POWER_RATIOS),
            "betas": list(BETAS),
            "probe_seed": PROBE_SEED,
            "walk_seed_stride": SEED_STRIDE,
            "gate_fraction": GATE,
            "q_tol": Q_TOL,
            "spread_tol": SPREAD_TOL,
            "drop_tol": DROP_TOL,
            "readout": "pred[:, -2] vs pat(x)[:, -1]",
            "arms": arms,
        },
        "r_star": {str(b): r_star(b) for b in BETAS},
        "generator": gen,
        "probe": {
            "seed": PROBE_SEED,
            "shape": "the arm's own per-band powers, beta = 0, no noise",
            "n_per": N_PER,
            "rms": {arm: float(probe(arm).square().mean().sqrt()) for arm in arms},
        },
        "arms": {},
        "wall": 0.0,
    }
    for arm in arms:
        rows = rows_by_arm[arm]
        rho, beta = ARMS[arm]["rho"], ARMS[arm]["beta"]
        powers = arm_powers(arm)
        bound = 1 - powers[TARGET] * (1 - r_star(beta) ** 2) / TOTAL_POWER
        s = summarize(rows)
        rec["arms"][arm] = {
            "rho": rho,
            "beta": beta,
            "powers": list(powers),
            "betas": arm_betas(arm),
            "r_star": r_star(beta),
            "analytic_bound": bound,
            "gate_passed": s["var_explained"][0] > GATE * bound,
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
        f"{c['layers']}L / {c['heads']} heads | SGD lr {c['lr']} / batch "
        f"{c['batch']} / {c['steps']} steps | seeds {c['seeds']}"
    )
    print(
        f"  target band b={c['target_b']} (f={c['target_f']}, period "
        f"{c['target_period_samples']:g} samples); controls "
        + ", ".join(f"b={BANDS[j]} (f={FREQS[j]})" for j in CONTROLS)
        + "; beta in "
        + ", ".join(str(b) for b in c["betas"])
        + " -> r* = "
        + ", ".join(f"{rec['r_star'][str(b)]:.3f}" for b in c["betas"])
    )
    print(
        f"  probe: seed {c['probe_seed']}, the arm's own per-band powers, "
        f"beta = 0, no noise; corpus band-power range = {c['power_ratios']}x"
    )


def print_generator(rec):
    gen, arms = rec["generator"], rec["config"]["arms"]
    tb = rec["config"]["target_b"]
    print("\ngenerator check, no training: corpus RMS and the probe shape match")
    table(
        [
            f"  {'arm':<7}",
            f"{'corpus rms':>11}",
            f"{'probe rms':>10}",
            *[f"{'P@' + str(b):>9}" for b in BANDS],
            *[f"{'probe/P@' + str(b):>12}" for b in BANDS],
        ],
        [
            [
                f"  {arm:<7}",
                f"{gen[arm]['rms'][0]:>11.5f}",
                f"{gen[arm]['probe_rms'][0]:>10.5f}",
                *[f"{gen[arm]['total_power'][str(b)][0]:>9.5f}" for b in BANDS],
                *[f"{gen[arm]['probe_ratio'][str(b)]:>12.4f}" for b in BANDS],
            ]
            for arm in arms
        ],
    )
    print(
        "  P@b is the band's **total** power, ``mean_p |Z_p|^2 / 2``; probe/P is "
        "the shape match,\n  and the probe uses the arm's own per-band powers, so "
        "it is 1.000 at every band."
    )

    print("\ncoherence, leakage, and where PM pushes the band's power")
    table(
        [
            f"  {'arm':<7}",
            *[f"{'coh/tot@' + str(b):>12}" for b in BANDS],
            *[f"{'in-band@' + str(b):>13}" for b in BANDS],
            f"{'mainshare@' + str(tb):>15}",
        ],
        [
            [
                f"  {arm:<7}",
                *[f"{gen[arm]['coherent_share'][str(b)]:>12.4f}" for b in BANDS],
                *[f"{gen[arm]['in_band'][str(b)][0]:>13.4f}" for b in BANDS],
                f"{gen[arm]['mainlobe'][str(tb)][0]:>15.4f}",
            ]
            for arm in arms
        ],
    )
    print(
        "  coh/tot is ``|mean_p Z|^2 / mean_p |Z|^2`` -- 1.000 at a fully "
        "coherent band and r*^2 =\n  0.105 at the target under beta=1.5: that is "
        "the redistribution the power axis must\n  not follow. in-band is the "
        "cross-talk diagonal (share of band b's measured power that\n  lives in "
        "band b's own part); integer-spaced carriers are orthogonal over a "
        "patch, so it\n  is 1.000 and there is no leakage to correct for. "
        f"mainshare@{tb} is the whole-window\n  DFT power inside +-0.5 of the "
        "carrier, where PM's sidebands at b +- 1 show up."
    )

    print("\nrealised beta, inverted from the unpredictable part (measured)")
    table(
        [f"  {'arm':<7}", *[f"{'beta@' + str(b):>11}" for b in BANDS]],
        [
            [
                f"  {arm:<7}",
                *[f"{gen[arm]['realized_beta'][str(b)][0]:>11.4f}" for b in BANDS],
            ]
            for arm in arms
        ],
    )
    print(
        "  nominal beta on the target band: "
        + ", ".join(f"{arm}={arms[arm]['beta']:.1f}" for arm in arms)
        + "; the controls read 0.0000"
    )

    print("\nexactly one axis moves at a time: the target band's two powers")
    table(
        [
            f"  {'rho':>4}",
            f"{'quantity':>22}",
            f"{'beta=0':>12}",
            f"{'beta=1.5':>12}",
            f"{'ratio':>8}",
        ],
        [
            [
                f"  {rho:>4}",
                f"{name:>22}",
                f"{gen[arm_id(rho, BETAS[0])][key][str(tb)][0]:>12.5f}",
                f"{gen[arm_id(rho, BETAS[-1])][key][str(tb)][0]:>12.5f}",
                f"{ratio:>8.4f}",
            ]
            for rho in POWER_RATIOS
            for name, key, ratio in (
                (
                    f"total power P@{tb}",
                    "total_power",
                    gen[arm_id(rho, BETAS[-1])]["total_power"][str(tb)][0]
                    / gen[arm_id(rho, BETAS[0])]["total_power"][str(tb)][0],
                ),
                (
                    "coherent power",
                    "coherent_power",
                    gen[arm_id(rho, BETAS[-1])]["coherent_power"][str(tb)][0]
                    / gen[arm_id(rho, BETAS[0])]["coherent_power"][str(tb)][0],
                ),
            )
        ],
    )
    print(
        "  the first row of each pair is the axis actually used and must be "
        "flat in beta (ratio\n  1.000 to float noise); the second is the "
        "quantity that is coupled to beta, and its\n  ratio must be "
        f"r*^2 = {r_star(BETAS[-1]) ** 2:.4f}."
    )

    print("\ncorpus band-power range, against T2b's dose curve")
    table(
        [f"  {'rho':>4}", f"{'range (max/min)':>16}", f"{'target share':>13}"],
        [
            [
                f"  {rho:>4}",
                f"{rng:>16.3f}",
                f"{share:>13.4f}",
            ]
            for rho, rng, share in (
                (
                    rho,
                    max(
                        gen[arm_id(rho, BETAS[0])]["total_power"][str(b)][0]
                        for b in BANDS
                    )
                    / min(
                        gen[arm_id(rho, BETAS[0])]["total_power"][str(b)][0]
                        for b in BANDS
                    ),
                    gen[arm_id(rho, BETAS[0])]["total_power"][str(tb)][0]
                    / sum(
                        gen[arm_id(rho, BETAS[0])]["total_power"][str(b)][0]
                        for b in BANDS
                    ),
                )
                for rho in POWER_RATIOS
            )
        ],
    )
    print(
        "  T2b's dose curve for comparison: a 617x corpus range gave a 7.89x "
        "decline in r,\n  81x gave 5.37x and 3.1x gave 1.12x -- the collapse "
        "sits between 81x and 3.1x, so\n  these three levels straddle it."
    )


def print_headline(rec):
    tb = rec["config"]["target_b"]
    print(
        f"\nthe 2x2: clean probe, target band b={tb}, r* = exp(-beta^2/2); "
        "q = r / r* is the headline\n  column, mean +- sd over the seeds. rnull "
        "is the amplitude the model emits at b=6.0,\n  where neither corpus nor "
        "probe has any content, in units of the target's true amplitude."
    )
    table(
        [
            f"  {'arm':<7}",
            f"{'rho':>4}",
            f"{'beta':>5}",
            f"{'r*':>6}",
            f"{f'r@{tb} target':>20}",
            f"{'q = r/r*':>15}",
            *[f"{f'r@{BANDS[j]} ctrl':>17}" for j in CONTROLS],
            f"{'rnull':>15}",
            f"{'fit gate':>18}",
        ],
        [
            [
                f"  {arm:<7}",
                f"{a['rho']:>4.0f}",
                f"{a['beta']:>5.1f}",
                f"{a['r_star']:>6.3f}",
                ms(s[f"r@{tb}"], 20),
                f"{rec['decision']['q'][arm][0]:.3f}±"
                f"{rec['decision']['q'][arm][1]:.3f}".rjust(15),
                *[ms(s[f"r@{b}"], 17) for b in (BANDS[j] for j in CONTROLS)],
                ms(s["rnull"], 15),
                (
                    f"{s['var_explained'][0]:+.3f} "
                    f"({100 * s['var_explained'][0] / a['analytic_bound']:.0f}%)"
                    + ("" if a["gate_passed"] else " FAIL")
                ).rjust(18),
            ]
            for arm, a in rec["arms"].items()
            for s in (a["mean_std"],)
        ],
    )
    print(
        "  neither control band is perturbed in any arm: those two columns are "
        "the internal control.\n  The gate's analytic bound is "
        "1 - P_t (1 - r*^2) / 0.5, the conditional mean's MSE floor."
    )


def ceilings(a):
    r"""Each band's retention ceiling: ``r*`` at the target band, 1 at the controls."""
    return [a["r_star"] if j == TARGET else 1.0 for j in range(len(BANDS))]


def print_marginal(rec):
    print("\nper arm: the marginal (its own held-out corpus), and the oracle")
    table(
        [
            f"  {'arm':<7}",
            f"{'var expl':>20}",
            f"{'floor':>8}",
            f"{'mean base':>10}",
            f"{'persist':>9}",
            *[f"{'rm@' + str(b):>17}" for b in BANDS],
        ],
        [
            [
                f"  {arm:<7}",
                f"{s['var_explained'][0]:>+9.4f}±{s['var_explained'][1]:<9.4f}",
                f"{s['unpredictable_floor'][0]:>+8.4f}",
                f"{s['baseline_mean'][0]:>+10.4f}",
                f"{s['baseline_persistence'][0]:>+9.4f}",
                *[ms(s[f"rm@{b}"], 17) for b in BANDS],
            ]
            for arm, a in rec["arms"].items()
            for s in (a["mean_std"],)
        ],
    )
    print(
        "  the marginal's ceiling at the target band is r* (0.325 at beta=1.5): "
        "the band cannot be\n  predicted beyond its coherent part, so rm@8 near "
        "0.32 is full marks there, not a\n  deficit. The floor is "
        "1 - (unpredictable power) / var."
    )
    print("\n  the same marginal readings as fractions of each band's ceiling")
    table(
        [f"  {'arm':<7}", *[f"{'rm/ceiling@' + str(b):>15}" for b in BANDS]],
        [
            [
                f"  {arm:<7}",
                *[f"{s[f'rm@{b}'][0] / ceil[j]:>15.3f}" for j, b in enumerate(BANDS)],
            ]
            for arm, a in rec["arms"].items()
            for s in (a["mean_std"],)
            for ceil in (ceilings(a),)
        ],
    )
    print("\n  probe oracle (contract 5): out-of-sample ridge z_held -> probe")
    table(
        [f"  {'arm':<7}", *[f"{'r_oracle@' + str(b):>13}" for b in BANDS]],
        [
            [
                f"  {arm:<7}",
                *[f"{s[f'r_oracle@{b}'][0]:>13.3f}" for b in BANDS],
            ]
            for arm, a in rec["arms"].items()
            for s in (a["mean_std"],)
        ],
    )
    print(
        "  the oracle is a linear readout of the same frozen representation, so "
        "a target-band\n  oracle near r* = "
        f"{rec['r_star']['1.5']:.3f} in the beta=1.5 arms says the information is "
        "there and any\n  head shortfall is an optimisation gap (P3), not a "
        "prior."
    )
    print("\n  oracle / r* per band (1.000 = the representation is at its optimum)")
    table(
        [f"  {'arm':<7}", *[f"{'oracle/ceiling@' + str(b):>19}" for b in BANDS]],
        [
            [
                f"  {arm:<7}",
                *[
                    f"{s[f'r_oracle@{b}'][0] / ceil[j]:>19.3f}"
                    for j, b in enumerate(BANDS)
                ],
            ]
            for arm, a in rec["arms"].items()
            for s in (a["mean_std"],)
            for ceil in (ceilings(a),)
        ],
    )
    print(
        "\n  the clean probe in Fredformer's units: dk = |Z_pred - Z_true| / |Z_true|"
    )
    table(
        [
            f"  {'arm':<7}",
            f"{'band':>5}",
            f"{'r':>8}",
            f"{'dk':>8}",
            f"{'dk-(1-r)':>10}",
        ],
        [
            [
                f"  {arm:<7}",
                f"{b:>5.1f}",
                f"{s[f'r@{b}'][0]:>8.4f}",
                f"{s[f'dk@{b}'][0]:>8.4f}",
                f"{s[f'dk@{b}'][0] - (1 - s[f'r@{b}'][0]):>10.4f}",
            ]
            for arm, a in rec["arms"].items()
            for s in (a["mean_std"],)
            for b in BANDS
        ],
    )
    print(
        "  dk is |r e^{i dphi} - 1|, so 1 - r is its floor and the excess is the "
        "phase's share\n  (bounded by 2r). An excess of 0.0000 says the emitted "
        "phase is locked and dk carries\n  nothing r does not; r stays the "
        "primary reading, and dk is the column to set beside\n  Fredformer's "
        "0.01 / 0.95."
    )


def step_label(lo, hi):
    r"""``1x -> 5x``: one step of the power axis."""
    return f"{lo:.0f}x -> {hi:.0f}x"


def print_interaction(rec):
    r"""The decomposition of the pre-registered decision into its parts."""
    d = rec["decision"]
    st = d.get("stepwise") or stepwise(rec)
    tb = rec["config"]["target_b"]
    rows = [
        [
            f"  {step_label(r['from'], r['to']):>12}",
            ms(r["drop"][str(BETAS[0])], 22, 3),
            ms(r["drop"][str(BETAS[-1])], 24, 3),
            ms(r["dod"], 24, 3),
        ]
        for r in st["rows"]
    ]
    if st["total"]:
        rows.append(
            [
                f"  {step_label(st['rows'][0]['from'], st['rows'][-1]['to']):>12}",
                ms(st["total"][str(BETAS[0])], 22, 3),
                ms(st["total"][str(BETAS[-1])], 24, 3),
                ms(st["total"]["dod"], 24, 3),
            ]
        )
    print("\nwhere the two axes stop composing additively (q, paired by seed)")
    table(
        [
            f"  {'power step':>12}",
            f"{'drop in q at beta=0':>22}",
            f"{'drop in q at beta=1.5':>24}",
            f"{'difference of the two':>24}",
        ],
        rows,
    )
    print(
        "  the drop at each beta on its own is the power effect; the last column "
        "is the pure\n  two-factor interaction -- zero means power acts on the "
        "band regardless of how\n  predictable it is, and the two axes are "
        "additive in q."
    )

    print(f"\nwhat the model actually emits at b={tb}, in absolute amplitude")
    table(
        [
            f"  {'arm':<7}",
            f"{'true A':>8}",
            f"{'r':>8}",
            f"{'emitted':>9}",
            f"{'rnull':>8}",
            f"{'floor amp':>10}",
            f"{'emit/floor':>11}",
        ],
        [
            [
                f"  {arm:<7}",
                f"{a['powers'][TARGET] ** 0.5 * 2**0.5:>8.4f}",
                f"{s[f'r@{tb}'][0]:>8.4f}",
                f"{s[f'r@{tb}'][0] * (2 * a['powers'][TARGET]) ** 0.5:>9.5f}",
                f"{s['rnull'][0]:>8.4f}",
                f"{s['rnull'][0] * (2 * a['powers'][TARGET]) ** 0.5:>10.5f}",
                f"{s[f'r@{tb}'][0] / s['rnull'][0]:>11.2f}",
            ]
            for arm, a in rec["arms"].items()
            for s in (a["mean_std"],)
        ],
    )
    print(
        "  emitted = r x the band's true amplitude; floor amp = the same "
        "conversion applied to\n  rnull, the band where neither corpus nor "
        "probe carries content. The floor is a\n  roughly constant absolute "
        "error term (0.012-0.019 in every arm), so it eats a fixed\n  fraction "
        "of the weaker bands' readings: the 25x, beta=1.5 cell emits 1.3x the "
        "floor,\n  i.e. essentially nothing, whatever r/r* says there."
    )


def print_per_seed(rec):
    print("\nper seed: fit gate, then the marginal and the clean probe")
    table(
        [
            f"  {'arm':<7}",
            f"{'seed':>4}",
            f"{'var expl':>9}",
            f"{'floor':>8}",
            *[f"{'rm@' + str(b):>8}" for b in BANDS],
            *[f"{'r@' + str(b):>8}" for b in BANDS],
            f"{'q':>8}",
            f"{'rnull':>8}",
        ],
        [
            [
                f"  {arm:<7}",
                f"{row['seed']:>4}",
                f"{row['var_explained']:>+9.4f}",
                f"{row['unpredictable_floor']:>+8.4f}",
                *[f"{row[f'rm@{b}']:>8.4f}" for b in BANDS],
                *[f"{row[f'r@{b}']:>8.4f}" for b in BANDS],
                f"{q_of(rec, arm, row):>8.4f}",
                f"{row['rnull']:>8.4f}",
            ]
            for arm, a in rec["arms"].items()
            for row in a["per_seed"]
        ],
    )


def print_decision(rec):
    d, arms = rec["decision"], rec["config"]["arms"]
    rho_seen = sorted({arms[a]["rho"] for a in arms})
    print("\nq = r / r*, the pre-registered grid (rows: control/target power ratio)")
    table(
        [f"  {'rho':>4}", *[f"{'beta=' + str(b):>20}" for b in BETAS]],
        [
            [
                f"  {rho:>4.0f}",
                *[
                    (
                        ms(d["q"][arm_id(rho, beta)], 20, 3)
                        if beta in [arms[a]["beta"] for a in arms]
                        and arm_id(rho, beta) in arms
                        else f"{'-':>20}"
                    )
                    for beta in BETAS
                ],
            ]
            for rho in rho_seen
        ],
    )
    print(
        "  a model at its optimum in every cell reads 1.000 in all six; two "
        "columns of identical\n  numbers are what orthogonality looks like."
    )
    print("\npre-registered decision")
    for beta in BETAS:
        b = str(beta)
        if d["drop"][b][0] is None:
            print(f"  beta={beta:<4} incomplete: not every power level was run")
            continue
        drop, sd = d["drop"][b]
        print(
            f"  beta={beta:<4} spread across power levels {d['spread'][b]:.3f} "
            f"(<= {SPREAD_TOL})   q(1x) - q(25x) {drop:+.3f}±{sd:.3f}"
        )
        print(
            "            interaction needs "
            f">= {DROP_TOL} and >= 3x its sd; monotone across the three "
            f"levels: {d['monotone'][b]}"
        )
    print(
        f"  cells off |q - 1| > {Q_TOL}: "
        + (", ".join(d["cells_off"]) if d["cells_off"] else "none")
    )
    print(
        "  arms failing the fit gate: "
        + (", ".join(d["unfitted"]) if d["unfitted"] else "none")
    )
    print(f"  -> {d['verdict']}")


def report(rec):
    print_config(rec)
    print_generator(rec)
    print_headline(rec)
    print_marginal(rec)
    print_interaction(rec)
    print_per_seed(rec)
    print_decision(rec)


def out_path(args):
    tag = f"_{args.tag}" if args.tag else ""
    return RUNS / f"power_x_pred{tag}.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=None, help="subset of arm ids")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="subset of seeds")
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
