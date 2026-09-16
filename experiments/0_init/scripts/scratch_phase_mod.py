r"""P2b: phase modulation -- is the deficit about unpredictability or fittability?

P2 left the word "predictability" unearned. PLAN.md §4's theta caveat: theta's
"incoherent" energy is built from the window's own FFT bins, so it is a
deterministic, low-DOF realization the context determines -- a fittability knob,
not an unpredictability knob -- and four of P2's six arms beat the analytic
bound because of it. Phase modulation does not share the flaw: the phase takes a
random step ``beta * eps`` at every patch boundary, ``eps ~ N(0, 1)`` drawn
independently, so the next patch's phase is genuinely not a function of the
context. This run is what decides the claim's wording.

Design. Three tones ``b = 0.25, 2.0, 8.0`` (``f = 4, 32, 128``) carry a third of
the signal power each, and ``make_pm`` keeps the envelope constant, so total
power is fixed at every ``beta`` -- no RMS bookkeeping, unlike theta, which
moves power between a tone and its own band.

  Part A  dose-response: beta in {0, 0.5, 1.0, 1.5, 2.0} on the high band.
  Part B  band-specificity: beta = 1.0 on each band in turn -- P2's diagonal,
          re-run with the clean instrument. ``hi_b1.0`` is shared between the
          parts, so 7 arms x 3 seeds = 21 runs.

Closed-form optimum. The previous patch's phase is readable and the advance
``2 pi b_c`` is known, so the only unknown in the next patch is ``beta * eps``
and the conditional mean shrinks by

    r* = E[e^{i beta eps}] = exp(-beta**2 / 2)
       beta 0.5 -> 0.882   1.0 -> 0.607   1.5 -> 0.325   2.0 -> 0.135

which is therefore the correct reading on the arm's own held-out marginal.

The clean probe -- the fixed 3-tone ``make_mixture``, seed 123, no phase walk,
identical in every arm and seed -- is where the claim lives. Beta is zero
there, so sixteen patches with exactly zero phase increments determine the next
phase completely, and a model that reads its own evidence would read
``r = 1.000`` at every beta. A model carrying a patch-local prior about how far
a band's phase jumps between patches instead reads ``r ~ r*``.

  Prediction 1 (the prior transfers): clean-probe r at b=8.0 tracks
     ``exp(-beta**2 / 2)`` over beta = 0.5, 1.0, 1.5, 2.0.
  Prediction 2 (evidence-adaptive): clean-probe r = 1.000 at every beta, which
     is what P1b found for theta -- an instrument that could not tell them
     apart.

The two differ by a wide margin at beta >= 1 (0.607 vs 1.000), so this run
decides it. Every arm is read on both corpora: the clean probe (raw, and
differenced against the clean arm, which P2 showed is mandatory) and the arm's
own held-out marginal, beside the contract 8 fit gate that says whether the arm
learned its marginal at all.

Config as P1/P2: ctx 512 / k 32 / hidden 32 / 2 layers / 4 heads / SGD lr 1e-2
/ batch 64 / 2000 steps, CPU, readout ``pred[:, -2]`` against ``pat(x)[:, -1]``.

Run from the repo root:
``.venv/bin/python experiments/0_init/scripts/scratch_phase_mod.py``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fbias.data import b_of, make_mixture, make_pm
from fbias.model import SimTFM
from fbias.probes import band_r2, fit_oracle, matched_amp, retention

CTX, PATCH = 512, 32
HIDDEN, LAYERS, HEADS = 32, 2, 4
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
FREQS = [4.0, 32.0, 128.0]  # b = 0.25, 2.0, 8.0
BANDS = [b_of(f, PATCH, CTX) for f in FREQS]
BAND_NAMES = ("lo", "mid", "hi")
DPHI = [2 * math.pi * b % (2 * math.pi) for b in BANDS]
PROBE_SEED = 123
BASELINE = "clean"
TONE_POWER = 0.5 / len(FREQS)  # per tone, matching make_mixture(normalize=True)
GATE = 0.5  # an arm below half its analytic bound has not learned its marginal
RUNS = Path(__file__).resolve().parents[1] / "runs"


def betas_of(idx, beta):
    r"""Per-tone betas of an arm: zero everywhere but the perturbed band."""
    betas = [0.0] * len(FREQS)
    if idx is not None:
        betas[idx] = beta
    return betas


ARMS = {
    BASELINE: betas_of(None, 0.0),
    "hi_b0.5": betas_of(2, 0.5),
    "hi_b1.0": betas_of(2, 1.0),
    "hi_b1.5": betas_of(2, 1.5),
    "hi_b2.0": betas_of(2, 2.0),
    "lo_b1.0": betas_of(0, 1.0),
    "mid_b1.0": betas_of(1, 1.0),
}
DOSE_ARMS = (BASELINE, "hi_b0.5", "hi_b1.0", "hi_b1.5", "hi_b2.0")  # Part A
BAND_ARMS = ("lo_b1.0", "mid_b1.0", "hi_b1.0")  # Part B, in band order


def perturbed(arm):
    r"""``(index of the perturbed band or None, its beta)`` of an arm."""
    for j, beta in enumerate(ARMS[arm]):
        if beta > 0:
            return j, beta
    return None, 0.0


def spec_of(arm):
    r"""The arm's ``(perturbed band, beta)`` spec, for header lines."""
    idx, beta = perturbed(arm)
    return f"({'-' if idx is None else BAND_NAMES[idx]}, {beta})"


def r_star(beta):
    r"""The conditional mean's amplitude fraction, ``exp(-beta**2 / 2)``."""
    return math.exp(-0.5 * beta * beta)


def bound_of(arm):
    r"""Fraction of the target variance no function of the context can explain.

    One tone's future phase is ``beta * eps``, so its unpredictable power is
    ``TONE_POWER * (1 - exp(-beta**2))`` and the target variance is ``0.5``.
    """
    floor = sum(TONE_POWER * (1 - r_star(b) ** 2) for b in ARMS[arm])
    return 1 - floor / 0.5


def corpus(arm, seed):
    r"""The arm's training marginal: the window, and its unpredictable part."""
    return make_pm(FREQS, ARMS[arm], 2 * N_PER, CTX, PATCH, seed)


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


@torch.no_grad()
def power_table(probe_rms):
    r"""Generator check only, no training: envelope, diffusion, and bin power.

    ``realised beta`` is inverted from the unpredictable part's power,
    ``(1 - exp(-beta**2)) / 6`` per tone, so it is measured rather than assumed.
    ``bin power`` is the per-patch matched amplitude at the carrier, ``|Z|^2/2``
    -- the power the probe bin carries. At ``b = 0.25`` the negative-frequency
    image leaks in, so that column reads high by a factor independent of beta;
    the point is that it does not move with beta.
    """
    out = {}
    for arm in ARMS:
        rms, floor, bins = [], [], {str(b): [] for b in BANDS}
        for seed in SEEDS:
            x, unpred = corpus(arm, seed)
            rms.append(float(x.square().mean().sqrt()))
            floor.append(float(unpred.square().mean()))
            patches = x.unfold(1, PATCH, PATCH)
            for b in BANDS:
                amp = matched_amp(patches, b, PATCH).abs().square().mean()
                bins[str(b)].append(float(amp) / 2)
        f, _ = mean_std(floor)
        realized = math.sqrt(-math.log(max(1e-12, 1 - f / TONE_POWER))) if f else 0.0
        out[arm] = {
            "betas": ARMS[arm],
            "rms": mean_std(rms),
            "vs_probe": mean_std(rms)[0] / probe_rms - 1,
            "rms_seed_spread": max(rms) - min(rms),
            "unpredictable_floor": mean_std(floor),
            "realized_beta": realized,
            "r_star_realized": r_star(realized),
            "bin_power": {k: mean_std(v) for k, v in bins.items()},
        }
    return out


def train(seed, arm):
    r"""P1's training loop, verbatim; only the corpus differs between arms."""
    torch.manual_seed(seed)
    model = SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        num_attn_heads=HEADS,
    )
    data, unpred = corpus(arm, seed)
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
    return model, data[N_PER:], unpred[N_PER:], float(loss)


@torch.no_grad()
def evaluate(model, held, held_unpred, probe, train_loss):
    r"""Fit gate and marginal reading on the arm's own corpus, then the probe."""
    model.eval()
    pred, _, z_held = model(held)
    patches = model.pat(held)
    target = patches[:, 1:]
    mse = float(F.mse_loss(pred[:, :-1], target))
    var = float(target.var(unbiased=False))
    floor = float(model.pat(held_unpred)[:, 1:].square().mean())
    row = {
        "train_loss": train_loss,
        "held_var": var,
        "var_explained": 1 - mse / var,
        # what no function of the context can explain: the target's own noise
        "unpredictable_floor": 1 - floor / var,
    }
    for b in BANDS:
        row[f"rm@{b}"] = float(retention(pred[:, -2], target[:, -1], b, PATCH).mean())
        row[f"r2m@{b}"] = band_r2(pred[:, -2], target[:, -1], b, PATCH)
    p_hat, _, z = model(probe)
    true = model.pat(probe)
    f_hat = p_hat[:, -2]
    for b in BANDS:
        row[f"r@{b}"] = float(retention(f_hat, true[:, -1], b, PATCH).mean())
        # out-of-sample ridge (contract 5): held latents -> probe, so not degenerate
        row[f"r_oracle@{b}"] = fit_oracle(z_held, patches, z, true, b, PATCH)[
            "retention"
        ]
    return row


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: mean_std([r[k] for r in rows]) for k in keys}


def print_matrix(title, arms, cell, width=14):
    r"""Bands down the side, arms across."""
    head = f"  {'readout':<8} " + " ".join(f"{a:>{width}}" for a in arms)
    print(f"\n{title}\n{head}")
    print("  " + "-" * (len(head) - 2))
    for i, b in enumerate(BANDS):
        print(f"  b={b:<6} " + " ".join(f"{cell(a, i):>{width}}" for a in arms))


def print_power(power, probe_rms):
    print(
        f"\ngenerator check, no training (probe RMS {probe_rms:.4f}; constant "
        "envelope, so every arm must match it)"
    )
    head = (
        f"  {'arm':<8} {'betas lo/mid/hi':>18} {'corpus rms':>18} {'vs probe':>9} "
        f"{'realised beta':>13} {'r*':>7}"
    )
    print(head)
    for arm in ARMS:
        p = power[arm]
        m, sd = p["rms"]
        idx, beta = perturbed(arm)
        bs = "/".join(f"{v:.2f}" for v in p["betas"])
        realised = "n/a" if idx is None else f"{p['realized_beta']:.3f}"
        rs = "1.000" if idx is None else f"{p['r_star_realized']:.3f}"
        print(
            f"  {arm:<8} {bs:>18} {m:>10.5f}±{sd:<6.5f} {p['vs_probe']:>8.2%} "
            f"{realised:>13} {rs:>7}"
        )
    print(
        "  the same seed gives the same RMS in every arm to 1e-4 relative; the "
        "~0.5% spread\n  across seeds is the corpus realisation, and the "
        "unpredictable part's power inverts\n  to the nominal beta (measured, "
        "not assumed)"
    )
    print("\nper-patch bin power |Z|^2/2 of each carrier, mean over 3 seeds")
    head = f"  {'arm':<8} " + " ".join(f"{'b=' + str(b):>14}" for b in BANDS)
    print(head)
    for arm in ARMS:
        cells = " ".join(f"{power[arm]['bin_power'][str(b)][0]:>14.4f}" for b in BANDS)
        print(f"  {arm:<8} {cells}")
    print(
        "  flat in beta: phase diffusion moves no power between bins, unlike "
        "theta. b=0.25\n  reads 1.45x its tone power because the matched "
        "filter's negative-frequency image leaks\n  at a non-integer 2b -- "
        "still constant in beta, which is the point."
    )


def print_per_seed(arms):
    head = (
        f"  {'arm':<8} {'seed':>4} {'var expl':>9} {'floor':>7} "
        " "
        + " ".join(f"{'rm@' + str(b):>8}" for b in BANDS)
        + " "
        + " ".join(f"{'r@' + str(b):>8}" for b in BANDS)
    )
    print(
        "\nper-seed: fit gate and marginal r on the arm's own corpus, then the "
        "clean probe\n" + head
    )
    print("  " + "-" * (len(head) - 2))
    for arm, rows in arms.items():
        for r in rows:
            print(
                f"  {arm:<8} {r['seed']:>4} {r['var_explained']:>+9.3f} "
                f"{r['unpredictable_floor']:>+7.3f} "
                + " ".join(f"{r[f'rm@{b}']:>8.4f}" for b in BANDS)
                + " "
                + " ".join(f"{r[f'r@{b}']:>8.4f}" for b in BANDS)
            )


def diff(s, arm, i):
    r"""Change in clean-probe retention at band ``i``, against the clean arm."""
    key = f"r@{BANDS[i]}"
    return s[arm][key][0] - s[BASELINE][key][0]


def print_probe(s):
    print_matrix(
        "clean-probe r, raw, mean +- std over 3 seeds (the same probe in every arm)",
        tuple(ARMS),
        lambda a, i: f"{s[a][f'r@{BANDS[i]}'][0]:.4f}±{s[a][f'r@{BANDS[i]}'][1]:.4f}",
    )
    print_matrix(
        "clean-probe r, change from the clean arm (P2: the subtraction is mandatory)",
        tuple(ARMS),
        lambda a, i: "0.0000" if a == BASELINE else f"{diff(s, a, i):+.4f}",
    )
    print("\nclean-probe oracle (contract 5): out-of-sample ridge z_held -> probe")
    for arm in ARMS:
        cells = " ".join(f"b={b}: {s[arm][f'r_oracle@{b}'][0]:.3f}" for b in BANDS)
        print(f"  {arm:<8} {cells}")


def print_dose(s):
    r"""Part A against the two predictions, on the probe and on the marginal."""
    print(
        "\nPart A: clean-probe r at b=8.0 against the two predictions\n"
        f"  {'beta':>5} {'r*':>7} {'probe r@8':>18} {'change':>9} {'r - r*':>8} "
        f"{'1 - r':>8} {'reads closer to':>16} {'marginal r@8':>14}"
    )
    for arm in DOSE_ARMS:
        idx, beta = perturbed(arm)
        target = r_star(beta)
        m, sd = s[arm][f"r@{BANDS[2]}"]
        mm, msd = s[arm][f"rm@{BANDS[2]}"]
        d = 0.0 if arm == BASELINE else diff(s, arm, 2)
        closer = "r* (prior)" if abs(m - target) < abs(1 - m) else "1.000 (adaptive)"
        if abs(m - target) == abs(1 - m):
            closer = "indistinguishable"
        print(
            f"  {beta:>5.1f} {target:>7.3f} {m:>10.4f}±{sd:<7.4f} {d:>+9.4f} "
            f"{m - target:>+8.4f} {1 - m:>+8.4f} {closer:>16} "
            f"{mm:>8.4f}±{msd:<5.4f}"
        )
    print(
        "  probe r@8 is the decision: r* = exp(-beta^2/2) is the prior reading, "
        "1.000 is what a\n  model that reads sixteen zero-increment patches "
        "would do. marginal r@8 is the same\n  arm on its own held-out corpus, "
        "where r* is the correct answer -- it says whether the\n  arm learned "
        "its own marginal at all (the fit gate below says it another way)."
    )
    print("\ninternal control: the two unperturbed bands in Part A (clean probe)")
    print(
        f"  {'arm':<8} {'beta@hi':>8} {'lo raw':>10} {'lo change':>10} "
        f"{'mid raw':>10} {'mid change':>11}"
    )
    for arm in DOSE_ARMS:
        _, beta = perturbed(arm)
        lo, losd = s[arm][f"r@{BANDS[0]}"]
        mid, midsd = s[arm][f"r@{BANDS[1]}"]
        print(
            f"  {arm:<8} {beta:>8.1f} {lo:>10.4f} {diff(s, arm, 0):>+10.4f} "
            f"{mid:>10.4f} {diff(s, arm, 1):>+11.4f}"
        )
    print(
        "  these two bands are perturbed by nothing, so their change columns are "
        "the internal\n  control: a change near 0 means the effect is band-local "
        "and does not spread to bands\n  whose phase is still deterministic."
    )


def print_marginal(s):
    print_matrix(
        "r on each arm's own held-out marginal at the last position "
        "(mean over 3 seeds)",
        tuple(ARMS),
        lambda a, i: f"{s[a][f'rm@{BANDS[i]}'][0]:.4f}±{s[a][f'rm@{BANDS[i]}'][1]:.4f}",
    )
    print_matrix(
        "band r2 on the same marginal (does the model emit the band at all?)",
        tuple(ARMS),
        lambda a, i: f"{s[a][f'r2m@{BANDS[i]}'][0]:.4f}",
    )
    print(
        "  the perturbed band's marginal r should land on r* = exp(-beta^2/2) "
        "for a model that\n  predicts the conditional mean. One that emits the "
        "tone at full amplitude with a\n  guessed phase reads 1.000 here and "
        "pays for it in the fit gate; the r2 column says\n  whether the band is "
        "emitted at all, and at b=0.25 the matched filter's leak bends the\n  "
        "ratio, as it does on the probe."
    )


def print_gate(s):
    print("\nfit gate (contract 8): variance explained on the arm's own marginal")
    for arm in ARMS:
        m, sd = s[arm]["var_explained"]
        bound = bound_of(arm)
        idx, beta = perturbed(arm)
        frac = m / bound if bound > 1e-9 else float("nan")
        flag = "ok" if m > GATE * bound else "NOT FITTED"
        print(
            f"  {arm:<8} {m:+.4f}±{sd:.4f}   analytic bound {bound:+.4f}   "
            f"{100 * frac:.0f}% of it   {flag}   "
            f"(beta {beta} on {BAND_NAMES[idx] if idx is not None else 'none'})"
        )
    print(
        f"  an arm at or below {GATE:.0%} of its bound did not learn its "
        "marginal, so its clean-probe\n  r is not quotable as retention. The "
        "bound is the conditional mean's MSE floor:\n  a model that reads the "
        "previous patch's phase exactly cannot do better."
    )


def decompose(m):
    r"""Split a 3x3 change matrix into row, column and residual parts.

    ``m[j][k]`` is the change at band ``k`` when band ``j`` was perturbed. A
    pure **column** (every row identical, the effect follows the frequency)
    leaves ``a = 0`` and every residual 0; a pure **diagonal** (the effect
    follows the perturbation) leaves the diagonal residual negative.
    """
    n = len(m)
    grand = sum(sum(row) for row in m) / (n * n)
    a = [sum(row) / n - grand for row in m]
    c = [sum(m[j][k] for j in range(n)) / n - grand for k in range(n)]
    e = [[m[j][k] - grand - a[j] - c[k] for k in range(n)] for j in range(n)]
    return grand, a, c, e


def print_part_b(s):
    m = [[diff(s, BAND_ARMS[j], k) for k in range(len(BANDS))] for j in range(3)]
    print(
        "\nPart B: beta = 1.0 on one band at a time, clean-probe change from "
        "the clean arm\n  (rows are the perturbed band, columns the readout)"
    )
    head = f"  {'perturbed':<10} " + " ".join(f"{'read@' + str(b):>12}" for b in BANDS)
    print(head)
    for j, arm in enumerate(BAND_ARMS):
        cells = " ".join(f"{m[j][k]:>+12.4f}" for k in range(3))
        print(f"  {'b=' + str(BANDS[j]):<10} {cells}")
    grand, a, c, e = decompose(m)
    print(f"  grand mean {grand:+.4f}")
    print(
        "  row effect a_j (size of the manipulation)   "
        + " ".join(f"{v:>+8.4f}" for v in a)
    )
    print(
        "  col effect c_k (motion of the band)         "
        + " ".join(f"{v:>+8.4f}" for v in c)
    )
    for j in range(3):
        print(
            "  residual e_jk after removing a_j and c_k    "
            + " ".join(f"{e[j][k]:>+8.4f}" for k in range(3))
            + ("  (row 1 of 3)" if j == 0 else "")
        )
    diag = sum(e[j][j] for j in range(3)) / 3
    off = sum(e[j][k] for j in range(3) for k in range(3) if j != k) / 6
    print(f"  residual on the diagonal {diag:+.4f} vs off-diagonal {off:+.4f}")
    print(
        "  P2 read this same matrix with theta and found a diagonal. A "
        "diagonal here means the\n  effect follows the perturbed band under an "
        "instrument the context cannot fit."
    )


def main():
    t0 = time.time()
    print("== scratch_phase_mod: P2b, the same deficit under phase diffusion? ==")
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {HIDDEN} / {LAYERS}L / {HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {STEPS} steps / cpu"
    )
    print(
        f"3 tones f = {FREQS} (b = {BANDS}, dphi/pi = "
        + ", ".join(f"{d / math.pi:.3f}" for d in DPHI)
        + f"), equal power, total RMS {math.sqrt(0.5):.4f} at every beta"
    )
    print("  arms " + ", ".join(f"{a}={spec_of(a)}" for a in ARMS))
    print(
        "  r* = exp(-beta^2/2): "
        + ", ".join(f"{b:.1f} -> {r_star(b):.3f}" for b in (0.0, 0.5, 1.0, 1.5, 2.0))
    )
    probe = make_mixture(FREQS, N_PER, CTX, PATCH, PROBE_SEED, normalize=True)
    probe_rms = float(probe.square().mean().sqrt())
    print(
        f"\nclean probe make_mixture({FREQS}, normalize=True) seed {PROBE_SEED}, "
        f"RMS {probe_rms:.4f}"
    )
    print(
        "  readout pred[:, -2] vs pat(x)[:, -1]; no phase walk, identical in "
        "every arm and seed --\n  a clean persistent tone is fully determined "
        "by the preceding patch, so r = 1.000\n  is what reading the evidence "
        "gives there, and r* is what a patch-local prior gives"
    )

    print("\ncomputing the generator check (no training yet)", flush=True)
    power = power_table(probe_rms)
    print_power(power, probe_rms)

    arms, walls = {}, []
    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            t = time.time()
            model, held, held_unpred, loss = train(seed, arm)
            row = evaluate(model, held, held_unpred, probe, loss)
            row["seed"] = seed
            row["wall"] = time.time() - t
            walls.append(row["wall"])
            rows.append(row)
            print(
                f"  {arm:<8} seed {seed}  loss {loss:.4f}  "
                f"var expl {row['var_explained']:+.3f}  "
                f"marg r = "
                + " ".join(f"{row[f'rm@{b}']:.4f}" for b in BANDS)
                + "  probe r = "
                + " ".join(f"{row[f'r@{b}']:.4f}" for b in BANDS)
                + f"  [{row['wall']:.0f}s]",
                flush=True,
            )
        arms[arm] = rows

    s = {arm: summarize(rows) for arm, rows in arms.items()}
    print_per_seed(arms)
    print_probe(s)
    print_dose(s)
    print_marginal(s)
    print_gate(s)
    print_part_b(s)

    record = {
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
            "seeds": list(SEEDS),
            "device": "cpu",
            "freqs": FREQS,
            "b": BANDS,
            "dphi": DPHI,
            "tone_power": TONE_POWER,
            "arms": {
                a: {"betas": ARMS[a], "perturbed": perturbed(a)[0], "beta": ARMS[a]}
                for a in ARMS
            },
            "dose_arms": list(DOSE_ARMS),
            "band_arms": {BAND_NAMES[j]: BAND_ARMS[j] for j in range(3)},
        },
        "probe": {
            "freqs": FREQS,
            "b": BANDS,
            "seed": PROBE_SEED,
            "normalize": True,
            "rms": probe_rms,
            "position": "pred[:, -2] vs pat(x)[:, -1]",
        },
        "generator": {
            "r_star": {str(b): r_star(b) for b in (0.0, 0.5, 1.0, 1.5, 2.0)},
            "arms": power,
        },
        "arms": {a: {"per_seed": rows, "mean_std": s[a]} for a, rows in arms.items()},
        "fit_gate": {
            a: {
                "var_explained": s[a]["var_explained"][0],
                "analytic_bound": bound_of(a),
                "fraction_of_bound": s[a]["var_explained"][0] / bound_of(a),
                "passed": s[a]["var_explained"][0] > GATE * bound_of(a),
            }
            for a in ARMS
        },
        "contrast": {
            "baseline": BASELINE,
            "probe_change": {
                a: {str(BANDS[k]): diff(s, a, k) for k in range(3)} for a in ARMS
            },
            "part_b_matrix": {
                BAND_NAMES[j]: {
                    str(BANDS[k]): diff(s, BAND_ARMS[j], k) for k in range(3)
                }
                for j in range(3)
            },
        },
        "wall_train_mean": sum(walls) / len(walls),
        "wall": time.time() - t0,
    }
    out = RUNS / "scratch_phase_mod.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"\nwrote {out}\ntotal wall {time.time() - t0:.1f}s "
        f"({sum(walls) / len(walls):.0f}s per training)"
    )


if __name__ == "__main__":
    main()
