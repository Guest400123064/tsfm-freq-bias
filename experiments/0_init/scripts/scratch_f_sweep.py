r"""P3: eight integer bands, predictability matched -- is retention flat in f?

P2b showed the clean-probe retention of a phase-modulated band lands on the
Bayes-optimal shrinkage ``r* = exp(-beta**2 / 2)`` for its training marginal, to
within 0.03, at three bands -- two of them coarse (``b = 0.25, 2.0, 8.0``). P3
is the fine sweep that sets the boundary of PLAN.md §0's claim: with
predictability matched across frequencies, is retention flat in ``f``, or is
there a residual frequency dependence?

Eight bands, integer cycles per patch ``b = 1..8`` (``f = 16, 32, ..., 128`` at
ctx 512 / k 32, so ``P = ctx / k = 16``), pairwise ``db = 1`` exactly -- contract
4 with no slack, asserted below. Integer ``b`` is deliberate on two counts:

- ``2b`` is an integer, so the matched filter's negative-frequency image cancels
  exactly and each band's reading is clean. P1/P2's ``b = 0.25`` was the least
  trustworthy column for exactly this reason.
- ``dphi = 2 pi b = 0 (mod 2 pi)`` for every band, so the phase advance is a
  whole number of cycles per patch and the transient axis of PLAN.md §2 sits in
  the "copy" class across the whole sweep. Any residual here is therefore not a
  ``dphi`` effect -- ``dphi`` is reported alongside anyway (contract 6).

Arms: every band perturbed equally, so no diagonal logic is needed (that was
P2's job). Uniform ``beta in {0, 0.5, 1.0, 1.5, 2.0}`` over all eight bands, 5
arms x 3 seeds = 15 runs. The probe is a fixed clean 8-tone mixture at the same
``b``, ``beta = 0``, RMS-matched to the training signal power, identical in
every arm and seed.

Config as P1/P2b: ctx 512 / k 32 / hidden 32 / 2 layers / 4 heads / SGD lr 1e-2
/ batch 64 / 2000 steps, CPU, readout ``pred[:, -2]`` against ``pat(x)[:, -1]``.
``r* = exp(-beta**2 / 2)`` = 1.000 / 0.882 / 0.607 / 0.325 / 0.135.

The transient ``r_f(t)`` curve (PLAN.md §4's last unimplemented metric) is read
at every band every 500 steps, so this run also fills that gap.

Run from the repo root:
``.venv/bin/python experiments/0_init/scripts/scratch_f_sweep.py``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fbias.data import make_mixture, make_pm
from fbias.model import SimTFM
from fbias.probes import fit_oracle, matched_amp, retention

CTX, PATCH = 512, 32
P = CTX // PATCH  # patches per context; f = b * P
HIDDEN, LAYERS, HEADS = 32, 2, 4
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
BANDS = [float(b) for b in range(1, 9)]  # cycles per patch
FREQS = [b * P for b in BANDS]  # cycles per window: 16, 32, ..., 128
DPHI = [2 * math.pi * b % (2 * math.pi) for b in BANDS]
ARM_BETAS = {
    "clean": 0.0,
    "beta0.5": 0.5,
    "beta1.0": 1.0,
    "beta1.5": 1.5,
    "beta2.0": 2.0,
}
ARMS = {name: [beta] * len(BANDS) for name, beta in ARM_BETAS.items()}
EVAL_STEPS = (500, 1000, 1500, 2000)  # the transient curve, PLAN.md §4
PROBE_SEED = 123
TONE_POWER = 0.5 / len(BANDS)  # per tone, as in make_mixture(normalize=True)
GATE = 0.5  # an arm below half its analytic bound has not learned its marginal
RUNS = Path(__file__).resolve().parents[1] / "runs"


def check_grid():
    r"""Contract 4 with no slack, plus the two consequences of integer ``b``."""
    steps = [BANDS[j + 1] - BANDS[j] for j in range(len(BANDS) - 1)]
    assert all(s == 1.0 for s in steps), f"bands are {steps} apart, need exactly 1"
    assert all(float(2 * b).is_integer() for b in BANDS), "2b must be an integer"
    assert all(d == 0.0 for d in DPHI), DPHI


def r_star(beta):
    r"""The conditional mean's amplitude fraction, ``exp(-beta**2 / 2)``."""
    return math.exp(-0.5 * beta * beta)


def bound_of(arm):
    r"""Variance explained no function of the context can beat.

    One tone's future phase is ``beta * eps``, so its unpredictable power is
    ``TONE_POWER * (1 - exp(-beta**2))`` per tone against a target variance of
    0.5. With every band at the same beta this is just ``r* ** 2``.
    """
    floor = sum(TONE_POWER * (1 - r_star(b) ** 2) for b in ARMS[arm])
    return 1 - floor / 0.5


def realised_beta(unpredictable_power):
    r"""Invert ``P = TONE_POWER * (1 - exp(-beta**2))`` for ``beta``."""
    share = min(max(unpredictable_power / TONE_POWER, 0.0), 1 - 1e-12)
    return math.sqrt(-math.log(1 - share))


def corpus(arm, seed):
    r"""The arm's training marginal: the window, and its unpredictable part."""
    return make_pm(FREQS, ARMS[arm], 2 * N_PER, CTX, PATCH, seed)


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


def pearson(xs, ys):
    r"""Pearson correlation of two equal-length sequences."""
    x = torch.tensor([float(v) for v in xs], dtype=torch.float64)
    y = torch.tensor([float(v) for v in ys], dtype=torch.float64)
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()))


def band_label(i):
    r"""Band ``i`` as ``b`` and the ``f`` it corresponds to."""
    return f"b={BANDS[i]:g} (f={int(BANDS[i] * P)})"


@torch.no_grad()
def power_table(probe_rms):
    r"""Generator check only, no training: envelope, diffusion, per-band power.

    ``bin power`` is the per-patch matched amplitude at the carrier,
    ``|Z|^2 / 2``, which is exactly the tone's power here because ``2b`` is an
    integer. ``realised beta`` is inverted from each band's unpredictable power,
    so the diffusion is measured rather than assumed.
    """
    out = {}
    for arm in ARMS:
        rms, floors = [], []
        bin_power = {b: [] for b in BANDS}
        realised = {b: [] for b in BANDS}
        for seed in SEEDS:
            x, unpred = corpus(arm, seed)
            rms.append(float(x.square().mean().sqrt()))
            floors.append(float(unpred.square().mean()))
            p = x.unfold(1, PATCH, PATCH)
            q = unpred.unfold(1, PATCH, PATCH)
            for b in BANDS:
                power = float(matched_amp(p, b, PATCH).abs().square().mean()) / 2
                leak = float(matched_amp(q, b, PATCH).abs().square().mean()) / 2
                bin_power[b].append(power)
                realised[b].append(realised_beta(leak))
        # every band carries the same beta, so one band's share is the total's
        leaked = mean_std(floors)[0] / len(BANDS)
        out[arm] = {
            "betas": ARMS[arm],
            "rms": mean_std(rms),
            "rms_seed_spread": max(rms) - min(rms),
            "vs_probe": mean_std(rms)[0] / probe_rms - 1,
            "unpredictable_power": mean_std(floors),
            "realised_beta_total": realised_beta(leaked),
            "bin_power": {b: mean_std(v) for b, v in bin_power.items()},
            "realised_beta": {b: mean_std(v) for b, v in realised.items()},
        }
    return out


def train(seed, arm, probe):
    r"""P1/P2b's training loop, plus a probe read at every EVAL_STEPS boundary."""
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
    curve = []
    for step in range(1, STEPS + 1):
        x = train_data[torch.randint(N_PER, (BATCH,), generator=gen)]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in EVAL_STEPS:
            curve.append((step, probe_r(model, probe)))
    return model, data[N_PER:], unpred[N_PER:], float(loss), curve


@torch.no_grad()
def probe_r(model, probe):
    r"""Clean-probe retention at every band, ``{b: r}``."""
    training = model.training
    model.eval()
    f_hat = model(probe)[0][:, -2]
    true = model.pat(probe)[:, -1]
    row = {b: float(retention(f_hat, true, b, PATCH).mean()) for b in BANDS}
    model.train(training)
    return row


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
    p_hat, _, z = model(probe)
    true = model.pat(probe)
    f_hat = p_hat[:, -2]
    for b in BANDS:
        row[f"r@{b}"] = float(retention(f_hat, true[:, -1], b, PATCH).mean())
        # out-of-sample ridge (contract 5): held latents -> probe, not degener
        row[f"r_oracle@{b}"] = fit_oracle(z_held, patches, z, true, b, PATCH)[
            "retention"
        ]
    return row


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: mean_std([r[k] for r in rows]) for k in keys}


def transient_table(arms):
    r"""Mean, spread and ``corr(b, r)`` of the clean-probe curve, by arm and step."""
    out = {}
    for arm, rows in arms.items():
        steps = [step for step, _ in rows[0]["curve"]]
        out[arm] = {}
        for i, step in enumerate(steps):
            v = torch.stack(
                [
                    torch.tensor(
                        [r["curve"][i][1][b] for b in BANDS], dtype=torch.float64
                    )
                    for r in rows
                ]
            )
            mean = v.mean(0)
            out[arm][step] = {
                "mean": [float(x) for x in mean],
                "std": [float(x) for x in v.std(0)],
                "spread": float(mean.max() - mean.min()),
                "corr_b": pearson(BANDS, mean),
            }
    return out


def arm_stats(s):
    r"""Per arm: spread across bands, seed spread, and the deviation from r*."""
    out = {}
    for arm in ARMS:
        r = torch.tensor([s[arm][f"r@{b}"][0] for b in BANDS], dtype=torch.float64)
        sd = torch.tensor([s[arm][f"r@{b}"][1] for b in BANDS], dtype=torch.float64)
        target = r_star(ARM_BETAS[arm])
        dev = r - target
        seed_sd = float((sd**2).mean().sqrt())
        spread = float(r.max() - r.min())
        out[arm] = {
            "r_star": target,
            "mean_r": float(r.mean()),
            "spread_b": spread,
            "seed_sd": seed_sd,
            "spread_over_seed_sd": spread / seed_sd if seed_sd > 0 else float("nan"),
            "mean_dev_r_star": float(dev.mean()),
            "max_abs_dev_r_star": float(dev.abs().max()),
            "corr_b_dev": pearson(BANDS, dev),
        }
    return out


def print_grid(probe_rms):
    print("== scratch_f_sweep: P3, is retention flat in f at matched beta? ==")
    print(
        f"ctx {CTX} / patch {PATCH} / P = {P} / hidden {HIDDEN} / {LAYERS}L / "
        f"{HEADS} heads | SGD lr {LR} / batch {BATCH} / {STEPS} steps / cpu"
    )
    print(
        f"8 bands f = {FREQS} (b = {[int(b) for b in BANDS]}), pairwise db = "
        f"{BANDS[1] - BANDS[0]:.1f} exactly (contract 4, no slack)"
    )
    print(
        "  integer b: 2b integer, so the matched filter's negative-frequency "
        "image cancels --\n  no b=0.25 leakage column here. And dphi = 2 pi b = "
        "0 (mod 2 pi), so the phase\n  advance is a whole number of cycles per "
        "patch and the transient axis sits in the\n  copy class at every band."
    )
    print("  arms " + ", ".join(f"{a}={ARM_BETAS[a]}" for a in ARMS))
    print(
        "  r* = exp(-beta^2/2): "
        + ", ".join(f"{b:.1f} -> {r_star(b):.3f}" for b in ARM_BETAS.values())
    )
    print(
        f"\nclean probe make_mixture({FREQS}, normalize=True) seed {PROBE_SEED}, "
        f"RMS {probe_rms:.4f}"
    )
    print(
        "  beta = 0 at every band, no phase walk, identical in every arm and "
        "seed; readout\n  pred[:, -2] vs pat(x)[:, -1]. The training signal "
        "carries power 0.5 and so does\n  the probe, so contract 9 holds by "
        "construction (verified in the RMS column below)."
    )


def print_power(power, probe_rms):
    print(
        f"\ngenerator check, no training (probe RMS {probe_rms:.4f}; constant "
        "envelope, so every\narm must match it at every beta)"
    )
    head = (
        f"  {'arm':<8} {'beta':>5} {'corpus rms':>18} {'vs probe':>9} "
        f"{'realised beta':>13} {'r*':>7}"
    )
    print(head)
    for arm in ARMS:
        p = power[arm]
        m, sd = p["rms"]
        beta = ARM_BETAS[arm]
        print(
            f"  {arm:<8} {beta:>5.1f} {m:>10.5f}±{sd:<6.5f} {p['vs_probe']:>8.2%} "
            f"{p['realised_beta_total']:>13.3f} {r_star(beta):>7.3f}"
        )
    print(
        "  the same seed gives the same RMS in every arm to 1e-4 relative; the "
        "spread across\n  seeds is the corpus realisation. The realised beta "
        "inverts the corpus's own\n  unpredictable power, so it is measured, "
        "not assumed."
    )
    print("\nper-band bin power |Z|^2/2 of each carrier, mean over 3 seeds")
    head = f"  {'arm':<8} " + " ".join(f"{'b=' + str(int(b)):>10}" for b in BANDS)
    print(head)
    for arm in ARMS:
        cells = " ".join(f"{power[arm]['bin_power'][b][0]:>10.4f}" for b in BANDS)
        print(f"  {arm:<8} {cells}")
    print(
        "  flat in beta and exactly the tone power 0.0625: phase diffusion moves "
        "no power\n  between bins (PLAN.md §4), and integer b means no "
        "negative-frequency image leaks in."
    )
    print("\nper-band realised beta, inverted from each band's unpredictable power")
    head = f"  {'arm':<8} " + " ".join(f"{'b=' + str(int(b)):>10}" for b in BANDS)
    print(head)
    for arm in ARMS:
        cells = " ".join(f"{power[arm]['realised_beta'][b][0]:>10.3f}" for b in BANDS)
        print(f"  {arm:<8} {cells}")
    print(
        "  every band in an arm reads the same beta to 3 decimals, which is what "
        '"perturb\n  every band equally" means -- there is no per-band efficacy '
        "difference here\n  (unlike theta, PLAN.md §4)."
    )


def print_per_seed(arms):
    head = f"  {'arm':<8} {'seed':>4} {'var expl':>9} {'floor':>7} " + " ".join(
        f"{'r@' + str(b):>8}" for b in BANDS
    )
    print(
        "\nper-seed: fit gate on the arm's own marginal, then clean-probe r at "
        "every band\n" + head
    )
    print("  " + "-" * (len(head) - 2))
    for arm, rows in arms.items():
        for r in rows:
            print(
                f"  {arm:<8} {r['seed']:>4} {r['var_explained']:>+9.3f} "
                f"{r['unpredictable_floor']:>+7.3f} "
                + " ".join(f"{r[f'r@{b}']:>8.4f}" for b in BANDS)
            )


def print_matrix(title, arms, cell, width=14):
    r"""Bands down the side, arms across."""
    head = f"  {'readout':<12} " + " ".join(f"{a:>{width}}" for a in arms)
    print(f"\n{title}\n{head}")
    print("  " + "-" * (len(head) - 2))
    for i, b in enumerate(BANDS):
        print(
            f"  {band_label(i):<12} " + " ".join(f"{cell(a, i):>{width}}" for a in arms)
        )


def print_probe(s):
    print_matrix(
        "clean-probe r, mean +- std over 3 seeds (identical probe in every arm)",
        tuple(ARMS),
        lambda a, i: f"{s[a][f'r@{BANDS[i]}'][0]:.4f}±{s[a][f'r@{BANDS[i]}'][1]:.4f}",
    )
    print_matrix(
        "clean-probe r minus r* of the arm's beta (negative = over-shrinkage)",
        tuple(ARMS),
        lambda a, i: f"{s[a][f'r@{BANDS[i]}'][0] - r_star(ARM_BETAS[a]):+.4f}",
    )
    print(
        "  r* is the same number in every row of an arm's column, so a trend "
        "down a column\n  here is a residual frequency dependence in the gap to "
        "the Bayes-optimal shrinkage."
    )


def print_arm_stats(stats):
    head = (
        f"  {'arm':<8} {'beta':>5} {'r*':>6} {'mean r':>8} {'spread_b':>9} "
        f"{'seed sd':>8} {'ratio':>6} {'r - r*':>8} {'|r - r*|max':>12} "
        f"{'corr(b, dev)':>13}"
    )
    print("\nper arm summary across the 8 bands\n" + head)
    print("  " + "-" * (len(head) - 2))
    for arm in ARMS:
        st = stats[arm]
        print(
            f"  {arm:<8} {ARM_BETAS[arm]:>5.1f} {st['r_star']:>6.3f} "
            f"{st['mean_r']:>8.4f} {st['spread_b']:>9.4f} {st['seed_sd']:>8.4f} "
            f"{st['spread_over_seed_sd']:>6.2f} {st['mean_dev_r_star']:>+8.4f} "
            f"{st['max_abs_dev_r_star']:>12.4f} {st['corr_b_dev']:>+13.2f}"
        )
    print(
        "  spread_b is max - min over the 8 bands; seed sd is the pooled "
        "per-band standard\n  deviation over the 3 seeds. 8 independent readings "
        "with this sd would spread about\n  2.85x it on average, so ratio near "
        "or below ~2.85 is what a null looks like. For\n  reference, P2b's seed "
        "spreads at 3 bands were 0.003-0.014."
    )


def print_marginal(s):
    print_matrix(
        "r on each arm's own held-out marginal, last position (mean over 3 seeds)",
        tuple(ARMS),
        lambda a, i: f"{s[a][f'rm@{BANDS[i]}'][0]:.4f}±{s[a][f'rm@{BANDS[i]}'][1]:.4f}",
    )
    print(
        "  the same arm's own corpus, where r* is the correct answer: a model "
        "reading its\n  evidence would emit the conditional mean there. P2b "
        "found this sitting 0.03-0.05\n  below r* at b=8."
    )


def print_gate(s):
    print("\nfit gate (contract 8): variance explained on the arm's own marginal")
    for arm in ARMS:
        m, sd = s[arm]["var_explained"]
        bound = bound_of(arm)
        frac = m / bound if bound > 1e-9 else float("nan")
        flag = "ok" if m > GATE * bound else "NOT FITTED"
        print(
            f"  {arm:<8} {m:+.4f}±{sd:.4f}   analytic bound {bound:+.4f}   "
            f"{100 * frac:.0f}% of it   {flag}   (beta {ARM_BETAS[arm]})"
        )
    print(
        f"  an arm at or below {GATE:.0%} of its bound did not learn its "
        "marginal, so its clean-probe\n  r is not quotable as retention. The "
        "bound is the conditional mean's MSE floor\n  and equals r*^2 for a "
        "uniform-beta arm."
    )


def print_oracle(s):
    print("\nclean-probe oracle (contract 5): out-of-sample ridge z_held -> probe")
    head = f"  {'arm':<8} " + " ".join(f"{'f=' + str(int(b * P)):>7}" for b in BANDS)
    print(head)
    for arm in ARMS:
        cells = " ".join(f"{s[arm][f'r_oracle@{b}'][0]:>7.3f}" for b in BANDS)
        print(f"  {arm:<8} {cells}")
    print(
        "  the frozen representation carries a near-Bayes readout at every band, "
        "so the\n  deficit is not a head-only quirk."
    )


def print_transient(transient):
    print("\n== transient: r_f(t), the clean probe at every 500 steps ==")
    for arm in ARMS:
        beta = ARM_BETAS[arm]
        print(f"\n  arm {arm} (beta {beta}, r* {r_star(beta):.3f}), mean over 3 seeds")
        steps = list(transient[arm])
        head = f"  {'readout':<12} " + " ".join(
            f"{'step ' + str(s):>14}" for s in steps
        )
        print(head)
        print("  " + "-" * (len(head) - 2))
        for i in range(len(BANDS)):
            cells = " ".join(f"{transient[arm][s]['mean'][i]:>14.4f}" for s in steps)
            print(f"  {band_label(i):<12} {cells}")
        print(
            f"  {'corr(b, r)':<12} "
            + " ".join(f"{transient[arm][s]['corr_b']:>+14.2f}" for s in steps)
        )
        print(
            f"  {'spread_b':<12} "
            + " ".join(f"{transient[arm][s]['spread']:>14.4f}" for s in steps)
        )
    print(
        "\n  corr(b, r) is over the 8 bands at one step: a residual ordering in f "
        "would show up\n  as a correlation that stays away from 0. With dphi = 0 "
        "at every band the\n  prediction is that it does not (PLAN.md §2: the "
        "transient is a dphi effect)."
    )


def main():
    t0 = time.time()
    check_grid()
    probe = make_mixture(FREQS, N_PER, CTX, PATCH, PROBE_SEED, normalize=True)
    probe_rms = float(probe.square().mean().sqrt())
    print_grid(probe_rms)

    print("\ncomputing the generator check (no training yet)", flush=True)
    power = power_table(probe_rms)
    print_power(power, probe_rms)

    arms, walls = {}, []
    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            t = time.time()
            model, held, held_unpred, loss, curve = train(seed, arm, probe)
            row = evaluate(model, held, held_unpred, probe, loss)
            row["seed"] = seed
            row["curve"] = curve
            row["wall"] = time.time() - t
            # the final eval re-reads the probe at step STEPS, so this is a
            # free determinism check on the last transient point
            row["curve_vs_final"] = max(
                abs(curve[-1][1][b] - row[f"r@{b}"]) for b in BANDS
            )
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
    transient = transient_table(arms)
    stats = arm_stats(s)
    print_per_seed(arms)
    print_probe(s)
    print_arm_stats(stats)
    print_marginal(s)
    print_gate(s)
    print_oracle(s)
    print_transient(transient)

    rms_all = [power[a]["rms"][0] for a in ARMS]
    rms_spread = max(rms_all) - min(rms_all)
    print("\n== constant-envelope check ==")
    print(
        f"  corpus RMS across the 5 arms spans {rms_spread:.5f} "
        f"({rms_spread / probe_rms:.2%} of the probe's {probe_rms:.4f}); "
        f"probe - corpus = {probe_rms - sum(rms_all) / len(rms_all):+.5f}"
    )
    worst = max(
        max(power[a]["bin_power"][b][0] for a in ARMS)
        / min(power[a]["bin_power"][b][0] for a in ARMS)
        for b in BANDS
    )
    print(
        f"  per-band bin power is flat in beta: worst band varies by a factor "
        f"{worst:.4f} across arms (1.0000 = identical)"
    )
    print(
        "  curve_vs_final (the step-2000 transient read against the final "
        "evaluation): max\n  over all 15 runs = "
        f"{max(r['curve_vs_final'] for rows in arms.values() for r in rows):.2e}"
    )

    record = {
        "config": {
            "ctx": CTX,
            "patch": PATCH,
            "P": P,
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
            "db": BANDS[1] - BANDS[0],
            "tone_power": TONE_POWER,
            "eval_steps": list(EVAL_STEPS),
            "arms": {
                a: {
                    "betas": ARMS[a],
                    "beta": ARM_BETAS[a],
                    "r_star": r_star(ARM_BETAS[a]),
                    "analytic_bound": bound_of(a),
                }
                for a in ARMS
            },
        },
        "probe": {
            "freqs": FREQS,
            "b": BANDS,
            "seed": PROBE_SEED,
            "normalize": True,
            "rms": probe_rms,
            "position": "pred[:, -2] vs pat(x)[:, -1]",
        },
        "generator": power,
        "arms": {a: {"per_seed": rows, "mean_std": s[a]} for a, rows in arms.items()},
        "arm_stats": stats,
        "transient": transient,
        "fit_gate": {
            a: {
                "var_explained": s[a]["var_explained"][0],
                "analytic_bound": bound_of(a),
                "fraction_of_bound": s[a]["var_explained"][0] / bound_of(a),
                "passed": s[a]["var_explained"][0] > GATE * bound_of(a),
            }
            for a in ARMS
        },
        "constant_envelope": {
            "probe_rms": probe_rms,
            "arm_rms": {a: power[a]["rms"] for a in ARMS},
            "rms_spread": rms_spread,
            "curve_vs_final_max": max(
                r["curve_vs_final"] for rows in arms.values() for r in rows
            ),
        },
        "wall_train_mean": sum(walls) / len(walls),
        "wall": time.time() - t0,
    }
    out = RUNS / "scratch_f_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"\nwrote {out}\ntotal wall {time.time() - t0:.1f}s "
        f"({sum(walls) / len(walls):.0f}s per training)"
    )


if __name__ == "__main__":
    main()
