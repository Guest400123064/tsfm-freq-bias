r"""P2: does the clean-probe deficit follow the perturbed band, or the frequency?

P1 held a clean probe fixed and moved only the *high* band's coherent fraction
theta; its clean-probe retention fell monotonically (r@8 = 0.9995 -> 0.0362,
PLAN.md §5). That cannot separate two readings, and they predict the same number
when the high band is the only band ever perturbed:

- "the model damps the band that was unpredictable" -- the effect follows the
  *perturbation*, and "frequency per se is not the variable" (PLAN.md §0);
- "the model damps high frequencies" -- the effect follows the *frequency*, and
  the §0 claim fails.

Here each of three bands is perturbed in turn. The corpus holds three tones
``b = 0.25, 2.0, 8.0`` (``f = 4, 32, 128`` cycles per window at ctx 512 /
patch 32), each carrying a third of the signal power, so the total RMS is the
same in every arm (PLAN.md §3.9). One band at a time loses part of its coherent
tone to incoherent energy confined to that band's own range -- that band's
theta -- while the other two stay perfectly coherent.

  arm     perturbed band   theta there
  clean   none             1.0 everywhere
  lo_50   b = 0.25         0.5
  mid_50  b = 2.0          0.5
  hi_50   b = 8.0          0.5
  lo_00   b = 0.25         0.0
  mid_00  b = 2.0          0.0
  hi_00   b = 8.0          0.0

The probe is the clean 3-tone mixture at those same ``b``, no noise,
RMS-normalised to the training signal power, identical in every arm and seed.

The readout is ``r`` at all three ``b`` on that probe, **minus the clean arm's
value**. The subtraction is the point: P1 saw perturbing the high band move the
*low* band by -0.12, and ``b = 0.25`` carries a different phase offset
(``dphi = pi / 2``) from the matched filter's negative-frequency image, so raw
``r`` mixes the perturbation's effect with effects that are there anyway.

A **diagonal** (each arm drops most at the band it perturbed) says the effect
follows the perturbation and supports §0. A **column** (the high band drops in
every arm) says the effect follows the frequency and §0 fails. Anything in
between is quantified below.

Bands for the incoherent part, cycles per window over the ``ctx + k = 544``
samples: low ``[2, 8]`` (FFT bins 3-8), mid ``[24, 40]`` (bins 26-42), high
``[96, 160]`` (bins 102-170). Disjoint, each containing its tone. They are not
equally wide, and the realised in-bin coherent:incoherent ratio at a given theta
scales with the bin count, so at theta = 0.5 the low-band arm is a much stronger
manipulation than the high-band arm. The script measures and prints the realised
ratio (P0's lesson) rather than trusting the nominal theta.

Config as P1: ctx 512 / k 32 / hidden 32 / 2 layers / 4 heads / SGD lr 1e-2 /
batch 64 / 2000 steps, CPU, readout ``pred[:, -2]`` against ``pat(x)[:, -1]``.

Run from the repo root:
``.venv/bin/python experiments/0_init/scripts/scratch_f_invariance.py``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fbias.data import b_of, make_banded, make_mixture
from fbias.model import SimTFM
from fbias.probes import band_r2, fit_oracle, matched_amp, retention

CTX, PATCH = 512, 32
HIDDEN, LAYERS, HEADS = 32, 2, 4
STEPS, BATCH, LR = 2000, 64, 1e-2
N_PER = 256
SEEDS = (0, 1, 2)
FREQS = [4.0, 32.0, 128.0]  # b = 0.25, 2.0, 8.0
BAND_NAMES = ("lo", "mid", "hi")
NOISE_BANDS = [(2.0, 8.0), (24.0, 40.0), (96.0, 160.0)]
BANDS = [b_of(f, PATCH, CTX) for f in FREQS]
DPHI = [2 * math.pi * b % (2 * math.pi) for b in BANDS]
PROBE_SEED = 123
BASELINE = "clean"
GROUPS = (
    ("theta = 0.5", ("lo_50", "mid_50", "hi_50")),
    ("theta = 0.0", ("lo_00", "mid_00", "hi_00")),
)
# arm -> (index of the perturbed band, or None, theta in that band)
ARMS = {
    "clean": (None, 1.0),
    "lo_50": (0, 0.5),
    "mid_50": (1, 0.5),
    "hi_50": (2, 0.5),
    "lo_00": (0, 0.0),
    "mid_00": (1, 0.0),
    "hi_00": (2, 0.0),
}
RUNS = Path(__file__).resolve().parents[1] / "runs"


def thetas_of(idx, theta):
    r"""Per-tone coherent fractions of an arm: 1 everywhere but the perturbed band."""
    thetas = [1.0, 1.0, 1.0]
    if idx is not None:
        thetas[idx] = theta
    return thetas


def spec_of(arm):
    r"""The arm's ``(perturbed band, theta)`` spec, for header lines."""
    idx, theta = ARMS[arm]
    return f"({'-' if idx is None else BAND_NAMES[idx]}, {theta})"


def band_label(idx):
    r"""The perturbed band of an arm, with its ``b``, or ``none``."""
    return "none" if idx is None else f"{BAND_NAMES[idx]} b={BANDS[idx]}"


def corpus(arm, seed):
    r"""The arm's training marginal, plus that arm's incoherent part."""
    idx, theta = ARMS[arm]
    thetas = thetas_of(idx, theta)
    return make_banded(FREQS, NOISE_BANDS, thetas, 2 * N_PER, CTX, PATCH, seed)


def mean_std(vals):
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.mean()), float(v.std())


@torch.no_grad()
def rms_table(probe_rms):
    r"""Corpus RMS per arm against the probe, and the incoherent part's share."""
    out = {}
    for arm in ARMS:
        total = [float(corpus(arm, s)[0].square().mean().sqrt()) for s in SEEDS]
        inc = [float(corpus(arm, s)[1].square().mean().sqrt()) for s in SEEDS]
        out[arm] = {
            "rms": mean_std(total),
            "incoherent_rms": mean_std(inc),
            "vs_probe": mean_std(total)[0] / probe_rms - 1,
        }
    return out


@torch.no_grad()
def bin_ratio(arm, seed):
    r"""Coherent/incoherent power in each probe bin, per patch, from the corpus.

    A persistent tone puts all its power in one bin while the incoherent energy
    spreads over the band, so this is far above ``theta / (1 - theta)``. It is
    the ratio the model actually faces, not the nominal theta.
    """
    x, incoherent = corpus(arm, seed)
    tone = (x - incoherent).unfold(1, PATCH, PATCH)
    noise = incoherent.unfold(1, PATCH, PATCH)
    out = {}
    for b in BANDS:
        coh = float(matched_amp(tone, b, PATCH).abs().square().mean())
        inc = float(matched_amp(noise, b, PATCH).abs().square().mean())
        out[str(b)] = coh / inc if inc > 1e-12 else None
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
    data, noise = corpus(arm, seed)
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
    return model, data[N_PER:], noise[N_PER:], float(loss)


@torch.no_grad()
def evaluate(model, held, held_noise, probe, train_loss):
    r"""Fit gate on the arm's own marginal (contract 8), then the clean probe."""
    model.eval()
    pred, _, z_held = model(held)
    patches = model.pat(held)
    target = patches[:, 1:]
    mse = float(F.mse_loss(pred[:, :-1], target))
    var = float(target.var(unbiased=False))
    floor = float(model.pat(held_noise)[:, 1:].square().mean())
    row = {
        "train_loss": train_loss,
        "held_var": var,
        "var_explained": 1 - mse / var,
        # the target's own unpredictable energy: no function of the past has it
        "unpredictable_floor": 1 - floor / var,
    }
    p_hat, _, z = model(probe)
    true = model.pat(probe)
    f_hat = p_hat[:, -2]
    for b in BANDS:
        row[f"r@{b}"] = float(retention(f_hat, true[:, -1], b, PATCH).mean())
        row[f"r2@{b}"] = band_r2(f_hat, true[:, -1], b, PATCH)
        # out-of-sample ridge (contract 5): held corpus -> probe, so not degenerate
        row[f"r_oracle@{b}"] = fit_oracle(z_held, patches, z, true, b, PATCH)[
            "retention"
        ]
    return row


def summarize(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {k: mean_std([r[k] for r in rows]) for k in keys}


def print_probe(probe_rms):
    print(
        f"clean probe make_mixture({FREQS}, normalize=True) seed {PROBE_SEED}, "
        f"b = {BANDS}, RMS {probe_rms:.4f}"
    )
    print(
        "  dphi/pi = "
        + ", ".join(f"{b}: {d / math.pi:.3f}" for b, d in zip(BANDS, DPHI))
        + "  (b = 0.25 differs, so the clean control must be subtracted)"
    )
    print(
        "  readout pred[:, -2] vs pat(x)[:, -1]; 3 tones, equal power, "
        "no noise, identical in every arm and seed"
    )


def print_rms(rms, probe_rms):
    print(f"\nRMS of each arm's corpus against the probe ({probe_rms:.4f})")
    head = (
        f"  {'arm':<8} {'corpus rms':>18} {'incoherent rms':>16} {'vs probe':>10} "
        f"{'perturbed band':>18}"
    )
    print(head)
    for arm in ARMS:
        idx, theta = ARMS[arm]
        r = rms[arm]
        print(
            f"  {arm:<8} {r['rms'][0]:>10.4f}±{r['rms'][1]:<6.4f} "
            f"{r['incoherent_rms'][0]:>10.4f}±{r['incoherent_rms'][1]:<4.4f} "
            f"{r['vs_probe']:>9.2%} {band_label(idx):>18}"
        )
    print(
        "  incoherent rms is sqrt((1 - theta) / 6): the power moves out of the "
        "tone, the total does not change"
    )


def print_bin_ratio(ratios):
    print("\nrealised coherent:incoherent power in each probe bin (per patch)")
    head = (
        f"  {'arm':<8} "
        + " ".join(f"{'b=' + str(b):>12}" for b in BANDS)
        + "   nominal"
    )
    print(head)
    for arm in ARMS:
        idx, theta = ARMS[arm]
        cells = " ".join(
            "n/a".rjust(12)
            if ratios[arm][str(b)] is None
            else f"{ratios[arm][str(b)]:>12.3f}"
            for b in BANDS
        )
        print(f"  {arm:<8} {cells}   theta {theta} in {band_label(idx)} band")
    print(
        "  read the perturbed band's own column. Every arm carries the same nominal\n"
        "  theta, but not the same realised ratio: the same incoherent power spread\n"
        "  over fewer FFT bins leaves more of it in the tone's own bin, so at a given\n"
        "  theta the low-band arm is the harsher manipulation. The other two columns\n"
        "  are what leaks into bands that were left fully coherent."
    )


def print_per_seed(arms):
    head = (
        f"{'arm':<8} {'seed':>4} {'var expl':>9} {'floor':>7} "
        + " ".join(f"{'r@' + str(b):>8}" for b in BANDS)
        + " "
        + " ".join(f"{'r2@' + str(b):>9}" for b in BANDS)
    )
    print(
        "\nper-seed: fit gate on the arm's own marginal, then the clean probe\n" + head
    )
    print("-" * len(head))
    for arm, rows in arms.items():
        for r in rows:
            print(
                f"{arm:<8} {r['seed']:>4} {r['var_explained']:>+9.3f} "
                f"{r['unpredictable_floor']:>+7.3f} "
                + " ".join(f"{r[f'r@{b}']:>8.4f}" for b in BANDS)
                + " "
                + " ".join(f"{r[f'r2@{b}']:>9.4f}" for b in BANDS)
            )


def print_matrix(title, cell, width=14):
    r"""3 bands x 7 arms, rows are the readout band."""
    head = f"{'readout':<8} " + " ".join(f"{a:>{width}}" for a in ARMS)
    print(f"\n{title}\n" + head)
    print("-" * len(head))
    for i, b in enumerate(BANDS):
        print(f"b={b:<6} " + " ".join(f"{cell(arm, i):>{width}}" for arm in ARMS))


def print_raw(s):
    print_matrix(
        "mean r over 3 seeds, raw, on the clean probe",
        lambda arm, i: (
            f"{s[arm][f'r@{BANDS[i]}'][0]:.4f}±{s[arm][f'r@{BANDS[i]}'][1]:.4f}"
        ),
    )
    print("  columns are arms; every arm sees the same clean probe")
    print(
        "\nfit gate (contract 8): variance explained on each arm's own marginal\n"
        "  the analytic bound treats the target's incoherent energy as unpredictable,\n"
        "  so an arm above it is predicting part of that energy, and an arm well\n"
        "  below it did not learn its marginal and its r is not evidence"
    )
    for arm in ARMS:
        m, sd = s[arm]["var_explained"]
        fl = s[arm]["unpredictable_floor"][0]
        room = 1 - fl  # the incoherent power, as a fraction of the target variance
        if room > 1e-9 and m - fl > 1e-9:
            note = (
                f"  above it by {m - fl:+.4f}, i.e. {100 * (m - fl) / room:.0f}% "
                "of the incoherent power"
            )
        else:
            note = f"  {m - fl:+.4f} against it"
        print(f"  {arm:<8} {m:+.4f}±{sd:.4f}   analytic bound {fl:+.4f}{note}")
    print("\nclean-probe oracle (contract 5): out-of-sample ridge z_held -> probe")
    for arm in ARMS:
        print(
            f"  {arm:<8} "
            + " ".join(f"b={b}: {s[arm][f'r_oracle@{b}'][0]:.3f}" for b in BANDS)
        )


def diff_of(s, arm, i):
    r"""Change in clean-probe retention at band ``i``, against the clean arm."""
    key = f"r@{BANDS[i]}"
    return s[arm][key][0] - s[BASELINE][key][0], s[arm][key][1] + s[BASELINE][key][1]


def print_diff(s):
    print_matrix(
        "change from the clean arm (negative = that band is damped)",
        lambda arm, i: (
            "0.0000"
            if arm == BASELINE
            else f"{diff_of(s, arm, i)[0]:+.4f}±{diff_of(s, arm, i)[1]:.4f}"
        ),
    )
    print(
        "  a diagonal means the perturbed band is the one that moves; a column means\n"
        "  one band moves whatever was perturbed"
    )
    print("\nown-band drop against the mean drop of the other two")
    head = (
        f"  {'arm':<8} {'perturbed':>10} {'own':>16} {'other two':>16} "
        f"{'own - other':>12}  {'beyond spread':>13}"
    )
    print(head)
    for _, group in GROUPS:
        for arm in group:
            idx = ARMS[arm][0]
            own, own_sd = diff_of(s, arm, idx)
            others = [i for i in range(len(BANDS)) if i != idx]
            off = sum(diff_of(s, arm, i)[0] for i in others) / len(others)
            off_sd = sum(diff_of(s, arm, i)[1] for i in others) / len(others)
            verdict = "yes" if abs(off - own) > own_sd + off_sd else "no"
            print(
                f"  {arm:<8} {'b=' + str(BANDS[idx]):>10} {own:>+8.4f}±{own_sd:<7.4f} "
                f"{off:>+8.4f}±{off_sd:<7.4f} {off - own:>+11.4f}  {verdict:>13}"
            )
    print(
        "  'own - other' is how much more the perturbed band moved than the rest; "
        "'beyond spread'\n  uses the summed seed spreads of own and other as the bar"
    )


def matrix(s, group):
    r"""``m[j][k]``: change at band ``k`` when band ``j`` was perturbed."""
    return [[diff_of(s, arm, k)[0] for k in range(len(BANDS))] for arm in group]


def decompose(m):
    r"""Split a 3x3 change matrix into row, column and residual parts.

    ``m[j][k]`` is the change at band ``k`` when band ``j`` was perturbed.
    ``grand + a[j] + c[k]`` is the additive row+column fit, so ``a`` is how much
    that perturbation moved *something* and ``c`` is how much that band moves on
    average. A pure **column** pattern (every row identical) has ``a = 0`` and
    every residual 0. A pure **diagonal** fits with much of the effect *not*
    in ``a``+``c``, leaving residuals negative on the diagonal.
    """
    n = len(m)
    grand = sum(sum(row) for row in m) / (n * n)
    a = [sum(row) / n - grand for row in m]
    c = [sum(m[j][k] for j in range(n)) / n - grand for k in range(n)]
    e = [[m[j][k] - grand - a[j] - c[k] for k in range(n)] for j in range(n)]
    return grand, a, c, e


def print_decomposition(s):
    for label, group in GROUPS:
        m = matrix(s, group)
        print(f"\n{label}: change in r by (perturbed band -> readout band)")
        head = f"  {'perturbed':<10} " + " ".join(
            f"{'read@' + str(b):>12}" for b in BANDS
        )
        print(head)
        for j, arm in enumerate(group):
            cells = " ".join(f"{m[j][k]:>+12.4f}" for k in range(len(BANDS)))
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
        diag = sum(e[j][j] for j in range(len(BANDS))) / len(BANDS)
        off = sum(
            e[j][k] for j in range(len(BANDS)) for k in range(len(BANDS)) if j != k
        )
        off /= len(BANDS) * (len(BANDS) - 1)
        print(
            "  residual e_jk after removing a_j and c_k    "
            + " ".join(f"{e[0][k]:>+8.4f}" for k in range(len(BANDS)))
            + "   (row 1 of 3)"
        )
        for j in range(1, len(BANDS)):
            print(" " * 45 + " ".join(f"{e[j][k]:>+8.4f}" for k in range(len(BANDS))))
        print(
            f"  residual on the diagonal {diag:+.4f} vs off-diagonal {off:+.4f} "
            f"(difference {off - diag:+.4f})"
        )
        print(
            "  a pure column would leave a_j = 0 and every residual 0; a pure "
            "diagonal\n  leaves the diagonal residual negative and the off-diagonal "
            "near 0"
        )


def main():
    t0 = time.time()
    print("== scratch_f_invariance: P2, does the deficit follow the band or the f? ==")
    print(
        f"ctx {CTX} / patch {PATCH} / hidden {HIDDEN} / {LAYERS}L / {HEADS} heads "
        f"| SGD lr {LR} / batch {BATCH} / {STEPS} steps / cpu"
    )
    print(
        f"3 tones f = {FREQS} (b = {BANDS}), equal power, total RMS sqrt(0.5) in "
        f"every arm\nincoherent bands {NOISE_BANDS} cycles/window, one arm "
        f"perturbs exactly one of them"
    )
    print("arms: " + ", ".join(f"{a}={spec_of(a)}" for a in ARMS))
    probe = make_mixture(FREQS, N_PER, CTX, PATCH, PROBE_SEED, normalize=True)
    probe_rms = float(probe.square().mean().sqrt())
    print_probe(probe_rms)

    rms = rms_table(probe_rms)
    print_rms(rms, probe_rms)

    ratios = {}
    print("\ncomputing the realised in-bin power ratios (no training yet)")
    for arm in ARMS:
        per_seed = [bin_ratio(arm, s) for s in SEEDS]
        ratios[arm] = {
            k: (
                mean_std([p[k] for p in per_seed])[0]
                if per_seed[0][k] is not None
                else None
            )
            for k in per_seed[0]
        }
    print_bin_ratio(ratios)

    arms, walls = {}, []
    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            t = time.time()
            model, held, held_noise, loss = train(seed, arm)
            row = evaluate(model, held, held_noise, probe, loss)
            row["seed"] = seed
            row["wall"] = time.time() - t
            walls.append(row["wall"])
            rows.append(row)
            print(
                f"  {arm:<8} seed {seed}  loss {loss:.4f}  "
                f"var expl {row['var_explained']:+.3f}  "
                f"r = "
                + " ".join(f"{row[f'r@{b}']:.4f}" for b in BANDS)
                + f"  [{row['wall']:.0f}s]",
                flush=True,
            )
        arms[arm] = rows

    s = {arm: summarize(rows) for arm, rows in arms.items()}
    print_per_seed(arms)
    print_raw(s)
    print_diff(s)
    print_decomposition(s)

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
            "noise_bands": NOISE_BANDS,
            "arms": {
                a: {"idx": i, "band": None if i is None else BAND_NAMES[i], "theta": th}
                for a, (i, th) in ARMS.items()
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
        "corpus_rms": {a: rms[a] for a in ARMS},
        "bin_ratio": ratios,
        "arms": {a: {"per_seed": rows, "mean_std": s[a]} for a, rows in arms.items()},
        "contrast": {
            "baseline": BASELINE,
            "matrix": {
                label: {
                    BAND_NAMES[j]: {
                        str(BANDS[k]): matrix(s, group)[j][k] for k in range(3)
                    }
                    for j in range(3)
                }
                for label, group in GROUPS
            },
            "decomposition": {
                label: dict(
                    zip(
                        ("grand", "row", "col", "residual"), decompose(matrix(s, group))
                    )
                )
                for label, group in GROUPS
            },
        },
        "wall_train_mean": sum(walls) / len(walls),
        "wall": time.time() - t0,
    }
    out = RUNS / "scratch_f_invariance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"\nwrote {out}\ntotal wall {time.time() - t0:.1f}s "
        f"({sum(walls) / len(walls):.0f}s per training)"
    )


if __name__ == "__main__":
    main()
