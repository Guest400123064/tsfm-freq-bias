# Run log

Append-only record of what was run, what came out, and what it means.
Design contract: [`PLAN.md`](../PLAN.md). Per-experiment intent: `experiments/<id>/README.md`.

---

## 2026-09-13 — `0_init`: harness, smoke, first reproduction

Intent: [`0_init/README.md`](0_init/README.md).

### Built

| File | Role |
| :-- | :-- |
| `src/fbias/data.py` | `make_mixture` (sum of tones, per-window amp/phase, `amp_jitter`/`phase_jitter`), `b_of`, and the `PLAN.md` §3.4 `Δb ≥ 1` guard |
| `src/fbias/probes.py` | `matched_amp`, `retention`, `phase_error`, `band_component`/`band_r2`, `fit_oracle` (ridge on frozen `z`), `ring_dim12`, `plane_overlap` |
| `src/fbias/cli/train.py` | train CLI: shifted MSE on `ctx + k` windows, SGD, checkpoint + sidecar JSON |
| `experiments/0_init/scripts/smoke.py` | driver |
| `tests/test_probes.py` | closed-form matched filter, retention, `Δb` guard |

Also fixed a pre-existing `NameError` in `src/fbias/cli/__init__.py`: the module-level `registry`
annotation referenced `_SubParsersAction`, which is only imported under `TYPE_CHECKING`, so
*every* CLI invocation raised before dispatch. One `from __future__ import annotations` fixes it.

### Smoke — PASS

`ctx 512 / k 32 / hidden 32 / 2L / 4 heads / SGD 1e-2 / batch 64 / 2000 steps`, two tones
`b = 0.125, 8` (`f = 2, 128`), deterministic. Wall **47.3 s** on CPU, loss `1.3006 → 0.003558`.
Checkpoint + sidecar in `0_init/runs/`.

```
     f       b  dphi/pi        r_model   r2_model  |phase err|      r_oracle  r2_oracle
   2.0   0.125    0.250      1.0193±0.0574   0.9987      0.0137     1.0003     1.0000
 128.0   8.000    0.000      1.0026±0.0305   0.9988      0.0129     1.0001     1.0000
```

### Reproduction check — FAILED, and the specification was wrong

The README's recorded check (`r_{b=8} < r_{b=0.125}`) did not hold: 1.0026 vs 1.0193, a 1.7%
difference well inside the ±0.03–0.06 window-to-window spread. Neither band shrank, and the
oracle had no gap to recover (1.0000 vs the model's 0.9988).

This is **my spec bug, not a harness failure or evidence against C1**. The corpus was specified
as deterministic, and `PLAN.md` **C4** — which I put there from lab It.29-30 — says shrinkage
vanishes at convergence on deterministic corpora. The spec asked a deterministic on-support
mixture to demonstrate damping, which C4 forbids. The two sections of the plan contradicted
each other.

The metric is not the problem. An untrained model at the same position reads retention
0.269 / 0.129 and r² +0.034 / −0.123, so `r_f` discriminates.

### Diagnostic D1 — no damping on a deterministic on-support corpus

8 tones `b = 1..8`, 2000 steps, same config, loss `0.01637`:

```
   b        r       r2          (all on-support)
 1.000   0.9870   +0.9943
 2.000   0.9851   +0.9942
 3.000   0.9840   +0.9958
 4.000   0.9924   +0.9953
 5.000   0.9858   +0.9948
 6.000   0.9780   +0.9950
 7.000   0.9923   +0.9958
 8.000   0.9883   +0.9955
```

Confirms the C4 reading: flat across `b`, ~0.98–0.99, no band structure. Consistent with C4.

### Diagnostic D2 — damping on an off-support frequency

Train on 8 tones `b = 1..8`; probe on 8 tones `b = 0.5, 2..8` (same tone count, same
amplitudes, `b=1` swapped for the off-support `b=0.5`), loss `0.01637`:

```
     f       b  support        r        r2
     8   0.500      OFF   0.6593   -0.7780
    32   2.000       on   1.0702   +0.8323
    48   3.000       on   1.0210   +0.8587
    64   4.000       on   1.0510   +0.9096
    80   5.000       on   1.0017   +0.9808
    96   6.000       on   1.0173   +0.9363
   112   7.000       on   1.0132   +0.9292
   128   8.000       on   1.0176   +0.9374
```

The off-support band is damped to **0.66×** with band r² **−0.78** (worse than predicting the
mean), while every on-support band sits at ~1.0 with r² 0.83–0.98. **The harness detects
frequency-selective damping.** This is a miniature C5.

**Two confounds to note, neither resolved here:**

1. **Amplitude/scale mismatch.** A first version of D2 probed 3 tones against an 8-tone training
   corpus and read retention ~1.7 across the *on-support* bands. `make_mixture` gives every tone
   amplitude 1, so total RMS scales as `sqrt(n_tones)` (√(8/3) ≈ 1.63, matching the 1.7 observed).
   The probe corpus must match the training corpus's tone count and amplitude distribution. This
   is a concrete instance of the `PLAN.md` §3.3 marginal-control requirement, and the design must
   state it explicitly.
2. **`b = 0.5` is simultaneously off-support *and* the `Δφ = π` case.** Lab It.29-30 identified
   `Δφ = π` as the transient laggard, so this single run cannot separate coverage from the `Δφ`
   ordering. Separating them is `3_deterministic_dphi`'s job (vary `Δφ` at fixed coverage), and
   `6_coverage`'s job (vary coverage at fixed `Δφ`).

### Verdict — corrected after reading It.27 / It.29 directly

**The attribution above is superseded.** It.27's damping was measured on `monash-direct-h32.pt` and
`geom-rich-input.pt`, **neither of which was trained on the 2-tone mixture it was probed on**;
`geom-rich-input` is deterministic within a window and still erased the band at r² = −0.46. It.29's
null came from a model trained and probed on the *same* pure-tone corpus (0.988–0.998).

So the discriminator is the **training marginal**, not determinism. `0_init`'s corpus is
deterministic, structurally simple, and *is* the probe — it sits in It.29's regime, which is why it
showed no damping. The spec bug is that it combined It.27's probe signal with It.29's training
condition; that combination appears nowhere in the lab.

### Diagnostic D3 — the marginal ladder (`diag_marginal.py`)

Probe identical in every arm (`make_mixture([2.0, 128.0])`, seed 123, `b = 0.125, 8.0`); only the
training marginal varies. 3 seeds each, 2000 steps, `ctx 512 / k 32 / hidden 32`.

```
arm                   r@0.125            r@8.0        r2@8.0   probe mse  final loss
A  mixture[2,128]   1.0258±0.0102    1.0009±0.0018    +0.9982    0.0038    0.004181
B  broad32          0.0339±0.0158    0.0407±0.0167    −0.0230    1.0468    0.499700
C  mixture[16..128] 0.1878±0.0414    2.1750±0.0859    −0.5047    2.0658    0.016213
D  broad32 high     0.0502±0.0173    0.1523±0.0559    −0.0056    1.0570    0.496606
   untrained ctrl   0.2489±0.0650    0.1835±0.0553    −0.1142    1.4081          —
```

**Arms B and D never fit.** B's final loss 0.4997 against a target variance of 0.5021 is **+0.5%**
variance explained — the model outputs the window mean. Its `r ≈ 0.04` at both bands means "no
signal anywhere", not "high band erased". First appearance of contract §3.8.

Arm C's `r@8 = 2.175` is the amplitude-scale artifact: its 8-tone corpus has RMS 2.0 against the
probe's 1.0, and a perfectly-tracking model reads ≈2.0 (its `band r2 = −0.50` confirms the reading
is not meaningful retention). Second appearance of the trap now recorded as §3.9.

### Diagnostic D4 — is the broad arm merely under-trained? (`diag_fit.py`)

One seed per arm, 20k steps with checkpoints, probe RMS-matched to every training corpus:

```
arm                        step   var expl (held)   r@0.125   r@8.0   r2@0.125   r2@8.0   r8/r1
N   mixture[2,128]         2000        +0.993        1.0268   0.9987    0.9967    0.9984   0.973
N                          20000       +0.999        1.0019   1.0000    0.9999    0.9999   0.998
BR  broad32, freqs redrawn 2000        −0.011        0.0164   0.0266    0.0044    0.0209   1.621
BR                         20000       −0.005        0.0213   0.0441   −0.0109    0.0614   2.073
BF  broad32, freqs fixed   2000        +0.178        0.5005   0.3084   −0.5096    0.2087   0.616
BF                         10000       +0.529        0.9385   0.5692   −1.0703    0.2826   0.606
BF                         20000       +0.691        1.1378   0.8022   −1.1103    0.4125   0.705
```

1. **BR is not under-trained; it is unlearnable at this scale.** Variance explained runs −0.011 →
   −0.005 across 2000→20000 with held-out MSE ≈ held-out variance throughout: it converged to the
   mean predictor, the optimum for an unpredictable target. The D3-B reading is confirmed worthless.
2. **BF fits** (+0.691 at 20k, still rising: 0.178 → 0.388 → 0.529 → 0.691) and shows
   `r8/r1 = 0.705`, ~40× the seed spread. **But `band_r2` does not support a clean high-band-damping
   story**: `r@0.125 = 1.14` with `r2@0.125 = −1.11` means the low band is *over*-predicted and
   phase-wrong, so the ratio is as much low-band overshoot as high-band suppression. BF had also
   not converged.
3. The BR/BF gap localises the D3-B failure: it was **per-window frequency identification**
   (32 unknown frequencies redrawn every window), not marginal breadth. Contract §3.10.

### Open

- `1_pred_side` still has no corpus it can run on. BF is the closest candidate, but it needs
  training to convergence and the low-band overshoot explained before `r8/r1` carries the meaning
  the lab contrast implies.
- The lab's rich corpus draws frequencies from **12 named clusters** — a low-dimensional family —
  which is plausibly why the lab's model fit its marginal while a uniform redraw could not.
  Reproducing that structure is the untried option (§3.10).
- `PLAN.md` §4's transient metric (`r_f(t)`) is still not implemented beyond four discrete
  checkpoints.
- The `b = 0.5` / `Δφ = π` confound from D2 remains unseparated.

---

## 2026-09-14 — P0: band-selective noise, fixed clean probe (scratch)

Script: `0_init/scripts/scratch_bandnoise.py` → `runs/scratch_bandnoise.json`. Intent: does the
training distribution's *band placement* of noise move clean-probe retention at a fixed frequency?

Config: `ctx 512 / k 32 / hidden 32 / 2L / 4 heads / SGD 1e-2 / batch 64 / 2000 steps`, 3 seeds.
Training signal `make_mixture([2.0, 128.0], normalize=True)` — identical in every arm, paired across
arms. Probe **clean** 2-tone, seed 123, identical everywhere. Bands (FFT over `ctx+k=544`): low
`[1.88, 32.0]`, high `[96.0, 160.0]`, `noise_all` = all bins. Same white realisation projected into
each band, so only spectral placement differs; noise RMS 0.4952 = 0.7 × signal RMS, so all noise arms
carry identical total power.

```
arm         var expl      r@0.125            r@8.0        r8/r1    r2@0.125
clean     +0.991±0.002   1.0487±0.0192   1.0005±0.0028   0.954     0.9943
noise_hi  +0.653±0.006   1.0301±0.0182   1.0428±0.0004   1.013     0.9900
noise_lo  +0.612±0.009   1.1303±0.0430   1.0410±0.0074   0.922     0.9240
noise_all +0.592±0.001   1.1240±0.0217   1.0250±0.0056   0.912     0.9382
```

Every arm fit (noisy arms 0.59–0.65 vs the `1 − var(noise patch)/var(target)` bound of 0.671; clean
0.991), so the §3.8 gate passes.

### Result: null, and **all** effects have the wrong sign

`noise_hi` does not depress `r@8` — it raises it, 1.0005 → 1.0428 (**+0.042, ~13× the combined seed
spread**). `noise_lo` raises `r@0.125`, 1.0487 → 1.1303 (+0.082). Nothing shrank anywhere. Every
non-zero effect is **gain inflation**.

### Why the manipulation could not have worked

This is the important part, and it invalidates the θ definition that was written into `PLAN.md` §4
earlier the same day.

The noise is band-limited but the **tone persists** across the window. The model integrates over ~16
context patches, and the injected noise is independent across patches, so it averages down by that
factor: the effective in-band SNR available to the model is ≈ `0.19/17 ≈ 0.011`. Under that marginal
only ~1% of shrinkage is justified, so `r ≈ 1` is the **correct** answer. The null therefore does not
discriminate "no learned prior" from "a prior correctly calibrated to a marginal where the tone
dominates its own bin". **Additive noise is the wrong instrument for building a band prior**, because
it makes the target's *content* unpredictable while leaving the *tone* perfectly recoverable from a
long context.

The contrast with D2 is instructive: coverage (off-support, `r = 0.66`) *did* produce a deficit. That
is the same phenomenon — the model has no coherent basis for the band — and it is the θ→0 endpoint.
So the right θ is **the fraction of a band's power that is coherent and persistent**, not noise level.

### Two measurement artifacts found

1. **Gain inflation tracks the training target's band power, not unpredictability.** Observed
   inflation sits near `sqrt(1 + injected band power ratio)`: hi@8 ×1.042 vs ×1.090 predicted,
   lo@0.125 ×1.078 vs ×1.085, all@8 ×1.025 vs ×1.029. It is **opposite in sign to shrinkage**, so it
   masks a prior. (Hypothesis, not a finding — `noise_all`@0.125 (×1.072 vs ×1.016) is the exception.)
2. **The clean control is not a flat baseline.** With zero noise it already reads `r@0.125 = 1.049`
   and `r8/r1 = 0.954`, so low-band overshoot is this config's default and any low-band claim must
   subtract it.

Minor: the §3.5 oracle (`fit_oracle` on the probe's own latents) reads 1.000 in every arm — an
in-sample ridge on a stationary probe is degenerate here and discriminates nothing. Present in the
JSON, but not evidence.

### Verdict

P0 is a **null, and not a clean test**. It does not falsify the §0 claim — it shows the first attempt
at θ was miscalibrated. Actions: θ redefined in `PLAN.md` §4 as coherent-energy fraction; P1's design
must train with the band's **tone removed** (energy replaced by incoherent content at fixed band
power), not with noise added on top of a persisting tone. The gain-inflation artifact must be
reported and subtracted, and the clean-arm baseline quoted whenever a low-band claim is made.

Wall 559 s for 12 trainings (~47 s each). Reproduce:
`.venv/bin/python experiments/0_init/scripts/scratch_bandnoise.py`.

Left undone: `0_init/README.md`'s script table does not list this script.

---

## 2026-09-15 — P1 (revamped P0): coherent-fraction sweep

Script: `0_init/scripts/scratch_coherent.py` → `runs/scratch_coherent.json`. 15 trainings, 680 s.

Design: fixed **clean** 2-tone probe (`b = 0.125, 8.0`), identical in every arm. The training
marginal holds total band power fixed and splits the high band `[96, 160]` between a **coherent**
tone at `f=128` (fraction θ) and **incoherent** band-limited energy (fraction 1−θ). The low band is
always a clean coherent tone at `f=2` — the internal control. `ctx 512 / k 32 / hidden 32 / 2L /
4 heads / SGD 1e-2 / 2000 steps`, 3 seeds.

```
arm           var expl (bound)   r@0.125           r@8.0        r2@8.0    in-bin ratio
θ=1.0 clean  +0.991±0.002 (1.000) 1.0490±0.0170  0.9995±0.0039  +0.9975      n/a
θ=0.5        +0.723±0.006 (0.750) 0.9315±0.0462  0.8195±0.0308  +0.9644      4.24
θ=0.2        +0.580±0.008 (0.600) 0.9211±0.0324  0.7035±0.0105  +0.9077      1.06
θ=0.0        +0.507±0.008 (0.500) 1.1017±0.0247  0.0362±0.0047  +0.0082      0.002
off_support  +0.991±0.001 (1.000) 1.2164±0.2141  0.1891±0.0592  −0.1642       n/a
```

Fit gate passes in all five arms (96–101% of the analytic bound `1 − var(unpredictable)/var(target)`),
so every `r` is quotable. RMS is constant across arms to **0.14%** (0.7064–0.7070 vs probe 0.7074):
contract 9 is satisfied, and θ=1.0 is distributionally identical to P0's `clean` arm
(`r@0.125 1.0490` vs `1.0487`, `r@8 0.9995` vs `1.0005`).

### A graded dose–response exists, and it falsifies the "just coverage" reading

Over θ ∈ [1.0, 0.2] the clean-probe deficit is **roughly linear in θ**: `r@8 ≈ 0.66 + 0.34θ`
(0.9995 → 0.8195 → 0.7035). Every step is beyond the combined seed spread. So the hypothesis "any
coherent tone present is fully extracted, and the effect is really only about coverage" is dead:
**θ=0.5 loses 18% of the amplitude on a probe that is completely clean.**

### And a cliff at θ=0 on top of that curve

Extrapolating the linear part to θ=0 predicts ≈0.66; observed **0.036** (band r² = 0.008, i.e. the
band is emitted at noise level). A single θ step of 0.2 costs 2.3× what the entire rest of the curve
costs.

### The two zero-coherence endpoints are **not** the same — PLAN §4 was wrong

**Empty band** (off_support, tone moved to `f=64`) reads **0.189±0.059**; **incoherent-filled band**
(θ=0) reads **0.036±0.005**. The incoherent-filled band is damped ~5× harder than the absent one, and
neither lies on the graded curve.

So PLAN §4's claim that "coverage is the θ→0 endpoint of the same axis" is **contradicted**. The
plausible mechanism: a band carrying energetic but unpredictable content is actively *punished* by the
loss — emitting ~0 there avoids large errors — whereas an absent band costs nothing either way. That
is **loss-driven suppression, not merely absence of a basis to predict.**

(off_support is also the least trustworthy row: its low-band reading spans 1.06–1.46 with
`r2@0.125` 0.55–0.96, so its `r8/r1` should not be leaned on. Its high-band deficit is real.)

### Cross-band coupling, and P0's gain hypothesis dies

The low band moves **−0.12** (θ=0.5, 0.2) and **+0.05…+0.17** (θ=0, off_support) despite essentially
unchanged coherent power there. P0's `sqrt(1 + injected band power ratio)` hypothesis cannot explain
either direction, because band power is constant across arms **by construction**. Any low-band claim
must quote the θ=1.0 baseline (1.0490).

### The main interpretive limit

At θ=0.5 the model's deficit equals **single-patch** Wiener shrinkage (observed 0.82 vs 0.81); at
θ=0.2 it reads 0.70 against single-patch 0.52 and context-integrated ≈0.94. So the model sits
*between* a single-patch and a 16-patch estimator — it is **not integrating the context coherently**
across patches. The deficit is therefore either a prior calibrated at the patch scale, or a limit on
coherent context averaging. **This run cannot separate them**, and the capacity/context-length
control is now the binding follow-up.

What it *does* establish cleanly is the causal part of the §0 claim: fixing the probe, the
architecture and the capacity, and changing **only** the training marginal's θ, moves the clean-probe
high-band retention monotonically from 1.00 to 0.03.

### Also note

The oracle column is weakly informative: `fit_oracle`'s ridge target is the *noisy observed* patch,
so it inherits the same marginal-driven attenuation and does not cleanly separate representation
from head. It is not independent evidence.

### Verdict

First positive signal. P1's causal part is supported; **P2 (`f`-invariance) is still untested**; and
the prior-vs-capacity question now needs a context-length or capacity control before the claim can be
worded as "Bayesian prior" rather than "learned attenuation".

---

## 2026-09-15 — P1b Step 1: context-length ablation — **P1b resolved**

Script `0_init/scripts/scratch_context_ablation.py`; JSON `runs/scratch_context_ablation.json`; 6
checkpoints `runs/ctx_ablation_theta_{0.5,1.0}_seed{0,1,2}.pt`. `scratch_coherent.py` saved no
checkpoints, so both arms were retrained with its `train()` verbatim — and the retrain is
**bit-identical to P1** (`train_loss`, `var_explained`, `r@8@N=16` match `scratch_coherent.json` to the
last float), which validates the truncation machinery.

Ablation: feed the last `(N+1)·k` timesteps, read `pred[:, -2]` vs `pat(x)[:, -1]`; left-truncation
only, so RoPE's relative positions transfer. Two corpora, deliberately: the **held-out θ=0.5 training
marginal** (where history should help) and the **fixed clean probe** (where it should not be needed).

**Next-patch MSE on the θ=0.5 marginal** (held out; target variance 0.50051, unpredictable floor 0.125)

```
arm              N=1              N=2              N=4              N=8              N=16
θ=0.5      0.16110±0.00678  0.14411±0.00135  0.13729±0.00208  0.13479±0.00123  0.13452±0.00141
θ=1.0 ctrl 0.28379±0.03057  0.25175±0.01996  0.23668±0.00911  0.22863±0.00732  0.22105±0.00605
oracle     0.15702          0.13951          0.13084          0.12739          0.12583
```

**Clean-probe `b = 8.0`**

```
N                                    1                2                4                8                16
θ=0.5 r@8                 0.80294±0.03494  0.79873±0.03783  0.80291±0.03704  0.81370±0.02966  0.81946±0.03078
θ=1.0 r@8                 0.97987±0.00635  0.97966±0.00996  0.99068±0.00429  0.99207±0.00514  0.99955±0.00389
Wiener calibrated to N    0.8093           0.8946           0.9444           0.9714           0.9855
evidence-adaptive         1.000            1.000            1.000            1.000            1.000
```

### The answer: it uses its history, but the attenuation is context-independent

- **Marginal error improves with `N`: yes, −16.5%**, monotone, and every step up to `N=8` is outside
  the seed spread (it saturates at `N≈8`). Against the least-squares oracle, which falls
  `0.15702 → 0.12583` (−19.9%), the model captures **85% of the available gain**. So it *does*
  coherently average across patches.
- **Clean-probe `r@8` rises with `N`: effectively no.** `+0.0165` total, inside the seed spread, dead
  flat over `N=1..4`. The Wiener factor calibrated to `N` moves `+0.1762` over the same range; the
  model captures **9.4%** of it. An estimator that adapted to the evidence it holds would read
  **1.000 at every `N`** — 16 clean patches determine the tone completely.

**This is row 1 of `PLAN.md` §5's P1b table: a learned prior.** Three independent reasons it is not a
capacity story: (i) it extracts 85% of the coherent-averaging gain available on the marginal, so
nothing stops it averaging; (ii) the *same architecture* trained on a clean marginal reads
`r@8 = 1.000`, so nothing architectural caps `b=8` at 0.82; (iii) the attenuation does not adapt to
evidence — with 16 clean patches it still emits 0.819 where correct is 1.000.

Combined with P1's datum that 0.82 ≈ the single-patch Wiener factor 0.8093, the sharpened wording is:
**the prior is patch-local, not context-global** — consistent with the lab's C3 (position-invariance),
since a model using its context would shrink *less* the further into the window it predicts.

### Caveats and things that contradicted the reasoning

- **The clean-probe drift is not θ-specific.** The `θ=1.0` control moves as much (`+0.0197` vs
  `+0.0165`) over the same range, and both arms' *low* band also gains with `N` (θ=0.5 `r@0.125`
  0.808→0.931, saturating by `N=4`). So `r@8` is flat in **both** arms; it should not be read as a
  θ-specific rising trend.
- **The marginal gain is not exclusively the incoherent band's.** Both bands' in-band errors fall with
  `N`. A band-restricted oracle puts 72% of the available gain in the high band, but the
  `0.5|ΔZ|²` split is approximate (matched-filter leakage at `b=0.125`) and its terms do not sum to the
  total MSE — treat it as a qualitative split, not an exact decomposition.
- **`N<16` truncation is technically out of distribution** (the model never saw a window shorter than
  17 patches). It cannot be inflating much: the oracle, a pure information effect with no shift,
  gains *more* (19.9%) than the model (16.5%).
- **Design premise confirmed.** The oracle is exactly 0 at every `N` on the clean probe and on the
  θ=1.0 corpus, and only falls on the θ=0.5 marginal. **Measuring on the clean probe alone would have
  been a false negative** — the split-corpus design was load-bearing.

### Consequences

- **P1b Step 2 (training-time context-length and `hidden` sweeps) is not needed** — the question it
  was written to answer is answered.
- Per the ladder's own trigger (§6), S1/S2 scaling is **deprioritized** from a validity check to a
  **generality check**: representation capacity is ruled out, and what remains is whether the learned
  prior survives a bigger model, which matters for the real-TSFM claim, not for the validity of the
  synthetic one.
- **P2 (`f`-invariance) is next.** Contract 8's fit gate and contamination checks hold: θ=0.5
  `+0.7233±0.0056`, θ=1.0 `+0.9912±0.0019`, identical to P1.

Wall 272 s (6 trainings). Reproduce: `.venv/bin/python experiments/0_init/scripts/scratch_context_ablation.py`.

---

## 2026-09-15 — P2: is the prior about the perturbation or the frequency? — **DIAGONAL**

Script `0_init/scripts/scratch_f_invariance.py`; JSON `runs/scratch_f_invariance.json`. 21 trainings,
936 s. `data.py` gained `make_banded(freqs, bands, thetas, ...)` — a sibling of `make_coherent` taking
per-tone `(band, θ)`.

Design: three tones `b ∈ {0.25, 2.0, 8.0}` (`f = 4, 32, 128`; `Δφ/π = 0.5, 0, 0`), each a third of the
signal power, total RMS `√0.5`. Incoherent bands `[2,8]` / `[24,40]` / `[96,160]` cycles/window. One
band perturbed per arm; clean 3-tone probe, identical everywhere. 7 arms × 3 seeds.

**Clean-differenced `r`** (each cell minus the `clean` arm's same-band reading; the subtraction is
mandatory because of the coupling below):

```
              lo_50          mid_50         hi_50          lo_00          mid_00         hi_00
b=0.25   -0.2426±0.0205  -0.0320±0.0393  -0.0924±0.0356  -0.3521±0.0356  +0.0331±0.0411  -0.0217±0.0476
b=2.0    +0.0053±0.0163  -0.1298±0.0158  -0.0737±0.0117  +0.0123±0.0123  -0.6894±0.0379  +0.0026±0.0096
b=8.0    +0.0039±0.0080  -0.0305±0.0093  -0.1193±0.0091  +0.0077±0.0082  +0.0280±0.0136  -0.9450±0.0099
```

| arm | own band | other two | own − other |
| :-- | --: | --: | --: |
| `lo_50` / `mid_50` / `hi_50` | −0.2426 / −0.1298 / −0.1193 | +0.0046 / −0.0313 / −0.0831 | **+0.2471 / +0.0985 / +0.0362** |
| `lo_00` / `mid_00` / `hi_00` | −0.3521 / −0.6894 / −0.9450 | +0.0100 / +0.0305 / −0.0095 | **+0.3622 / +0.7199 / +0.9354** |

All six beyond the summed seed spread.

### Answer: the effect follows the perturbed band. "The model damps high frequencies" is dead.

The high band is untouched by low/mid perturbations: `lo_50` → `r@8` **+0.0039**, `lo_00` → +0.0077,
`mid_50` → −0.0305, `mid_00` → +0.0026. A row+column decomposition of the 3×3 change matrix leaves a
non-zero residual only on the diagonal (θ=0.5: diag −0.0849 vs off-diag +0.0424; θ=0.0: −0.4483 vs
+0.2242). A pure column would leave **every** residual at 0.

### Two residuals, both stated plainly

1. **Coupling is real at θ=0.5 and one-directional** (`high → everything else`): `hi_50` moves
   `r@0.25` by **−0.0924±0.0356** (P1's −0.12 reproduced) and `r@2.0` by −0.0737±0.0117. `lo_50` moves
   nothing else (+0.0053, +0.0039). It is ~48% of the diagonal magnitude at θ=0.5 and **collapses to
   ≤0.0331 (5%) at θ=0.0**. So the "in between" in the decision table is entirely this shared term.
2. **A frequency dependence in the *size*, not the location.** At θ=0.0 the realized in-bin ratios are
   all ≈0 (0.023 / 0.035 / 0.008), yet own-band drops are −0.35 / −0.69 / −0.95 — monotone in `f`. So
   *which* band moves is not frequency-determined, but the magnitude is not frequency-free. **Confound:
   the three bands are not equally manipulable** (next section), so this ordering is not cleanly
   attributable to `f`.

### The instrument flaw: θ's "incoherent" energy is **not unpredictable**

The fit gate says four of six arms **beat** the analytic floor `1 − incoherent power / target variance`:

```
lo_00  +0.8671 vs bound +0.6668   → +0.2003  = 60% of the incoherent power explained
mid_00 +0.7864 vs bound +0.6670   → +0.1194  = 36%
lo_50  +0.8917 vs bound +0.8339   → +0.0578  = 35%
mid_50 +0.8561 vs bound +0.8336   → +0.0225  = 14%
hi_50  below by −0.0276 ; hi_00 below by −0.0116
```

**Mechanism.** `band_noise` builds the incoherent content from the *window's own* FFT bins, so it is a
fixed band-limited realization (≤69 bins), window-periodic and low-DOF, and the 512-sample context
determines its coefficients. A least-squares fit of each arm's generative basis (3 tones + that band's
FFT bins) to the 512 context samples reproduces the target patch at **context residual ≈ 1e-11 and
target MSE 0.00000000** for `lo_50`, `lo_00`, `mid_00`, `hi_00` (4/6 verified; `mid_50`/`hi_50` hit a
solver rank issue, not a data problem).

**Consequence — this is contract 11's distinction, and θ falls on the wrong side of it.** The
"incoherent" energy is deterministic, so in principle it is perfectly predictable from the window; it
is only unpredictable *relative to what the model can fit*. So θ manipulates **coherence/persistence
and fittability**, not irreducible unpredictability. And because the bands differ in DOF, θ is
**not equally strong across bands**: the hi arms are the strongest effective manipulation, the lo arms
the weakest — which is exactly what residual 2 above is confounded with.

**This applies to P0 and P1, not just P2.** P1's θ curve is therefore a curve in "the band is not a
persistent tone", not in "the band is unpredictable". That is why the fit gate mattered and why
`hi_00`'s near-floor reading (−0.0116) is the trustworthy end.

### Verdict and what it changes

- **§0's causal claim is supported**: holding the probe fixed and the architecture fixed, perturbing
  one band's marginal statistics moves *that band's* clean-probe retention, monotonically, with the
  other bands essentially untouched.
- **§0's invariance claim is mostly supported**: which band moves is not frequency-determined.
- **The word "predictability" is not yet earned.** θ is a coherence/fittability manipulation, so P1+P2
  support "a learned prior about the band's *fittability*", not "about its *predictability*".
- **P2b (phase modulation) is therefore promoted from an orthogonal check to the instrument that
  decides the claim's wording.** Its random-walk increments are genuinely unknown, so it does not share
  this flaw.
- `hi_50` reproducing P1's low-band coupling (−0.0924 vs −0.12) is a useful cross-check that the two
  scripts agree.

RMS constant to 0.10% across arms. Oracle reads ~1.000 in every *untouched* band — independent
confirmation that the readout is band-localized, not a probe artefact (but its target is still the
noisy patch, so it remains non-separable from readout calibration). Determinism verified by two runs.
Reproduce: `.venv/bin/python experiments/0_init/scripts/scratch_f_invariance.py`.

---

## 2026-09-15 — P2b: phase modulation — **Prediction 1: the prior transfers**

Script `0_init/scripts/scratch_phase_mod.py`; JSON `runs/scratch_phase_mod.json`. 21 trainings,
932.7 s. `data.py` gained `make_pm(freqs, betas, n_per, ctx, k, seed)` (purely additive).

Signal: piecewise-constant phase per patch, random walk across patches,
`phi_p = phi_{p-1} + 2*pi*b_c + beta*eps_p`, `eps ~ N(0,1)`. Constant envelope, so **total power is
fixed at every β by construction** — verified: corpus RMS identical across arms to 1e-4 relative, and
per-patch bin power `|Z|²/2` **flat in β** in every arm (PM moves no power between bins, as it
should). `r* = exp(−β²/2)`.

**Clean-probe `r` at `b=8.0` (Part A), identical probe in every arm:**

```
 beta    r*     probe r@8        r − r*     1 − r     reads closer to   marginal r@8
  0.0   1.000  1.0027±0.0031   +0.0027    -0.0027   indistinguishable    1.0001±0.0028
  0.5   0.882  0.8764±0.0127   -0.0061    +0.1236   r* (prior)           0.8501±0.0070
  1.0   0.607  0.6019±0.0139   -0.0047    +0.3981   r* (prior)           0.5700±0.0124
  1.5   0.325  0.3029±0.0129   -0.0217    +0.6971   r* (prior)           0.2727±0.0154
  2.0   0.135  0.1157±0.0143   -0.0196    +0.8843   r* (prior)           0.1040±0.0158
```

Every gap to `1.000` is ≥10× the seed spread; every gap to `r*` is ≤0.022. **Prediction 2
(evidence-adaptive, `r = 1.000` at every β) is dead.**

**Fit gate: all 7 arms sit at 98–99% of the analytic bound and none exceeds it** — the exact opposite
of θ, where 4/6 arms beat the bound. PM's unpredictable part really is unpredictable, so the
instrument is sound.

**Part B (β=1.0 on each band) — diagonal again, and now equal in magnitude:**

```
perturbed   read@0.25   read@2.0   read@8.0        own-band drop
b=0.25        -0.4226    +0.0049    +0.0034        -0.4226
b=2.0         -0.0009    -0.4058    +0.0005        -0.4058
b=8.0         -0.0070    -0.0020    -0.4008        -0.4008
```

All six off-diagonal cells ≤0.007. **This resolves both of P2's residuals:**

1. P2's θ=0.5 coupling (`hi_50` moving `r@0.25` by −0.0924) **does not appear under PM** — unperturbed
   bands move ≤0.007. So that coupling was a **θ-instrument artifact**.
2. P2's frequency-dependent drop *magnitudes* (−0.35 / −0.69 / −0.95) become **equal across bands**
   (−0.423 / −0.406 / −0.401). So that ordering was the θ instrument's unequal per-band efficacy
   (its caveat predicted exactly this), **not** a frequency effect.

**Internal control**: unperturbed bands on the clean probe move ≤0.007, i.e. none.

**Oracle** (contract 5, out-of-sample ridge `z_held → probe`) at `b=8`: 1.000 / 0.880 / 0.605 / 0.332 /
0.159 for clean…β=2.0 — the frozen representation carries a near-Bayes readout, so the deficit is not
a head-only quirk; the model's own head is 0.5–4 points below it.

### Caveats

- **The prior is not *purely* patch-local.** The clean-probe reading sits **above** the same arm's
  marginal reading at every β: +0.026 / +0.032 / +0.030 / +0.012. So the model does recover a small
  evidence-adaptive component (1–22% of the way from `r*` back to 1.000, decreasing with β) — the same
  order as P1b's "9.4% of the available gain", but here roughly a constant **+0.03 absolute** rather
  than a constant fraction.
- **The marginal reads slightly *below* `r*`** at `b=8` (0.850/0.570/0.273/0.104 vs
  0.883/0.607/0.325/0.135), i.e. ~0.03–0.05 of over-shrinkage. The fit gate says that costs almost
  nothing in MSE (98% of the Bayes bound). The clean arm's marginal reads 1.0001.
- **`b=0.25` remains the least trustworthy column**: the matched filter leaks at non-integer `2b` and
  the ratio is phase-sensitive there (`lo_b1.0` reads 0.7268 on its own marginal, above `r*`=0.607, but
  0.5535 on the clean probe, below), with 3–5× the seed spread. The diagonal is unambiguous because
  its off-diagonal cells are ≤0.007, but do not quote the low band's absolute number against `r*`.

### Generator bug found and fixed before the reported run

The first `make_pm` double-counted the inter-patch advance (the sample-time term already carried `2πb`
and the walk added it again), so only `b=0.25` was wrong (advance π instead of π/2 per patch); the
β=0 arm then read clean-probe `r@0.25 = 1.45` against P2's clean arm 0.98, because the low tone sat at
`f=8` instead of `f=4`. After the fix a β=0 corpus puts 100% of its power in bins 4/32/128, and β=0
reproduces P1/P2's clean arm almost exactly (0.9761 / 0.9987 / 1.0027 vs P2's 0.9765 / 0.9991 /
1.0035) — a useful cross-script check. **The reported run is the fixed one.**

### Consequences

- **The claim's wording is now earned: the prior is about *predictability*.** The clean-probe
  retention equals the Bayes-optimal shrinkage for the training marginal, to within 0.03, so the model
  behaves as if it knows the marginal's irreducible unpredictability and does not notice the input
  changed. §4's θ caveat is resolved in favour of "predictability" **for PM**; θ itself stays a
  fittability instrument.
- The prior is **marginal-calibrated with a small evidence-adaptive correction** (~+0.03 absolute).

Reproduce: `.venv/bin/python experiments/0_init/scripts/scratch_phase_mod.py`.

---

## 2026-09-15 — P3: fine `f` sweep, is there a residual at matched predictability?

Script `0_init/scripts/scratch_f_sweep.py`; JSON `runs/scratch_f_sweep.json`. 15 runs, 720 s.

Design: 8 integer bands `b ∈ {1..8}` (`f = 16…128`), `Δb = 1` exactly (contract 4 asserted with zero
slack), `2b` integer so the matched filter is exact, and `Δφ ≡ 0` for every band — so the whole sweep
sits in the "copy" class and any residual cannot be a `Δφ` effect. **Uniform β over all eight bands at
once** (β ∈ {0, 0.5, 1.0, 1.5, 2.0}), clean 8-tone probe, 3 seeds. Transient `r_f(t)` logged at steps
500/1000/1500/2000 — this fills `PLAN.md` §4's long-missing transient metric.

**Clean-probe `r`, mean ± sd over 3 seeds:**

```
       f     clean          beta0.5        beta1.0        beta1.5        beta2.0
 b=1  16   0.9320±0.0074  0.7771±0.0138  0.4565±0.0234  0.1540±0.0215  0.0639±0.0048
 b=4  64   0.9377±0.0166  0.7651±0.0192  0.4417±0.0577  0.1586±0.0240  0.0793±0.0027
 b=8 128   0.9399±0.0103  0.7605±0.0144  0.4594±0.0435  0.1654±0.0164  0.0765±0.0104
 r*         1.000          0.882          0.607          0.325          0.135
```

### Answer 1 — **flat in `f`: yes.** This is the boundary sentence the run existed for.

`spread_b` (max−min over the 8 bands) is **0.019–0.025** in every arm, against a pooled per-band seed
sd of 0.008–0.034. The ratio is 0.73–2.54, i.e. at or *below* what 8 iid readings would spread at that
noise level. Clean-differenced drops are equal across bands: −0.174±0.027 / −0.491±0.030 / −0.782±0.029
/ −0.865±0.030 for β = 0.5 / 1.0 / 1.5 / 2.0, each spread ≈ the differential's own seed sd.

### Answer 2 — **not at `r*`: the head sits uniformly *below* it.**

Mean deviation per arm: −0.117 / −0.159 / −0.168 / −0.061, i.e. `r/r*` = 0.867 / 0.738 / 0.483 /
0.547. Crucially the deviation has **no trend in `b`** (`corr(b, dev)` = −0.01 / +0.11 / −0.07 /
+0.20), so it is a **level** offset, not a frequency-selective one.

**The oracle resolves why.** An out-of-sample ridge readout of the arm's *frozen latents* reads, mean
over bands, 0.993 / 0.871 / 0.599 / 0.324 / 0.149 → **`oracle/r*` = 0.987 / 0.988 / 0.998 / 1.098**,
per-band spread 0.013–0.025, **flat in `f`**. So a closed-form linear readout of the same frozen
features **reproduces `r*` exactly at all 8 bands and all 5 β's**, while the SGD-trained head (also
linear on those features) reads 0.879 / 0.747 / 0.483 / 0.498 of its own oracle.

**Therefore the frequency claim is confirmed at the representation level, and the head's uniform
shortfall is an optimization / under-fitting gap — not a frequency-dependent miscalibration.**

### My design choice was load-bearing, not neutral — and it caused that gap

Perturbing **all eight bands at once** is a far harder joint objective than P2b's **one band at a
time** (eight independent phase walks to track simultaneously), and that is where the head's deficit
comes from. P2b's quantitative result (`r = r*` within 0.03) **does not carry over to the head in this
design**; it does carry over to the representation (oracle within 0.013). **Design lesson: for a
quantitative "`r` equals its optimum" claim, perturb one band at a time. For a "is it flat in `f`"
claim, uniform perturbation is fine and 3× cheaper.** P3's own answer (answer 1) is unaffected,
because flatness is a *relative* statement.

### Fit gate — two arms are not quotable

```
clean  +0.9534±0.0039 (95%)   β0.5 +0.6483±0.0130 (83%)   β1.0 +0.2364±0.0164 (64%)
β1.5   +0.0313±0.0042 (30%)  → NOT FITTED      β2.0 −0.0043±0.0011 (−23%) → NOT FITTED
```

So β1.5 and β2.0 appear in the table for completeness only. Corroborating the diagnosis: the **clean
(β=0) arm reads 0.939, not 1.000** — a ~6% ceiling loss with *zero* unpredictability in the corpus —
and the same config on P2b's 3-tone corpus read 1.0027. **Nothing has converged at 2000 steps** (every
arm except β=2.0 is still rising; the last 500 steps add +0.037 / +0.055 / +0.050 / +0.017). Contract
2's under-training warning applies to this whole run, and 2000 steps is simply too few for 8 tones.

### Other results and caveats

- **No transient ordering in `f`**: `corr(b, r)` at step 500 → 2000 is +0.30 → +0.56 for the clean arm
  (one drift of +0.008 across the whole range, under its own seed sd) and −0.26…+0.20 for the others.
  `spread_b` falls monotonically with training in every arm. With `Δφ ≡ 0` throughout, this is
  consistent with the transient being `Δφ`-driven rather than `f`-driven (§2) — but it does **not**
  separate the `Δφ` axis, only shows that fixing it at 0 produces no `f` ordering.
- **Constant-envelope check passed**: corpus RMS identical across the 5 arms to 4e-8, and per-band bin
  power exactly 0.0625 in every band × arm.
- The clean arm's own per-band wobble (0.932–0.955) is this config's intrinsic band noise, not a β
  effect.

### Verdict

- **P3's own question is answered**: at matched predictability there is **no residual dependence on
  `f`** — at the head level (flat `spread_b`, no trend in deviations) and, more rigorously, at the
  representation level (the oracle is flat in `f` and reproduces `r*` exactly).
- **P3's levels are not quotable** (two arms fail the gate; the clean arm shows a 6% ceiling loss), and
  its head-level shortfall is my design's fault, not a property of the prior.
- Open: whether the head's uniform deficit is pure under-training. The oracle says the information is
  there and correctly calibrated, so it is an optimization question, and the fix is the P2b design
  (one band at a time) or a longer run — not a different model.
