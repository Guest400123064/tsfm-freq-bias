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

---

## 2026-09-16 — T1: the explaining variable on a real corpus

New: `experiments/1_realdata/` (README + 3 scripts + 3 JSON) and `src/fbias/realdata.py` (a copy of the
lab's Monash loader; logic unchanged). **No model was trained** — T1 is pure data analysis. ≈60 s
total on CPU, nothing downloaded (the Monash cache is local; `GiftEvalPretrain` untouched).

### Corpus: `australian_electricity_demand_dataset`, 30-minute resolution

Survey of four candidates, all NaN-free and locally cached:

| dataset | series | length med (max) | interval | `b` of 24 h | windows @544 | scale ratio |
| :-- | --: | --: | :-- | --: | --: | --: |
| `solar_10_minutes` | 137 | 52 560 | 10 min | 0.222 | 13 152 | 10.7 |
| **`australian_electricity_demand`** | **5** | **230 736** (232 272) | **30 min** | **0.667** | **2 122** | **9.24** |
| `wind_4_seconds` | 1 | 7 397 147 | 4 s | 0.0015 | 13 597 | 1.0 |
| `oikolab_weather` | 8 | 100 057 | 1 h | 1.333 | 1 464 | 5 841 |

Chosen because the 24 h clock lands **inside** the band grid (`b = 0.667`; solar's is at 0.222, below
it, and wind puts 99.65% of its power below `b = 0.125`, i.e. one band wide), the two incommensurate
clocks (24 h + 7 d) make the deterministic part *structured* rather than a single line, there are 2 122
non-overlapping windows, and **solar is degenerate for the morphology diagnostics** — 50% exact zeros
at night. Construction: pool all 5 series, each centred and unit-variance-normalised over its whole
length (contract 9; scale ratio 9.24), stride 544 so the random train/eval split is leak-free, each
patch demeaned before the matched filter, de-clocking = subtract the 336-sample (7 d) folded mean.

### `P(b)` — the deliverable, plus the null controls

Matched-filter each patch at band `b`, ridge-regress the target patch's coefficient on the context's,
`P = 1 − residual_var / var(target)`. 32 bands `b ∈ [0.5, 16]`, `2b` integer so the filter is exact.
Selected rows (full tables in the README):

```
 b     T(h)    power     P_own   P_full   surrogate S_own   de-clocked D_own   in-band
 0.5   32.0   3.24e-1    0.894    0.937        0.872             0.604          0.894
 1.0   16.0   4.32e-1    0.917    0.939        0.902             0.668          0.962
 2.0    8.0   6.13e-2    0.826    0.900        0.874             0.615          0.343
 4.0    4.0   8.68e-3    0.749    0.881        0.847             0.577          0.146
 8.0    2.0   1.48e-3    0.699    0.831        0.854             0.445          0.103
16.0    1.0   7.00e-4    0.718    0.845        0.837             0.441          0.096
```

Train in-sample 0.799/0.903 vs eval 0.782/0.864 — no memorisation. The delivered estimator passes its
**null control** (`P_own = −0.002`, range [−0.006, 0.000]).

**Answer to T1's question.** Predictability varies with band, but **weakly and not monotonically**:
`P_own` runs 0.92 at `b = 1` → 0.68–0.83 for `b ≥ 3`, with wobbles at b = 3, 6, 8, 14; `P_full` is
0.05–0.15 higher everywhere; de-clocked, 0.60 at `b = 0.5` → 0.40–0.48 at `b ≥ 8`. The curves are
**floor-dominated, not structure-dominated**: at a 32-sample aperture the clock lines and the steep red
continuum leak into every bin, which the `in-band` column quantifies (0.96 at `b = 1`, 0.10 at `b = 8`).

**The one result that is clean and immediately useful:** across the grid, **power falls 463× while
`P_own` falls only ~1.25×.** Energy and predictability are demonstrably different objects **on real
data** — the direct real-data counterpart of the Fredformer disagreement (PLAN §2). Also
`P_own − S_own` ≈ +0.02 at the two lowest bands and ≈ **−0.09** above, i.e. the real high bands are
*less* predictable than a phase-randomised surrogate with the same spectrum.

### Morphology diagnostics — the §6.3 prediction is confirmed

```
                   envelope CV    gamma (phase-diffusion)   unwrapped phase sd   slip/env
clock band         0.307 (surr 0.279)      0.17                0.27 rad / 13 yr      0.606
mid / high bands   0.557 / 0.531           1.13 / 1.01         89.6 / 14.5 rad       0.12 / 0.14
surrogate                    —                     ~1.0        large                 0.12–0.14
```

**Clock-locked in the clock band, stationary-like elsewhere** — exactly what PLAN §6.3 predicted and
what it needed to be: the corpus is **not** PM-like in morphology, and its high-band unpredictability
is the stationary/observation-limited kind. The clock component is also amplitude-modulated (envelope
CV 0.307 ≠ 0), so "constant envelope" would have been the wrong description.

Two diagnostics reported as **degenerate rather than dressed up**: linewidth-vs-record-length converges
to a fixed width instead of narrowing as `1/L` (so it does not discriminate here); and the 24 h phase
fold reads `r² = 0.4692`, **exactly the surrogate's**, because that statistic is invariant under phase
randomisation — only harmonic magnitudes matter. The 7 d fold does discriminate: real **0.6188** vs
surrogate 0.5270 → **+0.092** of genuinely clock-locked structure beyond the spectrum.

### Three methodological findings, all reusable

1. **A 32-sample patch cannot isolate a band.** The high-band "power" is 90%+ leakage from the clock
   lines (`in-band` 0.10 at `b = 8`). Both `P(b)` and the model's `r_b` are measured through the same
   matched filter, so they are *consistently* contaminated — but the contamination must be reported,
   and **any comparison between them must hold the aperture fixed**.
2. **The obvious fix — pre-filtering to narrow bands — is silently degenerate**, and it is **the θ flaw
   in a new guise.** A 544-sample window band-limited to 0.5 cycles/patch holds only ~8.5 complex
   degrees of freedom, so the ridge predicts the 17th coefficient trivially and **white noise reads
   `P = 0.998`**. The identical low-DOF trap that made θ a fittability instrument (PLAN §1.3, §3) now
   appears on real data. Only the null control caught it.
3. **`P` is an R² and `r` is an amplitude ratio — they are not the same quantity.** For the AR(1)
   structure the PM instrument produces, `P = r*²`, so a direct T2 comparison must not equate them;
   the exact relation depends on the estimator's noise structure. **Flagged so T2 does not make this
   error.**

### What T2/T3 need, and the open design question

2 122 non-overlapping windows (1 061 / 1 061); stride 272 gives ~4 244 with a time-blocked split. The
fit gate is feasible: 62% of variance is clock-locked and `P_full ∈ [0.82, 0.94]` at every band, so a
mean predictor is not the optimum (the `broad32` failure mode does not apply).

**Open, and it gates T2:** whether a 32-sample aperture is even the right place to read `r_b`. The
measured `P(b)` is nearly flat *partly because* the aperture leaks, so a flat predicted `r_b` would be
a weak test. Options: (a) accept it and keep both readings on the same aperture (the comparison stays
valid, the dynamic range is just small); (b) read both over a longer aperture (e.g. the whole window),
which changes the "predict the next patch" framing; (c) choose a different corpus/resolution where the
clock lands lower `b` and the high bands are less contaminated. **Decide before T2.**

---

## 2026-09-16 — T1b: is T1's flatness an aperture artifact?

Script `experiments/1_realdata/scripts/aperture.py` (imports T1's estimator rather than rewriting it);
`runs/aperture.json`. 73.5 s, CPU. Motivated by T1's `in-band` column: at a 32-sample aperture only
10% of the measured "band power" at `b=8` is genuinely in-band, so the worry was that leakage biases
`P(b)` — and therefore any model's `r_b` — toward **flatness**, making T2 a weak test.

**Answer: no. The flatness is not an aperture artifact.**

| arm | ctx/k | P | window | windows (tr/ev) | null `P_own` | pre-filtered null | in-band @2 h |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| k32 (T1) | 512/32 | 16 | 544 | 2122 (1061/1061) | −0.002 | 0.998 | 0.103 |
| k64 | 1024/64 | 16 | 1088 | 1061 (530/531) | −0.005 | 0.998 | 0.271 |
| k128 | 1024/128 | 8 | 1152 | 1001 (500/501) | −0.005 | 0.995 | 0.546 |

1. **`in-band` does improve ~4× with the aperture**, at *matched physical* bands: 8 h 0.343 → 0.651 →
   0.869; 4 h 0.146 → 0.359 → 0.641; 2 h 0.103 → 0.271 → 0.546; 1 h 0.096 → 0.253 → 0.522.
2. **But `P_own` gains no structure.** On the six physical bands the `k=128` aperture actually
   isolates (`in-band` 0.153 → 0.613), `P_own` range = 0.148 / 0.167 / 0.148 and max/min = **1.218 /
   1.255 / 1.211** for k=32/64/128 — T1's flatness, unchanged. The apparent widening of the range over
   T1's 32 bands (1.35× → 2.30×) is **entirely at the 32 h and 16 h bands that the same arm shows to be
   leakage-dominated.**
3. **`P_own − S_own` keeps its sign at k=32/64** (−0.121 / −0.116 on the well-isolated bands, 6–8× the
   split noise) but is **not resolvable at k=128** (+0.009). Counterpoint, reported rather than buried:
   the **de-clocked** residual *does* gain slope with `k` (range ratio 1.67× → 2.86×), so part of T1's
   Table-B flatness was aperture mixing — **but the raw marginal a model trains on is the
   aperture-insensitive one.**
4. **Nulls clean at every `k`** (−0.002 / −0.005 / −0.005), and the pre-filtered 0.5-wide control stays
   **0.998 / 0.998 / 0.995** — the low-DOF trap is arm-independent and was not triggered. White-noise
   `in-band` is 0.770–0.775 at every aperture, confirming the matched filter is the *same object in
   `b` units* at every `k`.

### Recommendation for T2

**The binding invariant is that `P(b)` and `r_b` go through the same aperture — not the value of `k`.**

- **`k = 32` is safe.** `P` is aperture-insensitive, and T2 can proceed on T1's design. Carry the
  `in-band` column beside `r_b`, since both are leakage-dominated at the high bands.
- **`k = 64` is the better readout if patch size is free**: `in-band` at matched bands rises 1.9–2.6×
  with `P` unchanged, at the cost of half the windows and 1.5× the split noise.
- **Do not use `k = 128`** — 500 windows per split, only 8 context patches, 2.3× the noise, no
  informational gain, and its apparent structure is leakage.
- **Do not re-read `in-band` across apertures.**

### Two corrections to the brief, both conceptual

1. **"Vary only `k`" is not achievable on this corpus, because the aperture *is* the horizon.** The
   next patch is the next **16 h** at `k=32` and the next **64 h** at `k=128`. No arm is a pure
   aperture manipulation — the horizon moves with `k`. This is contract 1 (fix the horizon in patches)
   showing up in physical units, and it means T2 must state its forecasting horizon explicitly.
2. **`in-band` is not cross-`k` comparable.** The ±0.5-`b` window is a *relative* width `1/(2b)`, so
   the same `b` is a different physical filter at each `k` — at low bands it even ranks the better
   aperture worse (`b=1`: 0.962 / 0.095 / 0.065). The `in-band` column is only meaningful *within* one
   arm.

### Consequence for the claim

The worry that motivated this run is **resolved in the safe direction: T2's design at `k=32` is fine.**
And the substantive reading stands and sharpens: **on this corpus, per-band conditional predictability
is genuinely nearly flat across a 32× range of `b` (periods 32 h → 1 h), while power falls 463×.**

That makes T2's prediction sharp and falsifiable: a model trained on this corpus should show **nearly
flat per-band attenuation**, not the "high frequency is worse" pattern — because the prior is
calibrated to a flat predictability curve. T1's `P(b)` is flat, `S_own` shows most of it is
spectrum-implied, and `D_own` shows the clock accounts for a large share of the low-band value.

---

## 2026-09-16 — T2: real-data transfer — **prediction FALSIFIED, with a confound**

Script `experiments/1_realdata/scripts/t2.py` (+ `--summarize`, `--relation`); artifacts
`runs/t2_k32_s*.json`, `runs/t2_k64_s*.json`, `runs/t2_summary.json`, `runs/t2_relation.json`. Both
arms × 3 seeds, 20 000 steps, `hidden 32 / 2L / 4 heads`, SGD 1e-2, batch 64. Wall **1 647 s** for the
6-run batch (parallel). The script **asserts** (`atol 1e-9`) that its training windows reproduce T1's
band coefficients exactly, so the aperture invariant holds by construction.

### The result, stated plainly: the prediction failed

T1/T1b predicted **nearly flat per-band attenuation** on the clean probe, because the corpus's per-band
predictability is nearly flat. Observed:

```
k32 (16 h horizon)   spread_b = 1.2066   pooled per-band seed sd 0.0661   → 18×
k64 (32 h horizon)   spread_b = 0.8587   pooled per-band seed sd 0.0258   → 33×
corr(log2 b, r) = −0.87 (k32) / −0.96 (k64)
```

`r_probe` declines monotonically **7.9× (k32) / 5.8× (k64)** while `P(b)` spans only **1.31×**. The
shape is present at step 1 000 and per-band values move ≤0.13 out to step 20 000, so it is not a
training-length artifact. **T1b's prediction is falsified on this corpus.** (Per the brief, a sloped
result is the *more* interesting finding — report it as such, do not explain it away.)

### The confound, which may matter more than the result

**Contract A9 is insufficient on a red corpus.** Matching *total* RMS leaves the flat-spectrum probe
with 0.126 power in **every** band — **0.29× the corpus's lowest band and 180× its highest**. So a flat
probe against a **617×-red** corpus is out of distribution in **shape**, and T2 cannot separate:

- **(a)** a band-wise prior calibrated to per-band **predictability** (what we predicted), from
- **(b)** a learned response to **spectral shape / power** (a dynamic-range or masking effect: in a
  617×-red signal, the high band is buried in leakage from the low band — the `in-band` column says
  the *measurement* cannot isolate it, and neither can a 32-sample patch embedding).

**T3 (per-patch phase randomisation) is the experiment that separates them.** This is the same trap the
project has hit repeatedly: measuring something out of distribution.

### What stands regardless, and is worth keeping

- **The fit gate passes with a huge margin.** k32 `+0.9161±0.0016`, k64 `+0.8771±0.0015`, against
  **persistence −0.8995 / −1.0028** and **mean predictor +0.1526 / +0.1316** (margin +0.76). Per-band
  variance explained is positive everywhere (0.62–0.95 / 0.34–0.91). **A 32-hidden 2-layer model does
  fit this real corpus**, and no escalation to S1/S2 was needed. Persistence is negative because the
  16 h/32 h lag is not a multiple of 24 h.
- **`r ≈ sqrt(P)` is now measured, not assumed.** T1's own ridge, read as a retention: `r_ms/√P_own` =
  **0.998 (0.971–1.027)** over T1's 32 bands at k32 and **0.994 (0.938–1.105)** at k64. So T1's `sqrt`
  flag was right — **in the pooled-RMS convention** (the mean-of-ratios convention reads 26–36% above,
  because per-window coefficients are heavy-tailed). On the clean probe the denominator is exactly
  constant, so the two conventions coincide and `r_probe` is directly comparable to `√P` — where it is
  **4–5× below** at the top bands.
- **The decline is not a head-only optimisation gap** (which was P3's problem). The **oracle** on the
  clean probe declines too: 2.790 → 0.147, against the head's 1.382 → 0.175. So it is in the
  representation or in the probe's out-of-distribution-ness, not merely in the readout.
- **The power / predictability / retention ranges**: power 617× / 694×, `P_own` 1.31× / 1.32×,
  `r_b` **7.9× / 5.8×**. The model-side range is ~6× the predictability range.
- **`in-band` must be read beside `r_b`.** At k32 it is 0.05–0.15 for `b ≥ 4`, so `r_corpus ≈ 1` there
  is a **leakage cancellation, not retention**. The one well-isolated high band is k64's `b = 4` (8 h,
  `in-band` 0.651) — and the deficit is still there: `0.52` against `√P = 0.91`.

### Smaller notes

- k32's `b = 16` is the sampling Nyquist (± images coincide; a pure tone reads `|Z| = 2A`), flagged
  and reported separately throughout.
- **Phase error was not computed** (the brief asked for amplitude retention only) — recorded as a gap.
- Escalation to `hidden 64 / 4L` was measured but not run: the gate never raised the question.

### Where this leaves the real-data programme

T2 did **not** confirm the mechanism's extrapolation, and the test was not clean. Two candidate next
steps, in the order I would run them:

1. **Test the shape hypothesis directly — cheap.** Flatten the corpus's spectrum, re-measure `P(b)`,
   and re-run the clean probe. If the decline disappears, it was spectral shape, not predictability;
   if it survives, the shape account is dead and the decline needs another explanation. This is the
   decisive control and it costs one T2.
2. **T3 at a well-measurable band.** Per-patch phase randomisation preserves band power *exactly*, so
   the shape confound is absent by construction; and choosing a **low** `b` (where `in-band` ≈ 0.96)
   makes the measurement clean. That combination gives a causal test with no confound — unlike either
   T2 or T3 at the high bands.

---

## 2026-09-16 — T2b: the shape control — **T2's slope was spectral shape, not predictability**

Script `experiments/1_realdata/scripts/t2b.py` (imports `t2` and `predictability`, never reimplements
them); artifacts in `runs/t2b/` (in a subdirectory so T2's glob cannot pick them up). 15 trainings in
three waves, **4 249 s** total. **Arm A reproduced T2 bit-identically** — worst |Δ| = 3.33e-15 over
`P_own`/`P_full`/`in_band`/`power`, and `probe_mean`, `probe_sd`, `corpus_mean`,
`oracle_probe_mean`, `band_var_explained`, `ve_last`, the gate fields, `spread_b`, the seed sd and
`corr(log2 b, r)` differ by **exactly 0.0** — so the cross-arm comparison is sound.

Flattening = one linear filter `g(f) = S(f)^(−α/2)` with `S` the corpus's smoothed mean power
spectrum, applied to the whole series and followed by re-standardisation. Arm B is α = 1; arm C is the
α = 0.5 dose check.

### The answer

| arm | corpus max/min band power | decline `r(1)/r(12)` | `corr(log2 b, r)` | `spread_b` | seed sd | `P_own` max/min |
| :-- | --: | --: | --: | --: | --: | --: |
| A k32 (T2's) | **616.60×** | **7.89×** | −0.87 | 1.2066 | 0.0661 | 1.31 |
| A k64 | **693.70×** | **5.78×** | −0.96 | 0.8587 | 0.0258 | 1.32 |
| **B k32** | **3.11×** | **1.12×** | −0.30 | 0.3572 | 0.0159 | 1.87 |
| **B k64** | **3.04×** | **0.85×** | **+0.29** | 0.4001 | 0.0246 | 1.18 |
| C k32 (α = 0.5) | 81× | 5.37× | −0.94 | — | — | — |

**Flattening the corpus from 617× to 3.1× removes the decline.** T2's slope was **spectral shape /
dynamic range**, not a band-wise prior calibrated to predictability. All arms pass the fit gate
(A +0.9161/+0.8771, B +0.7224/+0.7475, C +0.8758) against persistence ≈ −0.9/−1.0 and a mean predictor
≈ +0.01, so the flattened corpora are still fittable and the comparison is not void.

**Two independent readings support the shape account, not just the arm-wise comparison:**

1. **The one arm where the two drivers are separable.** `corr(power, P_own)` is +0.91/+0.72 in arm A
   (collinear — arm A *cannot* discriminate), but **+0.19 in B/k64**. There,
   **`corr(r_probe, power) = +0.86` while `corr(r_probe, sqrt(P)) = −0.08`.** The retention tracks
   **band power**, not predictability.
2. **Dose check.** 617× → 81× → 3.11× of corpus range gives declines 7.89× → 5.37× → 1.12×, monotone
   but **not proportional**: a 7.6× cut in range buys only 1.5× off the decline, and the collapse
   happens between 81× and 3.1×. A log-dynamic-range story fits the three points better than a linear
   one (stated, not fitted).

### What does **not** vanish, and matters

A **band-flat deficit survives**: in arm B, `r / sqrt(P)` ∈ **[0.45, 0.80]** at k32 and **[0.32, 0.76]**
at k64, **with no band trend** (k64's correlation with `sqrt(P)` is −0.08). T2's 4.5–4.8× deficit at
`b = 8/10/12` becomes **1.5–1.8×**; **shape explains ~2/3 of it in log terms.** So **only the slope was
ever a frequency claim**, and a uniform under-retention of ~1.5–2.2× relative to the ridge's `sqrt(P)`
remains unexplained by either account.

Fit gates and RMS: corpus RMS 0.99995 → 0.99729 (B/k32) and 0.98448 → 0.98911 (B/k64), probe matched in
each arm; probe/corpus per-band power ratio went from **0.29…180** (A/k32) to **1.33…4.14** (B/k32).
`in-band` in arm B is **0.62–0.90** (vs 0.047–0.962 in A) — **T2's leakage caveat disappears with the
flattening**, so the arm-B reading is clean.

### Consequences

1. **The claim's driver is wrong on real corpora.** In controlled synthetic corpora, with band power
   held fixed and predictability varied, retention tracked predictability quantitatively (P2b). On a
   real corpus, with predictability nearly flat and power varying 617×, **retention tracks power**.
   Both are "learned from the training distribution" — so the §0 clarification that it is neither
   spectral bias nor an architectural defect **survives** — but the **specific driver** is
   band **power/dynamic range**, not conditional predictability. **The decisive test is a 2×2:
   vary power and predictability together, orthogonally** (this is P6's amplitude arm, promoted).
2. **This partially rehabilitates Fredformer.** It attributes the bias to band **energy** and is the
   only prior work in the forecasting setting. `corr(r_probe, power) = +0.86` is directional support
   for its mechanism — though *not* for its premise that the bias is removable by architecture, and
   the mechanism is a learned response to the *training* spectrum rather than attention over the input.
3. **A methodological correction: `spread_b / seed sd` is not a flatness statistic.** Arm B reports
   22.5 / 16.3 — *larger* than T2's 18.3 — while its decline is 1.12×. The ratio rose because the seed
   sd fell 4× (0.0661 → 0.0159), not because the spread grew. **Report `spread_b` and
   `corr(log2 b, r)`; keep the ratio, if at all, as a precision read.** (P3 used the ratio; its
   `corr` values carry the conclusion there and its conclusion is unchanged, but the ratio should not
   be quoted alone.)
4. **T3's design must change.** The agent's correction, accepted: **per-patch phase randomisation
   preserves band power exactly but does not remove the shape confound — a red corpus stays red.** So
   **T3 on this corpus must flatten it too**, or it will measure the same channel from the other side.
   My earlier suggestion (T3 at a low, well-isolated band) fixed the *probe-side* mismatch but not the
   *corpus-side* one; both are needed.
5. **Smaller notes.** k32's `b = 16` is still not flat in arm B (a single-bin Nyquist component
   survives the smoother); the k64 residual is a **harmonic comb** the one-pass equaliser cannot notch,
   and it is what arm B/k64's `r_probe` tracks (+0.86) — evidence *for* the shape account, but it caps
   how clean the k64 control is. Running the numpy-heavy T1 estimator with unrestricted threads cost
   70× wall time on one process; every command pins `OMP_NUM_THREADS=2`.

---

## 2026-09-16 — P6 (promoted): the synthetic 2×2, band power × predictability

New: `experiments/2_power_x_pred/` (README with the **pre-registration written before the run**,
`scripts/power_x_pred.py`, `scripts/transient.py`, `runs/`). Target band `b = 8.0` + two mutually
orthogonal control bands `b = 2.0, 4.0`; total power fixed at 0.5 (RMS 0.70711) in all six arms; each
control carries ρ× the target's power, ρ ∈ {1, 5, 25} → corpus band-power range **1× / 5× / 25×**; the
target alone gets PM, β ∈ {0, 1.5} (`r*` = 1.000 / 0.3247); shape-matched probe per arm; 6 arms × 3
seeds at P2b's config. Wall **3 088 s** for 28 trainings.

### The pre-registered 2×2 (2 000 steps, 3 seeds) — verdict: INTERACTION

| arm | ρ | β | `r*` | `r@8` | **`q = r/r*`** | `r@2` ctrl | `r@4` ctrl | gate |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| x1_b0 | 1 | 0.0 | 1.000 | 0.9945±0.0047 | **0.994±0.005** | 0.9957 | 0.9982 | +0.9937 (99%) |
| x1_b1.5 | 1 | 1.5 | 0.3247 | 0.2784±0.0231 | **0.857±0.071** | 0.9944 | 0.9967 | +0.6908 (98%) |
| x5_b0 | 5 | 0.0 | 1.000 | 0.9584±0.0132 | **0.958±0.013** | 0.9982 | 0.9980 | +0.9932 (99%) |
| x5_b1.5 | 5 | 1.5 | 0.3247 | 0.1460±0.0234 | **0.450±0.072** | 0.9993 | 0.9988 | +0.9082 (99%) |
| x25_b0 | 25 | 0.0 | 1.000 | 0.5399±0.0680 | **0.540±0.068** | 0.9990 | 0.9981 | +0.9888 (99%) |
| x25_b1.5 | 25 | 1.5 | 0.3247 | 0.1146±0.0141 | **0.353±0.043** | 0.9989 | 0.9983 | +0.9761 (99%) |

All six pass the gate (98–99% of the analytic bound; persistence baselines +1.000/+0.552/+1.000/+0.878/
+1.000/+0.974). **But the pre-registered criterion over-fired and the agent says so**: as written
("monotone drop ≥ 0.15 at β=1.5 **and/or** β=0") it also fires on a power-only effect — and the **β=0
row moved as much as the β=1.5 row** (+0.455±0.064 vs +0.505±0.034). The deciding statistic, a
difference-of-differences, was **not** pre-registered: 1×→5× −0.372, 5×→25× +0.322, 1×→25× −0.050 —
**a crossover, not a single interaction term**.

### Contract A2 broke the reading — and this is the third time it has

A 10 000-step transient (same seed/corpus/loop; the 2 000-step row reproduces the parent exactly):

```
q @ 10k    0.999   1.017   0.991   0.952   0.958±0.013   0.438±0.057
           x1_b0   x1_b1.5 x5_b0   x5_b1.5 x25_b0        x25_b1.5
```

- The **β=0 collapse was largely under-training** — both low-power rows gain **+0.42**.
- The β=1.5 row gains only **+0.086** (difference −0.333±0.048, 7× sd).
- **At 10 000 steps, five of six cells sit at their optimum** (`q` = 0.95–1.02). The lone exception is
  the **joint extreme** (25×, β=1.5), and it is at 99.8% of its Bayes bound with **96% of its residual
  MSE being exactly the target band's coherent power** — it withholds a band carrying **2% of the
  corpus power**. It is **still rising**, so 20k/40k is the open follow-up.

**So the 2 000-step `q` mixes prior with convergence speed, exactly as contract A2 warns.** The
`rnull` column is also not a floor — it is a constant *absolute* error term divided by a shrinking
`A_t`, and the (25×, β=1.5) cell is only 1.3× above it at 2 000 steps.

### The oracle localises the effect: it is head-side, not a prior

A ridge on the **frozen latents** reads `r*` in **all six cells** — oracle/ceiling = 1.000 / 0.999 /
0.997 at β=0 and 1.009 / 1.005 / 1.002 at β=1.5, **flat in power**. So the representation is orthogonal
and at its optimum in every cell, **and the power effect is a head-side optimisation effect**.

**This is a partial retraction of the "band power drives the prior" reading from T2b.** At 25× range and
10 000 steps the model sits at `r*` in five of six cells; only the joint extreme falls short, and the
representation has it. Note also that T2b's *flat* arm (range 3.11×) showed a decline of only 1.12×,
which is **consistent** with P7 at small range — so T2b's large slope really was the probe-OOD confound,
as concluded there.

### The three checks the brief demanded — all pass cleanly

1. **Power axis = total band power, orthogonal to β**: the total `P@8` ratio (β=1.5 / β=0) is
   **1.0000 / 1.0000 / 1.0000** at ρ = 1/5/25, while the *coherent* power — the quantity I told them not
   to use — falls to **0.1149**. The distinction was real and the right side was taken.
2. **Shape-matched probe**: probe/corpus per-band power ratio **1.0000 at every band in every arm**;
   corpus RMS = probe RMS = 0.70711 everywhere. And **`in-band` = 1.0000 everywhere** — with mutually
   orthogonal integer carriers on a non-red corpus, leakage is structurally absent, measured not
   assumed. (The whole-record ±0.5 mainlobe share is 0.850 at β=1.5 on the corpus side — PM sidebands,
   a caveat the patch-aligned readout does not see.)
3. **P2b replication near-miss**: the gap to `r*` at ρ=1, β=1.5 is **0.046** against the pre-registered
   0.03 tolerance — but it is 0.9 sd from P2b's own 0.3029, and it closes to `q = 1.017` at 10 000
   steps. So it is the same non-convergence, not a disagreement with P2b.

### Where this leaves the account

- **The two axes are largely orthogonal**: at convergence the model sits at `r*` in five of six cells.
  The pre-registered INTERACTION verdict is **not** supported once convergence is honoured — the
  apparent interaction was the β=0 row's under-training plus a crossover in the difference-of-
  differences.
- **The remaining effect is narrow and head-side**: only the joint extreme (very low power **and**
  unpredictable) falls short, and the frozen representation already carries `r*` there.
- **Open**: a converged 3-seed 2×2. The 10 000-step levels are 1 seed except the ρ=25 row, so a
  3-seed 10k grid is the experiment this should have been; the joint extreme additionally needs
  20k–40k. Also still unexplained: on **real** data the oracle *did* decline (T2, 2.79 → 0.15) whereas
  every synthetic deficit so far is head-side — that difference is a live thread.

## 2026-09-16 — `Δ_k` added to the reporting (P7 re-run, 2 000 steps)

**Why.** Fredformer's relative error is now a first-class metric (`fbias.probes.delta_k`, PLAN.md
Appendix B, contract 17), reported **beside** `r` and never instead of it — `Δ_k = |r e^{iΔφ} − 1|`
carries the phase error too, and by the triangle inequality `Δ_k ≥ |1 − r|`.

**What was run.** The P7 grid, unchanged, re-run under the new metric (`--tag dk`). This is a
**pure re-run, not a new experiment**: every `r` reproduces the stored 2 000-step run to **1e-12**
(`power_x_pred_dk.json` vs `power_x_pred.json`).

```
clean probe: r, then Fredformer's dk, then the phase's share of dk
  arm      band        r       dk   dk-(1-r)
  x1_b0     8.0   0.9945   0.0407     0.0352
  x5_b0     8.0   0.9584   0.0867     0.0451
  x25_b0    8.0   0.5399   0.5202     0.0601
  x1_b1.5   8.0   0.2784   0.7266     0.0049
  x5_b1.5   8.0   0.1460   0.8698     0.0158
  x25_b1.5  8.0   0.1146   0.9727     0.0872
```

**Three things this buys.**

1. **In this setting their metric is ours up to a small phase term.** The excess `Δ_k − (1 − r)` is
   **0.005–0.087** across all 18 cells, so `Δ_k` is a monotone reparametrisation of the deficit and
   nothing new is hidden in it. Their `Δ` figures convert directly: `Δ` 0.01 / 0.95 ↔ `r` 0.99 / 0.05.
2. **Their headline number is reachable — but only in the joint extreme, and only unconverged.** The
   (25×, β=1.5) cell reads `Δ = 0.9727`, within 0.02 of their 0.95; the pure power axis at the same
   range reads 0.52, and contract 2 took the β=0 row from 0.54 to 0.958 at 10 000 steps (Δ 0.52 →
   ≈0.09). So their 0.95 is what an **under-converged** model looks like on this design.
3. **Their 0.01 is below our floor.** Even a band the model recovers essentially perfectly
   (`r = 0.9945`) reads `Δ = 0.041`, because the metric is a per-sample mean of `|e^{iΔφ} − 1|` and the
   emitted phase jitters by ~0.05 rad. A `Δ` of 0.01 is therefore not reachable by "recovered
   exactly" in our readout — a difference of measurement convention (their `Δ` is over the
   prediction window of a directly fitted component), not necessarily of performance.

**No conclusion changes.** The pre-registered verdict on this grid stays what it was — `q` reads
INTERACTION at 2 000 steps and five of six cells sit at their optimum at 10 000 (contract 2). The
metric is new; the numbers under it are the same numbers.

## 2026-09-16 — P8: band power allocation, on its own — **power-locked at 2 000 steps, head-side**

The step-1 experiment the programme fixed (`experiments/3_power_alloc/`): three coherent carriers
`b ∈ {2, 4, 8}`, total power `0.5` in every arm, only the **allocation** varies. One band carries
`0.5/(2ρ+1)`, the other two `ρ×` that; `ρ ∈ {5, 25, 100, 1000}` plus an equal-power reference. At each
`ρ` the three arms are **permutations** of one another, so pooling a band over its family cancels the
power effect exactly and leaves frequency. Pre-registered in the experiment README before the run;
`x25_w2` reproduces P7's `x25_b0` bit-identically (both read `0.5399` pooled over 3 seeds).

**2 000-step grid (13 arms × 3 seeds, all fit gates pass):**

```
power effect (weak band vs loud bands, paired within an arm)
   rho    r at weak band    r at loud bands            gap  weak band MSE share
      5     0.9610±0.0038      0.9973±0.0009  0.0363±0.0046              0.09091
     25     0.5075±0.0359      0.9982±0.0003  0.4907±0.0357              0.01961
    100     0.2759±0.0045      0.9982±0.0005  0.7223±0.0042              0.00498
   1000     0.5278±0.0344      0.9985±0.0005  0.4707±0.0349              0.00050

frequency effect (each band pooled over its rho family): spread 0.0012 (tol 0.05)
-> power-locked: the deficit follows the assigned power
```

- **The deficit follows the rotation, not the frequency.** Whichever band is weak reads 0.47–0.54 at
  `ρ = 25` and 0.27–0.28 at `ρ = 100`; the other two read ≈ 1.00 in every arm. Pooled over the family,
  all three bands read 0.855 (spread 0.0012). This is the cleanest possible "the prior tracks power,
  not frequency" — and it is exactly Fredformer's direction.
- **In Fredformer's units the case-study pattern is reproduced.** Loud bands read `Δ ≈ 0.02–0.09`;
  weak bands read 0.52–1.12, bracketing their headline 0.95. Under full control, their Case-1 figure
  is the *transient* of a model learning bands in order of their power contribution.
- **The oracle has everything, at 2 000 steps.** The frozen-latent ridge reads ≈ 1.000 in **every**
  cell, weak bands included (e.g. `x100_w0` oracle@2.0 = 0.998 against a head reading of 0.275). So the
  deficit is entirely **head-side optimisation speed**, ordered by the band's share of the loss — the
  encoder sees the weak band fine; the head has not learned to emit it yet.
- **`ρ = 1000` exits the "learning" regime.** The weak band (0.05 % of the MSE) reads `r ≈ 0.53` with
  `Δ ≈ 1.09 ≈ sqrt(1 + r²)` — the signature of the **right amplitude at essentially random phase**: the
  head emits the band's marginal energy without having locked its phase. Reported as a regime boundary,
  not corrected.

**Why this is not yet the verdict.** Contract 2 binds: P7 already showed this same corpus (its
`x25_b0`) moving 0.51 → 0.958 between 2 000 and 10 000 steps. So the 2 000-step "power-locked" is the
*transient* reading: at any fixed training budget, retention is ordered by band power share, and since
natural corpora put their energy at low frequency, the transient *looks like* a frequency bias. Whether
it survives convergence — and at what `ρ` the deficit becomes permanent — is the 10 000-step leg
(`ρ ≥ 25`, 3 rotations × 3 seeds), running.

**Reconciliation this offers.** The two accounts stop competing: the **prior** (converged behavior on a
clean probe) tracks conditional predictability (P2b, P7 at 10k); the **practice-time bias** (what a
model exhibits at realistic training budgets) is ordered by band power share, frequency-independent
given the rotation, and head-side. Natural data couples the two by putting little energy *and* (via
flat-ish noise on a red signal) low SNR at high frequency — which is the interaction conjecture, landed
as a *speed* effect rather than a *prior*.

## 2026-09-16 — P8 10 000-step leg: the deficit is a **schedule, not a fixed point**

**10 000 steps, ρ ≥ 25, 3 rotations × 3 seeds (all fit gates pass):**

```
power effect (weak vs loud, paired within an arm)     2 000 steps -> 10 000 steps
   rho    r at weak band    r at loud bands            gap      r_weak      gap
     25     0.9615±0.0045      1.0000±0.0002  0.0385±0.0046   0.5075   0.4907
    100     0.5376±0.0346      0.9999±0.0000  0.4623±0.0346   0.2759   0.7223
   1000     0.3012±0.0130      0.9997±0.0002  0.6986±0.0130   0.5278   0.4707

frequency effect (band pooled over the family): spread 0.0066 (2k: 0.0012), tol 0.05
-> pre-registered verdict: power-locked (driven by rho = 100/1000)
```

**Three regimes, one mechanism.** The oracle reads ≈ 1.000 in **every** cell at 10 000 steps too —
including `x1000_w0` oracle@2.0 = 0.992 against a head reading of 0.303. The representation carries
the weak band at every ρ; only the head's learning time changes:

1. **ρ ≤ 25 — transient.** Closes by 10k (gap 0.039, inside tolerance). `x25_w2` reads 0.9582,
   matching P7's transient 0.958 for the same corpus.
2. **ρ ≈ 100 — practical transient.** Half-learned at 10k (0.54); a single-exponential fit of the
   2k → 10k improvement gives τ ≈ 20–50k steps — closure in principle, beyond typical budgets.
   `dk ≈ 1 − r`: phase locked, amplitude still growing — plain slow learning.
3. **ρ ≥ 1000 — below the SGD noise floor.** `r` *fell* from 0.53 (2k) to 0.30 (10k), with
   `dk ≈ 0.95–0.98 ≈ sqrt(1 + r²)` — the right-amplitude/wrong-phase signature of the head
   random-walking: the weak band's 0.05 % share of the gradient is below the optimizer's noise, so
   its weights diffuse instead of converging. The deficit is **permanent within any practical
   budget** here — but it is still gradient SNR, still head-side, still not a prior.

**What this settles.** Across four orders of magnitude of power disparity, at two budgets: the
deficit is **perfectly power-locked** (frequency spread ≤ 0.007 at both budgets — the rotation
control never wavers) and **never representation-side** (oracle ≈ 1.000 throughout). Power writes
the *learning schedule*, never the *fixed point*. The fixed point is r\* = 1 at every band, set by
predictability alone (P2b). Fredformer's Case-1 pattern is regime 2–3 observed at 50 epochs — a
budget effect, which is exactly why their `Δ` ordering tracks amplitude share and exactly why their
remedy (a preconditioner) can compress the schedule but cannot move the fixed point.

**Open follow-up from this leg:** a per-ρ transient (5+ time points) would fit the learning-curve
form τ(ρ) and locate the noise-floor boundary precisely; currently inferred from two time points
plus the non-monotone ρ = 1000 reading.

## 2026-09-17 — P11 2 000-step grid: the schedule is matched, and the marginal account's signature is already absent

**2 000 steps, 7 arms × 3 seeds (all fit gates pass):**

```
fixed-point decision table (b8 on each arm's probe)
  X    r@8 = 0.8288±0.0332   (homogeneous reference, b8 share 4% in every window)
  y4   r@8 = 0.8342±0.0330   marginal-account prediction 0.040
  y6   r@8 = 0.8376±0.0343   marginal-account prediction 0.060
  y8   r@8 = 0.8365±0.0302   marginal-account prediction 0.080
  c    r@8 = 0.5527±0.0662   (anchor: b8 share 2% in every window = P8's x25_w2)
  a    r@8 = 0.9955±0.0036   (flat homogeneous)
  b    r@8 = 0.9907±0.0219   (flat heterogeneous, marginal exactly flat)
```

- **The schedule check passes by construction and in the data.** X and all three Y arms
  carry b8's loss-share at exactly 4% (q·R = m8 = 0.02), and they read alike at 2 000 steps:
  0.829 / 0.834 / 0.838 / 0.837. Two arms, opposite occurrence structure, one schedule —
  the ladder isolates the fixed point exactly as designed.
- **The marginal account's signature is already absent at 2 000 steps.** The Y arms emit
  **~0.83** of a band whose corpus marginal share is 2% — 10–20× the 0.04–0.08 the
  marginal-prior account predicts at *every* budget. Whatever the model is doing with the
  loud-b8 probe, it is not integrating the allocation out. The formal fixed-point
  discrimination is the 10 000-step leg (running): conditional account says these go to
  ~1, marginal account says they stay at 0.04–0.08.
- **Controls are clean.** The flat arms read ≈ 0.995 everywhere; heterogeneity itself
  (arm b, three different allocations cycling) costs nothing detectable against the
  homogeneous flat arm (0.9907 vs 0.9955).
- **Anchor consistency.** `c` reads 0.553 at 2 000 steps against P8's `x25_w2` = 0.540 —
  statistically identical; not bit-matched, because P11's synthesiser draws phases through
  a different RNG path than P8's `make_pm` (a statistical anchor, not a reproduction).
- **Reporting artifact, noted:** y4's probe is a pure-b8 window (alloc (0, 0, 0.5)), so its
  b2/b4 "retention" cells divide by a zero true amplitude and print ~1e4 (same for its b2/b4
  oracle cells). The decision uses r@8 only; the display will NaN-out zero-power bands in a
  later edit.

**Interim reading.** At equal marginal and equal schedule, the model's response to a
context-visible loud band is ~10× its corpus marginal share — the response tracks what the
context shows, not what the corpus averages to. The 10k leg decides whether it converges
all the way to 1.

## 2026-09-17 — P11 10 000-step leg: **conditional account wins — the bias tracks loss-share, not the marginal**

**10 000 steps, x / y4 / y6 / y8 × 3 seeds (all fit gates pass):**

```
fixed point (b8 on each arm's probe)          2 000 steps   10 000 steps   marginal account
  X  (homogeneous, b8 share 4% every window)   0.8288        0.9868         —
  Y4 (4% of windows at full b8 power)          0.8342        0.9809         0.040
  Y6 (6% of windows at b8 = 1/3 of power)      0.8376        0.9770         0.060
  Y8 (8% of windows at b8 = 1/4 of power)      0.8365        0.9731         0.080
  -> conditional account: marginal-matched heterogeneity is free of bias;
     the operative variable is occurrence x in-pattern power (loss-share)
```

- **Every Y arm converges to 0.97–0.98** on a probe whose band carries a **2% corpus
  marginal** — 12–24× the marginal-prior account's prediction, which is excluded at every
  point of the ladder. A band that appears loud in 4% of windows is copied almost perfectly
  at convergence, marginal be damned.
- **The dose-response is flat**, as designed: because q·R = m₈ pins every arm's b8
  loss-share to 4%, the three Y arms move together from 0.83 (2k) to ~0.98 (10k). There is
  no occurrence threshold to find — loss-share is the whole story.
- The Y arms sit ~0.01 below X at 10k (0.973–0.981 vs 0.987): a whisper of heterogeneity
  cost, inside ~1 sd, not significant — consistent with the 2k control arms, where
  heterogeneity was also free.

**What this settles for the data-mixing question.** Matching the corpus marginal is
neither necessary (the Y arms are wildly non-flat in-window yet unbiased) nor the
mechanism (same marginal, opposite occurrence structure, identical outcome). What governs
both the schedule and the fixed point is the **context-resolved conditional mean**: the
model copies what the context shows, and learns it at a speed set by the pattern's share
of the loss. The actionable rule for data preparation: **supplement a corpus so that the
target behaviour — the allocation patterns you want the model to reproduce — occurs with
adequate loss-share; there is no need to flatten the marginal, which the per-window
generating process controls anyway.**

**Status of the branch.** P8 (power alone: schedule, not fixed point) + P11 (occurrence
structure: conditional, not marginal) close the "energy" axis of the programme: apparent
frequency bias from spectral shape is a finite-budget schedule effect on the head side,
and its fixed point is entirely predictability's. Reporting artifact carried: y4's pure-b8
probe leaves its b2/b4 cells as division-by-zero displays (decision uses r@8 only).

## 2026-09-17 — P11-16 2 000-step leg: the 16-band discrimination, **with a design flaw to record**

**2 000 steps, x16 / y16 × 3 seeds (both pass the fit gate):**

```
             r@16 (target)      oracle@16     min r over bands
  x16        2.0269±0.6699       1.6999        0.5013
  y16        1.0109±0.1926       1.2274        0.5580
  pre-registered: y16 >= 0.9 -> conditional account; marginal account predicts 0.100
  -> conditional account generalises
```

**The flaw, stated plainly: the target band b16 is Nyquist.** b16 = k/2 is its own
negative-frequency image, so the matched filter reads 2× the design power there and its phase
is degenerate (a bare sign). I caught this in the generator check *only after* writing the
arm definitions, fixed the *reporting* (marginal-deviation statistic excludes b16; P12 excludes
it from every decision statistic) but did **not** move this experiment's target off it. The
consequence is visible in the numbers: a homogeneous arm whose b16 share is 1.85% — and which
should read ≈ 0.9 on share-ordering — reads **2.03** at 2 000 steps and 1.31 at 10 000 (seed 0,
from the aborted first launch), i.e. the readout is biased **high** at this band and converges
toward ~1 from above rather than from below.

**What survives.** The discrimination is not near its boundary: y16 emits ≈ 1.0 of a band whose
corpus share is 1.85%, against a marginal-account prediction of 0.10 — an order of magnitude
apart. So "the fixed point tracks the context, not the corpus marginal" holds at 16-band
resolution too. But the *absolute* r@16 levels are not trustworthy, the x16-vs-y16 comparison at
this band is muddied (2.03 vs 1.01), and any tightening of this experiment should move the
target to a well-conditioned band (b15) before quoting numbers.

**Infrastructure note.** The first launch ran four jobs with torch's default intra-op threads
(12 each) on 12 cores: load average 21, a 2 000-step training went from 74 s to **669 s**, and the
10 000-step legs would have needed ~14 h. Relaunched at `OMP_NUM_THREADS=2` per job: 65–83 s per
2 000-step training, load ≈ 6.5. The 2 000-step grid was re-run from scratch under the fixed
threading (no partial results were kept), so all times in this log are comparable.

## 2026-09-17 — P12 2 000-step leg (16 bands, hidden 32): **capacity-limited — and that is the finding**

**Pre-registered verdict: `capacity story`.** At 16 coherent carriers with `hidden 32`, the
frozen-latent oracle is short at every arm — **0.898 / 0.837 / 0.765 / 0.689 / 0.617** for
α = 0 / 0.5 / 1 / 1.5 / 2 — so the per-band deficits cannot be attributed to the schedule; the
representation itself does not carry all sixteen bands. This is the pre-registered check doing
exactly its job: the *fit gate* passed everywhere (var_explained 0.72–0.92), because the low bands
dominate the variance. **At 16 bands the fit gate is not the binding check; the oracle is.**

**The ladder's shape is nevertheless informative** (α ladder, clean shape-matched probes, 3 seeds):

```
r(b1) / r(b15) at 2 000 steps        hidden 32      [hidden 64, α=1 probe]
  alpha = 0.0                          1.006         —
  alpha = 0.5                          1.464         —
  alpha = 1.0                          1.915        1.283
  alpha = 1.5                          1.978         —
  alpha = 2.0                          1.585         —
```

- Monotone over the first four rungs, then **α = 2 falls back** — and α = 2 is precisely the arm
  with the worst oracle (0.617). A capacity ceiling that bites hardest where the spectrum is most
  extreme is the natural reading.
- **A single-arm probe at `hidden 64` clears the bar**: α = 1 oracle minimum **0.9504** (against
  0.7645 at hidden 32), and its apparent bias ratio drops to **1.283**. So part of the small-model
  "frequency bias" here was capacity, exactly as contract 15's ladder anticipates.
- Within-arm Spearman(log share, r) came in at 0.943 / 0.957 / 0.861 for α = 0.5 / 1 / 1.5 (short
  of the 0.8 rule only at α = 2, 0.325 — the same arm).

**Two measurement bugs found and fixed.**
1. The dose-response ratio was printed with `BANDS[-1]` = **b16** in the denominator — the Nyquist
   band, whose readout is inflated ~1.3–1.6× (it reads 1.26–1.58 while every well-conditioned band
   reads < 1). Corrected to `b15`; the table above is recomputed from the stored JSON. The
   pre-registered collapse statistic already used `b1..b15`, so it was unaffected.
2. The per-seed progress line reported `r@16`; now reports `r@15`.

**Actions taken.** `--hidden` / `--layers` are now CLI knobs on the P12 driver, and the α ladder is
re-running at the **S3 rung (`hidden 64`)** for 2 000 and 10 000 steps. The hidden-32 grids stay in
the log as the *capacity-limited rung* of the same ladder, which is itself the trend comparison
contract 15 asks for. The 16-band occurrence experiment (P11-16) is unaffected by this specific
confound in its *target* reading (its target band is the degenerate Nyquist one, flagged
separately) but shares the small-model capacity question; its hidden-32 legs are read with that
caveat.

## 2026-09-17 — P11-16 10 000-step leg, and the corrected re-run

**10 000 steps, x16 / y16 × 3 seeds (hidden 32, target band = Nyquist b16):**

```
              r@16            dk@16          min r        oracle@16
  x16        1.5700±0.3201   1.0271±0.2685   0.7354       1.3278
  y16        1.0870±0.0614   0.3709±0.0600   0.7486       1.0882
  decision: conditional account (y16 = 1.087 against a marginal prediction of 0.100)
```

The direction is the same as the 3-band result and as the 2 000-step leg: a band whose corpus
marginal share is 1.85% is emitted at ~1.0 when the context shows it loud — **10× the
marginal-account prediction**. But the absolute levels are not usable: x16, the *homogeneous*
arm that should read ≈ 0.9 on share-ordering, reads 2.03 (2k) → 1.57 (10k), converging toward 1
from above, and its oracle reads 1.33 (> 1, impossible for a retention). Both are the Nyquist
artifact: b16 = k/2 is its own negative-frequency image, its phase is a bare sign, so the readout
is inflated and does not mean what the other bands' readout means.

**Corrected re-run launched** (`--target 15 --hidden 64`): the driver now builds the arms around
any target band, so the discrimination sits on **b15** (α=1 red share 1.97%, a well-conditioned
band) and the model is at the S3 rung, which the P12 capacity finding shows is required at 16
bands. Marginals are still pinned to RED16 exactly (`0.9c + 0.1d = 1` for every non-target band,
`0.1R = RED16[tgt]`), and the target's schedule share stays 1.97% in both arms. Both legs (2k, 10k)
are running; the hidden-32 N=1 runs stay in the log as the flagged version.

**Lesson recorded.** The Nyquist degeneracy was visible in the *first* generator check (probe b16
power 2× design) and I fixed only the reporting, not the design. Any future N-band grid should
either exclude b16 from the target role or treat "cycles/patch = k/2" as a special case from the
start.

## 2026-09-17 — P12 2 000-step leg at the S3 rung (hidden 64): capacity gone, collapse **narrowly fails**

**Pre-registered checks (5 α × 3 seeds, all fit gates pass, oracle now clears everywhere):**

```
1. oracle minimum over bands (>= 0.9)      a0 0.978  a05 0.969  a10 0.949  a15 0.927  a20 0.901
2. Spearman(log share, r)  (>= 0.8)        a05 +0.925  a10 +0.993  a15 +0.979  a20 +0.668  <-- short
3. collapse: worst spread over multi-arm octaves   0.1741  (tol 0.15)                      <-- fails

dose-response r(b1)/r(b15)     1.018 (a0)  1.160 (a05)  1.370 (a10)  1.597 (a15)  1.470 (a20)
```

- **Capacity resolved.** `hidden 64` lifts the oracle minimum from 0.617–0.898 to **0.901–0.978** at
  every α, and the whole dose-response curve flattens (α=1: 1.915 → **1.370**). The small-model
  ladder was inflated by capacity, as the contract-15 comparison anticipated.
- **The share ordering holds over three of the four red rungs**: Spearman 0.925 / 0.993 / 0.979 for
  α = 0.5 / 1 / 1.5 — the within-arm deficit really is ordered by the band's share.
- **It breaks at α = 2 (0.668), and the reason is visible in the band table**: that arm's retention
  declines to **0.53 at b8 and then stops** — b9–b15 read 0.57 / 0.60 / 0.61 / 0.55 / 0.62 / 0.67 /
  0.68, a plateau, not a continued decline. Those bands carry shares of 0.5–0.8% at 2 000 steps:
  this is P8's **below-the-noise-floor regime** appearing inside a single arm.
- **The collapse fails narrowly (0.174 vs 0.15), and it fails in one place.** The mid-range octaves
  are tight (share 0.0625–0.125: 23 points from all 5 arms, spread **0.062**; share 0.125–0.25:
  spread 0.015). The offender is octave −7 (share 0.0078–0.0156): there α = 1.5 retains 0.82 while
  α = 2 retains 0.55 at *matched share*. So share alone does not fully determine the deficit once
  the spectrum is extreme — the band's identity or its context matters there, or the plateau is a
  budget artifact.
- Note on the auto-verdict: the printed branch ("shape does not order the deficit as a share
  effect") is selected by the α = 2 Spearman alone; the checks are more nuanced than that label.

**What the 10 000-step leg decides.** If α = 2's plateau is a budget artifact, its ordering and the
octave −7 spread should both recover; if the plateau is structural (a per-band floor independent of
budget), it should persist and the collapse should fail for the same reason at 10k. Running.

## 2026-09-17 — P11-16 clean version at 2 000 steps (target b15, hidden 64): the schedule check passes again

```
              r@15            dk@15          min r        oracle
  x15        0.7147±0.0708   0.5697±0.1092   0.7147       0.9608
  y15        0.6669±0.1358   0.3797±0.1377   0.6669       0.9946
  marginal-account prediction 0.100   ->   auto-verdict "intermediate" (threshold is for 10k)
```

- **Capacity is not binding here**: the oracle clears 0.96 / 0.99 at the well-conditioned target
  band, unlike the Nyquist-targeted version whose oracle read > 1 (meaningless).
- **The schedule check passes, as in the 3-band design**: x15 and y15 read 0.715 vs 0.667 —
  within ~0.7 sd of each other — from corpora with opposite occurrence structure and an identical
  1.97% occurrence-weighted share of the target band. Both sit **6–7× above the marginal-account
  prediction of 0.10**, and both are about half-learned at 2 000 steps (share 1.97% is inside P8's
  slow-schedule regime; the 3-band P11 arms at share 4% read 0.83 at the same budget).
- y15's spread is twice x15's (0.136 vs 0.071) — expected: only 10% of its training windows carry
  the target band loudly, so fewer effective samples per batch.
- The 10 000-step leg is the fixed-point read: conditional account → both toward ~1, marginal
  account → y15 pinned near 0.10.

**Comparison with the flagged version**: the Nyquist-targeted run read x16 2.03 / y16 1.01 at 2k
and 1.57 / 1.09 at 10k — inflated and non-comparable. The clean run's absolute levels (0.71 / 0.67)
are in the same band as the 3-band design's transient behaviour, which is what a well-conditioned
readout should look like.

## 2026-09-17 — P11-16 clean version, 10 000 steps: **conditional account generalises** (final, on a well-conditioned band)

```
              r@15            dk@15          oracle     2k -> 10k
  x15        0.9144±0.0173   0.2965±0.0134   0.9722     0.715 -> 0.914
  y15        0.9160±0.0146   0.1270±0.0071   0.9958     0.667 -> 0.916
  pre-registered: min r >= 0.9 -> conditional account. Marginal-account prediction 0.100.
  -> conditional account generalises: full-spectrum marginal-matched heterogeneity is free of bias
```

- **The two arms converge to the same number to three digits** (0.9144 vs 0.9160) from corpora with
  opposite occurrence structure — one where the target band is present at 1.97% power in every
  window, one where it is **absent in 90% of windows** and loud in the rest. The marginal account's
  0.100 is excluded by a factor of nine, at a well-conditioned band (b15, not the degenerate
  Nyquist b16) and at adequate capacity (oracle 0.97 / 1.00).
- **A new detail: the concentrated corpus locks phase better.** At matched amplitude retention,
  `dk` (which carries the phase error) is **0.127 for y15 against 0.297 for x15** — implying phase
  errors of ~0.10 rad and ~0.30 rad. A band that appears weakly in every window gives a noisier
  phase estimate than the same band appearing strongly in a tenth of the windows: concentration
  buys phase precision even though it leaves the amplitude story untouched. Heterogeneity is not
  merely "free of bias" — on this axis it is slightly *better*.
- Worth putting beside the 3-band result: there, x/y at a 4% share reached 0.987 / 0.973–0.981 at
  10k; here, at half the share (1.97%), they reach 0.914 / 0.916. Same schedule law, one more
  budget rung down the curve.

**Final status of the occurrence branch.** 3-band P11 (0.987 vs 0.973–0.981 against 0.04–0.08) and
16-band P11-16 (0.914 vs 0.916 against 0.100) agree: with the corpus marginal fixed, the model
copies a band the context shows as loud **regardless of how rare that is in the corpus**. The
flagged Nyquist-targeted run's direction was the same (1.09 vs 0.10) but its levels were unusable;
the clean run replaces them.

## 2026-09-17 — P12 10 000-step leg at hidden 32: the oracle improves but stays short, and α=2's plateau was a budget artifact

```
1. oracle minimum (>= 0.9)      a0 0.953  a05 0.935  a10 0.834  a15 0.725  a20 0.630
                                (2k:  0.898  0.837  0.765  0.689  0.617)   <-- short for alpha >= 1
2. Spearman(log share, r)       a05 +0.929  a10 +0.982  a15 +1.000  a20 +0.989   all pass
3. collapse worst spread        0.1931   (tol 0.15)   <-- still fails
   dose-response r(b1)/r(b15)   0.788 / 0.836 / 0.726 / 0.668 / 0.707   (2k: 1.006 / 1.464 / 1.915 / 1.978 / 1.585)
```

- **Training shrinks the apparent bias substantially even at hidden 32** (α=1: 1.915 → 0.726), which
  is the schedule story's signature again — the deficit is largely thing-you-get-more-of.
- **But capacity is not merely an under-training artifact at this size**: the oracle improves
  (α=2: 0.617 → 0.630; α=1.5: 0.689 → 0.725; α=1: 0.765 → 0.834) and stays **short for every
  α ≥ 1**, with the shortfall ordered by redness. At `hidden 32` the representation cannot carry a
  strongly red 16-band spectrum, and more training does not fix it.
- **α = 2's within-arm plateau was a budget artifact** — its Spearman goes 0.325 (2k) → **0.989**
  (10k), so the share ordering does eventually impose itself even in the reddest arm at this size.
- **The collapse still fails (0.193)**, and here the reason is capacity, not the schedule: with
  oracles that differ across arms by 0.95 → 0.63, the same share does not mean the same thing in
  different arms.

Reading the two hidden-32 budgets together with the hidden-64 2k leg: the α ladder needs the S3 rung
before its cross-arm statistics mean anything, and at that rung the collapse was still marginal
(0.174 at 2k). The hidden-64 10 000-step leg is running and is the one that separates "marginal at
2k, tight at convergence" from "a real residual dependence on more than the share".

## 2026-09-17 — P12 10 000-step leg at the S3 rung (hidden 64): **the collapse holds** — the apparent bias is the schedule, read at spectral resolution

```
1. oracle minimum (>= 0.9)      a0 0.988  a05 0.982  a10 0.972  a15 0.953  a20 0.917   all pass
2. Spearman(log share, r)       a05 +0.911  a10 +0.964  a15 +0.989  a20 +0.996        all pass
3. collapse worst spread        0.1049   (tol 0.15)                                    HOLDS
   dose-response r(b1)/r(b15)   1.008 / 1.040 / 1.076 / 1.201 / 1.484   monotone in alpha
-> "one curve r = G(share) describes every spectrum"

collapse bins (share octave : points / arms / r_mean / spread)
  -7  0.0078-0.0156 :  8 / 2 / 0.885 / 0.105      <- the loosest, still inside tolerance
  -6  0.0156-0.0312 : 11 / 3 / 0.946 / 0.038
  -5  0.0312-0.0625 : 19 / 4 / 0.969 / 0.039
  -4  0.0625-0.1250 : 23 / 5 / 0.978 / 0.029
  -3  0.1250-0.2500 :  4 / 4 / 0.994 / 0.007
  -2  0.2500-0.5000 :  2 / 2 / 0.998 / 0.003
```

**Every pre-registered check passes**, and both 2 000-step anomalies were convergence artifacts:
α=2's within-arm Spearman recovers 0.668 → **0.996**, the dose-response stops dipping at the top
(1.470 → 1.484, now monotone across all five rungs), and the collapse tightens 0.174 → **0.105**.
That is the fourth time contract 2 has rescued a reading — and the first time it has *created* one.

**The two-rung comparison is the capacity lesson.** At `hidden 32` / 10k the same ladder gives
collapse 0.193 with the oracle short for α ≥ 1 (0.83 / 0.72 / 0.63); at `hidden 64` / 10k it gives
0.105 with every oracle ≥ 0.92. The small-model failure was not a property of the schedule but of
the representation's inability to carry a strongly red 16-band spectrum — which is exactly why
contract 15's ladder exists.

**The headline number, in blog form.** At 10 000 steps and `hidden 64`, the ratio of the model's
retention at the strongest band (b1) to its retention at the weakest (b15) is:

| α | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| r(b1)/r(b15) | 1.008 | 1.040 | 1.076 | 1.201 | 1.484 |

**Apparent frequency bias is a monotone, quantitative function of spectral redness at a fixed
budget, and its shape is a single curve in the band's share of the loss.** Nothing in the ladder
required a frequency-dependent mechanism: five spectra, sixteen bands each, one curve.

**Status: P12 complete.** Both rungs (hidden 32 as the capacity-limited rung, hidden 64 as S3) at
both budgets; all numbers in this log; the pre-registered verdicts are `capacity story` (hidden 32,
both budgets) and `collapse holds` (hidden 64, 10k).

## 2026-09-17 — P12 control analysis: is the cross-arm comparison confounded by overall fit quality?

Prompted by a fair question: the α ladder compares retention *across* corpora whose overall MSE
necessarily differs, so does the dose-response just reflect "the redder arms fit differently"?

**Recorded, and not matched — but it cannot be.** Every arm's held-out target variance is
**0.4992–0.49997** by construction (total power is fixed at 0.5), while the residual differs:

```
hidden 64, 10 000 steps      var_explained        held_var     MSE        sum_b (1-r_b)^2 P_b   ratio
  alpha 0.0                  0.9547±0.0013        0.49924    0.02261          0.00035          0.02
  alpha 0.5                  0.9564±0.0016        0.49955    0.02178          0.00035          0.02
  alpha 1.0                  0.9640±0.0018        0.49978    0.01797          0.00037          0.02
  alpha 1.5                  0.9722±0.0021        0.49991    0.01389          0.00062          0.04
  alpha 2.0                  0.9768±0.0021        0.49997    0.01161          0.00115          0.10
```

Two things follow.

1. **The redder arms fit *better* overall** (MSE 0.0226 → 0.0116). This is structural and not
   fixable by design: a red corpus concentrates its power in a few easy low bands, and the tiny
   high bands it neglects cost almost nothing in loss. You cannot hold total power fixed *and*
   match MSE across α. What *is* matched is the target variance (`held_var`), i.e. the signal
   power — so the only free quantity is how much of it the model explains, and `var_explained` is
   the normalised form of it.
2. **The amplitude deficit we measure is only 2–10 % of the total MSE.** The rest is phase error at
   the loud bands (the `dk` column). Matching MSE across arms would therefore be matching a
   quantity our deficit measurement barely contributes to.

**The control that does work — matching overall fit quality instead of MSE:**

```
octave (share)        arms in the bin              var_explained spread    r spread
-7  0.0078-0.0156     a15, a20                     0.0046  (matched)        0.105
-6  0.0156-0.0312     a10, a15, a20                0.0127                  0.038
-5  0.0312-0.0625     a05, a10, a15, a20           0.0204                  0.039
-4  0.0625-0.1250     a0, a05, a10, a15, a20      0.0221                  0.029
-3  0.1250-0.2500     a05, a10, a15, a20           0.0204                  0.007
```

- **The loosest octave is the one where overall fit is matched**: a15 and a20 differ by 0.0046 in
  `var_explained` (0.9722 vs 0.9768) yet differ by **0.105** in `r` at the same share. So the
  residual spread is **not** a fit-quality artifact.
- **The octaves with the largest fit differences have the smallest r spreads** (0.020–0.022 fit
  spread against 0.029–0.039 r spread). The residual does not track fit quality at all; if
  anything the relationship runs the other way.

**Where the residual actually comes from** (recorded, not resolved): in octave −7 the comparison is
a15's bands b10–b15 against a20's bands b7–b8 — the *same share* on *different bands*, sitting in
different surrounding spectra. So the residual is a band-identity / context effect, not a budget or
capacity effect. It is the one place in P12 where share is not the whole story, and it is inside the
pre-registered tolerance (0.105 ≤ 0.15) so it does not change the verdict.

## 2026-09-17 — P13 2 000-step leg (free shapes): share explains 98% of the variance, with a caveat about how check 3 is computed

**Pre-registered checks (10 profiles × 3 seeds, 150 pooled points, all fit gates pass):**

```
1. oracle minimum (>= 0.9)   r4 0.882 <-- SHORT, v2 0.891 <-- SHORT, rest 0.902-0.978
2. share-only monotone fit R^2 = 0.9830   (>= 0.9 required)          PASS
3. frequency adds nothing: worst |corr(r, band)| in a bin = 0.566 (tol 0.5)   <-- marginal fail
                           worst within-bin r spread       = 0.143 (tol 0.15)  PASS
-> auto-verdict: capacity (r4, v2 short) -- both only just short, at 2 000 steps
```

**The headline is check 2**: one monotone curve in loss-share explains **98.3 %** of the variance of
`r` across **150 points from ten structurally different spectra** — power laws, random log-normals,
random bumps, flat — where the loudest band lands on b1, b5, b7, b11 and b16 in different profiles.
Share is doing the work, not frequency.

**Check 3 is the one to read carefully, and the bin statistic is the problem.** A post-hoc
decomposition (labelled post-hoc, computed after seeing the 2k grid) says:

```
residual  = r - isotonic(share)            residual sd  0.0205   vs   r sd 0.1569
corr(residual, band index), all 150 pts    +0.084
corr(residual, arm)                        -0.105
per-arm residual mean                      -0.009 (v2) ... +0.029 (r2)
```

- **Globally, frequency adds nothing**: the pooled residual has essentially no band trend (+0.084)
  and no arm trend (−0.105), and its sd is 13 % of the spread of `r`.
- **The failing bin** (share 0.023–0.040, corr +0.566) is one of the low-share bins, where the points
  come disproportionately from a few arms — in a violet profile the low bands are the quiet ones, so
  "low share" and "low band" are correlated *through the arm*, not through the model. A bin-level
  correlation is therefore a noisy, arm-composition-dependent statistic at the ends of the range,
  whereas the pooled residual correlation is the honest form of the same question.
- **Within arms the residual is not flat** (r3 +0.589, r4 +0.340, v2 +0.314, flat −0.333): inside a
  single spectrum, at matched share, higher bands are retained slightly *better* in some arms. The
  effect is small (residual sd 0.02) but it is real and it is the same phenomenon P12's deepest
  octave showed (0.105 spread).

**What the 10 000-step leg decides.** Both short oracles (0.882, 0.891) should clear 0.9 with more
training, as they did in P12; and the residual structure should either sharpen or wash out. If the
freshly-trained R² stays near 0.98 with a band-free residual, "share is the sufficient statistic" is
the defensible claim, with the within-arm band trend reported as a small, honestly-flagged residual.

## 2026-09-17 — P13 10 000-step leg: **share is the sufficient statistic** (all three checks pass)

```
1. oracle minimum (>= 0.9)     0.9282 ... 0.9874 across all ten profiles        PASS
2. share-only monotone fit R^2 = 0.9558   (>= 0.9, 150 pooled points)           PASS
3. frequency adds nothing      worst |corr(r, band)| in a bin = 0.482 (tol 0.5) PASS
                               worst within-bin r spread      = 0.088 (tol 0.15) PASS
-> "share is the sufficient statistic: one monotone curve in loss-share predicts
    retention irrespective of frequency"
```

**Post-hoc residual decomposition**, computed the same way as at 2 000 steps so the two budgets are
comparable (labelled post-hoc: the isotonic curve is not part of the pre-registration):

```
                     r sd     residual sd    corr(residual, band)    R^2
  2 000 steps       0.1569      0.0205             +0.084            0.9830
 10 000 steps       0.0580      0.0122             -0.070            0.9558
  within-arm corr(residual, band), 10k:  v1 +0.13  v2 +0.35  r1 -0.06  r2 -0.11  r3 -0.05
                                         r4 -0.27  r5 +0.01  bump1 -0.09  bump2 -0.69  flat -0.48
  (2k, same arms:                        v1 +0.05  v2 +0.31  r1 -0.14  r2 +0.06  r3 +0.59
                                         r4 +0.34  r5 +0.17  bump1 -0.16  bump2 -0.19  flat -0.33)
```

- **The global frequency trend is gone** (−0.070 at 10k, against a positive +0.084 at 2k — the sign
  flips, which is what noise looks like, not a law).
- **The within-arm trends are unstable across budgets** (r3: +0.59 → −0.05; r4: +0.34 → −0.27;
  bump2: −0.19 → −0.69), so they read as transient rather than structural. With n = 15 per arm,
  |corr| ≈ 0.51 is the nominal 5 % threshold, so only bump2's −0.69 is even borderline.
- **R² falls from 0.983 to 0.956 while the residual halves** (0.0205 → 0.0122): convergence
  compresses `r`'s range from 0.157 to 0.058 faster than it shrinks the residual, so the same
  absolute scatter is a larger share of a smaller range. Both numbers are honest; the absolute one
  is the one to quote.

**What P13 adds over P12.** P12's α ladder could not separate share from band identity — share and
frequency were perfectly collinear inside a power-law family, and its collapse held only marginally
(0.105 vs a 0.15 tolerance) with a residual in the deepest octave. P13 broke that collinearity
directly: ten structurally different spectra (violet power laws, random log-normals, random bumps,
flat) in which the loudest band lands on **b1, b5, b7, b11 and b16** across the set, 150 pooled
points, `hidden 64`. One monotone curve in loss-share predicts retention with **R² = 0.96** and a
band-free residual of **sd 0.0122**. The violet arms — the mirrored case where energy sits at high
frequency — behave exactly as the share account requires.

**Status: P13 complete.** Both budgets; verdict at 2 000 steps was `capacity` (two oracles at
0.882 / 0.891, eight at ≥0.902) and at 10 000 steps `share is the sufficient statistic`.

## 2026-09-17 — P14: mixed-corpus learning dynamics — order yes, single curve no

One corpus (per-window lognormal spectra around b^−1, σ = 0.8, measured marginal spread **12.2×**),
16 bands, probe every 1 000 steps to 20 000, 3 seeds, `hidden 64`.

```
1. final checkpoint: fit gate ok, oracle minimum 0.9690 (>= 0.9)                 PASS
2. ordering: Spearman(log share, t80) = -0.993                                   PASS (sign bug, below)
3. time collapse under u = step x share: worst within-bin spread 0.186 (tol 0.15) FAIL
-> pre-registered branch: "power sets the order but not one universal time constant"
```

**A sign bug in my own pre-registration, stated plainly.** The criterion was written as
`Spearman(log share, t80) >= 0.8` while the prediction beside it was "higher power → *earlier*
t80". With `t80` = steps to reach `r = 0.8`, a bigger share means a *smaller* t80, so the correct
criterion is `≤ −0.8`. The measured value is **−0.993** — essentially a perfect rank ordering, and
the strongest possible support for the prediction. The stored JSON's verdict string was produced
with the buggy sign; the code now reads the sign correctly. This is a fix to the statistic, not a
relaxation: the prediction text always said "earlier".

**The fitted law is `t80 ∝ share^(−0.55)`** — i.e. **`t80 ∝ 1/amplitude`**, since share ∝ amplitude².
That matches the derivation from P8's mechanism: the loss contributes one factor of amplitude to a
band's gradient, so learning time scales as `1/√P`, not `1/P`. The two candidate laws are
visually separable in `figures/p14_scaling_law.png`, and the data sits on the amplitude law.

**But no exponent collapses the family.** Sweeping β in `u = step × share^β` from 0.1 to 1.1 leaves
the worst within-bin spread between **0.19 and 0.25** throughout — always above the 0.15 tolerance,
and the fitted β = 0.55 does no better than the pre-registered β = 1 (0.249 vs 0.244 at 8 bins).
So the failure is not a wrong time constant: **the curves differ in shape, not only in scale** — the
bands share a trunk, so their trajectories are not rescalings of one another. This is the one place
where the schedule picture is incomplete, and it is recorded as such.

**What holds.** The ordering is as clean as this project has produced: t80 runs 1 000 steps for
b1–b4 up to 3 000 for b13–b15, against a 12.2× spread in share, with b1–b4 censored at the 1 000
first-checkpoint floor (their true t80 is below the sampling grid). The frozen-latent oracle rises
from ~0.94 to 0.97 over the run — the representation itself learns the weak bands as training goes
on, which is the time-resolved form of P12's capacity finding.

**Figures** (`experiments/7_mixed_dynamics/figures/`, script `scripts/make_figures.py`): dynamics —
`p14_heatmap_r.png`, `p14_curves.png`, `p14_collapse.png`, `p14_t80_share.png`,
`p14_early_share_vs_r.png`, `p14_oracle.png`, `p14_scaling_law.png`; cross-experiment —
`p13_share_vs_r.png`; setup — `setup_bands.png`, `setup_shapes.png`, `setup_phase.png`. The Δk
panels (`p14_heatmap_dk.png`, `p14_curves_dk.png`) need the `_dk` rerun, which records `delta_k`
at every checkpoint; the first run did not record it.

**Infrastructure note.** `seaborn 0.13.2` (+ `pandas`) added via `uv add seaborn`, as requested.
The first render had 8 of 11 figures clipping text (`context="talk"` fonts on fixed-width
canvases); fixed by `context="notebook"` + `bbox_inches="tight"` on every save, and the 15-way
band labels on the curve plots reduced to b1/b5/b9/b13/b15 (37 overlapping label pairs → 0).

## 2026-09-17 — P14 Δk rerun and the figure set

The first P14 run recorded `r`, `r_oracle` and `var_explained` at each checkpoint but **not**
Fredformer's `Δk`, which the figure set needs. `delta_k` is now computed in the checkpoint
evaluator and the run was repeated (`mixed_dynamics_s20k_dk.json`).

**The rerun reproduces the first run exactly** — worst `|Δr|` between the two over all
seeds × bands × checkpoints is **0.0** (bit-identical training, same seeds and config), so the
Δk curves carry the same trajectory as the retention curves rather than a second sample of it.

```
Δk at step 1k    b1 0.14   ->  b15 0.68
Δk at step 20k   b1 0.073  ->  b15 0.245
```

i.e. the same ordering as retention, expressed in Fredformer's units: the weak band starts ~5×
worse and ends ~3.4× worse, and every band improves by roughly the same factor.

**Figures.** `scripts/make_figures.py` writes 13 PNGs to `experiments/7_mixed_dynamics/figures/`:
the dynamics set (`p14_heatmap_r`, `p14_heatmap_dk`, `p14_curves`, `p14_curves_dk`,
`p14_collapse`, `p14_t80_share`, `p14_early_share_vs_r`, `p14_oracle`, `p14_scaling_law`), the
cross-experiment `p13_share_vs_r`, and the setup trio (`setup_bands`, `setup_shapes`,
`setup_phase`).

Three defects found by an independent visual pass and fixed, in order of how much they mattered:

1. **8 of 11 figures clipped their titles** — `context="talk"` fonts on fixed-width canvases.
   Fixed with `context="notebook"` + `bbox_inches="tight"` on every save.
2. **A patch that did not apply**: the band-label subset (b1/b5/b9/b13/b15 instead of all 15, to
   kill 37 overlapping label pairs) silently missed its target string, so the shipped figures still
   carried 15 labels. Re-applied against the actual file text and verified by rendering.
3. **The Δk heatmap was built from the wrong run** and the `r` heatmap showed the same panel twice
   (the two runs' retention is bit-identical). Both fixed: each field is now drawn from the run that
   actually recorded it, single panel, with colour limits hugging the data (retention 0.64–0.99,
   Δk 0.07–0.68) instead of the previous 0–1 / 0–1.1.

The visual check was done by a subagent with image access, twice — once to find the defects, once to
confirm the fixes — because the code author cannot see the PNGs. Both rounds reported the on-disk
files as pixel-identical to the current script's output.

## 2026-09-17 — P15 (bimodal marginal): the ordering is by share, within one run

P14's corpus mean is red, so its share is monotone in band and the run cannot separate "ordered by
power" from "ordered by frequency". The bimodal variant removes that degeneracy in a single run:
two humps at b4 and b13, same per-window lognormal jitter (σ = 0.8) and 1% share floor.

```
1. final checkpoint: fit gate ok, oracle minimum 0.9645 (>= 0.9)                  PASS
2. ordering: Spearman(log share, t80) = -0.721   (<= -0.8 required)               marginal FAIL
3. time collapse: worst within-bin spread 0.196 (tol 0.15)                        FAIL
4. counter-monotone: t80(b1) = 2333 > t80(b4) = t80(b13) = 1000                   PASS
```

**Why check 2 failed, and why it is a measurement artefact.** 8 of the 15 bands reach r = 0.8 before
the first checkpoint, so their `t80` is tied at the 1 000-step floor; a rank correlation over a list
that is more than half ties is diluted. The bimodal corpus has a narrower share spread than the red
one (5.6× against 12.2×) and fewer very-quiet bands, so more bands saturate inside the first 1 000
steps. Floor-free versions of the same statistic, computed from the stored curves:

```
Spearman(log share, t80)            -0.721   <- pre-registered, floor-limited
Spearman(log share, mean deficit)   -0.921   <- floor-free
Spearman(log share, deficit at 3k)  -0.950   <- floor-free
```

**Check 4 is the one this variant exists for, and it passes decisively**: the lowest frequency is
2.3× slower than both humps, which "low frequency is preferred" cannot produce.

**The cleanest reading is by share group** — the bimodal design happens to place several bands of
*different* frequency at the *same* share, which is a within-run version of P13's cross-shape test:

```
share group        bands                mean deficit (20 checkpoints)
0.019              b8, b9               0.138
0.034              b1, b7, b10          0.070
0.064-0.066        b15, b2, b6, b11     0.037
0.087-0.104        b3, b5, b12, b13, b14, b4  0.028
```

Within a group the bands span low, middle and high frequency and agree; across groups the deficit
falls monotonically as the share rises. b1 — the lowest frequency in the set — sits in the
**second-worst** group, and b13 — a *high* band — sits in the best.

**The collapse fails again (0.196)**, as in P14 (0.186), for the same reason: the bands share a
trunk, so their curves differ in shape and not merely in scale. Two corpora, two budgets, same
outcome — this is now a reproducible negative result about the strongest form of the claim, and the
verdict branch was pre-registered.

**Red path unchanged.** `--shape red` reproduces P14's generator bit-for-bit (identical realised
shares), so P14's result stands as-is; the bimodal variant is additional evidence, not a revision.

**Open, cheap fix — now running.** A rerun with finer early checkpoints (every 250 steps to 5k, then
1k; 35 checkpoints per seed instead of 20) resolves `t80` for the fast bands and lets the
pre-registered statistic be read without ties. Same seeds and config, so the training is
bit-identical — the fine run's 1 500-step smoke reproduces the coarse smoke's `loss 3.229e-02`,
`r@1 0.806`, `r@15 0.910` exactly — and only the sampling of the trajectory changes. The
pre-registered criteria are unchanged; the verified prediction is that the 8 bands pinned at the
floor resolve to 250–1 000 steps and the Spearman moves toward the floor-free values (−0.921 on the
area statistic, −0.950 at 3k).

### P15 figures and two hardcoded labels

`figures_bimodal/` holds the same 13 panels for the bimodal run (`--out figures_bimodal`; the script
now takes an output subdirectory). The visual check confirmed the signatures this variant exists for:
in `p14_heatmap_r.png` the darkest row is a **mid-axis stripe** (b8/b9, row means 0.860/0.863 against
0.92–0.98 elsewhere) rather than an edge; in `p14_early_share_vs_r.png` points of different colours
share one rising curve, with b1 (lowest frequency) in the lower-middle; and `p14_t80_share.png` shows
the 8 floored bands as a horizontal row.

**Two labels were hardcoded to the red run and silently became wrong** when the corpus changed:
`p14_scaling_law.png` annotated "b1–b4 censored" (the bimodal run floors b2,b3,b4,b5,b11–b14 and
leaves b1 at 2 333) and `setup_shapes.png` titled the corpus "alpha=1.0". Both are now derived from
the run (`N of 15 bands censored`; the mean spectrum's name, via `config.get("shape", "red")` so the
older JSONs without the key still render correctly). A third fix: the early panels draw low-share
points last, so b1 is no longer overdrawn by b7/b10 (was ~40 % of its marker visible, now 100 %).

Worth recording as a process failure, because it happened three times in this session: **string
replacements against a file that a formatter has since rewritten fail silently.** Each time the
patch reported success (the script ran, the file was written, lint passed) while changing nothing,
and only the independent visual check caught it. The fix that worked was to read the file, then use
an edit tool that errors when its anchor is absent.

## 2026-09-17 — P15 fine-checkpoint rerun: the marginal failure of the ordering test was censoring, and the full t80 table is a share ladder

The bimodal run was repeated with `--ckpt-fine 250 --ckpt-fine-until 5000` (35 checkpoints per seed
instead of 20). Same seeds and config: **bit-identical training** — at the 60 checkpoints the two runs
share, the worst |Δ| over every `r@b` and `dk@b` is **0.0**. Only the sampling of the trajectory
changes, which is what the earlier marginal failure needed.

```
1. final checkpoint: fit gate ok, oracle minimum 0.9645 (>= 0.9)      PASS
2. ordering: Spearman(log share, t80) = -0.954  (<= -0.8 required)    PASS   (was -0.721)
3. time collapse: worst within-bin spread 0.244 (tol 0.15)            FAIL   (was 0.196)
4. counter-monotone: t80(b1) = 1917 > t80(b4) = 583, t80(b13) = 667   PASS
-> "power sets the order but not one universal time constant"
```

**Check 2 was failing on the quantisation, not on the data.** With 1 000-step sampling, 8 of 15 bands
tied at the floor; at 250-step resolution the whole table resolves and the correlation is −0.954 —
right where the floor-free statistics had put it (−0.921 on the area statistic, −0.950 at 3k). The
earlier verdict string is superseded.

**The resolved t80 table is a clean function of share, and it is not a function of frequency:**

```
share            bands                t80
0.099 / 0.090    b4,  b14             583      <- b4 is low-mid
0.094 / 0.104    b12, b13             667      <- b13 is the highest band in the set
0.087 / 0.089    b3,  b5              750
0.064-0.066      b15, b2, b6, b11     917      <- b15 is the *highest* frequency of all
0.034            b7, b1, b10          1750-2083  <- b1 is the lowest frequency of all
0.019-0.020      b9,  b8              3833, 4250
```

Every row of that table holds bands from opposite ends of the spectrum at the same `t80`: b15 (the
Nyquist-adjacent band) learns as fast as b2, and b1 (the lowest band) learns as slowly as b7 and b10.
The counter-monotone check is decisive — the lowest frequency needs **3.3× longer** than b4 and
**2.9× longer** than b13 — and no monotone frequency account produces the table's shape.

**The collapse fails again, and harder (0.244 vs 0.196).** That is the expected direction: with only
1 000-step sampling the statistic was blind to the steepest part of the rise, where the curves differ
most. Finer sampling does not rescue the one-curve claim, it strengthens the negative result. Three
corpora (P14 red, P15 bimodal coarse and fine) now agree that the family is ordered by share but not
a single rescaled curve.

**Figures**: `figures_bimodal_fine/` (13 panels) is generated from this run and supersedes
`figures_bimodal/` for visual reading, since the sampling is three times denser; the coarse JSON
remains as the first pre-registered record of the bimodal variant.

### Fine-grid figures and a tick-label bug

`figures_bimodal_fine/` (13 panels) is generated from the fine run; `figures/` and `figures_bimodal/`
were regenerated with the same code. One defect found by the visual check and fixed: the x-tick
thinning applied `step % stride` to the **step value** instead of the **column index**, so the fine
heatmap showed a duplicate `3k` and a misleading `0k` (750 ms read as "0k", 3 750 as "3k"). Now
index-based, labels read `0.25k, 1k, 1.75k, 2.5k, 3.25k, 4k, 4.75k, 7k, 10k, 13k, 16k, 19k` on the
35-column grid and `1k … 19k` on the 20-column one, both verified free of duplicates, overlap and
clipping, with the cell grids reconstructed against the run JSON (max abs error 0.0017).

That is the fourth silent failure of this kind in the session — the others were string replacements
that missed their anchor. This one at least produced a visible artefact that the check could catch.
