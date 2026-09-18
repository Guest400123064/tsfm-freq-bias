# `2_power_x_pred` — the 2×2: band **relative power** × band **predictability**

Design contract: [`PLAN.md`](../../PLAN.md) §3–§4 and appendices A–B. Run log:
[`experiments/README.md`](../README.md). Scripts: `scripts/`. Artifacts: `runs/`.

**Status:** design frozen and pre-registered below **before** the run. Results in the
`Results` section at the end (filled in after the run; nothing above it was changed).

---

## Why this run exists

P2b held a band's **power fixed** (constant envelope) and varied its **predictability**; clean-probe
retention tracked the closed-form optimum `r* = exp(−β²σ²/2)` to within **0.03**. T2b then found
that on a real corpus with predictability nearly flat (`P_own` range 1.31×) and band power varying
**617×**, retention tracked band **power** (`corr(r_probe, power) = +0.86`) and not predictability
(`corr(r_probe, sqrt(P)) = −0.08`).

**Both can hold only if the model responds to both drivers and whichever varies dominates.** That
claim is currently an inference from two experiments in *different* settings, one of which (T2b) is
a real corpus where the two axes are collinear in the arm that can discriminate them. This run puts
both axes under control in **one synthetic design where every cell has a closed-form predicted
retention**, so the two accounts can be told apart directly:

- if the model sits at its optimum in every cell, the axes are **orthogonal** and
  (power, predictability) is a sufficient description of a band;
- if `r/r*` falls systematically in the low-power rows at fixed β, **power modulates the
  predictability prior** and P2b + T2b must be unified into a two-factor account with an explicit
  interaction term.

This is P6's amplitude arm (PLAN §7's robustness backlog: "occurrence/power decoupling, amplitude
coupling"), promoted to a tracked experiment.

## Design

One **target band** plus two control bands, one target band's predictability perturbed at a time
(contract 16: *quantitative* claims perturb one band at a time).

| | |
| :-- | :-- |
| **Bands** | `b = 2.0, 4.0, 8.0` (`f = 32, 64, 128`). **Target = `b = 8.0`** (`f = 128`, period **4 samples** = `k/b`, i.e. 128 cycles per 512-sample window). Controls `b = 2.0` (16 samples) and `b = 4.0` (8 samples) |
| **Why these `b`** | all integers ⇒ `2b` integer ⇒ the matched filter is **exact**; integer spacing (gaps 2 and 4, contract 4 needs ≥ 1) ⇒ the carriers are **mutually orthogonal over a patch**, so no band can leak into another and `in-band` is 1.000 by construction — measured, not assumed. `b = 8.0` is P2b's headline band, so the 1× row is a cross-script replication |
| **Δφ** | `2πb (mod 2π) = 0` for every band, so the transient/`Δφ` axis is held fixed (P3's condition) |
| **Corpus** | `make_pm` per band, scaled and summed (see *Axis 1*) |
| **Config** | `ctx 512 / k 32 / hidden 32 / 2L / 4 heads / SGD 1e-2 / batch 64 / 2000 steps`, 256 windows/split (train/held-out), 3 seeds — **identical to P2b** so the numbers are comparable |
| **Probe** | clean (no phase walk), **shape-matched per arm**, fixed seed 123, RMS-matched exactly (see *Shape-matched probe*) |

### Axis 1 — the target band's **relative power**

The three bands carry shares of a **fixed total power 0.5** (RMS `√0.5`, the P2b/P1 convention, so
contract A9's total-RMS match holds by construction in all six arms). Each control band carries
`ρ`× the target band's power, with `ρ ∈ {1, 5, 25}` ⇒ corpus band-power range `max/min = ρ` =
**1× / 5× / 25×**, i.e. straddling the region where T2b's dose curve collapsed (617× → 81× → 3.1×).

| `ρ` | target `P_t` | each control `P_c` | target amplitude | control amplitude |
| --: | --: | --: | --: | --: |
| 1 | 0.16667 | 0.16667 | 0.577 | 0.577 |
| 5 | 0.045455 | 0.227273 | 0.302 | 0.674 |
| 25 | 0.009804 | 0.245098 | 0.140 | 0.700 |

**The quantity used is the band's *total* power, not its coherent power.** Under PM the envelope is
constant, so a band's total power is `P_t` at **every** β; PM only *redistributes* it between a
coherent part `r*²·P_t` and an unpredictable part `(1 − r*²)·P_t`. Operationally, the total power is
`mean_p |Z_p|²/2` — the mean *square* of the per-patch matched amplitude, which is β-invariant
because each patch is a whole number of carrier cycles. The coherent power `|mean_p Z_p|²/2` **is**
coupled to β (it falls by `r*²`), and using it as the power axis would collapse the 2×2 into a 1×2
diagonal. Both are computed and reported, and the orthogonality is verified numerically in the
generator check (below).

### Axis 2 — the target band's **predictability**

`β ∈ {0, 1.5}` on the target band only ⇒ `r* = exp(−β²/2) = 1.000` and **0.3247**. The control
bands stay fully coherent (`β = 0`, persistent tones) — they are the internal control.

| arm | `ρ` | β | `r*` |
| :-- | --: | --: | --: |
| `x1_b0` | 1 | 0.0 | 1.000 |
| `x1_b1.5` | 1 | 1.5 | 0.3247 |
| `x5_b0` | 5 | 0.0 | 1.000 |
| `x5_b1.5` | 5 | 1.5 | 0.3247 |
| `x25_b0` | 25 | 0.0 | 1.000 |
| `x25_b1.5` | 25 | 1.5 | 0.3247 |

**6 arms × 3 seeds = 18 runs.** The pairing is deliberate: every band's phase walk is drawn from the
same generator seed in every arm, and `make_pm` multiplies the *same* `ε_p` stream by β, so the β
contrast is paired and the `ρ` contrast holds the realised phases fixed.

### Shape-matched probe (T2b's lesson)

`make_mixture(..., normalize=True)` gives every tone amplitude 1, so a flat probe against a 25×-ranged
corpus is out of distribution in **shape** — exactly the confound that re-created T2's slope in T2b.
Each arm's probe is therefore

```
probe = Σ_j sqrt(2 P_j) · cos(2π f_j t / ctx + φ_j),   φ_j fixed per window, drawn from seed 123
```

— the arm's **own per-band powers**, the corpus's bands at β = 0 (clean tones), no noise. RMS is
matched exactly (`Σ P_j = 0.5` in probe and corpus alike), and the **probe/corpus per-band power
ratio is 1.00 at every band in every arm** (verified, see the generator check). The probe's target
band is a clean tone, i.e. the clean version of the corpus's target band.

## Readouts, per arm

1. **`r` at the target band on the clean probe and `r / r*`** — the headline column. A model at its
   optimum in every cell reads `r / r* = 1` in all six.
2. `r` at the two control bands on the probe (expected ≈ 1; the internal control).
3. **The fit gate (contract A8)** — held-out variance explained, against the analytic bound
   `1 − P_t(1 − r*²)/0.5`, plus its two baselines (**mean predictor** = the context mean;
   **persistence** = the previous patch) so an under-fit arm is visible.
4. `rnull` — the model's emitted amplitude at **`b = 6.0`**, a band with no content in either the
   corpus or the probe, in units of the target band's true amplitude. This is the target reading's
   noise floor: `rnull ≪ r` is what makes a `r = 0.3` reading meaningful rather than leakage.
5. Per-band marginal retention `rm@b` on the arm's own held-out corpus (beside its ceiling, which is
   `r*` at the target band and 1 at the controls) and the probe oracle (contract 5, out-of-sample
   ridge on the frozen latents).
6. Corpus band-power range; probe/corpus per-band power ratio; `in-band` (the cross-talk matrix's
   diagonal, per band, corpus and probe); the realised per-band β (inverted from the corpus's
   unpredictable part — measured, not assumed); RMS values; 3 seeds with spread.

## Pre-registered decision

Let `q(cell) = r(probe, target) / r*` (per seed, then mean ± sd over the 3 seeds). Written **before**
the run:

1. **Orthogonal** — every one of the six cells has `|q − 1| ≤ 0.10`, **and** at fixed β the spread of
   `q` across the three power levels (max − min) is `≤ 0.08`. Reading: the two axes are orthogonal
   and the model sits at its optimum in each cell, so **power and predictability each drive
   independently and (power, predictability) is a sufficient description of a band**.
2. **Power modulates the predictability prior** — at β = 1.5 (and/or β = 0) `q` **falls monotonically**
   as the target's power share falls (1× → 5× → 25×) with a total drop `Δq ≥ 0.15` that is also
   `≥ 3×` its own paired seed sd. Reading: the two drivers are **not independent**; P2b and T2b must
   be unified into a two-factor account, and the interaction size is `Δq`.
3. **Intermediate** — anything else: quantify the interaction (`Δq(β)` with its paired seed sd and
   monotonicity) rather than describing it.

Additional pre-registered rules:

- An arm whose held-out variance explained is below **half** its analytic bound is reported as
  **NOT FITTED** (P2b's `GATE = 0.5`) and its `r` is **not** quoted as a retention result.
- If the `ρ = 1` row does not reproduce P2b's `r = r*` within 0.03 (the same instrument, one control
  band moved from `b = 0.25` to `b = 4`), that is reported as a **replication failure** and every
  other cell is read with that caveat.
- **No knob is tuned to make a pattern appear.** Orthogonal, interacting and ambiguous outcomes are
  all acceptable; the one observed is the one reported. Steps, lr, model size, β, bands, power
  levels and seeds are fixed here and were not revisited after seeing results.
- Never report a number that was not computed; if an arm fails the gate, say so.

---

## Results

Run 2026-09-16, CPU. Artifacts in `runs/`; everything below is computed by the two scripts and
printed by them.

| artifact | what |
| :-- | :-- |
| `runs/power_x_pred.json` (+ `.log`, `.summary.log`) | the pre-registered 18-run 2×2 and its generator checks |
| `runs/power_x_pred_transient.json` (+ `.log`) | contract A2: all six cells to 10 000 steps, seed 0 |
| `runs/power_x_pred_transient_s12.json` (+ `.log`) | seeds 1 and 2 for the `ρ = 25` row to 10 000 steps |

**Wall: 3 088 s total for 28 trainings** — 833 s for the 18 pre-registered runs (46 s each), 1 353 s
for the six 10 000-step transient runs, 902 s for the four `ρ = 25` seeds-1/2 runs.

### 1. The pre-registered 2×2 — clean probe, target band `b = 8.0`

`r` is the fixed-position amplitude ratio on the clean probe, mean ± sd over 3 seeds; `q = r / r*` is
the headline column. `rnull` is the model's amplitude at `b = 6.0` (empty in both corpus and probe),
in units of the target band's true amplitude.

| arm | `ρ` | β | `r*` | `r@8` | **`q = r/r*`** | `r@2` ctrl | `r@4` ctrl | `rnull` | fit gate |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| `x1_b0` | 1 | 0.0 | 1.000 | 0.9945 ± 0.0047 | **0.994 ± 0.005** | 0.9957 ± 0.0048 | 0.9982 ± 0.0035 | 0.0267 | +0.9937 (99%) |
| `x1_b1.5` | 1 | 1.5 | 0.3247 | 0.2784 ± 0.0231 | **0.857 ± 0.071** | 0.9944 ± 0.0051 | 0.9967 ± 0.0047 | 0.0328 | +0.6908 (98%) |
| `x5_b0` | 5 | 0.0 | 1.000 | 0.9584 ± 0.0132 | **0.958 ± 0.013** | 0.9982 ± 0.0045 | 0.9980 ± 0.0023 | 0.0536 | +0.9932 (99%) |
| `x5_b1.5` | 5 | 1.5 | 0.3247 | 0.1460 ± 0.0234 | **0.450 ± 0.072** | 0.9993 ± 0.0023 | 0.9988 ± 0.0018 | 0.0491 | +0.9082 (99%) |
| `x25_b0` | 25 | 0.0 | 1.000 | 0.5399 ± 0.0680 | **0.540 ± 0.068** | 0.9990 ± 0.0024 | 0.9981 ± 0.0011 | 0.1103 | +0.9888 (99%) |
| `x25_b1.5` | 25 | 1.5 | 0.3247 | 0.1146 ± 0.0141 | **0.353 ± 0.043** | 0.9989 ± 0.0024 | 0.9983 ± 0.0016 | 0.0882 | +0.9761 (99%) |

Every arm passes the fit gate at 98–99% of its analytic bound, so all six `r` are quotable. The gate's
baselines: mean predictor ≈ `+0.000` in all six; persistence `+1.000 / +0.552 / +1.000 / +0.878 /
+1.000 / +0.974` — persistence is strong only where the corpus is predictable, as it should be. The
two **control bands are the internal control**: perturbed by nothing in any arm, they read
0.994–0.999 everywhere, so the effect is band-local (P2b's result again).

### 2. The pre-registered decision: **INTERACTION**

| β | `q` at 1× → 5× → 25× | monotone | spread | `q(1×) − q(25×)`, paired |
| --: | :-- | :-- | --: | --: |
| 0.0 | 0.994 → 0.958 → 0.540 | yes | **0.455** (> 0.08) | **+0.455 ± 0.064** (7× its sd) |
| 1.5 | 0.857 → 0.450 → 0.353 | yes | **0.505** (> 0.08) | **+0.505 ± 0.034** (15× its sd) |

Four of six cells are outside `|q − 1| ≤ 0.10` (`x1_b1.5`, `x5_b1.5`, `x25_b0`, `x25_b1.5`); no arm
fails the gate. **Criterion 2 fires at both β, so the pre-registered answer is criterion 2: power
modulates the predictability prior — the two axes are not independent.**

Two things must be said immediately, and both are in the data above:

1. **The criterion as I pre-registered it cannot distinguish that from a power-only effect.** It says
   "at β = 1.5 **and/or** β = 0" and requires only a monotone drop ≥ 0.15, so it also fires when the
   β = 0 row moves — and the β = 0 row *does* move (0.455, i.e. as much as the β = 1.5 row). The
   statistic that actually separates the two accounts is the **difference-of-differences**, which I
   did not pre-register. It is reported in §3 below, and it is not resolvable over the full range.
2. **The β = 0 row's drop is contract A2 under-training, not a converged property.** §5 shows it
   going to 0.958 at 10 000 steps. This is why the run below exists.

**Replication check (pre-registered, tolerance 0.03).** The `ρ = 1` row is P2b's instrument with one
control band moved from `b = 0.25` to `b = 4`: P2b read `r@8 = 0.3029 ± 0.0129` at β = 1.5 against
`r* = 0.325`; this run reads **0.2784 ± 0.0231** (0.9 sd apart, so consistent), but its gap to `r*` is
**0.046**, above the 0.03 tolerance. **The replication is a near-miss by the letter of the
pre-registration** — the β = 0 arm reads 0.9945 (gap 0.006, inside tolerance) — and every cell above
is read with the caveat that this config's head sits ~0.02 lower on this corpus than on P2b's.
§5 shows that gap is also under-training: at 10 000 steps the same cell reads 0.3302, i.e. `q = 1.017`.

### 3. What the interaction is made of

`q = a(ρ) + b(β)` is the additive account; the last column is the pure two-factor interaction
(paired by seed). A drop that is the same at both β is power acting *independently* of predictability.

| power step | drop in `q` at β = 0 | drop in `q` at β = 1.5 | difference of the two |
| :-- | --: | --: | --: |
| 1× → 5× | 0.036 ± 0.009 | 0.408 ± 0.004 | **−0.372 ± 0.005** |
| 5× → 25× | 0.419 ± 0.055 | 0.097 ± 0.033 | **+0.322 ± 0.049** |
| 1× → 25× | 0.455 ± 0.064 | 0.505 ± 0.034 | −0.050 ± 0.052 |

So the interaction is **not** a uniform rescaling of the predictability term, and over the whole range
it is **unresolved** (−0.050 ± 0.052). It is instead a crossover: from 1× to 5× the predictability
penalty deepens sharply while the coherent band is essentially untouched (0.994 → 0.958), and from 5×
to 25× the opposite happens. My pre-registered "interaction" verdict is therefore right about
*non-independence* and wrong to suggest a single interaction term.

### 4. Where it lives: the representation, or the head?

The oracle is a ridge readout of the arm's **frozen latents** (contract 5), out of sample.

| `oracle / ceiling` at band | ρ = 1 | ρ = 5 | ρ = 25 |
| :-- | --: | --: | --: |
| β = 0 (ceiling 1) | 1.000 | 0.999 | 0.997 |
| β = 1.5 (ceiling `r*`) | 1.009 | 1.005 | 1.002 |

**The representation is at its optimum in all six cells, and it is flat in power** (the oracle reads
1.000 / 0.999 / 0.997 at β = 0 and 0.328 / 0.326 / 0.325 — exactly `r*` — at β = 1.5). So the power
effect is *not* in the learned representation: it is in the head's readout, i.e. an optimisation
effect, exactly P3's split (representation at `r*`, head below it) — but here the head's shortfall is
**power-dependent**, which P3's uniform 8-tone design could not see.

That immediately raises the question of whether the head's shortfall is a converged property or a
training budget, and the empty-band floor says how much room for interpretation the low-power readings
have. In absolute amplitude at the target band (true amplitude `sqrt(2 P_t)`):

| arm | true `A` | `r` | emitted | `rnull` | floor amplitude | emitted / floor |
| :-- | --: | --: | --: | --: | --: | --: |
| `x1_b0` | 0.5774 | 0.9945 | 0.5742 | 0.0267 | 0.0154 | 37.2 |
| `x1_b1.5` | 0.5774 | 0.2784 | 0.1607 | 0.0328 | 0.0190 | 8.5 |
| `x5_b0` | 0.3015 | 0.9584 | 0.2890 | 0.0536 | 0.0162 | 17.9 |
| `x5_b1.5` | 0.3015 | 0.1460 | 0.0440 | 0.0491 | 0.0148 | 3.0 |
| `x25_b0` | 0.1400 | 0.5399 | 0.0756 | 0.1103 | 0.0155 | 4.9 |
| `x25_b1.5` | 0.1400 | 0.1146 | 0.0160 | 0.0882 | 0.0124 | **1.3** |

The floor is a roughly **constant absolute error term (0.012–0.019 in every arm)** — so `rnull` grows
only because `A_t` shrinks, and it eats a fixed fraction of the weak bands' readings. The
`(25×, β = 1.5)` cell emits 1.3× the floor: essentially nothing. Its `q = 0.353` is therefore the
floor, not a calibrated shrink, and is the one cell whose value should not be read as a ratio to `r*`.

### 5. Contract A2: what survives training (10 000 steps)

Contract A2 is binding ("report the converged state and log the transient"), and P3 is the precedent
that this config does not converge in 2 000 steps on hard corpora. The transient changes the **step
count and nothing else** — corpus builder, amplitudes, phase walks, model, probe and readout are
imported from the parent script — and it is reported whatever it shows.

The 2 000-step rows reproduce the parent run's per-seed rows (same seed, same corpus, same data
order, and the parent evaluates only after training, so the trajectory is the same; checked against
`power_x_pred.json` at the printed precision). `q` at 2 000 vs 10 000 steps, seed 0 (the `ρ = 25`
cells at 3 seeds):

| arm | `q` @2 000 | `q` @10 000 | change | `r` @10 000 | oracle@8 @10 000 | reading |
| :-- | --: | --: | --: | --: | --: | :-- |
| `x1_b0` | 0.995 | 0.999 | +0.004 | 0.9988 | 1.000 | at optimum throughout |
| `x1_b1.5` | 0.936 | **1.017** | +0.081 | 0.3302 | 0.329 | at optimum at 10k |
| `x5_b0` | 0.955 | 0.991 | +0.036 | 0.9908 | 1.000 | at optimum throughout |
| `x5_b1.5` | 0.528 | **0.952** | +0.424 | 0.3092 | 0.328 | **under-training** |
| `x25_b0` | 0.540 | **0.958 ± 0.013** | +0.418 | 0.9582 ± 0.0135 | 0.996 | **under-training** |
| `x25_b1.5` | 0.353 | **0.438 ± 0.057** | +0.086 | 0.1424 ± 0.0186 | 0.321 | stable below optimum |

The 10 000-step 2×2 (row `ρ = 25` at 3 seeds; the other two rows one seed, the parent's 3-seed 2 000-
step spreads quoted in §1):

```
q:          beta=0.0          beta=1.5
rho=1       0.999             1.017
rho=5       0.991             0.952
rho=25      0.958 ± 0.013     0.438 ± 0.057      -> one cell off; rho=25 at 3 seeds, rest at 1
```

Three things follow.

1. **The 2 000-step verdict is mostly an artefact of the training budget.** Both low-power rows rise
   by +0.42 over 5× the steps, and the cell that looked most dramatic in §1 (`x25_b0`, a *fully
   predictable* band emitted at 54%) reaches **0.958 ± 0.013**, with all three seeds entering it
   (0.951 / 0.950 / 0.974) and every seed's 2 000-step value (0.509 / 0.493 / 0.618) rising
   monotonically to it. Reading §1's β = 0 row as a converged property of the learned map would have
   been wrong; contract A2 is what caught it, and P3 gave the warning.
2. **The curve shape in the first rows is not a learning-rate story that runs one way.**
   `x25_b1.5` *falls* from 0.87 at step 250 to 0.39 at 2 000 before rising to 0.44: the head first
   learns the strong bands and suppresses the weak unpredictable one, then partially recovers it.
3. **The interaction survives — in one row and one cell.** At 10 000 steps the β = 0 row is flat
   (spread 0.041, drop 0.041) while the β = 1.5 row falls 1.017 → 0.952 → 0.438 (spread 0.579). Applying
   the pre-registered rule to this configuration gives the same verdict as §2, but now with the sign
   of the effect isolated: **power and predictability are orthogonal in five of six cells, and the
   exception is the joint extreme** (weakest band *and* unpredictable), which is at 0.44 of its
   optimum. The *training gain* is itself power × predictability dependent: from 2 000 to 10 000 steps
   the β = 0 cell gains **+0.418 ± 0.046** and the β = 1.5 cell **+0.086 ± 0.011** (difference
   −0.333 ± 0.048, 7× its sd).

**The one off-optimum cell is not an under-fitting arm.** At 10 000 steps `x25_b1.5` is at **+0.9805**
variance explained against its analytic bound of **+0.9825** — 99.8% of it — with `held_var` exactly
0.5. The excess MSE over the bound is `(0.9825 − 0.9805) × 0.5 = 0.0010`, and the target band's
coherent power is `P_t·r*² = 0.009804 × 0.1054 = 0.00103`. **96% of the arm's remaining error is
exactly the coherent power of the band it under-emits** — the model is at the Bayes bound except for
withholding a band that carries 2% of the corpus power. The same ratio for the other five cells is
0.005 / 0.076 / 0.015 / 0.230 / 0.072, i.e. their residual error is mostly *not* that band. Nor is that
cell's reading its floor any more: by 10 000 steps it emits 0.0224 against a floor of 0.0071 (3.2×),
where §4's 2 000-step table had it at 1.3×. Whether it would reach `r*` given yet more steps is
**not settled** (it is still rising, +0.086 over 5 000 → 10 000 in all three seeds; 20k/40k is the
obvious follow-up).

*All numbers in §1–§4 are the pre-registered 2 000-step runs; §5 onward is the transient.*

### 6. The checks the brief asked for: both pass

| check | result |
| :-- | :-- |
| **Power axis is the band's total power, not its coherent power** | Target total power at β = 1.5 / β = 0: **1.0000 / 1.0000 / 1.0000** at ρ = 1 / 5 / 25 (realisations paired, so the ratio is exact to float noise); the coherent power over the same pairs falls to **0.1149**, i.e. the quantity I did *not* use is the one coupled to β. **Orthogonality verified numerically: the two axes are orthogonal by construction and measurement.** |
| **Probe shape-matched to each arm's corpus** | Probe/corpus per-band power ratio = **1.0000 at all three bands in all six arms**; corpus RMS = probe RMS = **0.70711** everywhere (contract A9's total match holds too, and is not doing the work). **Pass.** |
| `in-band` (leakage) | The corpus's cross-talk diagonal is **1.0000 in all six arms and all three bands**: integer-spaced carriers are orthogonal over a patch, so the matched filter at `b = 8` sees the `b = 8` tone and nothing else. On the probe side the whole-window ±0.5 mainlobe share is **1.0000 at every band and every arm**. The brief's "a low-power band is leakage-prone" failure mode is structurally absent here — measured, not assumed. |
| realised β (per band) | **1.4981** on the target (nominal 1.5), **0.0000** on both controls, in every arm — inverted from the corpus's unpredictable part. |
| corpus band-power range | **1.000× / 5.000× / 25.000×**, target share 0.3333 / 0.0909 / 0.0196. Against T2b's dose curve (617× → 7.89× decline, 81× → 5.37×, 3.1× → 1.12×) these levels straddle the low end of the collapse region. |

One extra generator measurement worth recording, since the whole design rests on "constant envelope ⇒
constant band power": in the **corpus**, the whole-window DFT power inside ±0.5 `b` of the target
carrier is **0.850×** of that band's total power at β = 1.5 (1.000 at β = 0), in all three power
levels — PM's sidebands at `Δb = 1` move 15% of the band's power out of the matched filter's mainlobe.
That is a whole-record statement the patch-aligned readout does not see (each patch holds a constant
phase, so `|Z_p|²/2` is untouched), but it is the honest bound on how "the same band" survives β. And
the target band's **marginal** retention (`rm@8`, its own held-out corpus, where `r*` is the correct
answer) tracks the probe reading closely in every arm, so the probe is not measuring something the
corpus does not.

### 7. What this does to P2b and T2b

- **P2b survives, and is sharpened at convergence.** With band power held fixed (ρ = 1) the clean-probe
  retention equals `exp(−β²/2)`: at 10 000 steps `r = 0.3302` against `r* = 0.3247` (`q = 1.017`); at
  2 000 steps the same cell reads 0.2784 (`q = 0.857`) where P2b's own readout was 0.3029
  (`q = 0.933`). Marginals and the oracle agree with that.
- **T2b's "retention tracks power" is reproduced in a controlled corpus — as an interaction with the
  training budget.** At the pre-registered 2 000 steps, a 25× corpus range costs 1.84× (β = 0) / 2.43×
  (β = 1.5) of the target band's retention on a *perfectly shape-matched* probe, i.e. with no possible
  probe-shape confound; at 10 000 steps the β = 0 cost is 1.05× and the β = 1.5 cost 2.07×, entirely
  from the single `(25×, β = 1.5)` cell. So T2b's driver is real, but on this evidence it is a
  **convergence-rate / optimisation allocation** effect of the band's power share, not a converged
  power-calibrated prior — and it is not the same object as P2b's predictability prior, which lives at
  `r*` in five of six cells.
- **This does not directly explain T2b's band-flat residual deficit** (`r/√P ∈ [0.45, 0.80]` at a 3.1×
  range). Here a 5× range costs the *coherent* band only 0.036 in `q` even at 2 000 steps. T2b's arms
  were also not at their Bayes bound (gate +0.72), so its residual is consistent with the same
  optimisation story, but that is an inference, not a measurement from this run.

### 8. Caveats, contradictions and gaps

- **The pre-registered criterion over-fires** (§2.1): as written it cannot separate "power modulates the
  predictability prior" from "power independently costs retention". I report the verdict it gives and
  the statistic that would have decided it, and I did not rewrite the criterion after seeing the result.
- **The `ρ = 1` row misses the pre-registered 0.03 replication tolerance** (gap 0.046 to `r*` at
  β = 1.5, against P2b's 0.022), even though it is 0.9 sd from P2b's own value. Every absolute level in
  §1 is read with that caveat; §5 shows the gap closes at 10 000 steps.
- **The 2 000-step configuration is not converged**, and I only found that by adding the transient. The
  brief fixed 2 000 steps (P2b's config) for comparability, which is defensible, but every `q` in §1 is
  a mixed quantity: part prior, part convergence speed. A converged 3-seed 2×2 at 10 000 steps is the
  experiment §1 *should* have been; I ran only one row of it at 3 seeds.
- **The transient is 1 seed except for the `ρ = 25` row** (3 seeds). Its conclusions are carried by
  effect sizes of 0.4 in `q` against 2 000-step seed sds of 0.07, but the 10 000-step *levels* for
  `ρ = 1` and `ρ = 5` are single-seed.
- **`rnull` is not a flat baseline** — it is a constant *absolute* error term divided by a shrinking
  `A_t`. The `(25×, β = 1.5)` cell sits 1.3× above it, which is why its `q` is reported with that
  warning rather than as a calibration.
- **A 25× range is what the brief asked for, and it is where the two drivers are least separable.** The
  single off-optimum cell is the compound extreme; nothing here says what happens at 100× or 500×.
- **The target band is one band at one `b`** (`b = 8`, `Δφ = 0`). P3 established flatness in `b` and P2b
  the diagonal, so this is not re-tested; `b = 8` is P2b's headline band.
- The brief's expectation "`r / r* ≈ 1` everywhere if the model is at its optimum in every cell" held —
  **at 10 000 steps** — in five of six cells, and failed at the pre-registered 2 000 steps in four. That
  is the substantive contradiction with the brief's framing, and it is a statement about this config's
  convergence, not about the design.

### 9. Reproduce

```
.venv/bin/python experiments/2_power_x_pred/scripts/power_x_pred.py                # 833 s
.venv/bin/python experiments/2_power_x_pred/scripts/power_x_pred.py --summarize
.venv/bin/python experiments/2_power_x_pred/scripts/transient.py                   # 1353 s
.venv/bin/python experiments/2_power_x_pred/scripts/transient.py \
    --arms x25_b0 x25_b1.5 --seeds 1 2 --tag s12                                   # 902 s
```

`power_x_pred.py` also takes `--arms` / `--seeds` / `--tag` for subsets (a partial run writes a
tagged JSON and prints "partial run: no verdict"). `ruff check` and `ruff format --check` are clean on
both scripts; `pytest tests/ -q` stays at 8 passed; no file outside this directory was modified.
