# T5 — the real-data power intervention: profiling, then a controlled balance

**The intervention.** Take the real corpus (Monash `australian_electricity_demand_dataset`), never
touch its own spectrum, and **add synthetic windows** until the corpus's *marginal* per-band power is
balanced. Then ask whether the target bands' retention improves at a fixed number of update steps.

**Why this ordering.** On real data we cannot control a band's conditional predictability, so the
first question is how much **power** alone contributes. That is exactly what the synthetic programme
measured (P8/P12/P13/P14/P15): power sets the learning *schedule*. This experiment asks whether that
schedule law transfers to a real corpus under a real probe.

## The setting, fixed before anything is trained

- **Normalization** — per window, statistics from the **context only**:
  `mu = mean(context)`, `sigma = std(context)`, applied to the whole `ctx + k` window. The model only
  ever sees normalized data, the loss is computed in that space, and nothing is denormalized.
  Context-only is the point: whole-window statistics would put part of the target's own amplitude into
  its normalization. This choice also matches the standing synthetic convention (every window has unit
  variance), so the schedule law applies without a conversion.
- **Aperture** — `k = 32`, `ctx = 16` patches, integer bands `b = 1..16` (periods 16 h … 1 h).
  Integer bands are required: at half-integer `b` the matched filter has no bin of its own and its
  reading is pure leakage (see the profiling entry). `k = 32` over `k = 64` because it gives twice the
  windows at the same measurable range; table below.
- **Split** — T1's random half-split of the non-overlapping windows, so the held-out half covers the
  whole period and is exchangeable with the training half by construction. **The probe is a real
  held-out draw of the training marginal**, so contract 9 is satisfied by construction — the T2
  probe-shape confound cannot arise here.
- **Metric** — `r_b = |Z(pred,b)| / |Z(observ|ed target,b)|`. On real data `r_b < 1` is *expected*
  (the target patch contains content no function of the context determines), so a deficit is only ever
  reported **relative to the per-band oracle**, `oracle_b − r_b`, never against 1.

## The three checks this design owes (agreed before the baseline)

1. **Split-seed replication** — T1's estimator takes a split seed; run 2–3 of them and confirm the
   per-band profile and `P(b)` do not move. This is the direct test of whether "iid" is doing any work.
2. **Per-half consistency** — the corpus is **non-stationary in level** (profiling: median 26 %
   relative difference in per-band power between the first and second halves of the windows). Compute
   every reading separately on the two halves of the held-out set; an effect that appears in one half
   only is reported as such.
3. **Blocked split as a separate, practical arm** — train on the first 80 % of the windows by time and
   evaluate on the last 20 %. There the probe is **not** exchangeable with the training marginal and
   the 26 % drift is a live confound, so this arm answers the *forecasting* question and is never
   mixed with the controlled one. Two purposes, two splits.

## Profiling (step 1) — done, `scripts/corpus_profile.py`

Four findings, all in the run log; the two that shape the design:

- **integer bands only** — at 0.5 spacing, half the grid is interpolation between DFT bins
  (`in-band` 0.09–0.90 against exactly 1.000 at integer bands);
- **no hard power floor** — all sixteen integer periods 16 h … 1 h are measurable, power falling
  smoothly by ~600×, with `in-band` = 1.000 at both apertures. `k = 32` (2 122 windows, horizon 16 h)
  and `k = 64` (1 061 windows, horizon 32 h) differ in **horizon and statistical mass, not in reach**;
- `P_own` spans only **1.5×** (0.63–0.92) while power spans **1664×**, and the estimator's null
  (white noise through the identical pipeline) reads **≈ 0.000**;
- the profile's **shape** is reproducible across halves (rank corr +0.98) while its **levels** drift by
  ~26 %.

## Baseline (step 2) — `scripts/baseline.py`

Trains the unmodified corpus and produces everything the intervention is compared against:
`MASE` (per-window, seasonal-naive denominator `m = 48` samples = 24 h) with persistence and
seasonal-naive baselines, `MSE`/`MAE`, the per-band `r`, `Δk` and phase-error **learning curves**
(checkpoints every 100 steps to 2 000, then every 500 to 10 000), the per-band oracle, and two
reference quantities.

**The model-free per-band rule.** For each band, a ridge from the *context's own coefficients at
that band* to the *target patch's coefficient at that band*, fitted on training windows and scored
on held-out ones. Its explained-variance fraction is `P(b)`; under the pooled-RMS convention the
retention a perfect linear predictor at that band achieves is `sqrt(P(b))`. It is a **baseline
predictor, not a strict upper bound** — it is linear in one band's own history and sees no other
band — but it says what a band is worth on its own.

**The headroom**, `(MSE_model − MSE_oracle) / MSE_model`, where the oracle is the same kind of ridge
but read out from the frozen representation. Fraction of the error a perfect head on the frozen
representation removes. No retention enters it, so it stays meaningful at the bands where `r > 1`
(the emission floor). If it is small the intervention is cosmetic, which is why it comes first.

Results are in the run log. Two readings that decide what the intervention can even ask:

- the model-free rule reads `sqrt(P(b))` = 0.82–0.96 with **no trend across the sixteen bands**,
  against a null (white noise through the identical pipeline) of −0.001. Power spans 537× and
  `P(b)` spreads only 1.5×, so on this corpus the bands differ in *how much energy they carry*, not
  in *how recoverable* they are;
- `res_b / floor_b`, the model's per-band residual against what that rule leaves (`floor_b =
  (1 − P(b)) * P_b`), starts far above 1 at the weak bands and **closes as the budget grows**: at
  10 000 steps 15 of 16 bands are at or below 1, only b16 is left at 1.4. Under the blocked split
  the model still *loses* to the rule at b = 2–7. The headroom is therefore a **transient**, i.e. a
  property of the budget and not of the band — which is exactly what makes the schedule the thing
  left to measure.

## Intervention (step 3) — `scripts/intervene.py`, frozen before the run

The real windows are never touched. To the real training half (1 061 windows) every synthetic arm
adds the same 1 061 synthetic windows (half as many as the real ones), built by
`fbias.data.make_pm_profile` so that each band's power and its *conditional predictability* are
separate knobs: band `b` gets a chosen mean power and a phase-diffusion rate `β_b`, and the fraction
of that band any function of the context can recover is `exp(−β_b²)`. Setting
`β_b = sqrt(−ln P(b))` from the corpus's own measured `P(b)` reproduces it: the calibration check
inside the run recovered 0.67–0.91 against targets of 0.68–0.92, every band within 0.03 and
systematically low by a median 0.015. **Only power moves.**

Every synthetic window is given the real corpus's *total* matched-band power (0.686 per window), so
no arm can win by carrying more energy than the real windows — only by putting it in different
bands. Fraction `λ` of that budget is spread equally over the bands the real corpus leaves below the
equal-split level (bands 3–15), the rest keeps the real profile:

| arm | λ | synthetic marginal | corpus marginal range |
|---|---|---|---|
| `real` | — | none | 537× |
| `dilute` | 0.0 | the real profile exactly | 542× |
| `bal_mid` | 0.4 | real profile blended halfway to flat | ~38× |
| `bal_hard` | 1.0 | flat over bands 3–15, nothing at 1–2 | ~10× |

`dilute` is the control: it doubles the corpus exactly as the `bal_*` arms do with the marginal
held fixed, so `bal_* − dilute` is the marginal's effect and not "more data"'s. Note what the
manipulation is and is not: the balance arms carry **no band-1 content**, so they halve the
loudest band's marginal while lifting the quiet ones. It is a flattening, not a selective raise.

**Probe.** The real held-out half (1 061 windows), and the same `floor_b` in every arm — the floor
is a property of the probe, not of the training corpus, so `res/floor` is comparable across arms by
construction.

**Pre-registered before the run:**

1. the manipulation landed — each `bal_*` corpus marginal is flatter than `dilute`'s and its quiet
   bands carry far more power;
2. `dilute` matches `real` band by band within seed noise — extra windows of the same marginal buy
   nothing;
3. in every band a `bal_*` arm lifts, `res/floor` declines **faster** than in `real` and `dilute`,
   and the ordering of decline rates across arms follows the ordering of that band's corpus share;
4. at 10 000 steps the arms agree — power writes a schedule, not a ceiling.

**Amendment, recorded before any arm result was read** (the run was already training; the only
output seen was the 100-step smoke that produced this design). Check 2 as first written was wrong,
not merely optimistic: doubling the corpus with windows from a *different source* leaves the real
training windows at half weight, so `dilute` is expected to be **worse** than `real`, not equal —
that is dilution, and it is a confound the additive design cannot remove. The check as it should
have been written, and as it is now read:

2. dilution is not a free lunch — `dilute` is *no better* than `real` at any band;
3. at the bands a `bal_*` arm lifts, `res/floor` declines faster than in `dilute` — which holds the
   corpus size, the number of real windows and the synthetic windows' total energy fixed, so it is
   the marginal's effect alone. Whether a `bal_*` arm also beats the undiluted `real` is reported
   band by band rather than predicted, because that contrast mixes the marginal's gain with
   dilution's cost;
4. unchanged.

Falsification: a band failing (3) falsifies the transfer to real data; a corpus failing (2) is not a
controlled comparison and is reported as such.

**Known limits.** b16 (Nyquist) is excluded from every statistic, as in P12/P13: a real tone there
is its own negative-frequency image, so the matched filter reads twice its power. The synthetic
windows are phase-diffusing tones, so they match the real corpus band by band in power and in
conditional predictability but not in any structure beyond that; the `dilute` arm bounds what that
is worth. And a null on the real probe would not distinguish "power does not schedule real data"
from "the added windows are too unlike the real ones" — a corpus-matched synthetic probe is the
follow-up that separates them.

## Horizon profile (step 5) — `scripts/horizon_profile.py`

Model-free: for each horizon `h` the band-only ridge behind `P(b)` is refitted with the target moved
`h` patches past the context, over the same 1 126 non-overlapping windows at every `h`. Horizons
1–16 patches (the target's last sample 16–256 h past the context). Three corpora: the real one, a
phase-diffusing `pm` corpus whose predictability is **analytically** `P_1(b)**h` (its phase walks by an
independent `beta eps` per patch boundary), and a white-noise `null` through the same pipeline.

The question is whether a predictability gradient *appears* with the lead, since the mechanisms that
are supposed to make high frequencies hard — turbulence's cascade, autocorrelation decay,
microstructure noise — are all statements about decay with lead time.

**Answer: no.** The real corpus's `P_h(b)` is flat in both `h` and `b` (span across bands 1.12 / 1.20
/ 1.19 / 1.12 / 1.10 at h = 1/2/4/8/16; Spearman with band −0.175 / −0.379 / −0.232 / −0.268 /
+0.025), so the one-patch flatness is not a horizon artifact. And the corpus decays *far slower* than
the memoryless benchmark — `P_h / P_1**h` reaches 4.7× (b1) to 92× (b8) at h = 16, with the 16 h band
still 88 % predictable at a 128 h lead. Quasi-periodic demand simply has no fast-decorrelating
high-frequency component for those mechanisms to act on. Checks: null `|P|` ≤ 0.0266; the PM benchmark
reproduces `P_1**h` to a mean absolute deviation of 0.0155 over the 50 cells above the estimator's
floor (worst 0.068 in one cell, against a pre-registered max tolerance of 0.03 that is reported as
missed rather than rewritten).

## Status of step 4 (pure tones): check 3 fails, in the opposite direction

4 arms x 4 seeds x 10 000 steps. The manipulation is exact -- dose 0.0944/0.0921/0.0918 at the tone
bands, every non-tone band at 0.909x `real` identically in all three tone arms, and the largest
|delta| between tone arms is **0.017** at the non-tone bands against **0.301** at the tone bands.

**Check 3 fails, and `real` is the minimum in every column.** Each tone arm is *worse* at its own band:
paired seed-matched contrasts +0.057 ± 0.023 (b6), +0.211 ± 0.040 (b10), +0.301 ± 0.061 (b14). Check 4
also fails -- the diagonal is 0.547 / 0.834 / 0.985 against a dose spread of 2.9 %. The effect is
sharply band-specific and present from the first checkpoint.

The mechanism is visible: the tone is learned **perfectly** (`own` probe MSE 0.0006-0.0008 at 10k,
against 1.757 for `real`), the band's real-profile content is *not* helped (`mimic` probe 5.7/11.0/13.8
against 4.6/5.9/6.5), so the band's output is specialised to the tone at the real content's expense.
Interference, not a schedule.

**Design flaw, owned.** With a fixed absolute tone power the tone's dominance over the band's real
content is graded 24x/67x/90x, so the arms differ in band *and* relative dose and check 4 is confounded.
Equal *absolute* dose gives equal share but graded dominance; equal *relative* dose gives equal
dominance but graded share. **Pure tones cannot deliver a same-share/different-frequency test.**

**What it settles.** Steps 3 and 4 agree, from two constructions: adding synthetic energy at a band does
not buy that band on real windows. And it forces the scope statement -- the share law is an
*in-distribution* law, and it locates the deficit without licensing "add data to remove it".

## Correction (step 5): what `in-band` does and does not measure

Reading step 5 exposed a measurement error of the profiling step. Zeroing every window-DFT bin below
band 1 — the 24 h cycle, the weekly cycle, all trends — and recomputing the matched power leaves
**0.245 at band 1** and **0.60–0.91 at bands 2–15**: band 1's reading is three-quarters sub-band-1
leakage. The profiling's `in-band` column reads exactly 1.000 at integer bands and cannot see it,
because it band-limits each *patch* to `|f − b| ≤ 0.5` cycles/patch, which at integer `b` keeps exactly
the bin the matched filter reads — a tautology, and the sub-band-1 continuum has no bin of its own.
`in-band` remains meaningful at half-integer `b` and across apertures; "1.000" must not be read as
"no leakage".

It does not change the story, and that is measured rather than argued: the same band-only ridge on the
high-passed corpus gives `P` = 0.77–0.99 (span **1.29**, Spearman −0.66) against 0.72–0.90 raw (span
1.27, Spearman −0.26). Removing the continuum *raises* `P`, so it was not inflating the flat reading.
The corrected power span is 546× → **170×**. Power 170× against predictability 1.3× — the direction
account stands.

## Status — the intervention ran, and it does not settle the question

`scripts/intervene.py` ran at 10 000 steps × 3 seeds; results in the run log, and the harness check
inside it (`real` bit-identical to the baseline) passed. **Check 3 failed**: on the thirteen bands
`bal_hard` lifts, `res/floor` is worse than `dilute` at 13 of 13, and the crossing step is later at
eight of them. Check 2 passed one-sided.

Two things follow, and both are about the design rather than the law. First, *no band is left
unmanipulated* — adding power where the corpus is quiet means the weak bands and the manipulated
bands are the same set, so this run cannot separate "weak bands are hard" from "weak bands were
manipulated". Second, the treatment was not neutral: bands 3–15, the ones the synthetic profile
touches, are worse than `dilute` by 1.13–1.51× while b1 and b2 (1.03× and 0.96×) — the two bands it
does *not* touch — are the two that do not move, and at the top bands the change is amplitude up,
phase down. The added power looks like interference with the real-window band, not a schedule.

The corpus loss share, the quantity P13/P15's law is written in, was **not** measured: it needs the
model's residual on the synthetic windows and no checkpoints were saved.

**The follow-up this points to** is a *single-band boost*: synthetic profile = the real profile plus
a bump at one band only, so the corpus marginal moves at exactly one band and the other fourteen are
an internal control in the same arm over the same corpus size and window type — plus a corpus-matched
synthetic probe, which the pre-registration already listed as the diagnostic that separates "power
does not schedule real data" from "the added windows are too unlike the real ones". That is a new arm
set, not a rerun of these.

## Intervention, second form (step 4) — `scripts/intervene_tone.py`, frozen before the run

Step 3 lifted the bands that *are* the weak bands, so "weak bands are hard" and "weak bands were
manipulated" were the same contrast. The first repair tried here was a **single-band boost**: keep
the synthetic windows as real-profile mimics and add a bump at chosen bands, so the corpus marginal
moves at exactly those bands and every other band is an exact internal control. **That design was
abandoned before it ran**, because it is not the intervention anyone would actually perform: a
practitioner adding synthetic data to a red corpus adds **pure tones**, not another realisation of the
real spectrum with a bump on it.

The pure-tone version is what runs here, and it is why the design changed shape.

```
synthetic window = one clean tone at a single band (beta = 0), nothing elsewhere
one tone window per ten real windows; each tone carries power 1.0
```

| arm | synthetic profile | corpus marginal |
|---|---|---|
| `real` | none | the real corpus |
| `tone_b6` | a tone at b6 (2.67 h) | 0.1034 at b6 (24× real), exactly 0.909× real everywhere else |
| `tone_b10` | a tone at b10 (1.60 h) | 0.1012 at b10 (67× real), exactly 0.909× real everywhere else |
| `tone_b14` | a tone at b14 (1.14 h) | 0.1009 at b14 (90× real), exactly 0.909× real everywhere else |

**The dose is deliberately gentle.** An earlier draft used half a corpus of tone windows each carrying
0.686, which puts 0.344 at the tone band — *above* band 1's 0.273, i.e. it inverts the corpus's
spectrum instead of flattening it. One tone window in ten, each at unit power, moves the corpus's
total matched-band energy only from 0.686 to 0.718 (+4.6 %) while taking its marginal from a 537×
range to **5.9×** — the flattening the intervention is actually for.

**The price, stated up front.** A pure-tone window has no energy at the other fifteen bands, so mixing
them in dilutes every other band's marginal. There is no arm in which some band is untouched. That is
inherent to "add pure tones" and it is why step 3 could not be read.

**What replaces the missing control.** The dose a tone arm puts at its band is the same whichever band
it is — 0.1034 / 0.1012 / 0.1009 for b6 / b10 / b14, equal to within 2.5 % in absolute power **and**
in share (0.0809 / 0.0791 / 0.0789). The three tone bands therefore *end* the intervention with the
same share even though their real shares differed by 3.8×, and the question is whether their retention
ends up equal too. And the twelve non-tone bands sit at exactly 0.909× real, *identically* in all
three arms, so they are a null control **among the tone arms**. `real` is *not* a control there (it is
not diluted), only the anchor.

Equal share, three frequencies, twelve identical control bands: the project's central question in
its cleanest available form, *is it the share or the band?*

**Probes, all held out.** `real` (the real held-out half, the primary reading); `mimic` (a fixed
512-window synthetic corpus from the real profile, never trained on — does anything transfer to a
synthetic window whose band content is the real corpus's?); and `own` (a fresh draw from each arm's
own corpus, read as a whole-window MSE — a pure tone is fully predictable, so it should fall to ~0;
if it does not, the intervention is broken and nothing else can be read).

**Pre-registered before the run:**

1. the manipulation landed — the tone arms' corpus marginals agree with each other at every non-tone
   band to the last digit, sit at 0.909× `real` there, and carry the same dose at their own band
   (within 2.5 % in absolute power and in share);
2. internal control — at the twelve non-tone bands the three tone arms agree with *each other* within
   three seed-paired sd, on the real and mimic probes;
3. **the diagonal** — in the 3 × 3 table of res/floor at (arm, tone band), each column's minimum is
   the arm that carries the dose at that band, on the real probe;
4. **equal share, equal effect** — the three diagonal values agree with each other within seed noise,
   i.e. the improvement is predicted by the corpus *share*, which the intervention equalised, and not
   by the frequency. A systematic gradient along the diagonal is a residual frequency effect beyond
   share, and is reported as such.

**Caveats recorded before the run.** The tones are pure (`beta = 0`), so the added band is also made
fully *predictable*; a positive result is therefore not attributable to power alone, and a
matched-beta control arm (same dose, `beta_b = sqrt(-ln P(b))`) is the follow-up — it only needs
running if this one moves. And the across-seed spread of `res/floor` reaches 0.10–0.13 at the weakest
bands, which is why the reading is a *seed-paired* contrast (the arms share each seed's init and its
real-window batch schedule, only the synthetic half differs) over four seeds, not a single band's
number.

## Status

Steps 1–3 are reported above (profiling, baseline, and the balance intervention that could not settle
the question). Step 4, the pure-tone version at one tone window in ten and unit tone power, is the run
in flight; `scripts/intervene_boost.py` (the mimic-plus-bump draft) never ran and has been removed.
