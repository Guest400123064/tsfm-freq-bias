> **Status:** earlier draft. It predates the current three-axis programme and is retained
> for reference only. The active plan is `.agents/PLAN.md`; current positioning,
> related work, and the working storyline are in `docs/README.md`.

# "Frequency bias" is two biases

*What a time-series foundation model is actually doing to a signal's spectrum, and why the word
"frequency" keeps hiding it.*

---

## 0. Two words, six meanings

"Frequency bias" is two words and at least six objects. In the time-series forecasting literature it
usually means one of two of them, and they are not the same thing:

1. **the frequency content of the signal** — the Fourier spectrum of the series you feed in; and
2. **the frequency content of the map** from history to future — the object the classical
   *spectral bias* literature is about.

The confusion is easy to state and easy to miss:

> A **high-frequency pure tone** is high-frequency in sense 1, but the map that generates it is
> **trivial** — it is a rotation. Conversely a *low-frequency* signal can require a hard map.

So "the model handles low-frequency signal content better" does **not** follow from "the map is
biased toward low-frequency functions". Rahaman et al. (2019) and Xu (2020), the papers usually cited
here, are about sense 2, in a setting where **the target is in the training set** — function fitting
on a fully observed domain. Forecasting is a different regime: the target is held out, the conditional
distribution has irreducible spread, and *shrinking toward the conditional mean is the correct
answer*. That difference is not cosmetic; it is why the classical framework has no cell for the
phenomenon the field calls frequency bias.

"Bias" hides a second split, orthogonal to the first: is it about **how fast** a band is learned (a
learning-dynamics statement, removable with more training), or about **how much survives at
convergence** (a fixed point)? Much of the disagreement in the literature is two papers reporting one
of each and both calling it "the frequency bias".

This post is about the gap that split leaves open. To our knowledge no prior work studies a
*converged, non-removable, data-distribution-driven* frequency bias in forecasting. The nearest
misses are precise: Basri et al. (2020) makes the data distribution a genuine variable but is a
**rate** result; Bietti & Mairal (2019) and Yu et al. (2025a) make non-removable claims but attribute
them to **architecture** and **initialisation**; Fredformer (Piao et al., 2024) is the only prior work
in the forecasting setting, and it attributes the bias to band **energy** and treats it as
**removable** — which is the point of its remedy.

Our claim, after a lot of controlled experiments, is that the persistent high-frequency suppression
everyone observes is **not** an architectural defect, and that it is set by **two properties of the
training distribution**:

- the **marginal power** of each band, which sets the **learning schedule** — how fast that band is
  learned; and
- the **conditional predictability** of each band, which sets the **fixed point** — how much of it
  survives at convergence.

And that on real corpora these two are almost always tilted the same way, which is exactly why the
field has been able to read the whole thing as a property of *frequency*.

---

## 1. What has to be controlled, and what we control it with

**The task.** Forecasting only. The target patch is held out; the model must generalise. We report
"frequency" in **cycles per patch**, `b = f · k / ctx`, with the phase advance `Δφ = 2πb` stated
alongside, because that is the variable the model actually operates on. The tokenizer is the
transducer between the two senses above: **patching is what creates frequency selectivity at all** — with `k = 1` the model is frequency-blind and
frequencies at `f = 16`, `32`, `64` give identical geometry (plane overlap 1.000).

**The model.** A deliberately small, patched, input-space next-patch Transformer we call `SimTFM`:
no RevIN, no SIGReg, RoPE, trained on `context_size + patch_size` windows. It is an instrument, not a
production model. Everything here is a statement about *how a model with a frequency axis behaves
given a data distribution*; the production-model question is §4 and §5.

Three design choices carry most of the weight, and we got two of them wrong first.

**Choice 1: the probe must be clean.** Shrinking an unpredictable band is *correct* MSE behaviour.
A learned prior is only visible on an input where shrinking is still wrong. An in-distribution probe
cannot distinguish "the model hedged appropriately" from "the model carries a prior". This is the
single most load-bearing decision in the project, and our very first experiment failed for exactly
this reason.

**Choice 2: additive noise is not a usable instrument.** Our first attempt to make a band "noisy"
added band-limited i.i.d. noise on top of a persisting tone. It produced no shrinkage in any arm —
correctly. The context integrates over ~16 patches, so the effective in-band SNR was ≈ **0.011**, and
`r ≈ 1` was the right answer. What the experiment *did* produce was gain *inflation* (1.0005 → 1.0428),
about 13× the seed spread, in the direction opposite to shrinkage. The lesson generalises: **whether
your noise can be averaged away determines whether it is an instrument or a control.** §3 is about
which noise does which.

**Choice 3: reduce the quantity under test to a closed-form optimum.** If a corpus's optimum is known,
the reading stops being "bigger or smaller" and becomes "does the model sit at the optimum". We get
that from **per-patch phase diffusion**: the carrier's phase advances by `2πb` plus an independent
`β·ε` at every patch boundary. The envelope is constant, so total power is fixed at every `β`; future
increments are genuinely not a function of the past; and the Bayes-optimal retention against the clean
target is

```
r* = exp(−β²σ²/2)   →   exp(−β²)
```

Three metrics come out of this: **retention** `r` (matched-filter amplitude against the **clean**
target — so `r = 1` means "the deterministic component was recovered exactly"), **`Δk`**, Fredformer's
relative error `|r e^{iΔφ} − 1|` (which carries the phase too, so we report it beside `r`, never
alone), and on real data a **model-free per-band rule**: a ridge from the context's own coefficients
at a band to the target patch's coefficient at that band, fitted on training windows and read on
held-out ones. That last one is what lets us talk about a deficit without a model in the loop.

[FIGURE 1] *The instrument. Left: a clean multi-tone probe. Right: the same probe under per-patch
phase diffusion — constant envelope, so total power is unchanged while the coherent part falls.*

---

## 2. The power account: share sets the schedule

**Question.** Does a band's *share of the spectrum's energy* change how much of it the model keeps,
when nothing else varies?

**Setting.** Fully coherent multi-tone corpora — deterministic, so the Bayes optimum is `r* = 1` at
every band and any shortfall is a *failure to fit*, not a correct shrink. The knob is one band's share
of a fixed total power; the other bands are held equal; and **the weak band rotates** between `b = 2`,
`4`, `8` across arms, which is what separates "power" from "this particular frequency".

**Result 1 — share is the sufficient statistic.** Ten structurally different spectra (red and violet
power laws, random log-normal profiles, random bumps, flat), in which the loudest band lands on
`b = 1, 5, 7, 11` and `16` across the set; 150 pooled band-arm points. A single **monotone** curve in
loss share predicts retention with **R² = 0.96**, leaving a band-free residual of sd **0.012**. Adding
band index as a second variable does not help: inside equal-count share bins, the correlation between
retention and band index is ≤ 0.48 and the within-bin spread of retention ≤ 0.09.

**Result 2 — the deficit follows the manipulation.** Rotate the power ratio to a different band and
the deficit moves with it. It is not "this frequency".

**Result 3 — it is a schedule, not a ceiling.** At 2 000 steps the ordering by share is sharp and the
retention range is wide; at 10 000 steps everything has converged toward `r* = 1` and the same
statistics get noisy — between the two budgets the retention range compresses 2.7× (0.157 → 0.058)
while the residual only shrinks 1.7× (0.021 → 0.012), so the same absolute scatter is a larger share of
a smaller range. The quantity is **transient**, and every statistic in this project needed a stated
budget window before it meant anything. Relatedly, a band's time-to-80%-retention scales approximately
as `1/share`.

**Result 4 — and it lives in the head, not the representation.** A ridge readout on the model's
**frozen latents** is *flat in power*: over a 25× power range the oracle reads 1.000 / 0.999 / 0.997.
The representation has the band; the head has not got there yet. So a low-share band is learned
*slower*, not learned *less*.

**What this buys.** You can put the deficit wherever you want. Raise the marginal power of a band and
that band's retention climbs; lower it and the deficit appears there. We can make the "frequency bias"
appear at a low frequency, at a high frequency, at two frequencies at once, or nowhere. That is the
sense in which it is a property of the data and not of the architecture.

[FIGURE 2] *The share collapse. Retention against loss share, one point per band per arm, ten spectra,
150 points, one monotone curve, R² = 0.96.*

---

## 3. The noise account: which noise sets the fixed point

The intuitive story for a high-frequency bias is "high frequencies are noisier". True, but "noisy" is
not one thing, and the difference decides whether the model *should* damp the band.

**Amplitude noise is integrable.** If a persisting tone is buried in additive noise, the context can
average across patches and the effective SNR grows with the number of patches. This is why our first
experiment was a null (§1, choice 2): with ~16 context patches the in-band SNR was ≈ 0.011 and reading
`r ≈ 1` was correct. A model that *should* recover the band, does.

**Phase noise is not.** A per-patch phase increment is never averaged away: the target patch's phase
is genuinely not a function of the past, and no amount of context fixes it. That is the irreducibility
we need, and it is what the `β` instrument isolates.

**The model sits at the optimum.** Sweeping the phase-walk depth:

| `β` | `r* = exp(−β²)` | clean-probe `r` | `r − r*` |
| --: | --: | --: | --: |
| 0.0 | 1.000 | 1.0027 ± 0.0031 | +0.003 |
| 0.5 | 0.882 | 0.8764 ± 0.0127 | −0.006 |
| 1.0 | 0.607 | 0.6019 ± 0.0139 | −0.005 |
| 1.5 | 0.325 | 0.3029 ± 0.0129 | −0.022 |
| 2.0 | 0.135 | 0.1157 ± 0.0143 | −0.020 |

Every gap to **1.000** is at least 10× the seed spread; every gap to **`r*`** is ≤ 0.022. The model
is not failing to learn the band — it is implementing the conditional mean of the distribution it was
trained on. **That is not a defect, and it is not removable by more training, because it is not an
error.**

Three corollaries, each of which kills a version of the "frequency" story:

- **Position does not matter.** Move a modulated line anywhere in `b = 3…13`, at three modulation
  depths, and the deficit does not move: `|Δr| ≤ 0.008`, `|Δ(Δk)| ≤ 0.036`. The model does not care
  where in the spectrum a line sits, given its share.
- **Dimension does not matter (as far as we can push it).** Four arms with the *same* per-band shares
  to six decimals, differing only in the rank of the state generating them (1, 2, 4, continuous), give
  one curve: convergence values 0.988–0.995 with no ordering by rank, time-to-threshold ratios ≤ 1.14.
- **The head is uniformly below, and the representation is at `r*`.** On the 8-band version the head
  sits at `r/r*` = 0.87 / 0.74 / 0.48 / 0.55 with **no trend in band** — a uniform optimisation gap
  caused by our own decision to perturb all eight bands at once, and the frozen-latent oracle reads
  `r*` exactly.

**So of {share, position, dimension, predictability}, only share and predictability enter the model's
behaviour — and in different roles.** Share writes the schedule; predictability writes the fixed
point. Two other things the literature's frequency story predicts to matter, do not.

**Where this is weak, said plainly.** "Conditional predictability" and "conditional function
complexity" are not the same thing, and our handle on the second is thin: rank 1 vs rank 2 via phase
structure is a narrow family of task complexities, and it is the one knob we have. Real high-frequency
content may be complex in ways a phase-modulated tone cannot express — heavy tails, intermittency,
regime switches, unobserved covariates. We do not claim to have separated complexity from
predictability; we claim only that the *one* complexity knob we can turn cleanly does not move
retention at fixed share.

---

## 4. Real data: why the field still sees it

If the bias is a property of the training distribution, why do production models keep showing it? We
profiled a real corpus — Monash's `australian_electricity_demand`, 30-minute sampling, 1 155 264
points, non-overlapping windows — under the same band definitions, and the answer is in the numbers.

**The two candidate asymmetries differ by two orders of magnitude.**

| | span across 15 bands | Spearman with band index |
| :-- | --: | --: |
| marginal **power** | **546×** | **−0.989** |
| band-only ridge **predictability** | **1.35×** | −0.396 |

And the 1.35× is not even a gradient: it is a two-step drop out of bands 1–2; from `b = 3` to `b = 15`
the predictability is **0.732 ± 0.026**, flat.

**The deficit follows share, and only share.**

| | vs. band index | vs. log share | vs. predictability |
| :-- | --: | --: | --: |
| per-band deficit (`res/floor`) | **+0.911** | **−0.914** | −0.225 |
| retention | +0.721 | −0.714 | −0.004 |
| share of the loss | −0.982 | +0.964 | +0.396 |

Note the first column: on this corpus, "frequency" and "share" are the same variable to three
decimals (−0.989 collinear). The field's reading is not *wrong* about the data — it is naming the axis
by its proxy. And the second and third columns say predictability is neither.

**It is not a horizon artefact.** Everything above forecasts one patch (32 samples, 16 h). The TSFM
literature forecasts 96–720 samples, and the mechanisms that are supposed to make high frequencies
intrinsically harder — turbulence's scale-local cascade, autocorrelation decay, microstructure noise
in markets — are all statements about *decay with lead time*. They predict a predictability gradient
that grows with the lead. We swept the target from 1 to 16 patches out (lead up to 256 h): the span
across bands is 1.12 / 1.20 / 1.19 / 1.12 / 1.10 and the rank correlation −0.18 / −0.38 / −0.23 /
−0.27 / **+0.03**. Nothing emerges. The corpus also decays far *slower* than a memoryless per-patch
process would — `P_h / P_1^h` reaches 4.7× to 92× at `h = 16` — because quasi-periodic demand has no
fast-decorrelating high-frequency component for those mechanisms to act on.

**So why do production models still show it?** This is the weakest link in our story and we do not
claim to have closed it. Three resolutions are consistent with what we measured, and only two of them
are about training length:

- **Distribution shift.** Our *controlled* split closes its headroom with training — at 10 000 steps
  the model is at or below a trivial per-band linear rule at 15 of 16 bands. The *practical* split does
  not: train on the first 80 % by time, evaluate on the last 20 %, and the model still loses to that
  same trivial rule at `b = 2…7`, with MASE 0.94 against 0.60. A temporal split — which is what the
  field usually reports — is a regime where the deficit does not close with more steps.
- **The schedule is set by the share *ratio*, not by the step count.** A larger, more heterogeneous
  corpus has a wider per-band ratio and a wider ratio stretches the schedule in a way no fixed budget
  repairs. "Trained for far more steps" is not the same as "trained past the transient".
- **The metric mixes amplitude and phase.** Fredformer's `Δk` is `|r e^{iΔφ} − 1|`, so a pure phase
  error reads as a frequency deficit. In our own balance arms we saw amplitude retention *improve*
  (0.926 → 1.022) while `Δk` got 20 % worse (1.374 → 1.667). A `Δk`-based frequency profile cannot
  tell those two apart.

**And we have no successful intervention.** This is the part we would most like someone else to
solve, and we report our two attempts because they both failed in informative ways.

*Attempt 1 — balance the marginal.* Keep the real windows untouched and add synthetic windows until
the corpus's marginal is flatter. The synthetic windows were built so that their per-band power **and**
their per-band conditional predictability both matched the real corpus, so only power moved. The
result: on the bands the balance lifted, the per-band deficit was worse than the dilution-matched
control at **13 of 13**, and the undiluted real corpus beat everything everywhere. The design also
cannot identify: adding power where the corpus is quiet makes the lifted bands *be* the weak bands, so
"weak bands are hard" and "weak bands were manipulated" are the same contrast.

*Attempt 2 — add pure tones at the target band,* which is what a practitioner would actually do: one
synthetic window in ten, each a single unit-variance tone, so the target band's corpus power rises
~25–90× while every other band moves identically in all arms. The result was the **opposite** of the
prediction:

| tone at | `real` | the tone arm | seed-paired contrast | crossing step, `real` vs tone |
| --: | --: | --: | --: | :-- |
| `b = 6` | 0.469 | 0.547 | **+0.057 ± 0.023** | 200 vs 300 |
| `b = 10` | 0.596 | 0.834 | **+0.211 ± 0.040** | 500 vs 1400 |
| `b = 14` | 0.668 | 0.985 | **+0.301 ± 0.061** | 800 vs 1700 |

(values are the per-band deficit; lower is better). `real` — the arm with **no synthetic data at all**
— was the best arm in every column. The mechanism is visible: the tone is learned **perfectly**
(whole-window MSE on a fresh draw from the arm's own corpus falls to **0.0007**, against 1.757 for
`real`), while the band's *real* content gets worse. The model's output at that band is specialised to
the tone at the real content's expense. Interference, not a schedule.

The honest reading of both: the share law is an **in-distribution** law. It was established on corpora
where the probe and the corpus are the same distribution. Step off that footing and the model's
per-band output is one function serving a *mixture*, and the synthetic component — easier, and
carrying a share of the loss — takes it over. **The share account locates the deficit; it does not
license "add data to remove it".**

[FIGURE 3] *Real-corpus profile: power is monotone red across two orders of magnitude; predictability
is flat.*

[FIGURE 4] *The failed intervention: each tone arm is worse than the no-synthetic arm at its own band,
and the harm is sharply band-specific.*

---

## 5. Limitations, and what we would like the field to take away

**Limitations, in the order a reviewer would raise them.**

- **The data-generating processes are controlled to the point of being unrealistic.** That is the
  trade: it is the only way to attribute a change, and it is also why our biggest results are negative
  ones on real data. The two are the same coin.
- **No RevIN, small model, one scale.** `SimTFM` is RevIN-free and small. RevIN is load-bearing for
  the representation's geometry, so a RevIN twin is mandatory before any claim about production TSFMs.
  Model size is a variable, not an upgrade.
- **Phase diffusion is an instrument, not a data model.** Unbounded phase growth requires an
  autonomous limit cycle; a stationary narrowband process has random but *non-diffusing* phase. Our
  constant envelope is a construction. We use it because it isolates the property under test and has a
  closed-form optimum.
- **"Not architectural" must be read precisely.** Patching *does* create frequency selectivity. What we
  refute is that the converged *suppression* is an architectural defect.
- **The complexity axis is under-controlled** (§3).
- **No real-world support for the predictability account.** Our corpus's predictive profile is flat at
  every horizon we tested, so we cannot say the direction comes from predictability. We can only say
  the direction is consistent with power and that predictability does not supply it *here*.
- **A caution for anyone profiling real spectra.** Our own leakage column read exactly 1.000 at every
  integer band, and it was a tautology — it band-limits each *patch* to the band, which at integer
  bands keeps exactly the bin the matched filter reads. Measured directly, band 1's power in that
  corpus was **75 %** leakage from below band 1. The direction account survived (power 546× → 170×),
  but anyone reusing the metric should know it cannot see the continuum.

**What we would like the field to take away.** Every practical implication here is about *which*
frequency and *what* bias you mean.

- **When you say "frequency", say which object.** Signal spectrum and map spectrum are not
  interchangeable, and a claim about one is routinely cited as the other.
- **When you report a frequency bias, say whether it is a rate or a floor, and report the budget.**
  Most of the statistics here are transient. Our own corpus closes its headroom with training: at
  10 000 steps the model is at or below a trivial per-band linear rule at 15 of 16 bands.
- **Control share and predictability before attributing anything to the architecture.** Report both
  per band, not a band index. In a red corpus they are collinear with band index and with each other,
  and telling them apart needs a corpus where they disagree.
- **The same prior that looks like a defect on one corpus is correct denoising on another.** A band
  that is noise *should* be shrunk; the "bias" and the "denoising" are one mechanism evaluated under
  two data distributions. That symmetry is why the literature can report both directions from the same
  model.
- **Check whether your frequency metric is measuring amplitude.** `Δk` carries the phase error; a model
  with the right amplitude and the wrong phase still reads a deficit.
- **Where the levers actually are.** If share writes the schedule, then dataset construction that
  balances per-band power changes learning dynamics, and per-band loss weighting is a way to spend the
  budget where you want it — whereas architectural changes that do not alter the effective per-band
  gradient have less to act on. Our failed interventions are a warning that the obvious version of the
  first lever does not work on its own; we would like to see one that does.

---

## Appendix — notation, metrics, and the walls we walked into

**Notation.** `f` in cycles per window; `b = f · k / ctx` cycles per patch; `Δφ = 2πb (mod 2π)`.
Retention `r_b = |Z(pred, b)| / |Z(clean target, b)|`. `Δk_b = |Z(pred, b) − Z(clean target, b)| /
|Z(clean target, b)| = |r e^{iΔφ} − 1|`, so `Δk ≥ |1 − r|` with equality iff the phase is locked.
`r` measures **coherent** power, not total power: for a random phase offset, the coherent amplitude
fraction is `|E[e^{iφ}]|`, which is `1` for a fixed phase, `0` for uniform, `e^{−σ²/2}` for Gaussian,
and `J_0(β)` for sinusoidal PM. Total power is `A²/2` always. For PM, `Σ_n J_n(β)² = 1` conserves
total power while the carrier's share `J_0(β)²` falls — that is the whole trick.

**The contracts we would put in the next person's README.** The ones we paid for:

1. Fix the horizon in patches; configs are comparable only at the same number of predicted patches.
2. Report the converged state *and* log the transient.
3. Probe frequencies must be ≥ 1 cycle/patch apart — the orthogonality threshold is exactly one cycle
   per patch.
4. Gate every retention reading on the model's fit on its own marginal. A mean predictor reads `r ≈ 0`
   at every band with `r8/r1 ≈ 1`, which looks exactly like uniform damping and is not.
5. A band's **bandwidth** is not its **centre frequency**. Our own θ instrument's efficacy tracked a
   band's degrees of freedom: a low band with 6 FFT bins beat its analytic bound by 60 % while a
   69-bin high band did not beat it at all.
6. Separate the three sources of unpredictability — *irreducible*, *observation-limited*, and
   *capacity-limited*. Every claim should say which one it means.
7. Noise **placement** is the instrument; noise **level** is not. White noise is unpredictable in
   every band, so it can only produce uniform damping — it is a control, not a lever.
8. Match the perturbation design to the claim: perturb every band at once for a *relative* claim,
   one band at a time for a *quantitative* one. Eight simultaneous phase walks are a much harder joint
   objective and cost us a 6 % ceiling.
9. RMS-match the probe to the training corpus's *signal* power, and keep signal power fixed across
   arms. This trap corrupted two independent readings before we caught it.

**Two bugs worth naming**, because both produced confident nonsense:

- `sqrt(-max(log P, 0))` evaluates to `0` for every `P < 1`. It silently zeroed every phase-diffusion
  rate and turned our analytic benchmark into a perfectly predictable corpus that read 1.000
  everywhere. Caught only because the benchmark was *analytic*.
- Our leakage column was a tautology at integer bands (§5). Caught only because a number looked too
  good — `P = 0.80` at a 128 h lead on 1 h-period content.

**References.** Rahaman et al., *On the Spectral Bias of Neural Networks*, ICML 2019. Xu et al.,
*Frequency Principle*, 2019. Yang & Salman, *A Fine-Grained Spectral Perspective on Neural Networks*,
2019. Basri et al., *The Convergence Rate of Neural Networks for Learned Functions of Different
Frequencies*, ICML 2020. Bietti & Mairal, *On the Inductive Bias of Neural Tangent Kernels*, NeurIPS
2019. Piao et al., *Fredformer: Frequency Debiased Transformer for Time Series Forecasting*, KDD 2024.
Yu et al., *Frequency Bias in State Space Models*, ICLR 2025. Yu et al., *embedding spectra*, 2025.
