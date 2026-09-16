# Frequency is overloaded: what TSFMs are actually doing to a signal's spectrum

**Status:** working design + storyline document. The operative experiment contracts and metric
definitions are in the appendices (§A–§C); the main body is the argument.

**Run log:** `experiments/README.md`. **Sandbox:** `experiments/0_init/`.

---

## 0. Thesis

The time-series forecasting literature uses one word — *frequency* — for at least **six different
objects**, and the resulting claims do not compose: a result about one of them is routinely cited as if
it were about another. The most consequential confusion is between the **frequency content of a
signal** and the **frequency content of the map from history to future**, which are not the same
object and are not related in the way the borrowing of the term implies.

Once the task is restricted to what TSFMs actually do — **forecasting**, i.e. generalising to a
held-out target — and "frequency" is fixed to mean the **signal's spectrum**, the phenomenon that the
field calls "frequency bias" turns out to be a **learned prior calibrated to the training
distribution's per-band conditional predictability**, quantitatively at the Bayes-optimal value for
that distribution, and otherwise **independent of frequency**.

Two consequences:

1. **Clarification.** The persistent high-frequency suppression observed in TSFMs is **not** the
   spectral bias of the learned map, and it is **not** a converged architectural defect. The
   architecture decides *whether frequency is a variable at all* (patching does that); the data
   decides *how much each band is attenuated*.
2. **A practical lever.** Because the attenuation is set by the training marginal, it is **predictable
   from the corpus before any training** (§6, T1), and it is not moved by the interventions the field
   usually reaches for. The same prior that looks like a defect on one corpus is the *desirable*
   denoising behaviour on another — which is why the literature can report both, from the same
   mechanism.

**Scope, stated up front.** The mechanism results are established on a small, **RevIN-free**,
input-space next-patch Transformer (`SimTFM`) over synthetic corpora. Real-data transfer is designed
(§6) but not yet run. §7 lists what is *not* claimed.

---

## 1. The term is overloaded: six distinct objects

| # | Object | Whose variable | How it is measured |
| :-- | :-- | :-- | :-- |
| **1** | **Signal frequency** — the Fourier content of the observed series | TSFM / forecasting critiques | FFT of `x` |
| **2** | **Function frequency** — the eigenfrequency of the map from input to output | classical spectral bias | NTK / RKHS eigenvalues |
| **3** | **Per-patch phase advance** `Δφ = 2π·b (mod 2π)`, `b = f·k/ctx` | *the model's own operational variable* | the patch-level rotation the next-patch map must implement |
| **4** | **Bandwidth / effective dimension** of a band | rank-structure results | `rank_ε`, stable rank |
| **5** | **Occurrence / input density** — how often a frequency appears | input-density theory | `p(x)` |
| **6** | **Spectral energy / amplitude** | energy-based debiasing | band power |

### 1.1 The confusion that matters: #1 vs #2

The spectral-bias literature is about **#2**. The TSFM literature reports **#1** and cites **#2** as
the explanation. They are different objects, and the difference is easy to see:

> A **high-frequency pure tone** is high-frequency in sense #1, but the map that generates it is
> **trivial** — a rotation. The lab measured this directly: a single *affine* map over the frozen
> encoder matches the transformer head (It.41), and the next-patch map for a tone is a phase advance
> (It.10). Conversely, a *low-frequency* signal can require a hard map.

So "the model handles low-frequency signal content better" is **not implied** by "the map is biased
toward low-frequency functions". Nothing about the spectral bias of the map tells you how much of a
band's energy will survive in the forecast.

### 1.2 The transducer between #1 and #3 is the tokenizer

A signal frequency `f` (cycles per window) is seen by the model as a **per-patch phase advance**

```
b = f · k / ctx      Δφ = 2π b (mod 2π)
```

where `k` is the patch size. This is not a cosmetic change of units. The lab measured two sharp
consequences:

- **With `k = 1` the model is frequency-blind**: f = 16/32/64 give *identical* geometry, plane overlap
  **1.000** (It.15). Frequency selectivity is created by patching, not by the transformer.
- **The orthogonality threshold is exactly one cycle per patch**: `d = k` in raw units, i.e. `Δb = 1`
  (It.34; overlap 0.985 / 0.29 / **0.04** at `Δf = 1 / 8 / 16` in a 1024/64 config).

**Consequence for reading the literature:** any claim about "the model's frequency bias" is
underdetermined without the tokenizer, because #3 is what the model sees. A patch-size change moves
#3 for every #1 — which is why patch-size comparisons are so easily misread.

### 1.3 Why #4 gets read as #1

The classical theory's native variable is a **bandwidth**, not a centre frequency. The same slip
reappears in the TSFM literature, and **we made it ourselves**: our θ instrument's efficacy turned out
to track a band's **degrees of freedom** — i.e. its bandwidth — rather than its centre frequency (§4,
P2). Low band `[2,8]` (6 FFT bins) beat its analytic bound by **60%** while high band `[96,160]`
(69 bins) did not beat it at all. Anyone who uses band power or band index interchangeably with
"frequency" will reproduce this error.

---

## 2. What the prior work actually studied

Two axes separate the literature from our problem cleanly:

- **rate vs floor** — is the claim about *how fast* a frequency is learned (a training-dynamics
  statement, removable by more training), or about *how much* survives at convergence (a fixed point)?
- **whose variable** — architecture / initialization, or the **data distribution**?

| Work | Setting | Target held out? | rate / floor | "frequency" = | Driver |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **Rahaman et al. 2019** (ICML, 1806.08734) | fit a known target on a fully observed domain | no | **rate** | #2 | architecture + data-manifold geometry |
| **Xu 2020** (F-Principle, 1901.06523) | same | no | **rate** + an interpolation-selection corollary | #2 | activation regularity |
| **Yang & Salman 2019** (1907.10599) | kernel spectra | n/a | floor (at init) | #2 | architecture |
| **Basri et al. 2020** (ICML, 2003.04560) | fit a known harmonic, fully observed | no | **rate** (`O(κ^d/p(x))`) | #2 **+ #5** | **input density** |
| **Bietti & Mairal 2019** (NeurIPS, 1905.12173) | NTK / RKHS | n/a | **floor** (null-space modes unreachable) | #2 | architecture |
| **Piao et al. 2024 — Fredformer** (KDD, 2406.09009) | **forecasting, real data** | **yes** | post-hoc / converged-ish | **#1 + #6** | **energy (#6)** |
| **Yu et al. 2025a** (ICLR, 2410.02035) | LTI transfer functions (SSMs) | n/a | **floor** ("conventional training does not alter this bias") | #1 (transfer function) | **initialization** |
| **Yu et al. 2025b** (2510.03358) | embedding singular spectra | n/a | floor | **#4** | bandwidth / patch size |
| **arXiv 2510.19236** (attribution unverified) | embedding geometry + TSFM experiments | no for Thm 1 | floor | **#4**, stated as #1 | bandwidth |

|  | **Architecture / init** | **Data distribution** |
| :-- | :-- | :-- |
| **rate** (removable) | Rahaman 2019; Xu 2020; Yang & Salman 2019 | Basri 2020 — a convergence-**time** result |
| **floor** (non-removable) | Bietti & Mairal 2019; Yu 2025a | **empty** |

### 2.1 The empty cell, and why it is empty

**No paper in this set studies a converged, non-removable, data-distribution-driven frequency bias.**
The nearest misses are precise: Basri 2020 has the data distribution as a genuine variable but is a
**rate** result; Bietti & Mairal 2019 and Yu 2025a make non-removable claims but attribute them to
**architecture** and **initialization**. Fredformer is the only prior work in the *forecasting*
setting, and it attributes the bias to band **energy** and treats it as **removable** — which is the
point of its remedy.

**The structural reason.** Every work above except Fredformer studies *fitting a known target on a
fully observed domain*. There the target is in the training set, so there is no Bayes error and
"regressing to the mean" is never correct; the only possible limitation is optimisation, hence
inevitably a rate story. **Forecasting is a different regime**: the target is held out, the conditional
distribution has irreducible spread, and shrinking toward the conditional mean is the *correct* answer.
The classical framework cannot express the phenomenon we study — it has no cell for "the target is not
determined by the input".

### 2.2 The one paper whose frequency section needs a direct response

arXiv 2510.19236 §2 is the closest TSFM work. Its gap can be named precisely:

1. **Its theorem is a bandwidth (#4) result read as a frequency (#1) result.** The native variable is
   the bandwidth ω (`rank_ε = O(ε⁻²ω)`); statement 1 is exactly "bandwidth bounds the effective
   dimension". The step to "low frequencies are learned better" is asserted, not derived — the paper
   concedes as much ("even without going into that level of detail, we have an intuitive
   understanding"). **We were burned by this exact conflation ourselves** (§1.3).
2. **Its "intuitive understanding" is a data argument wearing an architecture's clothes.** The sentence
   invokes the training marginal ("high-frequency content often appears more irregular or noisy") and
   then slides to `W_Q, W_K, W_V` without separating the two accounts.
3. **Its frequency figure is our It.27 design**: a *pretrained* real-data model probed with a clean
   synthetic two-mode signal measures **transfer of a predictability prior**, not an architectural
   property. No power control; no predictability manipulation; `k=1` vs `k=16` compares two separately
   pretrained checkpoints, so architecture and training both vary.
4. **A frequency sweep must be reported in cycles/patch.** `Δφ` is periodic, so a smooth monotone
   "low is better" curve in raw frequency needs justifying against that periodicity (lab It.29-30).

Our response is **not** a refutation. Their intuition is directionally right; our experiments turn it
from an aside into a tested claim.

### 2.3 The symmetry that explains why the field reports both things

The same learned prior is a **defect** when the high band is signal in the probe, and a **feature** when
the high band is noise in the corpus:

- probed on a clean high-frequency tone, the suppression looks like a failure (It.27: the model emits
  **7%** of a perfectly copyable band);
- reported as outlier robustness on large patches, the *same* suppression looks like denoising
  (arXiv 2510.19236 finds larger patches "more effective at denoising" when outliers act as
  high-frequency perturbations).

Both are the model outputting the conditional mean. **So "frequency bias" cannot be evaluated without
specifying whether the band is signal or noise in the distribution** — a property of the data, not the
model. This is the clearest single demonstration that the term is doing too much work.

---

## 3. Our setting, defined precisely

Everything below is stated in these terms; nothing is borrowed from #2 or #4.

| | |
| :-- | :-- |
| **Task** | **Forecasting only.** The target patch is held out; the model must generalise to an unseen next patch. Not function fitting, not interpolation |
| **"Frequency"** | **#1, the signal's spectrum.** Reported in `cycles/patch` (`b = f·k/ctx`, contract 6), with `Δφ = 2πb` stated alongside (#3) |
| **Model** | `SimTFM`: input-space next-patch, patched (`k`), **no RevIN**, **no SIGReg**, RoPE, `context_size + patch_size` training windows |
| **Metric** | `r_f` = \|matched-filter amplitude of the prediction at `b`\| ÷ \|matched-filter amplitude of the **clean** target\|. Measures **coherent** power, not total power (Appendix B) |
| **Probe** | A **fixed, clean, deterministic** multi-tone signal, identical in every arm |

**Why the clean probe is the load-bearing design choice.** Shrinking an unpredictable band is
*correct* MSE behaviour. A learned prior is visible only if the deficit **survives on an input where
shrinking is wrong**. An in-distribution probe cannot distinguish "the model hedged appropriately"
from "the model carries a prior" — this is exactly what separates the lab's It.27 from It.29, and it is
the design error we made first (P0).

**The instruments, and why two of the three were superseded.** These supersessions are themselves
reviewable findings:

| Instrument | What it manipulates | Fate |
| :-- | :-- | :-- |
| **Additive band-limited noise** | total band power | **Superseded (P0).** With a *persisting* tone, i.i.d. noise is averaged over ~16 context patches; effective in-band SNR ~**0.011**, so `r ≈ 1` is the *correct* answer and the null discriminates nothing |
| **θ — coherent-energy fraction** | how much of a band's power is a persistent tone | **Superseded for mechanism, kept as a fittability manipulation (P2).** Its "incoherent" energy is built from the window's own FFT bins, so it is a deterministic, low-DOF realization the context fully determines — an LS fit reproduces the target at **MSE 0.00000000**. So θ moves *difficulty*, not the Bayes floor |
| **PM — per-patch phase random walk** | **irreducible** unpredictability at *constant envelope* | **Kept.** Future increments are genuinely unknowable, total power is fixed by construction, and it has a closed-form optimum `r* = exp(−β²σ²/2)` |

---

## 4. The argument, as a chain of killed alternatives

Full numbers in `experiments/README.md`. Each step removes one explanation.

### P0 — kills "additive noise is a usable instrument"

Fixed clean 2-tone probe (`b = 0.125, 8.0`); only the *band placement* of training noise varies.
**No shrinkage in any arm**; every non-zero effect is gain *inflation* (hi@8: 1.0005 → 1.0428, ~13× the
seed spread). Retrospectively not a test: effective in-band SNR ~0.011 (§3 table above).

### P1 — establishes causation

θ sweep at a fixed clean probe, band power constant. Clean-probe `r@8`:

```
θ = 1.0    0.9995 ± 0.0039        θ = 0.2    0.7035 ± 0.0105
θ = 0.5    0.8195 ± 0.0308        θ = 0.0    0.0362 ± 0.0047      off-support   0.1891 ± 0.0592
```

Monotone, every step beyond the seed spread; all arms pass the fit gate; RMS constant to 0.14%.
**Only the training marginal's per-band statistics were changed.** Two results beyond the headline:
the two zero-coherence endpoints differ — an incoherent-**filled** band (0.036) is damped ~5× harder
than an **empty** one (0.189) — so coverage and incoherence are **two mechanisms**, not one axis; and
energetic-but-unpredictable content is actively *punished* by the loss, which is stronger than merely
having no basis to predict.

### P1b — kills "it is a capacity limit"

Context ablation (`N` ∈ {1,2,4,8,16}) on the trained models, read on **two** corpora.

```
                       N=1       N=16                     N=1       N=16
θ=0.5 marginal MSE   0.16110   0.13452   (−16.5%)   θ=0.5 clean-probe r@8   0.80294   0.81946  (+0.0165, inside spread)
  oracle             0.15702   0.12583   (−19.9%)     Wiener calibrated to N  0.8093    0.9855   (+0.1762) → captures 9.4%
                                                      evidence-adaptive       1.000     1.000
```

The model captures **85%** of the coherent-averaging gain available on its own marginal — so it *does*
integrate across patches — yet its clean-probe deficit **does not move** with the number of context
patches, where an evidence-adaptive estimator would read **1.000 at every `N`**. The *same
architecture* trained on a clean marginal reads `r@8 = 1.000`. **It is a learned prior, and it is
patch-local rather than context-global** — consistent with the lab's C3 (position-invariance), since a
model using its context would shrink *less* the further into the window it predicts.

*Design note:* the oracle is exactly **0** at every `N` on the clean probe and only falls on the θ=0.5
marginal. **Measuring on the clean probe alone would have been a false negative** — a persistent clean
tone is extrapolable from a single patch.

### P2 — kills "the model damps high frequencies"

Three bands (`b = 0.25, 2, 8`), exactly one perturbed per arm, all bands read on every arm, clean arm
subtracted. Own-band drops, clean-differenced:

```
                own              other two        own − other
lo_50/mid_50/hi_50   −0.243/−0.130/−0.119   +0.005/−0.031/−0.083   +0.247/+0.099/+0.036
lo_00/mid_00/hi_00   −0.352/−0.689/−0.945   +0.010/+0.031/−0.010   +0.362/+0.720/+0.935
```

**Diagonal.** The high band is untouched by low/mid perturbations (`lo_50` → `r@8` **+0.0039**).
A row+column decomposition leaves a residual only on the diagonal. **"The model damps high
frequencies" is dead.**

Two residuals were recorded, and **P2b showed both were θ-instrument artifacts**. P2 also surfaced the
θ flaw that §1.3 and §3 describe: **4 of 6 arms beat the analytic fit bound** (by 14–60% of the
incoherent power), and an LS fit of the generative basis reproduces the target at MSE 0.

### P2b — establishes the quantitative claim, with an irreducible instrument

Per-patch phase random walk; constant envelope so total power is fixed at every β; optimal shrinkage
`r* = exp(−β²σ²/2)`. Clean-probe `r@8`:

```
 β      r*      probe r@8        r − r*      1 − r      reads closer to
 0.0   1.000   1.0027±0.0031    +0.0027    −0.0027    indistinguishable
 0.5   0.882   0.8764±0.0127    −0.0061    +0.1236    r* (prior)
 1.0   0.607   0.6019±0.0139    −0.0047    +0.3981    r* (prior)
 1.5   0.325   0.3029±0.0129    −0.0217    +0.6971    r* (prior)
 2.0   0.135   0.1157±0.0143    −0.0196    +0.8843    r* (prior)
```

Every gap to **1.000** is ≥10× the seed spread; every gap to `r*` is ≤**0.022**. All 7 arms sit at
98–99% of the analytic bound and **none exceeds it** — the opposite of θ, so the instrument is sound.
Part B (β=1.0 per band) is diagonal again with **equal** magnitudes (−0.4226 / −0.4058 / −0.4008) and
every off-diagonal cell ≤0.007 — which is what dissolved P2's two residuals.

### P3 — kills "there is a residual frequency dependence"

Eight integer bands `b ∈ {1..8}` (`Δb = 1` exactly, matched filters exact, `Δφ ≡ 0` so the transient
axis is held fixed), **uniform β over all bands**, clean 8-tone probe. `spread_b` (max−min over bands)
= **0.019–0.025** in every arm against a pooled per-band seed sd of 0.008–0.034 — at or *below* what 8
iid readings would spread. Clean-differenced drops are equal across bands
(−0.174 / −0.491 / −0.782 / −0.865). **No residual frequency dependence.**

The head sat uniformly *below* `r*` (`r/r*` = 0.867 / 0.738 / 0.483 / 0.547, no trend in `b`). The
oracle resolves it: a ridge readout of the **frozen latents** gives `oracle/r*` = **0.987 / 0.988 /
0.998 / 1.098**, flat in `f`. **The representation reproduces `r*` exactly; the head's shortfall is a
uniform optimisation gap.** That gap is my own design's fault — perturbing all eight bands at once is a
much harder joint objective than one at a time (contract 16) — and it is why the headline quantitative
result is P2b's, not P3's.

### The claim, in its final form

> The next-patch map's amplitude contraction at frequency `f` is set by the **conditional
> predictability of `f` in the training distribution**, not by `f` itself. Given matched
> predictability:
> **(1)** perturbing one band moves *that band's* clean-probe retention (P1, P2);
> **(2)** which band moves is not determined by its frequency (P2, P2b);
> **(3)** there is no residual dependence on `f` (P3);
> **(4)** the magnitude is the **Bayes-optimal shrinkage for that marginal**, to within 0.03 (P2b).

---

## 5. What this does and does not clarify

**Clarified.**

- The persistent "high-frequency suppression" of a TSFM is **not** the spectral bias of the learned
  map (#2). Nothing about the map's spectral bias predicts how much of a band's *signal* energy
  survives; the map for a high-frequency tone is a rotation (It.10, It.41).
- It is **not** a converged architectural defect. The attenuation is a function of the training
  marginal, quantitatively matching the Bayes-optimal shrink, and frequency-independent at matched
  predictability.
- It is **not** removable by "more training" either — because it is not an error. It is the conditional
  mean.
- The architecture's real role is different and upstream: **patching decides whether frequency is a
  variable at all** (It.15: `k=1` is frequency-blind), and the tokenizer maps #1 to #3.

**Not clarified / not claimed.** See §7.

**The symmetry (§2.3) is the practical clarification.** Whether the suppression is a defect or a
feature depends entirely on whether the band is signal or noise in the training distribution. The
literature reports both, from the same mechanism, and attributes both to the architecture.

---

## 6. What this implies, and the real-data programme

### 6.1 The lever is the data, and it is measurable before training

If per-band attenuation is the Bayes-optimal shrink for the training marginal, then:

- **it is predictable from the corpus alone**, without training a model: estimate each band's
  conditional predictability given the observed context, and you have predicted the model's per-band
  retention;
- **band-wise predictability, not band index, is the quantity to report.** A corpus in which high
  frequencies are genuinely predictable should produce a model that preserves them — and P2b is the
  synthetic demonstration that it does;
- the same account explains the *desirable* case: when a band is noise, suppressing it is correct, so
  "denoising" and "frequency bias" are one phenomenon evaluated under two data distributions.

This is a **data-preparation** insight, not an architectural one. It is also falsifiable cheaply, which
is the point of T1.

### 6.2 Real-data transfer (designed, not yet run)

| | What | Cost | What it buys |
| :-- | :-- | :-- | :-- |
| **T1** | **Measure the explaining variable**: a corpus's per-band conditional predictability `P(b)`, after phase-folding out clock-locked components. Includes four morphology diagnostics (§6.3) | low, no training | the independent variable of the whole claim; predicted to explain any model's `r_b` |
| **T2** | Train `SimTFM` on a real corpus, probe clean, check `r_b` against `P(b)` | medium | P2b with a real marginal |
| **T3** | **Causal**: manipulate one band's predictability by **per-patch phase randomization** — preserves band power *exactly* — then train and probe | medium | the strongest real-data test; also kills the energy account by construction |
| **T4** | Audit a released TSFM's per-band retention against `P(b)` of a proxy for its training distribution | medium-high | the blog's "so what" |

**Anti-Monash-checkpoint note.** Reusing the lab's Monash checkpoints is not possible (different model
class). **Real-data work must use a single corpus**, which is better anyway: it permits a fine
`P(b)` spectrum, a within-corpus contrast, and a check for structure in `P(b)` at the model's own
scales (`b = 1`, the DFT orthogonality threshold; and `Δφ = π`, the lab's persistent laggard).

**The gating risk.** Contract A8 (fit gate) is the binding constraint for T2/T3: `SimTFM` is small, and
real-corpus forecasting skill is poor. Mitigations: a high-signal single domain; the S2/S3 ladder
(Appendix C); a **per-band** fit gate rather than a global one. T1 and T4 do not depend on our model
fitting anything.

### 6.3 Why solar / electricity are the right corpora — and what they are *not*

Their dominant components are **externally clocked** (diurnal, weekly), so their phase is *known* and
does not diffuse. That is the **opposite** of an autonomous oscillator, and it has two consequences:

- **They are not PM-like in morphology.** Good: it keeps the instrument/model distinction honest (§7).
- **Their unpredictable high-frequency content is "observation-limited"** (weather is deterministic but
  chaotic and only partially observed) rather than strictly irreducible. **This does not weaken the
  claim**: the model's Bayes-optimal prediction is the same conditional mean in both cases. It means
  PM (irreducible) and real data (observation-limited) are **two different sources hitting the same
  mechanism** — a robustness argument, not a gap. The lab's own It.31 residual was already attributed
  to observation-limited predictability.

Because the deterministic part is clock-locked, it is **removable by construction** (phase folding),
which leaves a well-defined unpredictable residual whose `P(b)` is clean to estimate. Four cheap
diagnostics settle the morphology question and belong in T1:

1. **Narrowband envelope CV** — oscillator/PM ≈ 0 (constant envelope); stationary Rayleigh ≈ 0.52.
2. **Phase diffusion vs phase slips** (*decisive*) — does unwrapped phase variance grow linearly in
   time, and do the large jumps coincide with envelope fades? Oscillator phase noise does not need
   fades; stationary-Gaussian "diffusion" is nothing but fades.
3. **Linewidth vs record length** — non-stationary phase ⇒ narrows with longer records; stationary
   ⇒ converges.
4. **Is the periodic component's phase locked?** Phase folding answers forced-vs-autonomous directly.

---

## 7. Limitations, and what is deliberately not claimed

1. **No RevIN.** `SimTFM` is RevIN-free; the lab showed RevIN is load-bearing for the representation's
   geometry. The claim is currently about a RevIN-free input-space forecaster. **A RevIN twin is
   mandatory before any claim about production TSFMs** (Appendix A, P4).
2. **Small model, one scale.** Capacity is ruled out *at this scale* by the θ=1.0 control (same
   architecture trained clean reads 1.000), not across scales. S1/S2 are designed, not run.
3. **Synthetic.** Real-data transfer is designed (T1–T4), not run.
4. **PM is an instrument, not a data model.** Phase diffusion is a well-founded, citable model of
   *self-sustained narrowband components* (Demir et al. 2000; lasers, NEMS resonators, circadian
   clocks). It is **not** a model of stationary broadband content: for a stationary narrowband Gaussian
   process the phase does **not** diffuse — unbounded phase growth requires an autonomous limit cycle —
   and the standard narrowband model of, e.g., ocean swell has random but **non-diffusing** phase.
   **Our constant envelope is therefore a construction, not an empirical fact.** We use PM because it
   isolates the property under test and has a closed-form optimum; the real-data bridge is T1/T3, not
   PM's realism.
5. **"Not architectural" must be read precisely.** Patching *does* create frequency selectivity
   (It.15). What we refute is that the converged *suppression* is an architectural defect.
6. **θ remains a fittability instrument.** Its P2/P3-era use is superseded for mechanism; any future
   use must respect Appendix A's contract 10 (bandwidth ≠ centre frequency).
7. **The Δφ axis is not separated in the real-data plan.** P3 held `Δφ ≡ 0`; half-integer `b`
   (`Δφ = π`) is the lab's persistent laggard and needs its own arm.
8. **Not a "fix" paper.** The deliverable is a claim about mechanism plus a data-preparation insight;
   real-data forecasting gains are out of scope.

**Open work.** P4 (RevIN twin, scaling S1/S2, a real-data instance), P5 (gradient probe: which
frequencies actually lower the loss), T1–T4, and the robustness backlog (P6: phase structure, tone
count, occurrence/power decoupling, amplitude coupling, bin alignment, non-stationarity).

---
---

# Appendices — the operative material

## Appendix A — experiment contracts

Confounds the lab paid dozens of iterations to learn. Requirements, not suggestions.

1. **Fix the horizon in patches.** It.35/36: the "smaller patches are better" result was a
   mismatched-horizon artifact. Configs are comparable only at the same number of predicted patches.
2. **Report the converged state and log the transient.** It.21: past 10k steps on one fixed config the
   `k=1` retention moved 0.574 → 0.843 — half the apparent "bias" was under-training.
3. **Count frequencies evenly in the probe corpus.** It.45/49: the "elliptical ring" conclusion from an
   uneven cluster density was misattributed; the real cause was training duration.
4. **Probe frequencies must be spaced ≥ `ctx/k` apart** (1 cycle/patch). It.34: the orthogonality
   threshold is exactly `d = k` (overlap 0.985 / 0.29 / **0.04** at `Δf = 1 / 8 / 16`).
5. **Every experiment carries an oracle upper bound.** It.46: a detect→rotate→recombine pipeline on
   frozen `z` reaches MSE **0.043–0.070** against the transformer head's **0.44–1.55**. Without an
   oracle, "did not learn" and "cannot be learned" are indistinguishable. **Caveat learned in P0/P2b**:
   `fit_oracle`'s target is the *noisy observed* patch, so its absolute scale is unreliable and on a
   stationary probe it is degenerate (it reads 1.000); only use it for the representation-vs-head gap.
6. **Report in `cycles/patch`** and state `Δφ` alongside.
7. **At least 3 seeds per config**, reporting spread. A single run's difference is not a conclusion.
8. **Gate every retention reading on the model's fit.** `r_f` is meaningless for a model that did not
   learn its corpus. Report variance explained (`1 − mse / var(target_patch)`) on a held-out corpus of
   the *same marginal* beside every `r_f`. A mean predictor yields `r ≈ 0` at **every** band with
   `r8/r1 ≈ 1`, which reads exactly like uniform damping and is not.
9. **RMS-match the probe to the training corpus's *signal* power.** Every tone in a mixture has
   amplitude 1, so a corpus's RMS scales as `sqrt(n_tones)`; a perfectly-tracking model reads
   `r ≈ 2.0` when an 8-tone corpus is probed with a 2-tone signal. Keep signal power fixed across arms,
   add noise on top, and match the probe to the signal power. This trap corrupted two independent
   readings before it was caught.
10. **A band's bandwidth is not its centre frequency.** θ's efficacy tracked a band's **degrees of
    freedom**: low band `[2,8]` (6 FFT bins) beat its analytic bound by **60%** while high band
    `[96,160]` (69 bins) did not beat it at all. Report bin counts alongside band centres.
11. **Separate the three sources of "unpredictability"** — they are different mechanisms:
    - *irreducible* — the future is genuinely not a function of the past;
    - *observation-limited* — determined by the past but not identifiable from the observed context;
    - *capacity-limited* — caught by contract 8, but unchecked it masquerades as damping.
    Every claim states which one it means, and the corpus isolates it.
12. **Noise *placement* is the instrument; noise *level* is not.** White noise is unpredictable in
    every band, so it can only produce uniform damping — it is a control.
13. **Probe with a clean signal to demonstrate a prior.** A deficit that survives a clean probe is a
    prior; one that does not is correct hedging. This is what separates It.27 from It.29.
14. **`SimTFM` has no RevIN, so absolute scale enters directly.** Marginal power density sets both the
    input scale and the per-window loss weight.
15. **Scale one knob at a time, and hold the `b` grid fixed when you scale.** Frequencies are
    comparable only at fixed `b = f·k/ctx`, so changing `ctx` means re-deriving `f = b·P` with
    `P = ctx/k`. **Model size is a variable, not an upgrade** — rungs are compared as a trend.
16. **Match the perturbation design to the kind of claim.** For a *relative* claim ("is retention flat
    in `f`?", "does the effect follow the perturbed band?") perturb every band at once: 3× cheaper and
    sufficient. For a *quantitative* claim ("does `r` equal its optimum?") perturb **one band at a
    time** — eight simultaneous phase walks are a much harder joint objective, and in P3 the head fell
    uniformly to `r/r*` = 0.48–0.87 with a 6% ceiling loss even at β=0, while the same architecture
    perturbing one band at a time read `r = r*` within 0.03 (P2b). The oracle is what caught it.

## Appendix B — variables and metrics

Notation: `f` in cycles per window (`ctx`); `b = f·k/ctx` cycles/patch; `Δφ = 2πb (mod 2π)`.

**Measure against the clean target, not the realised one.** Every corpus returns both the observed
window and the noise-free target patch:

```
r_f = |Z(pred, b)| / |Z(clean_target, b)|
```

so `r_f = 1` means "the deterministic component was recovered exactly", *unconditionally*. Against a
realised noisy target the correct prediction (the conditional mean) reads `r < 1` whenever the
evaluation corpus carries noise — precisely the regime where a deficit is least interpretable. On a
clean corpus the two definitions coincide. (This does **not** fix scale mismatch; that is contract 9.)

| Metric | Definition |
| :-- | :-- |
| **retention** `r_f` | Fixed forecast position, `\|Z(pred, b)\| / \|Z(clean target, b)\|` |
| **phase error** | `angle(Z(pred, b)) − angle(Z(clean target, b))`, unwrapped |
| **ring** `dim1+2` | Per `f`, PCA of the phase-loop latents; variance fraction of the top two components |
| **plane overlap** | Grassmann `mean(cos²)` between two frequencies' top-2 subspaces. 1 = identical, 0 = orthogonal, 0.5 = sharing one direction |
| **transient metric** | the `r_f(t)` curve, and steps to reach a threshold `ε` (implemented in P3) |
| **fit gate** | variance explained on a held-out corpus of the same marginal (contract 8) |

### Why `r` measures **coherent** power, not total power

Write a signal as `x(t) = A(t)·cos(2πbt/k + φ(t))`, with `A(t)` the **envelope**. For a random phase
offset `φ`, the **coherent amplitude fraction** is `|E[e^{iφ}]|`, the characteristic function of the
phase distribution:

| phase distribution | `\|E[e^{iφ}]\|` |
| :-- | :-- |
| fixed | `1` |
| uniform | `0` |
| Gaussian `N(0, σ²)` | `e^{−σ²/2}` |
| sinusoidal PM, depth β | `J_0(β)` |

Total power is `A²/2` **always**; coherent power is `A²·|E[e^{iφ}]|²/2`. The matched filter is
phase-locked, so **`r_f` measures the coherent part only**:

| manipulation | coherent power | total power |
| :-- | :-- | :-- |
| P0: additive band-limited noise | unchanged | **rises**; in-bin SNR diluted by the fill fraction (nominal 0.49 → in-bin 0.19) |
| θ: coherent-energy fraction | falls with θ | fixed by construction |
| PM: phase diffusion | falls as `J_0(β)` / `e^{−β²σ²/2}` | **fixed — constant envelope** |
| coverage: band absent | zero | zero |

**Constant envelope power does not mean constant bin power.** For PM, `Σ_n J_n(β)² = 1` conserves
total power while the carrier's share `J_0(β)²` falls (β=1 → `J_0 = 0.765`, 58.5% of the power;
β=2 → 5.0%). The rest goes to sidebands, which can break contract 4 once β is large. Always record the
**clean target's bin power**.

### Reference probe and the predictability knob

**Reference probe `R`**: defined once, shared by every arm — clean, deterministic, RMS-normalised to
the training *signal* power, frequencies spaced ≥ 1 cycle/patch, fixed seed.

**θ (superseded for mechanism)** = the fraction of a band's power that is coherent and persistent.
Endpoints: *coherent* (θ=1) and *fully incoherent* (θ=0). **Correction:** coverage — a frequency
absent from training — is **not** the θ→0 endpoint. An incoherent-*filled* band (clean-probe
`r@8 = 0.036`) is damped far harder than an *empty* one (`0.189`), and neither lies on the graded curve
running over θ ∈ [0.2, 1.0] (`r@8 ≈ 0.66 + 0.34θ`). Coverage and incoherence are **two mechanisms**.

**PM (kept)** = per-patch phase random walk, `φ_p = φ_{p−1} + 2πb_c + β·ε_p`. Constant envelope, so
total power is fixed at every β. Optimal shrinkage **`r* = exp(−β²σ²/2)`**; report `r / r*`.

## Appendix C — infrastructure

| Path | Contents |
| :-- | :-- |
| `src/fbias/data.py` | corpus builders (`make_mixture`, `make_broad`, `band_noise`, `make_coherent`, `make_banded`, `make_pm`) |
| `src/fbias/probes.py` | the Appendix B metrics + `fit_oracle` |
| `src/fbias/cli/train.py` | training loop; windows **must** be `context_size + patch_size` |
| `experiments/<id>/scripts/`, `runs/` | driver + `state_dict` + sidecar JSON |
| `experiments/README.md` | run log |

**Config.** Development: `ctx 512 / k 32 / hidden 32 / 2L / 4 heads / SGD 1e-2 / batch 64` (both
`ctx/k` choices give `P = 16` patches, so the frequency grid matches the lab's). Headline: the lab's
canonical direct config `ctx 1024 / k 64 / hidden 64`. A 2000-step run is ~47 s on CPU; 20k ≈ 8 min.

### Scaling ladder

Model size is a **variable, not an upgrade** — every rung runs the same sweep so curves can be overlaid
(contract 15).

| | ctx / k | P | hidden | layers | why |
| :-- | :-- | -- | -- | -- | :-- |
| **S0** | 512 / 32 | 16 | 32 | 2 | current; the lab's `monash-direct-h32`, which produced C1's 7% |
| **S1** | 512 / 32 | 16 | 32 | **4** | **depth** — coherent averaging is a multi-step algorithm |
| **S2** | **1024** / 32 | **32** | 32 | 2 | **more patches** — pushes the θ=0.5 optimum from 0.94 to ≈0.97 |
| S3 | 512 / 32 | 16 | 64 | 4 | width + depth. Deferred |
| S4 | 1024 / 64 | 16 | 64 | 2 | lab canonical. Deferred |

**Decided: S1 and S2 only**, as a **generality** check (P1b already refuted capacity at S0), after the
claim's own experiments. `b` is held fixed: at `P=32` the same `b` values correspond to `f = b·P`. The
patch embedding is **already non-linear** (`use_glu=True`: `Linear → RMSNorm → SwiGLU` + residual).
**Prerequisite:** `cli/train.py` pins the window at `ctx + k` and needs a context-length knob for S2.

### Corpus-builder axes

*Required now:* return `(observed_window, clean_target_patch)`; a **per-patch phase random walk** with
an explicit β and a carrier decoupled from the band; **signal power fixed** across arms with the probe
normalised to signal power; an **off-support** variant; a fixed frequency set within a corpus with
per-window phases; frequencies spaced ≥ 1 cycle/patch with `b` and `Δφ` reported; variance explained on
a held-out same-marginal corpus; the clean-arm baseline and the gain-inflation correction.

*Deferred (P6 robustness):* occurrence vs marginal power density (currently **identical by
construction** — marginal power ∝ occurrence, measured `power/count` std = 1e-2 over a 3469× range);
conditional amplitude; per-window tone count; phase structure (independent vs locked); bin alignment;
non-stationarity; per-window power CV (bandwidth-dependent: 15% for tight `peaks` vs 5.6% for `broad`).
