# Experiment Design: the origin of frequency bias in TSFMs

This file is the **experiment design contract**: the claim, the mandatory controls, and the metric definitions.

- Run log / results / conclusions → `experiments/README.md`
- Per-experiment intent, config, pass criteria → `experiments/<id>/README.md`
- Sandbox (harness, generator rewrite, debugging) → `experiments/0_init/`

Scope note: the target output is an **ICLR blog-track post**, so the deliverable is one sharp,
falsifiable claim with a clean figure — not exhaustive coverage of every axis.

## 0. The claim

> **Claim.** The next-patch map's amplitude contraction at frequency `f` is set by the
> **conditional predictability of `f` in the training distribution**, not by `f` itself. Given
> matched predictability, there is no residual dependence on frequency.

This is the falsifiable form of "frequency bias is a learned prior". It decomposes into three
independently testable parts, which are also the three levels of the research plan (§5):

1. **Causal.** Perturbing the training predictability of one band moves *that band's* retention on
   a **clean** probe.
2. **Invariant.** The same perturbation applied to a low band vs a high band moves the *perturbed*
   band, not the frequency. Without this step the claim is not established.
3. **No residual.** With predictability matched, retention is flat in `f`. Any residual is
   quantified, and that number is the claim's honest boundary.

If 1–3 hold, the punchline is that **"frequency bias" is a misnomer**: it is a bias against low
predictability, and high frequencies are merely where real data happens to be unpredictable.

| | |
| :-- | :-- |
| **Model** | `SimTFM`: input-space next-patch, **no RevIN**, **no SIGReg**, RoPE, `context_size + patch_size` windows |
| **Data** | synthetic corpora with controlled spectra |
| **Output** | one falsifiable claim + the figure that carries it; **no fix** — real-data forecasting gains are out of scope |

**Why the framing changed.** Earlier drafts made "reproduce the lab's conclusions" the goal. That
stalled three times, and the reason was structural: C1 as recorded describes an *observation* (the
model emits 7% of the high band) without the condition that produced it (what the training marginal
was), so it was not a well-posed target. It is a better fit as **P1's real-data data point** than as
a gate (§1).

## 1. Motivation: what the lab already showed

From `projects/temporal-batch-comp-sigreg`. This is the evidence the claim is built on — context and
motivation, not a reproduction checklist.

| ID | Lab finding | Role here |
| :-- | :-- | :-- |
| C1 | The bias is on the **prediction side**, not a resolution loss. It.27-28: on `monash-direct-h32.pt` and `geom-rich-input.pt` — **neither trained on the `f=2+f=128` mixture it was probed on** — the high band is copyable at r²=**0.9995** while the model emits r²≈0 / **7%** amplitude (r²=**−0.46** on the rich-corpus model); the low slope is kept at ~54%. It.29 is the *other* regime: trained and probed on one pure-tone corpus, retention **0.988–0.998** | **The phenomenon to explain.** P1's real-data instance |
| C2 | **Patching encodes frequency**; the transformer does not (It.15: at `k=1`, f=16/32/64 agree, plane overlap **1.000**) | Architecture context, not a target |
| C3 | Shrinkage is **position-invariant** (It.14: flat across `p=2..14`; f=32 **0.537**, f=64 **0.524**) | A signature the prior must reproduce — cheap check |
| C4 | Shrinkage **vanishes** on deterministic corpora; ordering set by `Δφ` (It.29-30: finals **0.988–0.998**, `Δφ=π` always the laggard) | The **transient**, which P3 must separate from the fixed point (§2) |
| C5 | **Coverage**: frequencies absent from training stay damped even when deterministic (It.31: converges at 0.865, worst cluster 0.715) | The **extreme endpoint of the predictability axis**, not a separate phenomenon |
| C6 | The phase-ring geometry is **purely architectural** (It.9; It.26: `dim1+2 ≥ 0.94`) | Architecture context, not a target |

**The lab's evidence is on RevIN models; `SimTFM` has none.** The journal's E3 showed no-revin gives
eccentric rings and collapses the trend tilt, and It.49-51 showed SIGReg stretches rings into
ellipses. So geometry is *expected* to diverge — a documented difference, not a failure. Any claim
about **real** TSFMs therefore needs the RevIN twin (P4).

## 2. Two kinds of bias, and they must not be pooled

- **Transient (optimization order).** Set by `Δφ = 2π·b (mod 2π)`, **not by `f`** — It.29-30 showed
  on deterministic corpora that it is transient and vanishes at convergence.
- **Fixed point (converged damping).** Set by the conditional predictability of `f` in the training
  marginal. `Δφ` cannot explain it. This is what the claim in §0 is about.

**Consequence:** `(high-freq error − low-freq error) / (high-freq error + low-freq error)` is
unusable as an index. Error is a periodic function of `Δφ mod 2π`, so differencing high against low
buckets averages over a periodic function — changing `k` or `ctx` flips the sign. The two are
measured separately, and frequencies are reported in `cycles/patch` throughout.

## 3. Experiment contracts

Confounds the lab paid dozens of iterations to learn. Requirements, not suggestions.

1. **Fix the horizon in patches.** It.35/36: the "smaller patches are better" result was a
   mismatched-horizon artifact. Configs are only comparable at the same number of predicted patches.
2. **Report the converged state and log the transient.** It.21: past 10k steps on one fixed config,
   the `k=1` retention moved from 0.574 to 0.843 — half the apparent "bias" was under-training.
3. **Count frequencies evenly in the probe corpus.** It.45/49: the "elliptical ring" conclusion from
   an uneven cluster density was misattributed; the real cause was training duration.
4. **Probe frequencies must be spaced ≥ `ctx/k` apart** (1 cycle/patch). It.34: the orthogonality
   threshold is exactly `d = k` (`Δf=1` → overlap 0.985, `Δf=8` → 0.29, `Δf=16` → **0.04**). Closer
   and DFT bins leak into each other, making per-frequency retention meaningless.
5. **Every experiment carries an oracle upper bound.** It.46: a detect→rotate→recombine pipeline on
   frozen `z` reaches MSE **0.043–0.070** against the transformer head's **0.44–1.55**, a 10–35× gap.
   Without an oracle, "did not learn" and "cannot be learned" are indistinguishable.
6. **Report in `cycles/patch`** and state `Δφ` alongside.
7. **At least 3 seeds per config**, reporting spread. A single run's difference is not a conclusion.
8. **Gate every retention reading on the model's fit.** `r_f` is only meaningful for a model that
   learned its corpus. Report variance explained (`1 − mse / var(target_patch)`) on a held-out corpus
   of the *same marginal* beside every `r_f`. A mean predictor yields `r ≈ 0` at **every** band with
   `r8/r1 ≈ 1`, which reads exactly like uniform damping and is not. Earned 2026-09-13: a 32-tone
   corpus with per-window redrawn frequencies sat at **−0.005** after 20k steps.
9. **RMS-match the probe to the training corpus's *signal* power.** Every tone in a mixture has
   amplitude 1, so a corpus's RMS scales as `sqrt(n_tones)`; a perfectly-tracking model reads
   `r ≈ 2.0` when an 8-tone corpus is probed with a 2-tone signal. Keep **signal** power fixed across
   all arms, add noise on top, and match the probe to the signal power — then a correct model has no
   reason to rescale. This trap corrupted two independent readings before it was caught.
10. **"Broad marginal" must mean a structured frequency family, not a uniform draw.** At
    `ctx 512 / k 32 / hidden 32`, a 32-tone uniform-redraw corpus plateaued at **−0.005** variance
    explained while the *same* 32-tone set held fixed across windows reached **+0.691** and was still
    rising at 20k. The failure was per-window frequency identification, not marginal breadth.
11. **Separate the three sources of "unpredictability"** — they are different mechanisms, and the
    lab's own residual (It.31) is attributed to the second, not the first:
    - *Irreducible noise* — the future is genuinely not a function of the past.
    - *Observation-limited* — determined by the past but not identifiable from the observed context
      (32 phases to infer; only a few cycles of `f` visible). It.31 invokes this; it is **not** noise.
    - *Capacity-limited* — caught by contract 8, but unchecked it masquerades as damping.
    Every claim must state which one it means, and the corpus must isolate it.
12. **Noise *placement* is the instrument; noise *level* is not.** White noise is unpredictable in
    **every** band, so it can only produce uniform damping — it is a control. Only **band-limited**
    noise separates "the model damps the unpredictable band" from "the model damps high frequencies".
13. **Probe with a clean signal to demonstrate a prior.** Shrinking an unpredictable band is
    *correct* MSE behaviour. A prior is demonstrated only if the deficit **survives on a clean
    probe**. An in-distribution probe cannot tell the two apart — this is exactly what separates
    It.27 from It.29.
14. **`SimTFM` has no RevIN, so absolute scale enters directly.** RevIN removed most of the input
    scale in the lab; here marginal power density sets both the input scale and the per-window loss
    weight (§4).

## 4. Variables and metrics

Notation: `f` in cycles per window (`ctx`). Hence

```
b = f · k / ctx        Δφ = 2π b (mod 2π)
```

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
| **retention** `r_f` | Fixed forecast position, `\|Z(pred, b)\| / \|Z(clean target, b)\|`. The frequency-domain form of "radius shrinkage" |
| **phase error** | `angle(Z(pred, b)) − angle(Z(clean target, b))`, unwrapped |
| **ring** `dim1+2` | Per `f`, collect the latents around the phase loop, PCA, variance fraction of the first two components |
| **plane overlap** | Grassmann `mean(cos²)` between two frequencies' top-2 subspaces. 1 = identical, 0 = orthogonal, 0.5 = sharing one direction |
| **transient metric** | the `r_f(t)` curve, and steps to reach a threshold `ε` |
| **fit gate** | variance explained on a held-out corpus of the same marginal (contract 8) |

For the oracle, `r_f` uses the oracle's prediction in the numerator — which is what separates "is it
in the representation" from "does the head use it".

**Reference probe `R`.** Defined once, shared by every arm: clean, deterministic, RMS-normalised to
the training signal power, frequencies spaced ≥ 1 cycle/patch, fixed seed. The prior is read off as
`r_f(R)` versus the training marginal.

**The predictability knob θ.** Band-limited additive noise at a swept SNR, with two endpoints:
*clean* (θ = 1) and *off-support* (the frequency is absent from training entirely, so the model has
no basis to predict it). **Coverage is therefore the extreme of the same axis, not a separate
phenomenon** — which is what makes C5's D2-style result (off-support at `r = 0.66`) a point on the
main curve rather than a side result.

## 5. Research plan

Ordered so the load-bearing figure appears as early as possible, and so each step can kill the claim.

### P0 — instrument check (go/no-go) — *running*

**Question.** Does *any* manipulation of the training distribution move clean-probe retention at a
fixed frequency? Design: fixed clean 2-tone probe, only the *band placement* of training noise
varies (`clean` / `noise_hi` / `noise_lo` / `noise_all`), 3 seeds.
**Kills the project early if:** no arm moves the probe. **Cost:** minutes.

### P1 — the dose–response (the load-bearing figure)

**Question.** Does the clean-probe retention of a band track that band's predictability in the
training marginal? Design: with signal power, occurrence and tone structure held fixed, sweep the
SNR of band-limited noise on one band across ~4 levels, plus the off-support endpoint. Read `r_f(R)`
at that band.
**Supports the claim if:** monotone in training predictability, starting at ≈1 for the clean arm.
**Also a check on itself:** report `r_f` against the *fit gate* and the seed spread (contracts 8, 7).
**Cost:** ~15 runs × ~1 min at the smoke config.

### P2 — invariance in `f` (the money claim)

**Question.** Does the damping follow the *perturbed* band or the frequency? Design: P1's
perturbation applied to a **low** band, giving a 2×2 (perturbed band × predictability), then extend
to a proper `f` sweep with an intermediate band.
**This is the step that converts "predictability matters" into "frequency per se is not the
variable".** Without it the claim is not established, however clean P1 looks.

### P3 — residual in `f` (the claim's boundary)

**Question.** With predictability matched across frequencies, what is left? Report `r_f` across
`b = 1..8` at matched θ, plus `Δφ` alongside. Any residual is either the transient (§2) or
architectural — say which, and give it a number. **The honest wording of the claim depends on this.**

### P4 — controls

- **Reversibility.** P1's clean arm already covers it; state it explicitly.
- **RevIN twin.** Mandatory before any claim about real TSFMs (contract 14, §1).
- **Capacity sweep** (`hidden`): rules out "this is just a small model".

### P5 — mechanism (the blog's payoff)

**Gradient probe.** Measure which frequencies actually lower the loss. This upgrades the behavioural
claim to a mechanistic one: the model is doing predict-the-conditional-mean, and the high band
contributes little *because* it is unpredictable.

## 6. Infrastructure

| Path | Contents |
| :-- | :-- |
| `src/fbias/data.py` | corpus builders |
| `src/fbias/probes.py` | the §4 metrics + the oracle |
| `src/fbias/cli/train.py` | training loop; windows **must** be `context_size + patch_size` |
| `experiments/<id>/scripts/`, `runs/` | driver + `state_dict` + sidecar JSON |
| `experiments/README.md` | run log |

**Config.** Smoke/development: `ctx 512 / k 32 / hidden 32 / 2L / 4 heads / SGD 1e-2 / batch 64`
(both `ctx/k` choices give `P = 16` patches, so the frequency grid is identical to the lab's and
only Nyquist differs). Headline runs: the lab's canonical *direct* config
`ctx 1024 / k 64 / hidden 64`.

### Corpus builder: what to build now, what to defer

The generator is being rewritten. **Build only what P0–P2 need; add an axis when a specific
experiment requires it.** Letting the generator become the project is the failure mode this repo
inherits from the lab (50 iterations, most of them infrastructure).

**Required for P0–P2:**

| Requirement | Why |
| :-- | :-- |
| Return `(observed_window, clean_target_patch)` | §4's measurement definition |
| **Band-limited noise** with sweepable SNR, band chosen per arm | contracts 12, 13 |
| **Signal power fixed** across arms; noise added on top; probe normalised to signal power | contract 9 |
| **Off-support** variant (a frequency held out entirely) | θ's extreme endpoint (C5) |
| Fixed frequency set within a corpus; per-window phases | contract 10 |
| Frequencies spaced ≥ 1 cycle/patch; report `b` and `Δφ` | contracts 4, 6 |
| Report variance explained on a held-out same-marginal corpus | contract 8 |

**Deferred to P6 (robustness) — do not build yet:**

| Axis | Note |
| :-- | :-- |
| Occurrence vs marginal power density | currently **identical by construction** (equal per-tone power ⇒ marginal power ∝ occurrence; measured `power/count` std = 1e-2 over a 3469× range). Decoupling needs its own experiment |
| Conditional amplitude (amplitude given presence) | the amplitude–frequency coupling axis |
| Per-window tone count | per-signal complexity ≠ population coverage |
| Phase structure (independent vs locked) | untested anywhere; plausibly a large effect |
| Bin alignment (integer vs off-grid `b`) | matters for the transient, not the fixed point |
| Non-stationarity: offset, trend, level shift | our model has no RevIN, so this is currently assumed away |
| Per-window power CV | bandwidth-dependent (measured 15% tight `peaks` vs 5.6% `broad`); report it when a corpus has clustered bandwidths |

## 7. Robustness backlog (P6+)

Only after P0–P5. Each is a small experiment, not a research line: phase structure; tone count;
occurrence/power decoupling; amplitude coupling; bin alignment; non-stationarity.

## 8. Related work

| Work | Claim | Relation |
| :-- | :-- | :-- |
| FreIE (ICDM 2025) | Spectral bias stems from **autocorrelation** | A predictability manipulation separates "autocorrelation" from "data predictability" |
| Basri et al. (ICML 2020) | NTK: the bias relates to **input density** | The occurrence arm of P6 is the direct empirical test |
| Fredformer (KDD 2024) | The bias stems from **over-attention to high-energy frequencies** | The conditional-amplitude arm of P6 tests this |
| Maddix et al. (arXiv 2510.19236) | The temporal bias induced by patching | P2's `f` sweep, since `Δφ` is `k`-dependent |
| Yu et al. (arXiv 2510.03358) | Rank structure of TS transformers | the `dim1+2` measurement |
