# Experiment Design: the origin of frequency bias in TSFMs

This file is the **experiment design contract**: scope, target claims, mandatory controls, and metric definitions.

- Run log / results / conclusions → `experiments/README.md`
- Per-experiment intent, config, and pass criteria → `experiments/<id>/README.md`

## 0. Scope

| | |
| :-- | :-- |
| **Question** | Why do next-patch forecasting models exhibit frequency bias? |
| **Model** | `SimTFM`: input-space next-patch, **no RevIN**, **no SIGReg**, RoPE, `context_size + patch_size` training windows |
| **Data** | synthetic corpora with controlled spectra |
| **Output** | a mechanistic explanation with synthetic evidence. **No fix** — real-data forecasting gains are out of scope |
| **v1 goal** | **reproduce the lab's conclusions** on a clean testbed, not propose new ones |

Why reproduce first: `SimTFM` drops two components the journal showed to be load-bearing — RevIN and SIGReg. Until the basic phenomena are confirmed to survive without them, any new claim is uninterpretable.

## 1. Starting point: conclusions already established in the lab

From `projects/temporal-batch-comp-sigreg`. v1 rebuilds this table on `SimTFM`.

| ID | Conclusion | Lab evidence | Reproduced by |
| :-- | :-- | :-- | :-- |
| C1 | The bias is on the **prediction side**, not a resolution loss | It.27-28: on `monash-direct-h32.pt` and `geom-rich-input.pt` — **neither of which was trained on the `f=2+f=128` mixture it was probed on** — the high band is copyable at r²=**0.9995** while the model emits r²≈0 / **7%** amplitude (r²=**−0.46** on the rich-corpus model); the low-frequency slope is kept at ~54%. It.29 is the matching control in the *other* regime: trained and probed on one pure-tone corpus, retention is **0.988–0.998**. The training marginal, not determinism, is the variable | `1_pred_side` |
| C2 | **Patching encodes frequency**; the transformer does not | It.15: at `k=1`, f=16/32/64 have identical geometry, plane overlap **1.000** | `4_patch_granularity` |
| C3 | Shrinkage is a **learned, position-invariant prior** | It.14: the ratio is flat across forecast positions `p=2..14` (f=32 **0.537**, f=64 **0.524**) | `2_pos_invariance` |
| C4 | Shrinkage **vanishes** on deterministic corpora; the ordering is set by `Δφ` | It.29-30: after 15k steps finals are **0.988–0.998**, and the `Δφ=π` frequency is always the laggard | `3_deterministic_dphi` |
| C5 | **Coverage**: frequencies absent from training stay damped even when deterministic | It.31: the uneven rich corpus converges at 0.865, worst cluster 0.715 | `6_coverage` |
| C6 | The phase-ring geometry is **purely architectural** | It.9: rings survive in a direct input-space model with no z-regularizer; It.26: `dim1+2 ≥ 0.94` | `5_ring_geometry` |

**This is a conceptual reproduction, not a numerical one.** `SimTFM` has no RevIN; the journal's E3 showed no-revin gives eccentric rings and collapses the trend tilt. It has no SIGReg; It.49-51 showed SIGReg stretches rings into ellipses. So C6's geometry is *expected* to diverge from the lab — that is a **documented difference, not a failure**. Pass criteria are therefore directional, never exact values.

**Decision rule.** Every headline lab result was collected on a model **with RevIN** (the no-revin run was only an ablation). So if `1_pred_side` (C1) does **not** reproduce here, the lab's conclusion is not thereby refuted — a RevIN twin must be run first to separate "architectural difference" from "wrong conclusion". This is v1's largest interpretive risk.

## 2. First, separate two different kinds of bias

The draft plan conflated these into a single metric. This is the most important correction:

- **Transient (optimization order).** Low frequencies are learned first. This is set by `Δφ = 2π·f·k/ctx (mod 2π)`, **not by `f`** — It.29-30 showed on purely deterministic corpora that it is transient and disappears at convergence.
- **Fixed point (converged damping).** Set by the **conditional predictability** of a frequency in the training marginal. `Δφ` cannot explain it. This is what It.31 and It.14 measure.

**Consequence:** the `(high-freq error − low-freq error) / (high-freq error + low-freq error)` "spectral bias index" is unusable. Error is a periodic function of `Δφ mod 2π`, so differencing high against low buckets averages over a periodic function — changing `k` or `ctx` flips the sign. The two must be measured separately, and frequencies reported in `cycles/patch` throughout.

## 3. Experiment contracts

These are confounds the lab paid dozens of iterations to learn. They are requirements, not suggestions:

1. **Fix the horizon in patches.** It.35/36: the "smaller patches are better" result turned out to be an artifact of mismatched horizons. Configs are only comparable at the same number of predicted patches.
2. **Report the converged state and log the transient.** It.21: on one fixed config, going past 10k steps moved the `k=1` retention from 0.574 to 0.843 — half the apparent "bias" was under-training.
3. **Count frequencies evenly in the probe corpus.** It.45/49: the "elliptical ring" conclusion drawn from an uneven cluster density was misattributed; the real cause was training duration.
4. **Probe frequencies must be spaced ≥ `ctx/k` apart** (i.e. 1 cycle/patch). It.34: the orthogonality threshold is exactly `d = k` (`Δf=1` → overlap 0.985, `Δf=8` → 0.29, `Δf=16` → **0.04**). Any closer and DFT bins leak into each other, making per-frequency retention meaningless.
5. **Every experiment carries an oracle upper bound.** It.46: a hand-built detect→rotate→recombine pipeline on frozen `z` reaches MSE **0.043–0.070**, while the transformer head reaches **0.44–1.55** — a 10–35× gap. Without an oracle, "did not learn" and "cannot be learned" are indistinguishable.
6. **Report in `cycles/patch`** and state `Δφ` alongside.
7. **At least 3 seeds per config**, reporting spread. A single run's difference is not a conclusion.
8. **Gate every retention reading on the model's fit.** `r_f` is only meaningful for a model that has actually learned its corpus. Report variance explained (`1 − mse / var(target_patch)`) on a held-out corpus of the *same marginal* alongside every `r_f`. A model converged to the mean predictor yields `r ≈ 0` at **every** band with `r8/r1 ≈ 1`, which reads exactly like uniform damping and is not. Earned 2026-09-13: a 32-tone corpus with per-window redrawn frequencies sat at **−0.005** variance explained after 20k steps and would have been written up as a result without this gate.
9. **RMS-match the probe to the training corpus.** Every tone in `make_mixture` has amplitude 1, so a corpus's RMS scales as `sqrt(n_tones)`; a model calibrated to one scale systematically over- or under-predicts on a probe drawn at another, and a perfectly-tracking model reads `r ≈ 2.0` when an 8-tone corpus is probed with a 2-tone signal. `make_mixture(..., normalize=True)` and `make_broad` both divide by `sqrt(n_tones)`, holding RMS at ≈0.707 regardless of tone count. This trap corrupted two independent readings before it was caught.
10. **"Broad marginal" must mean a structured frequency family, not a uniform draw.** Redrawing each window's frequencies uniformly across the band makes the window incompressible. At `ctx 512 / k 32 / hidden 32`, a 32-tone uniform-redraw corpus plateaued at **−0.005** variance explained (mean predictor) while the *same* 32-tone set held fixed across windows reached **+0.691** and was still rising at 20k — so the failure was per-window frequency identification, not marginal breadth. The lab's rich corpus samples from 12 named clusters, i.e. a low-dimensional family; reproduce that structure rather than a uniform draw.

## 4. Variables and metrics

Notation: `f` is in cycles per window (window length `ctx`). Hence

```
cycles/patch = f · k / ctx        Δφ = 2π · f · k / ctx (mod 2π)
```

| Metric | Definition |
| :-- | :-- |
| **retention** `r_f` | At a **fixed forecast position**, the ratio of predicted to true complex amplitude in bin `f`: `\|P̂_f\| / \|P_f\|`. This is "radius shrinkage" in the frequency domain. |
| **phase error** | `angle(P̂_f) − angle(P_f)`, unwrapped |
| **ring** `dim1+2` | Per `f`, collect the latents around the phase loop, PCA, and take the variance fraction of the first two components |
| **plane overlap** | Grassmann `mean(s²)` between the top-2 subspaces of two frequencies. 1 = identical plane, 0 = orthogonal, 0.5 = sharing one direction |
| **transient metric** | the `r_f(t)` curve, and the number of steps to reach a threshold `ε` |

For the oracle, `r_f` is defined as `‖P̂_f^oracle‖ / ‖P_f‖`, which is what separates "is it in the representation" from "does the head use it".

## 5. v1 experiment list

Ordered by infrastructure dependency. `0_init` already exists in the repo.

### `0_init` — infrastructure + smoke + first reproduction

Bring the harness up: `data.py` corpus builders, `cli/train.py` training loop (T+patch_size windows, input-space shifted MSE, SGD), `probes.py` metrics. Run once at the smallest config, confirm training reduces loss, and confirm retention can be read out.
**Pass criterion**: end-to-end run producing a readable sidecar JSON.

### `1_pred_side` — the bias is on the prediction side (C1, headline)

A mixed corpus with a low- and a high-frequency component, spaced ≥ 1 cycle/patch so the bins are separable. After training, measure two things: whether an oracle on frozen `z` can reconstruct the next patch's high-frequency band, and how much the model's own forecast retains.
**Pass criterion**: oracle r² ≥ 0.9 in the high band, model retention ≤ 0.3, and a gap ≥ 3× between them.
**By-product**: this repo has no SIGReg, so if the readout gap reproduces here, the hypothesis "SIGReg is the cause" is exonerated outright (one branch of journal backlog #1).

### `2_pos_invariance` — shrinkage is position-invariant (C3)

Fix `f`, sweep the forecast position. Rules out context-length bottleneck explanations.
**Pass criterion**: relative variation of `r_f` across positions < 20%.

### `3_deterministic_dphi` — deterministic corpora and the Δφ ordering (C4)

Pure-tone / two-tone deterministic corpora, several `f`, probing every N steps during training.
**Pass criterion**: at convergence every `f` has retention ≥ 0.9; the ordering of convergence steps correlates with `Δφ mod 2π`, with `Δφ≈π` slowest.

### `4_patch_granularity` — patching encodes frequency (C2)

`k=1` vs `k>1` over the same set of `f`.
**Pass criterion**: at `k=1` plane overlap across `f` ≈ 1.000; at `k>1` substantially below 1.

### `5_ring_geometry` — the phase ring (C6)

**Pass criterion**: `dim1+2 ≥ 0.90` in the mid band. Eccentricity or collapse caused by the missing RevIN is recorded as a difference per §1.

### `6_coverage` — coverage (C5)

Hold out some `f` entirely from training, then probe those `f` with pure tones.
**Pass criterion**: off-support retention is substantially below on-support, and remains so on deterministic corpora.

## 6. Infrastructure

| Path | Contents |
| :-- | :-- |
| `src/fbias/data.py` | corpus builders: pure tone, multi-tone mixtures, rich clusters, off-support variants |
| `src/fbias/probes.py` | the §4 metrics plus the oracle (ridge / MLP on frozen `z`) |
| `src/fbias/cli/train.py` | training loop; windows **must** be `context_size + patch_size` |
| `experiments/<id>/scripts/` | driver scripts for that experiment |
| `experiments/<id>/runs/` | `state_dict` + sidecar JSON |
| `experiments/README.md` | run log |

**Default config**: the lab's canonical *direct* configuration — `ctx 1024 / k 64 / hidden 64 / 2L / 4 heads / SGD 1e-2 / batch 64` — so runs stay comparable against the journal. Smoke runs use `ctx 512 / k 32 / hidden 32`.

## 7. v2 candidates (new claims; to be settled after v1)

- **Amplitude axis.** Fix `f` and predictability, vary only the amplitude distribution (lognormal). This directly tests Fredformer's "over-attention to high-energy frequencies". The journal only measured this on the encoder side (It.5); the readout side is untested.
- **Gradient probe.** Measure which frequencies actually lower the loss, as mechanistic evidence for the "predictability prior" account. The journal never did this.
- **Mechanism of the readout gap.** Why the head ignores frequency information that `z` demonstrably holds. This is the main v2 target.
- **Frequency density.** Uniform / log-uniform / bimodal.

## 8. Related work

| Work | Claim | Relation to this design |
| :-- | :-- | :-- |
| FreIE (ICDM 2025) | Spectral bias stems from **autocorrelation** | Controlling coverage and noise separates the contributions of "autocorrelation" from "data predictability" |
| Basri et al. (ICML 2020) | NTK theory: the bias relates to **input density** | `6_coverage` is a direct empirical test |
| Fredformer (KDD 2024) | The bias stems from **over-attention to high-energy frequencies** | v2's amplitude axis tests this directly |
| Maddix et al. (arXiv 2510.19236) | The temporal bias induced by patching | `4_patch_granularity` |
| Yu et al. (arXiv 2510.03358) | Rank structure of TS transformers | the low-dimensionality measured in `5_ring_geometry` |
