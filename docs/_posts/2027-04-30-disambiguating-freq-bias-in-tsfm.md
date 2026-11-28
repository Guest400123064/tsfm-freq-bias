---
layout: distill
title: "Model Flaw or Data Property? Disambiguating Frequency Bias in Neural Forecasting"
description: Controlled experiments manipulate the data-generating process while holding each neural forecaster fixed, separating signal spectra, latent map complexity, and finite-budget learning schedules.
date: 2027-04-30
future: true
htmlwidgets: true
authors:
  - name: Anonymous
bibliography: 2027-04-30-disambiguating-freq-bias-in-tsfm.bib
toc:
  - name: Introduction
  - name: The Data Spectrum Alone Can Organize the Learning Schedule
  - name: Can Noise Stretch a Schedule into an Apparent Deficit?
  - name: Latent Map Complexity Also Shapes Learning
  - name: Conclusion
  - name: Appendix
---

## Introduction

Neural time-series forecasters frequently exhibit larger errors on fast oscillations, and papers routinely invoke “frequency bias” to explain the phenomenon, often citing classical spectral bias.<d-cite key="rahaman2019spectral,xu2020frequency,basri2019convergence,basri2020frequency,piao2024fredformer,he2025sempo,sun2025freie"></d-cite> Yet these two bodies of work describe different combinations of the problem. Classical spectral bias concerns **learning dynamics on a target function**: neural networks tend to fit low-frequency components of a map before high-frequency ones. Empirical reports in time-series forecasting usually measure **end-state errors on an input or output signal**: a trained model retains less amplitude or incurs higher error in high-frequency bands. Collapsing these distinctions under one label invites inaccurate attribution of empirical results.

Untangling this confusion starts with separating what **frequency** refers to. On one axis lies the **spectrum of the signal itself**: carrier positions, band power, and signal-to-noise ratio (SNR). On another lies the **forecasting map** from history to future and the complexity of its underlying dynamics. The two can vary independently. A pure tone $$x_t=\cos(\omega t+\phi)$$ obeys the linear recurrence

$$
x_{t+1}=2\cos(\omega)x_t-x_{t-1}
$$

for every carrier $$\omega$$. Rapid signal oscillation can therefore coexist with a simple forecasting map.

Alongside this distinction, **bias** depends on the stage of training. Finite-budget learning speed differs from converged end-state behavior. Under squared loss, shrinking unpredictable noisy components toward the conditional mean is statistically optimal.

To test these mechanisms in isolation, we use a standard next-patch Transformer, `SimTFM`, and manipulate the **synthetic data-generating process** across full learning trajectories. Each experiment holds the architecture, objective, optimizer, and training budget fixed across its arms. The Appendix gives full architecture and training details. Three findings emerge:

1. **Data spectra organize signal-domain schedules.** When spectral power is randomly permuted across frequencies, learning speed and fixed-budget error closely track allocated band power, with negligible dependence on carrier frequency.
2. **Noise stretches recovery schedules.** Full-spectrum Gaussian noise at fixed signal power substantially slows clean-signal recovery, allowing an unfinished schedule to resemble a persistent spectral deficit.
3. **Latent map complexity also slows learning.** Increasing the nonlinear harmonic order of a latent phase transition slows learning relative to a pure rotation. Phase modulation simultaneously alters realized spectra, making this a bounded analogy to classical spectral bias.

Throughout the post, we use the following notation to separate signal bands, amplitude and phase errors, and latent map complexity:

| Symbol                                                                                                   | Meaning                                                             |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| $$L,k$$                                                                                                  | Context length 512 and patch size                                   |
| $$b=fk/L$$                                                                                               | Normalized carrier frequency in cycles per patch                    |
| $$r_b=\lvert Z_{\mathrm{pred},b}\rvert/\lvert Z_{\mathrm{target},b}\rvert$$                              | Coherent amplitude retention at band $$b$$                          |
| $$\Delta k_b=\lvert Z_{\mathrm{pred},b}-Z_{\mathrm{target},b}\rvert/\lvert Z_{\mathrm{target},b}\rvert$$ | Normalized complex coefficient error, capturing amplitude and phase |
| $$m$$                                                                                                    | Nonlinear harmonic order of the latent phase transition             |

## The Data Spectrum Alone Can Organize the Learning Schedule

Fredformer observed that band-wise forecasting error tracks marginal spectral power and introduced an architectural intervention in response.<d-cite key="piao2024fredformer"></d-cite> We begin with a simpler question: how much structure emerges from the data spectrum when the model stays fixed?

**Design.** Each run generates 32 deterministic sinusoidal tones with independent phases, fixed total power, and no observation noise. Eight power levels span a $$128\times$$ ratio, with four tones per level. A fresh random permutation pairs frequencies with power levels in each run. Carrier positions are expressed in cycles per patch,

$$
b=\frac{fk}{L},
$$

where $$f$$ counts cycles per context, $$k$$ is patch size, and $$L=512$$ is context length.

For each tone, we recover a complex coefficient $$Z_b$$ from the predicted and target patches. Our primary error measure combines amplitude and phase discrepancies:

$$
\Delta k_b=\frac{|Z_{\mathrm{pred},b}-Z_{\mathrm{target},b}|}{|Z_{\mathrm{target},b}|}.
$$

**Results.** Learning curves separate by allocated power. Across 96 band–run measurements at step 32,768, the Spearman correlation of $$\Delta k$$ is **−0.992 with log power** and **−0.090 with carrier frequency**. The mean time to reach $$\Delta k\leq0.5$$ follows

$$
t\propto P^{-1.083},\qquad R^2=0.999.
$$

This exponent closely matches a simple linear-readout model in which a band’s curvature scales with its power, giving a $$P^{-1}$$ time-to-threshold schedule.

{% include figure.liquid path="assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/power.png" class="img-fluid" alt="Learning curves, threshold times, and final errors for the power-allocation experiment." %}

_Figure 1. Allocated power organizes learning in a noise-free corpus. **(a)** Mean coefficient error by power level. **(b)** Time to reach $$\Delta k\leq0.5$$; marker shapes denote runs and the line shows the power-law fit. **(c)** Final error by carrier position, colored by power. The loudest level reaches the threshold by the first checkpoint at 128 steps._

**Reversing the assignment.** A second control makes the carrier comparison direct. The **red** corpus places loud tones at low frequencies; the **anti-red** corpus places them at high frequencies. Both use the same power multiset and noise variance. Final clean errors at matched power levels differ by at most **0.018**. Excluding the weakest level, which never reaches $$\Delta k\leq0.5$$, the maximum difference is **0.011**. After removing power-level means, the residual Spearman correlation between carrier frequency and final error is only **0.036**.

{% include figure.liquid path="assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/assignment.png" class="img-fluid" alt="Red and anti-red frequency-power assignments produce closely matching final errors and threshold times." %}

_Figure 2. Reversing which carriers are loud leaves the learning profile nearly unchanged. **(a)** Final clean coefficient error by power level. **(b)** Time to clean $$\Delta k\leq0.5$$. Small marks show individual band measurements; lines show means. Level 0 is absent from panel b because neither assignment reaches the threshold._

Within this controlled family, the deficit follows allocated power. Reversing the carrier–power assignment changes the spectral color of the data without producing a corresponding preference for low or high carrier positions.

## Can Noise Stretch a Schedule into an Apparent Deficit?

The first experiment identifies a strong schedule effect in clean data. Large forecasting models also train for many steps, so we need a mechanism that can make these schedules long enough to matter in practice. Real-world spectra often combine declining band power with broadband observation noise, leaving high-frequency bands with low SNR. Our next experiment tests the noise component of that hypothesis at fixed signal power.

**Design.** We generate 32 equal-power tones and add independent Gaussian white noise to both context and target. Four arms vary global noise RMS from zero to the signal RMS while holding carriers and signal power fixed. Every band therefore has the same nominal SNR within an arm.

Evaluation uses **clean-context-to-clean-target probes**. These probes measure recovery of the deterministic tones without asking the model to reproduce unpredictable noise. Alongside $$\Delta k$$, we report coherent amplitude retention,

$$
r_b=\frac{|Z_{\mathrm{pred},b}|}{|Z_{\mathrm{target},b}|}.
$$

Retention isolates amplitude shrinkage; $$\Delta k$$ also includes phase error.

**Results.** Mean time to clean-probe $$\Delta k\leq0.5$$ rises from **723 steps without noise to 3,119 steps at equal noise and signal RMS**, a **$$4.3\times$$ slowdown**. At step 32,768, mean clean retention is **0.995** without noise and **0.844** at the highest noise level. The highest-noise arm never reaches $$\Delta k\leq0.1$$ within the budget.

{% include figure.liquid path="assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/snr.png" class="img-fluid" alt="Clean and noisy probes under four noise doses, together with time to clean-error thresholds." %}

_Figure 3. Noise lengthens clean-signal recovery at fixed signal power. **(a)** Clean-probe coefficient error. **(b)** Error when the context contains noise and the target is clean. **(c)** Mean time to three clean-probe thresholds. Missing points indicate thresholds that were not reached._

The highest-noise curves continue improving through the final checkpoint. Their final attenuation may also include a mismatch between noisy training inputs and clean probe inputs. We therefore read these measurements as schedule evidence at a fixed budget.

The experiment establishes that broadband noise can strongly delay deterministic-signal recovery. Connecting this mechanism to production forecasters requires band-wise SNR measurements in real corpora and evidence about convergence. If real high-frequency bands combine low power with low SNR, an unfinished learning schedule offers one plausible explanation for their persistent-looking deficits.

## Latent Map Complexity Also Shapes Learning

The first two studies manipulate the signal spectrum. We now hold carrier positions fixed and change the latent rule that generates each oscillator:

$$
x_i(t)=A_i\cos\theta_i(t),\qquad
\theta_i(t+1)=\theta_i(t)+\omega_i+
\sum_{j=1}^{m}\gamma_j\sin(j\theta_i(t)+\phi_j).
$$

Here $$m=0$$ is a pure rotation. Levels $$m=1,2,3$$ add nonlinear harmonics to the phase transition while keeping a single scalar phase state per carrier. Carriers, total power, and modulation RMS stay fixed. Initialization and batch order are paired across map orders within each seed.

{% include figure.liquid path="assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/map-instrument.png" class="img-fluid" alt="Waveforms, phase-map derivatives, and realized spectra for four latent map orders." %}

_Figure 4. The map-order intervention illustrated for one carrier at $$b=1$$. **(a)** Waveforms, vertically offset by map order. **(b)** Derivative of the latent phase transition. **(c)** Realized spectra. Phase modulation introduces sidebands even though total power and modulation RMS are fixed. Training uses carriers 2.5, 4.5, and 6.5 cycles per patch._

Because modulation spreads energy into sidebands, this study evaluates the full predicted waveform. We normalize MSE by target power and measure variance explained separately.

At normalized MSE $$\leq0.02$$, $$m=0$$ reaches threshold in **384–512 steps across seeds**, compared with **768 steps for $$m=3$$**. The intermediate orders take 512–640 steps. Final variance explained decreases monotonically from **0.9997** at $$m=0$$ to **0.9964** at $$m=3$$, and every arm continues improving through step 32,768.

{% include figure.liquid path="assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/map.png" class="img-fluid" alt="Learning curves, threshold times, and final errors as latent phase-map order increases." %}

_Figure 5. Higher latent map order slows learning relative to a pure rotation, with the clearest separation between $$m=0$$ and $$m=3$$. **(a)** Held-out normalized MSE; faint curves show seeds and solid curves show means. **(b)** Time to three thresholds. **(c)** Final normalized MSE. Small marks show seeds and black dashes show means._

The latent transition contains exactly $$m$$ nonlinear harmonics. The forecasting map seen by the model also includes phase inference and a cosine observation function, giving it Fourier content beyond those $$m$$ harmonics. The realized signal spectrum changes as well. These results support a bounded claim: **increasing latent phase-map complexity slows learning relative to pure rotation**. A direct reproduction of the F-Principle would require tracking Fourier components of the target map itself.

## Conclusion

Within each experiment, the model architecture, objective, and training recipe never changed. Manipulating the data-generating process alone was enough to reshape how the forecaster behaved across both signal frequencies and latent map complexity through band power, full-spectrum noise, and latent harmonic order.

That is why treating “frequency bias” as a single umbrella term is so costly. When signal spectra, map complexity, and finite-budget learning dynamics are collapsed together, **properties of the data and training schedule are easily mistaken for inherent deficiencies of the model**. Spectral structure clearly matters in neural forecasting, but it becomes explanatory only when we state explicitly which frequency and which stage of training we are measuring.

## Appendix

### Architecture and optimization

All studies use `SimTFM`, a patched Transformer with a nonlinear patch embedding, two decoder layers, four attention heads, rotary position embeddings, RMS normalization, and a linear patch predictor. The context contains 512 samples. Models are trained with next-patch MSE for 32,768 SGD steps, using three seeds and checkpoints every 128 steps. Each training corpus contains 4,096 windows; probes contain 512 held-out windows.

| Setting          | Signal studies | Latent-map study |
| ---------------- | -------------: | ---------------: |
| Patch size $$k$$ |             64 |               16 |
| Hidden width     |            128 |               64 |
| Batch size       |            256 |              128 |
| Learning rate    |           0.04 |             0.01 |

Within each study, these settings are constant across arms. The latent-map study additionally pairs initialization and batch order across map orders within each seed.

### Signal generation and coefficient recovery

Signal carriers start on the grid $$b=0.5,1.5,\ldots,31.5$$ cycles per patch and receive independent jitter in $$[-0.25,0.25]$$. Total signal power is 0.5. In the power study, relative band weights follow $$1,2,4,\ldots,128$$ and are randomly permuted across carriers in each seed. All 32 tones have independent phases and no observation noise.

The red/anti-red control uses the same power ladder and fixes absolute noise RMS to 0.12524, calibrated so the weakest band has in-band SNR 1. The noise-dose study assigns equal power to every tone and uses noise/signal RMS ratios $$0,0.25,0.5,1$$. Training noise is independently sampled for each context and target.

A 64-sample patch has exactly enough real degrees of freedom to recover 32 complex tone coefficients. We jointly invert the full cosine/sine basis and then compute

$$
r_b=\frac{|Z_{\mathrm{pred},b}|}{|Z_{\mathrm{target},b}|},\qquad
\Delta k_b=\frac{|Z_{\mathrm{pred},b}-Z_{\mathrm{target},b}|}{|Z_{\mathrm{target},b}|}
=|r_b e^{i\Delta\psi_b}-1|.
$$

Here $$\Delta\psi_b$$ is phase error. Clean probes contain clean context and clean target; noisy probes add noise only to the context while retaining a clean target. All threshold times are first-hitting times on the 128-step checkpoint grid. In the power study, excluding the loudest level, which reaches $$\Delta k\leq0.5$$ by the first checkpoint, changes the fitted exponent from 1.083 to 1.078.

### Latent phase maps

The latent-map study uses carriers $$b=2.5,4.5,6.5$$ cycles per patch and modulation scale $$\beta=0.1$$. For $$m>0$$, every active harmonic has

$$
\gamma_j=\beta\sqrt{\frac{2}{m}},\qquad j=1,\ldots,m,
$$

which fixes RMS phase modulation across map orders. Harmonic phases are shared between training and probe corpora. Each window is RMS-normalized to power 0.5, and the generator asserts positive phase advance to prevent locking. The recurrence is evaluated in float64 before casting observations to float32.

Normalized MSE in Figure 5 is waveform MSE divided by target power 0.5. Variance explained is computed against the variance of each held-out target patch. Figure 4 is a separate one-carrier diagnostic at $$b=1$$; it illustrates the intervention and its realized sidebands without entering training.
