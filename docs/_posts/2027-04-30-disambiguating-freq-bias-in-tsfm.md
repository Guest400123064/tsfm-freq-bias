---
layout: distill
title: Disambiguating Frequency Bias in Time-Series Forecasting
description: Signal spectrum, map complexity, learning-time bias, and converged behavior are different axes. Controlled forecasting experiments show why they should not be collapsed into one notion of frequency bias.
date: 2027-04-30
future: true
htmlwidgets: true
authors:
  - name: Anonymous
bibliography: 2027-04-30-disambiguating-freq-bias-in-tsfm.bib
toc:
  - name: Two words, four questions
  - name: What prior work measures
  - name: A controlled forecasting instrument
  - name: In the signal domain, power organizes the schedule
  - name: When slow learning looks like a fixed preference
  - name: Latent map complexity is a distinct axis
  - name: Which frequency, which bias?
---

## Two words, four questions

“Frequency bias” has only two words, but in time-series forecasting each word can refer to a different object. **Frequency** may describe the spectrum of an input signal or the Fourier content of the map from history to future—or, more loosely, the complexity of that map. **Bias** may describe how quickly a component is learned or how much of it survives in a converged model. When these distinctions are collapsed, a learning-dynamics result about functions can become an architectural claim about signals.

The classical spectral-bias literature primarily makes the first kind of statement. Neural networks can learn low-frequency components of a target function before high-frequency ones, and input density can affect those convergence rates.<d-cite key="rahaman2019spectral,xu2020frequency,basri2019convergence,basri2020frequency"></d-cite> Here, frequency belongs to the **function being learned**, and bias concerns **learning dynamics**.

Time-series work often measures something else: the error or retained amplitude of Fourier components of an observed series. Fredformer diagnoses disproportionate focus on high-energy bands; SEMPO targets the same axis with energy-aware spectral decomposition.<d-cite key="piao2024fredformer,he2025sempo"></d-cite> Related sequence-modeling work also uses frequency to describe architectural transfer functions.<d-cite key="yu2025tuning"></d-cite> These measurements are valuable, but a result about the frequency of a learned map does not, by itself, explain suppression of high-frequency content in an input signal.

A pure tone makes the difference concrete:

$$
x_t=\cos(\omega t+\phi),\qquad
x_{t+1}=2\cos(\omega)\,x_t-x_{t-1}.
$$

The signal may oscillate at any carrier frequency $$\omega$$, yet its one-step forecast is a fixed linear functional of two past samples. Changing the carrier changes the single coefficient $$2\cos\omega$$ and the rotation angle, not the function class. A high-frequency signal therefore need not define a high-frequency or complex forecasting map. With multiple tones and randomized frequencies, the target becomes a conditional family indexed by latent frequency states rather than one global linear functional; the narrower point still holds: carrier frequency does not by itself raise map complexity.

There is a second, orthogonal distinction. A component learned slowly at a fixed training budget need not remain poorly learned at convergence. Conversely, a converged predictor may correctly suppress an unpredictable component: the conditional mean, not the noisy realization, minimizes expected squared error. A final-checkpoint deficit is not automatically a converged architectural preference.

These distinctions produce four different questions.

| Frequency of… | Learning dynamics | Converged behavior: floor or preference |
| --- | --- | --- |
| **Input or output signal** | How quickly is each spectral band learned? **Power and SNR studies** | Does a deficit remain after convergence, and is it optimal shrinkage or an optimization failure? **Not tested here** |
| **Target function or latent map** | Target-function frequency: classical spectral bias. Latent phase-map order: **map study**—an analogy, not the same variable. | Is there a non-removable map-dependent preference? **Not tested here** |

Our goal is not to prove that frequency never matters, nor to explain every production checkpoint. It is to separate these claims and test which variables move them in controlled forecasting experiments.

## What prior work measures

The classical results concern function fitting. Spectral bias and the F-Principle describe the order in which a network fits Fourier components of a target; Basri et al. make convergence rate the explicit object and show that non-uniform input density matters.<d-cite key="rahaman2019spectral,xu2020frequency,basri2019convergence,basri2020frequency"></d-cite> Other work locates spectral preferences in architecture or initialization through neural tangent kernels and their null spaces.<d-cite key="bietti2019inductive,yang2019finegrained,yu2025tuning"></d-cite> These are not interchangeable with a statement about the Fourier spectrum of a time series.

Forecasting adds a regime change. In fully observed function fitting, the target is available on the training domain. In forecasting, the target patch is held out and may have irreducible conditional variance. The conditional mean is then the relevant squared-error reference, and shrinkage toward it can be correct rather than pathological.

The measurements in the forecasting literature also differ. Fredformer relates component error to energy, SEMPO targets energy-aware spectral utilization, and FreIE reports frequency-domain learning behavior across time-series models.<d-cite key="piao2024fredformer,he2025sempo,sun2025freie"></d-cite> Filter then Attend frames its intervention through spectral filtering, with learned filters that need not be low-pass.<d-cite key="dayag2025filter"></d-cite> Across this work, “frequency” can denote band position, band energy, a learned filter, or an architectural transfer function. Our review is not a rejection of these approaches; it is a reason to stop treating their variables as synonyms.

## A controlled forecasting instrument

We use a family of small patched Transformers, `SimTFM`, trained to predict the next patch from a 512-sample context. Patch size and width vary across studies. These models are deliberately instruments rather than production foundation models. All three studies use three seeds and a 32,768-step training budget; learning curves are part of the result, not just the final checkpoint.

For signal-domain experiments, frequencies are expressed in cycles per patch, $$b=fk/L$$, where $$f$$ is cycles per context, $$k$$ is patch size, and $$L$$ is context length. We recover all tone coefficients jointly from each patch. Two readouts distinguish amplitude from total coefficient error:

$$
r_b=\frac{|Z_{\mathrm{pred},b}|}{|Z_{\mathrm{target},b}|},
\qquad
\Delta k_b=\frac{|Z_{\mathrm{pred},b}-Z_{\mathrm{target},b}|}{|Z_{\mathrm{target},b}|}
=|r_b e^{i\Delta\psi_b}-1|.
$$

Retention $$r$$ isolates coherent amplitude; $$\Delta k$$ also contains phase error $$\Delta\psi$$. A model with the correct amplitude but wrong phase can therefore show a large $$\Delta k$$. We use clean context-to-clean-target probes to measure recovery of deterministic signal even when training data contain noise. This avoids mistaking appropriate denoising for a failure to represent a band.

## In the signal domain, power organizes the schedule

Our first experiment asks whether a band’s carrier frequency predicts how quickly it is learned once spectral power is controlled. Each corpus contains 32 deterministic sinusoidal tones with independent random phases, fixed total power, and no observation noise. Eight power levels span a $$128\times$$ ratio, with four bands per level. Frequencies and power levels are paired by a fresh random permutation in each run, so no carrier retains a privileged power assignment.

The learning curves separate sharply by power. At the final checkpoint, per-band $$\Delta k$$ has Spearman correlation **−0.992 with log power**, but only **−0.090 with carrier frequency**. The mean time to reach $$\Delta k\leq0.5$$ follows

$$
t\propto P^{-1.083},\qquad R^2=0.999,
$$

close to the $$P^{-1}$$ schedule predicted by a linear-readout baseline. Excluding the fastest level, whose threshold time reaches the first checkpoint, changes the exponent only slightly, to 1.078.

{% include figure.liquid path="assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/power.png" class="img-fluid" alt="Learning curves, threshold times, and final errors for the power-allocation experiment." %}

*Figure 1. Assigned band power organizes both learning curves and final error. Marker shapes denote independent runs. No band is censored at $$\Delta k\leq0.5$$; the fastest level is floor-limited at the 128-step checkpoint. These are fixed-budget measurements, not an asymptotic convergence claim.*

A separate noisy control from the SNR suite makes the frequency comparison especially direct. We hold the power multiset and noise variance fixed, but reverse their assignment: the **red** corpus places loud tones at low frequencies, while the **anti-red** corpus places them at high frequencies. Matched power levels differ by at most **0.018 in final clean $$\Delta k$$**; across levels 1–7 the maximum is 0.011, while level 0 is censored at the $$\Delta k\leq0.5$$ threshold. After removing power-level means, the residual correlation between frequency and error is only **0.036**.

{% include figure.liquid path="assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/assignment.png" class="img-fluid" alt="Red and anti-red frequency-power assignments produce closely matching final-error curves." %}

*Figure 2. Reversing which frequencies are loud leaves final-error curves nearly unchanged. Threshold times differ by about 10% at some levels, with no consistent sign; level 0 is censored. Power and in-band SNR are collinear within this control, so it isolates carrier position, not power from SNR.*

Within this controlled family, the deficit follows the allocation rather than the carrier. This does not show that architecture is irrelevant in general. It does show that a low-frequency preference is not identified without decoupling frequency from power.

## When slow learning looks like a fixed preference

If power primarily changes a learning schedule, why do some reported Transformer forecasts still appear to suppress high-frequency signal?<d-cite key="piao2024fredformer,sun2025freie"></d-cite> One plausible answer is that real spectra combine declining band power with broadband noise. High-frequency bands then have especially low SNR and may remain on the slow part of their learning schedule at any practical budget.

This is a **hypothesis about practice**, not something our synthetic study directly proves. We test one ingredient: whether adding noise at fixed signal power slows clean-signal learning. The study uses 32 equal-power tones and independent Gaussian noise added to both context and target. Carrier frequencies and signal power stay fixed; only the global noise level changes across four arms. Because every band has the same nominal SNR, this does not test the joint red-spectrum-plus-white-noise pattern or the frequency dependence of SNR in real data.

Increasing noise substantially slows clean-signal learning. Mean time to clean-probe $$\Delta k\leq0.5$$ rises from **723 steps without noise to 3,119 steps** when noise and signal have equal RMS—a $$4.3\times$$ slowdown. At 32,768 steps, mean clean retention is 0.995 without noise and 0.844 at the highest noise level.

{% include figure.liquid path="assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/snr.png" class="img-fluid" alt="Clean and noisy probes under four noise doses." %}

*Figure 3. Noise slows recovery of clean deterministic tones and increases attenuation at the fixed budget. Missing threshold points were never reached; in particular, the highest-noise arm does not reach $$\Delta k\leq0.1$$.*

The clean probe matters. For a model that knows its input is clean, the input determines the tone and optimal retention is one. Our trained model never sees clean inputs, however, so attenuation can reflect a learned prior, finite training, or train/probe mismatch; it is not an information-theoretic requirement. The highest-noise curves are also still improving. We therefore do not interpret their final errors as converged floors.

More generally, convergence alone does not imply recovering every component of a noisy target. If a component is conditionally unpredictable, suppressing it can be optimal. The practical question is whether an observed deficit is an appropriate fixed point or a transient that has been mistaken for one. Our experiment establishes that noise can strongly affect the schedule at fixed signal power; it does not establish that real spectra place high-frequency bands at low SNR, nor how much of any production TSFM’s profile this mechanism explains.

## Latent map complexity is a distinct axis

The final study returns to the other meaning of frequency. Rather than comparing carrier positions, we change the latent rule generating a fixed set of oscillators:

$$
x_i(t)=A_i\cos\theta_i(t),\qquad
\theta_i(t+1)=\theta_i(t)+\omega_i+
\sum_{j=1}^{m}\gamma_j\sin(j\theta_i(t)+\phi_j).
$$

Here $$m=0$$ is a pure rotation, while $$m=1,2,3$$ add nonlinear harmonics to the phase transition. Carriers, total power, modulation RMS, and a single scalar phase state per carrier are controlled; initialization and batch order are paired within each seed. The carriers remain in a no-phase-locking regime. Because modulation creates sidebands, we evaluate the full predicted waveform rather than a single carrier coefficient.

{% include figure.liquid path="assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/map-instrument.png" class="img-fluid" alt="Waveforms, phase-map derivatives, and realized spectra for four latent map orders." %}

*Figure 4. An illustrative one-carrier pilot at $$b=1$$ shows what the map-order knob changes. Total RMS is fixed, but the realized spectra acquire sidebands. Training uses carriers 2.5, 4.5, and 6.5 cycles per patch, where sidebands above Nyquist can alias.*

At target-variance-normalized MSE $$\leq0.02$$, $$m=0$$ reaches threshold in **384–512 steps across seeds**, compared with **768 steps for $$m=3$$**. Final variance explained decreases monotonically from 0.9997 at $$m=0$$ to 0.9964 at $$m=3$$, although threshold times for $$m=1$$ and $$m=2$$ are close. At looser thresholds the ordering is not strict, and all arms continue improving through the final checkpoint.

{% include figure.liquid path="assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/map.png" class="img-fluid" alt="Learning curves, threshold times, and final errors as latent phase-map order increases." %}

*Figure 5. At the strict 0.02 threshold and final checkpoint, higher latent map order slows learning relative to a pure rotation, with the clearest separation between $$m=0$$ and $$m=3$$. Thin curves and small marks are individual seeds.*

This is evidence that latent map complexity can slow learning, not evidence that higher realized signal frequency is learned more slowly. In the pilot, neither occupied bandwidth nor spectral entropy increases monotonically with $$m$$. The latent phase transition has exactly $$m$$ nonlinear harmonics, but the observable map composes phase inference, that transition, and a cosine observation function, and generally has a dense spectrum. Phase modulation changes the input spectrum along with $$m$$, so the experiment does not isolate map order from signal spectrum. It is at most a loose analogy to classical spectral bias—not a validation of the F-Principle, the observation that networks fit low-frequency target components first.

## Which frequency, which bias?

The controlled results support a narrower but more useful account than “Transformers prefer low frequencies”:

- **Signal-domain schedules depend strongly on power.** Randomized and reversed assignments show that deficits follow allocated power rather than carrier position.
- **SNR can substantially extend those schedules.** This makes a slow transient a plausible contributor to persistent-looking high-frequency deficits, but the production-model account remains a hypothesis.
- **Latent map complexity is a distinct axis, but not fully isolated.** The highest-order phase map is slower at the strict threshold and final checkpoint, while observable spectra also change; this does not license a direct inference from map frequency to signal frequency.
- **None of these studies establishes a converged floor.** A fixed training budget cannot distinguish a permanent preference from an unfinished schedule.

A frequency-bias claim should therefore answer four questions: Which object’s frequency is being measured? Is bias about learning speed or converged behavior? Are band power and SNR reported separately from band position? And does the metric mix amplitude with phase?

Spectral structure matters—but “frequency bias” alone is not an explanation.
