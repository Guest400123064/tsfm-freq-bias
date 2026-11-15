# Positioning and related work

This document holds the current storyline and the literature map for the ICLR 2027
blog post. The post itself lives in `docs/_posts/`; this file is the working bridge
between the controlled experiments and the claims we want to make.

## One-sentence position

Time-series forecasting calls several different objects "frequency"; the field's
observed low-frequency preference is better described as a data-driven learning
schedule set by power and SNR, with map complexity as a separate axis.

## Three objects behind "frequency bias"

| object | what it means | what changes in our experiments | primary evidence |
| --- | --- | --- | --- |
| signal frequency | where the Fourier energy of the observed series sits | carrier frequency and spectral placement | dense power ladder |
| effective power / SNR | how much coherent signal a band carries relative to noise or to other bands | power allocation, white-noise level | power and SNR controls |
| map complexity | how nonlinear the history-to-future transformation is | harmonic order of a phase map | map-complexity study |

A second distinction cuts across all three: **schedule** versus **floor**. A band can
be learned more slowly without having a worse asymptotic error. The blog reports
learning curves and time-to-threshold, not only terminal error.

## Transition: signal frequency is not map complexity

Take one pure tone:

\[
x_t=\cos(\omega t+\phi).
\]

The waveform is nonlinear in time and phase, but its one-step forecast is a fixed
**linear functional** of the last two samples:

\[
x_{t+1}=2\cos\omega\,x_t-x_{t-1}.
\]

The `cos` belongs to the observation/generative constraint that places the data on a
two-dimensional linear state space. It is not a nonlinear operation on the forecasting
input. `2cosω` is a constant coefficient for a fixed tone.

For `K` known tones, the same argument gives an order-`2K` linear recurrence:

\[
x_{t+1}=\sum_{j=1}^{2K}a_j x_{t-j+1}.
\]

Different carrier frequencies change the coefficients and rotation angle, not the
function class: each fixed frequency support admits a linear forecast map. If the
frequency set varies across windows, there is no single global linear functional; the
target is then a conditional family indexed by the latent frequency state. That
qualification matters, but it does not turn signal frequency into map frequency.

A nonlinear circle map changes the recurrence itself. Then the next value is no longer
a fixed linear functional of the observed history, and the target-map function class
becomes a separate object from the carrier spectrum. That is the object manipulated in
the map-complexity experiment.

## What the literature measures

### Classical spectral bias

- Rahaman et al. (2019) and Xu et al. (2019, F-Principle) study the frequency content
  of the learned **function**, not the frequency content of the input signal.
- Basri et al. (2019) gives a convergence-rate result; Basri et al. (2020) makes input
  density a genuine variable but remains a deterministic, fully observed fitting
  problem.
- Bietti & Mairal (2019) and Yang & Salman (2019) describe architectural or
  initialization-induced null spaces / kernel spectra.

These are mostly **rate or floor results for fitting a known target**, where the
target is in the training set. They do not model the forecasting regime, in which the
target is held out and the conditional mean is a meaningful reference.

### Frequency and energy in TSFMs

- Fredformer (Piao et al., 2024) reports "frequency bias" but measures a component's
  relative error against its amplitude share. Its manipulation is band energy, and its
  remedy presumes an architectural floor.
- SEMPO (He et al., 2025) is explicit that its axis is **energy**, not band position:
  low-energy components are suppressed across low, mid, and high frequency.
- FreIE (Sun et al., 2025) imports a spectral-bias narrative but reports per-band RMSE,
  another signal-domain measurement.
- FilterFormer (Dayag et al., 2025) frames self-attention as low-pass while its learned
  filters can be high-pass.
- Yu et al. (2024, 2025a, 2025b) study SSM transfer functions and embedding rank /
  bandwidth. Their native variables are architecture, patch size, and effective
  dimension, not a measured signal spectrum.

The recurring pattern is:

> signal-frequency language, energy measurement, function-frequency citation.

"Frequency" is often a proxy for low energy, low SNR, or a particular tokenizer scale.
Those are different physical variables and should be measured separately.

## What prior work does not separate

1. **Power / SNR from carrier position.** A red spectrum makes low frequency and high
   power collinear. Without a controlled assignment, a frequency effect and an energy
   effect predict the same data.
2. **Learning schedule from fixed point.** A deficit at a fixed training budget need
   not survive convergence. Most evaluations report one budget and one terminal score.
3. **Clean-probe behaviour from in-distribution behaviour.** A model that correctly
   shrinks an unpredictable band is doing its job; a learned prior is visible only on a
   clean input where shrinking is wrong.
4. **Carrier frequency from the actual spectrum after modulation.** Once a signal is
   phase- or amplitude-modulated, its power is not localized at a single carrier. A
   "carrier" parameter is not the same object as the dominant signal frequency.

## What this project contributes

1. A dense-spectrum power ladder with fixed total power and randomized
   frequency-power assignment. Final error follows assigned power almost perfectly and
   has no meaningful relationship with carrier frequency.
2. A white-noise dose response plus a red/anti-red assignment control. Noise slows the
   schedule; swapping which frequencies are loud leaves matched power levels almost
   unchanged.
3. A separate map-complexity experiment with fixed carriers, fixed total power, and a
   single scalar phase state per carrier. Nonlinear maps are slower than a pure
   rotation, but harmonic count is not a monotone difficulty scale: map realization
   matters, and `m=2` is on average slower than `m=3`.
4. A vocabulary and evaluation protocol: report signal spectrum, effective power/SNR,
   map complexity, schedule, and floor as different quantities.

## Scope and open questions

- The controlled evidence is from a small `SimTFM`; the blog should not silently
  generalize every number to production TSFMs.
- The map-complexity experiment is fixed-carrier. Its treatment is the Fourier order of
  the latent phase map `F_m`, not the Fourier rank of the ambient history-to-next
  function. The observable map adds phase inference and a nonlinear observation
  function. It does not claim cross-carrier invariance; that is handled by the power
  and SNR experiments.
- The SNR experiment still needs a scale-matched noiseless control.
- An illustrative real-data projection can combine the measured corpus spectrum with a
  noise floor and the synthetic power-scaling exponent. It is an extrapolation, not a
  causal experiment.
- How production checkpoints behave under the same clean-probe protocol remains future
  work.

## Key references

- Rahaman et al., *On the Spectral Bias of Neural Networks*, ICML 2019.
- Xu et al., *Frequency Principle: Fourier Analysis Sheds Light on Deep Neural
  Networks*, 2019.
- Basri et al., *The Convergence Rate of Neural Networks for Learned Functions of
  Different Frequencies*, NeurIPS 2019.
- Basri et al., *Frequency Bias in Neural Networks for Input of Non-Uniform Density*,
  ICML 2020.
- Bietti & Mairal, *On the Inductive Bias of Neural Tangent Kernels*, NeurIPS 2019.
- Piao et al., *Fredformer: Frequency Debiased Transformer for Time Series
  Forecasting*, KDD 2024.
- He et al., *SEMPO: Energy Bias in Time-Series Forecasting*, 2025.
- Sun et al., *FreIE*, 2025.
- Dayag et al., *Filter then Attend / FilterFormer*, 2025.
- Yu et al., *Tuning Frequency Bias of State Space Models*, ICLR 2025; and the
  rank/bandwidth papers (arXiv:2510.03358, arXiv:2510.19236).

## Submission layout

The ICLR 2027 blog track accepts only post-specific files. Draft paths:

- `docs/_posts/2027-04-30-disambiguating-freq-bias-in-tsfm.md`
- `docs/assets/bibliography/2027-04-30-disambiguating-freq-bias-in-tsfm.bib`
- `docs/assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/`
- `docs/assets/html/2027-04-30-disambiguating-freq-bias-in-tsfm/`
