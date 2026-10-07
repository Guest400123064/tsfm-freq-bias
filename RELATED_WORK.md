# Related work, re-read against our taxonomy

Every paper below that uses the words "frequency bias", "spectral bias" or an equivalent, classified
by **which object it measures** and **how it tests it**. Supersedes `PLAN.md` §2, which was written
from abstracts and contains two errors this pass corrected (noted at the end).

**Method.** One reader per paper, working only from the paper's own text, answering a fixed protocol:
what they call it (verbatim), which object they measure, whether the claim is a *rate* or a *floor*,
whose variable they attribute it to, and — the part that matters — whether their measurement could
distinguish "the model learned the conditional mean of its training distribution" from "the model has
a hard-wired prior or defect". No inference beyond the text; `not stated` where absent.

---

## The taxonomy

**"Frequency" can mean six different objects.**

| | object |
| :-- | :-- |
| **F1** | **signal frequency** — the Fourier content of the observed series |
| **F2** | **function/map frequency** — the eigenfrequency of the learned map input→output (the classical *spectral bias* object: NTK / RKHS eigenvalues) |
| **F3** | **per-patch phase advance** Δφ = 2πb (mod 2π), b = f·k/ctx — what a *patched* model actually operates on |
| **F4** | **bandwidth / effective dimension** of a band (stable rank, degrees of freedom) |
| **F5** | **occurrence / input density** — how often a frequency appears |
| **F6** | **energy / amplitude** of a band |

**"Bias" splits again.** *Rate* (learned slower; removable with training) vs *floor* (a converged,
non-removable property). And **relative to what** — (i) a single corpus's in-distribution behaviour,
(ii) the *difference* between training and evaluation distributions, or (iii) an **idealised
reference** ("a perfect model would keep every band").

A high-frequency pure tone is F1-high but its generating map is a rotation, so **F1 ≠ F2**. Any claim
that reports F1 while citing F2 is two claims.

---

## Table A — what they are actually measuring

| paper | what they call it | object **measured** | object **claimed** | rate / floor | whose variable |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Rahaman et al. 2019 (ICML) | "spectral bias" | **F2** | F2 | **rate**, explicit | architecture + optimisation |
| Xu et al. 2019 (F-Principle) | "Frequency Principle"; "implicit bias" | **F2**, cleanly | F2 | rate explicit; floor via theorems | activation regularity, init, loss |
| Basri et al. 2020 (ICML) | "frequency bias" | **F2**, modulated by **F5** | F2+F5 | **rate**, explicit | **data distribution** (input density p(x)) |
| Bietti & Mairal 2019 (NeurIPS) | "inductive bias" | **F2** (+F4) | F2/F4 | **floor**, provable null space | architecture / init |
| Yang & Salman 2019 | "simplicity bias" | **F2** (+F4/F6 mixed) | F2 | floor (kernel spectrum) | architecture, init, input measure |
| **Fredformer** KDD 2024 | "frequency bias" | **F6**, with F1+F5 in the story | F1 → F6 | floor (inferred; fix is architectural) | **data distribution** (relative amplitude) |
| **FreIE** 2510.25800 | "spectral bias" | **F1** (per-band RMSE) | **F2**, transplanted | rate explicit, floor-flavoured table | architecture, init, activation |
| **EASD / SEMPO** 2510.19710 (NeurIPS) | "**energy** bias" | **F6** | F6 (F1 explicitly orthogonal) | floor (inferred) | architecture (self-attention weighting) |
| **Yu et al.** 2410.02035 (ICLR) | "frequency bias" = "spectral bias" | **F2** (transfer function) | F2/F1 | **floor**, explicit ("conventional training does not alter this bias") | **architecture + init**, then optimisation |
| Maddix et al. 2510.03358 | *no bias claimed* | **F4** (singular spectra) | — | — | architecture (patch size) |
| Maddix et al. 2510.19236 | "frequency bias" | **F1** (synthetic), **F4** (theorem) | F1, F4, cites F2 | floor (inferred) | **patch size** |
| FilterFormer 2508.20206 | "bias toward low-frequencies" | **F1** (synthetic), F3 operationally | F2-flavoured cause ("self-attention acts as a low-pass filter") | floor (inferred) | architecture |

**Read the "measured" column, not the "claimed" one.** The two strongest forecasting-side papers —
Fredformer and EASD — measure **energy**, and EASD's own definition says the deficit falls on
low-*energy* components "distributed across mid-, high-, or even low-frequency bands": their axis is
amplitude, band position is explicitly orthogonal. Fredformer's Definition 2 and Eq. (2) assert
`−|Δ_k| ∝ P(ã_k)` — again an amplitude share. **The forecasting literature's "high-frequency" is a
proxy for "low-energy".** FreIE is the reverse error: it imports the F2 claim and measures F1.

---

## Table B — how they test it

| paper | target held out? | eval signal | deficit measured against | metric | manipulates training dist.? | controls **power**? | controls **predictability**? | **separates prior from conditional mean?** |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Rahaman 2019 | **no** — target in training set | analytic, in-dist. | idealised (target's own amplitude) | amplitude-only | yes: noise freq/amp, tilt | shown not to matter | **no** (target deterministic) | **No** |
| Xu 2019 | **no** — training inputs only | analytic, in-dist. | 0 relative error on training target | amplitude+phase | **no** | no | no | **No** |
| Basri 2020 | **no** — fully observed domain | analytic, noiseless | analytic optimum (time-to-δ) | time-to-threshold | yes: input density, target freq | amplitudes fixed | **no** (deterministic) | **No** |
| Bietti & Mairal 2019 | **no** — theory, no target fit | MNIST + synthetic | another kernel | amplitude-only | **no** | no | no | **No** |
| Yang & Salman 2019 | **no** — target in training set | analytic / none | raw loss | amplitude-only | target degree, input measure | no | no | **No** |
| **Fredformer** | **yes** (forecasting) | **in-distribution**, A ≈ Â | idealised: Δ_k *independent* of P | **amplitude+phase** | **yes: band-power ordering** | **yes** — flips the order, error follows amplitude | no | **No** (but see below) |
| **FreIE** | mixed | ETTH2 in-dist. + synthetic | nothing but per-band RMSE | not stated | amplitude on synthetic only | no | no | **No** |
| **EASD / SEMPO** | **yes** (zero/few-shot) | real held-out | "ground-truth spectral peaks" (visual) | amplitude-only | no (masks the *input*) | no | no | **No** |
| **Yu et al.** 2410.02035 | **no** — analytic target fit | synthetic + OOD noise at inference | loss numbers | amplitude-only | **no** — varies **init** and loss | no | no | **No** |
| Maddix 2510.03358 | n/a — embedding geometry | 6 pretrained Chronos-Bolt | nothing band-wise | amplitude-only | no | no | no | **No** |
| Maddix 2510.19236 | **yes** (synthetic forecast) | **synthetic**, not the pretraining dist. | another model | "frequency loss", formula absent | **evaluation signal only** | no | no | **No** |
| FilterFormer | **yes** | in-dist. real + one synthetic | another model | amplitude-only | no | no | no | **No** |

---

## Four findings

**1. Nobody controls per-band conditional predictability. Not one paper, in either set.** Twelve of
twelve. Power/energy is varied by four of them; predictability never. So every attribution to
"frequency" or to "energy" is made in a regime where the *other* axis of the training distribution is
free to move with it — and in every forecasting corpus the two are collinear with band index.

**2. The classical set cannot ask our question, structurally.** All five fit a **known target on a
fully observed domain**, where the target is *in* the training set: the conditional distribution has
no spread, so "learned the conditional mean" and "fit the truth" coincide and no test can separate
them. Basri 2020 is the nearest miss — the data distribution is a genuine variable — but it is a
**rate** result in a noiseless deterministic setting. Bietti & Mairal is the only paper in the set
with a real **floor** (a provable null space, μ_k = 0), and it is a statement about a kernel's RKHS
that never touches a corpus.

**3. The forecasting set has held-out targets but the wrong reference.** Fredformer, EASD, FreIE,
Maddix-19236 and FilterFormer all measure a deficit against **1.0, an idealised reference, or another
model**. None measures against what the conditional mean of *that corpus* would predict, and none
against a trivial baseline. A model that is correctly shrinking a band whose future is not determined
by its past reads as "biased" under every one of these metrics.

**4. Where Fredformer is right, and where it stops.** Its synthetic Case 1 holds amplitudes fixed
between input and ground truth (`A ≈ Â`), so the target is effectively **deterministic**: the Bayes
optimum is `r = 1` at every band, and any residual `Δ_k` is a genuine **fitting gap** — not correct
hedging. That is a real result, and it is the same logic as our own P7/P8 design. And their
amplitude-order flip genuinely separates *band index* from *energy*: the error follows energy. **So
Fredformer's mechanism survives; what does not survive is the framing.** A fitting gap on a
deterministic corpus is a statement about the *optimisation schedule*, and our work locates it there
("energy sets how fast a band is learned"). Their remedy is architectural, which presumes a floor;
their own evidence is consistent with a rate.

**One more, for the record:** FilterFormer's own learned filters come out **high-pass** (§5.1.4), on
top of the same "self-attention is a low-pass filter" framing. The narrative and the measurement point
in opposite directions, which is the bookkeeping problem in miniature.

---

## What this leaves us

Our position is not "the previous work is wrong". It is narrower and, we think, harder to argue with:

- **The field's "frequency" is a proxy.** F1-language, F6-measurement, F2-citation — in the same
  paragraph, in several of the papers above.
- **The measurement has no baseline.** A per-band deficit against 1.0 is not evidence of a defect when
  the band's future is partly undetermined. The reference has to be the conditional mean of the corpus
  under test — which is computable before training a model.
- **The two axes that do move the per-band profile — marginal power and conditional predictability —
  have never been separated by anyone.** We did it, and they play different roles: **share sets the
  schedule, predictability sets the fixed point.**
- **No paper in this set tests whether the reported phenomenon survives a clean probe**, i.e. an input
  on which shrinking is *wrong*. Our P0 is the demonstration that an in-distribution probe cannot tell
  hedging from a prior, and it is the test the field has not run.

---

## Corrections to `PLAN.md` §2

1. **Title/ID mismatch.** `arXiv:2003.04560` is Basri, Galun, Geifman, Jacobs, Kasten & Kritchman,
   *"Frequency Bias in Neural Networks for Input of Non-Uniform Density"* (ICML 2020). The title
   `PLAN.md` attaches to it, *"The Convergence Rate of Neural Networks for Learned Functions of
   Different Frequencies"*, is a different paper: Basri, Jacobs, Kasten & Kritchman, **NeurIPS 2019,
   arXiv:1906.00425**. Both should be listed and distinguished — the NeurIPS one is the rate result,
   the ICML one adds input density.
2. **Maddix et al. `2510.03358` claims no frequency bias at all.** It is a rank-structure paper
   (singular spectra, flow-of-ranks, compressibility) on *embeddings*, and it never measures a target.
   It belongs in a list of F4/bandwidth work, not in a list of frequency-bias claims. `PLAN.md`'s entry
   "#4 stated as #1" overstates it.
3. **EASD `2510.19710` is the paper that makes the energy claim cleanest**, and it is not in
   `PLAN.md` at all. It should be, because it independently arrives at the axis we argue is the real
   one — and then attributes it to self-attention rather than to the data.
4. **`PLAN.md` §2's "empty cell" claim survives**, but for a sharper reason than it gives. The cell is
   empty not because nobody looked at the data distribution, but because **nobody measured against a
   conditional-mean reference** — without one, a data-driven *rate* and an architectural *floor* are
   indistinguishable in the numbers everyone reports.
