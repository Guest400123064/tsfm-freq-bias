# P17 — rank: is the manifold's dimension a cost, and is it proportionate?

**Question.** P16 settled **position**: two carriers with identical share profiles read identically
to |Δr| = 0.008. The one axis nothing has touched is the **dimension of the state the context must
resolve** — the rank of the map `∂y_{T+1}/∂y_{t<T}`. PM offers one operation that changes rank and
provably nothing else.

```
y(t) = cos(2π b_h t / k + β · cos(2π b_m t / k + φ))
```

The n-th sideband carries power `J_n(β)²/2` — **independent of φ** — while its phase carries `nφ`.

| | state | rank | sideband powers | relative phases |
| :-- | :-- | :-- | :-- | :-- |
| φ fixed | (τ) | **1** | `J_n(β)²` | identical in every window |
| φ random per window | (τ, φ) | **2** | **bit-identical** | vary per window |

Powers, positions, carrier, bandwidth, total power: all unchanged. Only the sidebands' relative-phase
structure varies, and the model must resolve it from the context.

**Why φ and not the carrier's phase**: the carrier's phase enters linearly and is absorbed by a time
shift (`ψ → ψ + 2πb_h·d` is `τ → τ + d`), so it is not a degree of freedom at all.

**This is a dynamics experiment.** The target is a deterministic function of the context, the oracle
already reads 0.98–1.00 at hidden 32 (P16), and the corpus is fit to 97–98 % of variance by 2 000
steps. Any rank effect is therefore a **time** effect, not a representational one. The readout is the
transient, sampled every 100 steps to 2 000, and the 10 000-step state serves only as the convergence
claim.

## Arms

| phase ladder | φ ∈ | state |
| :-- | :-- | :-- |
| `phiK1` | {0} | rank 1 (the P16 configuration) |
| `phiK2` | {0, π} | 1 + 1 bit |
| `phiK4` | {0, π/2, π, 3π/2} | 1 + 2 bits |
| `phiInf` | Uniform[0, 2π) | rank 2 |

× carriers `b_h ∈ {6, 10}` = **8 arms**, `β = 1.5` (shares .262 / .311 / .054 — every significant
line well above the noise floor), one common modulator `b_m = 1`, constant envelope, no noise.
`hidden 32 / 2L / 4 heads`, 3 seeds, checkpoints every 100 steps to 2 000 then every 500 to 10 000.

**One fairness point, recorded because it is what makes the comparison legitimate.** With φ fixed the
corpus **already sweeps its whole one-dimensional state space**: window offsets are uniform over
1 024 samples and the modulator runs 32 cycles in that span, so the window-start modulator phase
covers [0, 2π) uniformly. In `(θ_c, θ_m)` coordinates the fixed-φ windows lie on the curve
`θ_c = 8(θ_m − φ)` and the random-φ windows fill the torus. Both arms span a full family; the random
arm adds one orthogonal dimension rather than "more samples of the same thing".

## Pre-registered decision (written before the run)

| | check |
| :-- | :-- |
| **A** | **design holds** — measured per-band shares match `J_n(β)²` in every arm and are equal between arms of the same carrier (the rank operation did not move the spectrum) |
| **B** | **capacity** — the frozen-latent oracle ≥ 0.9 at every significant line of every arm at the final checkpoint |
| **C** | **convergence** — at 10 000 steps every significant line of every arm reads `r ≥ 0.95`, i.e. the endpoint is rank-independent |
| **D** | **transient (discriminator)** — `t80(K)/t80(K=1)` at matched lines; the rank effect is present if any line exceeds **1.25×**, reported as a dose-response over K |
| **E** | **interaction** — `|log₂ effect(b6) − log₂ effect(b10)|` against the same 1.25× threshold; exceeded means the rank cost is **disproportionate** across position |
| **F** | **residual** — if D is present but C holds, the cost is purely dynamical (a time constant); if C fails, a gap survives convergence and is reported as such |

Verdicts: B failing → capacity; C failing → a gap survives convergence; D absent → no rank effect;
D present with E exceeded → disproportionate; D present with E within tolerance → a uniform time cost.

## Status — DONE

```
A. design       share error vs J_n^2 = 0.000000; cross-arm spread = 0.000000   PASS
B. capacity     oracle 0.9932 ... 0.9993                                       PASS
C. convergence  final min r 0.9881 ... 0.9949 across all 8 arms                PASS
D. rank effect  worst t80(K)/t80(K=1) = 1.143 (threshold 1.25)                 ABSENT
E. interaction  worst |Δlog2 effect| = 0.193 (threshold 0.322)                 within tolerance
-> no rank effect: identical dynamics at rank 1 and rank 2
```

The carrier line of the four b6 arms, step by step — four state-space dimensions, one curve:

```
step      phiK1    phiK2    phiK4    phiInf
   100    0.597    0.601    0.597    0.604
   500    0.890    0.887    0.892    0.896
  1000    0.968    0.964    0.971    0.970
  2000    0.992    0.989    0.996    0.992
 10000    0.999    1.000    1.002    1.001
```

The largest deviation is on the weakest line (share .054): ~0.04 at 1 000–2 000 steps and ~0.008 at
convergence, and it does **not** order monotonically in K (`K1 ≈ K2 > K4 ≈ Inf` — a two-way split, not
a ladder). Check F therefore resolves as the *pure dynamics* branch, with no time constant to report.

**Limitation recorded**: the runner stores only the seed-mean trajectory, so the seed sd of the t80
ratios cannot be recovered afterwards. The pre-registered threshold was applied as written; the
stronger evidence is the convergence column, which spans 0.988–0.995 with no ordering in K.

**Branch summary.** Of {share, position, dimension, predictability}, only **share** (which sets the
schedule — P8/P14/P15, reproduced here on a third constructor) and **predictability** (which sets the
fixed point — P2b) enter the model's behaviour. Position (P16) and dimension (P17) do not.
