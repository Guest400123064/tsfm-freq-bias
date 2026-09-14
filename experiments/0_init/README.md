# `0_init` — playground

**This is a sandbox, not a tracked experiment.** Code debugging, reproduction attempts, and setup
for later experiments all happen here, without strict experiment tracking. Promoted experiments get
their own `experiments/<id>/` folder with a strict README; this one does not.

Design contract: [`PLAN.md`](../../PLAN.md). Run log: [`experiments/README.md`](../README.md).

## Status

| | |
| :-- | :-- |
| harness | working — `data.py`, `probes.py`, `cli/train.py`, `tests/test_probes.py` green |
| smoke | passes — loss `1.3006 → 0.0036`, 47 s, checkpoint + sidecar written |
| C1 reproduction | **not achieved** — the corpus that was specified cannot exhibit the effect |

## What the ladder established

Full numbers and the reasoning trail: [`experiments/README.md`](../README.md). Summary:

| arm | training marginal | variance explained @20k | `r@0.125` | `r@8` | `r8/r1` |
| :-- | :-- | --: | --: | --: | --: |
| N | 2 tones, train == probe | +0.999 | 1.002 | 1.000 | 0.998 |
| BR | 32 tones, frequencies redrawn every window | **−0.005** | 0.021 | 0.044 | 2.073 |
| BF | 32 tones, fixed set + per-window phases | +0.691 (rising) | 1.138 | 0.802 | **0.705** |

Three conclusions:

1. **BR is unlearnable at this scale**, not under-trained — it plateaus at the mean predictor within 2k steps and makes no further progress over the next 18k. Any `r_f` from it is meaningless. This is why the first marginal ladder looked like it showed something.
2. **BF fits and does show `r8/r1 < 1`** (0.705 at 20k, ~40× the seed spread), but it had not converged.
3. The lab's It.27 damping was measured on models trained on **Monash real data** or the **rich 12-cluster corpus**, neither of which was the 2-tone mixture it was probed on. A corpus that is deterministic, simple, and *is* the probe sits in It.29's regime instead, where the lab also measures no damping (0.988–0.998).

## Re-specified reproduction target

Train `make_broad(n_tones=32, resample_freqs=False, normalize=True)` — a fixed 32-tone set with
per-window phases — long enough to fit (≥50% variance explained; 20k steps reached 0.691 and was
still climbing). Then probe a normalized 2-tone mixture `[2.0, 128.0]` and read `r@0.125`, `r@8`
and `r8/r1`.

**Unresolved caveat.** At 20k the low band is *over*-predicted (`r@0.125 = 1.14`, `band r2 =
−1.11`) while `r@8 = 0.80`, so `r8/r1 = 0.705` is partly low-band overshoot rather than clean
high-band suppression. A defensible "the high band is damped" claim needs BF trained to
convergence first — and a check that the low-band overshoot is not itself the fit deficit.

## Scripts

| script | what it does |
| :-- | :-- |
| `scripts/smoke.py` | trains the two-tone smoke model, writes checkpoint + sidecar, prints the metric table |
| `scripts/diag_marginal.py` | 4 training marginals × 3 seeds, fixed 2-tone probe (`runs/diag_marginal.json`) |
| `scripts/diag_fit.py` | 3 marginals × 20k steps with fit/retention checkpoints (`runs/diag_fit.json`) |

## Contracts this produced

Now in `PLAN.md` §3: **8** gate `r_f` on variance explained, **9** RMS-match the probe to the
training corpus, **10** "broad" must be a structured frequency family. Items 8 and 9 each cost a
corrupted reading before being caught.
