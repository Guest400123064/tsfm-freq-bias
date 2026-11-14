# Agent guidance

- Ask clarification questions if the intent is unclear before acting.
- Prefer planning before action; use [`.agents/PLAN.md`](./PLAN.md) as a scratch pad.
- Prefer concise code that fails fast.
- Before finishing a code change, run the `ponytail` skill as a self-review: question
  whether the code needs to exist, reuse existing helpers, and keep the diff minimal.
- Prefer minimal comments; code should be self-explanatory.
- Prefer NumpyDoc docstrings.

## Experiment layout

Every maintained experiment lives in `experiments/<NN>_<slug>/` and keeps this structure:

```text
<NN>_<slug>/
├── README.md
├── Makefile
├── scripts/
├── outputs/
├── figs/
└── notebooks/   # optional; only for interactive exploration
```

- `scripts/`: runnable Python entry points. No result should exist only in a notebook.
- `outputs/`: metrics, JSON/NPZ, logs, tables. `figs/`: figures only.
- `notebooks/`: add only when interactive inspection is genuinely needed. Any result
  shown in the README must also be reproducible from `scripts/` + `Makefile`.
- Keep raw data out of git; put it in an ignored path or outside the repo.
- Never leave an empty README or Makefile; fill it or omit the experiment.
- `experiments/_archive/` is read-only reference material from the exploratory phase.
  Do not treat it as a maintained or reproducible experiment group.

## Makefile

- `make` (default target) runs the experiment end to end.
- Every file in `outputs/` or `figs/` is produced by a Makefile target; no manual steps.
- Pass parameters as Make variables or CLI flags; document the defaults in the README.
- Fix seeds. A rerun should produce the same or quantitatively similar results.
- Keep one target per script with explicit inputs and outputs.

Minimal skeleton:

```make
.PHONY: all clean
all: outputs/metrics.json figs/result.png

outputs/metrics.json: scripts/run.py
	.venv/bin/python $< --seed 0 > $@

figs/result.png: scripts/plot.py outputs/metrics.json
	.venv/bin/python $< --input outputs/metrics.json --output $@

clean:
	rm -rf outputs/* figs/*
```

## Experiment README

Every experiment README starts with `## TLDR` and always contains `## How to run`.
Keep the prose short (target: under ~300 words excluding code, tables, and figures).

- `## TLDR`: 1–3 sentences: what the experiment tests and the main result.
- `## What it does`: brief design, controls, and metrics.
- `## Results`: after the run, a short material summary; replace placeholders.
- `## How to run`: exact `make` commands, key parameters, and where artifacts land.
- Embed figures inline with relative paths, e.g. `![retention](figs/retention.png)`.
- Prefer bullets and one small table over paragraphs.

## Visualization

- Reuse shared plotting helpers in `src/bias/`; do not create per-experiment styles or
  add plotting dependencies without a demonstrated need.
- Use consistent figure widths: 3.25 inches for compact figures and 7 inches for
  full-width figures. Keep typography readable at final size (about 8 pt), with
  sans-serif fonts and a white background.
- Use black for actual values, grey for context/auxiliary information, and shared
  model colors for forecasts. Emphasize the result; mute gridlines and weak evidence.
- Label every panel explicitly when bands or conditions differ; never use a shared row
  label that implies a shared condition.
- Plot analytic continuous ground truth when appropriate, but leave model forecasts as
  discrete outputs. Avoid straight-line rendering artifacts that resemble modulation.
- Mark phase estimates as weakly identified when energy retention `r < 0.05` using open
  grey markers; do not emphasize them as reliable results.
- Generate 300 dpi PNG only, not PDF. Remove superseded or duplicate figures and their
  Makefile/README references.
- Inspect the rendered image before finishing; code running successfully is not enough.

## ICLR 2027 blog post

The submission lives under `docs/` using the official ICLR layout:

- Post: `docs/_posts/2027-04-30-disambiguating-freq-bias-in-tsfm.md`.
- Bibliography: `docs/assets/bibliography/2027-04-30-disambiguating-freq-bias-in-tsfm.bib`.
- Static figures: `docs/assets/img/2027-04-30-disambiguating-freq-bias-in-tsfm/`.
- Interactive figures: `docs/assets/html/2027-04-30-disambiguating-freq-bias-in-tsfm/`.

Keep the review draft anonymous. When preparing the submission repository, copy only
these post-specific files into the official blog repository; do not copy this research
repository's other files into it.
