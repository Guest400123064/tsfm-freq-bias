[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/) [![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

# disambiguating frequency bias in time-series foundation models

Controlled experiments behind an ICLR 2027 blog post on the overloaded meaning of
"frequency" in time-series forecasting. Signal frequency, band power/SNR, and
history-to-future map complexity are different objects; this project separates their
effects on learning schedules and finite-budget forecasting error.

## Abstract

A high-frequency signal can have a trivial linear forecast map, while a low-frequency
signal can require a complex one. We use a small next-patch Transformer and controlled
synthetic corpora to disentangle carrier frequency, power allocation, observation
noise, and latent map complexity. The current evidence identifies strong learning-speed
differences without supporting an intrinsic preference for low carrier frequencies.

## How to start

```bash
uv sync
uv run pytest
make -C experiments/2_map_complexity smoke
```

Full experiments use CUDA. Each maintained experiment has its own README and Makefile
with reproduction commands and result summaries.

## Navigate

- [Current narrative, related work, and evaluation protocol](docs/README.md)
- [ICLR blog draft](docs/_posts/2027-04-30-disambiguating-freq-bias-in-tsfm.md)
- [Experiment index](experiments/README.md)
- [Model, data generators, and probes](src/bias/)

## Notes

The evidence comes from controlled `SimTFM` experiments, not pretrained production
TSFMs. Historical exploratory runs live in `experiments/_archive/` and are not
maintained reproduction targets.

Licensed under [Apache 2.0](LICENSE).
