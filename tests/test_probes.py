import math

import pytest
import torch

from fbias.data import b_of, make_mixture
from fbias.probes import matched_amp, phase_error, retention

K = 32


def test_matched_amp_pure_tone():
    # b = 4.5 has 2b integer, so the negative-frequency image cancels exactly.
    b, k = 4.5, K
    t = torch.arange(k, dtype=torch.float64)
    x = 2.5 * torch.cos(2 * math.pi * b * t / k + 0.7)

    z = matched_amp(x, b, k)
    assert z.abs().item() == pytest.approx(2.5, rel=1e-12)
    assert torch.angle(z).item() == pytest.approx(0.7, rel=1e-12)


def test_retention_and_phase_error():
    b, k = 4.5, K
    t = torch.arange(k, dtype=torch.float64)
    true = torch.stack(
        [torch.cos(2 * math.pi * b * t / k + phi) for phi in (0.0, 0.4, 2.9)]
    )

    shrunk = 0.25 * true
    assert torch.allclose(retention(shrunk, true, b, k), 0.25 * torch.ones(3).double())
    assert torch.allclose(
        phase_error(shrunk, true, b, k), torch.zeros(3, dtype=torch.float64)
    )

    shifted = torch.cos(2 * math.pi * b * t / k + 0.3)
    assert retention(shifted, true[0], b, k).item() == pytest.approx(1.0, rel=1e-12)
    assert phase_error(shifted, true[0], b, k).item() == pytest.approx(0.3, rel=1e-12)


def test_make_mixture_spacing():
    # b = 0.125 and 0.1875, i.e. 0.0625 cycles/patch apart.
    with pytest.raises(AssertionError, match="cycles/patch"):
        make_mixture([2.0, 3.0], 4, 512, K, 0)

    # Exactly 1 cycle/patch apart is allowed.
    x = make_mixture([2.0, 18.0], 4, 512, K, 0)
    assert x.shape == (4, 512 + K)
    assert b_of(18.0, K, 512) - b_of(2.0, K, 512) == 1.0
