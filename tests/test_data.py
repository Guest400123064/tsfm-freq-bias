import math

import pytest
import torch

from bias.data import (
    b_of,
    circle_rule,
    make_coeff_recurrence,
    make_mixture,
    make_phase_map,
    make_pm_profile,
)
from bias.probes import matched_amp

K = 32


def test_make_mixture_spacing():
    with pytest.raises(AssertionError, match="cycle/patch"):
        make_mixture([2.0, 3.0], 4, 512, K, 0)

    x = make_mixture([2.0, 18.0], 4, 512, K, 0)
    assert x.shape == (4, 512 + K)
    assert b_of(18.0, K, 512) - b_of(2.0, K, 512) == 1.0


def test_pm_profile_preserves_power():
    freqs = torch.tensor([16.0, 32.0])
    powers = torch.tensor([0.4, 0.1])
    x, unpredictable = make_pm_profile(freqs, powers, [0.0, 0.0], 8, 512, K, 0)
    assert x.square().mean().item() == pytest.approx(0.5, rel=1e-5)
    assert torch.equal(unpredictable, torch.zeros_like(unpredictable))

    for j, b in enumerate((1.0, 2.0)):
        z = matched_amp(x.unfold(1, K, K), b, K)
        assert (z.abs() ** 2 / 2).mean().item() == pytest.approx(
            powers[j].item(), rel=1e-4
        )


def test_coeff_recurrence_preserves_power_and_rule():
    bands = [2.0, 4.0]
    powers = [0.3, 0.2]
    rule = circle_rule(math.pi / 3, 0.5)
    x = make_coeff_recurrence(bands, rule, powers, 8, 512, K, 0)
    patches = x.unfold(1, K, K)

    for j, b in enumerate(bands):
        z = matched_amp(patches, b, K)
        assert (z.abs() ** 2 / 2).mean().item() == pytest.approx(powers[j], rel=1e-4)
        assert torch.allclose(z[:, 1:], rule(z[:, :-1]), atol=2e-6, rtol=2e-6)


def test_phase_map_power_and_zero_beta():
    bands = [0.5, 1.5, 2.5, 3.5]
    x = make_phase_map(bands, [0, 1, 2, 3], 0.1, 16, 512, 16, 0)
    assert x.shape == (16, 528)
    assert x.square().mean().item() == pytest.approx(0.5, rel=0.02)

    pure = make_phase_map(bands, [0, 0, 0, 0], 0.1, 4, 512, 16, 3)
    zero = make_phase_map(bands, [1, 2, 3, 0], 0.0, 4, 512, 16, 3)
    assert torch.equal(pure, zero)
