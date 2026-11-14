import math

import pytest
import torch

from bias.probes import delta_k, fit_oracle, matched_amp, phase_error, retention

K = 32


def test_matched_amp_pure_tone():
    b = 4.5
    t = torch.arange(K, dtype=torch.float64)
    x = 2.5 * torch.cos(2 * math.pi * b * t / K + 0.7)
    z = matched_amp(x, b, K)
    assert z.abs().item() == pytest.approx(2.5, rel=1e-12)
    assert torch.angle(z).item() == pytest.approx(0.7, rel=1e-12)


def test_retention_and_phase_error():
    b = 4.5
    t = torch.arange(K, dtype=torch.float64)
    true = torch.stack(
        [torch.cos(2 * math.pi * b * t / K + phi) for phi in (0.0, 0.4, 2.9)]
    )
    shrunk = 0.25 * true
    assert torch.allclose(retention(shrunk, true, b, K), 0.25 * torch.ones(3).double())
    assert torch.allclose(phase_error(shrunk, true, b, K), torch.zeros(3).double())

    shifted = torch.cos(2 * math.pi * b * t / K + 0.3)
    assert retention(shifted, true[0], b, K).item() == pytest.approx(1.0, rel=1e-12)
    assert phase_error(shifted, true[0], b, K).item() == pytest.approx(0.3, rel=1e-12)


def test_delta_k_is_amplitude_and_phase():
    b = 4.5
    t = torch.arange(K, dtype=torch.float64)
    true = torch.cos(2 * math.pi * b * t / K)
    assert delta_k(0.4 * true, true, b, K).item() == pytest.approx(0.6, rel=1e-12)

    quadrature = torch.cos(2 * math.pi * b * t / K + math.pi / 2)
    assert delta_k(quadrature, true, b, K).item() == pytest.approx(
        math.sqrt(2), rel=1e-12
    )

    pred = 0.7 * torch.cos(2 * math.pi * b * t / K + 0.9)
    r = retention(pred, true, b, K).item()
    dphi = phase_error(pred, true, b, K).item()
    assert delta_k(pred, true, b, K).item() == pytest.approx(
        math.sqrt(1 + r**2 - 2 * r * math.cos(dphi)), rel=1e-12
    )


def test_fit_oracle_recovers_linear_readout():
    gen = torch.Generator().manual_seed(0)
    z_fit = torch.randn(16, 5, 4, generator=gen)
    z_eval = torch.randn(8, 5, 4, generator=gen)
    weights = torch.randn(4, 3, generator=gen)
    bias = torch.randn(3, generator=gen)
    target_fit = torch.zeros(16, 5, 3)
    target_fit[:, 1:] = z_fit[:, :-1] @ weights + bias

    pred = fit_oracle(z_fit, target_fit, z_eval, ridge=1e-8)
    expected = z_eval[:, :-1] @ weights + bias
    assert torch.allclose(pred, expected, atol=1e-4, rtol=1e-4)
