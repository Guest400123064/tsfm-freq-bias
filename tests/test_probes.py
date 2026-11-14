import math

import pytest
import torch

from bias.probes import (
    band_coefficients,
    delta_k,
    matched_amp,
    phase_error,
    retention,
)

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


def test_band_coefficients_recovers_dense_tones():
    k = 64
    bands = torch.linspace(0.25, 31.75, 32, dtype=torch.float64)
    torch.manual_seed(0)
    z = torch.randn(5, len(bands), dtype=torch.float64) + 1j * torch.randn(
        5, len(bands), dtype=torch.float64
    )
    angles = 2 * math.pi * bands[:, None] * torch.arange(k, dtype=torch.float64) / k
    x = torch.real(z[:, :, None] * torch.exp(1j * angles)).sum(dim=1)
    assert torch.allclose(band_coefficients(x, bands, k), z, atol=1e-12)
