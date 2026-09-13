import pytest
import torch

from fbias.model import SimTFM

CTX, PATCH = 128, 32


def tiny_model(**kwargs):
    torch.manual_seed(0)
    return SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=16,
        num_layers=2,
        num_attn_heads=4,
        **kwargs,
    )


@pytest.mark.parametrize("use_rope", [True, False])
def test_pat(use_rope):
    m = tiny_model(use_rope=use_rope)
    x = torch.randn(2, CTX)
    p = m.pat(x)
    assert p.shape == (2, CTX // PATCH, PATCH)
    assert torch.equal(p.reshape(2, -1), x)
    assert torch.equal(m.pat(x.unsqueeze(-1)), p)

    with pytest.raises(AssertionError):
        m.pat(torch.randn(2, CTX + 1))


@pytest.mark.parametrize("use_rope", [True, False])
def test_rollout(use_rope):
    m = tiny_model(use_rope=use_rope)
    x = torch.randn(2, CTX)

    assert m.rollout(x, num_patches=1).shape == (2, 1, PATCH)

    # The context is used as given: step 1 is exactly forward's
    # last-position prediction.
    with torch.no_grad():
        full = m.forward(x)[0][:, -1]
    assert torch.allclose(m.rollout(x, num_patches=1)[:, 0], full, atol=1e-6)

    # Every context patch can move the forecast.
    for sl in (slice(0, PATCH), slice(-PATCH, None)):
        x2 = x.clone()
        x2[:, sl] = torch.randn_like(x2[:, sl])
        assert not torch.allclose(
            m.rollout(x2, num_patches=1), m.rollout(x, num_patches=1)
        )

    # The window slides: step 2 is not a copy of step 1.
    p = m.rollout(x, num_patches=2)
    assert p.shape == (2, 2, PATCH)
    assert not torch.allclose(p[:, 1], p[:, 0])


def test_training_window():
    # T + patch_size: one patch longer than the rollout window, so the
    # shifted loss has a target for the window's last position. RoPE only:
    # emb_pos has num_patches rows, one short of this window.
    m = tiny_model()
    x = torch.randn(2, CTX + PATCH)
    pred, _, z = m(x)
    assert z.shape == (2, CTX // PATCH + 1, 16)
    assert pred[:, :-1].shape == m.pat(x)[:, 1:].shape
