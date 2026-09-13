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

    p, xh = m.rollout(x, num_steps=PATCH + 5)
    assert p.shape == (2, 2, PATCH)
    assert torch.equal(xh, p.reshape(2, -1)[:, : PATCH + 5])

    # The first patch is dropped, so scrambling it changes nothing.
    x2 = x.clone()
    torch.manual_seed(1)
    x2[:, :PATCH] = torch.randn_like(x2[:, :PATCH])
    assert torch.equal(m.rollout(x2, PATCH)[0], m.rollout(x, PATCH)[0])

    # The last patch is kept, so scrambling it changes the forecast.
    x3 = x.clone()
    torch.manual_seed(2)
    x3[:, -PATCH:] = torch.randn_like(x3[:, -PATCH:])
    assert not torch.allclose(m.rollout(x3, PATCH)[0], m.rollout(x, PATCH)[0])

    # The window slides: step 2 is not a copy of step 1.
    assert not torch.allclose(p[:, 1], p[:, 0])
