import torch

from bias.model import SimTFM

CTX, PATCH = 128, 32


def tiny_model():
    torch.manual_seed(0)
    return SimTFM(
        context_size=CTX,
        patch_size=PATCH,
        hidden_size=16,
        num_layers=2,
        num_attn_heads=4,
    )


def test_pat():
    model = tiny_model()
    x = torch.randn(2, CTX)
    patches = model.pat(x)
    assert patches.shape == (2, CTX // PATCH, PATCH)
    assert torch.equal(patches.reshape(2, -1), x)
    assert torch.equal(model.pat(x.unsqueeze(-1)), patches)


def test_rollout():
    model = tiny_model()
    x = torch.randn(2, CTX)
    assert model.rollout(x, num_patches=1).shape == (2, 1, PATCH)

    with torch.no_grad():
        full = model.forward(x)[0][:, -1]
    assert torch.allclose(model.rollout(x, num_patches=1)[:, 0], full, atol=1e-6)

    for sl in (slice(0, PATCH), slice(-PATCH, None)):
        changed = x.clone()
        changed[:, sl] = torch.randn_like(changed[:, sl])
        assert not torch.allclose(model.rollout(changed), model.rollout(x))

    patches = model.rollout(x, num_patches=2)
    assert patches.shape == (2, 2, PATCH)
    assert not torch.allclose(patches[:, 1], patches[:, 0])


def test_training_window_has_target_for_last_position():
    model = tiny_model()
    x = torch.randn(2, CTX + PATCH)
    pred, _, z = model(x)
    assert z.shape == (2, CTX // PATCH + 1, 16)
    assert pred[:, :-1].shape == model.pat(x)[:, 1:].shape


def test_predict_next_ignores_held_out_target():
    model = tiny_model()
    x = torch.randn(2, CTX + PATCH)
    pred, target = model.predict_next(x)
    changed = x.clone()
    changed[:, -PATCH:] = torch.randn_like(changed[:, -PATCH:])
    changed_pred, changed_target = model.predict_next(changed)

    assert torch.equal(pred, changed_pred)
    assert not torch.equal(target, changed_target)
