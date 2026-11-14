from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from fbias.data import b_of, make_mixture
from fbias.logging import get_logger
from fbias.model import SimTFM
from fbias.probes import phase_error, retention

logger = get_logger(__name__)


def _probe(model, args, device):
    r"""Retention and phase error per probe frequency, at one forecast position."""
    x = make_mixture(args.freqs, args.n_per, args.ctx, args.patch, args.seed + 1).to(
        device
    )
    model.eval()
    with torch.no_grad():
        pred, _, _ = model(x)
        patches = model.pat(x)
        # pred[:, j] predicts patch j + 1, so the last entry has no target.
        f_hat = pred[:, :-1][:, args.probe_pos]
        true = patches[:, 1:][:, args.probe_pos]

    rows = []
    for f in args.freqs:
        b = b_of(f, args.patch, args.ctx)
        rows.append(
            {
                "f": f,
                "b": b,
                "dphi": (2 * math.pi * b) % (2 * math.pi),
                "retention": float(retention(f_hat, true, b, args.patch).mean()),
                "phase_error": float(phase_error(f_hat, true, b, args.patch).mean()),
            }
        )
    model.train()
    return rows


def main(args):
    device = torch.device(args.device)
    assert args.ctx % args.patch == 0, "context_size must be a multiple of patch"
    torch.manual_seed(args.seed)

    model = SimTFM(
        context_size=args.ctx,
        patch_size=args.patch,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_attn_heads=args.heads,
    ).to(device)
    data = make_mixture(args.freqs, args.n_per, args.ctx, args.patch, args.seed).to(
        device
    )
    n_patches = args.ctx // args.patch
    assert -n_patches <= args.probe_pos < n_patches, "probe pos out of range"
    # ctx + patch: one patch longer than the rollout context, so the shifted
    # loss has a target for the window's last position (see SimTFM.rollout).
    assert data.shape[1] == args.ctx + args.patch, "window must be ctx + patch"

    gen = torch.Generator().manual_seed(args.seed)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    log_every = max(1, args.steps // 10)
    history = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(data.shape[0], (args.batch,), generator=gen)
        x = data[idx]
        pred, _, _ = model(x)
        loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % log_every == 0 or step == args.steps:
            history.append([step, float(loss)])
            logger.info(f"step {step}/{args.steps} loss {float(loss):.6f}")

    probe = _probe(model, args, device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    record = {
        "config": {k: v for k, v in model.config.items() if not k.startswith("__")},
        "corpus": {
            "freqs": list(args.freqs),
            "b": [b_of(f, args.patch, args.ctx) for f in args.freqs],
            "n_per": args.n_per,
            "ctx": args.ctx,
            "patch": args.patch,
            "seed": args.seed,
            "amp_jitter": 0.0,
            "phase_jitter": 0.0,
        },
        "training": {
            "optimizer": "SGD",
            "steps": args.steps,
            "batch": args.batch,
            "lr": args.lr,
            "seed": args.seed,
            "device": str(device),
        },
        "final_loss": history[-1][1],
        "loss_history": history,
        "probe": {"pos": args.probe_pos, "seed": args.seed + 1, "per_freq": probe},
    }
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    logger.info(f"wrote {out} and {sidecar}")
    return 0


def add_cmd(subparsers):
    p = subparsers.add_parser(
        "train",
        help="Next patch prediction training.",
        description=(
            "Train a TFM model object through next patch prediction over synthetic "
            "data with controlled frequency distribution and noise level."
        ),
    )
    p.add_argument("--ctx", type=int, default=512, help="context_size timesteps")
    p.add_argument("--patch", type=int, default=32, help="patch_size timesteps")
    p.add_argument("--hidden", type=int, default=32, help="hidden_size")
    p.add_argument("--layers", type=int, default=2, help="decoder depth")
    p.add_argument("--heads", type=int, default=4, help="attention heads")
    p.add_argument("--steps", type=int, default=2000, help="training steps")
    p.add_argument("--batch", type=int, default=64, help="batch size")
    p.add_argument("--lr", type=float, default=1e-2, help="SGD learning rate")
    p.add_argument("--seed", type=int, default=0, help="corpus and init seed")
    p.add_argument("--n-per", type=int, default=256, help="windows in the corpus")
    p.add_argument(
        "--freqs",
        type=float,
        nargs="+",
        default=[2.0, 128.0],
        help="tone frequencies in cycles per window (>= ctx/patch apart)",
    )
    p.add_argument("--out", default="runs/train.pt", help="state_dict path")
    p.add_argument(
        "--probe-pos",
        type=int,
        default=-1,
        help="forecast position to probe; -1 is the one-step forecast",
    )
    p.add_argument("--device", default="cpu", help="torch device")
    p.set_defaults(func=main)
    return p
