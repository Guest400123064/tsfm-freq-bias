r"""Contract A2's transient, run over the whole 2x2.

``power_x_pred.py`` reads the state at 2 000 steps -- P2b's config, held fixed so
the six cells are comparable -- and the low-power rows of its 2x2 land far below
their optimum. Two readings are consistent with that, and the experiment cannot
tell them apart on its own:

  (a) converged: the head really does withhold a band that carries a small share
      of the corpus power, which is what T2b's ``corr(r_probe, power) = +0.86``
      would look like in a controlled corpus;
  (b) under-training: contract A2's failure mode. P3 is the precedent -- on its
      8-tone corpora *nothing* had converged at 2 000 steps (its beta=0 arm lost
      6% with zero unpredictability in the corpus, while P2b's 3-tone corpus read
      1.0027 at the same step count).

This script changes one thing and nothing else: it trains every cell of the 2x2
for 10 000 steps, reading at 250 / 500 / 1 000 / 2 000 / 5 000 / 10 000. Same
corpus builder, same amplitudes and phase walks, same model, same probe, same
readout, same seed for every band -- all imported from ``power_x_pred`` rather
than re-implemented. The 2 000-step row must therefore reproduce the parent
run's per-seed row exactly (the parent evaluates only after training, so the
parameter trajectory to 2 000 steps is identical).

Seed count: this is a *training-length* diagnostic over one seed (the parent's
3-seed spreads at 2 000 steps are quoted beside it), not a sixth experiment. It
is reported as such.

Run from the repo root:
``.venv/bin/python experiments/2_power_x_pred/scripts/transient.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import power_x_pred as px
import torch
import torch.nn.functional as F

SEEDS = (0,)
STEPS_AT = (250, 500, 1000, 2000, 5000, 10000)
RUNS = Path(__file__).resolve().parents[1] / "runs"


def run(arm, seed):
    r"""P2b's loop, verbatim, with the readouts taken at each checkpoint."""
    torch.manual_seed(seed)
    model = px.SimTFM(
        context_size=px.CTX,
        patch_size=px.PATCH,
        hidden_size=px.HIDDEN,
        num_layers=px.LAYERS,
        num_attn_heads=px.HEADS,
    )
    data, floors = px.corpus(arm, seed)
    train_data, held = data[: px.N_PER], data[px.N_PER :]
    held_floor = floors.sum(0)[px.N_PER :]
    prb = px.probe(arm)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=px.LR)
    rows, step, loss = [], 0, torch.tensor(0.0)
    for target in STEPS_AT:
        model.train()
        while step < target:
            x = train_data[torch.randint(px.N_PER, (px.BATCH,), generator=gen)]
            pred, _, _ = model(x)
            loss = F.mse_loss(pred[:, :-1], model.pat(x)[:, 1:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        row = px.evaluate(model, held, held_floor, prb, float(loss))
        row["step"] = step
        rows.append(row)
    return rows


def cells(r, r_st):
    tb = px.BANDS[px.TARGET]
    return [
        f"{r['train_loss']:>11.4f}",
        f"{r['var_explained']:>+9.4f}",
        f"{r[f'rm@{tb}']:>9.4f}",
        f"{r[f'r@{tb}']:>9.4f}",
        f"{r[f'r@{tb}'] / r_st:>9.4f}",
        f"{r[f'r@{px.BANDS[0]}']:>8.4f}",
        f"{r[f'r@{px.BANDS[1]}']:>8.4f}",
        f"{r['rnull']:>8.4f}",
        f"{r[f'r_oracle@{tb}']:>9.4f}",
    ]


def report(rec):
    tb = px.BANDS[px.TARGET]
    print("\ncontract A2 transient, cell by cell (r* = the arm's optimum)")
    for arm, a in rec["arms"].items():
        for seed_rec in a["seeds"]:
            print(
                f"\n  {arm}  (rho = {px.ARMS[arm]['rho']:.0f}, beta = "
                f"{px.ARMS[arm]['beta']}, seed {seed_rec['seed']}, target band "
                f"b = {tb}, r* = {a['r_star']:.3f})"
            )
            px.table(
                [
                    f"  {'step':>6}",
                    f"{'train loss':>11}",
                    f"{'var expl':>9}",
                    f"{f'rm@{tb}':>9}",
                    f"{f'r@{tb}':>9}",
                    f"{'q = r/r*':>9}",
                    f"{'r@2.0':>8}",
                    f"{'r@4.0':>8}",
                    f"{'rnull':>8}",
                    f"{'oracle@8':>9}",
                ],
                [
                    [f"  {r['step']:>6}", *cells(r, a["r_star"])]
                    for r in seed_rec["rows"]
                ],
            )
    print("\n2 000 vs 10 000 steps: what survives training (q = r / r*)")
    px.table(
        [
            f"  {'arm':<7}",
            f"{'q @2000':>9}",
            f"{'q @10000':>10}",
            f"{'change':>9}",
            f"{'r @2000':>9}",
            f"{'r @10000':>10}",
            f"{'oracle@8':>9}",
            f"{'verdict':>28}",
        ],
        [
            [
                f"  {arm:<7}",
                *[
                    f"{a['seeds'][0]['rows'][i][f'r@{tb}'] / a['r_star']:>9.3f}"
                    for i in (3, 5)
                ],
                f"{q10 - q2:>+9.3f}",
                f"{a['seeds'][0]['rows'][3][f'r@{tb}']:>9.4f}",
                f"{a['seeds'][0]['rows'][5][f'r@{tb}']:>10.4f}",
                f"{a['seeds'][0]['rows'][5][f'r_oracle@{tb}']:>9.3f}",
                f"{read(a['seeds'][0], a['r_star']):>28}",
            ]
            for arm, a in rec["arms"].items()
            for q2, q10 in (
                (
                    a["seeds"][0]["rows"][3][f"r@{tb}"] / a["r_star"],
                    a["seeds"][0]["rows"][5][f"r@{tb}"] / a["r_star"],
                ),
            )
        ],
    )


def read(seed_rec, r_st):
    r"""One word for whether the cell's 2 000-step reading survives to 10 000."""
    tb = px.BANDS[px.TARGET]
    q2 = seed_rec["rows"][3][f"r@{tb}"] / r_st
    q10 = seed_rec["rows"][5][f"r@{tb}"] / r_st
    if abs(q2 - 1) <= px.Q_TOL and abs(q10 - 1) <= px.Q_TOL:
        return "at optimum throughout"
    if q10 > q2 + 0.10:
        return f"under-training (+{q10 - q2:.2f})"
    if q10 < q2 - 0.10:
        return f"still falling ({q10 - q2:+.2f})"
    return "stable below optimum"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arms", nargs="+", default=None, help="subset of arm ids")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--tag", default="", help="suffix for the output JSON")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    arms = list(args.arms or px.ARMS)
    print("== power_x_pred transient: contract A2 over the 2x2 ==")
    print(
        f"config as power_x_pred: ctx {px.CTX} / patch {px.PATCH} / hidden "
        f"{px.HIDDEN} / {px.LAYERS}L / {px.HEADS} heads | SGD lr {px.LR} / "
        f"batch {px.BATCH}"
    )
    print(f"arms {arms}; seeds {args.seeds}; readouts at steps {STEPS_AT}")
    print(
        f"target band b = {px.BANDS[px.TARGET]}, null band b = {px.NULL_B}; "
        "the 2 000-step row must reproduce power_x_pred.json's"
    )
    rec = {
        "config": {
            "arms": arms,
            "seeds": list(args.seeds),
            "steps_at": list(STEPS_AT),
            "target_b": px.BANDS[px.TARGET],
        },
        "arms": {},
        "wall": 0.0,
    }
    for arm in arms:
        entry = {
            "rho": px.ARMS[arm]["rho"],
            "beta": px.ARMS[arm]["beta"],
            "r_star": px.r_star(px.ARMS[arm]["beta"]),
            "seeds": [],
            "wall": 0.0,
        }
        for seed in args.seeds:
            t = time.time()
            rows = run(arm, seed)
            entry["seeds"].append({"seed": seed, "rows": rows})
            entry["wall"] += time.time() - t
            print(f"  {arm} seed {seed} done [{time.time() - t:.0f}s]", flush=True)
        rec["arms"][arm] = entry
    report(rec)
    rec["wall"] = time.time() - t0
    tag = f"_{args.tag}" if args.tag else ""
    out = RUNS / f"power_x_pred_transient{tag}.json"
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"\nwrote {out}\ntotal wall {rec['wall']:.1f}s")


if __name__ == "__main__":
    main()
