"""
G1 test for trained encoders, without retraining: out of time, does the encoder do better with the real macro window
than with the macro window of another period? (docs/archive/report_2026-11-01_macro_search.md)

    cd src && ../xbrl2vec/bin/python macro_placebo_eval.py <run> [<run> ...]

For runs whose macro input is the same for every firm (--macro_weight_mode none): every out-of-time window gets the
macro window of the quarter `shift` quarters earlier (8, 12, ..., 60), the company inputs unchanged; for --scenario_past runs, the past columns of the scenario are shifted the same way (all the macro
known at t), the realised ones kept. Windows older
than the panel are rebuilt from exog.parquet: the stored windows are an exact affine map (per series) of the symmetric
log of the exog series (transform_dataset, then the training scale); the map is fitted on the stored windows and the
script stops if it does not reproduce them to 1e-4. Scored with services.similarity.evaluate: fit and fit2 of the macro model (and, for --scenario runs, fit2
of the scenario-reweighted embedding, and fit2_scen_nomacro: the no-macro encoder with its own head, where the macro
enters only through the head). G1 needs the real macro above almost every shifted one.
Output: data/out/similarity/<run>/explain/08_macro_placebo.csv, printed.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from explain_run import RUNS, load
from models.similarity_encoder import ScenarioHead
from services.config import DEVICE
from services.similarity import embed, evaluate
from services.transforms import symmetric_log

OOT = ("out of time", "unseen firms, out of time")


def _qn(q: str) -> int:
    return int(q[:4]) * 4 + int(q[-1]) - 1


def check(run: str, shifts=range(8, 61, 4)) -> pd.DataFrame:
    cfg, args, sets, _, models, _ = load(run)
    if args.macro_weight_mode != "none":
        raise SystemExit(f"{run}: macro input is firm-specific ({args.macro_weight_mode}); this test needs 'none'")
    cutoff = cfg["cutoff"]
    tr, te = sets["train firms"], sets["unseen firms"]
    data = {"out of time": tr.subset(tr.meta.quarter.gt(cutoff).to_numpy()),
            "unseen firms, out of time": te.subset(te.meta.quarter.gt(cutoff).to_numpy())}
    window = {}                                                        # quarter -> its macro window (same for all firms)
    for d in (tr, te):
        for i, q in zip(range(len(d.meta)), d.meta.quarter):
            window.setdefault(q, d.X_macro[i])
    e = pd.read_parquet(args.panels_dir / "exog.parquet").set_index("observation_date")
    full = [f"{k // 4}Q{k % 4 + 1}" for k in range(_qn(e.index.min()), _qn(e.index.max()) + 1)]
    R = symmetric_log(torch.tensor(e.reindex(full)[tr.macro_cols].to_numpy(), dtype=torch.float32)).numpy()
    raw = lambda q: R[full.index(q) - args.t_in + 1: full.index(q) + 1]
    A = np.concatenate([raw(q) for q in window])
    W = np.concatenate([window[q].numpy() for q in window])
    coef = [np.polyfit(A[:, j], W[:, j], 1) for j in range(W.shape[1])]
    fitted = np.stack([a * A[:, j] + b for j, (a, b) in enumerate(coef)], axis=1)
    resid = float(np.abs(fitted - W).max())
    if resid > 1e-4:
        raise SystemExit(f"{run}: rebuilt macro windows differ from the stored ones by {resid:.2e}")
    for q in full[args.t_in - 1:]:
        x = raw(q)
        if q not in window and np.isfinite(x).all():
            window[q] = torch.tensor(np.stack([a * x[:, j] + b for j, (a, b) in enumerate(coef)], axis=1), dtype=torch.float32)
    print(f"[INFO] {run}: macro windows rebuilt from exog back to {min(window)} (max residual {resid:.1e})")
    head = head0 = None
    if args.scenario:
        head = ScenarioHead(__import__('main_similarity').scenario_width(args), args.latent_dim)
        head.load_state_dict(torch.load(RUNS / run / "scenario_macro.pth", map_location="cpu"))
        head0 = ScenarioHead(__import__('main_similarity').scenario_width(args), args.latent_dim)
        head0.load_state_dict(torch.load(RUNS / run / "scenario_no_macro.pth", map_location="cpu"))
        ms = __import__("main_similarity")
        qs = sorted(window)
        raw0 = ms.scenarios_raw(args, qs)
        train_q = [q for q in qs if q <= cfg["last_train_quarter"]]
        mu, sd = raw0.loc[train_q].mean(), raw0.loc[train_q].std()
        scen_of = lambda shift: (lambda S: (S - mu) / sd)(
            raw0.assign(**ms.scenarios_raw(args, qs, shift=shift).filter(like="_past"))
            if (args.scenario_past or args.scenario_past_only) else raw0)
    rows = []
    for shift in [0] + list(shifts):
        for set_name, d in data.items():
            src = [f"{(_qn(q) - shift) // 4}Q{(_qn(q) - shift) % 4 + 1}" for q in d.meta.quarter]
            if not all(q in window for q in src):
                continue
            orig = d.X_macro
            d.X_macro = torch.stack([window[q] for q in src])
            if head is not None:
                d.scenario = scen_of(shift).reindex(d.meta.quarter).to_numpy(np.float32)
            r = evaluate(embed(models["macro"], d, DEVICE, "real"), d)
            row = {"set": set_name, "shift": shift, "fit": r["fit"], "fit2": r["fit2"]}
            if head is not None:
                row["fit2_scen"] = evaluate(embed(models["macro"], d, DEVICE, "real", head=head), d)["fit2"]
                row["fit2_scen_nomacro"] = evaluate(embed(models["no macro"], d, DEVICE, "none", head=head0), d)["fit2"]
            d.X_macro = orig
            rows.append(row)
        print(f"[INFO] {run} shift {shift}: " + "; ".join(f"{r['set']} fit2 {r['fit2']:.4f}" for r in rows if r["shift"] == shift), flush=True)
    res = pd.DataFrame(rows)
    out = RUNS / run / "explain"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "08_macro_placebo.csv", index=False)
    for set_name, g in res.groupby("set"):
        real, plc = g[g["shift"] == 0].iloc[0], g[g["shift"] != 0]
        for k in [c for c in ("fit", "fit2", "fit2_scen", "fit2_scen_nomacro") if c in g]:
            print(f"  {run} {set_name:26s} {k:9s}: real {real[k]:.4f}, placebo mean {plc[k].mean():.4f}, "
                  f"max {plc[k].max():.4f}, placebos >= real {int((plc[k] >= real[k]).sum())}/{len(plc)}")
    return res


if __name__ == "__main__":
    for run in sys.argv[1:]:
        check(run)
