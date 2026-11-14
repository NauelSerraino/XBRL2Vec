"""
G2 check for scenario-conditioned runs (main_similarity.py --scenario): does a rate scenario pull banks together?
(docs/archive/report_2026-11-01_macro_search.md)

    cd src && ../xbrl2vec/bin/python bank_scenario_check.py <run> [<run> ...]

For every quarter (at most 1500 firms per quarter, training and unseen firms together), the firms' embeddings z of the
run's encoder (macro and no-macro) are reweighted by its scenario head under the neutral scenario (s = 0) and under a
rate scenario (Fed funds change +2 or -2 standard deviations, the other series 0; for heads with the past macro,
also the past Fed funds change: "after a hike year"). For the banks (the 5 bank SIC
descriptions):
  bank share of 10-NN   share of a bank's 10 nearest neighbours that are banks, neutral vs rate scenario
  bank cos gap          mean cosine bank-bank minus bank-other, neutral vs rate scenario
  rate sensitivity      per firm, r_i = Σ_k w_k z_ik² / Σ_k z_ik², w = the head's Fed funds weights (how much a rate
                        rise reweights the firm's own dimensions); the banks' rank among sectors with >= 15 firms
Split into the training quarters and the quarters after the cutoff. A placebo run (--scenario_shift) is checked the
same way and should show nothing.
Output: data/out/similarity/<run>/explain/07_banks_scenario.csv, printed.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from explain_run import RUNS, load
from models.similarity_encoder import ScenarioHead
from services.config import DEVICE
from services.similarity import embed

BANKS = {"State Commercial Banks", "National Commercial Banks", "Commercial Banks, NEC",
         "Savings Institution, Federally Chartered", "Savings Institutions, Not Federally Chartered"}


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def bank_stats(Z: np.ndarray, bank: np.ndarray, k: int = 10) -> tuple[float, float]:
    U = _unit(Z)
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S[bank], k, axis=1)[:, :k]
    share = float(bank[nn].mean())
    np.fill_diagonal(S, np.nan)
    gap = float(np.nanmean(S[np.ix_(bank, bank)]) - np.nanmean(S[np.ix_(bank, ~bank)]))
    return share, gap


def check(run: str, max_firms: int = 1500, seed: int = 0) -> pd.DataFrame:
    cfg, args, sets, _, models, sector = load(run)
    if not args.scenario:
        raise SystemExit(f"{run} is not a scenario run")
    ms = __import__("main_similarity")
    names = list(ms.scenarios_raw(args, ["2015Q1"]).columns)          # the head's inputs, in order
    feds = {lab: names.index(c) for lab, c in (("FF", "DFF"), ("FF past", "DFF_past")) if c in names}
    rng = np.random.default_rng(seed)
    rows = []
    for name, model in models.items():
        head = ScenarioHead(__import__('main_similarity').scenario_width(args), args.latent_dim)
        head.load_state_dict(torch.load(RUNS / run / f"scenario_{name.replace(' ', '_')}.pth", map_location="cpu"))
        head.eval()
        w = head.net.weight[:, feds.get("FF past", feds.get("FF"))].detach().numpy()
        for set_name, D in sets.items():
            Z = embed(model, D, DEVICE, "real" if name == "macro" else "none")
            r = (w * Z ** 2).sum(1) / (Z ** 2).sum(1)
            meta = D.meta.assign(r=r, sector=D.meta.ticker.map(sector))
            for q, g in meta.groupby("quarter"):
                idx = g.index.to_numpy()
                if len(idx) > max_firms:
                    idx = rng.choice(idx, max_firms, replace=False)
                bank = meta.loc[idx, "sector"].isin(BANKS).to_numpy()
                if bank.sum() < 20:
                    continue
                z = torch.tensor(Z[idx])
                row = {"model": name, "firms": set_name, "quarter": q,
                       "period": "train" if q <= cfg["last_train_quarter"] else ("after cutoff" if q > cfg["cutoff"] else "embargo")}
                scen = [("neutral", None, 0.0)] + [(f"{lab} {v:+.0f}sd", j, v) for lab, j in feds.items() for v in (2.0, -2.0)]
                for label, j, value in scen:
                    s = torch.zeros(len(idx), ms.scenario_width(args))
                    if j is not None:
                        s[:, j] = value
                    with torch.no_grad():
                        zs = head(z, s).numpy()
                    row[f"bank share 10-NN {label}"], row[f"bank cos gap {label}"] = bank_stats(zs, bank)
                rows.append(row)
            sec = meta.groupby("sector").r.agg(["mean", "size"])
            sec = sec[sec["size"] >= 15].sort_values("mean", ascending=False)
            ranks = [i + 1 for i, s in enumerate(sec.index) if s in BANKS]
            print(f"[INFO] {run} {name:8s} {set_name:12s}: rate sensitivity, bank sectors at ranks {ranks} of {len(sec)}; "
                  f"top: {list(sec.index[:3])}")
    res = pd.DataFrame(rows)
    out = RUNS / run / "explain"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "07_banks_scenario.csv", index=False)
    cols = [c for c in res.columns if c.startswith("bank")]
    summary = res.groupby(["model", "period"])[cols].mean()
    for lab in [f"{k} {v}sd" for k in feds for v in ("+2", "-2")]:
        summary[f"share change {lab}"] = summary[f"bank share 10-NN {lab}"] - summary["bank share 10-NN neutral"]
        summary[f"cos gap change {lab}"] = summary[f"bank cos gap {lab}"] - summary["bank cos gap neutral"]
    pd.set_option("display.width", 250)
    print(summary[[c for c in summary.columns if "change" in c or "neutral" in c]].round(4).to_string())
    return res


if __name__ == "__main__":
    for run in sys.argv[1:]:
        check(run)
