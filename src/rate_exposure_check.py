"""
G2 check for encoders whose macro acts inside the encoder (e.g. --macro_conditioning cross_attention): which sectors
does each macro series move, and does the rate path pull banks together? (docs/archive/report_2026-11-01_macro_search.md)

    cd src && ../xbrl2vec/bin/python rate_exposure_check.py <run> [<run> ...]

Out of time (training and unseen firms after the cutoff), for each macro series m: a counterfactual in which m's path
in the encoder's macro window, and m's past change in the scenario head, are set to their training average (0 on the
standardised scale), everything else unchanged. Per firm-quarter, the displacement of the Pillar 2 embedding z̃,
d = 1 - cos(z̃ real, z̃ counterfactual): how much the firm's position depends on that series. Per sector (SIC
description, >= 15 firms): mean displacement and rank. For the banks (the 5 bank SIC descriptions): rank for each
series (G2 wants Fed funds first, and banks more exposed to Fed funds than to the other series), and the share of a
bank's 10 nearest neighbours that are banks, real vs Fed-funds counterfactual (does the real rate path pull banks
together?).
Output: data/out/similarity/<run>/explain/09_rate_exposure.csv (sector x series), printed.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from bank_scenario_check import BANKS, bank_stats
from explain_run import RUNS, load
from models.similarity_encoder import ScenarioHead
from services.config import DEVICE
from services.similarity import embed


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def check(run: str, max_firms: int = 1500, seed: int = 0) -> pd.DataFrame:
    cfg, args, sets, _, models, sector = load(run)
    ms = __import__("main_similarity")
    cutoff = cfg["cutoff"]
    tr, te = sets["train firms"], sets["unseen firms"]
    data = {"out of time": tr.subset(tr.meta.quarter.gt(cutoff).to_numpy()),
            "unseen firms, out of time": te.subset(te.meta.quarter.gt(cutoff).to_numpy())}
    head = None
    if args.scenario:
        head = ScenarioHead(ms.scenario_width(args), args.latent_dim)
        head.load_state_dict(torch.load(RUNS / run / "scenario_macro.pth", map_location="cpu"))
        quarters = sorted(set(tr.meta.quarter) | set(te.meta.quarter))
        scen = ms.scenarios(args, quarters, [q for q in quarters if q <= cfg["last_train_quarter"]])
        names = list(scen.columns)
        for d in data.values():
            d.scenario = scen.reindex(d.meta.quarter).to_numpy(np.float32)
    series = [c.replace("_DIFF_Y", "") for c in tr.macro_cols]
    rng = np.random.default_rng(seed)
    disp, nn_rows = {}, []
    for set_name, d in data.items():
        Zr = embed(models["macro"], d, DEVICE, "real", head=head)
        sec = d.meta.ticker.map(sector)
        for j, s in enumerate(series):
            X0, S0 = d.X_macro, d.scenario
            d.X_macro = d.X_macro.clone()
            d.X_macro[:, :, j] = 0.0
            if head is not None:
                d.scenario = d.scenario.copy()
                for k, n in enumerate(names):
                    if n.split("_past")[0] == s and "_past" in n:
                        d.scenario[:, k] = 0.0
            Zc = embed(models["macro"], d, DEVICE, "real", head=head)
            d.X_macro, d.scenario = X0, S0
            disp[(set_name, s)] = pd.Series(1 - (_unit(Zr) * _unit(Zc)).sum(1), index=d.meta.index)
            if s == "DFF":
                for q, g in d.meta.groupby("quarter"):
                    idx = g.index.to_numpy()
                    if len(idx) > max_firms:
                        idx = rng.choice(idx, max_firms, replace=False)
                    bank = sec.loc[idx].isin(BANKS).to_numpy()
                    if bank.sum() >= 20:
                        nn_rows.append({"set": set_name, "quarter": q, "bank share real": bank_stats(Zr[idx], bank)[0],
                                        "bank share no FF path": bank_stats(Zc[idx], bank)[0]})
        firms = d.meta.assign(sector=sec).groupby("sector").ticker.nunique()
        big = firms[firms >= 15].index
        table = pd.DataFrame({s: disp[(set_name, s)].groupby(sec).mean().reindex(big) for s in series})
        ranks = table.rank(ascending=False)
        bank_rank = ranks.loc[ranks.index.isin(BANKS)]
        print(f"\n[{run}] {set_name}: {len(big)} sectors; mean displacement x1000 when a series' path is removed")
        print("  all firms : " + ", ".join(f"{s} {table[s].mean() * 1000:.2f}" for s in series))
        print("  banks     : " + ", ".join(f"{s} {table.loc[table.index.isin(BANKS), s].mean() * 1000:.2f}" for s in series))
        print("  bank ranks: " + ", ".join(f"{s} {sorted(bank_rank[s].astype(int).tolist())}" for s in series) + f" (of {len(big)})")
        print("  top 3 sectors for DFF: " + "; ".join(table["DFF"].sort_values(ascending=False).index[:3]))
        table.assign(set=set_name).to_csv(RUNS / run / "explain" / f"09_rate_exposure_{'oot' if set_name == 'out of time' else 'unseen_oot'}.csv")
    nn = pd.DataFrame(nn_rows)
    if len(nn):
        g = nn.groupby("set")[["bank share real", "bank share no FF path"]].mean()
        g["real minus no FF"] = g["bank share real"] - g["bank share no FF path"]
        print("\nbank share of 10-NN, real Fed funds path vs average path:")
        print(g.round(4).to_string())
    return nn


if __name__ == "__main__":
    for run in sys.argv[1:]:
        (RUNS / run / "explain").mkdir(exist_ok=True)
        check(run)
