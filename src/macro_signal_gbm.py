"""
Nonlinear version of macro_signal_check.py: can gradient-boosted trees, which build interactions by themselves, find
macro signal in how firms move that the linear test missed? No training of the encoder
(docs/archive/report_2026-11-01_macro_search.md).

    cd src && ../xbrl2vec/bin/python macro_signal_gbm.py [--h 4] [--outcomes all|growth+]

Target: the change of the fingerprint over the next h quarters (demeaned per quarter, unit variance per ratio); with
--outcomes growth+ also the log growth of trailing revenue and of total assets. Features: base = the firm's
fingerprint today, its change over the previous 4 quarters, its sector (40 largest SIC descriptions, the rest
pooled); + the macro change s (Fed funds, CPI, GDP, unemployment) over t - h -> t ("past") or t -> t + h
("realised"). XGBoost, one output per tree, 200 trees of depth 4, learning rate 0.05, the same for every variant.
Trained up to 2023Q3 - h, scored from 2023Q4 (out of time): R² and fit2 (Spearman of the predicted vs the actual
change cosines, 1500 firms per quarter). Placebo: s shifted back by 8, 12, ..., 60 quarters.
Output: data/out/macro_signal/gbm_h<h>_<outcomes>.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

import macro_signal_check as m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h", type=int, default=4)
    p.add_argument("--outcomes", choices=["all", "growth+"], default="all")
    p.add_argument("--shifts", type=int, nargs="*", default=list(range(8, 61, 4)))
    p.add_argument("--n_jobs", type=int, default=6)
    args = p.parse_args()
    h = args.h
    F = pd.read_parquet(m.PANELS / "filled" / "targets" / "fingerprint.parquet")
    sector = pd.read_parquet(m.PANELS / "metadata.parquet").drop_duplicates("ticker").set_index("ticker").sector
    cols = [c for c in F.columns if c not in ("ticker", "quarter")]
    D = m.changes(F, h)
    D = D[D.quarter >= m.FIRST]
    outcomes = cols
    if args.outcomes == "growth+":
        D = D.merge(m.growth(m.PANELS / "filled", h), on=["ticker", "quarter"], how="left").dropna(subset=m.GROWTH)
        outcomes = cols + m.GROWTH
    keys = pd.MultiIndex.from_frame(D[["ticker", "quarter"]])
    Fi = F.set_index(["ticker", "quarter"])
    f = Fi.loc[keys, cols].to_numpy(np.float32)
    past = m.changes(F, 4).assign(quarter=lambda d: d.quarter.map(lambda q: m.nq(m.qn(q) + 4)))
    past = past.set_index(["ticker", "quarter"]).reindex(keys)[cols].to_numpy(np.float32)    # f(t) - f(t - 4)
    sec = D.ticker.map(sector).fillna("unknown")
    top = list(sec.value_counts().index[:40])
    code = sec.map({s: i for i, s in enumerate(top)}).fillna(40).to_numpy(np.float32)[:, None]
    qv = D.quarter.to_numpy()
    quarters = sorted(set(qv))
    Y = D[outcomes].to_numpy(np.float32)
    Y = Y - pd.DataFrame(Y).groupby(qv).transform("mean").to_numpy()
    Y = Y / Y.std(0)
    tr = np.array([m.qn(q) <= m.qn("2023Q3") - h for q in qv])
    te = qv >= "2023Q4"
    base_X = np.hstack([f, past, code])
    rng = np.random.default_rng(0)
    samples = [rng.choice(np.flatnonzero(qv == q), min(1500, int((qv == q).sum())), replace=False) for q in sorted(set(qv[te]))]
    unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

    def score(X):
        model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, tree_method="hist",
                                 multi_strategy="one_output_per_tree", n_jobs=args.n_jobs, max_bin=64, random_state=0)
        model.fit(X[tr], Y[tr])
        P = np.zeros_like(Y)
        P[te] = model.predict(X[te])
        r2 = 1 - ((Y[te] - P[te]) ** 2).sum() / (Y[te] ** 2).sum()
        fit2 = np.mean([spearmanr((unit(P[i]) @ unit(P[i]).T)[np.triu_indices(len(i), 1)],
                                  (unit(Y[i]) @ unit(Y[i]).T)[np.triu_indices(len(i), 1)])[0] for i in samples])
        return float(r2), float(fit2)

    L = m.macro_levels()
    rows = [{"macro": "none", "shift": 0, **dict(zip(("R2", "fit2"), score(base_X)))}]
    print(f"[INFO] base: R2 {rows[0]['R2']:.4f}, fit2 {rows[0]['fit2']:.4f}", flush=True)
    for timing in ("past", "realised"):
        for s in [0] + args.shifts:
            S = pd.DataFrame(m.scenario(L, quarters, h, timing, s), index=quarters).loc[qv].to_numpy(np.float32)
            r2, fit2 = score(np.hstack([base_X, S]))
            rows.append({"macro": timing, "shift": s, "R2": r2, "fit2": fit2})
            print(f"[INFO] {timing} shift {s}: R2 {r2:.4f}, fit2 {fit2:.4f}", flush=True)
    res = pd.DataFrame(rows)
    m.OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(m.OUT / f"gbm_h{h}_{args.outcomes.replace('+', 'plus')}.csv", index=False)
    base = res.iloc[0]
    print(f"\nh {h}, outcomes {args.outcomes}; base R2 {base.R2:.4f}, fit2 {base.fit2:.4f}")
    for timing in ("past", "realised"):
        g = res[res.macro == timing]
        real, plc = g[g["shift"] == 0].iloc[0], g[g["shift"] != 0]
        for k in ("R2", "fit2"):
            print(f"  {timing:8s} {k:4s}: gain real {real[k] - base[k]:+.4f}, placebo max {plc[k].max() - base[k]:+.4f}, "
                  f"placebos >= real {int((plc[k] >= real[k]).sum())}/{len(plc)}")


if __name__ == "__main__":
    main()
