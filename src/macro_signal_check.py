"""
Model-free test: is there macro signal in how firms move, beyond their structure, and where? No training
(docs/archive/report_2026-11-01_macro_search.md, step 1).

    cd src && ../xbrl2vec/bin/python macro_signal_check.py

Linear model of the fingerprint change of every firm-quarter, Δf_i(t) = f_i(t + h) − f_i(t) (demeaned per quarter,
each ratio scaled to unit variance), on
  base           the firm's structure today f_i(t) (the 32 ratios) and its sector (SIC description: the 40 largest,
                 the rest pooled)
  + s ⊗ f        the structure interacted with the macro change s(t): firms with a given structure move more when the
                 macro moves (exposures carried by the structure; what a macro-aware encoder could learn)
  + s ⊗ sector   the sector interacted with the macro change (sector exposures, the sector ridge idea)
s(t) = the changes of Fed funds, CPI, GDP and unemployment (exog.parquet levels) over the target period t -> t + h
("realised", the scenario side) or over t - h -> t ("past", what the encoder's input knows), standardised.
Score: out-of-sample R² gain of the interactions over the base, (a) leave-one-year-out over 2013–2024, leaving out of
training the firm-quarters whose window overlaps the test year (± h quarters), and (b) the scorecard split (train up
to 2023Q3 − h, test from 2023Q4). Placebo: s shifted back in time by 8, 12, ..., 60 quarters (same series, wrong
dates). Outcome sets: all 32 ratios, 7 economic ratios, growth (log change of trailing revenue and of total assets
over t -> t + h, clipped 1/99 % per quarter). Ridge with α = 0.01, 0.1, 1 x the number of training rows, all reported (the penalty is not chosen on the
real macro). All from per-quarter sufficient statistics, so every macro version costs a few small solves per fold. Variants: horizon h, all 32 ratios or 7 economic ones as outcomes, with or
without the windows that overlap the pandemic (2020Q1–2021Q2).
Output: data/out/macro_signal/results.csv (every configuration and macro version) and summary.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "preprocess"))
from fingerprint import changes, load_raw  # noqa: E402

PANELS = ROOT / "data" / "in" / "panels_2026-10-25"
OUT = ROOT / "data" / "out" / "macro_signal"
SERIES = ["DFF", "CPIAUCSL", "GDP", "UNEMPLOY"]
FOCUSED = ["Net Income", "Operating Income", "Interest Expense", "Operating Cash Flow", "End Cash Position",
           "Long Term Debt", "Current Debt"]
GROWTH = ["revenue growth", "asset growth"]
FIRST, PAN = "2012Q1", ("2020Q1", "2021Q2")


def qn(q: str) -> int:
    return int(q[:4]) * 4 + int(q[-1]) - 1


def nq(n: int) -> str:
    return f"{n // 4}Q{n % 4 + 1}"


def macro_levels() -> pd.DataFrame:
    e = pd.read_parquet(PANELS / "exog.parquet").set_index("observation_date")[SERIES]
    return e.reindex([nq(k) for k in range(qn(e.index.min()), qn(e.index.max()) + 1)]).interpolate(limit_area="inside")


def scenario(L: pd.DataFrame, quarters: list[str], h: int, timing: str, shift: int) -> np.ndarray:
    """[Q, 4] standardised macro change per quarter: over t -> t + h (realised) or t - h -> t (past), dates - shift."""
    a, b = (h, 0) if timing == "realised" else (0, -h)
    S = np.stack([L.loc[nq(qn(q) - shift + a)].to_numpy() - L.loc[nq(qn(q) - shift + b)].to_numpy() for q in quarters])
    return (S - S.mean(0)) / S.std(0)


def growth(panel: Path, h: int) -> pd.DataFrame:
    """ticker, quarter t, log change of trailing-12-month revenue and of total assets over t -> t + h."""
    L, _ = load_raw(panel)
    rev = L["Total Revenue"].T.rolling(4, min_periods=4).sum().T
    out = {}
    for name, v in (("revenue growth", rev), ("asset growth", L["Total Assets"])):
        v = v.where(v > 0)
        g = np.log(v.shift(-h, axis=1) / v).replace([np.inf, -np.inf], np.nan)
        out[name] = g.clip(g.quantile(0.01), g.quantile(0.99), axis=1).stack()
    return pd.DataFrame(out).rename_axis(["ticker", "quarter"]).reset_index()


def quarter_stats(F: pd.DataFrame, sector: pd.Series, h: int, outcomes: list[str], drop_pandemic: bool,
                  G: pd.DataFrame | None = None):
    """Per quarter: Z'Z, Z'Y, Σ y², n for Z = [structure, sector dummies], Y = the demeaned, scaled change."""
    D = changes(F, h)
    D = D[D.quarter >= FIRST]
    if G is not None:
        D = D.merge(G, on=["ticker", "quarter"], how="left")
    D = D.dropna(subset=outcomes)
    t = D.quarter.map(qn)
    if drop_pandemic:                                                  # windows t-h..t+h touching the pandemic
        D = D[~((t - h <= qn(PAN[1])) & (t + h >= qn(PAN[0])))]
    cols = [c for c in F.columns if c not in ("ticker", "quarter")]
    keys = pd.MultiIndex.from_frame(D[["ticker", "quarter"]])
    f = F.set_index(["ticker", "quarter"]).loc[keys, cols].to_numpy(float)
    Y = D[outcomes].to_numpy(float)
    Y = Y - pd.DataFrame(Y).groupby(D.quarter.to_numpy()).transform("mean").to_numpy()
    Y = Y / Y.std(0)
    sec = D.ticker.map(sector).fillna("unknown")
    top = sec.value_counts().index[:40]
    dummies = pd.get_dummies(sec.where(sec.isin(top), "other")).to_numpy(float)
    Z = np.hstack([f, dummies])
    quarters = sorted(D.quarter.unique())
    st = {}
    qv = D.quarter.to_numpy()
    for q in quarters:
        m = qv == q
        st[q] = (Z[m].T @ Z[m], Z[m].T @ Y[m], float((Y[m] ** 2).sum()), int(m.sum()))
    return quarters, st, len(cols), dummies.shape[1]


def fold_scores(quarters, st, S, pf, pd_, folds, alphas=(0.01, 0.1, 1.0)) -> dict:
    """{alpha: pooled out-of-sample R² of base / + s⊗f / + s⊗sector} over the folds [(train quarters, test quarters)]."""
    pz = pf + pd_
    a = np.hstack([np.ones((len(S), 1)), S])                           # [Q, 5]: 1, s
    blk = lambda b, part: list(range(b * pz, b * pz + pf)) if part == "f" else list(range(b * pz + pf, (b + 1) * pz))
    base = list(range(pz))
    idx = {"base": base, "s x f": base + sum((blk(b, "f") for b in range(1, 5)), []),
           "s x sector": base + sum((blk(b, "d") for b in range(1, 5)), [])}
    XtX = {q: np.kron(np.outer(a[i], a[i]), st[q][0]) for i, q in enumerate(quarters)}
    XtY = {q: np.kron(a[i][:, None], st[q][1]) for i, q in enumerate(quarters)}
    sse = {(al, k): 0.0 for al in alphas for k in idx}
    sst = 0.0
    for train, test in folds:
        A = sum(XtX[q] for q in train)
        B = sum(XtY[q] for q in train)
        n = sum(st[q][3] for q in train)
        At, Bt = sum(XtX[q] for q in test), sum(XtY[q] for q in test)
        yy = sum(st[q][2] for q in test)
        sst += yy
        for al in alphas:
            for k, ix in idx.items():
                beta = np.linalg.solve(A[np.ix_(ix, ix)] + al * n * np.eye(len(ix)), B[ix])
                sse[al, k] += yy - 2 * (beta * Bt[ix]).sum() + (beta * (At[np.ix_(ix, ix)] @ beta)).sum()
    r2 = {key: 1 - v / sst for key, v in sse.items()}
    return {al: {"R2 base": r2[al, "base"], "gain s x f": r2[al, "s x f"] - r2[al, "base"],
                 "gain s x sector": r2[al, "s x sector"] - r2[al, "base"]} for al in alphas}


def folds_for(quarters: list[str], h: int) -> dict:
    loyo = []
    for y in range(2013, 2025):
        test = [q for q in quarters if q.startswith(str(y))]
        lo, hi = qn(f"{y}Q1") - h, qn(f"{y}Q4") + h
        train = [q for q in quarters if not lo <= qn(q) <= hi]
        if test:
            loyo.append((train, test))
    oot = [([q for q in quarters if qn(q) <= qn("2023Q3") - h], [q for q in quarters if q >= "2023Q4"])]
    return {"leave one year out": loyo, "out of time": oot}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", type=Path, default=PANELS / "filled")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--shifts", type=int, nargs="*", default=list(range(8, 61, 4)))
    p.add_argument("--outcome_sets", nargs="+", default=["all 32", "7 economic"], choices=["all 32", "7 economic", "growth"])
    p.add_argument("--tag", default="", help="Suffix of the output files.")
    args = p.parse_args()
    F = pd.read_parquet(args.panel / "targets" / "fingerprint.parquet")
    sector = pd.read_parquet(PANELS / "metadata.parquet").drop_duplicates("ticker").set_index("ticker").sector
    L = macro_levels()
    allc = [c for c in F.columns if c not in ("ticker", "quarter")]
    rows = []
    for h in args.horizons:
        G = growth(args.panel, h) if "growth" in args.outcome_sets else None
        for oname, outcomes in ((o, {"all 32": allc, "7 economic": FOCUSED, "growth": GROWTH}[o]) for o in args.outcome_sets):
            for drop in (False, True):
                quarters, st, pf, pd_ = quarter_stats(F, sector, h, outcomes, drop, G)
                folds = folds_for(quarters, h)
                for timing in ("realised", "past"):
                    for s in [0] + args.shifts:
                        S = scenario(L, quarters, h, timing, s)
                        for fname, fl in folds.items():
                            for al, r in fold_scores(quarters, st, S, pf, pd_, fl).items():
                                rows.append({"h": h, "outcomes": oname, "pandemic": "excluded" if drop else "included",
                                             "macro": timing, "folds": fname, "alpha": al, "shift": s, **r})
                print(f"[INFO] h {h}, {oname}, pandemic {'excluded' if drop else 'included'}: done", flush=True)
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"results{args.tag}.csv", index=False)
    keys = ["h", "outcomes", "pandemic", "macro", "folds", "alpha"]
    out = []
    for k, g in res.groupby(keys):
        real, plc = g[g["shift"] == 0].iloc[0], g[g["shift"] != 0]
        row = dict(zip(keys, k)) | {"R2 base": real["R2 base"]}
        for m in ("gain s x f", "gain s x sector"):
            row |= {f"{m}": real[m], f"{m} placebo max": plc[m].max(), f"{m} placebos >= real": int((plc[m] >= real[m]).sum())}
        out.append(row)
    summ = pd.DataFrame(out)
    summ.to_csv(OUT / f"summary{args.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    fmt = summ.copy()
    for c in fmt.columns:
        if c.startswith(("gain", "R2")) and not c.endswith("real"):
            fmt[c] = (fmt[c] * 1000).round(2)                         # in thousandths of R²
    print("\nR² and gains in thousandths (x 1000); 'placebos >= real' out of", len(args.shifts))
    print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
