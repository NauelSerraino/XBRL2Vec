"""
Exposure-based target, feasibility check (no training): is a firm's sensitivity of its accounts to the macro a stable
property that separates firms (banks in particular) better than chance? Decides whether "similarity of macro
sensitivities" can be a Y (docs/pillar1_benchmark.md).

    cd src && ../xbrl2vec/bin/python exposure_check.py              # quarter-on-quarter changes
    cd src && ../xbrl2vec/bin/python exposure_check.py --change y   # year-on-year changes (lagged responses)
    cd src && ../xbrl2vec/bin/python exposure_check.py --change y --by sector [--fed_only]   # one regression per sector

For every firm i and quarter t: ridge regression, over the W quarters up to t (pandemic quarters 2020Q1–2021Q2 left
out), of K accounting changes on M macro changes; β_i(t) = the M x K slopes (macro standardised in the window).
  outcomes   changes of revenue / assets, net income / assets, interest expense / assets, operating cash flow / assets
             (flows over the trailing 12 months), cash / assets, liabilities / assets, log trailing revenue; reported
             values only (interpolated cells of the filled panel left out), from 2012Q1; per quarter clipped at 1/99 %
             and minus the cross-sectional median (relative exposure: what separates firms, not what moves them all)
  macro      changes of Fed funds, CPI, GDP, unemployment (exog.parquet)
Placebo: the same with the macro shifted back in time by 8, 12, ..., 60 quarters (same series, wrong dates). Tests:
  stability  own      mean cos(β_i(t), β_i(t + h)), h = 20 (windows do not overlap)
             other    the same against a random other firm / a random firm of the same sector
             rank     cross-sectional Spearman of the Fed funds -> revenue / assets slope between t and t + h
  sectors    share of the cross-sectional variance of β explained by sector (SIC description, >= 15 firms)
  banks      bank minus non-bank mean of the Fed funds slopes (in cross-sectional standard deviations); mean cosine
             bank-bank minus bank-other; rank of the banks among sectors on the Fed funds -> revenue / assets slope
--by sector: one ridge regression per sector (>= 15 firms) and outcome, the sector's firm-quarters pooled with firm
fixed effects (`sector_betas`), on the first and the second half of the sample: split-half agreement of the sector
betas (same coefficient, sectors ranked alike; each sector's beta vector, cosine) and the banks' Fed funds slopes
(z-score across sectors, mean rank of the 5 bank sectors) on the whole sample.
Output: data/out/exposure_check/summary_<by>_<change>.csv (real and every placebo), banks_by_quarter_<change>.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "preprocess"))
from fingerprint import _shift, load_raw  # noqa: E402

PANELS = ROOT / "data" / "in" / "panels_2026-10-25"
OUT = ROOT / "data" / "out" / "exposure_check"
MACRO = {"Fed funds": "DFF", "CPI": "CPIAUCSL", "GDP": "GDP", "unemployment": "UNEMPLOY"}
BANKS = {"State Commercial Banks", "National Commercial Banks", "Commercial Banks, NEC",
         "Savings Institution, Federally Chartered", "Savings Institutions, Not Federally Chartered"}
PANDEMIC = {"2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2"}
FIRST = "2012Q1"                                                       # wrong tags before (26 Oct report)
KEY = ("Fed funds", "revenue / assets")


def outcomes(panel: Path, step: int) -> dict[str, pd.DataFrame]:
    """{outcome: ticker x quarter} of relative changes over `step` quarters, reported values only."""
    L, _ = load_raw(panel)
    mask = pd.read_parquet(panel / "mask.parquet").drop(columns="stmt")
    for k in L:
        m = mask[mask.features == k].set_index("ticker").drop(columns="features").reindex_like(L[k])
        L[k] = L[k].where(~m.isin([0, 3]) & m.notna())
    ttm = lambda df: df.T.rolling(4, min_periods=4).sum().T
    ta = L["Total Assets"].where(L["Total Assets"] > 0)
    rev = ttm(L["Total Revenue"])
    levels = {"revenue / assets": rev / ta, "net income / assets": ttm(L["Net Income"]) / ta,
              "interest expense / assets": ttm(L["Interest Expense"]) / ta,
              "operating cash flow / assets": ttm(L["Operating Cash Flow"]) / ta,
              "cash / assets": L["End Cash Position"] / ta,
              "liabilities / assets": L["Total Liabilities Net Minority Interest"] / ta,
              "log revenue": np.log(rev.where(rev > 0))}
    first = _shift(FIRST, step + 3)                                    # both ends of the change from 2012Q1 on, TTM
    out = {}
    for k, v in levels.items():
        d = v.replace([np.inf, -np.inf], np.nan).diff(step, axis=1)
        d = d[[q for q in d.columns if q >= first]]
        d = d.clip(d.quantile(0.01), d.quantile(0.99), axis=1)
        d = d - d.median()
        out[k] = d / np.nanstd(d.to_numpy())
    return out


def macro(step: int, shift: int = 0) -> pd.DataFrame:
    """quarter x macro changes; shift s > 0: the value at q is the one of q - s (placebo)."""
    e = pd.read_parquet(PANELS / "exog.parquet").set_index("observation_date")
    X = e[[v + ("_DIFF_Q" if step == 1 else "_DIFF_Y") for v in MACRO.values()]].set_axis(list(MACRO), axis=1)
    return X.set_axis([_shift(q, shift) for q in X.index]) if shift else X


def betas(Y: dict[str, pd.DataFrame], X: pd.DataFrame, W: int, ridge: float, min_obs: int) -> dict[str, pd.DataFrame]:
    """{t: ticker x (macro, outcome) slopes} for every window end t."""
    names = list(Y)
    cols = list(Y[names[0]].columns)
    A = np.stack([Y[k][cols].to_numpy() for k in names], axis=-1)      # N, Q, K
    tick = Y[names[0]].index
    res = {}
    for t in cols[W - 1:]:
        win = [q for q in cols if _shift(t, -(W - 1)) <= q <= t and q not in PANDEMIC and q in X.index]
        y = A[:, [cols.index(q) for q in win], :]
        x = X.loc[win].to_numpy()
        x = (x - x.mean(0)) / x.std(0)
        v = np.isfinite(y).all(-1)
        n = v.sum(1)
        keep = n >= min_obs
        y, v, n = np.where(v[..., None], y, 0.0)[keep], v[keep], n[keep]
        xc = (x[None] - (v @ x / n[:, None])[:, None]) * v[..., None]
        yc = (y - (y.sum(1) / n[:, None])[:, None]) * v[..., None]
        B = np.linalg.solve(np.einsum("nwm,nwl->nml", xc, xc) + ridge * np.eye(x.shape[1]),
                            np.einsum("nwm,nwk->nmk", xc, yc))
        res[t] = pd.DataFrame(B.reshape(len(B), -1), index=tick[keep],
                              columns=pd.MultiIndex.from_product([list(X.columns), names]))
    return res


def sector_betas(Y: dict[str, pd.DataFrame], X: pd.DataFrame, quarters: list[str], sector: pd.Series, ridge: float,
                 min_firms: int = 15) -> pd.DataFrame:
    """sector x (macro, outcome): one ridge regression per sector and outcome over the given quarters, all the sector's
    firm-quarters pooled, each firm demeaned (firm fixed effects); ridge scaled by the number of firms."""
    names = list(Y)
    qs = [q for q in quarters if q in X.index and q not in PANDEMIC and q in Y[names[0]].columns]
    y = np.stack([Y[k][qs].to_numpy() for k in names], axis=-1)         # N, w, K
    x = X.loc[qs].to_numpy()
    x = (x - x.mean(0)) / x.std(0)
    v = np.isfinite(y).all(-1)
    n = v.sum(1)
    ok = n >= 8
    y, v, n = np.where(v[..., None], y, 0.0)[ok], v[ok], n[ok]
    xc = (x[None] - (v @ x / n[:, None])[:, None]) * v[..., None]
    yc = (y - (y.sum(1) / n[:, None])[:, None]) * v[..., None]
    XtX, XtY = np.einsum("nwm,nwl->nml", xc, xc), np.einsum("nwm,nwk->nmk", xc, yc)
    sec = sector.reindex(Y[names[0]].index[ok]).to_numpy()
    out = {}
    for s in pd.unique(sec[pd.notna(sec)]):
        g = sec == s
        if g.sum() >= min_firms:
            B = np.linalg.solve(XtX[g].sum(0) + ridge * g.sum() * np.eye(x.shape[1]), XtY[g].sum(0))
            out[s] = B.reshape(-1)
    return pd.DataFrame(out, index=pd.MultiIndex.from_product([list(X.columns), names])).T


def sector_scores(Y: dict[str, pd.DataFrame], X: pd.DataFrame, sector: pd.Series, ridge: float) -> dict:
    """Split-half check of the sector betas (first vs second half of the sample) and the banks' place in them."""
    cols = list(Y[next(iter(Y))].columns)
    half = [q for q in cols if q not in PANDEMIC and q in X.index]
    a = sector_betas(Y, X, half[:len(half) // 2], sector, ridge)
    b = sector_betas(Y, X, half[len(half) // 2:], sector, ridge)
    full = sector_betas(Y, X, cols, sector, ridge)
    common = a.index.intersection(b.index)
    za = (a.loc[common] - a.loc[common].mean()) / a.loc[common].std()     # each coefficient across sectors
    zb = (b.loc[common] - b.loc[common].mean()) / b.loc[common].std()
    agree = np.mean([za[c].corr(zb[c]) for c in za.columns])           # same coefficient, sectors ranked alike?
    unit = lambda d: d.to_numpy() / np.linalg.norm(d.to_numpy(), axis=1, keepdims=True)
    own = (unit(za) * unit(zb)).sum(1)                                 # sector's beta vector, first vs second half
    banks = [s for s in full.index if s in BANKS]
    zf = (full - full.mean()) / full.std()
    res = {"sector split-half agreement": agree, "sector split-half own cos": own.mean(),
           "sectors": len(common), "sector rank of banks FF->net income": np.nan}
    for o in ("net income / assets", "interest expense / assets", "revenue / assets"):
        c = zf[("Fed funds", o)]
        res[f"banks z FF->{o}"] = c[banks].mean()
        res[f"banks rank FF->{o}"] = float(c.rank(ascending=False)[banks].mean())
    res.pop("sector rank of banks FF->net income")
    return res


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def _unit(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def scores(B: dict[str, pd.DataFrame], sector: pd.Series, h: int, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    stab = []
    for t, a in B.items():
        if _shift(t, h) not in B:
            continue
        b = B[_shift(t, h)]
        common = a.index.intersection(b.index)
        a_, b_ = a.loc[common].to_numpy(), b.loc[common].to_numpy()
        sec = sector.reindex(common).to_numpy()
        partner = np.arange(len(common))
        for s in pd.unique(sec[pd.notna(sec)]):
            g = np.flatnonzero(sec == s)
            if len(g) > 1:
                sh = rng.permutation(g)
                partner[sh] = np.roll(sh, 1)
        same = partner != np.arange(len(common))
        rank = pd.Series(a.loc[common, KEY].to_numpy()).corr(pd.Series(b.loc[common, KEY].to_numpy()), method="spearman")
        stab.append([_cos(a_, b_).mean(), _cos(a_, b_[rng.permutation(len(common))]).mean(),
                     _cos(a_[same], b_[partner[same]]).mean(), rank])
    rows = []
    for t, b in B.items():
        sec = sector.reindex(b.index)
        big = (sec.map(sec.value_counts()) >= 15).to_numpy()
        x = b[big]
        x = (x - x.mean()) / x.std()
        r2 = float((x.groupby(sec[big].to_numpy()).transform("mean") ** 2).sum().sum() / (x ** 2).sum().sum())
        bank = sec.isin(BANKS).to_numpy()
        z = _unit(b.to_numpy())
        within = z[bank] @ z[bank].T
        gaps = {}
        for o in ("revenue / assets", "net income / assets", "interest expense / assets"):
            c = b[("Fed funds", o)]
            c = (c - c.mean()) / c.std()
            gaps[f"bank gap FF->{o}"] = c[bank].mean() - c[~bank].mean()
        group = sec.where(~sec.isin(BANKS), "BANKS")[big | bank]
        means = b.loc[(big | bank), KEY].groupby(group.to_numpy()).mean().sort_values(ascending=False)
        rows.append({"quarter": t, "sector R2": r2, **gaps,
                     "bank cos within - cross": (within.sum() - np.trace(within)) / (bank.sum() * (bank.sum() - 1))
                                                - (z[bank] @ z[~bank].T).mean(),
                     "bank rank FF->revenue": int(np.flatnonzero(means.index == "BANKS")[0]) + 1,
                     "sectors": len(means), "firms": len(b), "banks": int(bank.sum())})
    q = pd.DataFrame(rows)
    s = np.mean(stab, axis=0) if stab else np.full(4, np.nan)          # one window (whole sample): no stability
    summary = {"stability own": s[0], "stability random other": s[1], "stability same-sector other": s[2],
               "stability rank FF->revenue": s[3], **q.drop(columns=["quarter", "firms", "banks", "sectors"]).mean().to_dict()}
    return summary, q


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", type=Path, default=PANELS / "filled")
    p.add_argument("--change", choices=["q", "y"], default="q", help="Quarter-on-quarter or year-on-year changes.")
    p.add_argument("--window", type=int, default=20, help="Quarters per regression (W).")
    p.add_argument("--ridge", type=float, default=2.0, help="Ridge penalty on the standardised macro slopes.")
    p.add_argument("--min_obs", type=int, default=12)
    p.add_argument("--shifts", type=int, nargs="*", default=list(range(8, 61, 4)), help="Placebo shifts (quarters).")
    p.add_argument("--by", choices=["firm", "sector"], default="firm", help="One beta per firm-quarter or per sector.")
    p.add_argument("--fed_only", action="store_true", help="Fed funds as the only macro regressor.")
    args = p.parse_args()
    step = 1 if args.change == "q" else 4
    sector = pd.read_parquet(PANELS / "metadata.parquet").drop_duplicates("ticker").set_index("ticker").sector
    Y = outcomes(args.panel, step)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in [0] + args.shifts:
        X = macro(step, s)[["Fed funds"]] if args.fed_only else macro(step, s)
        if args.by == "sector":
            rows.append({"macro": "real" if s == 0 else f"shifted {s}", **sector_scores(Y, X, sector, args.ridge)})
            continue
        B = betas(Y, X, args.window, args.ridge, args.min_obs)
        summary, by_q = scores(B, sector, h=args.window)
        rows.append({"macro": "real" if s == 0 else f"shifted {s}", **summary})
        if s == 0:
            by_q.to_csv(OUT / f"banks_by_quarter_{args.change}.csv", index=False)
            print(f"[INFO] real macro: {len(B)} windows {min(B)}–{max(B)}, {by_q.firms.mean():.0f} firms "
                  f"({by_q.banks.mean():.0f} banks) per window, {by_q.sectors.mean():.0f} sectors")
        print(f"[INFO] {rows[-1]['macro']}: done")
    res = pd.DataFrame(rows).set_index("macro")
    res.to_csv(OUT / f"summary_{args.by}_{args.change}{'_fed' if args.fed_only else ''}.csv")
    real, plc = res.loc["real"], res.drop(index="real")
    table = pd.DataFrame({"real": real, "placebo mean": plc.mean(), "placebo max": plc.max(),
                          "share of placebos >= real": (plc >= real).mean()})
    pd.set_option("display.width", 200)
    print(f"\n{args.change}-changes, {args.by} betas{', Fed funds only' if args.fed_only else ''}, "
          f"{'W ' + str(args.window) + ', ' if args.by == 'firm' else ''}ridge {args.ridge}, {len(plc)} placebos\n")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
