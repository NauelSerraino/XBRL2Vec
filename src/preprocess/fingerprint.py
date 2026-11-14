"""
Pillar 1 target: the financial fingerprint of every firm-quarter (notebook 01_pillar1_financial_fingerprints).

    python src/preprocess/fingerprint.py --panel data/in/panels_2026-10-25/filled

One vector per firm-quarter, from the raw statements of a panel ({bs,ins,cf}-raw.parquet):
  balance sheet and end cash   line / total assets
  income statement             trailing-12-month line / trailing-12-month revenue
  cash flow                    trailing-12-month line / total assets
without the lines that are empty, mirrored or redundant (DROP). Each quarter's cross-section is clipped at its 1st and
99th percentiles and standardised; a missing ratio (line not reported) is 0, the cross-sectional mean. --scaling rank
(add-on, <panel>/targets/fingerprint_rank.parquet): instead of the z-score, each ratio's normal score of its rank in
the quarter, Φ⁻¹((rank − 0.5) / n), so that a few extreme firms (tiny revenue) do not squeeze every other firm to ~0,
as the z-score does with the income-statement ratios. Firm-quarters
without positive total assets, or without any ratio, get no fingerprint. Training uses cos(f_i, f_j) of firms in the same quarter.

Output: <panel>/targets/fingerprint.parquet, columns ticker, quarter and one per ratio. Also, for each --horizons h
(default 4), the Pillar 2 target <panel>/targets/fingerprint_change_h<h>.parquet: Δf = f(t + h) − f(t), the change of
the firm's fingerprint over the next h quarters, indexed by t (rows only where both quarters have a fingerprint).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: lines left out: the denominators, the share count, always-empty and lease lines, and lines that mirror others
DROP = {"Total Assets", "Shares Outstanding", "Other Non-Current Assets", "Other Non-Current Liabilities",
        "Current Capital Lease Obligation", "Long Term Capital Lease Obligation",
        "Total Liabilities Net Minority Interest", "Total Non Current Assets", "Gross Profit", "Total Revenue"}
STATEMENTS = {"bs": "balance sheet", "ins": "income statement", "cf": "cash flow"}


def load_raw(panel: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """{line: ticker x quarter} and {line: statement} from a panel's raw files."""
    wide = pd.concat([pd.read_parquet(panel / f"{s}-raw.parquet").assign(stmt=s) for s in STATEMENTS], ignore_index=True)
    quarters = sorted(c for c in wide.columns if len(str(c)) == 6 and str(c)[4] == "Q")
    lines = {k: g.set_index("ticker")[quarters].astype(float) for k, g in wide.groupby("features")}
    tickers = lines["Total Assets"].index
    return {k: v.reindex(tickers) for k, v in lines.items()}, wide.groupby("features").stmt.first().to_dict()


def scale(F: pd.DataFrame, scaling: str = "zscore") -> pd.DataFrame:
    """Per quarter (index level "quarter") and ratio: z-score, or normal score of the rank among the reported values,
    or (slogz) the z-score of the signed log sign(x) log(1 + |x|), which compresses the extreme ratios before they set
    the standard deviation and keeps the size of differences; a missing ratio is 0 in all three."""
    if scaling == "slogz":
        F, scaling = np.sign(F) * np.log1p(F.abs()), "zscore"
    by_q = F.groupby(level="quarter")
    if scaling == "zscore":
        return ((F - by_q.transform("mean")) / by_q.transform("std")).fillna(0.0)
    from scipy.stats import norm
    u = (by_q.rank(method="average") - 0.5) / by_q.transform("count")
    return pd.DataFrame(norm.ppf(u), index=F.index, columns=F.columns).fillna(0.0)


def fingerprints(L: dict[str, pd.DataFrame], stmt_of: dict[str, str], scaling: str = "zscore") -> pd.DataFrame:
    """Long frame (ticker, quarter, one column per ratio), standardised per quarter (see scale)."""
    ttm = lambda df: df.T.rolling(4, min_periods=4).sum().T
    ta = L["Total Assets"].where(L["Total Assets"] > 0)
    rev = ttm(L["Total Revenue"]).where(lambda v: v > 0)
    parts = {}
    for line, s in stmt_of.items():
        if line in DROP:
            continue
        if s == "bs" or line == "End Cash Position":
            parts[line] = L[line] / ta
        elif s == "ins":
            parts[line] = ttm(L[line]) / rev
        else:
            parts[line] = ttm(L[line]) / ta
    F = pd.concat({k: v.stack(future_stack=True) for k, v in parts.items()}, axis=1)
    F.index.names = ["ticker", "quarter"]
    F = F.replace([np.inf, -np.inf], np.nan)
    F = F[ta.stack(future_stack=True).reindex(F.index).notna().to_numpy()]          # firm-quarters with assets
    by_q = F.groupby(level="quarter")
    F = F.clip(by_q.transform(lambda c: c.quantile(0.01)), by_q.transform(lambda c: c.quantile(0.99)))
    reported = F.notna().any(axis=1)
    F = scale(F, scaling)
    return F[reported & (F.abs().sum(axis=1) > 0)].reset_index()     # no fingerprint without any ratio


def _shift(q: str, h: int) -> str:
    n = int(q[:4]) * 4 + int(q[-1]) - 1 + h
    return f"{n // 4}Q{n % 4 + 1}"


def changes(F: pd.DataFrame, h: int) -> pd.DataFrame:
    """Δf = f(t + h) − f(t) for every firm-quarter t whose t + h has a fingerprint; same layout as F, indexed by t."""
    cur = F.set_index(["ticker", "quarter"])
    later = F.assign(quarter=F.quarter.map(lambda q: _shift(q, -h))).set_index(["ticker", "quarter"])
    both = cur.index.intersection(later.index)
    return (later.loc[both] - cur.loc[both]).reset_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", type=Path, required=True, help="Folder with {bs,ins,cf}-raw.parquet.")
    p.add_argument("--out", type=Path, default=None, help="Default: <panel>/targets/fingerprint.parquet.")
    p.add_argument("--horizons", type=int, nargs="*", default=[4], help="Horizons h of the change targets (none: skip).")
    p.add_argument("--scaling", choices=["zscore", "rank", "slogz"], default="zscore",
                   help="rank: normal scores of the ranks (fingerprint_rank.parquet); slogz: z-score of the signed log "
                        "(fingerprint_slogz.parquet); neither writes change targets.")
    p.add_argument("--firms", type=Path, default=None,
                   help="Parquet with a ticker column: the fingerprint of these firms only, scaled on them (no change "
                        "targets; main_similarity.py --sample balanced).")
    args = p.parse_args()
    L, stmt_of = load_raw(args.panel)
    if args.firms:
        firms = set(pd.read_parquet(args.firms).ticker)
        L = {k: v[v.index.isin(firms)] for k, v in L.items()}
    F = fingerprints(L, stmt_of, args.scaling)
    name = {"zscore": "fingerprint.parquet", "rank": "fingerprint_rank.parquet", "slogz": "fingerprint_slogz.parquet"}
    out = args.out or args.panel / "targets" / name[args.scaling]
    if args.scaling != "zscore" or args.firms:
        args.horizons = []
    out.parent.mkdir(parents=True, exist_ok=True)
    F.to_parquet(out)
    print(f"fingerprints: {F.ticker.nunique():,} firms, {len(F):,} firm-quarters, {F.shape[1] - 2} ratios, "
          f"{F.quarter.min()}–{F.quarter.max()} -> {out}")
    for h in args.horizons:
        C = changes(F, h)
        path = out.with_name(f"fingerprint_change_h{h}.parquet")
        C.to_parquet(path)
        print(f"changes over {h} quarters: {len(C):,} firm-quarters, {C.quarter.min()}–{C.quarter.max()} -> {path}")


if __name__ == "__main__":
    main()
