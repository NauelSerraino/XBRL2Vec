"""
The financial fingerprint of every firm-quarter: 32 ratios that describe the firm's statements without its size. It is
both the encoder's input (12 quarters of it) and its Pillar 1 target (the last quarter's).

    python src/preprocess/fingerprint.py --panel data/in/panels_2026-10-25/filled
    python src/preprocess/fingerprint.py --panel data/in/panels_2026-10-25/filled --firms <panel>/targets/firms_balanced.parquet

From the raw statements of a panel ({bs,ins,cf}-raw.parquet: amounts in dollars, one row per firm and line, one column
per quarter), for each firm-quarter:
  balance-sheet lines and the end cash position   line / total assets
  income-statement lines                          trailing-12-month line / trailing-12-month revenue
  cash-flow lines                                 trailing-12-month line / total assets
without the lines of DROP (the denominators, empty or mirrored lines): 17 + 10 + 5 = 32 ratios. Then, per quarter and
ratio, across the firms:
  1. clip at the 1st and 99th percentiles
  2. signed log, sign(x) · log(1 + |x|): compresses the extreme ratios (a firm with almost no revenue) before they set
     the standard deviation, and keeps the sign
  3. z-score (minus the mean, over the standard deviation)
A ratio that is not reported becomes 0, the mean of the quarter. Firm-quarters without positive total assets, or
without any ratio, get no fingerprint.

Output: <panel>/targets/fingerprint_slogz.parquet, columns ticker, quarter and one per ratio. With --firms (a parquet
with a ticker column): the fingerprint of those firms only, scaled on them, in fingerprint_slogz_balanced.parquet (the
model's sample; both runs are step 5 of src/main_download.py).
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
    """{line: ticker x quarter, in dollars} and {line: statement ("bs", "ins" or "cf")} from a panel's raw files."""
    wide = pd.concat([pd.read_parquet(panel / f"{s}-raw.parquet").assign(stmt=s) for s in STATEMENTS], ignore_index=True)
    quarters = sorted(c for c in wide.columns if len(str(c)) == 6 and str(c)[4] == "Q")
    lines = {k: g.set_index("ticker")[quarters].astype(float) for k, g in wide.groupby("features")}
    tickers = lines["Total Assets"].index
    return {k: v.reindex(tickers) for k, v in lines.items()}, wide.groupby("features").stmt.first().to_dict()


def scale(F: pd.DataFrame) -> pd.DataFrame:
    """Per quarter (index level "quarter") and ratio: the z-score of the signed log; a missing ratio becomes 0."""
    F = np.sign(F) * np.log1p(F.abs())
    by_q = F.groupby(level="quarter")
    return ((F - by_q.transform("mean")) / by_q.transform("std")).fillna(0.0)


def fingerprints(L: dict[str, pd.DataFrame], stmt_of: dict[str, str]) -> pd.DataFrame:
    """Long frame (ticker, quarter, one column per ratio) from the lines L of load_raw (see the module docstring)."""
    ttm = lambda df: df.T.rolling(4, min_periods=4).sum().T              # trailing 12 months: the sum of 4 quarters
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
    F = scale(F)
    return F[reported & (F.abs().sum(axis=1) > 0)].reset_index()     # no fingerprint without any ratio


def main() -> None:
    """Fingerprints of the panel (or of --firms only), written to <panel>/targets/."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", type=Path, required=True, help="Folder with {bs,ins,cf}-raw.parquet.")
    p.add_argument("--firms", type=Path, default=None,
                   help="Parquet with a ticker column: the fingerprint of these firms only, scaled on them.")
    args = p.parse_args()
    L, stmt_of = load_raw(args.panel)
    if args.firms:
        firms = set(pd.read_parquet(args.firms).ticker)
        L = {k: v[v.index.isin(firms)] for k, v in L.items()}
    F = fingerprints(L, stmt_of)
    out = args.panel / "targets" / ("fingerprint_slogz_balanced.parquet" if args.firms else "fingerprint_slogz.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    F.to_parquet(out)
    print(f"fingerprints: {F.ticker.nunique():,} firms, {len(F):,} firm-quarters, {F.shape[1] - 2} ratios, "
          f"{F.quarter.min()}–{F.quarter.max()} -> {out}")


if __name__ == "__main__":
    main()
