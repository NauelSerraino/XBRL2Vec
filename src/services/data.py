"""
Loading the preprocessed statements of a panel and cutting them into windows of consecutive quarters.

A panel folder (data/in/panels_<date>/filled) holds, from src/preprocess/preprocess.py, six files
{bs,ins,cf}_global_{train,test}.parquet: one row per firm-quarter, columns ticker, quarter and, for every statement line,
its scaled amount ("Net PPE") and its changes ("Net PPE_DIFF_Q", "Net PPE_DIFF_Y"). "train" / "test" is that script's
split of the companies; the model uses its own split (main_similarity.balanced_firms), so here the two files are only
two halves of the same data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from services.config import EXCLUDED_TICKERS

KEYS = ["ticker", "quarter"]
#: quarters left out: the data used start in 2013Q1 (the raw panels start in 2008Q4)
EXCLUDED_QUARTERS = [f"{y}Q{q}" for y in (2011, 2012) for q in range(1, 5)]
#: lines left out (almost always empty)
EXCLUDED_LINES = {"Other Non-Current Assets", "Other Non-Current Liabilities"}


def zero_unreported(df: pd.DataFrame) -> pd.DataFrame:
    """
    In a quarter where the firm reports this statement (at least one value), a missing line is a line it does not
    report: 0. Quarters with no value at all (the firm is not reporting yet, or any more) stay missing.
    """
    cols = [c for c in df.columns if c not in KEYS]
    reported = df[cols].notna().any(axis=1)
    df.loc[reported, cols] = df.loc[reported, cols].fillna(0.0)
    return df


def load_statements(panel: Path, split: str) -> dict[str, pd.DataFrame]:
    """{"bs", "ins", "cf"} -> the firm-quarters of one half of the companies (split "train" or "test"): ticker, quarter,
    and each line's amount and year-on-year change (the quarterly changes and the excluded lines are dropped)."""
    out = {}
    for s in ("bs", "ins", "cf"):
        df = pd.read_parquet(panel / f"{s}_global_{split}.parquet")
        df = df[~df.quarter.isin(EXCLUDED_QUARTERS) & ~df.ticker.isin(EXCLUDED_TICKERS)].copy()
        zero_unreported(df)
        keep = [c for c in df.columns if c not in KEYS and ("_DIFF_" not in c or "DIFF_Y" in c)
                and c.split("_DIFF_")[0] not in EXCLUDED_LINES]
        out[s] = df[KEYS + keep]
    return out


def sliding_windows(df: pd.DataFrame, t_in: int) -> tuple[torch.Tensor, pd.DataFrame, list[str]]:
    """
    Every window of t_in consecutive quarters of every firm in which no value is missing.

    df: one row per firm-quarter (ticker, quarter, then the feature columns).
    Returns X [N, t_in, F] (float32), meta [N] (ticker, quarter = the window's last quarter) and the F feature names.
    A firm with quarters q1..qn gives n - t_in + 1 windows; a missing quarter or value cuts its history into runs, and
    windows never span two runs.
    """
    df = df.sort_values(KEYS)
    cols = [c for c in df.columns if c not in KEYS]
    X, meta = [], []
    for ticker, g in df.groupby("ticker"):
        q_num = g.quarter.str[:4].astype(int) * 4 + g.quarter.str[-1].astype(int)
        valid = np.isfinite(g[cols].to_numpy(dtype=float)).all(axis=1)
        run_id = np.cumsum(~valid | (q_num.diff().fillna(1).to_numpy() != 1))     # a new run after every gap
        for _, run in g[valid].groupby(run_id[valid]):
            if len(run) < t_in:
                continue
            X.append(np.lib.stride_tricks.sliding_window_view(run[cols].values, (t_in, len(cols))).squeeze(1))
            meta += [{"ticker": ticker, "quarter": q} for q in run.quarter.values[t_in - 1:]]
    return torch.tensor(np.concatenate(X), dtype=torch.float32), pd.DataFrame(meta), cols
