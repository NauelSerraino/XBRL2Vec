"""
Loading the preprocessed statements of a panel, cutting them into windows of consecutive quarters, and the model's
sample (universe, balanced_firms; written once by src/main_download.py, step 5).

A panel folder (data/in/panels_<date>/filled) holds, from src/preprocess/preprocess.py, six files
{bs,ins,cf}_global_{train,test}.parquet: one row per firm-quarter, columns ticker, quarter and, for every statement line,
its scaled amount ("Net PPE") and its changes ("Net PPE_DIFF_Q", "Net PPE_DIFF_Y"). "train" / "test" is that script's
split of the companies; the model uses its own split (balanced_firms), so here the two files are only two halves of
the same data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from services.config import EXCLUDED_TICKERS
from services.similarity import attach_fingerprints

KEYS = ["ticker", "quarter"]
T_IN = 12                                # quarters in a window
LAST_QUARTER = "2025Q4"                  # whole years: the 2026 quarters are left out
UNSEEN_SHARE, SPLIT_SEED = 0.15, 0       # the unseen firms: their share of the balanced firms, and the seed of the draw
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


def windows(panel: Path, split: str, t_in: int, fp: pd.DataFrame | None = None):
    """sliding_windows of one half of the companies (split "train" / "test" of preprocess.py). With fp: windows of the
    fingerprint, on the firm-quarters of the balance sheet file. Without: windows of the amounts and their yearly
    changes of the three statements (used only to define the universe)."""
    S = load_statements(panel, split)
    df = S["bs"].merge(S["ins"], on=KEYS).merge(S["cf"], on=KEYS) if fp is None else fp.merge(S["bs"][KEYS])
    return sliding_windows(df, t_in)


def universe(panel: Path, t_in: int) -> pd.MultiIndex:
    """
    The (ticker, quarter) pairs that can be used: those that end a complete window of t_in quarters in the amounts and
    their yearly changes of all three statements, and in the fingerprint of every firm (targets/fingerprint_slogz.parquet),
    up to LAST_QUARTER.
    """
    fp = pd.read_parquet(panel / "targets" / "fingerprint_slogz.parquet")
    keys = None
    for use_fp in (False, True):
        parts = []
        for split in ("train", "test"):
            X, meta, cols = windows(panel, split, t_in, fp if use_fp else None)
            parts.append(attach_fingerprints(X, meta, cols, fp).meta[KEYS])       # the last quarter has a fingerprint
        k = pd.concat(parts)
        keys = k if keys is None else keys.merge(k)
    return pd.MultiIndex.from_frame(keys[keys.quarter.le(LAST_QUARTER)])


def balanced_firms(keep: pd.MultiIndex) -> pd.DataFrame:
    """ticker, split: the firms with a window at every quarter of the universe `keep`, a random UNSEEN_SHARE of them
    "unseen" (the others "train")."""
    k = keep.to_frame(index=False)
    n = k.groupby("ticker").quarter.nunique()
    firms = np.sort(n.index[n.eq(k.quarter.nunique())].to_numpy())
    unseen = np.random.default_rng(SPLIT_SEED).permutation(firms)[:round(UNSEEN_SHARE * len(firms))]
    return pd.DataFrame({"ticker": firms, "split": np.where(np.isin(firms, unseen), "unseen", "train")})
