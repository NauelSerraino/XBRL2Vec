"""
Step 4 of src/main_download.py: the panel's statements as one row per firm-quarter, with each line's changes, scaled,
and split into two halves of firms.

    python src/preprocess/preprocess.py --out_dir data/in/panels_2026-10-25/filled

For each statement (bs, ins, cf) of the panel folder, from {bs,ins,cf}-raw.parquet (one row per firm and line, one
column per quarter, in dollars):
  1. reshape to one row per firm-quarter, one column per line (melt_pivot)
  2. add each line's change over 1 quarter (<line>_DIFF_Q) and over 4 quarters (<line>_DIFF_Y) (create_diffs)
  3. split the firms at random (seed 42) into "train" (80 %) and "test" (20 %); a firm's whole history goes to one side
  4. z-score every column with the mean and standard deviation of the "train" firms (StandardScaler), 4 decimals
Missing values stay missing. Only firms present in all three statements are kept.

Output, in --out_dir: {bs,ins,cf}_global_{train,test}.parquet.

What the model uses of this: only which firm-quarters have complete windows in these amounts and their yearly changes
(services/data.universe). The model's own split of the firms (services/data.balanced_firms) replaces this one, and
its inputs are the fingerprint (src/preprocess/fingerprint.py), not these scaled amounts.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_RATIO = 0.2


def melt_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Wide (one row per ticker and line, one column per quarter) -> long (one row per ticker and quarter, one column
    per line), sorted by ticker and quarter."""
    return (df.melt(id_vars=["ticker", "features"], var_name="quarter", value_name="value")
              .pivot(index=["ticker", "quarter"], columns="features", values="value")
              .reset_index())


def create_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Add <line>_DIFF_Q (change over 1 quarter) and <line>_DIFF_Y (over 4) for every line, within each ticker. The
    first 1 / 4 quarters of a firm get NaN."""
    df = df.copy().sort_values(["ticker", "quarter"])
    for col in [c for c in df.columns if c not in ("ticker", "quarter")]:
        df[f"{col}_DIFF_Q"] = df.groupby("ticker")[col].diff(1)
        df[f"{col}_DIFF_Y"] = df.groupby("ticker")[col].diff(4)
    return df


def ticker_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test): every firm goes, with all its rows, to one side; TEST_RATIO of the firms to "test" (seed SEED)."""
    tickers = df.ticker.unique()
    np.random.default_rng(SEED).shuffle(tickers)
    test_tickers = set(tickers[:max(1, int(len(tickers) * TEST_RATIO))])
    mask = df.ticker.isin(test_tickers)
    return df[~mask].copy().reset_index(drop=True), df[mask].copy().reset_index(drop=True)


def normalize_global(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """z-score of every column with the mean and standard deviation of train (NaN ignored), applied to both."""
    cols = [c for c in train.columns if c not in ("ticker", "quarter")]
    scaler = StandardScaler()
    train, test = train.copy(), test.copy()
    train[cols] = scaler.fit_transform(train[cols]).round(4)
    test[cols] = scaler.transform(test[cols]).round(4)
    return train, test


def main(out_dir: str) -> None:
    """Steps 1-4 of the module docstring for each statement of the panel folder out_dir."""
    dfs = {s: melt_pivot(pd.read_parquet(os.path.join(out_dir, f"{s}-raw.parquet"))) for s in ("bs", "ins", "cf")}
    common = set.intersection(*(set(df.ticker) for df in dfs.values()))
    print(f"firms in all three statements: {len(common):,}")
    for s, df in dfs.items():
        df = create_diffs(df[df.ticker.isin(common)].reset_index(drop=True))
        train, test = normalize_global(*ticker_split(df))
        train.to_parquet(os.path.join(out_dir, f"{s}_global_train.parquet"), index=False)
        test.to_parquet(os.path.join(out_dir, f"{s}_global_test.parquet"), index=False)
        print(f"{s}: train {train.ticker.nunique():,} firms ({len(train):,} rows), test {test.ticker.nunique():,} firms "
              f"({len(test):,} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True, help="Panel folder: reads {bs,ins,cf}-raw.parquet, writes the outputs.")
    main(p.parse_args().out_dir)
