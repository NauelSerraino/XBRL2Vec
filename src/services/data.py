"""
Data loading, filtering, alignment and transformation pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from services.config import SEQ_LEN, TICKERS as _EXCLUDED_TICKERS


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ColumnFilter(str, Enum):
    DIFF_Y = "DIFF_Y"        # year-on-year changes (the forecasting pipeline)
    LEVELS = "LEVELS"        # the values themselves
    BOTH = "BOTH"            # values and year-on-year changes


class ModelType(str, Enum):
    CONTEXTUAL = "contextual"
    BLIND = "blind"


class SaliencyMode(str, Enum):
    LATENT = "latent"
    RECONSTRUCTION = "reconstruction"


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"


# ---------------------------------------------------------------------------
# Config / Hyperparameters
# ---------------------------------------------------------------------------

class TrainConfig(BaseModel):
    latent_factors: list[float] = Field(default=[0.5, 1.0, 2.0, 3.0])
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42
    t_in: int = 12
    t_out: int = 4
    norm_mode: str = "per_ticker"
    macro_weight_mode: str = "corr"  # "corr" | "ridge" | "xgboost"

    @classmethod
    def from_args(cls, args) -> "TrainConfig":
        return cls(
            latent_factors=args.latent_factors,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            t_in=args.t_in,
            t_out=args.t_out,
            norm_mode=args.norm_mode,
            macro_weight_mode=args.macro_weight_mode,
        )


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class AlignedDataset:
    X_fin: torch.Tensor          # [N, T, F]
    X_macro: torch.Tensor        # [N, T, M]
    Y_fin: torch.Tensor          # [N, T, F]  (reconstruction target = X_fin)
    meta_df: pd.DataFrame        # [N] rows: ticker, end_quarter
    fin_cols: list[str]
    macro_cols: list[str]

    @property
    def fin_dim(self) -> int:
        return self.X_fin.shape[-1]

    @property
    def macro_dim(self) -> int:
        return self.X_macro.shape[-1]

    @property
    def n_samples(self) -> int:
        return self.X_fin.shape[0]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

_QUARTERS_TO_DELETE = [f"201{j}Q{i}" for j in range(1, 3) for i in range(1, 5)]

_EXCLUDED_COLS = {
    "Other Non-Current Assets_DIFF_Y",
    "Other Non-Current Liabilities_DIFF_Y",
    "Other Non-Current Assets",
    "Other Non-Current Liabilities",
}


def _keep_filtered_cols(df: pd.DataFrame, cond: ColumnFilter) -> pd.DataFrame:
    """Keep the feature columns of the kind `cond` (plus id columns): values (no _DIFF_ suffix), _DIFF_Y, or both."""
    id_cols = [c for c in ["ticker", "quarter"] if c in df.columns]
    is_level = lambda c: "_DIFF_" not in c
    keep = {ColumnFilter.DIFF_Y: lambda c: "DIFF_Y" in c,
            ColumnFilter.LEVELS: is_level,
            ColumnFilter.BOTH: lambda c: is_level(c) or "DIFF_Y" in c}[ColumnFilter(cond)]
    feature_cols = [
        c for c in df.columns
        if c not in id_cols and keep(c) and c not in _EXCLUDED_COLS and c.split("_DIFF_")[0] not in _EXCLUDED_COLS
    ]
    return df[id_cols + feature_cols]


def zero_unreported(df: pd.DataFrame) -> pd.DataFrame:
    """
    In a quarter where the firm reports this statement (at least one value), a missing line is a line it does not
    report: 0, as in the complete panel and finqual. Quarters with no value at all (not reporting yet / any more)
    stay missing. The filled panel keeps such lines as NaN; the complete panel has none.
    """
    cols = [c for c in df.columns if c not in ("ticker", "quarter")]
    reported = df[cols].notna().any(axis=1)
    df.loc[reported, cols] = df.loc[reported, cols].fillna(0.0)
    return df


def filter_columns(
    bs_df: pd.DataFrame,
    is_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    cond: ColumnFilter = ColumnFilter.DIFF_Y,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _keep_filtered_cols(bs_df, cond),
        _keep_filtered_cols(is_df, cond),
        _keep_filtered_cols(cf_df, cond),
        _keep_filtered_cols(macro_df, ColumnFilter.DIFF_Y),      # macro: always year-on-year changes
    )


def load_raw_data(
    preprocess_dir: Path,
    meta_dir: Path,
    norm_mode: str = "per_ticker",
) -> tuple[pd.DataFrame, ...]:
    """Load and pre-filter all raw parquet files. Returns (bs, is, cf, macro, metadata).

    Args:
        preprocess_dir: directory produced by preprocess.py (contains train/test parquets).
        meta_dir: directory containing exog.parquet and metadata.parquet.
        norm_mode: normalization mode used in preprocess.py (e.g. 'per_ticker', 'global').
    """
    bs_df    = pd.read_parquet(preprocess_dir / f"bs_{norm_mode}_train.parquet")
    is_df    = pd.read_parquet(preprocess_dir / f"ins_{norm_mode}_train.parquet")
    cf_df    = pd.read_parquet(preprocess_dir / f"cf_{norm_mode}_train.parquet")
    macro_df = pd.read_parquet(meta_dir / "exog.parquet").rename(columns={"observation_date": "quarter"})
    metadata = pd.read_parquet(meta_dir / "metadata.parquet")[["ticker", "sector"]].drop_duplicates()

    for df in [bs_df, is_df, cf_df, macro_df]:
        mask = df["quarter"].isin(_QUARTERS_TO_DELETE)
        df.drop(df[mask].index, inplace=True)

    for df in [bs_df, is_df, cf_df]:
        df.drop(df[df["ticker"].isin(_EXCLUDED_TICKERS)].index, inplace=True)
        zero_unreported(df)

    return bs_df, is_df, cf_df, macro_df, metadata


def load_test_data(
    preprocess_dir: Path,
    macro_df: pd.DataFrame,
    norm_mode: str = "per_ticker",
    cond: ColumnFilter = ColumnFilter.DIFF_Y,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load test parquet files and apply same quarter filter."""
    bs  = pd.read_parquet(preprocess_dir / f"bs_{norm_mode}_test.parquet")
    is_ = pd.read_parquet(preprocess_dir / f"ins_{norm_mode}_test.parquet")
    cf  = pd.read_parquet(preprocess_dir / f"cf_{norm_mode}_test.parquet")

    for df in [bs, is_, cf]:
        df.drop(df[df["quarter"].isin(_QUARTERS_TO_DELETE)].index, inplace=True)

    for df in [bs, is_, cf]:
        df.drop(df[df["ticker"].isin(_EXCLUDED_TICKERS)].index, inplace=True)
        zero_unreported(df)

    bs, is_, cf, _ = filter_columns(bs, is_, cf, macro_df, cond)
    return bs, is_, cf


def create_aligned_dataset(
    bs_df: pd.DataFrame,
    is_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
) -> AlignedDataset:
    """Merge all statements, build sliding-window tensors."""
    df = (
        bs_df
        .merge(is_df, on=["ticker", "quarter"])
        .merge(cf_df, on=["ticker", "quarter"])
        .merge(macro_df, on="quarter")
        .sort_values(["ticker", "quarter"])
    )

    macro_cols = [c for c in macro_df.columns if c != "quarter"]
    fin_cols   = [c for c in df.columns if c not in ["ticker", "quarter"] + macro_cols]

    X_fin_list, X_macro_list, Y_fin_list, meta = [], [], [], []

    for ticker, group in df.groupby("ticker"):
        # windows only inside runs of consecutive quarters with every value present (the filled panel keeps NaN
        # where a firm does not report yet / any more; the complete panel has none)
        q_num = group["quarter"].str[:4].astype(int) * 4 + group["quarter"].str[-1].astype(int)
        valid = np.isfinite(group[fin_cols + macro_cols].to_numpy(dtype=float)).all(axis=1)
        run_id = np.cumsum(~valid | (q_num.diff().fillna(1).to_numpy() != 1))
        for _, run in group[valid].groupby(run_id[valid]):
            if len(run) < seq_len:
                continue

            f_vals = run[fin_cols].values
            m_vals = run[macro_cols].values
            q_vals = run["quarter"].values

            x_fin  = np.lib.stride_tricks.sliding_window_view(f_vals, (seq_len, len(fin_cols))).squeeze(1)
            x_mac  = np.lib.stride_tricks.sliding_window_view(m_vals, (seq_len, len(macro_cols))).squeeze(1)

            X_fin_list.append(x_fin)
            X_macro_list.append(x_mac)
            Y_fin_list.append(x_fin)  # AE: target == input

            meta.extend([
                {"ticker": ticker, "end_quarter": q_vals[i + seq_len - 1]}
                for i in range(len(run) - seq_len + 1)
            ])

    return AlignedDataset(
        X_fin   = torch.tensor(np.concatenate(X_fin_list),   dtype=torch.float32),
        X_macro = torch.tensor(np.concatenate(X_macro_list), dtype=torch.float32),
        Y_fin   = torch.tensor(np.concatenate(Y_fin_list),   dtype=torch.float32),
        meta_df = pd.DataFrame(meta),
        fin_cols   = fin_cols,
        macro_cols = macro_cols,
    )