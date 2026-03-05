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

from services.config import SEQ_LEN


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ColumnFilter(str, Enum):
    DIFF_Y = "DIFF_Y"


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

    @classmethod
    def from_args(cls, args) -> "TrainConfig":
        return cls(
            latent_factors=args.latent_factors,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
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
}


def _keep_filtered_cols(df: pd.DataFrame, cond: ColumnFilter) -> pd.DataFrame:
    """Keep only columns whose name contains `cond` (plus id columns)."""
    id_cols = [c for c in ["ticker", "quarter"] if c in df.columns]
    feature_cols = [
        c for c in df.columns
        if cond.value in c and c not in _EXCLUDED_COLS
    ]
    return df[id_cols + feature_cols]


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
        _keep_filtered_cols(macro_df, cond),
    )


def load_raw_data(in_dir: Path) -> tuple[pd.DataFrame, ...]:
    """Load and pre-filter all raw parquet files. Returns (bs, is, cf, macro, metadata)."""
    bs_df    = pd.read_parquet(in_dir / "bs_pct_train.parquet")
    is_df    = pd.read_parquet(in_dir / "ins_pct_train.parquet")
    cf_df    = pd.read_parquet(in_dir / "cf_pct_train.parquet")
    macro_df = pd.read_parquet(in_dir / "exog.parquet").rename(columns={"observation_date": "quarter"})
    metadata = pd.read_parquet(in_dir / "metadata.parquet")[["ticker", "sector"]].drop_duplicates()

    for df in [bs_df, is_df, cf_df, macro_df]:
        mask = df["quarter"].isin(_QUARTERS_TO_DELETE)
        df.drop(df[mask].index, inplace=True)

    return bs_df, is_df, cf_df, macro_df, metadata


def load_test_data(in_dir: Path, macro_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load test parquet files and apply same quarter filter."""
    bs = pd.read_parquet(in_dir / "bs_pct_test.parquet")
    is_ = pd.read_parquet(in_dir / "ins_pct_test.parquet")
    cf = pd.read_parquet(in_dir / "cf_pct_test.parquet")

    for df in [bs, is_, cf]:
        df.drop(df[df["quarter"].isin(_QUARTERS_TO_DELETE)].index, inplace=True)

    bs, is_, cf, _ = filter_columns(bs, is_, cf, macro_df)
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
        if len(group) < seq_len:
            continue

        f_vals = group[fin_cols].values
        m_vals = group[macro_cols].values
        q_vals = group["quarter"].values

        x_fin  = np.lib.stride_tricks.sliding_window_view(f_vals, (seq_len, len(fin_cols))).squeeze(1)
        x_mac  = np.lib.stride_tricks.sliding_window_view(m_vals, (seq_len, len(macro_cols))).squeeze(1)

        X_fin_list.append(x_fin)
        X_macro_list.append(x_mac)
        Y_fin_list.append(x_fin)  # AE: target == input

        meta.extend([
            {"ticker": ticker, "end_quarter": q_vals[i + seq_len - 1]}
            for i in range(len(group) - seq_len + 1)
        ])

    return AlignedDataset(
        X_fin   = torch.tensor(np.concatenate(X_fin_list),   dtype=torch.float32),
        X_macro = torch.tensor(np.concatenate(X_macro_list), dtype=torch.float32),
        Y_fin   = torch.tensor(np.concatenate(Y_fin_list),   dtype=torch.float32),
        meta_df = pd.DataFrame(meta),
        fin_cols   = fin_cols,
        macro_cols = macro_cols,
    )