"""
preprocess.py — financial statement preprocessing for DL models.

Fixes vs the original notebook approach:
  1. No data leakage: scaler is fit on train split only.
  2. Scalers are persisted to disk for inference.
  3. NaNs from differencing are resolved before any scaler sees the data.
  4. Per-ticker normalization is available (default) to remove company-scale bias.
  5. Company-based train/test split (random, seed=42): each company's full time
     series goes entirely to train or entirely to test.
  6. Output files named _train / _test, not _pct.

Usage:
    python src/preprocess/preprocess.py --out_dir data/out/preprocess --norm_mode per_ticker
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_RATIO = 0.2


# ── helpers ───────────────────────────────────────────────────────────────────

def melt_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape from wide format (one row per ticker×feature, quarters as columns)
    to long format (one row per ticker×quarter, features as columns).
    """
    return (
        df.melt(id_vars=["ticker", "features"], var_name="quarter", value_name="value")
          .pivot(index=["ticker", "quarter"], columns="features", values="value")
          .reset_index()
    )


def create_diffs(df: pd.DataFrame, entity: str = "ticker", time: str = "quarter") -> pd.DataFrame:
    """
    Add quarter-over-quarter (_DIFF_Q) and year-over-year (_DIFF_Y) differences
    for every feature column, computed within each ticker.
    Introduces NaNs for the first 1 / 4 rows per ticker — handled by handle_nans().
    """
    df = df.copy().sort_values([entity, time])
    feature_cols = [c for c in df.columns if c not in [entity, time]]
    for col in feature_cols:
        df[f"{col}_DIFF_Q"] = df.groupby(entity)[col].diff(1)
        df[f"{col}_DIFF_Y"] = df.groupby(entity)[col].diff(4)
    return df


def handle_nans(df: pd.DataFrame, entity: str = "ticker", time: str = "quarter") -> pd.DataFrame:
    """
    Resolve NaNs introduced by create_diffs BEFORE any scaler sees the data.
    Strategy:
      - Forward-fill per ticker (covers internal gaps if any).
      - Drop rows that still have NaNs (the first ~4 rows of each ticker
        where no historical diff can be computed).
    """
    df = df.copy()
    non_key = [c for c in df.columns if c not in [entity, time]]
    df[non_key] = df.groupby(entity)[non_key].ffill()
    before = len(df)
    df.dropna(subset=non_key, inplace=True)
    df.reset_index(drop=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"    dropped {dropped} rows with unresolvable NaNs (expected: first ~4 rows per ticker)")
    return df


def get_complete_tickers(df: pd.DataFrame, entity: str = "ticker", time: str = "quarter") -> set:
    """
    Return the set of tickers that have no NaNs in any feature column across all quarters.
    """
    feature_cols = [c for c in df.columns if c not in [entity, time]]
    has_any_nan = df.groupby(entity)[feature_cols].apply(lambda g: g.isna().any().any())
    return set(has_any_nan[~has_any_nan].index)


def ticker_split(df: pd.DataFrame, test_ratio: float = 0.2, entity: str = "ticker"):
    """
    Randomly assign each company to train or test (seed=SEED).
    Every row for a given ticker goes entirely to one split,
    so each company's full time series is preserved intact.
    """
    tickers = df[entity].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(tickers)
    n_test = max(1, int(len(tickers) * test_ratio))
    test_tickers = set(tickers[:n_test])
    mask = df[entity].isin(test_tickers)
    return df[~mask].copy().reset_index(drop=True), df[mask].copy().reset_index(drop=True)


# ── normalization ──────────────────────────────────────────────────────────────

def normalize_global(
    train: pd.DataFrame,
    test: pd.DataFrame,
    entity: str = "ticker",
    time: str = "quarter",
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a single StandardScaler on train, transform both splits.
    Preserves cross-company scale information.
    """
    cols = [c for c in train.columns if c not in [entity, time]]
    scaler = StandardScaler()
    train = train.copy()
    test = test.copy()
    train[cols] = scaler.fit_transform(train[cols]).round(4)
    test[cols] = scaler.transform(test[cols]).round(4)
    return train, test, scaler


def normalize_per_ticker(
    train: pd.DataFrame,
    test: pd.DataFrame,
    entity: str = "ticker",
    time: str = "quarter",
) -> tuple[pd.DataFrame, pd.DataFrame, dict, StandardScaler]:
    """
    Fit one StandardScaler per ticker using only that ticker's train rows.
    Removes company-size bias so the model learns temporal dynamics,
    not scale differences between companies.

    For tickers that appear in test but not in train (unseen companies),
    a global fallback scaler (fit on all train data) is used.

    Returns: (train, test, per_ticker_scalers_dict, fallback_scaler)
    """
    cols = [c for c in train.columns if c not in [entity, time]]
    train = train.copy()
    test = test.copy()
    scalers: dict[str, StandardScaler] = {}

    # Fit and transform each ticker's train rows
    for ticker, grp in train.groupby(entity):
        sc = StandardScaler()
        train.loc[grp.index, cols] = sc.fit_transform(grp[cols]).round(4)
        scalers[ticker] = sc

    # Fallback for tickers unseen during training
    fallback = StandardScaler().fit(train[cols])

    # Transform test rows using the appropriate scaler
    for ticker, grp in test.groupby(entity):
        sc = scalers.get(ticker, fallback)
        test.loc[grp.index, cols] = sc.transform(grp[cols]).round(4)

    unseen = set(test[entity].unique()) - set(scalers)
    if unseen:
        print(f"    {len(unseen)} tickers in test not seen in train — used fallback scaler")

    return train, test, scalers, fallback


# ── main ──────────────────────────────────────────────────────────────────────

def preprocess_dataset(
    name: str,
    df: pd.DataFrame,
    out_dir: str,
    norm_mode: str,
) -> None:
    """
    `df` is already reshaped (melt_pivot) and filtered to the common ticker set.
    """
    print(f"\n── {name.upper()} ──")

    # Step 1: save clean baseline
    df.to_parquet(os.path.join(out_dir, f"{name}_raw_clean.parquet"), index=False)

    # Step 2: add diffs
    df = create_diffs(df)

    # Step 3: resolve NaNs from diffs BEFORE any scaler
    df = handle_nans(df)
    print(f"    rows: {len(df)}, tickers: {df['ticker'].nunique()}")

    # Step 4: company-based split (full time series per company, seed=SEED)
    train, test = ticker_split(df, test_ratio=TEST_RATIO)
    print(f"    train: {len(train)} rows, {train['ticker'].nunique()} tickers")
    print(f"    test:  {len(test)} rows,  {test['ticker'].nunique()} tickers")

    if len(train) == 0:
        raise ValueError("Train split is empty.")
    if len(test) == 0:
        raise ValueError("Test split is empty.")

    # Step 5: normalize (fit on train only)
    scaler_dir = os.path.join(out_dir, "scalers")
    os.makedirs(scaler_dir, exist_ok=True)

    if norm_mode == "global":
        train, test, scaler = normalize_global(train, test)
        joblib.dump(scaler, os.path.join(scaler_dir, f"{name}_{norm_mode}_scaler.joblib"))
        print(f"    saved global scaler → scalers/{name}_{norm_mode}_scaler.joblib")

    else:  # per_ticker
        train, test, scalers, fallback = normalize_per_ticker(train, test)
        joblib.dump(scalers,  os.path.join(scaler_dir, f"{name}_{norm_mode}_scalers.joblib"))
        joblib.dump(fallback, os.path.join(scaler_dir, f"{name}_{norm_mode}_fallback_scaler.joblib"))
        print(f"    saved {len(scalers)} per-ticker scalers → scalers/{name}_{norm_mode}_scalers.joblib")
        print(f"    saved fallback scaler → scalers/{name}_{norm_mode}_fallback_scaler.joblib")

    # Step 6: save splits
    train.to_parquet(os.path.join(out_dir, f"{name}_{norm_mode}_train.parquet"), index=False)
    test.to_parquet(os.path.join(out_dir,  f"{name}_{norm_mode}_test.parquet"),  index=False)
    print(f"    saved {name}_{norm_mode}_train.parquet and {name}_{norm_mode}_test.parquet")


def main(out_dir: str, norm_mode: str) -> None:
    print(f"Settings: test_ratio={TEST_RATIO}, seed={SEED}, norm_mode={norm_mode}")

    raw_paths = {
        "bs":  os.path.join(out_dir, "bs-raw.parquet"),
        "ins": os.path.join(out_dir, "ins-raw.parquet"),
        "cf":  os.path.join(out_dir, "cf-raw.parquet"),
    }

    # Step 1: reshape all available datasets and compute the common complete-ticker set
    dfs: dict[str, pd.DataFrame] = {}
    complete_per_dataset: dict[str, set] = {}
    for name, raw_path in raw_paths.items():
        if not os.path.exists(raw_path):
            print(f"WARNING: {raw_path} not found — skipping {name}")
            continue
        df = melt_pivot(pd.read_parquet(raw_path))
        dfs[name] = df
        complete_per_dataset[name] = get_complete_tickers(df)

    if not dfs:
        print("No datasets found. Done.")
        return

    common_tickers = set.intersection(*complete_per_dataset.values())
    total_per_ds = {n: len(s) for n, s in complete_per_dataset.items()}
    print(
        f"\nCommon complete tickers across all datasets: {len(common_tickers)} "
        f"(per-dataset complete: {total_per_ds})"
    )

    # Step 2: filter each dataset to the common ticker set, then preprocess
    for name, df in dfs.items():
        before = df["ticker"].nunique()
        df = df[df["ticker"].isin(common_tickers)].copy().reset_index(drop=True)
        dropped = before - df["ticker"].nunique()
        if dropped:
            print(f"  {name}: dropped {dropped} tickers not in common set")
        preprocess_dataset(name, df, out_dir, norm_mode)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess financial statements for DL models.")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(__file__), "data", "out", "bulk"),
        help="Directory containing raw parquets and where outputs are written.",
    )
    parser.add_argument(
        "--norm_mode",
        choices=["global", "per_ticker"],
        default="per_ticker",
        help=(
            "per_ticker: fit one scaler per company (removes company-size bias, recommended). "
            "global: one scaler across all companies (preserves cross-company scale)."
        ),
    )
    args = parser.parse_args()
    main(args.out_dir, args.norm_mode)
