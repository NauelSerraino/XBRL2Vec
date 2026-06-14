import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess.preprocess import (
    melt_pivot,
    create_diffs,
    handle_nans,
    get_complete_tickers,
    ticker_split,
    normalize_global,
    normalize_per_ticker,
)


def _make_long_df(tickers=("A", "B", "C"), n_quarters=8, n_features=3):
    """Build a minimal long-format DataFrame (ticker, quarter, feat_*)."""
    rows = []
    for t in tickers:
        for q in range(n_quarters):
            row = {"ticker": t, "quarter": f"202{q // 4}Q{q % 4 + 1}"}
            for f in range(n_features):
                row[f"feat_{f}"] = float(q + f + hash(t) % 10)
            rows.append(row)
    return pd.DataFrame(rows)


def _make_wide_df(tickers=("A", "B"), quarters=("2020Q1", "2020Q2", "2020Q3"), features=("rev", "cost")):
    """Build a wide-format DataFrame suitable for melt_pivot."""
    rows = []
    for t in tickers:
        for feat in features:
            row = {"ticker": t, "features": feat}
            for q in quarters:
                row[q] = float(hash((t, feat, q)) % 100)
            rows.append(row)
    return pd.DataFrame(rows)


class TestMeltPivot(unittest.TestCase):
    def setUp(self):
        self.quarters = ("2020Q1", "2020Q2", "2020Q3")
        self.features = ("Revenue", "COGS")
        self.tickers  = ("AAPL", "MSFT")
        self.wide_df  = _make_wide_df(self.tickers, self.quarters, self.features)

    def test_output_shape(self):
        result = melt_pivot(self.wide_df)
        expected_rows = len(self.tickers) * len(self.quarters)
        self.assertEqual(len(result), expected_rows)

    def test_columns_present(self):
        result = melt_pivot(self.wide_df)
        self.assertIn("ticker", result.columns)
        self.assertIn("quarter", result.columns)
        for feat in self.features:
            self.assertIn(feat, result.columns)

    def test_no_duplicate_ticker_quarter(self):
        result = melt_pivot(self.wide_df)
        dupes = result.duplicated(subset=["ticker", "quarter"]).sum()
        self.assertEqual(dupes, 0)


class TestCreateDiffs(unittest.TestCase):
    def setUp(self):
        self.df = _make_long_df(tickers=("X", "Y"), n_quarters=8, n_features=2)

    def test_diff_columns_created(self):
        result = create_diffs(self.df)
        for col in ["feat_0", "feat_1"]:
            self.assertIn(f"{col}_DIFF_Q", result.columns)
            self.assertIn(f"{col}_DIFF_Y", result.columns)

    def test_diff_q_nans_only_first_row_per_ticker(self):
        result = create_diffs(self.df)
        for ticker, grp in result.groupby("ticker"):
            nan_mask = grp["feat_0_DIFF_Q"].isna()
            self.assertEqual(nan_mask.sum(), 1, f"Expected 1 NaN for DIFF_Q in ticker {ticker}")

    def test_diff_y_nans_only_first_four_rows_per_ticker(self):
        result = create_diffs(self.df)
        for ticker, grp in result.groupby("ticker"):
            nan_mask = grp["feat_0_DIFF_Y"].isna()
            self.assertEqual(nan_mask.sum(), 4, f"Expected 4 NaNs for DIFF_Y in ticker {ticker}")

    def test_original_columns_preserved(self):
        result = create_diffs(self.df)
        self.assertIn("feat_0", result.columns)
        self.assertIn("feat_1", result.columns)


class TestHandleNans(unittest.TestCase):
    def setUp(self):
        self.df = create_diffs(_make_long_df(tickers=("A", "B"), n_quarters=8))

    def test_no_nans_after_handling(self):
        result = handle_nans(self.df)
        self.assertFalse(result.isnull().any().any())

    def test_rows_dropped(self):
        result = handle_nans(self.df)
        self.assertLess(len(result), len(self.df))

    def test_ticker_column_intact(self):
        result = handle_nans(self.df)
        self.assertIn("ticker", result.columns)
        self.assertEqual(set(result["ticker"].unique()), {"A", "B"})


class TestGetCompleteTickers(unittest.TestCase):
    def test_returns_only_complete_tickers(self):
        df = _make_long_df(tickers=("good1", "good2", "bad"), n_quarters=6)
        df.loc[df["ticker"] == "bad", "feat_0"] = np.nan
        result = get_complete_tickers(df)
        self.assertIn("good1", result)
        self.assertIn("good2", result)
        self.assertNotIn("bad", result)

    def test_all_complete_when_no_nans(self):
        df = _make_long_df(tickers=("A", "B", "C"), n_quarters=6)
        result = get_complete_tickers(df)
        self.assertEqual(result, {"A", "B", "C"})


class TestTickerSplit(unittest.TestCase):
    def setUp(self):
        self.df = _make_long_df(tickers=tuple(f"T{i}" for i in range(20)), n_quarters=8)

    def test_no_ticker_overlap(self):
        train, test = ticker_split(self.df, test_ratio=0.2)
        train_tickers = set(train["ticker"].unique())
        test_tickers  = set(test["ticker"].unique())
        self.assertEqual(len(train_tickers & test_tickers), 0)

    def test_all_tickers_covered(self):
        train, test = ticker_split(self.df, test_ratio=0.2)
        all_tickers = set(self.df["ticker"].unique())
        self.assertEqual(set(train["ticker"].unique()) | set(test["ticker"].unique()), all_tickers)

    def test_test_ratio_approximate(self):
        train, test = ticker_split(self.df, test_ratio=0.2)
        n_total = self.df["ticker"].nunique()
        n_test  = test["ticker"].nunique()
        self.assertGreaterEqual(n_test, 1)
        self.assertLessEqual(n_test / n_total, 0.35)

    def test_deterministic(self):
        train1, test1 = ticker_split(self.df, test_ratio=0.2)
        train2, test2 = ticker_split(self.df, test_ratio=0.2)
        self.assertEqual(set(test1["ticker"].unique()), set(test2["ticker"].unique()))


class TestNormalizeGlobal(unittest.TestCase):
    def setUp(self):
        df = _make_long_df(tickers=tuple(f"T{i}" for i in range(10)), n_quarters=8)
        self.train, self.test = ticker_split(df)

    def test_train_approx_zero_mean(self):
        feat_cols = [c for c in self.train.columns if c not in ["ticker", "quarter"]]
        train_norm, _, _ = normalize_global(self.train, self.test)
        means = train_norm[feat_cols].mean()
        self.assertTrue((means.abs() < 1e-4).all(), f"Means not near zero: {means}")

    def test_test_transformed_with_train_scaler(self):
        feat_cols = [c for c in self.train.columns if c not in ["ticker", "quarter"]]
        _, test_norm, scaler = normalize_global(self.train, self.test)
        self.assertEqual(len(test_norm), len(self.test))
        self.assertFalse(test_norm[feat_cols].isnull().any().any())

    def test_original_not_mutated(self):
        original_mean = self.train["feat_0"].mean()
        normalize_global(self.train, self.test)
        self.assertAlmostEqual(self.train["feat_0"].mean(), original_mean)


class TestNormalizePerTicker(unittest.TestCase):
    def setUp(self):
        df = _make_long_df(tickers=tuple(f"T{i}" for i in range(10)), n_quarters=8)
        self.train, self.test = ticker_split(df)

    def test_per_ticker_mean_near_zero(self):
        feat_cols = [c for c in self.train.columns if c not in ["ticker", "quarter"]]
        train_norm, _, _, _ = normalize_per_ticker(self.train, self.test)
        for ticker, grp in train_norm.groupby("ticker"):
            means = grp[feat_cols].mean()
            self.assertTrue((means.abs() < 1e-4).all(), f"Ticker {ticker} mean not near zero")

    def test_scalers_dict_has_train_tickers(self):
        _, _, scalers, _ = normalize_per_ticker(self.train, self.test)
        for t in self.train["ticker"].unique():
            self.assertIn(t, scalers)

    def test_unseen_test_tickers_use_fallback(self):
        extra_rows = [{"ticker": "UNSEEN", "quarter": "2020Q1", "feat_0": 1.0, "feat_1": 2.0, "feat_2": 3.0}]
        test_with_unseen = pd.concat([self.test, pd.DataFrame(extra_rows)], ignore_index=True)
        _, test_norm, _, _ = normalize_per_ticker(self.train, test_with_unseen)
        self.assertFalse(test_norm[[c for c in test_norm.columns if c not in ["ticker", "quarter"]]].isnull().any().any())


if __name__ == "__main__":
    unittest.main()
