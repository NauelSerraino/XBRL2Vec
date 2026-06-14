import sys
import os
import unittest
import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.data import (
    AlignedDataset,
    ColumnFilter,
    TrainConfig,
    filter_columns,
    create_aligned_dataset,
)


def _make_aligned_dataset(N=10, T=12, F=5, M=3):
    return AlignedDataset(
        X_fin    = torch.randn(N, T, F),
        X_macro  = torch.randn(N, T, M),
        Y_fin    = torch.randn(N, T, F),
        meta_df  = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(N)],
            "end_quarter": ["2021Q1"] * N,
        }),
        fin_cols   = [f"fin_{i}" for i in range(F)],
        macro_cols = [f"mac_{i}" for i in range(M)],
    )


def _make_statement_df(tickers, quarters, prefix, n_features=3):
    """Build a minimal statement DataFrame with ticker, quarter, and feature columns."""
    rows = []
    for t in tickers:
        for q in quarters:
            row = {"ticker": t, "quarter": q}
            for i in range(n_features):
                row[f"{prefix}_feat_{i}_DIFF_Y"] = float(i)
                row[f"{prefix}_feat_{i}_DIFF_Q"] = float(i * 0.1)
                row[f"{prefix}_feat_{i}"]        = float(i * 10)
            rows.append(row)
    return pd.DataFrame(rows)


class TestAlignedDatasetProperties(unittest.TestCase):
    def setUp(self):
        self.ds = _make_aligned_dataset(N=10, T=12, F=5, M=3)

    def test_fin_dim(self):
        self.assertEqual(self.ds.fin_dim, 5)

    def test_macro_dim(self):
        self.assertEqual(self.ds.macro_dim, 3)

    def test_n_samples(self):
        self.assertEqual(self.ds.n_samples, 10)


class TestTrainConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.epochs, 20)
        self.assertEqual(cfg.batch_size, 32)
        self.assertAlmostEqual(cfg.learning_rate, 1e-3)
        self.assertEqual(cfg.seed, 42)

    def test_custom_values(self):
        cfg = TrainConfig(epochs=50, batch_size=64, t_in=20, t_out=4)
        self.assertEqual(cfg.epochs, 50)
        self.assertEqual(cfg.t_in, 20)
        self.assertEqual(cfg.t_out, 4)

    def test_latent_factors_list(self):
        cfg = TrainConfig(latent_factors=[0.5, 1.0])
        self.assertEqual(cfg.latent_factors, [0.5, 1.0])


class TestFilterColumns(unittest.TestCase):
    def setUp(self):
        tickers   = ["A", "B"]
        quarters  = ["2020Q1", "2020Q2"]
        self.bs    = _make_statement_df(tickers, quarters, "bs")
        self.is_   = _make_statement_df(tickers, quarters, "is")
        self.cf    = _make_statement_df(tickers, quarters, "cf")
        self.macro = _make_statement_df(tickers, quarters, "mac")

    def test_only_diff_y_columns_kept(self):
        bs_f, is_f, cf_f, mac_f = filter_columns(self.bs, self.is_, self.cf, self.macro)
        for df in [bs_f, is_f, cf_f, mac_f]:
            feat_cols = [c for c in df.columns if c not in ["ticker", "quarter"]]
            for col in feat_cols:
                self.assertIn("DIFF_Y", col, f"Column {col} should contain DIFF_Y")

    def test_id_columns_preserved(self):
        bs_f, _, _, _ = filter_columns(self.bs, self.is_, self.cf, self.macro)
        self.assertIn("ticker",  bs_f.columns)
        self.assertIn("quarter", bs_f.columns)

    def test_non_diff_y_columns_excluded(self):
        bs_f, _, _, _ = filter_columns(self.bs, self.is_, self.cf, self.macro)
        raw_cols = [c for c in bs_f.columns if c not in ["ticker", "quarter"] and "DIFF_Y" not in c]
        self.assertEqual(len(raw_cols), 0)


class TestCreateAlignedDataset(unittest.TestCase):
    def _build_inputs(self, tickers, quarters):
        bs    = pd.DataFrame([{"ticker": t, "quarter": q, "bs_A_DIFF_Y": 1.0, "bs_B_DIFF_Y": 2.0}
                               for t in tickers for q in quarters])
        is_   = pd.DataFrame([{"ticker": t, "quarter": q, "is_A_DIFF_Y": 3.0}
                               for t in tickers for q in quarters])
        cf    = pd.DataFrame([{"ticker": t, "quarter": q, "cf_A_DIFF_Y": 4.0}
                               for t in tickers for q in quarters])
        macro = pd.DataFrame([{"quarter": q, "GDP_DIFF_Y": 0.5, "CPI_DIFF_Y": 0.3}
                               for q in quarters])
        return bs, is_, cf, macro

    def test_output_tensor_shapes(self):
        quarters = [f"201{y}Q{q}" for y in range(4, 8) for q in range(1, 5)]  # 16 quarters
        tickers  = ["T1", "T2", "T3"]
        seq_len  = 8
        bs, is_, cf, macro = self._build_inputs(tickers, quarters)
        ds = create_aligned_dataset(bs, is_, cf, macro, seq_len=seq_len)

        expected_windows = len(tickers) * (len(quarters) - seq_len + 1)
        self.assertEqual(ds.X_fin.shape[0],  expected_windows)
        self.assertEqual(ds.X_fin.shape[1],  seq_len)
        self.assertEqual(ds.X_macro.shape[1], seq_len)
        self.assertEqual(ds.X_fin.shape, ds.Y_fin.shape)

    def test_tickers_with_too_few_quarters_skipped(self):
        quarters_long  = [f"201{y}Q{q}" for y in range(4, 8) for q in range(1, 5)]  # 16
        quarters_short = ["2020Q1", "2020Q2"]  # only 2
        seq_len = 8

        bs    = pd.concat([
            pd.DataFrame([{"ticker": "long", "quarter": q, "feat_DIFF_Y": 1.0} for q in quarters_long]),
            pd.DataFrame([{"ticker": "short", "quarter": q, "feat_DIFF_Y": 1.0} for q in quarters_short]),
        ])
        is_   = bs.rename(columns={"feat_DIFF_Y": "feat2_DIFF_Y"}).assign(feat2_DIFF_Y=2.0)
        cf    = bs.rename(columns={"feat_DIFF_Y": "feat3_DIFF_Y"}).assign(feat3_DIFF_Y=3.0)
        all_q = sorted(set(quarters_long) | set(quarters_short))
        macro = pd.DataFrame([{"quarter": q, "mac_DIFF_Y": 0.1} for q in all_q])

        ds = create_aligned_dataset(bs, is_, cf, macro, seq_len=seq_len)
        tickers_in_meta = set(ds.meta_df["ticker"].unique())
        self.assertIn("long",  tickers_in_meta)
        self.assertNotIn("short", tickers_in_meta)

    def test_col_names_populated(self):
        quarters = [f"2020Q{q}" for q in range(1, 5)] + [f"2021Q{q}" for q in range(1, 5)]
        tickers  = ["T1"]
        bs, is_, cf, macro = self._build_inputs(tickers, quarters)
        ds = create_aligned_dataset(bs, is_, cf, macro, seq_len=4)
        self.assertGreater(len(ds.fin_cols), 0)
        self.assertGreater(len(ds.macro_cols), 0)

    def test_target_equals_input(self):
        quarters = [f"2020Q{q}" for q in range(1, 5)] + [f"2021Q{q}" for q in range(1, 5)]
        bs, is_, cf, macro = self._build_inputs(["T1"], quarters)
        ds = create_aligned_dataset(bs, is_, cf, macro, seq_len=4)
        self.assertTrue(torch.equal(ds.X_fin, ds.Y_fin))


if __name__ == "__main__":
    unittest.main()
