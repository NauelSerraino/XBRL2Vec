import sys
import os
import unittest
import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.transforms import symmetric_log, _weights_corr, _weights_ridge, transform_dataset
from services.data import AlignedDataset


def _make_dataset(N=8, T=12, F=5, M=3):
    return AlignedDataset(
        X_fin    = torch.randn(N, T, F),
        X_macro  = torch.randn(N, T, M),
        Y_fin    = torch.randn(N, T, F),
        meta_df  = pd.DataFrame({"ticker": [f"T{i}" for i in range(N)], "end_quarter": ["2020Q4"] * N}),
        fin_cols = [f"fin_{i}" for i in range(F)],
        macro_cols = [f"mac_{i}" for i in range(M)],
    )


class TestSymmetricLog(unittest.TestCase):
    def test_zero_maps_to_zero(self):
        t = torch.zeros(3, 3)
        self.assertTrue(torch.allclose(symmetric_log(t), torch.zeros(3, 3)))

    def test_positive_values_compressed(self):
        t = torch.tensor([1.0, 10.0, 100.0])
        result = symmetric_log(t)
        self.assertTrue((result > 0).all())
        self.assertLess(result[2].item(), t[2].item())

    def test_negative_values_stay_negative(self):
        t = torch.tensor([-1.0, -10.0, -100.0])
        result = symmetric_log(t)
        self.assertTrue((result < 0).all())

    def test_sign_preserved(self):
        t = torch.tensor([-5.0, 0.0, 5.0])
        result = symmetric_log(t)
        self.assertEqual(result[0].sign().item(), -1.0)
        self.assertEqual(result[2].sign().item(),  1.0)

    def test_output_shape_preserved(self):
        t = torch.randn(4, 8, 3)
        self.assertEqual(symmetric_log(t).shape, t.shape)

    def test_antisymmetric(self):
        t = torch.tensor([3.0, -3.0])
        r = symmetric_log(t)
        self.assertAlmostEqual(r[0].item(), -r[1].item(), places=6)


class TestWeightsCorr(unittest.TestCase):
    def setUp(self):
        self.N, self.T, self.F, self.M = 6, 10, 4, 3
        self.X_fin   = torch.randn(self.N, self.T, self.F)
        self.X_macro = torch.randn(self.N, self.T, self.M)

    def test_output_shape(self):
        w = _weights_corr(self.X_fin, self.X_macro)
        self.assertEqual(w.shape, (self.N, self.M))

    def test_values_in_range(self):
        w = _weights_corr(self.X_fin, self.X_macro)
        self.assertTrue((w >= -1.0).all() and (w <= 1.0).all())

    def test_perfectly_correlated(self):
        fin   = torch.linspace(0, 1, self.T).unsqueeze(0).unsqueeze(-1).expand(1, self.T, self.F)
        macro = torch.linspace(0, 1, self.T).unsqueeze(0).unsqueeze(-1).expand(1, self.T, 1)
        w = _weights_corr(fin, macro)
        self.assertAlmostEqual(w[0, 0].item(), 1.0, places=4)


class TestWeightsRidge(unittest.TestCase):
    def test_output_shape(self):
        N, T, F, M = 5, 8, 3, 4
        w = _weights_ridge(torch.randn(N, T, F), torch.randn(N, T, M))
        self.assertEqual(w.shape, (N, M))


class TestTransformDataset(unittest.TestCase):
    def setUp(self):
        self.ds = _make_dataset(N=6, T=10, F=4, M=3)

    def test_output_tensors_same_shape(self):
        out = transform_dataset(self.ds, macro_weight_mode="corr")
        self.assertEqual(out.X_fin.shape,   self.ds.X_fin.shape)
        self.assertEqual(out.X_macro.shape, self.ds.X_macro.shape)
        self.assertEqual(out.Y_fin.shape,   self.ds.Y_fin.shape)

    def test_symmetric_log_applied(self):
        out = transform_dataset(self.ds, macro_weight_mode="corr")
        # After symmetric_log the absolute max is much smaller than raw for large values
        large = torch.full((2, 5, 3), 1000.0)
        ds2 = _make_dataset(N=2, T=5, F=3, M=2)
        ds2.X_fin = large
        out2 = transform_dataset(ds2, macro_weight_mode="corr")
        self.assertLess(out2.X_fin.abs().max().item(), 1000.0)

    def test_meta_and_cols_preserved(self):
        out = transform_dataset(self.ds, macro_weight_mode="corr")
        self.assertEqual(out.fin_cols,   self.ds.fin_cols)
        self.assertEqual(out.macro_cols, self.ds.macro_cols)
        self.assertEqual(len(out.meta_df), len(self.ds.meta_df))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            transform_dataset(self.ds, macro_weight_mode="invalid_mode")

    def test_ridge_mode_runs(self):
        out = transform_dataset(self.ds, macro_weight_mode="ridge")
        self.assertEqual(out.X_macro.shape, self.ds.X_macro.shape)


if __name__ == "__main__":
    unittest.main()
