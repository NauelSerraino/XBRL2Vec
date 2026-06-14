import sys
import os
import unittest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.autoencoder_dlinear_forecaster import (
    MovingAvg,
    SeriesDecomp,
    DLinearEncoder,
    MacroConditioner,
    FinancialDecoder,
    ForecastingAE,
)
from models.autoencoder_dlinear_blind import FinancialOnlyAE


class TestMovingAvg(unittest.TestCase):
    def test_output_shape_preserved(self):
        B, T, F = 4, 12, 6
        layer = MovingAvg(kernel_size=5)
        x = torch.randn(B, T, F)
        out = layer(x)
        self.assertEqual(out.shape, (B, T, F))

    def test_constant_signal_unchanged(self):
        """A constant signal should pass through moving average unchanged."""
        B, T, F = 2, 8, 3
        x = torch.ones(B, T, F) * 5.0
        out = MovingAvg(kernel_size=3)(x)
        self.assertTrue(torch.allclose(out, x, atol=1e-5))


class TestSeriesDecomp(unittest.TestCase):
    def test_seasonal_plus_trend_equals_input(self):
        B, T, F = 3, 10, 4
        x = torch.randn(B, T, F)
        seasonal, trend = SeriesDecomp(kernel_size=5)(x)
        self.assertTrue(torch.allclose(seasonal + trend, x, atol=1e-5))

    def test_output_shapes(self):
        B, T, F = 3, 10, 4
        x = torch.randn(B, T, F)
        seasonal, trend = SeriesDecomp(kernel_size=5)(x)
        self.assertEqual(seasonal.shape, (B, T, F))
        self.assertEqual(trend.shape,    (B, T, F))


class TestDLinearEncoder(unittest.TestCase):
    def test_output_shape(self):
        B, T, F = 5, 12, 8
        enc = DLinearEncoder(T=T, F=F)
        out = enc(torch.randn(B, T, F))
        self.assertEqual(out.shape, (B, 2 * F))

    def test_grad_flows(self):
        B, T, F = 3, 8, 4
        enc = DLinearEncoder(T=T, F=F)
        x = torch.randn(B, T, F, requires_grad=True)
        loss = enc(x).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestMacroConditioner(unittest.TestCase):
    def test_gamma_beta_shapes(self):
        B, M, fin_dim = 6, 5, 10
        h_macro = torch.randn(B, 2 * M)
        cond = MacroConditioner(M=2 * M, fin_dim=fin_dim)
        gamma, beta = cond(h_macro)
        self.assertEqual(gamma.shape, (B, fin_dim))
        self.assertEqual(beta.shape,  (B, fin_dim))

    def test_film_modulation_changes_representation(self):
        """FiLM should change h_fin (unless gamma=1, beta=0 by coincidence)."""
        B, M, fin_dim = 4, 3, 8
        h_macro = torch.randn(B, 2 * M)
        h_fin   = torch.randn(B, fin_dim)
        cond    = MacroConditioner(M=2 * M, fin_dim=fin_dim)
        gamma, beta = cond(h_macro)
        h_modulated = gamma * h_fin + beta
        self.assertFalse(torch.equal(h_modulated, h_fin))


class TestFinancialDecoder(unittest.TestCase):
    def test_output_shape(self):
        B, T_out, F, latent_dim = 4, 4, 6, 16
        dec = FinancialDecoder(latent_dim=latent_dim, T=T_out, F=F)
        z   = torch.randn(B, latent_dim)
        out = dec(z)
        self.assertEqual(out.shape, (B, T_out, F))

    def test_grad_flows(self):
        B, T_out, F, latent_dim = 3, 2, 5, 10
        dec = FinancialDecoder(latent_dim=latent_dim, T=T_out, F=F)
        z   = torch.randn(B, latent_dim, requires_grad=True)
        dec(z).sum().backward()
        self.assertIsNotNone(z.grad)


class TestForecastingAE(unittest.TestCase):
    def setUp(self):
        self.B, self.T_in, self.T_out = 8, 12, 4
        self.F, self.M, self.latent   = 6, 3, 16
        self.model = ForecastingAE(
            T_in=self.T_in, T_out=self.T_out,
            F=self.F, M=self.M, latent_dim=self.latent,
        )
        self.x_fin   = torch.randn(self.B, self.T_in, self.F)
        self.x_macro = torch.randn(self.B, self.T_in, self.M)

    def test_output_shapes_with_macro(self):
        z, x_hat = self.model(self.x_fin, self.x_macro)
        self.assertEqual(z.shape,     (self.B, self.latent))
        self.assertEqual(x_hat.shape, (self.B, self.T_out, self.F))

    def test_output_shapes_without_macro(self):
        z, x_hat = self.model(self.x_fin, x_macro=None)
        self.assertEqual(z.shape,     (self.B, self.latent))
        self.assertEqual(x_hat.shape, (self.B, self.T_out, self.F))

    def test_macro_changes_output(self):
        """Conditioning on different macro tensors should produce different latents."""
        z1, _ = self.model(self.x_fin, self.x_macro)
        z2, _ = self.model(self.x_fin, torch.randn_like(self.x_macro) * 10)
        self.assertFalse(torch.equal(z1, z2))

    def test_no_macro_vs_zero_macro_differ(self):
        """x_macro=None skips FiLM; x_macro=zeros still passes through conditioner."""
        z_none, _ = self.model(self.x_fin, x_macro=None)
        z_zero, _ = self.model(self.x_fin, torch.zeros_like(self.x_macro))
        self.assertFalse(torch.equal(z_none, z_zero))

    def test_eval_mode_no_grad(self):
        self.model.eval()
        with torch.no_grad():
            z, x_hat = self.model(self.x_fin, self.x_macro)
        self.assertFalse(z.requires_grad)

    def test_backward_pass(self):
        z, x_hat = self.model(self.x_fin, self.x_macro)
        loss = x_hat.mean()
        loss.backward()
        for p in self.model.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
                break


class TestFinancialOnlyAE(unittest.TestCase):
    def setUp(self):
        self.B, self.T_in, self.T_out = 6, 10, 4
        self.F, self.latent = 5, 12
        self.model = FinancialOnlyAE(
            T_in=self.T_in, T_out=self.T_out,
            F=self.F, latent_dim=self.latent,
        )
        self.x_fin = torch.randn(self.B, self.T_in, self.F)

    def test_output_shapes(self):
        z, x_hat = self.model(self.x_fin)
        self.assertEqual(z.shape,     (self.B, self.latent))
        self.assertEqual(x_hat.shape, (self.B, self.T_out, self.F))

    def test_macro_argument_ignored(self):
        z1, x1 = self.model(self.x_fin, x_macro=None)
        z2, x2 = self.model(self.x_fin, x_macro=torch.randn(self.B, self.T_in, 7))
        self.assertTrue(torch.equal(z1, z2))
        self.assertTrue(torch.equal(x1, x2))

    def test_drop_in_compatibility_with_forecasting_ae(self):
        """FinancialOnlyAE must accept x_macro kwarg like ForecastingAE does."""
        x_macro = torch.randn(self.B, self.T_in, 3)
        z, x_hat = self.model(self.x_fin, x_macro=x_macro)
        self.assertEqual(z.shape[0], self.B)


if __name__ == "__main__":
    unittest.main()
