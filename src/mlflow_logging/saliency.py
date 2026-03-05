"""
Saliency analysis using Captum Integrated Gradients.
Covers both per-company (gradient-based) and global (IG) attribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

import mlflow

from mlflow_logging.artifacts import ArtifactGroup, ArtifactLogger
from services.config import DEVICE
from services.data import SaliencyMode


# ---------------------------------------------------------------------------
# Wrapper for captum
# ---------------------------------------------------------------------------

class _LatentWrapper(torch.nn.Module):
    """Wraps a model to expose a scalar output for IG attribution."""

    def __init__(self, model: torch.nn.Module, mode: SaliencyMode, Y_fin: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        self.mode  = mode
        self.Y_fin = Y_fin  # forecast target; if None, uses X_fin (autoencoder case)

    def forward(self, X_fin: torch.Tensor, X_macro: torch.Tensor) -> torch.Tensor:
        z, x_hat = self.model(X_fin, X_macro)
        if self.mode == SaliencyMode.LATENT:
            return torch.norm(z, dim=1)
        if self.Y_fin is not None:
            # Captum expands the batch to [N * n_steps, ...]; tile Y_fin to match.
            n_orig = self.Y_fin.shape[0]
            repeats = x_hat.shape[0] // n_orig
            target = self.Y_fin.repeat(repeats, 1, 1)
        else:
            target = X_fin
        return ((x_hat - target) ** 2).mean(dim=(1, 2))


# ---------------------------------------------------------------------------
# Per-company gradient saliency
# ---------------------------------------------------------------------------

def compute_saliency_per_company(
    model: torch.nn.Module,
    X_fin: torch.Tensor,
    X_macro: torch.Tensor,
    fin_cols: list[str],
    macro_cols: list[str],
    meta_df: pd.DataFrame,
    metadata_sector_df: pd.DataFrame,
    mode: SaliencyMode,
    logger: ArtifactLogger,
    device: torch.device = DEVICE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[INFO] Per-company saliency ({mode.value})")
    model.eval().to(device)
    tickers = meta_df["ticker"].unique()
    rows = []

    for ticker in tickers:
        idx = meta_df[meta_df["ticker"] == ticker].index.tolist()
        if not idx:
            continue

        xf = X_fin[idx].to(device).requires_grad_(True)
        xm = X_macro[idx].to(device).requires_grad_(True)

        if mode == SaliencyMode.LATENT:
            z, _ = model(xf, xm)
            target = z.abs().sum()
        else:
            _, x_hat = model(xf, xm)
            target = x_hat.abs().sum()

        target.backward()

        sal_fin   = xf.grad.abs().mean(dim=(0, 1)).cpu().numpy()
        sal_macro = xm.grad.abs().mean(dim=(0, 1)).cpu().numpy()

        row = {"ticker": ticker}
        row.update(dict(zip(fin_cols, sal_fin)))
        row.update(dict(zip(macro_cols, sal_macro)))
        rows.append(row)

    df_company = (
        pd.DataFrame(rows)
        .merge(metadata_sector_df, on="ticker", how="left")
        .set_index("ticker")
    )
    logger.log_table(df_company, ArtifactGroup.SALIENCY, f"per_company_{mode.value}")

    # Sector aggregation
    df_sector = (
        df_company.reset_index()
        .groupby("sector")[fin_cols + macro_cols]
        .mean()
    )
    df_sector["fin_exposure"]   = df_sector[fin_cols].sum(axis=1)
    df_sector["macro_exposure"] = df_sector[macro_cols].sum(axis=1)
    df_sector["macro_fin_ratio"] = df_sector["macro_exposure"] / (df_sector["fin_exposure"] + 1e-9)
    df_sector = df_sector.sort_values("macro_exposure", ascending=False)

    logger.log_table(df_sector, ArtifactGroup.SALIENCY, f"per_sector_{mode.value}")

    return df_company, df_sector


# ---------------------------------------------------------------------------
# Global IG saliency (both modes at once)
# ---------------------------------------------------------------------------

def compute_full_saliency(
    model: torch.nn.Module,
    X_fin: torch.Tensor,
    X_macro: torch.Tensor,
    fin_cols: list[str],
    macro_cols: list[str],
    meta_df: pd.DataFrame,
    metadata_sector_df: pd.DataFrame,
    logger: ArtifactLogger,
    device: torch.device = DEVICE,
    top_n: int = 30,
    Y_fin: torch.Tensor | None = None,
) -> dict[str, dict]:
    """
    Run Integrated Gradients for both SaliencyMode values.

    Returns a dict keyed by mode: {df_global, df_sector, macro_ratio}
    """
    print(f"[INFO] Full IG saliency – {logger.run_name}")
    model.eval().to(device)

    xf = X_fin.to(device)
    xm = X_macro.to(device)
    yf = Y_fin.to(device) if Y_fin is not None else None
    baseline_fin   = torch.zeros_like(xf)
    baseline_macro = torch.zeros_like(xm)

    results = {}

    for mode in SaliencyMode:
        wrapper = _LatentWrapper(model, mode, Y_fin=yf)
        ig = IntegratedGradients(wrapper)

        attr_fin, attr_macro = ig.attribute(
            inputs=(xf, xm),
            baselines=(baseline_fin, baseline_macro),
            n_steps=50,
        )

        # ---- Global feature saliency ----
        attr_fin_g   = attr_fin.abs().mean(dim=(0, 1)).cpu().numpy()
        attr_macro_g = attr_macro.abs().mean(dim=(0, 1)).cpu().numpy()
        total_fin    = attr_fin_g.sum()
        total_macro  = attr_macro_g.sum()
        macro_ratio  = total_macro / (total_fin + total_macro + 1e-9)

        mlflow.log_metric(f"{mode.value}_macro_ratio", macro_ratio)

        df_global = pd.concat([
            pd.DataFrame({"feature": fin_cols,   "saliency": attr_fin_g,   "type": "financial"}),
            pd.DataFrame({"feature": macro_cols,  "saliency": attr_macro_g, "type": "macro"}),
        ]).sort_values("saliency", ascending=False)

        logger.log_table(df_global, ArtifactGroup.SALIENCY, f"global_features_{mode.value}")

        # Top-N bar chart
        df_top = df_global.head(top_n)
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        sns.barplot(
            data=df_top, x="saliency", y="feature", hue="type",
            palette={"financial": "#1f77b4", "macro": "#ff7f0e"}, dodge=False,
        )
        plt.title(f"Top {top_n} Feature Saliency ({mode.value}) – {logger.run_name}")
        plt.tight_layout()

        logger.log_figure(plt.gcf(), ArtifactGroup.SALIENCY, f"global_features_{mode.value}")

        # ---- Sector exposure ----
        attr_macro_sample = attr_macro.abs().mean(dim=1).cpu().numpy()  # (N, M)
        macro_exp_per_sample = attr_macro_sample.sum(axis=1)

        df_sector = (
            pd.DataFrame({"ticker": meta_df["ticker"].values, "macro_exposure": macro_exp_per_sample})
            .merge(metadata_sector_df, on="ticker", how="left")
            .groupby("sector")["macro_exposure"]
            .mean()
            .sort_values(ascending=False)
            .to_frame()
        )
        logger.log_table(df_sector, ArtifactGroup.SALIENCY, f"sector_exposure_{mode.value}")

        # Top / Bottom 15 bar charts
        for label, subset, invert in [
            ("top15",    df_sector.head(15),                         True),
            ("bottom15", df_sector.tail(15).sort_values("macro_exposure"), False),
        ]:
            plt.figure(figsize=(8, 6))
            data = subset["macro_exposure"]
            idx  = subset.index
            plt.barh(idx[::-1] if invert else idx, data[::-1] if invert else data)
            if not invert:
                plt.gca().invert_yaxis()
            plt.title(f"{label.replace('15',' 15').title()} Macro-Exposed Sectors – {mode.value}")
            plt.tight_layout()

            logger.log_figure(plt.gcf(), ArtifactGroup.SALIENCY, f"sector_exposure_{label}_{mode.value}")

        results[mode.value] = {"df_global": df_global, "df_sector": df_sector, "macro_ratio": macro_ratio}

    # ---- Sector composition bar ----
    n_companies = (
        metadata_sector_df.groupby("sector")["ticker"]
        .nunique()
        .sort_values(ascending=False)
        .rename("n_companies")
        .reset_index()
    )
    logger.log_table(n_companies, ArtifactGroup.SALIENCY, "companies_per_sector")

    # ---- Bubble chart (one per mode) ----
    df_lat   = results["latent"]["df_sector"].rename(columns={"macro_exposure": "latent_exposure"})
    df_recon = results["reconstruction"]["df_sector"].rename(columns={"macro_exposure": "recon_exposure"})
    df_bubble_base = df_lat.join(df_recon, how="inner")
    df_bubble_base["n_companies"] = df_bubble_base.index.map(
        metadata_sector_df.groupby("sector")["ticker"].nunique()
    )
    df_bubble_base = df_bubble_base.dropna()

    for sort_col, sort_label in [("recon_exposure", "reconstruction"), ("latent_exposure", "latent")]:
        df_b = df_bubble_base.nlargest(10, sort_col)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            df_b["latent_exposure"], df_b["recon_exposure"],
            s=df_b["n_companies"] * 20, alpha=0.6,
            color="#1f77b4", edgecolors="white", linewidths=0.5,
        )
        for sector, row in df_b.iterrows():
            ax.annotate(sector, (row["latent_exposure"], row["recon_exposure"]),
                        fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points")

        ax.set_xlabel("Latent Macro Exposure")
        ax.set_ylabel("Reconstruction Macro Exposure")
        ax.set_title(
            f"Top 10 Sector Macro Exposure (by {sort_label.upper()})\n"
            f"(bubble = n companies) – {logger.run_name}"
        )
        plt.tight_layout()

        logger.log_figure(fig, ArtifactGroup.SALIENCY, f"sector_bubble_{sort_label}")

    print(f"  Latent macro ratio:         {results['latent']['macro_ratio']:.4f}")
    print(f"  Reconstruction macro ratio: {results['reconstruction']['macro_ratio']:.4f}")

    return results