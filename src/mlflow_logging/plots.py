"""
All matplotlib / seaborn plot loggers.
Each function: builds figure → saves via ArtifactLogger → logs to MLflow.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

import mlflow

from mlflow_logging.artifacts import ArtifactGroup, ArtifactLogger
from services.config import DEVICE


# Key variables shown by default in the aggregate forecast plot.
# Strip the suffix before display; only variables present in fin_cols are used.
_DEFAULT_KEY_VARS = [
    "Total Revenue_DIFF_Y",
    "Net Income_DIFF_Y",
    "Gross Profit_DIFF_Y",
    "Operating Income_DIFF_Y",
    "Operating Cash Flow_DIFF_Y",
    "Total Assets_DIFF_Y",
    # DIFF_Q variants (used when model is trained on quarterly diffs)
    "Total Revenue_DIFF_Q",
    "Net Income_DIFF_Q",
    "Gross Profit_DIFF_Q",
    "Operating Income_DIFF_Q",
    "Operating Cash Flow_DIFF_Q",
    "Total Assets_DIFF_Q",
]


# ---------------------------------------------------------------------------
# DISTRIBUTION plots
# ---------------------------------------------------------------------------

def log_zero_sparsity(
    X_fin: torch.Tensor,
    X_macro: torch.Tensor,
    fin_cols: list[str],
    macro_cols: list[str],
    logger: ArtifactLogger,
) -> None:
    print("[INFO] Zero sparsity plot")
    fin_flat   = X_fin.cpu().numpy().reshape(-1, len(fin_cols))
    macro_flat = X_macro.cpu().numpy().reshape(-1, len(macro_cols))

    fin_zeros   = (fin_flat   == 0).mean(axis=0) * 100
    macro_zeros = (macro_flat == 0).mean(axis=0) * 100

    df = pd.concat([
        pd.DataFrame({"Feature": fin_cols,   "Zero_Pct": fin_zeros,   "Type": "Financial"}),
        pd.DataFrame({"Feature": macro_cols,  "Zero_Pct": macro_zeros, "Type": "Macro"}),
    ]).sort_values("Zero_Pct")

    plt.figure(figsize=(10, len(df) * 0.25 + 2))
    sns.barplot(
        data=df, x="Zero_Pct", y="Feature", hue="Type",
        palette={"Financial": "#1f77b4", "Macro": "#ff7f0e"}, dodge=False,
    )
    plt.axvline(50, color="red", linestyle="--", alpha=0.5)
    plt.title(f"Zero Observation Sparsity (%) – {logger.run_name}")
    plt.xlabel("Percentage of Zero Values")
    plt.xlim(0, 100)
    plt.tight_layout()

    logger.log_figure(plt.gcf(), ArtifactGroup.DISTRIBUTION, "zero_sparsity")


def log_financial_boxplots(
    X_fin: torch.Tensor,
    fin_cols: list[str],
    logger: ArtifactLogger,
    chunk_size: int = 10,
) -> None:
    print("[INFO] Financial boxenplots")
    fin_flat = X_fin.cpu().numpy().reshape(-1, len(fin_cols))

    for i in range(0, len(fin_cols), chunk_size):
        subset_cols = fin_cols[i: i + chunk_size]
        df = pd.DataFrame(fin_flat[:, i: i + chunk_size], columns=subset_cols)

        plt.figure(figsize=(14, 8))
        sns.boxenplot(data=df, orient="h", palette="Blues_d", k_depth="proportion")
        plt.title(f"Financial Distributions – Chunk {i // chunk_size + 1} – {logger.run_name}")
        plt.xlabel("Value")
        plt.grid(axis="x", alpha=0.3, linestyle="--")
        plt.tight_layout()

        logger.log_figure(plt.gcf(), ArtifactGroup.DISTRIBUTION, f"financial_boxenplot_chunk_{i // chunk_size}")


def log_macro_boxplots(
    X_macro: torch.Tensor,
    macro_cols: list[str],
    logger: ArtifactLogger,
    chunk_size: int = 10,
) -> None:
    print("[INFO] Macro boxenplots")
    macro_flat = X_macro.cpu().numpy().reshape(-1, len(macro_cols))

    for i in range(0, len(macro_cols), chunk_size):
        subset_cols = macro_cols[i: i + chunk_size]
        df = pd.DataFrame(macro_flat[:, i: i + chunk_size], columns=subset_cols)

        plt.figure(figsize=(14, 8))
        sns.boxenplot(data=df, orient="h", color="#ff7f0e", k_depth="proportion")
        plt.title(f"Macro Distributions – Chunk {i // chunk_size + 1} – {logger.run_name}")
        plt.xlabel("Value")
        plt.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()

        logger.log_figure(plt.gcf(), ArtifactGroup.DISTRIBUTION, f"macro_boxenplot_chunk_{i // chunk_size}")


def log_correlation_matrix(
    X_fin: torch.Tensor,
    X_macro: torch.Tensor,
    fin_cols: list[str],
    macro_cols: list[str],
    logger: ArtifactLogger,
) -> pd.DataFrame:
    print("[INFO] Correlation matrix")
    fin_flat   = X_fin.cpu().numpy().reshape(-1, len(fin_cols))
    macro_flat = X_macro.cpu().numpy().reshape(-1, len(macro_cols))

    corr = pd.concat([
        pd.DataFrame(fin_flat,   columns=fin_cols),
        pd.DataFrame(macro_flat, columns=macro_cols),
    ], axis=1).corr()

    logger.log_table(corr, ArtifactGroup.DISTRIBUTION, "correlation_matrix")

    plt.figure(figsize=(16, 12))
    im = plt.imshow(corr.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, label="Pearson Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(f"Feature Correlation Matrix – {logger.run_name}")
    plt.tight_layout()

    logger.log_figure(plt.gcf(), ArtifactGroup.DISTRIBUTION, "correlation_matrix")

    return corr


# ---------------------------------------------------------------------------
# TRAINING plots
# ---------------------------------------------------------------------------

def log_loss_comparison(
    metrics_ctx: list,
    metrics_blind: list,
    logger: ArtifactLogger,
) -> None:
    print("[INFO] Loss comparison plot")
    epochs = [m.epoch + 1 for m in metrics_ctx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (attr, label) in enumerate([("mse", "MSE"), ("mae", "MAE"), ("smooth", "SmoothL1")]):
        axes[i].plot(epochs, [getattr(m, attr) for m in metrics_ctx],   label="Contextual")
        axes[i].plot(epochs, [getattr(m, attr) for m in metrics_blind], label="Blind")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(label)
        axes[i].set_title(label)
        axes[i].legend()

    plt.suptitle(f"Loss Comparison – {logger.run_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    logger.log_figure(fig, ArtifactGroup.TRAINING, "loss_comparison")


# ---------------------------------------------------------------------------
# IMPORTANCE plots
# ---------------------------------------------------------------------------

def log_importance_matrix(
    df: pd.DataFrame,
    label: str,
    logger: ArtifactLogger,
) -> None:
    print(f"[INFO] Importance matrix ({label})")
    logger.log_table(df.reset_index(), ArtifactGroup.IMPORTANCE, f"matrix_{label}")

    data = df.values.copy()
    masked = np.ma.masked_where(data <= 0, data)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="lightgray")

    positive = data[data > 0]
    if len(positive) == 0:
        vmin, vmax = 0, 1
    else:
        vmin = np.percentile(positive, 1)
        vmax = np.percentile(positive, 99)
        if vmin == vmax:
            vmax = vmin + 1e-8

    plt.figure(figsize=(14, 10))
    im = plt.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Δ MSE")
    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.index)), df.index)
    plt.title(f"Importance Matrix ({label}) – {logger.run_name}")
    plt.tight_layout()

    logger.log_figure(plt.gcf(), ArtifactGroup.IMPORTANCE, f"matrix_{label}")


def log_importance_summary(
    imp_ctx: pd.DataFrame,
    imp_blind: pd.DataFrame,
    fin_cols: list[str],
    macro_cols: list[str],
    logger: ArtifactLogger,
) -> None:
    print("[INFO] Importance summary")

    def _summarize(df):
        return pd.Series({
            "financial": df[fin_cols].abs().mean().sum(),
            "macro":     df[[c for c in df.columns if c in macro_cols]].abs().mean().sum(),
        })

    df_summary = pd.DataFrame({
        "Contextual": _summarize(imp_ctx),
        "Blind":      _summarize(imp_blind),
    })
    logger.log_table(df_summary.reset_index(), ArtifactGroup.IMPORTANCE, "summary_contextual_vs_blind")

    plt.figure(figsize=(6, 4))
    df_summary.plot.bar()
    plt.title(f"Importance Summary – {logger.run_name}")
    plt.tight_layout()

    logger.log_figure(plt.gcf(), ArtifactGroup.IMPORTANCE, "summary_contextual_vs_blind")


# ---------------------------------------------------------------------------
# TOURNAMENT plots
# ---------------------------------------------------------------------------

def log_macro_embedding_tournament(
    model_ctx: torch.nn.Module,
    model_blind: torch.nn.Module,
    X_fin: torch.Tensor,
    X_macro: torch.Tensor,
    logger: ArtifactLogger,
    device: torch.device = DEVICE,
) -> pd.DataFrame:
    print("[INFO] Macro embedding tournament")

    results = []
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_fin, X_macro), batch_size=32
    )

    for label, model in [("Contextual", model_ctx), ("Blind", model_blind)]:
        model.eval()
        cos_dists, euc_dists = [], []

        for xf, xm in loader:
            xf = xf.to(device); xm = xm.to(device)
            xm_perm = xm[torch.randperm(xm.size(0), device=device)]

            with torch.no_grad():
                z_real, _ = model(xf, xm)
                z_perm, _ = model(xf, xm_perm)

            cos_dists.append((1 - F.cosine_similarity(z_real, z_perm, dim=-1)).cpu())
            euc_dists.append(torch.norm(z_real - z_perm, dim=-1).cpu())

        cos_dists = torch.cat(cos_dists).numpy()
        euc_dists = torch.cat(euc_dists).numpy()

        results.append({
            "Model": label,
            "Cosine_Distance_mean": cos_dists.mean(), "Cosine_Distance_std": cos_dists.std(),
            "Euclidean_Distance_mean": euc_dists.mean(), "Euclidean_Distance_std": euc_dists.std(),
        })

    df = pd.DataFrame(results)
    logger.log_table(df, ArtifactGroup.TOURNAMENT, "macro_embedding_tournament")

    x = np.arange(len(df))
    for metric, col_mean, col_std, color, name in [
        ("Cosine",    "Cosine_Distance_mean",    "Cosine_Distance_std",    "#1f77b4", "cosine"),
        ("Euclidean", "Euclidean_Distance_mean",  "Euclidean_Distance_std",  "#ff7f0e", "euclidean"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.bar(x, df[col_mean], yerr=df[col_std], capsize=5, color=color)
        plt.xticks(x, df["Model"])
        plt.ylabel(f"{metric} distance")
        plt.title(f"Embedding Shift from Macro ({metric})\n{logger.run_name}")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        logger.log_figure(plt.gcf(), ArtifactGroup.TOURNAMENT, f"macro_embedding_{name}")

    return df


def log_macro_exposure_density(
    exposure_blind: np.ndarray,
    exposure_contextual: np.ndarray,
    metric: str,       # "cosine" or "l2"
    logger: ArtifactLogger,
) -> pd.DataFrame:
    print(f"[INFO] Macro exposure density ({metric})")

    df = pd.DataFrame({"blind": exposure_blind, "contextual": exposure_contextual})
    logger.log_table(df, ArtifactGroup.TOURNAMENT, f"macro_exposure_{metric}")

    for label, arr in [("blind", exposure_blind), ("contextual", exposure_contextual)]:
        mean_v = float(np.mean(arr))
        std_v  = float(np.std(arr))
        mlflow.log_metric(f"{metric}_{label}_mean", mean_v)
        mlflow.log_metric(f"{metric}_{label}_std",  std_v)
        mlflow.log_metric(f"{metric}_{label}_cv",   std_v / (mean_v + 1e-9))

    def _plot_series(arr, label):
        if np.std(arr) < 1e-10:
            plt.axvline(np.mean(arr), label=f"{label} (constant={np.mean(arr):.4f})", linewidth=2)
        else:
            pd.Series(arr).plot.kde(label=label, linewidth=2)

    plt.figure(figsize=(7, 4))
    _plot_series(exposure_blind,      "Blind")
    _plot_series(exposure_contextual, "Contextual")
    plt.title(f"Macro Exposure Density – {metric.upper()}")
    plt.xlabel("Exposure"); plt.ylabel("Density")
    plt.legend(); plt.tight_layout()

    logger.log_figure(plt.gcf(), ArtifactGroup.TOURNAMENT, f"macro_exposure_density_{metric}")

    return df


# ---------------------------------------------------------------------------
# EMBEDDING plots
# ---------------------------------------------------------------------------

def log_company_distance_scatter(
    model: torch.nn.Module,
    X_fin: torch.Tensor,
    X_macro: torch.Tensor,
    logger: ArtifactLogger,
    device: torch.device = DEVICE,
) -> None:
    print("[INFO] Company distance scatter (cosine + euclidean)")
    model.eval().to(device)

    with torch.no_grad():
        z, _ = model(X_fin.to(device), torch.zeros_like(X_macro).to(device))

    z_np  = z.cpu().numpy()
    fin   = X_fin.cpu().numpy().reshape(X_fin.shape[0], -1)

    def _upper(mat): return mat[np.triu_indices_from(mat, k=1)]

    cos_fin_flat = _upper(cosine_similarity(fin))
    cos_lat_flat = _upper(cosine_similarity(z_np))
    euc_fin_flat = _upper(euclidean_distances(fin))
    euc_lat_flat = _upper(euclidean_distances(z_np))

    cos_corr, _ = spearmanr(cos_fin_flat, cos_lat_flat)
    euc_corr, _ = spearmanr(euc_fin_flat, euc_lat_flat)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    hb0 = axes[0].hexbin(cos_fin_flat, cos_lat_flat, gridsize=80, cmap="Blues", mincnt=1, bins="log")
    axes[0].set_xlabel("Cosine Similarity Financial")
    axes[0].set_ylabel("Cosine Similarity Latent")
    axes[0].set_title(f"Cosine – Spearman={cos_corr:.3f}")
    plt.colorbar(hb0, ax=axes[0], label="log(Counts)")

    hb1 = axes[1].hexbin(euc_fin_flat, euc_lat_flat, gridsize=80, cmap="Oranges", mincnt=1, bins="log")
    axes[1].set_xlabel("Euclidean Distance Financial")
    axes[1].set_ylabel("Euclidean Distance Latent")
    axes[1].set_title(f"Euclidean – Spearman={euc_corr:.3f}")
    plt.colorbar(hb1, ax=axes[1], label="log(Counts)")

    plt.tight_layout()
    logger.log_figure(fig, ArtifactGroup.EMBEDDING, "distance_scatter_cosine_euclidean")


def log_macro_sensitivity_barplot(
    model: torch.nn.Module,
    X_fin: torch.Tensor,
    X_macro: torch.Tensor,
    macro_cols: list[str],
    logger: ArtifactLogger,
    device: torch.device = DEVICE,
) -> None:
    print("[INFO] Macro sensitivity barplot")
    model.eval().to(device)
    xf = X_fin.to(device); xm = X_macro.to(device)

    with torch.no_grad():
        z_base, _ = model(xf, xm)

    sensitivities = []
    for m in range(xm.shape[2]):
        xm_pert = xm.clone()
        xm_pert[:, :, m] = xm_pert[torch.randperm(xm.shape[0]), :, m]
        with torch.no_grad():
            z_pert, _ = model(xf, xm_pert)
        sensitivities.append((1 - F.cosine_similarity(z_base, z_pert, dim=1)).mean().item())

    order = np.argsort(-np.array(sensitivities))
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(macro_cols)[order], np.array(sensitivities)[order])
    plt.xlabel("1 − cosine similarity")
    plt.title("Macro Sensitivity (Cosine Permutation)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    logger.log_figure(plt.gcf(), ArtifactGroup.EMBEDDING, "macro_sensitivity_barplot", save_kwargs={"dpi": 150, "bbox_inches": "tight"})


def log_variance_analysis_plot(
    r2_df: pd.DataFrame,
    logger: ArtifactLogger,
) -> None:
    """Save the R² variance analysis table as CSV (plot is optional)."""
    logger.log_table(r2_df.reset_index(), ArtifactGroup.EMBEDDING, "variance_r2_analysis")


# ---------------------------------------------------------------------------
# FORECAST plots
# ---------------------------------------------------------------------------

def log_forecast_aggregate_plot(
    timeseries_df: pd.DataFrame,
    logger: ArtifactLogger,
    key_vars: list[str] | None = None,
    split_label: str = "",
) -> None:
    """
    Aggregate (cross-sectional mean ± std) predicted vs actual for key variables.

    Each quarter appears exactly once (walk-forward evaluation, one prediction
    per company per quarter).  The full quarterly time series is shown, starting
    from the earliest available forecast quarter (determined by T_in/T_out).

    Parameters
    ----------
    timeseries_df : output of ``compute_forecast_timeseries``
    key_vars      : fin_col names to plot; defaults to the most economically
                    meaningful ones present in the data.
    split_label   : 'in_sample' or 'oos' – used in title and artifact name.
    """
    print(f"[INFO] Forecast aggregate plot ({split_label})")

    available_vars = timeseries_df["feature"].unique()
    if key_vars is None:
        key_vars = [v for v in _DEFAULT_KEY_VARS if v in available_vars]
    else:
        key_vars = [v for v in key_vars if v in available_vars]

    if not key_vars:
        print("[WARN] No matching key variables found; skipping forecast plot.")
        return

    df = timeseries_df.copy()

    if df.empty:
        print("[WARN] No forecast data available; skipping forecast plot.")
        return

    sorted_quarters = sorted(df["quarter"].unique())
    tick_pos = list(range(0, len(sorted_quarters), 4))

    n_vars = len(key_vars)
    n_cols = min(3, n_vars)
    n_rows = math.ceil(n_vars / n_cols)

    modes = [
        ("mean_std",     "actual_mean",   "actual_std",    "predicted_mean",   "predicted_std",    "Mean ± Std"),
        ("median_iqr",   "actual_median", "actual_halfiqr","predicted_median",  "predicted_halfiqr","Median ± Half-IQR"),
    ]

    lbl = f" ({split_label})" if split_label else ""
    artifact_base = f"aggregate_timeseries_{split_label}" if split_label else "aggregate_timeseries"

    for mode_key, act_center, act_spread, pred_center, pred_spread, mode_label in modes:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

        for idx, var in enumerate(key_vars):
            ax = axes[idx // n_cols][idx % n_cols]
            var_df = (
                df[df["feature"] == var]
                .set_index("quarter")
                .reindex(sorted_quarters)
            )
            x = range(len(var_df))

            ac = var_df[act_center]
            as_ = var_df[act_spread].fillna(0)
            pc = var_df[pred_center]
            ps = var_df[pred_spread].fillna(0)

            ax.plot(x, ac, label="Actual",   color="#1f77b4", linewidth=1.8)
            ax.fill_between(x, ac - as_, ac + as_, alpha=0.12, color="#1f77b4")
            ax.plot(x, pc, label="Forecast", color="#ff7f0e", linewidth=1.8, linestyle="--")
            ax.fill_between(x, pc - ps, pc + ps, alpha=0.12, color="#ff7f0e")

            display_name = var.replace("_DIFF_Y", "").replace("_DIFF_Q", "")
            ax.set_title(display_name, fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([sorted_quarters[t] for t in tick_pos], rotation=45, fontsize=7)

        for idx in range(n_vars, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        plt.suptitle(
            f"Cross-sectional Forecast vs Actual – {mode_label}{lbl} – {logger.run_name}",
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        logger.log_figure(fig, ArtifactGroup.FORECAST, f"{artifact_base}_{mode_key}")

    logger.log_table(df, ArtifactGroup.FORECAST, artifact_base)