"""
XBRL2Vec – Forecasting Autoencoder Experiment
==============================================
MLflow experiment loop over multiple latent dimensions.
Objective: predict next quarter's financials from T-1 past quarters + macro.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import mlflow
import pandas as pd
import torch

from models.autoencoder_dlinear_forecaster import ForecastingAE
from models.autoencoder_dlinear_blind import FinancialOnlyAE
from services.config import DEVICE
from services.data import (
    AlignedDataset,
    ColumnFilter,
    ModelType,
    TrainConfig,
    create_aligned_dataset,
    filter_columns,
    load_raw_data,
    load_test_data,
)
from services.evaluation import (
    compute_forecast_timeseries,
    compute_importance_matrix,
    compute_macro_exposure,
    compute_variance_analysis,
    evaluate_oos,
)
from services.training import MaskedAETrainer
from services.transforms import transform_dataset
from mlflow_logging import (
    ArtifactGroup,
    ArtifactLogger,
    compute_full_saliency,
    compute_saliency_per_company,
    log_company_distance_scatter,
    log_correlation_matrix,
    log_financial_boxplots,
    log_forecast_aggregate_plot,
    log_importance_matrix,
    log_importance_summary,
    log_loss_comparison,
    log_macro_boxplots,
    log_macro_embedding_tournament,
    log_macro_exposure_density,
    log_macro_sensitivity_barplot,
    log_variance_analysis_plot,
    log_zero_sparsity,
)
from services.data import SaliencyMode


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def create_sliding_windows(
    X_fin: torch.Tensor,    # [N, T_total, F]
    X_macro: torch.Tensor,  # [N, T_total, M]
    T_in: int,
    T_out: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X_win, X_mac_win, Y_win) with leading dim N * n_windows."""
    T_total = X_fin.shape[1]
    n_windows = T_total - T_in - T_out + 1
    if n_windows < 1:
        raise ValueError(
            f"Not enough time steps for sliding window: "
            f"T_total={T_total}, T_in={T_in}, T_out={T_out}"
        )
    X_wins, X_mac_wins, Y_wins = [], [], []
    for w in range(n_windows):
        X_wins.append(X_fin[:, w : w + T_in, :])
        X_mac_wins.append(X_macro[:, w : w + T_in, :])
        Y_wins.append(X_fin[:, w + T_in : w + T_in + T_out, :])
    return (
        torch.cat(X_wins,     dim=0),  # [N * n_windows, T_in,  F]
        torch.cat(X_mac_wins, dim=0),  # [N * n_windows, T_in,  M]
        torch.cat(Y_wins,     dim=0),  # [N * n_windows, T_out, F]
    )


# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_factors", nargs="+", type=float, default=[0.5, 0.8, 1, 2])
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--learning_rate",  type=float, default=1e-3)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--t_in",           type=int,   default=20)
    parser.add_argument("--t_out",          type=int,   default=4)
    parser.add_argument(
        "--norm_mode",
        choices=["global", "per_ticker"],
        default="per_ticker",
        help="Must match the norm_mode used when running preprocess.py.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# One experiment run
# ---------------------------------------------------------------------------

def run_experiment(
    config: TrainConfig,
    train_ds: AlignedDataset,
    test_ds: AlignedDataset,
    metadata_sector_df,
    latent_dim: int,
) -> None:
    t_in, t_out = config.t_in, config.t_out
    run_name = f"tin{t_in}-tout{t_out}-latent{latent_dim}"
    print(f"\n{'='*60}")
    print(f"[INFO] Run: {run_name}  |  latent_factor leads to dim={latent_dim}")
    print(f"{'='*60}")

    # ---- All sliding windows (training + OOS eval) ----
    X_fin_in, X_mac_in, Y_fin = create_sliding_windows(
        train_ds.X_fin, train_ds.X_macro, t_in, t_out
    )
    X_fin_in_t, X_mac_in_t, Y_fin_t = create_sliding_windows(
        test_ds.X_fin, test_ds.X_macro, t_in, t_out
    )

    # ---- Last window per company (diagnostics / geometry / saliency) ----
    X_fin_last  = train_ds.X_fin[:,   -(t_in + t_out):-t_out, :]   # [N, t_in,  F]
    X_mac_last  = train_ds.X_macro[:, -(t_in + t_out):-t_out, :]   # [N, t_in,  M]
    Y_fin_last  = train_ds.X_fin[:,   -t_out:,                :]   # [N, t_out, F]

    with mlflow.start_run(run_name=run_name):
        logger = ArtifactLogger(run_name)
        mlflow.log_params(config.model_dump())
        mlflow.log_param("latent_dim", latent_dim)
        mlflow.log_param("norm_mode", config.norm_mode)

        # ----------------------------------------------------------------
        # 1. Distribution diagnostics  (last window, one row per company)
        # ----------------------------------------------------------------
        log_correlation_matrix(X_fin_last, X_mac_last, train_ds.fin_cols, train_ds.macro_cols, logger)
        log_financial_boxplots(X_fin_last, train_ds.fin_cols, logger)
        log_macro_boxplots(X_mac_last, train_ds.macro_cols, logger)
        log_zero_sparsity(X_fin_last, X_mac_last, train_ds.fin_cols, train_ds.macro_cols, logger)

        # ----------------------------------------------------------------
        # 2. Train contextual model (FiLM-conditioned forecaster)
        # ----------------------------------------------------------------
        print("[INFO] Training contextual model")
        model_ctx = ForecastingAE(t_in, t_out, train_ds.fin_dim, train_ds.macro_dim, latent_dim)
        trainer_ctx = MaskedAETrainer(config, ModelType.CONTEXTUAL)
        model_ctx, metrics_ctx = trainer_ctx.train(
            model_ctx, X_fin_in, X_mac_in, Y_fin=Y_fin, device=DEVICE,
        )

        # ----------------------------------------------------------------
        # 3. Train blind model (financial-only forecaster)
        # ----------------------------------------------------------------
        print("[INFO] Training blind model")
        model_blind = FinancialOnlyAE(t_in, t_out, train_ds.fin_dim, latent_dim)
        trainer_blind = MaskedAETrainer(config, ModelType.BLIND)
        model_blind, metrics_blind = trainer_blind.train(
            model_blind, X_fin_in, Y_fin=Y_fin, device=DEVICE,
        )

        mlflow.log_metric("final_mse_contextual",  metrics_ctx[-1].mse)
        mlflow.log_metric("final_mse_macro_blind", metrics_blind[-1].mse)

        # ----------------------------------------------------------------
        # 4. Training plots
        # ----------------------------------------------------------------
        log_loss_comparison(metrics_ctx, metrics_blind, logger)

        # ----------------------------------------------------------------
        # 5. Importance matrices  (last window, one row per company)
        # ----------------------------------------------------------------
        imp_ctx = compute_importance_matrix(
            model_ctx, X_fin_last, X_mac_last,
            train_ds.fin_cols, train_ds.macro_cols, "contextual", Y_fin=Y_fin_last,
        )
        imp_blind = compute_importance_matrix(
            model_blind, X_fin_last, X_mac_last,
            train_ds.fin_cols, train_ds.macro_cols, "blind", Y_fin=Y_fin_last,
        )
        log_importance_matrix(imp_ctx,   "contextual", logger)
        log_importance_matrix(imp_blind, "blind",      logger)
        log_importance_summary(imp_ctx, imp_blind, train_ds.fin_cols, train_ds.macro_cols, logger)

        # ----------------------------------------------------------------
        # 6. Embedding geometry  (last window, one row per company)
        # ----------------------------------------------------------------
        log_company_distance_scatter(model_ctx, X_fin_last, X_mac_last, logger)
        log_macro_sensitivity_barplot(model_ctx, X_fin_last, X_mac_last, train_ds.macro_cols, logger)

        r2_df = compute_variance_analysis(model_ctx, X_fin_last, X_mac_last)
        log_variance_analysis_plot(r2_df, logger)
        mlflow.log_table(r2_df.reset_index(), "r2_metrics.json")

        # ----------------------------------------------------------------
        # 7. Tournament: contextual vs blind  (last window)
        # ----------------------------------------------------------------
        log_macro_embedding_tournament(model_ctx, model_blind, X_fin_last, X_mac_last, logger)

        exposure = compute_macro_exposure(model_ctx, model_blind, X_fin_last, X_mac_last)
        log_macro_exposure_density(exposure.blind_cosine, exposure.contextual_cosine, "cosine", logger)
        log_macro_exposure_density(exposure.blind_l2,     exposure.contextual_l2,     "l2",     logger)

        # ----------------------------------------------------------------
        # 8. Saliency (Integrated Gradients)  (last window, one row per company)
        # ----------------------------------------------------------------
        compute_full_saliency(
            model_ctx, X_fin_last, X_mac_last,
            train_ds.fin_cols, train_ds.macro_cols,
            train_ds.meta_df, metadata_sector_df, logger,
            Y_fin=Y_fin_last,
        )

        for mode in SaliencyMode:
            compute_saliency_per_company(
                model_ctx, X_fin_last, X_mac_last,
                train_ds.fin_cols, train_ds.macro_cols,
                train_ds.meta_df, metadata_sector_df,
                mode, logger,
            )

        # ----------------------------------------------------------------
        # 9. Save model weights
        # ----------------------------------------------------------------
        for label, model in [("ctx", model_ctx), ("blind", model_blind)]:
            path = logger.model_path(label)
            torch.save(model.state_dict(), path)
            logger.log(path)

        # ----------------------------------------------------------------
        # 10. OOS evaluation
        # ----------------------------------------------------------------
        print("[INFO] OOS evaluation")
        oos_ctx   = evaluate_oos(model_ctx,   X_fin_in_t, X_mac_in_t, "contextual", Y_fin=Y_fin_t)
        oos_blind = evaluate_oos(model_blind, X_fin_in_t, X_mac_in_t, "blind",      Y_fin=Y_fin_t)

        mlflow.log_metrics({
            "oos_mse_contextual": oos_ctx.mse,
            "oos_mae_contextual": oos_ctx.mae,
            "oos_mse_blind":      oos_blind.mse,
            "oos_mae_blind":      oos_blind.mae,
            "oos_macro_gain":     oos_blind.mse - oos_ctx.mse,
        })
        print(f"[INFO] OOS Macro Gain: {oos_blind.mse - oos_ctx.mse:.6f}")

        # ----------------------------------------------------------------
        # 11. Metrics summary table (in-sample vs OOS, overfitting check)
        # ----------------------------------------------------------------
        rows = []
        for label, insample, oos in [
            ("contextual", metrics_ctx[-1],   oos_ctx),
            ("blind",      metrics_blind[-1], oos_blind),
        ]:
            rows.append({"model": label, "split": "in_sample",
                         "mse": insample.mse, "mae": insample.mae, "smooth_l1": insample.smooth})
            rows.append({"model": label, "split": "oos",
                         "mse": oos.mse,      "mae": oos.mae,      "smooth_l1": oos.smooth})
            rows.append({"model": label, "split": "oos_minus_insample",
                         "mse": oos.mse - insample.mse,
                         "mae": oos.mae - insample.mae,
                         "smooth_l1": oos.smooth - insample.smooth})
        metrics_df = pd.DataFrame(rows)
        mlflow.log_table(metrics_df, "metrics_summary.json")

        # ----------------------------------------------------------------
        # 12. Forecast aggregate timeseries (cross-sectional mean ± std)
        # ----------------------------------------------------------------
        print("[INFO] Forecast aggregate timeseries")
        for ds, split in [(train_ds, "in_sample"), (test_ds, "oos")]:
            ts_df = compute_forecast_timeseries(
                model_ctx,
                ds.X_fin, ds.X_macro, ds.meta_df, ds.fin_cols,
                T_in=t_in, T_out=t_out,
            )
            log_forecast_aggregate_plot(ts_df, logger, split_label=split)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = TrainConfig.from_args(args)
    seed_everything(config.seed)

    BASE_DIR       = Path("/home/nauel/vscode/XBRL2Vec/data")
    META_DIR       = BASE_DIR / "in"
    PREPROCESS_DIR = BASE_DIR / "out" / "preprocess"

    # ---- Load & transform train data ----
    print("[INFO] Loading train data")
    bs_df, is_df, cf_df, macro_df, metadata_sector_df = load_raw_data(
        PREPROCESS_DIR, META_DIR, norm_mode=config.norm_mode
    )
    bs_df, is_df, cf_df, macro_df = filter_columns(bs_df, is_df, cf_df, macro_df)

    print("[INFO] Building aligned dataset")
    raw_train_ds = create_aligned_dataset(bs_df, is_df, cf_df, macro_df)
    train_ds     = transform_dataset(raw_train_ds)

    # ---- Load & transform test data ----
    print("[INFO] Loading test data")
    bs_test, is_test, cf_test = load_test_data(PREPROCESS_DIR, macro_df, norm_mode=config.norm_mode)
    raw_test_ds = create_aligned_dataset(bs_test, is_test, cf_test, macro_df)
    test_ds     = transform_dataset(raw_test_ds)

    # ---- MLflow setup ----
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Forecaster_vs_Blind_Comparison")

    # ---- Experiment loop ----
    print("[INFO] Starting experiments")
    for factor in config.latent_factors:
        latent_dim = max(1, math.ceil(train_ds.fin_dim * factor))
        run_experiment(config, train_ds, test_ds, metadata_sector_df, latent_dim)

    print("[INFO] All experiments completed!")


if __name__ == "__main__":
    main()
