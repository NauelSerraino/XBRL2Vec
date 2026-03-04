"""
XBRL2Vec – Contextual vs Blind Autoencoder Experiment
======================================================
MLflow experiment loop over multiple latent dimensions.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import mlflow
import torch

from models.autoencoder_dlinear_conditioner import CompanyEmbeddingAE
from services.config import DEVICE, SEQ_LEN
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

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_factors", nargs="+", type=float, default=[0.5, 0.8, 1, 2])
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--learning_rate",  type=float, default=1e-3)
    parser.add_argument("--mask_prob",      type=float, default=0.2)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--use_mask",       type=int,   default=0)
    return TrainConfig.from_args(parser.parse_args())


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
    run_name = f"latent_dim-{latent_dim}"
    print(f"\n{'='*60}")
    print(f"[INFO] Run: {run_name}  |  latent_factor leads to dim={latent_dim}")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=run_name):
        logger = ArtifactLogger(run_name)
        mlflow.log_params(config.model_dump())
        mlflow.log_param("latent_dim", latent_dim)

        # ----------------------------------------------------------------
        # 1. Distribution diagnostics (data-level, run once per experiment)
        # ----------------------------------------------------------------
        log_correlation_matrix(train_ds.X_fin, train_ds.X_macro, train_ds.fin_cols, train_ds.macro_cols, logger)
        log_financial_boxplots(train_ds.X_fin, train_ds.fin_cols, logger)
        log_macro_boxplots(train_ds.X_macro, train_ds.macro_cols, logger)
        log_zero_sparsity(train_ds.X_fin, train_ds.X_macro, train_ds.fin_cols, train_ds.macro_cols, logger)

        # ----------------------------------------------------------------
        # 2. Train contextual model
        # ----------------------------------------------------------------
        print("[INFO] Training contextual model")
        model_ctx = CompanyEmbeddingAE(SEQ_LEN, train_ds.fin_dim, train_ds.macro_dim, latent_dim)
        trainer_ctx = MaskedAETrainer(config, ModelType.CONTEXTUAL)
        model_ctx, metrics_ctx = trainer_ctx.train(
            model_ctx, train_ds.X_fin, train_ds.X_macro,
            alpha=0.0, repeats=10, device=DEVICE,
        )

        # ----------------------------------------------------------------
        # 3. Train blind model (zeroed macro)
        # ----------------------------------------------------------------
        print("[INFO] Training blind model")
        model_blind = CompanyEmbeddingAE(SEQ_LEN, train_ds.fin_dim, train_ds.macro_dim, latent_dim)
        trainer_blind = MaskedAETrainer(config, ModelType.BLIND)
        model_blind, metrics_blind = trainer_blind.train(
            model_blind, train_ds.X_fin, torch.zeros_like(train_ds.X_macro),
            alpha=0.0, repeats=10, device=DEVICE,
        )

        mlflow.log_metric("final_mse_contextual",  metrics_ctx[-1].mse)
        mlflow.log_metric("final_mse_macro_blind", metrics_blind[-1].mse)

        # ----------------------------------------------------------------
        # 4. Training plots
        # ----------------------------------------------------------------
        log_loss_comparison(metrics_ctx, metrics_blind, logger)

        # ----------------------------------------------------------------
        # 5. Importance matrices
        # ----------------------------------------------------------------
        imp_ctx = compute_importance_matrix(
            model_ctx, train_ds.X_fin, train_ds.X_macro,
            train_ds.fin_cols, train_ds.macro_cols, "contextual",
        )
        imp_blind = compute_importance_matrix(
            model_blind, train_ds.X_fin, torch.zeros_like(train_ds.X_macro),
            train_ds.fin_cols, train_ds.macro_cols, "blind",
        )
        log_importance_matrix(imp_ctx,   "contextual", logger)
        log_importance_matrix(imp_blind, "blind",      logger)
        log_importance_summary(imp_ctx, imp_blind, train_ds.fin_cols, train_ds.macro_cols, logger)

        # ----------------------------------------------------------------
        # 6. Embedding geometry
        # ----------------------------------------------------------------
        log_company_distance_scatter(model_ctx, train_ds.X_fin, train_ds.X_macro, logger)
        log_macro_sensitivity_barplot(model_ctx, train_ds.X_fin, train_ds.X_macro, train_ds.macro_cols, logger)

        r2_df = compute_variance_analysis(model_ctx, train_ds.X_fin, train_ds.X_macro)
        log_variance_analysis_plot(r2_df, logger)
        mlflow.log_table(r2_df.reset_index(), "r2_metrics.json")

        # ----------------------------------------------------------------
        # 7. Tournament: contextual vs blind
        # ----------------------------------------------------------------
        log_macro_embedding_tournament(model_ctx, model_blind, train_ds.X_fin, train_ds.X_macro, logger)

        exposure = compute_macro_exposure(model_ctx, model_blind, train_ds.X_fin, train_ds.X_macro)
        log_macro_exposure_density(exposure.blind_cosine, exposure.contextual_cosine, "cosine", logger)
        log_macro_exposure_density(exposure.blind_l2,     exposure.contextual_l2,     "l2",     logger)

        # ----------------------------------------------------------------
        # 8. Saliency (Integrated Gradients)
        # ----------------------------------------------------------------
        compute_full_saliency(
            model_ctx, train_ds.X_fin, train_ds.X_macro,
            train_ds.fin_cols, train_ds.macro_cols,
            train_ds.meta_df, metadata_sector_df, logger,
        )

        for mode in SaliencyMode:
            compute_saliency_per_company(
                model_ctx, train_ds.X_fin, train_ds.X_macro,
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
        oos_ctx   = evaluate_oos(model_ctx,   test_ds.X_fin, test_ds.X_macro,              "contextual")
        oos_blind = evaluate_oos(model_blind, test_ds.X_fin, torch.zeros_like(test_ds.X_macro), "blind")

        mlflow.log_metrics({
            "oos_mse_contextual": oos_ctx.mse,
            "oos_mae_contextual": oos_ctx.mae,
            "oos_mse_blind":      oos_blind.mse,
            "oos_mae_blind":      oos_blind.mae,
            "oos_macro_gain":     oos_blind.mse - oos_ctx.mse,
        })
        print(f"[INFO] OOS Macro Gain: {oos_blind.mse - oos_ctx.mse:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = parse_args()
    seed_everything(config.seed)

    IN_DIR = Path("/home/nauel/vscode/XBRL2Vec/data/in")

    # ---- Load & transform train data ----
    print("[INFO] Loading train data")
    bs_df, is_df, cf_df, macro_df, metadata_sector_df = load_raw_data(IN_DIR)
    bs_df, is_df, cf_df, macro_df = filter_columns(bs_df, is_df, cf_df, macro_df)

    print("[INFO] Building aligned dataset")
    raw_train_ds = create_aligned_dataset(bs_df, is_df, cf_df, macro_df)
    train_ds     = transform_dataset(raw_train_ds)

    # ---- Load & transform test data ----
    print("[INFO] Loading test data")
    bs_test, is_test, cf_test = load_test_data(IN_DIR, macro_df)
    raw_test_ds = create_aligned_dataset(bs_test, is_test, cf_test, macro_df)
    test_ds     = transform_dataset(raw_test_ds)

    # ---- MLflow setup ----
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Macro_vs_Blind_Comparison")

    # ---- Experiment loop ----
    print("[INFO] Starting experiments")
    for factor in config.latent_factors:
        latent_dim = max(1, math.ceil(train_ds.fin_dim * factor))
        run_experiment(config, train_ds, test_ds, metadata_sector_df, latent_dim)

    print("[INFO] All experiments completed!")


if __name__ == "__main__":
    main()