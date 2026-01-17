import os
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from services.utils import create_sequences_for_forecast, get_fae_metrics_and_embeddings, analyze_by_quantile, prepare_forecasting_data
from services.config import FORECAST_LEN, MACRO_COLUMNS, SEQ_LEN, TICKERS
from services.paths import IN_DIR

class ArtifactsManager:
    """Handles all post-training artifact generation, plotting, and MLflow logging."""
    
    def __init__(self, model, X_fin, X_macro, Y_fin, args):
        self.model = model
        self.X_fin = X_fin
        self.X_macro = X_macro
        self.Y_fin = Y_fin
        self.args = args

    def analyze_and_log_quantiles(self):
        """Calculates error quantiles, generates the plot, and logs it to MLflow."""
        
        all_errors = get_fae_metrics_and_embeddings(
            self.model, self.X_fin, self.X_macro, self.Y_fin
        )
        quantile_labels, quantile_means = analyze_by_quantile(all_errors, num_quantiles=30)
        
        plt.figure(figsize=(10, 5))
        plt.bar(quantile_labels, quantile_means)
        plt.title(f"{self.args.model_name} Error by Quantile")
        
        plot_filename = "quantile_error.png"
        plt.savefig(plot_filename) 
        mlflow.log_artifact(plot_filename)
        plt.close()
        
    def register_final_model(self):
        """Registers the final PyTorch model structure and weights with MLflow."""
        mlflow.pytorch.log_model(
            pytorch_model=self.model, 
            artifact_path="model", 
            registered_model_name=f"FAE_{self.args.model_name}"
        )
        
    def test_performance(self):
        """
        Loads out-of-sample (OOS) data, processes it, and evaluates performance.
        Logs OOS metrics and a comparison plot to MLflow.
        """
        print(f"\n--- Starting Out-of-Sample Performance Test for {self.args.model_name} ---")
        
        # 1. Load Test Parquets
        try:
            bs_test = pd.read_parquet(os.path.join(IN_DIR, "bs_pct_test.parquet"))
            ins_test = pd.read_parquet(os.path.join(IN_DIR, "ins_pct_test.parquet"))
            cf_test = pd.read_parquet(os.path.join(IN_DIR, "cf_pct_test.parquet"))
            exog = pd.read_parquet(os.path.join(IN_DIR, "exog.parquet"))
        except Exception as e:
            print(f"CRITICAL: Test data loading failed: {e}")
            return

        X_fin_t, Y_fin_t, X_macro_t = prepare_forecasting_data(
            bs_test, ins_test, cf_test, exog, 
            tickers_to_exclude=TICKERS, 
            seq_len=SEQ_LEN, 
            forecast_len=FORECAST_LEN, 
            macro_cols=MACRO_COLUMNS
        )

        # Convert to Tensors
        X_fin_test_t = torch.tensor(X_fin_t, dtype=torch.float32)
        X_macro_test_t = torch.tensor(X_macro_t, dtype=torch.float32)
        Y_fin_test_t = torch.tensor(Y_fin_t, dtype=torch.float32)

        # 3. Get Metrics
        oos_errors = get_fae_metrics_and_embeddings(
            self.model, X_fin_test_t, X_macro_test_t, Y_fin_test_t
        )
        oos_mse = oos_errors.mean()
        
        # 4. Log to MLflow
        mlflow.log_metric("oos_mse_loss", oos_mse)
        print(f"OUT-OF-SAMPLE MSE: {oos_mse:.6f}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(oos_errors, bins=50, alpha=0.5, label='Test (Out-of-Sample)', density=True, color='red')
        plt.title(f"Error Distribution Test: {self.args.model_name}")
        plt.xlabel("MSE")
        plt.legend()
        
        compare_plot = "test_error.png"
        plt.savefig(compare_plot)
        mlflow.log_artifact(compare_plot)
        plt.close()
    
    def run_all(self):
        """Executes all artifact generation and registration steps."""
        self.analyze_and_log_quantiles()
        self.test_performance()
        self.register_final_model()