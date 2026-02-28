# --- utils.py (CRITICAL FUNCTIONS) ---
import os
import random
import numpy as np
import pandas as pd
from sklearn.isotonic import spearmanr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import mlflow # Required for logging
from models import SpecializedHyperbolicFAE
from services.config import FORECAST_LEN, SEQ_LEN # Required for isinstance check
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import umap


# Define missing globals used in the functions (if not passed as args)
# NOTE: BATCH_SIZE must be available globally or passed if used inside get_fae_metrics_and_embeddings
BATCH_SIZE = 32
SIZE_METRIC_COLUMN = 'Total Revenue'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def create_sequences_for_forecast(df: pd.DataFrame, entity_col="ticker", time_col="quarter", features=None, seq_len=4, forecast_len=4):
#     """Creates input sequences, target sequences, and metadata."""
#     # ... (body of create_sequences_for_forecast remains the same)
#     # 
#     input_sequences = []; target_sequences = []; metadata = []
#     # Simplified logic for brevity:
#     # (function body must be the full one you provided previously)
#     return np.array(input_sequences), np.array(target_sequences), pd.DataFrame(metadata)


def apply_time_mask(x, mask_ratio=0.25):
    """
    Randomly masks chunks of the time series to force the model 
    to learn temporal correlations.
    """
    x_masked = x.clone()
    batch_size, seq_len, dim = x.shape
    
    # Create a random binary mask
    mask = torch.rand(batch_size, seq_len, device=x.device) > mask_ratio
    # Expand mask to match feature dimension
    mask = mask.unsqueeze(-1).expand_as(x)
    
    return x_masked * mask, mask

def evaluate_oos(model, X_fin, X_macro, label, device=DEVICE):

    print(f"[INFO] Running OOS evaluation ({label})")

    model.eval().to(device)

    X_fin = X_fin.to(device)
    X_macro = X_macro.to(device)

    with torch.no_grad():

        _, x_hat = model(X_fin, X_macro)

        mse = F.mse_loss(x_hat, X_fin).item()
        mae = F.l1_loss(x_hat, X_fin).item()
        smooth = F.smooth_l1_loss(x_hat, X_fin).item()

    print(f"[INFO] OOS {label} MSE={mse:.6f} MAE={mae:.6f}")

    return {
        "mse": mse,
        "mae": mae,
        "smooth": smooth
    }

def bert_like_train(model, X_fin, X_macro, epochs, batch_size, lr, mask_ratio=0.3, num_augmentations=5):
    """
    PHASE 1: Self-Supervised Learning. 
    Reconstructs X_fin from a corrupted version of itself.
    """
    print(f"--- Starting Pre-training (Masked Reconstruction) ---")
    print(f"Augmentations per Epoch: {num_augmentations} | Mask Ratio: {mask_ratio}")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(X_fin, X_macro)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_recon_loss = 0
        for _ in range(num_augmentations):
            for x_fin_batch, x_macro_batch in loader:
                x_fin_batch = x_fin_batch.to(DEVICE)
                x_macro_batch = x_macro_batch.to(DEVICE)
                
                # 1. Corrupt the input
                x_masked, mask = apply_time_mask(x_fin_batch, mask_ratio=mask_ratio)
                
                optimizer.zero_grad()
                
                # 2. Forward pass (Model must output reconstruction of shape [B, SEQ_LEN * DIM])
                # If your model only forecasts future, you may need a separate reconstruction head.
                recon_flat, _, _ = model(x_masked, x_macro_batch)
                
                # 3. Calculate loss ONLY on the masked values (standard MAE practice)
                target_truncated = x_fin_batch[:, :FORECAST_LEN, :].reshape(x_fin_batch.size(0), -1)
                loss = criterion(recon_flat, target_truncated)
                
                loss.backward()
                optimizer.step()
                total_recon_loss += loss.item()
            
        avg_loss = total_recon_loss / (len(loader) * num_augmentations)
        mlflow.log_metric("pretrain_recon_loss", avg_loss, step=epoch)
        print(f"Pre-train Epoch {epoch+1}, Recon Loss: {avg_loss:.6f}")
        
    return avg_loss

def log_macro_sensitivity_barplot(
    model,
    X_fin,
    X_macro,
    macro_cols,
    run_name,
    device=DEVICE
):

    print("[INFO] Macro sensitivity barplot (cosine-based)")

    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import mlflow

    model.eval().to(device)

    X_fin = X_fin.to(device)
    X_macro = X_macro.to(device)

    # baseline embedding
    with torch.no_grad():
        z_base, _ = model(X_fin, X_macro)

    sensitivities = []

    # permutation importance using cosine similarity
    for m in range(X_macro.shape[2]):

        xm_pert = X_macro.clone()

        perm = torch.randperm(X_macro.shape[0])
        xm_pert[:, :, m] = xm_pert[perm, :, m]

        with torch.no_grad():
            z_pert, _ = model(X_fin, xm_pert)

        # cosine similarity per company
        cos_sim = F.cosine_similarity(z_base, z_pert, dim=1)

        # convert to sensitivity (higher = more impact)
        sensitivity = (1 - cos_sim).mean().item()

        sensitivities.append(sensitivity)

    sensitivities = np.array(sensitivities)

    # sort descending
    order = np.argsort(-sensitivities)
    sensitivities_sorted = sensitivities[order]
    macro_cols_sorted = np.array(macro_cols)[order]

    # plot
    plt.figure(figsize=(10, 6))

    plt.barh(macro_cols_sorted, sensitivities_sorted)

    plt.xlabel("1 − cosine similarity (embedding sensitivity)")
    plt.title("Macro sensitivity (cosine permutation importance)")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    path = f"plots/macro_sensitivity_{run_name}.png"

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(path)

def log_variance_analysis(
    model,
    X_fin,
    X_macro,
    run_name,
    device=DEVICE
):

    print("[INFO] Variance analysis (latent z and reconstruction x_hat)")

    model.eval().to(device)

    with torch.no_grad():
        z, x_hat = model(X_fin.to(device), X_macro.to(device))

    z = z.cpu().numpy()
    x_hat = x_hat.cpu().numpy().reshape(x_hat.shape[0], -1)

    fin = X_fin.cpu().numpy().reshape(X_fin.shape[0], -1)
    macro = X_macro.cpu().numpy().reshape(X_macro.shape[0], -1)

    # -----------------
    # LATENT PROBES
    # -----------------

    reg_macro_z = LinearRegression().fit(z, macro)
    macro_pred_z = reg_macro_z.predict(z)
    r2_macro_z = r2_score(macro, macro_pred_z)

    reg_fin_z = LinearRegression().fit(z, fin)
    fin_pred_z = reg_fin_z.predict(z)
    r2_fin_z = r2_score(fin, fin_pred_z)

    # -----------------
    # RECONSTRUCTION PROBES
    # -----------------

    reg_macro_xhat = LinearRegression().fit(x_hat, macro)
    macro_pred_xhat = reg_macro_xhat.predict(x_hat)
    r2_macro_xhat = r2_score(macro, macro_pred_xhat)

    reg_fin_xhat = LinearRegression().fit(x_hat, fin)
    fin_pred_xhat = reg_fin_xhat.predict(x_hat)
    r2_fin_xhat = r2_score(fin, fin_pred_xhat)

    # -----------------
    # LOGGING
    # -----------------

    mlflow.log_metric("macro_linear_probe_r2_latent", r2_macro_z)
    mlflow.log_metric("financial_linear_probe_r2_latent", r2_fin_z)

    mlflow.log_metric("macro_linear_probe_r2_recon", r2_macro_xhat)
    mlflow.log_metric("financial_linear_probe_r2_recon", r2_fin_xhat)

    # -----------------
    # PRINT
    # -----------------

    print(f"Macro R2 - Latent (z): {r2_macro_z:.4f}")
    print(f"Financial R2 - Latent (z): {r2_fin_z:.4f}")

    print(f"Macro R2 - Reconstruction (x_hat): {r2_macro_xhat:.4f}")
    print(f"Financial R2 - Reconstruction (x_hat): {r2_fin_xhat:.4f}")

    return pd.DataFrame(data={
        "latent": {
            "macro_r2": r2_macro_z,
            "financial_r2": r2_fin_z
        },
        "reconstruction": {
            "macro_r2": r2_macro_xhat,
            "financial_r2": r2_fin_xhat
        }
    }).round(3)
    
    
def log_latent_projection(
    model,
    X_fin,
    company_names,
    run_name,
    device=DEVICE
):

    print("[INFO] Computing PCA and UMAP")

    model.eval().to(device)

    with torch.no_grad():
        z, _ = model(X_fin.to(device), torch.zeros_like(X_fin[:,:,:1]).to(device))

    z = z.cpu().numpy()

    projections = {
        "pca": PCA(n_components=2).fit_transform(z),
        "umap": umap.UMAP().fit_transform(z)
    }

    for name, proj in projections.items():

        plt.figure(figsize=(8,6))

        plt.scatter(proj[:,0], proj[:,1])

        for i, txt in enumerate(company_names):
            plt.annotate(txt, (proj[i,0], proj[i,1]), fontsize=6)

        plt.title(f"{name.upper()} projection")

        path = f"plots/{name}_latent_{run_name}.png"
        plt.savefig(path)
        plt.close()

        mlflow.log_artifact(path)

def log_distance_preservation(
    model,
    X_fin,
    run_name,
    device=DEVICE
):
    print("[INFO] Computing distance preservation")

    model.eval().to(device)

    with torch.no_grad():
        z, _ = model(X_fin.to(device), torch.zeros_like(X_fin[:,:,:1]).to(device))

    z = z.cpu().numpy()
    fin = X_fin.cpu().numpy().reshape(X_fin.shape[0], -1)

    d_fin = euclidean_distances(fin)
    d_lat = euclidean_distances(z)

    idx = np.triu_indices_from(d_fin, k=1)

    corr, _ = pearsonr(d_fin[idx], d_lat[idx])

    mlflow.log_metric("distance_preservation_corr", corr)

    plt.figure()
    plt.scatter(d_fin[idx], d_lat[idx], alpha=0.3)
    plt.xlabel("Financial distance")
    plt.ylabel("Latent distance")
    plt.title(f"Distance preservation corr={corr:.3f}")

    path = f"plots/distance_preservation_{run_name}.png"
    plt.savefig(path)
    plt.close()

    mlflow.log_artifact(path)

        

# --- UPDATED FUNCTION WITH MLFLOW AND TENSORBOARD LOGGING ---
def train_hybrid_fae(model, X_fin_past, X_macro_past, Y_fin_future, epochs, batch_size, lr, lambda_ortho=0.0, writer=None):
    """Trains the FAE model, logging loss per epoch to MLflow and detailed stats to TensorBoard."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_fin_past, X_macro_past, Y_fin_future)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    global_step = 0 # Initialize a step counter for TensorBoard/batch logging
    
    print(f"Starting FAE training on {len(dataset)} sequences...")
    for epoch in range(epochs):
        model.train(); total_loss = 0
        for batch_idx, (x_fin_batch, x_macro_batch, y_fin_batch) in enumerate(loader):
            x_fin_batch = x_fin_batch.to(DEVICE)
            x_macro_batch = x_macro_batch.to(DEVICE)
            y_fin_batch = y_fin_batch.to(DEVICE)
            y_fin_flat_future = y_fin_batch.view(y_fin_batch.size(0), -1) 
            optimizer.zero_grad()
            forecast_flat, z_fin_e, z_macro_e = model(x_fin_batch, x_macro_batch)
            
            loss_forecast = criterion(forecast_flat, y_fin_flat_future)
            
            # Ortho loss calculation remains the same
            if lambda_ortho > 0.0 and isinstance(model, SpecializedHyperbolicFAE):
                cos_sim = F.cosine_similarity(z_fin_e, z_macro_e, dim=1)
                loss_ortho = lambda_ortho * torch.mean(cos_sim**2) 
            else:
                loss_ortho = torch.tensor(0.0).to(DEVICE)
                
            loss = loss_forecast + loss_ortho
            
            loss.backward()
            
            # --- TENSORBOARD LOGGING (More Robust) ---
            if writer and global_step % 100 == 0: 
                writer.add_scalar('Loss/train_batch', loss_forecast.item(), global_step)
                
                # Iterate only over parameters that require gradients
                for name, param in model.named_parameters():
                    # Check if the parameter has a gradient AND matches your naming filter
                    if param.grad is not None and ('fin_encoder' in name or 'decoder' in name):
                        writer.add_histogram(f'{name}/weights', param.data, global_step)
                        writer.add_histogram(f'{name}/gradients', param.grad.data, global_step)
            # ---------------------------

            optimizer.step()
            total_loss += loss_forecast.item() * x_fin_batch.size(0) 
            global_step += 1 # Increment step counter
            
        avg_loss = total_loss / len(dataset)
        
        # --- MLFLOW LOGGING: Epoch loss ---
        mlflow.log_metric("epoch_avg_loss", avg_loss, step=epoch)
        # ----------------------------------
        
        print(f"Epoch {epoch+1:02d}, Avg Forecast Loss: {avg_loss:.6f} (Ortho Loss: {loss_ortho.item():.9f})")
    
    return avg_loss


import matplotlib.pyplot as plt

def plot_losses(epoch_metrics_contextual, epoch_metrics_blind, latent_dim, save_dir="plots"):
    """
    Plots MSE, MAE, and SmoothL1 losses for contextual vs blind macro models
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = [m["epoch"]+1 for m in epoch_metrics_contextual]

    # Extract losses
    mse_ctx = [m["mse"] for m in epoch_metrics_contextual]
    mse_blind = [m["mse"] for m in epoch_metrics_blind]
    mae_ctx = [m["mae"] for m in epoch_metrics_contextual]
    mae_blind = [m["mae"] for m in epoch_metrics_blind]
    smooth_ctx = [m["smooth"] for m in epoch_metrics_contextual]
    smooth_blind = [m["smooth"] for m in epoch_metrics_blind]

    plt.figure(figsize=(12, 4))

    # MSE
    plt.subplot(1, 3, 1)
    plt.plot(epochs, mse_ctx, label="Contextual")
    plt.plot(epochs, mse_blind, label="Blind")
    plt.title(f"MSE Loss (latent_dim={latent_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    # MAE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, mae_ctx, label="Contextual")
    plt.plot(epochs, mae_blind, label="Blind")
    plt.title(f"MAE Loss (latent_dim={latent_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()

    # SmoothL1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, smooth_ctx, label="Contextual")
    plt.plot(epochs, smooth_blind, label="Blind")
    plt.title(f"SmoothL1 Loss (latent_dim={latent_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"loss_comparison_latent{latent_dim}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Saved loss comparison plot at {plot_path}")
    return plot_path


def create_aligned_dataset(bs_df, is_df, cf_df, macro_df, seq_len=SEQ_LEN):
    # Merge all dataframes on ticker and quarter where applicable
    df = bs_df.merge(is_df, on=['ticker','quarter']).merge(cf_df, on=['ticker','quarter']).merge(macro_df, on='quarter')
    df = df.sort_values(['ticker','quarter'])
    
    # Identify column groups
    macro_cols = [c for c in macro_df.columns if c != 'quarter']
    fin_cols = [c for c in df.columns if c not in ['ticker','quarter'] + macro_cols]
    
    X_fin, X_macro, Y_fin, meta = [], [], [], []
    
    for ticker, group in df.groupby('ticker'):
        if len(group) < seq_len:
            continue
            
        f_vals = group[fin_cols].values
        m_vals = group[macro_cols].values
        q_vals = group['quarter'].values
        
        # Sliding window for features
        # [Num_Windows, Seq_Len, Features]
        curr_x_fin = np.lib.stride_tricks.sliding_window_view(f_vals, (seq_len, len(fin_cols))).squeeze(1)
        curr_x_mac = np.lib.stride_tricks.sliding_window_view(m_vals, (seq_len, len(macro_cols))).squeeze(1)
        
        X_fin.append(curr_x_fin)
        X_macro.append(curr_x_mac)
        
        # For an Autoencoder, Y is the reconstruction of the input financials
        Y_fin.append(curr_x_fin) 
        
        meta.extend([{'ticker':ticker, 'end_quarter':q_vals[i+seq_len-1]} for i in range(len(group)-seq_len+1)])
    
    # Concatenate all tickers into single tensors
    X_fin = torch.tensor(np.concatenate(X_fin), dtype=torch.float32)
    X_macro = torch.tensor(np.concatenate(X_macro), dtype=torch.float32)
    Y_fin = torch.tensor(np.concatenate(Y_fin), dtype=torch.float32)
    
    meta_df = pd.DataFrame(meta)
    
    return X_fin, X_macro, Y_fin, meta_df, len(fin_cols), len(macro_cols), fin_cols


def log_company_distance_heatmaps(model, X_fin, X_macro, tickers, run_name, device=DEVICE):

    model.eval().to(device)

    with torch.no_grad():
        z, _ = model(
            X_fin.to(device),
            torch.zeros_like(X_macro).to(device)  # correct
        )

    z = z.cpu().numpy()

    # flatten financials per company
    fin = X_fin.cpu().numpy().reshape(X_fin.shape[0], -1)

    cos_fin = cosine_similarity(fin)
    cos_lat = cosine_similarity(z)

    euc_fin = euclidean_distances(fin)
    euc_lat = euclidean_distances(z)

    
    def flatten_upper(mat):
        return mat[np.triu_indices_from(mat, k=1)]
    
    # flatten
    cos_fin_flat = flatten_upper(cos_fin)
    cos_lat_flat = flatten_upper(cos_lat)

    euc_fin_flat = flatten_upper(euc_fin)
    euc_lat_flat = flatten_upper(euc_lat)
   
    # Compute correlations
    cos_corr, _ = spearmanr(cos_fin_flat, cos_lat_flat)
    euc_corr, _ = spearmanr(euc_fin_flat, euc_lat_flat)

    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    # ----- Cosine -----
    hb0 = axes[0].hexbin(
        cos_fin_flat, cos_lat_flat, gridsize=80, cmap="Blues", mincnt=1, bins='log'
    )
    axes[0].set_xlabel("Cosine Similarity Financial")
    axes[0].set_ylabel("Cosine Similarity Latent")
    axes[0].set_title(f"Cosine Similarity Comparison\nSpearman={cos_corr:.3f}")
    plt.colorbar(hb0, ax=axes[0], label='log(Counts)')

    # ----- Euclidean -----
    hb1 = axes[1].hexbin(
        euc_fin_flat, euc_lat_flat, gridsize=80, cmap="Oranges", mincnt=1, bins='log'
    )
    axes[1].set_xlabel("Euclidean Distance Financial")
    axes[1].set_ylabel("Euclidean Distance Latent")
    axes[1].set_title(f"Euclidean Distance Comparison\nSpearman={euc_corr:.3f}")
    plt.colorbar(hb1, ax=axes[1], label='log(Counts)')

    plt.tight_layout()
    fig_path = f"plots/scatter_cosine_eucl_fin_vs_latent_{run_name}.png"
    plt.savefig(fig_path)
    plt.close()
    mlflow.log_artifact(fig_path)

def filter_columns(bs_df, is_df, cf_df, macro_df, cond="DIFF_Y"):
    def keep(df):
        cols = [c for c in df.columns if cond in c]
        cols = [c for c in cols if c not in [f"Other Non-Current Assets_{cond}", f"Other Non-Current Liabilities_{cond}"]]
        if "ticker" in df.columns:
            return df[["ticker", "quarter"] + cols]
        else:
            return df[["quarter"] + cols]

    return keep(bs_df), keep(is_df), keep(cf_df), keep(macro_df)

class LatentWrapper(torch.nn.Module):

    def __init__(self, model, mode="latent"):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, X_fin, X_macro):

        z, x_hat = self.model(X_fin, X_macro)

        if self.mode == "latent":

            # attribution of representation structure
            return torch.norm(z, dim=1)

        elif self.mode == "reconstruction":

            # attribution of reconstruction mechanics
            loss = ((x_hat - X_fin) ** 2).mean(dim=(1,2))
            return loss

        else:
            raise ValueError("mode must be latent or reconstruction")
import seaborn as sns

def compute_full_saliency(
    model,
    X_fin,
    X_macro,
    fin_cols,
    macro_cols,
    meta_df,
    metadata_sector_df,
    run_name,
    device=DEVICE,
    top_n=30
):
    from captum.attr import IntegratedGradients

    print(f"[INFO] Computing saliency (both modes) – {run_name}")

    model.eval().to(device)
    xf = X_fin.to(device)
    xm = X_macro.to(device)
    baseline_fin = torch.zeros_like(xf)
    baseline_macro = torch.zeros_like(xm)

    results = {}

    for mode in ["latent", "reconstruction"]:

        wrapper = LatentWrapper(model, mode=mode)
        ig = IntegratedGradients(wrapper)

        attr_fin, attr_macro = ig.attribute(
            inputs=(xf, xm),
            baselines=(baseline_fin, baseline_macro),
            n_steps=50
        )

        # ---------------- GLOBAL FEATURE SALIENCY ----------------

        attr_fin_global = attr_fin.abs().mean(dim=(0, 1)).cpu().numpy()
        attr_macro_global = attr_macro.abs().mean(dim=(0, 1)).cpu().numpy()

        total_fin = attr_fin_global.sum()
        total_macro = attr_macro_global.sum()
        macro_ratio = total_macro / (total_fin + total_macro + 1e-9)

        mlflow.log_metric(f"{mode}_macro_ratio", macro_ratio)

        df_global = pd.concat([
            pd.DataFrame({"feature": fin_cols, "saliency": attr_fin_global, "type": "financial"}),
            pd.DataFrame({"feature": macro_cols, "saliency": attr_macro_global, "type": "macro"})
        ]).sort_values("saliency", ascending=False)

        csv_path = f"tables/full_saliency_{mode}_{run_name}.csv"
        df_global.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        # ----------- Top N features plot -----------
        df_sorted = df_global.head(top_n)
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        sns.barplot(
            data=df_sorted,
            x="saliency",
            y="feature",
            hue="type",
            palette={"financial": "#1f77b4", "macro": "#ff7f0e"},
            dodge=False
        )
        plt.title(f"Top {top_n} Feature Saliency ({mode}) – {run_name}")
        fig_path = f"plots/full_feat_saliency_{mode}_{run_name}.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path)

        # ---------------- SECTOR EXPOSURE ----------------

        attr_macro_sample = attr_macro.abs().mean(dim=1).cpu().numpy()  # (N, M)
        macro_exposure_per_sample = attr_macro_sample.sum(axis=1)

        df_sector = pd.DataFrame({
            "ticker": meta_df["ticker"].values,
            "macro_exposure": macro_exposure_per_sample
        })
        df_sector = df_sector.merge(metadata_sector_df, on="ticker", how="left")
        df_sector = (
            df_sector
            .groupby("sector")["macro_exposure"]
            .mean()
            .sort_values(ascending=False)
            .to_frame()
        )

        csv_sector = f"tables/sector_macro_exposure_{mode}_{run_name}.csv"
        df_sector.to_csv(csv_sector)
        mlflow.log_artifact(csv_sector)

        # -------- TOP / BOTTOM 15 --------
        for label, subset, invert in [
            ("top15", df_sector.head(15), True),
            ("bottom15", df_sector.tail(15).sort_values("macro_exposure"), False)
        ]:
            plt.figure(figsize=(8, 6))
            data = subset["macro_exposure"]
            index = subset.index
            if invert:
                plt.barh(index[::-1], data[::-1])
            else:
                plt.barh(index, data)
                plt.gca().invert_yaxis()
            plt.title(f"{label.replace('15', ' 15').title()} Macro-Exposed Sectors – {mode}")
            plt.tight_layout()
            fig_path = f"plots/{label}_macro_sectors_{mode}_{run_name}.png"
            plt.savefig(fig_path)
            plt.close()
            mlflow.log_artifact(fig_path)

        results[mode] = {
            "df_global": df_global,
            "df_sector": df_sector,
            "macro_ratio": macro_ratio
        }

    # ---------------- COMPANIES PER SECTOR BAR ----------------

    n_companies = (
        metadata_sector_df
        .groupby("sector")["ticker"]
        .nunique()
        .sort_values(ascending=False)
        .rename("n_companies")
        .reset_index()
    )

    csv_path = f"tables/companies_per_sector_{run_name}.csv"
    n_companies.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    # ---------------- BUBBLE CHART ----------------

    df_latent = results["latent"]["df_sector"].rename(columns={"macro_exposure": "latent_exposure"})
    df_recon = results["reconstruction"]["df_sector"].rename(columns={"macro_exposure": "recon_exposure"})

    df_bubble = df_latent.join(df_recon, how="inner")
    df_bubble["n_companies"] = df_bubble.index.map(
        metadata_sector_df.groupby("sector")["ticker"].nunique()
    )
    df_bubble = df_bubble.dropna()

    # Keep only top 15 by average exposure across both modes
    df_bubble["avg_exposure"] = (df_bubble["recon_exposure"])
    df_bubble = df_bubble.nlargest(10, "avg_exposure").drop(columns="avg_exposure")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        df_bubble["latent_exposure"],
        df_bubble["recon_exposure"],
        s=df_bubble["n_companies"] * 20,
        alpha=0.6,
        color="#1f77b4",
        edgecolors="white",
        linewidths=0.5
    )

    for sector, row in df_bubble.iterrows():
        ax.annotate(
            sector,
            (row["latent_exposure"], row["recon_exposure"]),
            fontsize=7,
            alpha=0.8,
            xytext=(4, 4),
            textcoords="offset points"
        )

    ax.set_xlabel("Latent Macro Exposure")
    ax.set_ylabel("Reconstruction Macro Exposure")
    ax.set_title(f"Top 10 Sector Macro Exposure (RECON) – Latent vs Reconstruction\n(bubble size = n companies) – {run_name}")
    plt.tight_layout()

    fig_path = f"plots/bubble_sector_exposure_{run_name}_recon.png"
    plt.savefig(fig_path)
    plt.close()
    mlflow.log_artifact(fig_path)

    print(f"[INFO] Latent macro ratio:         {results['latent']['macro_ratio']:.4f}")
    print(f"[INFO] Reconstruction macro ratio: {results['reconstruction']['macro_ratio']:.4f}")

    # ---------------- BUBBLE CHART ----------------

    df_latent = results["latent"]["df_sector"].rename(columns={"macro_exposure": "latent_exposure"})
    df_recon = results["reconstruction"]["df_sector"].rename(columns={"macro_exposure": "recon_exposure"})

    df_bubble = df_latent.join(df_recon, how="inner")
    df_bubble["n_companies"] = df_bubble.index.map(
        metadata_sector_df.groupby("sector")["ticker"].nunique()
    )
    df_bubble = df_bubble.dropna()

    # Keep only top 15 by average exposure across both modes
    df_bubble["avg_exposure"] = (df_bubble["latent_exposure"]) 
    df_bubble = df_bubble.nlargest(10, "avg_exposure").drop(columns="avg_exposure")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        df_bubble["latent_exposure"],
        df_bubble["recon_exposure"],
        s=df_bubble["n_companies"] * 20,
        alpha=0.6,
        color="#1f77b4",
        edgecolors="white",
        linewidths=0.5
    )

    for sector, row in df_bubble.iterrows():
        ax.annotate(
            sector,
            (row["latent_exposure"], row["recon_exposure"]),
            fontsize=7,
            alpha=0.8,
            xytext=(4, 4),
            textcoords="offset points"
        )

    ax.set_xlabel("Latent Macro Exposure")
    ax.set_ylabel("Reconstruction Macro Exposure")
    ax.set_title(f"Top 10 Sector Macro Exposure (LATENT) – Latent vs Reconstruction\n(bubble size = n companies) – {run_name}")
    plt.tight_layout()

    fig_path = f"plots/bubble_sector_exposure_{run_name}_latent.png"
    plt.savefig(fig_path)
    plt.close()
    mlflow.log_artifact(fig_path)

    print(f"[INFO] Latent macro ratio:         {results['latent']['macro_ratio']:.4f}")
    print(f"[INFO] Reconstruction macro ratio: {results['reconstruction']['macro_ratio']:.4f}")
    
    return (
        results["latent"]["df_global"],
        results["reconstruction"]["df_global"],
        results["latent"]["df_sector"],
        results["reconstruction"]["df_sector"],
        results["latent"]["macro_ratio"],
        results["reconstruction"]["macro_ratio"],
    )

def seed_everything(seed=42):
# 1. Python & OS
    # Python's built-in random module
    random.seed(seed)
    
    # Numpy's random module
    np.random.seed(seed)
    
    # PyTorch seed for CPU
    torch.manual_seed(seed)
    
    # PyTorch seed for all GPU devices (if using CUDA)
    torch.cuda.manual_seed_all(seed)
    
    # Make sure to disable CuDNN's non-deterministic optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
        

def create_sequences(
    df: pd.DataFrame, 
    entity_col="ticker", 
    time_col="quarter", 
    features=None, 
    seq_len=4, 
    window_len=4
    ):
    """
    Creates input sequences (X_past), target sequences (Y_future), and metadata 
    from the time series DataFrame.
    
    Args:
        df: Merged DataFrame containing financial and macro features.
        entity_col: Column identifying the time series group (e.g., 'ticker').
        time_col: Column identifying the time step (e.g., 'quarter').
        features: List of columns to use as features (e.g., fin_columns or MACRO_COLUMNS).
        seq_len: Length of the input sequence (X_past).
        forecast_len: Length of the target sequence (Y_future).
        
    Returns:
        input_sequences: NumPy array (N, seq_len, F)
        target_sequences: NumPy array (N, forecast_len, F)
        metadata_df: DataFrame with sequence context (ticker, quarter, size).
    """
    # This global dependency is a terrible practice but required to fix your code:
    global SIZE_METRIC_COLUMN 
    
    input_sequences = []
    target_sequences = []
    metadata = []
    
    # Ensure all required columns are present in the subset
    required_cols = list(set(features + [time_col, entity_col, SIZE_METRIC_COLUMN]))
    
    # Filter for tickers that meet the required minimum length, though this
    # should ideally be done before calling the function.
    tickers = df[entity_col].unique()
    required_length = seq_len + window_len
    
    print(f"DEBUG: Starting sequence creation for {len(tickers)} unique entities...")

    for tkr in tickers:
        # Get all data for the current ticker, sorted by time
        sub = df[df[entity_col] == tkr].sort_values(time_col)[required_cols]
        
        # Check 1: Data presence and length
        if len(sub) < required_length:
            continue
            
        # Extract only the feature values as a NumPy array
        vals = sub[features].values
        
        # Check 2: Feature integrity (ensure no all-NaN columns were passed, though input cleanup should handle this)
        if vals.shape[1] == 0:
            print(f"WARNING: Ticker {tkr} has zero features in its subset. Skipping.")
            continue

        # Iterate through the array to create all possible (X, Y) sequence pairs
        for i in range(len(vals) - required_length + 1): 
            seq_input = vals[i : i + seq_len]
            seq_target = vals[i + seq_len : i + required_length]
            
            # Metadata: Context for the sequence (taken from the first row of the forecast window)
            forecast_start_row = sub.iloc[i + seq_len]
            forecast_quarter = forecast_start_row[time_col]
            
            # Handle potential missingness in size metric
            size_metric_value = forecast_start_row.get(SIZE_METRIC_COLUMN, np.nan)

            input_sequences.append(seq_input)
            target_sequences.append(seq_target)
            
            metadata.append({
                'ticker': tkr, 
                'forecast_quarter': forecast_quarter, 
                'size_metric': size_metric_value
            })

    if not input_sequences:
        num_features = len(features) if features else 0
        print(f"\nFATAL WARNING: Returned 0 sequences. Check your data split and minimum length requirements (current is {required_length}).")
        return np.empty((0, seq_len, num_features)), np.empty((0, window_len, num_features)), pd.DataFrame(metadata)
    
    print(f"DEBUG: Sequence creation successful. Final count: {len(input_sequences)}")
    return np.array(input_sequences), np.array(target_sequences), pd.DataFrame(metadata)

def get_fae_metrics_and_embeddings(model, X_fin_past, X_macro_past, Y_fin_future):
    """Calculates per-sequence MSE and returns errors."""
    model.eval(); model = model.to(DEVICE)
    dataset = TensorDataset(X_fin_past, X_macro_past, Y_fin_future)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_errors = []
    
    with torch.no_grad():
        for x_fin_batch, x_macro_batch, y_fin_batch in loader:
            x_fin_batch = x_fin_batch.to(DEVICE); 
            x_macro_batch = x_macro_batch.to(DEVICE)
            y_fin_flat = x_fin_batch.view(x_fin_batch.size(0), -1)
            
            x_masked, _ = apply_time_mask(x_fin_batch, mask_ratio=0.25)
            
            # Model returns: forecast, z_fin_e, z_macro_e
            forecast_flat, _, _ = model(x_masked, x_macro_batch)
            
            # Calculate MSE per sequence
            squared_errors = (forecast_flat - y_fin_flat)**2
            sequence_mse = squared_errors.mean(dim=-1).cpu().numpy()
            
            all_errors.extend(sequence_mse.tolist())
            
    return np.array(all_errors) # We only need the errors for analysis

def analyze_by_quantile(errors, num_quantiles=20):
    """Divides errors into quantiles and computes mean error per quantile."""
    quantiles = np.linspace(0, 100, num_quantiles + 1)[1:]
    thresholds = np.percentile(errors, quantiles)
    quantile_means = []
    for i in range(num_quantiles):
        upper_bound = thresholds[i]
        if i == 0:
            bin_errors = errors[errors <= upper_bound]
        else:
            lower_bound = thresholds[i-1]
            bin_errors = errors[(errors > lower_bound) & (errors <= upper_bound)]
        quantile_means.append(bin_errors.mean() if len(bin_errors) > 0 else 0.0)
    quantile_labels = [f"Q{i+1}" for i in range(num_quantiles)]
    return quantile_labels, quantile_means

def analyze_by_group(df_results, group_col, num_groups=5):
    """Computes mean error per group (e.g., Year or Size Quintile/Decile)."""
    
    # 1. Size Metric Grouping (Decide the granularity here)
    if group_col == 'size_metric':
        # Enforce 10 groups for size analysis, regardless of input
        num_groups = 10 
        group_labels = [f"Size Q{i+1}" for i in range(num_groups)]
        
        # Filter out infinities/NaT before qcut
        valid_df = df_results.replace([np.inf, -np.inf], np.nan).dropna(subset=[group_col])
        
        if len(valid_df) < num_groups:
             # Not enough data for the desired granularity - a data problem, not a code problem
             print(f"WARNING: Not enough unique size data points for {num_groups} quantiles. Using simple rank grouping.")
             valid_df['group'] = pd.cut(valid_df[group_col], bins=num_groups, labels=group_labels, include_lowest=True, duplicates='drop')
        else:
             # Use qcut to create equal-sized quantiles (deciles)
             valid_df['group'] = pd.qcut(valid_df[group_col], q=num_groups, labels=group_labels, duplicates='drop')
             
    # 2. Other Grouping (e.g., Year)
    else:
        valid_df = df_results
        valid_df['group'] = valid_df[group_col]
        group_labels = sorted(valid_df['group'].unique())
    
    # 3. Calculate Means
    grouped_errors = valid_df.groupby('group')['error'].mean().reset_index()
    
    # 4. Reindex and Return (to ensure all labels appear, even if mean is 0)
    result_means = grouped_errors.set_index('group').reindex(group_labels)['error'].fillna(0).tolist()
        
    return group_labels, result_means

def plot_comparison(all_models_means, all_model_names, labels, title, xlabel):
    """Plots the mean error per group/quantile for multiple models."""
    x = np.arange(len(labels))
    width = 0.8 / len(all_models_means)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, means in enumerate(all_models_means):
        ax.bar(x + i * width - 0.4, means, width, label=all_model_names[i])
    
    ax.set_ylabel('Mean Forecasting Error (MSE)')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x + 0.4 - width/2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()

def filter_tickers(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    return df[~df.ticker.isin(tickers)]

def prepare_data(bs, ins, cf, exog, tickers_to_exclude, seq_len, window_len, macro_cols):
    """
    Unified processing pipeline for both Train and Test datasets.
    """
    # 1. Merge
    financials = bs.merge(ins, on=["ticker", "quarter"], how="outer").merge(cf, on=["ticker", "quarter"], how="outer")
    merged = financials.merge(exog, left_on="quarter", right_on="observation_date", how="left")
    
    # 2. Filter & Clean
    merged = merged[~merged.ticker.isin(tickers_to_exclude)]
    
    imputation_cols = [c for c in merged.columns if c not in ['ticker', 'quarter', 'observation_date']]
    merged[imputation_cols] = merged.groupby('ticker')[imputation_cols].ffill() 
    merged.dropna(subset=imputation_cols, inplace=True)
    
    # 3. Sequence Validation
    required_len = seq_len + window_len
    ticker_counts = merged.groupby('ticker')['quarter'].count()
    valid_tickers = ticker_counts[ticker_counts >= required_len].index.tolist()
    merged = merged[merged['ticker'].isin(valid_tickers)]

    # 4. Feature Splitting
    all_features = [c for c in merged.columns if c not in ["ticker", "quarter", "observation_date"]]
    fin_columns = [c for c in all_features if c not in macro_cols]

    # 5. Sequence Generation
    X_fin, Y_fin, _ = create_sequences(merged, features=fin_columns, seq_len=seq_len, window_len=window_len)
    X_macro, _, _ = create_sequences(merged, features=macro_cols, seq_len=seq_len, window_len=window_len)

    return (
        torch.tensor(X_fin, dtype=torch.float32), 
        torch.tensor(Y_fin, dtype=torch.float32), 
        torch.tensor(X_macro, dtype=torch.float32)
    )