import math
import os
import argparse
import random
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import mlflow

from autoencoder_dlinear_conditioner import CompanyEmbeddingAE
from services.config import DEVICE, MACRO_COLUMNS, SEQ_LEN
from services.utils import filter_columns, create_aligned_dataset, seed_everything

# ----------------------------
# 1. ARGPARSE
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--latent_factors", nargs="+", type=float, default=[0.5, 1, 2, 3])
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--mask_prob", type=float, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_mask", type=int, default=False)
args = parser.parse_args()

def symmetric_log_transform(tensor):
    """
    Applies sign(x) * log(1 + |x|) to a tensor.
    This preserves the sign and squashes extreme outliers.
    """
    return torch.sign(tensor) * torch.log1p(torch.abs(tensor))

# ----------------------------
# 2. SEED EVERYTHING
# ----------------------------
print("[INFO] Seeding everything")
seed_everything(args.seed)

def compute_company_macro_weights(X_fin, X_macro):
    """
    X_fin: [N, T, F]
    X_macro: [N, T, M] or [T, M] broadcasted

    Returns:
        weights: [N, M]
    """
    N, T, F = X_fin.shape
    M = X_macro.shape[-1]

    # Aggregate financial signal per company
    fin_signal = X_fin.mean(dim=2)  # [N, T]

    weights = []

    for i in range(N):
        f = fin_signal[i]
        m = X_macro[i] if X_macro.ndim == 3 else X_macro

        f_center = f - f.mean()
        f_std = f_center.std() + 1e-8

        w_i = []
        for j in range(M):
            m_j = m[:, j]
            m_center = m_j - m_j.mean()
            m_std = m_center.std() + 1e-8

            corr = (f_center * m_center).mean() / (f_std * m_std)
            w_i.append(corr)

        weights.append(torch.tensor(w_i))

    return torch.stack(weights)  # [N, M]

def customize_macro_input(X_macro, weights):
    """
    X_macro: [N, T, M]
    weights: [N, M]

    Returns:
        X_macro_custom: [N, T, M]
    """
    return X_macro * weights.unsqueeze(1)

# ----------------------------
# 3. LOAD DATA
# ----------------------------
print("[INFO] Loading data")
IN_DIR = "/home/nauel/vscode/XBRL2Vec/data/in/"

bs_df = pd.read_parquet(f"{IN_DIR}/bs_pct_train.parquet")
is_df = pd.read_parquet(f"{IN_DIR}/ins_pct_train.parquet")
cf_df = pd.read_parquet(f"{IN_DIR}/cf_pct_train.parquet")
macro_df = pd.read_parquet(f"{IN_DIR}/exog.parquet").rename(columns={"observation_date": "quarter"})

quarters_to_delete = [f"201{j}Q{i}" for j in range(1, 3) for i in range(1, 5)]
bs_df = bs_df[~bs_df.quarter.isin(quarters_to_delete)]
is_df = is_df[~is_df.quarter.isin(quarters_to_delete)]
cf_df = cf_df[~cf_df.quarter.isin(quarters_to_delete)]
macro_df = macro_df[~macro_df.quarter.isin(quarters_to_delete)]

bs_df, is_df, cf_df = filter_columns(bs_df, is_df, cf_df, cond="DIFF_Q")

# ----------------------------
# 4. CREATE ALIGNED DATASET
# ----------------------------
print("[INFO] Creating aligned dataset")
X_fin_raw, X_macro_raw, Y_fin, meta_df, FIN_DIM, MACRO_DIM, fin_cols = create_aligned_dataset(
    bs_df, is_df, cf_df, macro_df
)

X_fin = symmetric_log_transform(X_fin_raw)
X_macro = symmetric_log_transform(X_macro_raw)


print("[INFO] Customizing macro per company")

macro_weights = compute_company_macro_weights(X_fin, X_macro)
X_macro = customize_macro_input(X_macro, macro_weights)

# Inside the mlflow.start_run() block:

# ----------------------------
# 5. MLFLOW SETUP
# ----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Macro_vs_Blind_Comparison")

os.makedirs("plots", exist_ok=True)
os.makedirs("tables", exist_ok=True)
os.makedirs("runs", exist_ok=True)

# ----------------------------
# 6. FUNCTION
# ----------------------------

import pandas as pd
import torch

def inspect_company_ae(
    model,
    X_fin,
    X_macro,
    n_obs=100,
    device=DEVICE
):
    model.eval().to(device)

    xf = X_fin[:n_obs].to(device)
    xm = X_macro[:n_obs].to(device)

    out = {}

    with torch.no_grad():

        # -------- Financial encoder --------
        seas_fin, trend_fin = model.fin_encoder.decomp(xf)
        h_fin_seas = model.fin_encoder.seas_proj(
            seas_fin.permute(0,2,1)
        ).squeeze(-1)
        h_fin_trend = model.fin_encoder.trend_proj(
            trend_fin.permute(0,2,1)
        ).squeeze(-1)
        h_fin = torch.cat([h_fin_seas, h_fin_trend], dim=-1)

        # -------- Macro encoder --------
        seas_macro, trend_macro = model.macro_encoder.decomp(xm)
        h_macro_seas = model.macro_encoder.seas_proj(
            seas_macro.permute(0,2,1)
        ).squeeze(-1)
        h_macro_trend = model.macro_encoder.trend_proj(
            trend_macro.permute(0,2,1)
        ).squeeze(-1)
        h_macro = torch.cat([h_macro_seas, h_macro_trend], dim=-1)

        # -------- Conditioning --------
        gamma, beta = model.conditioner(h_macro)
        h_fin_cond = gamma * h_fin + beta

        # -------- Latent + decode --------
        z = model.to_latent(h_fin_cond)
        x_hat = model.decoder(z)

    # -------- Helper to flatten --------
    def to_df(x, name):
        x = x.cpu().numpy()
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        cols = [f"{name}_{i}" for i in range(x.shape[1])]
        return pd.DataFrame(x, columns=cols)

    dfs = {
        "h_fin": to_df(h_fin, "h_fin"),
        "h_macro": to_df(h_macro, "h_macro"),
        "gamma": to_df(gamma, "gamma"),
        "beta": to_df(beta, "beta"),
        "h_fin_cond": to_df(h_fin_cond, "h_fin_cond"),
        "z": to_df(z, "z"),
        "x_hat": to_df(x_hat, "x_hat"),
    }

    return dfs

def train_masked_ae(model, X_fin, X_macro, num_epochs=10, lr=1e-3, batch_size=32, 
                    device=DEVICE, mask_prob=0.2, alpha=1.0, repeats=1, seed=42, use_mask=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_fin, X_macro)
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0
    )

    metrics = []
    for epoch in range(num_epochs):
        model.train()
        total_mse = total_mae = total_smooth = 0.0
        n = 0

        for x_fin_batch, x_macro_batch in loader:
            x_fin_batch = x_fin_batch.to(device)
            x_macro_batch = x_macro_batch.to(device)

            for r in range(repeats):
                optimizer.zero_grad()
                
                if use_mask and mask_prob > 0:
                    mask_gen = torch.Generator(device='cpu')
                    mask_gen.manual_seed(seed + r + n)
                    mask = torch.rand(x_fin_batch.shape, generator=mask_gen).to(device) < mask_prob
                    
                    x_fin_input = x_fin_batch.clone()
                    x_fin_input[mask] = 0.0
                    
                    _, x_hat = model(x_fin_input, x_macro_batch)
                    
                    mse_masked = F.mse_loss(x_hat[mask], x_fin_batch[mask])
                    mse_full = F.mse_loss(x_hat, x_fin_batch)
                    loss = alpha * mse_masked + (1 - alpha) * mse_full
                else:
                    # Standard AE training: No mask, direct reconstruction
                    _, x_hat = model(x_fin_batch, x_macro_batch)
                    loss = F.mse_loss(x_hat, x_fin_batch)

                loss.backward()
                optimizer.step()

                batch_size_eff = x_fin_batch.size(0)
                n += batch_size_eff

                total_mse += loss.item() * batch_size_eff
                with torch.no_grad():
                    total_mae += F.l1_loss(x_hat, x_fin_batch, reduction='mean').item() * batch_size_eff
                    total_smooth += F.smooth_l1_loss(x_hat, x_fin_batch, reduction='mean').item() * batch_size_eff

        avg_mse = total_mse / n
        avg_mae = total_mae / n
        avg_smooth = total_smooth / n
        metrics.append({"epoch": epoch, "mse": avg_mse, "mae": avg_mae, "smooth": avg_smooth})

        print(f"[INFO] Epoch {epoch+1}/{num_epochs} MSE={avg_mse:.6f} MAE={avg_mae:.6f} Smooth={avg_smooth:.6f}")

    return model, metrics

def log_zero_sparsity(X_fin, X_macro, fin_cols, macro_cols, run_name):
    print(f"[INFO] Computing zero sparsity – {run_name}")
    fin_flat = X_fin.cpu().numpy().reshape(-1, len(fin_cols))
    macro_flat = X_macro.cpu().numpy().reshape(-1, len(macro_cols))
    
    fin_zeros = (fin_flat == 0).mean(axis=0) * 100
    macro_zeros = (macro_flat == 0).mean(axis=0) * 100
    
    df_fin = pd.DataFrame({'Feature': fin_cols, 'Zero_Pct': fin_zeros, 'Type': 'Financial'})
    df_macro = pd.DataFrame({'Feature': macro_cols, 'Zero_Pct': macro_zeros, 'Type': 'Macro'})
    df = pd.concat([df_fin, df_macro]).sort_values(by='Zero_Pct', ascending=True)
    
    plt.figure(figsize=(10, len(df) * 0.25 + 2))
    sns.barplot(data=df, x='Zero_Pct', y='Feature', hue='Type', palette={'Financial': '#1f77b4', 'Macro': '#ff7f0e'}, dodge=False)
    plt.axvline(50, color='red', linestyle='--', alpha=0.5)
    plt.title(f"Zero Observation Sparsity (%) – {run_name}")
    plt.xlabel("Percentage of Zero Values")
    plt.xlim(0, 100)
    plt.tight_layout()
    
    fig_path = f"plots/sparsity_{run_name}.png"
    plt.savefig(fig_path)
    plt.close()
    mlflow.log_artifact(fig_path)

def log_financial_boxplots(X_fin, fin_cols, run_name, chunk_size=10):
    print(f"[INFO] Generating financial boxenplots – {run_name}")
    
    # Flatten N e T: (N, T, F) -> (N*T, F)
    fin_flat = X_fin.cpu().numpy().reshape(-1, len(fin_cols))
    
    for i in range(0, len(fin_cols), chunk_size):
        subset_cols = fin_cols[i : i + chunk_size]
        subset_data = fin_flat[:, i : i + chunk_size]
        df = pd.DataFrame(subset_data, columns=subset_cols)
        
        plt.figure(figsize=(14, 8))
        # Boxenplot: ideale per dataset grandi con outliers pesanti
        sns.boxenplot(data=df, orient="h", palette="Blues_d", k_depth="proportion")
        
        plt.title(f"Financial Distributions (Boxenplot) Chunk {i//chunk_size + 1} – {run_name}")
        plt.xlabel("Value")
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        fig_path = f"plots/boxen_fin_{run_name}_chunk_{i}.png"
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path)

def log_macro_boxplots(X_macro, macro_cols, run_name, chunk_size=10):
    print(f"[INFO] Generating macro boxenplots – {run_name}")
    
    macro_flat = X_macro.cpu().numpy().reshape(-1, len(macro_cols))
    
    for i in range(0, len(macro_cols), chunk_size):
        subset_cols = macro_cols[i : i + chunk_size]
        subset_data = macro_flat[:, i : i + chunk_size]
        df = pd.DataFrame(subset_data, columns=subset_cols)
        
        plt.figure(figsize=(14, 8))
        # Colore arancione per distinguere macro da fin
        sns.boxenplot(data=df, orient="h", color="#ff7f0e", k_depth="proportion")
        
        plt.title(f"Macro Distributions (Boxenplot) Chunk {i//chunk_size + 1} – {run_name}")
        plt.xlabel("Value")
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        fig_path = f"plots/boxen_macro_{run_name}_chunk_{i}.png"
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path)

def log_pairplot_financial(X_fin, fin_cols, run_name):
    print(f"[INFO] Generating Global Financial pairplot (Flattened) – {run_name}")
    
    # Flatten (N, T, F) -> (N*T, F)
    # This treats every quarterly report from every company as a unique observation.
    fin_flat = X_fin.cpu().numpy().reshape(-1, len(fin_cols))
    
    df_fin = pd.DataFrame(fin_flat, columns=fin_cols)
    
    # Drop rows that are all zeros (padding or inactive periods)
    df_fin = df_fin[(df_fin.T != 0).any()] 
    
    if df_fin.empty: 
        print("[WARNING] df_fin is empty after removing zeros.")
        return

    # Square matrix: Global internal dynamics
    # Note: Increased figsize slightly and reduced alpha because you have more points now.
    plt.figure(figsize=(20, 20))
    pg = sns.pairplot(df_fin, 
                     diag_kind="kde", 
                     corner=True, 
                     plot_kws={'alpha': 0.15, 's': 3, 'color': '#1f77b4'})
    
    pg.fig.suptitle(f"Global Financial Dynamics (All Quarters) – {run_name}", y=1.02)
    
    path = f"plots/pairplot_FIN_ONLY_{run_name}.png"
    pg.savefig(path)
    plt.close('all')
    mlflow.log_artifact(path)

def log_pairplot_macro_impact(X_fin, X_macro, fin_cols, macro_cols, run_name):
    print(f"[INFO] Generating Macro-vs-Financial pairplot – {run_name}")
    
    fin_flat = X_fin.cpu().numpy().reshape(-1, len(fin_cols))
    macro_flat = X_macro.cpu().numpy().reshape(-1, len(macro_cols))
    
    df_all = pd.concat([
        pd.DataFrame(fin_flat, columns=fin_cols),
        pd.DataFrame(macro_flat, columns=macro_cols)
    ], axis=1)
    df_all = df_all[(df_all.T != 0).any()]

    if df_all.empty: return

    # Rectangular matrix: Macro (X) vs Financial (Y)
    # This is the strategic view for the ECB.
    plt.figure(figsize=(len(macro_cols) * 3, len(fin_cols) * 2))
    pg = sns.pairplot(df_all, 
                      x_vars=macro_cols, 
                      y_vars=fin_cols, 
                      kind="scatter",
                      plot_kws={'alpha': 0.3, 's': 7, 'color': '#ff7f0e'})
    
    pg.fig.suptitle(f"Macroeconomic Impact on Financials – {run_name}", y=1.005)
    
    path = f"plots/pairplot_MACRO_IMPACT_{run_name}.png"
    pg.savefig(path)
    plt.close('all')
    mlflow.log_artifact(path)

def get_importance_matrix(model, x_f, x_m, fin_cols, macro_cols, label, seed=42):
    print(f"[INFO] Computing importance matrix ({label})")
    model.eval(); model.to(DEVICE)
    x_f = x_f.to(DEVICE); x_m = x_m.to(DEVICE)
    N, T, F = x_f.shape
    all_inputs = fin_cols + macro_cols
    out_feats = fin_cols
    with torch.no_grad():
        _, x_hat = model(x_f, x_m)
        base_loss = ((x_hat - x_f)**2).mean(dim=1)
    imp_matrix = np.zeros((len(out_feats), len(all_inputs)))
    for j, feat in enumerate(all_inputs):
        gen = torch.Generator().manual_seed(seed+j)
        perm = torch.randperm(N, generator=gen)
        xf_p = x_f.clone(); xm_p = x_m.clone()
        if feat in fin_cols:
            idx = fin_cols.index(feat); xf_p[:,:,idx] = xf_p[perm,:,idx]
        else:
            idx = macro_cols.index(feat); xm_p[:,:,idx] = xm_p[perm,:,idx]
        with torch.no_grad():
            _, x_hat_p = model(xf_p, xm_p)
            perm_loss = ((x_hat_p - x_f)**2).mean(dim=1)
        delta = (perm_loss - base_loss).mean(dim=0)
        imp_matrix[:, j] = delta.cpu().numpy()
    df = pd.DataFrame(imp_matrix, index=out_feats, columns=all_inputs)
    print(f"[INFO] Importance matrix ({label}) shape={df.shape}")
    return df

def compute_macro_exposure_both(
    model_ctx,
    model_blind,
    X_fin,
    X_macro,
    device=DEVICE
):

    model_ctx.eval().to(device)
    model_blind.eval().to(device)

    X_fin = X_fin.to(device)
    X_macro = X_macro.to(device)

    with torch.no_grad():

        perm_idx = torch.randperm(X_macro.size(0), device=device)
        X_macro_perm = X_macro[perm_idx]

        # -------- CONTEXTUAL --------

        z_ctx_real, _ = model_ctx(X_fin, X_macro)
        z_ctx_cf, _ = model_ctx(X_fin, X_macro_perm)

        ctx_l2 = torch.norm(z_ctx_real - z_ctx_cf, dim=1)

        ctx_cos = 1 - F.cosine_similarity(
            z_ctx_real,
            z_ctx_cf,
            dim=1
        )

        # -------- BLIND --------

        z_blind_real, _ = model_blind(X_fin, X_macro)
        z_blind_cf, _ = model_blind(X_fin, X_macro_perm)

        blind_l2 = torch.norm(z_blind_real - z_blind_cf, dim=1)

        blind_cos = 1 - F.cosine_similarity(
            z_blind_real,
            z_blind_cf,
            dim=1
        )

    return {
        "contextual_l2": ctx_l2.cpu().numpy(),
        "contextual_cosine": ctx_cos.cpu().numpy(),
        "blind_l2": blind_l2.cpu().numpy(),
        "blind_cosine": blind_cos.cpu().numpy(),
    }

def log_macro_exposure(exposure, run_name, company_names, label):
    """
    exposure: np.ndarray
    company_names: aligned with exposure
    label: contextual_l2, contextual_cosine, blind_l2, blind_cosine
    """

    df_exposure = pd.DataFrame({
        "company_name": company_names,
        "macro_exposure": exposure
    }).sort_values("macro_exposure", ascending=False)

    # ---------- SAVE CSV ----------
    csv_path = f"tables/macro_exposure_{label}_{run_name}.csv"

    df_exposure.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    # ---------- LOG METRICS ----------
    mean_val = float(df_exposure["macro_exposure"].mean())
    std_val = float(df_exposure["macro_exposure"].std())
    cv_val = std_val / (mean_val + 1e-9)

    mlflow.log_metric(f"{label}_mean", mean_val)
    mlflow.log_metric(f"{label}_std", std_val)
    mlflow.log_metric(f"{label}_cv", cv_val)

    # ---------- HISTOGRAM ----------
    plt.figure(figsize=(6,4))

    plt.hist(
        df_exposure["macro_exposure"],
        bins=30,
        color="#1f77b4"
    )

    plt.title(f"Macro Exposure Distribution – {label}")
    plt.xlabel("Exposure")
    plt.ylabel("Frequency")

    plt.tight_layout()

    fig_path = f"plots/macro_exposure_{label}_{run_name}.png"

    plt.savefig(fig_path)
    plt.close()

    mlflow.log_artifact(fig_path)

    print(f"[INFO] Logged macro exposure ({label})")

    return df_exposure


def log_importance_matrix(df, run_name, label):
    csv_path = f"tables/importance_matrix_{label}_{run_name}.csv"
    df.reset_index().to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    data = df.values.copy()
    masked = np.ma.masked_where(data <= 0, data)

    from matplotlib.colors import LinearSegmentedColormap

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="lightgray")

    # ---- robust scaling ----
    positive_data = data[data > 0]

    if len(positive_data) == 0:
        vmin, vmax = 0, 1
    else:
        vmin = np.percentile(positive_data, 1)
        vmax = np.percentile(positive_data, 99)
        if vmin == vmax:
            vmax = vmin + 1e-8

    plt.figure(figsize=(14, 10))

    im = plt.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(im, label="Δ MSE")
    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.index)), df.index)
    plt.title(f"Importance matrix ({label}) – {run_name}")
    plt.tight_layout()

    fig_path = f"plots/importance_matrix_{label}_{run_name}.png"
    plt.savefig(fig_path)
    plt.close()

    mlflow.log_artifact(fig_path)
    print(f"[INFO] Logged importance matrix ({label}) CSV + heatmap")

def log_correlation_matrix(X_fin, X_macro, fin_cols, macro_cols, run_name):
    print(f"[INFO] Computing correlation matrix – {run_name}")
    
    # Flatten T (sequence) dimension to treat all observations as data points
    # N, T, F -> (N*T), F
    fin_flat = X_fin.cpu().numpy().reshape(-1, len(fin_cols))
    macro_flat = X_macro.cpu().numpy().reshape(-1, len(macro_cols))
    
    # Combine into a single DataFrame
    combined_df = pd.concat([
        pd.DataFrame(fin_flat, columns=fin_cols),
        pd.DataFrame(macro_flat, columns=macro_cols)
    ], axis=1)
    
    corr_matrix = combined_df.corr()
    
    # Save CSV
    csv_path = f"tables/correlation_matrix_{run_name}.csv"
    corr_matrix.to_csv(csv_path)
    mlflow.log_artifact(csv_path)
    
    # Plot Heatmap
    plt.figure(figsize=(16, 12))
    # Using RdBu_r for correlations (Red = Positive, Blue = Negative)
    im = plt.imshow(corr_matrix.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, label="Pearson Correlation")
    
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
    plt.title(f"Feature Correlation Matrix – {run_name}")
    plt.tight_layout()
    
    fig_path = f"plots/correlation_matrix_{run_name}.png"
    plt.savefig(fig_path)
    plt.close()
    
    mlflow.log_artifact(fig_path)
    print(f"[INFO] Logged correlation matrix CSV + heatmap")
    return corr_matrix

def plot_loss_comparison_combined(metrics_ctx, metrics_blind, run_name):
    print("[INFO] Plotting combined loss curves")
    epochs = [m["epoch"]+1 for m in metrics_ctx]
    fig, axes = plt.subplots(1,3,figsize=(18,5))
    for i, metric in enumerate(["mse","mae","smooth"]):
        axes[i].plot(epochs, [m[metric] for m in metrics_ctx], label="Contextual")
        axes[i].plot(epochs, [m[metric] for m in metrics_blind], label="Blind")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric.upper())
        axes[i].set_title(metric.upper())
        axes[i].legend()
    plt.suptitle(f"Loss comparison {run_name}")
    plt.tight_layout(rect=[0,0,1,0.95])
    out = f"plots/loss_comparison_{run_name}.png"
    plt.savefig(out); plt.close()
    mlflow.log_artifact(out)
    print(f"[INFO] Logged combined loss comparison {out}")

def prove_macro_utility(
    model_ctx,
    model_blind,
    val_loader,
    run_name,
    device=DEVICE,
):
    print(f"[INFO] Computing Macro Embedding Tournament – {run_name}")

    results = []

    for label, model in {"Contextual": model_ctx, "Blind": model_blind}.items():

        model.eval()

        cosine_dists = []
        euclidean_dists = []

        for xf, xm in val_loader:

            xf = xf.to(device)
            xm = xm.to(device)

            # permute macro across batch
            idx = torch.randperm(xm.size(0), device=device)
            xm_perm = xm[idx]

            with torch.no_grad():

                z_real, _ = model(xf, xm)
                z_perm, _ = model(xf, xm_perm)

                # cosine distance
                cos_sim = F.cosine_similarity(z_real, z_perm, dim=-1)
                cos_dist = 1 - cos_sim

                # euclidean distance
                euc_dist = torch.norm(z_real - z_perm, dim=-1)

                cosine_dists.append(cos_dist.cpu())
                euclidean_dists.append(euc_dist.cpu())

        cosine_dists = torch.cat(cosine_dists).numpy()
        euclidean_dists = torch.cat(euclidean_dists).numpy()

        results.append({
            "Model": label,
            "Cosine_Distance_mean": cosine_dists.mean(),
            "Cosine_Distance_std": cosine_dists.std(),
            "Euclidean_Distance_mean": euclidean_dists.mean(),
            "Euclidean_Distance_std": euclidean_dists.std(),
        })

        print(f"\n--- {label} Model ---")
        print(f"Cosine distance mean:    {cosine_dists.mean():.6f}")
        print(f"Euclidean distance mean: {euclidean_dists.mean():.6f}")

    df = pd.DataFrame(results)

    # save CSV
    csv_path = f"tables/macro_embedding_tournament_{run_name}.csv"
    df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    # -------- COSINE PLOT --------

    plt.figure(figsize=(8,5))

    x = np.arange(len(df))

    plt.bar(
        x,
        df["Cosine_Distance_mean"],
        yerr=df["Cosine_Distance_std"],
        capsize=5,
        color="#1f77b4"
    )

    plt.xticks(x, df["Model"])
    plt.ylabel("Cosine distance")
    plt.title(f"Embedding Shift from Macro (Cosine)\n{run_name}")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    out_cos = f"plots/macro_embedding_cosine_{run_name}.png"
    plt.tight_layout()
    plt.savefig(out_cos)
    plt.close()

    mlflow.log_artifact(out_cos)

    # -------- EUCLIDEAN PLOT --------

    plt.figure(figsize=(8,5))

    plt.bar(
        x,
        df["Euclidean_Distance_mean"],
        yerr=df["Euclidean_Distance_std"],
        capsize=5,
        color="#ff7f0e"
    )

    plt.xticks(x, df["Model"])
    plt.ylabel("Euclidean distance")
    plt.title(f"Embedding Shift from Macro (Euclidean)\n{run_name}")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    out_euc = f"plots/macro_embedding_euclidean_{run_name}.png"
    plt.tight_layout()
    plt.savefig(out_euc)
    plt.close()

    mlflow.log_artifact(out_euc)

    return df


def plot_importance_summary(imp_ctx, imp_blind, run_name):
    financial_cols = [c for c in imp_ctx.columns if c in fin_cols]
    macro_cols = [c for c in imp_ctx.columns if c in MACRO_COLUMNS]

    summary_ctx = pd.Series({
        "financial": imp_ctx[financial_cols].abs().mean().sum(),
        "macro": imp_ctx[macro_cols].abs().mean().sum()
    })
    summary_blind = pd.Series({
        "financial": imp_blind[financial_cols].abs().mean().sum(),
        "macro": imp_blind[macro_cols].abs().mean().sum()
    })

    df_summary = pd.DataFrame({
        "Contextual": summary_ctx,
        "Blind": summary_blind
    })
    csv_path = f"tables/importance_summary_{run_name}.csv"
    df_summary.reset_index().to_csv(csv_path)
    mlflow.log_artifact(csv_path)
    print(f"[INFO] Logged importance summary CSV")

    plt.figure(figsize=(6,4))
    df_summary.plot.bar()
    plt.title(f"Importance Summary (financial vs macro) {run_name}")
    plt.tight_layout()
    out = f"plots/importance_summary_{run_name}.png"
    plt.savefig(out); plt.close()
    mlflow.log_artifact(out)
    print(f"[INFO] Logged importance summary plot {out}")
    

# ----------------------------
# 11. EXPERIMENT LOOP
# ----------------------------


print("[INFO] Starting experiments")
for factor in args.latent_factors:
    latent_dim = max(1,int(math.ceil(FIN_DIM*factor)))
    run_name = f"latent_dim-{latent_dim}"
    print(f"[INFO] Run {run_name} | latent_factor={factor}")
    
    
    with mlflow.start_run(run_name=run_name):
        
        log_correlation_matrix(X_fin, X_macro, fin_cols, MACRO_COLUMNS, run_name)
        log_financial_boxplots(X_fin, fin_cols, run_name)
        log_macro_boxplots(X_macro, MACRO_COLUMNS, run_name)
        # log_pairplot_macro_impact(X_fin, X_macro, fin_cols, MACRO_COLUMNS, run_name)
        # log_pairplot_financial(X_fin, fin_cols, run_name)
        log_zero_sparsity(X_fin, X_macro, fin_cols, MACRO_COLUMNS, run_name)
        mlflow.log_params(vars(args))
        mlflow.log_param("latent_dim", latent_dim)


        # Train contextual
        print("[INFO] Training contextual model")
        model_ctx = CompanyEmbeddingAE(SEQ_LEN, FIN_DIM, MACRO_DIM, latent_dim)
        model_ctx, metrics_ctx = train_masked_ae(
            model=model_ctx,
            X_fin=X_fin,
            X_macro=X_macro,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            mask_prob=args.mask_prob,
            device=DEVICE,
            alpha=0,
            repeats=10,
            seed=args.seed,
            use_mask=args.use_mask
        )


        # Train blind
        print("[INFO] Training blind model")
        model_blind = CompanyEmbeddingAE(SEQ_LEN, FIN_DIM, MACRO_DIM, latent_dim)
        model_blind, metrics_blind = train_masked_ae(
            model=model_blind,
            X_fin=X_fin,
            X_macro=torch.zeros_like(X_macro),
            num_epochs=args.epochs,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            mask_prob=args.mask_prob,
            device=DEVICE,
            alpha=0,
            repeats=10,
            seed=args.seed
        )

        # Log final metrics
        mlflow.log_metric("final_mse_contextual", metrics_ctx[-1]["mse"])
        mlflow.log_metric("final_mse_macro_blind", metrics_blind[-1]["mse"])

        # Combined loss plot
        plot_loss_comparison_combined(metrics_ctx, metrics_blind, run_name)

        # Importance matrices
        imp_mat_ctx = get_importance_matrix(model_ctx, X_fin, X_macro, fin_cols, MACRO_COLUMNS, "contextual")
        imp_mat_blind = get_importance_matrix(model_blind, X_fin, torch.zeros_like(X_macro), fin_cols, MACRO_COLUMNS, "blind")
        log_importance_matrix(imp_mat_ctx, run_name, "contextual")
        log_importance_matrix(imp_mat_blind, run_name, "blind")

        # Importance summary plot
        plot_importance_summary(imp_mat_ctx, imp_mat_blind, run_name)

        # Macro utility / tournament plot
        loader = DataLoader(TensorDataset(X_fin, X_macro), batch_size=32)
        df_tournament = prove_macro_utility(model_ctx, model_blind, loader, run_name, device=DEVICE)

        exposures = compute_macro_exposure_both(
            model_ctx,
            model_blind,
            X_fin,
            X_macro
        )

        log_macro_exposure(
            exposures["contextual_l2"],
            run_name,
            meta_df["ticker"].values,
            label="contextual_l2"
        )

        log_macro_exposure(
            exposures["contextual_cosine"],
            run_name,
            meta_df["ticker"].values,
            label="contextual_cosine"
        )

        log_macro_exposure(
            exposures["blind_l2"],
            run_name,
            meta_df["ticker"].values,
            label="blind_l2"
        )

        log_macro_exposure(
            exposures["blind_cosine"],
            run_name,
            meta_df["ticker"].values,
            label="blind_cosine"
        )
        
        # Save models
        ctx_path = f"runs/{run_name}_ctx.pth"
        blind_path = f"runs/{run_name}_blind.pth"
        torch.save(model_ctx.state_dict(), ctx_path)
        torch.save(model_blind.state_dict(), blind_path)
        mlflow.log_artifact(ctx_path)
        mlflow.log_artifact(blind_path)
        
        # dfs = inspect_company_ae(model_ctx, X_fin, X_macro, n_obs=100)


print("[INFO] All experiments completed!")
