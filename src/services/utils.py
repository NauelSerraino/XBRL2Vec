# --- utils.py (CRITICAL FUNCTIONS) ---
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import mlflow # Required for logging
from models import SpecializedHyperbolicFAE
from services.config import FORECAST_LEN # Required for isinstance check

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

def bert_like_train(model, X_fin, X_macro, epochs, batch_size, lr, mask_ratio=0.3,num_augmentations=5):
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
    

def create_sequences_for_forecast(df: pd.DataFrame, entity_col="ticker", time_col="quarter", features=None, seq_len=4, forecast_len=4):
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
    required_length = seq_len + forecast_len
    
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
        return np.empty((0, seq_len, num_features)), np.empty((0, forecast_len, num_features)), pd.DataFrame(metadata)
    
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

def prepare_forecasting_data(bs, ins, cf, exog, tickers_to_exclude, seq_len, forecast_len, macro_cols):
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
    required_len = seq_len + forecast_len
    ticker_counts = merged.groupby('ticker')['quarter'].count()
    valid_tickers = ticker_counts[ticker_counts >= required_len].index.tolist()
    merged = merged[merged['ticker'].isin(valid_tickers)]

    # 4. Feature Splitting
    all_features = [c for c in merged.columns if c not in ["ticker", "quarter", "observation_date"]]
    fin_columns = [c for c in all_features if c not in macro_cols]

    # 5. Sequence Generation
    X_fin, Y_fin, _ = create_sequences_for_forecast(merged, features=fin_columns, seq_len=seq_len, forecast_len=forecast_len)
    X_macro, _, _ = create_sequences_for_forecast(merged, features=macro_cols, seq_len=seq_len, forecast_len=forecast_len)

    return (
        torch.tensor(X_fin, dtype=torch.float32), 
        torch.tensor(Y_fin, dtype=torch.float32), 
        torch.tensor(X_macro, dtype=torch.float32)
    )