# XBRL2Vec — Contextual vs Blind Financial Autoencoder

A research framework for quantifying **macro exposure** in corporate financial statements. It trains two D-Linear autoencoders side-by-side:

- **Contextual model** — reconstructs financial time series using both financial inputs and macroeconomic context.
- **Blind model** — same architecture, but macro inputs are zeroed out.

The difference in out-of-sample reconstruction error (*Macro Gain = blind MSE − contextual MSE*) measures how much macroeconomic information the financial data implicitly carries.

---

## Project structure

```text
XBRL2Vec/
├── src/
│   ├── main.py                          # Entry point & experiment loop
│   ├── models/
│   │   └── autoencoder_dlinear_conditioner.py  # CompanyEmbeddingAE (active model)
│   ├── services/
│   │   ├── config.py                    # Global constants (SEQ_LEN, DEVICE, tickers)
│   │   ├── data.py                      # Data loading, alignment, dataset container
│   │   ├── evaluation.py                # OOS metrics, importance matrix, variance analysis
│   │   ├── training.py                  # MaskedAETrainer
│   │   └── transforms.py               # symmetric_log, macro weighting
│   └── mlflow_logging/
│       ├── plots.py                     # All MLflow plot helpers
│       ├── saliency.py                  # Integrated Gradients (Captum)
│       └── artifacts.py                 # Artifact path management
├── data/in/                             # Parquet data files (not tracked by git)
├── notebooks/                           # Exploratory and post-training analysis
├── tests/                               # Unit tests
├── docker-compose.yaml                  # PostgreSQL backend for MLflow
├── MLproject                            # MLflow project definition
├── python_env.yaml                      # Conda environment spec
└── requirements.txt                     # Pip dependencies
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Data

Place the following parquet files in `data/in/`:

| File | Description |
| --- | --- |
| `bs_pct_train.parquet` | Balance Sheet — train |
| `ins_pct_train.parquet` | Income Statement — train |
| `cf_pct_train.parquet` | Cash Flow Statement — train |
| `bs_pct_test.parquet` | Balance Sheet — test |
| `ins_pct_test.parquet` | Income Statement — test |
| `cf_pct_test.parquet` | Cash Flow Statement — test |
| `exog.parquet` | Macroeconomic indicators (GDP, CPI, DFF, …) |
| `metadata.parquet` | Company metadata (ticker, sector) |

### 3. Start the MLflow backend

```bash
# Spin up the PostgreSQL backend
docker compose up -d

# Start the MLflow tracking server
mlflow server --host 127.0.0.1 --port 5000
```

The UI will be available at `http://localhost:5000`.

### 4. (Optional) TensorBoard

```bash
tensorboard --logdir=runs --port=6006
```

---

## Running experiments

```bash
python src/main.py \
    --latent_factors 0.5 1.0 2.0 3.0 \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --seed 42
```

The loop trains one MLflow run per latent factor. `latent_dim` is computed as `ceil(fin_dim × factor)`, so the same set of factors adapts to the number of financial features in the data.

### CLI reference

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--latent_factors` | `float …` | `0.5 1.0 2.0 3.0` | Multipliers of `fin_dim` used to derive `latent_dim` |
| `--epochs` | `int` | `20` | Training epochs per model |
| `--batch_size` | `int` | `32` | Mini-batch size |
| `--learning_rate` | `float` | `0.001` | Adam learning rate |
| `--mask_prob` | `float` | `0.2` | Feature masking probability (active when `use_mask=1`) |
| `--use_mask` | `0 / 1` | `0` | Enable BERT-style masked reconstruction loss |
| `--seed` | `int` | `42` | Global random seed |

---

## Model architecture

`CompanyEmbeddingAE` ([src/models/autoencoder_dlinear_conditioner.py](src/models/autoencoder_dlinear_conditioner.py)) is a D-Linear autoencoder with macro conditioning:

1. **DLinearEncoder** — decomposes the financial time series `[B, T, F]` into seasonal and trend components, then projects to a financial embedding `h_fin ∈ ℝᴴ`.
2. **MacroEncoder** — projects the macroeconomic series `[B, T, M]` to a macro embedding `h_mac ∈ ℝᴴ`.
3. **MacroConditioner** — learns affine parameters `(γ, β)` from `h_mac` and applies them to `h_fin`: `h_cond = γ ⊙ h_fin + β`.
4. **FinancialDecoder** — reconstructs `[B, T, F]` from `h_cond`.

The blind model uses the same architecture but receives `zeros_like(X_macro)` as input.

---

## Metrics logged to MLflow

| Metric | Description |
| --- | --- |
| `final_mse_contextual` | Train-set MSE — contextual model |
| `final_mse_macro_blind` | Train-set MSE — blind model |
| `oos_mse_contextual` | Out-of-sample MSE — contextual |
| `oos_mae_contextual` | Out-of-sample MAE — contextual |
| `oos_mse_blind` | Out-of-sample MSE — blind |
| `oos_mae_blind` | Out-of-sample MAE — blind |
| `oos_macro_gain` | `oos_mse_blind − oos_mse_contextual` (key metric) |

Plots logged per run: correlation matrix, loss curves, feature importance heatmaps, embedding geometry, variance analysis (R²), macro exposure density, and Integrated Gradients saliency maps.
