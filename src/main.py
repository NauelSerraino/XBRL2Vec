import argparse
import mlflow
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models import * 
from services.paths import IN_DIR, MODEL_REGISTRY
from services.utils import * 
from services.config import *
from services.artifacts_manager import ArtifactsManager
from torch.utils.tensorboard import SummaryWriter


# --- 1. ARGUMENT PARSING & MLFLOW SETUP ---
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Euclidean FAE")
parser.add_argument("--latent_dim", type=int, default=150)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--lambda_ortho", type=float, default=1e-4)
parser.add_argument("--pretrain", action="store_true", help="Enable Phase 1 Masked Reconstruction")
parser.add_argument("--num_iterations_pretrain", type=int, default=1)
args = parser.parse_args()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("BERT-like training")


class ModelTrain:
    model_class_map = {
        'Euclidean FAE': FlattenedEuclideanFAE,
        'GRU FAE': RecurrentGRUFAE,
        'LSTM FAE': LSTMFAE,
        'RNN FAE': RNNFAE,
        'Transformer FAE': TransformerFAE,
        'D-Linear FAE': DLinearFAE,
    }
    
    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter(f"runs/{self.args.model_name}")
        self.model = None
        self.lambda_ortho = 0.0 
        self.fin_dim = 0
        self.macro_dim = 0
        self.X_fin, self.Y_fin, self.X_macro = None, None, None
        
    def train(self):
        self.load_data()
        self.process_data()
        self.load_model()
        self.train_model()
        self.save_model()
        self.create_artifacts()
        
        
    def load_data(self):
        print("Loading data...")
        try:
            self.bs = pd.read_parquet(os.path.join(IN_DIR, "bs_pct_train.parquet"))
            self.ins = pd.read_parquet(os.path.join(IN_DIR, "ins_pct_train.parquet"))
            self.cf = pd.read_parquet(os.path.join(IN_DIR, "cf_pct_train.parquet"))
            self.exog = pd.read_parquet(os.path.join(IN_DIR, "exog.parquet"))
        except FileNotFoundError as e:
            print(f"ERROR: Parquet file not found at {e.filename}. Check your 'IN_DIR' path.")
            raise
    
    def process_data(self):
        self.X_fin_tensor_past, self.Y_fin_tensor_future, self.X_macro_tensor_past = prepare_forecasting_data(
            self.bs, self.ins, self.cf, self.exog, 
            tickers_to_exclude=TICKERS, 
            seq_len=SEQ_LEN, 
            forecast_len=FORECAST_LEN, 
            macro_cols=MACRO_COLUMNS
        )
        
        self.fin_dim = self.X_fin_tensor_past.size(2)
        self.macro_dim = self.X_macro_tensor_past.size(2)
        print(f"Data Prepared. Fin Dim: {self.fin_dim}, Macro Dim: {self.macro_dim}")
    
    def load_model(self):
        # 5. LOG PARAMETERS
        mlflow.log_params(vars(args))
        mlflow.log_param("fin_dim", self.fin_dim)
        mlflow.log_param("macro_dim", self.macro_dim)
        
        # 6. MODEL INITIALIZATION (FIXED)
        
        ModelClass = ModelTrain.model_class_map[self.args.model_name]
        self.model = ModelClass(
            SEQ_LEN, self.fin_dim, self.macro_dim, self.args.latent_dim, FORECAST_LEN
        ).to(DEVICE)
        
        self.lambda_ortho = self.args.lambda_ortho if 'Euclidean' in self.args.model_name else 0.0

        print(f"--- Training {self.args.model_name} ---")
        
        print("\n--- DEBUG: MODEL PARAMETER NAMES ---")
        logged_names = []
        for name, param in self.model.named_parameters():
            print(f"Name: {name}")
            if 'fin_encoder' in name or 'decoder' in name:
                logged_names.append(name)
                
        if not logged_names:
            print("CRITICAL ERROR: No parameter names matched the logging filter!")
            
        print("--------------------------------------\n")

    def train_model(self):
        # Log whether pre-training was used
        mlflow.log_param("use_pretraining", self.args.pretrain)

        # PHASE 1: PRE-TRAINING (Optional)
        if self.args.pretrain:
            print(f"--- BERT-LIKE TRAINING: Reconstructing Xt -> Xt ---")
            final_loss = bert_like_train(
                self.model, 
                self.X_fin_tensor_past, 
                self.X_macro_tensor_past, 
                epochs=self.args.epochs, 
                batch_size=self.args.batch_size, 
                lr=self.args.learning_rate,
                mask_ratio=0.25, #TODO: keep the macro data
                num_augmentations=self.args.num_iterations_pretrain
            )
        else:
            print(f"--- SKIPPING PRE-TRAINING: Direct Fine-tuning ---")

        mlflow.log_metric("final_mse_loss", final_loss)
        
    def save_model(self):
        model_save_dir = os.path.join(MODEL_REGISTRY, self.args.model_name)
        os.makedirs(model_save_dir, exist_ok=True)
        save_path = os.path.join(model_save_dir, f"{self.args.model_name}_state.pth")
        
        torch.save(self.model.state_dict(), save_path)
        mlflow.log_artifact(save_path)
        
    def create_artifacts(self):        
        manager = ArtifactsManager(
            self.model, 
            self.X_fin_tensor_past, 
            self.X_macro_tensor_past, 
            self.Y_fin_tensor_future, 
            self.args
        )
        manager.run_all()


if __name__=="__main__":
    with mlflow.start_run(run_name=args.model_name) as run:
        trainer = ModelTrain(args)
        trainer.train()