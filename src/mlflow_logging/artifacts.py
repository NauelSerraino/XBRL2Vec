"""
Central artifact logger.

Naming convention for MLflow artifacts:
    {GROUP} - {description}_{run_name}.{ext}

Groups:
    DISTRIBUTION  – raw data distributions
    EMBEDDING     – latent space geometry
    IMPORTANCE    – permutation importance
    SALIENCY      – integrated gradients
    TOURNAMENT    – contextual vs blind comparisons
    TRAINING      – loss curves
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import mlflow


class ArtifactGroup(str, Enum):
    DISTRIBUTION = "DISTRIBUTION"
    EMBEDDING    = "EMBEDDING"
    IMPORTANCE   = "IMPORTANCE"
    SALIENCY     = "SALIENCY"
    TOURNAMENT   = "TOURNAMENT"
    TRAINING     = "TRAINING"


class ArtifactLogger:
    """
    Manages output directories and MLflow artifact logging.

    Usage:
        logger = ArtifactLogger(run_name="latent_dim-32")
        path   = logger.plot_path(ArtifactGroup.DISTRIBUTION, "zero_sparsity")
        plt.savefig(path)
        logger.log(path)
    """

    PLOTS_DIR  = Path("plots")
    TABLES_DIR = Path("tables")
    RUNS_DIR   = Path("runs")

    def __init__(self, run_name: str):
        self.run_name = run_name
        for d in [self.PLOTS_DIR, self.TABLES_DIR, self.RUNS_DIR]:
            d.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ paths

    def plot_path(self, group: ArtifactGroup, name: str, ext: str = "png") -> Path:
        filename = f"{group.value} - {name}_{self.run_name}.{ext}"
        return self.PLOTS_DIR / filename

    def table_path(self, group: ArtifactGroup, name: str, ext: str = "csv") -> Path:
        filename = f"{group.value} - {name}_{self.run_name}.{ext}"
        return self.TABLES_DIR / filename

    def model_path(self, label: str) -> Path:
        return self.RUNS_DIR / f"{self.run_name}_{label}.pth"

    # ------------------------------------------------------------------ log

    def log(self, path: Path) -> None:
        mlflow.log_artifact(str(path))

    def log_table(self, df, group: ArtifactGroup, name: str) -> Path:
        path = self.table_path(group, name)
        df.to_csv(path)
        self.log(path)
        return path