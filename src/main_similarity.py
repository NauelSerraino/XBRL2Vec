"""
Step 2 of 3 (src/main.py): train the encoders the evaluation needs, the paper model and its two ablations, 10 seeds
each (30 runs). A run already trained on the same panels is skipped, so re-running resumes.

    xbrl2vec/bin/python src/main_similarity.py                                   # every run not trained yet
    xbrl2vec/bin/python src/main_similarity.py --variants paper --seeds 42       # one run

SETTINGS is the paper model; each entry of VARIANTS changes some settings (a new experiment = one line there). The
data come from step 1 (src/main_download.py), which writes the balanced firms and their fingerprint.

What one run does:
  1. Sample (sample_sets): the windows of the 2,115 balanced firms (full data 2013Q1-2025Q4, services/data.py), split
     into 1,798 training firms and 317 unseen firms, never used in training.
  2. Windows and targets. Each firm gives one window per quarter: its fingerprint over the 12 quarters up to it (the
     input), the fingerprint of that quarter (Pillar 1 target) and the window deltas (Pillar 2 target).
  3. Training (services/similarity.train) on the windows of the training firms that end up to the cutoff.
  4. Output, in data/out/similarity/<time>_<tag>_s<seed>/: encoder.pth (the weights) and config.json (the settings,
     the panels and the loss of each epoch).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.xbrl2vec import XBRL2VecEncoder
from services.config import DEVICE
from services.data import LAST_QUARTER, T_IN, windows
from services.similarity import SimilarityData, attach_fingerprints, attach_window_deltas, train

DATA = Path(__file__).resolve().parents[1] / "data"
RUNS = DATA / "out" / "similarity"
PANEL = "filled"                         # the panel with gaps filled (src/preprocess/build_panels.py)
#: the paper model
SETTINGS = {
    "t_in": T_IN,                        # quarters in each input window
    "hidden": 128,                       # width of the encoder's MLP
    "latent_dim": 32,                    # size of the embedding
    "epochs": 20,
    "batch_size": 256,                   # windows per batch, all from one quarter
    "learning_rate": 1e-3,
    "cutoff": "2022Q4",                  # last quarter used in training; the later ones are the test period
    "lambda2": 0.5,                      # loss = (1 - lambda2) Pillar 1 + lambda2 Pillar 2
    "input_mask": 0.2,                   # share of the input cells (ratio x quarter) set to 0 in every training batch
    "pillar2_decay": 0.8,                # Pillar 2: the move ending k quarters before the last weighs decay ** k
}
#: name -> (tag of the run folders, settings changed); the evaluation finds a variant's runs by the tag
VARIANTS = {
    "paper": ("B_z32_mask02_lam05_slogz_dec08", {}),
    "unmasked": ("B_z32_lam05_slogz_dec08", {"input_mask": 0.0}),
    "decay1": ("B_z32_mask02_lam05_slogz", {"pillar2_decay": 1.0}),
}
SEEDS = [42, *range(1, 10)]


def latest_panels_dir() -> Path:
    """The newest data/in/panels_<date> that has a filled/ panel."""
    return sorted(p for p in (DATA / "in").glob("panels_*") if (p / PANEL).is_dir())[-1]


def seed_of(run: str) -> int:
    """The training seed in a run folder's name (..._s<seed>)."""
    return int(re.search(r"_s(\d+)$", run).group(1))


def runs(variant: str, panels_dir: Path | None = None) -> list[str]:
    """The finished runs (folder names) of a variant trained on panels_dir (default: the latest), seed 42 first, then
    by seed."""
    tag, panels_dir = VARIANTS[variant][0], panels_dir or latest_panels_dir()
    rx = re.compile(rf"_{tag}_s\d+$")
    rs = [d.name for d in RUNS.glob(f"*_{tag}_s*") if rx.search(d.name) and (d / "encoder.pth").exists()
          and (d / "config.json").exists()                      # written last: the run finished
          and Path(json.loads((d / "config.json").read_text())["panels_dir"]) == panels_dir]
    return sorted(rs, key=lambda r: (seed_of(r) != 42, seed_of(r)))


def resplit(sets: dict, split: pd.Series) -> dict:
    """The windows of both sets regrouped by `split` (ticker -> "train" / "unseen"); firms not in it are dropped."""
    a, u = sets["train firms"], sets["unseen firms"]
    both = SimilarityData(torch.cat([a.X, u.X]), np.concatenate([a.target, u.target]),
                          pd.concat([a.meta, u.meta], ignore_index=True), a.cols)
    s = both.meta.ticker.map(split)
    return {"train firms": both.subset(s.eq("train").to_numpy()), "unseen firms": both.subset(s.eq("unseen").to_numpy())}


def sample_sets(panel: Path, t_in: int) -> tuple[dict, pd.DataFrame]:
    """
    {"train firms", "unseen firms"}: the windows of the balanced firms that end up to LAST_QUARTER (input X and Pillar 1
    target), and the fingerprint they are made of. From <panel>/targets, written by src/main_download.py (step 5):
      firms_balanced.parquet               the balanced firms and their split
      fingerprint_slogz_balanced.parquet   the fingerprint of those firms, scaled on them
    """
    firms, fp_file = panel / "targets" / "firms_balanced.parquet", panel / "targets" / "fingerprint_slogz_balanced.parquet"
    if not fp_file.exists():
        raise SystemExit(f"{fp_file} missing: run step 1, src/main_download.py (its step 5 writes it)")
    fp = pd.read_parquet(fp_file)
    sets = {}
    for name, split in (("train firms", "train"), ("unseen firms", "test")):
        d = attach_fingerprints(*windows(panel, split, t_in, fp), fp)
        sets[name] = d.subset(d.meta.quarter.le(LAST_QUARTER).to_numpy())
    return resplit(sets, pd.read_parquet(firms).set_index("ticker").split), fp


def train_run(sets: dict, fp: pd.DataFrame, settings: dict, seed: int, out: Path) -> list[float]:
    """One run: the targets with the settings' decay, the training windows up to the cutoff, the training; writes
    encoder.pth in `out` and returns the loss of each epoch."""
    tr = attach_window_deltas(sets["train firms"], fp, settings["t_in"], settings["pillar2_decay"])
    train_data = tr.subset(tr.meta.quarter.le(settings["cutoff"]).to_numpy())
    print(f"[INFO] training {len(train_data.meta):,} windows, {train_data.meta.ticker.nunique():,} firms, "
          f"{train_data.meta.quarter.min()}–{train_data.meta.quarter.max()}", flush=True)
    torch.manual_seed(seed)
    model = XBRL2VecEncoder(
        T=settings["t_in"],
        F=train_data.X.shape[-1],
        latent_dim=settings["latent_dim"],
        hidden=settings["hidden"],
        )
    history = train(
        model=model,
        data=train_data,
        epochs=settings["epochs"],
        lr=settings["learning_rate"],
        batch_size=settings["batch_size"],
        device=DEVICE,
        seed=seed,
        lambda2=settings["lambda2"],
        input_mask=settings["input_mask"],
        )
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "encoder.pth")
    return history


def main() -> None:
    """Train every (variant, seed) asked for that has no finished run on these panels yet."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--panels_dir", type=Path, default=None, help="Default: the latest data/in/panels_*.")
    args = p.parse_args()
    panels_dir = (args.panels_dir or latest_panels_dir()).resolve()
    data = None                                                  # loaded once, for the first run to train
    for name in args.variants:
        tag, changes = VARIANTS[name]
        done = {seed_of(r) for r in runs(name, panels_dir)}
        for seed in args.seeds:
            if seed in done:
                print(f"[INFO] {name} seed {seed}: already trained", flush=True)
                continue
            data = data or sample_sets(panels_dir / PANEL, T_IN)
            settings = {**SETTINGS, **changes}
            out = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}_s{seed}"
            print(f"[INFO] {name} seed {seed} -> {out}", flush=True)
            history = train_run(*data, settings, seed, out)
            (out / "config.json").write_text(json.dumps({**settings, "variant": name, "seed": seed,
                                                         "panels_dir": str(panels_dir), "loss_history": history}, indent=2))


if __name__ == "__main__":
    main()
