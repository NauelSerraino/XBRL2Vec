"""
Train the XBRL2Vec encoder: one run = one seed.

    cd src && ../xbrl2vec/bin/python main_similarity.py --seed 42 --out ../data/out/similarity/<name>

The defaults are the paper model; the ablations change one flag: --input_mask 0 (no input mask), --pillar2_decay 1
(all moves weigh the same), --lambda2 (the weight of the dynamics).

What happens:
  1. Sample (sample_sets). The universe is every firm-quarter up to LAST_QUARTER that ends a complete 12-quarter
     window, both in the amounts and their yearly changes and in the fingerprint. The balanced sample is the firms that
     have such a window at EVERY quarter of the universe (2015Q4-2025Q4, 41 quarters), that is firms with full data
     from 2013Q1: 2,115 firms. 15 % of them, drawn at random once (seed 0), are the unseen firms: never used in
     training. The fingerprint is rescaled on the balanced firms.
  2. Windows and targets. Each firm gives one window per quarter: its fingerprint over the 12 quarters up to it (the
     input), the fingerprint of that quarter (Pillar 1 target) and the window deltas (Pillar 2 target).
  3. Training (services/similarity.train) on the windows of the training firms that end up to --cutoff.
  4. Output, in --out: encoder.pth (the weights) and config.json (the settings and the loss of each epoch).

Scoring is separate: usefulness_eval.py, then the scripts of model-evaluation/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.xbrl2vec import XBRL2VecEncoder
from services.config import DEVICE
from services.data import KEYS, load_statements, sliding_windows
from services.similarity import SimilarityData, attach_fingerprints, attach_window_deltas, train

DATA = Path(__file__).resolve().parents[1] / "data"
PANEL = "filled"                         # the panel with gaps filled (src/preprocess/build_panels.py)
LAST_QUARTER = "2025Q4"                  # whole years: the 2026 quarters are left out
UNSEEN_SHARE, SPLIT_SEED = 0.15, 0       # the unseen firms: their share of the balanced firms, and the seed of the draw


def parse_args():
    """The training options; the defaults are the paper model."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panels_dir", type=Path, default=None, help="Default: the latest data/in/panels_*.")
    p.add_argument("--t_in", type=int, default=12, help="Quarters in each input window.")
    p.add_argument("--hidden", type=int, default=128, help="Width of the encoder's MLP.")
    p.add_argument("--latent_dim", type=int, default=32, help="Size of the embedding.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256, help="Windows per batch (all from one quarter).")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42, help="Seed of the weights' initialisation, the batches and the mask.")
    p.add_argument("--cutoff", default="2022Q4", help="Last quarter used in training; the later ones are the test period.")
    p.add_argument("--lambda2", type=float, default=0.5,
                   help="Weight of Pillar 2 in the loss, in [0, 1]: (1 - lambda2) Pillar 1 + lambda2 Pillar 2.")
    p.add_argument("--input_mask", type=float, default=0.2,
                   help="Share of the input cells (ratio x quarter) set to 0 at random in every training batch.")
    p.add_argument("--pillar2_decay", type=float, default=0.8,
                   help="Pillar 2: the move ending k quarters before the window's last quarter weighs decay ** k.")
    p.add_argument("--out", type=Path, default=None, help="Run folder (default: data/out/similarity/<timestamp>).")
    return p.parse_args()


def latest_panels_dir() -> Path:
    """The newest data/in/panels_<date> that has a filled/ panel."""
    return sorted(p for p in (DATA / "in").glob("panels_*") if (p / PANEL).is_dir())[-1]


def windows(panel: Path, split: str, t_in: int, fp: pd.DataFrame | None = None):
    """sliding_windows of one half of the companies (split "train" / "test" of preprocess.py). With fp: windows of the
    fingerprint, on the firm-quarters of the balance sheet file. Without: windows of the amounts and their yearly
    changes of the three statements (used only to define the universe)."""
    S = load_statements(panel, split)
    df = S["bs"].merge(S["ins"], on=KEYS).merge(S["cf"], on=KEYS) if fp is None else fp.merge(S["bs"][KEYS])
    return sliding_windows(df, t_in)


def universe(panel: Path, t_in: int, fp: pd.DataFrame) -> pd.MultiIndex:
    """
    The (ticker, quarter) pairs that can be used: those that end a complete window of t_in quarters in the amounts and
    their yearly changes of all three statements, and in the fingerprint, up to LAST_QUARTER. Cached in
    data/out/similarity/universe_<panels>_<panel>_tin<t_in>_to<LAST_QUARTER>.parquet (delete it after a new download).
    """
    path = DATA / "out" / "similarity" / f"universe_{panel.parent.name}_{panel.name}_tin{t_in}_to{LAST_QUARTER}.parquet"
    if path.exists():
        return pd.MultiIndex.from_frame(pd.read_parquet(path))
    print("[INFO] Building the universe (once per panel and t_in)")
    keys = None
    for use_fp in (False, True):
        parts = []
        for split in ("train", "test"):
            X, meta, cols = windows(panel, split, t_in, fp if use_fp else None)
            parts.append(attach_fingerprints(X, meta, cols, fp).meta[KEYS])       # the last quarter has a fingerprint
        k = pd.concat(parts)
        keys = k if keys is None else keys.merge(k)
    keys = keys[keys.quarter.le(LAST_QUARTER)]
    path.parent.mkdir(parents=True, exist_ok=True)
    keys.reset_index(drop=True).to_parquet(path)
    return pd.MultiIndex.from_frame(keys)


def balanced_firms(keep: pd.MultiIndex) -> pd.DataFrame:
    """ticker, split: the firms with a window at every quarter of the universe `keep`, a random UNSEEN_SHARE of them
    "unseen" (the others "train")."""
    k = keep.to_frame(index=False)
    n = k.groupby("ticker").quarter.nunique()
    firms = np.sort(n.index[n.eq(k.quarter.nunique())].to_numpy())
    unseen = np.random.default_rng(SPLIT_SEED).permutation(firms)[:round(UNSEEN_SHARE * len(firms))]
    return pd.DataFrame({"ticker": firms, "split": np.where(np.isin(firms, unseen), "unseen", "train")})


def resplit(sets: dict, split: pd.Series) -> dict:
    """The windows of both sets regrouped by `split` (ticker -> "train" / "unseen"); firms not in it are dropped."""
    a, u = sets["train firms"], sets["unseen firms"]
    both = SimilarityData(torch.cat([a.X, u.X]), np.concatenate([a.target, u.target]),
                          pd.concat([a.meta, u.meta], ignore_index=True), a.cols)
    s = both.meta.ticker.map(split)
    return {"train firms": both.subset(s.eq("train").to_numpy()), "unseen firms": both.subset(s.eq("unseen").to_numpy())}


def sample_sets(panel: Path, t_in: int) -> tuple[dict, pd.DataFrame]:
    """
    {"train firms", "unseen firms"}: the windows of the balanced sample (input X and Pillar 1 target), and the
    fingerprint they are made of. Files in <panel>/targets:
      fingerprint_slogz.parquet            every firm (preprocess/fingerprint.py); defines the universe
      firms_balanced.parquet               the balanced firms and their split; written here the first time
      fingerprint_slogz_balanced.parquet   the fingerprint of those firms, scaled on them (fingerprint.py --firms)
    """
    targets = panel / "targets"
    keep = universe(panel, t_in, pd.read_parquet(targets / "fingerprint_slogz.parquet"))
    # shortcut: the firm list and its fingerprint are not tied to LAST_QUARTER; delete both if it changes
    firms, fp_file = targets / "firms_balanced.parquet", targets / "fingerprint_slogz_balanced.parquet"
    if not fp_file.exists():
        balanced_firms(keep).to_parquet(firms)
        raise SystemExit(f"{fp_file} missing: run src/preprocess/fingerprint.py --panel {panel} --firms {firms}")
    fp = pd.read_parquet(fp_file)
    sets = {}
    for name, split in (("train firms", "train"), ("unseen firms", "test")):
        d = attach_fingerprints(*windows(panel, split, t_in, fp), fp)
        sets[name] = d.subset(pd.MultiIndex.from_frame(d.meta[KEYS]).isin(keep))
    return resplit(sets, pd.read_parquet(firms).set_index("ticker").split), fp


def main() -> None:
    """Sample, targets, training, then encoder.pth and config.json in the run folder."""
    args = parse_args()
    torch.manual_seed(args.seed)
    panels_dir = args.panels_dir or latest_panels_dir()
    panel = panels_dir / PANEL
    out = args.out or DATA / "out" / "similarity" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Data: {panel}")

    sets, fp = sample_sets(panel, args.t_in)
    tr = attach_window_deltas(sets["train firms"], fp, args.t_in, args.pillar2_decay)
    train_data = tr.subset(tr.meta.quarter.le(args.cutoff).to_numpy())
    for name, d in [("training", train_data), ("unseen firms", sets["unseen firms"])]:
        print(f"[INFO] {name:13s} {len(d.meta):7,} windows, {d.meta.ticker.nunique():5,} firms, "
              f"{d.meta.quarter.min()}–{d.meta.quarter.max()}")

    model = XBRL2VecEncoder(
        T=args.t_in,
        F=train_data.X.shape[-1],
        latent_dim=args.latent_dim,
        hidden=args.hidden,
        )
    history = train(
        model=model,
        data=train_data,
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        device=DEVICE,
        seed=args.seed,
        lambda2=args.lambda2,
        input_mask=args.input_mask,
        )
    torch.save(model.state_dict(), out / "encoder.pth")
    (out / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                 "panels_dir": str(panels_dir), "loss_history": history}, indent=2))
    print(f"[INFO] Run: {out}")


if __name__ == "__main__":
    main()
