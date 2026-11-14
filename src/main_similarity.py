"""
XBRL2Vec, Pillar 1 benchmark: the DLinear + FiLM encoder (company data + macro) trained so that cosine similarity
between embeddings reproduces the fingerprint similarity of firms in the same quarter (services/similarity.py).

    cd src && ../xbrl2vec/bin/python main_similarity.py --panel filled

Splits: training firms (preprocess.py's company split) up to --cutoff (default: 8 quarters before the last); evaluated
on the training firms (in sample), unseen firms (up to the cutoff), the training firms after the cutoff (out of time)
and unseen firms after it. Variants: the model with its macro input real, zeroed and shuffled across firm-quarters
(the macro ablation); the same encoder trained without macro; the raw fingerprint; PCA of the fingerprint to the
same dimension (the ceiling: it compresses the target itself); PCA of the model's input window to the same dimension
(the fair baseline: same information as the model, no learning); for the Pillar 2 metrics, the past change of the
fingerprint ("firms react like last time"). Add-ons, all off by default (the defaults reproduce the 31 Oct starting
point, except FiLM, which starts at the identity since 31 Oct): --loss pillar1+2 (second target), --film_init default
(the paper's FiLM), --macro_weight_mode none, --macro_scale train, --weight_moves, --embargo, --scenario
(scenario-conditioned Pillar 2, with --scenario_shift as its placebo), --pillar2 window_deltas (Pillar 2 = the
year-on-year moves inside the input window), --skip_macro_model (only the encoder without macro), --pair_loss neighbourhood, --input_mask, --train_frac
(a random share of the training firms: data stability). Compare runs with
compare_runs.py. Results:
printed, written to <out>/results.csv and logged to MLflow.
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from models.similarity_encoder import ResponseHead, ScenarioHead, SimilarityEncoder
from services.config import DEVICE
from services.data import ColumnFilter, create_aligned_dataset, filter_columns, load_raw_data, load_test_data
from services.similarity import (SimilarityData, attach_changes, attach_fingerprints, attach_scenario,
                                 attach_window_deltas, attach_window_mean, embed, evaluate, sic2_peers_r2, train)
from services.transforms import transform_dataset

DATA = Path(__file__).resolve().parents[1] / "data"
LAST_QUARTER = "2025Q4"                                                # whole years: the 2026 quarters are left out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", choices=["filled", "complete"], default="filled")
    p.add_argument("--panels_dir", type=Path, default=None, help="Default: the latest data/in/panels_* with exog.parquet.")
    p.add_argument("--t_in", type=int, default=12, help="Quarters in each input window.")
    p.add_argument("--inputs", choices=["levels", "diff_y", "both", "common_size"], default="common_size",
                   help="Company inputs: the preprocessed values, their year-on-year changes, both, or common-size "
                        "statements (the fingerprint ratios: lines / total assets or / revenue; the default since "
                        "31 Oct 2026, see docs/pillar1_benchmark.md). Macro: always changes.")
    p.add_argument("--hidden", type=int, default=128,
                   help="Width of a 2-layer MLP between FiLM and the embedding (default 128 since 31 Oct 2026; "
                        "0: linear, as ForecastingAE).")
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256, help="Firms per batch (all from one quarter).")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cutoff", default=None, help="Last training quarter (default: 8 quarters before the last).")
    p.add_argument("--macro_weight_mode", choices=["corr", "ridge", "xgboost", "none"], default="corr",
                   help="Firm-specific weighting of the macro inputs (none: the raw macro, the same for every firm).")
    p.add_argument("--macro_scale", choices=["global", "train"], default="global",
                   help="Macro standardisation: as in exog.parquet (all quarters), or refitted on the training windows "
                        "only (no look-ahead in the scaling).")
    p.add_argument("--film_init", choices=["default", "identity"], default="identity",
                   help="FiLM conditioner: IdentityFiLM (starts as the identity: macro has no effect until it lowers "
                        "the loss; the default since 31 Oct 2026, docs/pillar1_benchmark.md) or 'default', the paper's.")
    p.add_argument("--loss", choices=["pillar1", "pillar1+2"], default="pillar1",
                   help="pillar1: fingerprint similarity (pair_loss). pillar1+2: (1 - lambda2) x that + lambda2 x the "
                        "similarity of the Pillar 2 target (--pillar2; default the fingerprint changes over the next --horizon "
                        "quarters) (two_target_loss).")
    p.add_argument("--lambda2", type=float, default=0.5,
                   help="Weight of the Pillar 2 term (loss pillar1+2), in [0, 1]: 0 = Pillar 1 only, 1 = Pillar 2 only.")
    p.add_argument("--lambda_form", choices=["convex", "additive"], default="convex",
                   help="convex: (1 - lambda2) L1 + lambda2 L2. additive: L1 + lambda2 L2, the form of the runs before "
                        "1 Nov 2026 (trains like convex with lambda2 / (1 + lambda2)).")
    p.add_argument("--weight_moves", action="store_true",
                   help="Pillar 2 term: weight pairs by the size of the two changes (firms that barely move count less).")
    p.add_argument("--horizon", type=int, default=4, help="Quarters ahead of the Pillar 2 target (fingerprint_change_h<h>).")
    p.add_argument("--pillar2", choices=["future", "window_deltas"], default="future",
                   help="Pillar 2 target (loss pillar1+2 and the fit2 metrics). future: the fingerprint change over the "
                        "next --horizon quarters. window_deltas: the year-on-year changes of the fingerprint inside the "
                        "input window, f(s) - f(s - 4) for its last t_in - 4 quarters, stacked (firms that moved in the "
                        "same direction at the same time; nothing after t, so no embargo); the next-horizon change "
                        "stays as the fit2_future metric.")
    p.add_argument("--fingerprint", choices=["zscore", "rank", "slogz"], default="zscore",
                   help="rank: the fingerprint with rank normal scores (targets/fingerprint_rank.parquet, from "
                        "preprocess/fingerprint.py --scaling rank) as targets and common-size inputs; slogz: the "
                        "z-score of the signed log of the ratios (fingerprint_slogz.parquet, --scaling slogz).")
    p.add_argument("--sample", choices=["all", "balanced"], default="all",
                   help="balanced: only the firms with a window at every quarter up to LAST_QUARTER (full data from "
                        "2013Q1), split anew with 15 %% unseen (seed 0), on the fingerprint rescaled on them "
                        "(targets/fingerprint*_balanced.parquet, from preprocess/fingerprint.py --firms).")
    p.add_argument("--pillar2_decay", type=float, default=1.0,
                   help="--pillar2 window_deltas: weight decay ** k on the move ending k quarters before t, so that recent "
                        "moves count more in the Pillar 2 cosine (1: all moves equal, the default).")
    p.add_argument("--pillar2_fingerprint", choices=["same", "zscore", "rank", "slogz"], default="same",
                   help="--pillar2 window_deltas: the fingerprint the moves are computed on (same: --fingerprint's, the "
                        "default). --fingerprint rank --pillar2_fingerprint zscore: levels compared by rank (inputs and "
                        "Pillar 1), moves measured in z-scores, whose size the rank shrinks.")
    # not an option: the fingerprint inputs (common_size) go to the encoder as they are since 8 Nov 2026; runs from
    # before had a second signed log on them, and apply_reuse sets this to True to rebuild their inputs
    p.set_defaults(input_slog=False)
    p.add_argument("--embargo", type=int, default=None,
                   help="Training quarters end this many quarters before the cutoff, so that no training label falls "
                        "after it (default: --horizon with loss pillar1+2 and the future Pillar 2, 0 otherwise).")
    p.add_argument("--pillar1", choices=["last", "window_mean"], default="last",
                   help="Pillar 1 target: the fingerprint of the window's last quarter, or its average over the "
                        "window's quarters (window_mean: the firm's structure over the window; fit is then measured "
                        "against it).")
    p.add_argument("--pair_loss", choices=["mse", "neighbourhood"], default="mse",
                   help="Form of each pillar's loss: mse, the squared error between all pair cosines; neighbourhood, "
                        "each firm's softmax(cos / --tau) over the batch matches the target's (KL): the nearest firms "
                        "count most (peers are local).")
    p.add_argument("--tau", type=float, default=0.1, help="Temperature of --pair_loss neighbourhood.")
    p.add_argument("--input_mask", type=float, default=0.0,
                   help="Training only: this share of the input cells (ratio x quarter) set missing (0, as in the "
                        "data) at random in every batch, targets unchanged (robustness to holes).")
    p.add_argument("--train_frac", type=float, default=1.0,
                   help="Train on this random share of the training firms (drawn with --subsample_seed); the evaluation "
                        "sets are unchanged (data stability: how much the embedding depends on the sample of firms).")
    p.add_argument("--subsample_seed", type=int, default=0, help="Seed of the --train_frac draw of firms.")
    p.add_argument("--skip_macro_model", action="store_true",
                   help="Train and score only the encoder without macro (DLinear -> MLP), not the macro one.")
    p.add_argument("--scenario", action="store_true",
                   help="Scenario-conditioned Pillar 2 (loss pillar1+2): the Pillar 2 term compares cos of the embeddings "
                        "reweighted by the realised macro change over the next --horizon quarters (ScenarioHead, "
                        "z~ = (1 + W s + b) * z, starts at the identity); z itself never sees it.")
    p.add_argument("--scenario_cols", nargs="+", default=["DFF", "CPIAUCSL", "GDP", "UNEMPLOY"],
                   help="Macro series of the scenario (exog.parquet; their year-on-year change ending at t + horizon).")
    p.add_argument("--scenario_past", action="store_true",
                   help="The scenario head also gets the macro change over the past --horizon quarters (known at t): "
                        "the Pillar 2 metric may depend on what just happened (transmission lags).")
    p.add_argument("--scenario_past_only", action="store_true",
                   help="The scenario head gets only the past macro change (implies --scenario_past): nothing after t is "
                        "used anywhere, so z and the Pillar 2 metric are both available in real time.")
    p.add_argument("--scenario_past_lags", type=int, nargs="+", default=[0],
                   help="Past macro of the scenario head: the year-on-year change ending at t - lag, for each lag "
                        "(e.g. 0 4: the last year and the year before, the phase of the cycle).")
    p.add_argument("--response_weight", type=float, default=0.0,
                   help="Auxiliary loss (needs --scenario with the realised scenario): weight of the error of a "
                        "ResponseHead predicting the next-h-quarter change from z and the realised macro.")
    p.add_argument("--macro_conditioning", choices=["film", "cross_attention", "mean_pool"], default="film",
                   help="How the macro enters the encoder: FiLM (--film_init) or cross-attention of the firm's encoding "
                        "over the macro quarters (MacroCrossAttention, zero-initialised), or mean_pool: the same with "
                        "every quarter weighted equally (an additive, common macro read; ablation of the attention).")
    p.add_argument("--macro_target", choices=["both", "pillar2"], default="both",
                   help="pillar2: the embedding z (Pillar 1) stays macro-free; the macro-conditioned embedding is used "
                        "only in the Pillar 2 term (and by the scenario head).")
    p.add_argument("--macro_dropout", type=float, default=0.0,
                   help="Dropout on the macro embedding before FiLM (limits memorising the training quarters).")
    p.add_argument("--scenario_shift", type=int, default=0,
                   help="Placebo: the scenario of quarter t is the one of t - shift (same series, wrong dates).")
    p.add_argument("--skip_in_sample", action="store_true",
                   help="Do not score the training windows (the slowest set; G1 is decided out of time).")
    p.add_argument("--norm_mode", choices=["global", "per_ticker"], default="global")
    p.add_argument("--mlflow_uri", default="http://localhost:5000")
    p.add_argument("--experiment", default="Pillar1_similarity")
    p.add_argument("--out", type=Path, default=None, help="Default: data/out/similarity/<timestamp>.")
    p.add_argument("--reuse", default=None,
                   help="Re-score an earlier run with the current metrics, without training: its saved encoders and its "
                        "configuration (data, inputs, encoder, loss) are used; the new run folder records reused_from.")
    return p.parse_args()


#: fingerprint file of each --fingerprint option
FINGERPRINT = {"zscore": "fingerprint.parquet", "rank": "fingerprint_rank.parquet", "slogz": "fingerprint_slogz.parquet"}

#: the settings a re-scored run takes from the original run (everything that defines its data and its encoders)
REUSED = ["panel", "panels_dir", "t_in", "inputs", "hidden", "latent_dim", "epochs", "batch_size", "learning_rate", "seed",
          "cutoff", "macro_weight_mode", "macro_scale", "film_init", "loss", "lambda2", "lambda_form", "weight_moves",
          "horizon", "embargo", "norm_mode", "scenario", "scenario_cols", "scenario_shift", "scenario_past",
          "macro_dropout", "macro_conditioning", "scenario_past_only", "response_weight", "scenario_past_lags",
          "macro_target", "pillar2", "skip_macro_model", "pair_loss", "tau", "input_mask", "pillar1", "train_frac",
          "subsample_seed", "pillar2_decay", "fingerprint", "pillar2_fingerprint", "sample"]


def apply_reuse(args) -> dict:
    """With --reuse: copy the original run's settings into args (defaults of options it predates are kept)."""
    cfg = json.loads((DATA / "out" / "similarity" / args.reuse / "config.json").read_text())
    types = {"t_in": int, "hidden": int, "latent_dim": int, "epochs": int, "batch_size": int, "seed": int, "horizon": int,
             "learning_rate": float, "lambda2": float, "weight_moves": lambda v: v == "True",
             "panels_dir": lambda v: None if v == "None" else Path(v), "embargo": lambda v: None if v == "None" else int(v),
             "cutoff": lambda v: None if v == "None" else v, "scenario": lambda v: v == "True",
             "scenario_cols": ast.literal_eval, "scenario_shift": int, "scenario_past": lambda v: v == "True",
             "macro_dropout": float, "scenario_past_only": lambda v: v == "True", "response_weight": float,
             "scenario_past_lags": ast.literal_eval, "skip_macro_model": lambda v: v == "True", "tau": float, "input_mask": float,
             "train_frac": float, "subsample_seed": int, "pillar2_decay": float}
    for k in REUSED:
        if k in cfg:
            setattr(args, k, types.get(k, str)(cfg[k]))
    if "film_init" not in cfg:                                          # runs from before the option used the paper's FiLM
        args.film_init = "default"
    if "hidden" not in cfg:                                             # runs from before the option were linear
        args.hidden = 0
    if "lambda_form" not in cfg:                                        # runs from before the option: L1 + lambda2 L2
        args.lambda_form = "additive"
    args.input_slog = cfg.get("input_slog", "True") == "True"         # runs before 8 Nov: second signed log on the inputs
    args.cutoff = cfg.get("cutoff", args.cutoff)                       # the cutoff the run actually used
    return cfg


def scenario_width(args) -> int:
    past = len(args.scenario_cols) * len(getattr(args, "scenario_past_lags", [0]))
    if getattr(args, "scenario_past_only", False):
        return past
    return len(args.scenario_cols) + (past if args.scenario_past else 0)


def scenarios_raw(args, quarters: list[str], shift: int | None = None) -> pd.DataFrame:
    """quarter t -> the realised macro scenario of its Pillar 2 target: the year-on-year change of each --scenario_cols
    series ending at t + horizon (of t - shift + horizon for the placebo); with --scenario_past also the change ending
    at t (columns <series>_past). A quarter missing in exog.parquet is interpolated from its neighbours."""
    shift = args.scenario_shift if shift is None else shift
    e = pd.read_parquet(args.panels_dir / "exog.parquet").set_index("observation_date")
    X = e[[c + "_DIFF_Y" for c in args.scenario_cols]].set_axis(args.scenario_cols, axis=1)
    n = lambda q: int(q[:4]) * 4 + int(q[-1]) - 1
    q_of = lambda k: f"{k // 4}Q{k % 4 + 1}"
    X = X.reindex([q_of(k) for k in range(n(X.index.min()), n(X.index.max()) + 1)]).interpolate(limit_area="inside")
    S = X.reindex([q_of(n(q) + args.horizon - shift) for q in quarters]).set_axis(quarters)
    if args.scenario_past or getattr(args, "scenario_past_only", False):
        past = pd.concat([X.reindex([q_of(n(q) - lag - shift) for q in quarters]).set_axis(quarters)
                          .add_suffix("_past" if lag == 0 else f"_past{lag}") for lag in getattr(args, "scenario_past_lags", [0])], axis=1)
        S = past if getattr(args, "scenario_past_only", False) else pd.concat([S, past], axis=1)
    return S


def scenarios(args, quarters: list[str], train_quarters: list[str]) -> pd.DataFrame:
    """scenarios_raw standardised on the training quarters."""
    S = scenarios_raw(args, quarters)
    mu, sd = S.loc[train_quarters].mean(), S.loc[train_quarters].std()
    return (S - mu) / sd


def revenue_growth(panel: Path) -> pd.Series:
    """Year-on-year log growth of quarterly revenue, (ticker, quarter) -> value (for the peer test)."""
    ins = pd.read_parquet(panel / "ins-raw.parquet")
    q = sorted(c for c in ins.columns if len(str(c)) == 6 and str(c)[4] == "Q")
    rev = ins[ins.features == "Total Revenue"].set_index("ticker")[q].astype(float)
    rev = rev.where(rev > 0)
    g = np.log(rev / rev.shift(4, axis=1)).replace([np.inf, -np.inf], np.nan)
    g = g.clip(g.quantile(0.01), g.quantile(0.99), axis=1)
    return g.stack().rename_axis(["ticker", "quarter"])


def windows(args, panel: Path, inputs: str, fp: pd.DataFrame, transform: bool = True) -> dict:
    """{"train firms", "unseen firms"}: SimilarityData of every complete t_in-quarter window with a fingerprint."""
    bs, is_, cf, macro, _ = load_raw_data(panel, args.panels_dir, norm_mode=args.norm_mode)
    cond = {"levels": ColumnFilter.LEVELS, "diff_y": ColumnFilter.DIFF_Y, "both": ColumnFilter.BOTH,
            "common_size": ColumnFilter.LEVELS}[inputs]
    bs, is_, cf, macro = filter_columns(bs, is_, cf, macro, cond)
    bs_t, is_t, cf_t = load_test_data(panel, macro, norm_mode=args.norm_mode, cond=cond)
    if inputs == "common_size":
        # the fingerprint ratios as input series, on the same firm-quarters as the other inputs
        bs, bs_t = fp.merge(bs[["ticker", "quarter"]]), fp.merge(bs_t[["ticker", "quarter"]])
        is_ = cf = bs[["ticker", "quarter"]]                          # no other statements: all ratios are in bs
        is_t = cf_t = bs_t[["ticker", "quarter"]]
    sets = {}
    for name, (b, i, c) in {"train firms": (bs, is_, cf), "unseen firms": (bs_t, is_t, cf_t)}.items():
        ds = create_aligned_dataset(b, i, c, macro, seq_len=args.t_in)
        if transform:
            ds = transform_dataset(ds, macro_weight_mode=args.macro_weight_mode,   # amounts need the signed log, the
                                   fin_slog=inputs != "common_size" or getattr(args, "input_slog", False))  # fingerprint not
        sets[name] = attach_fingerprints(ds, fp)
    return sets


def universe(args, panel: Path, fp: pd.DataFrame) -> pd.MultiIndex:
    """
    The firm-quarters every run trains and is evaluated on: those with a complete window under every input option
    (amounts and changes, which is the strictest of levels / diff_y / both, and common-size). Without it, runs with
    different inputs would be scored on different firms. Windows end at LAST_QUARTER at the latest. Cached per panel,
    t_in and last quarter (the cache without a last quarter in its name is the one of the runs before 14 Nov, to 2025Q3).
    """
    path = DATA / "out" / "similarity" / f"universe_{args.panels_dir.name}_{args.panel}_tin{args.t_in}_to{LAST_QUARTER}.parquet"
    if path.exists():
        return pd.MultiIndex.from_frame(pd.read_parquet(path))
    print("[INFO] Building the evaluation universe (once per panel and t_in)")
    keys = None
    for inputs in ("both", "common_size"):
        k = pd.concat([d.meta[["ticker", "quarter"]] for d in windows(args, panel, inputs, fp, transform=False).values()])
        keys = k if keys is None else keys.merge(k)
    keys = keys[keys.quarter.le(LAST_QUARTER)]
    path.parent.mkdir(parents=True, exist_ok=True)
    keys.reset_index(drop=True).to_parquet(path)
    return pd.MultiIndex.from_frame(keys)


BALANCED_UNSEEN, SPLIT_SEED = 0.15, 0                                  # --sample balanced: share of unseen firms, draw


def balanced_firms(keep: pd.MultiIndex) -> pd.DataFrame:
    """ticker, split of --sample balanced: the firms with a window at every quarter of the universe (full data from
    2013Q1 to LAST_QUARTER), a random 15 % of them unseen."""
    k = keep.to_frame(index=False)
    n = k.groupby("ticker").quarter.nunique()
    firms = np.sort(n.index[n.eq(k.quarter.nunique())].to_numpy())
    unseen = np.random.default_rng(SPLIT_SEED).permutation(firms)[:round(BALANCED_UNSEEN * len(firms))]
    return pd.DataFrame({"ticker": firms, "split": np.where(np.isin(firms, unseen), "unseen", "train")})


def resplit(sets: dict, split: pd.Series) -> dict:
    """The windows of both sets regrouped by split (ticker -> "train" / "unseen"); firms not in it are dropped."""
    a, u = sets["train firms"], sets["unseen firms"]
    both = SimilarityData(torch.cat([a.X_fin, u.X_fin]), torch.cat([a.X_macro, u.X_macro]),
                          np.concatenate([a.target, u.target]), pd.concat([a.meta, u.meta], ignore_index=True),
                          a.fin_cols, a.macro_cols)
    s = both.meta.ticker.map(split)
    return {"train firms": both.subset(s.eq("train").to_numpy()), "unseen firms": both.subset(s.eq("unseen").to_numpy())}


def sample_sets(args, panel: Path) -> tuple[dict, pd.DataFrame]:
    """{"train firms", "unseen firms"} on the universe, and the fingerprint they use. --sample balanced: the firms and
    split of targets/firms_balanced.parquet, on the fingerprint rescaled on them."""
    fp = pd.read_parquet(panel / "targets" / FINGERPRINT[args.fingerprint])
    keep = universe(args, panel, fp)                                   # the same firm-quarters for every run
    if args.sample == "balanced":
        # shortcut: the firm list and its fingerprint are not tied to LAST_QUARTER; delete both if it changes
        firms = panel / "targets" / "firms_balanced.parquet"
        path = firms.with_name(FINGERPRINT[args.fingerprint].replace(".parquet", "_balanced.parquet"))
        if not path.exists():
            balanced_firms(keep).to_parquet(firms)
            raise SystemExit(f"{path} missing: run src/preprocess/fingerprint.py --panel {panel} --scaling "
                             f"{args.fingerprint} --firms {firms} --out {path}")
        fp = pd.read_parquet(path)
    sets = windows(args, panel, args.inputs, fp)
    sets = {k: v.subset(pd.MultiIndex.from_frame(v.meta[["ticker", "quarter"]]).isin(keep)) for k, v in sets.items()}
    if args.sample == "balanced":
        sets = resplit(sets, pd.read_parquet(firms).set_index("ticker").split)
    return sets, fp


def main() -> None:
    args = parse_args()
    original = apply_reuse(args) if args.reuse else None
    torch.manual_seed(args.seed)
    if args.panels_dir is None:
        args.panels_dir = sorted(p for p in (DATA / "in").glob("panels_*") if (p / "exog.parquet").exists())[-1]
    panel = args.panels_dir / args.panel
    out = args.out or DATA / "out" / "similarity" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Data: {panel}")

    # ---- windows of t_in quarters, company split (preprocess.py's or --sample's), fingerprint of the last quarter ----
    sets, fp = sample_sets(args, panel)
    change_file = panel / "targets" / f"fingerprint_change_h{args.horizon}.parquet"
    if change_file.exists():                                           # Pillar 2 target (and its metrics) for every run
        ch = pd.read_parquet(change_file)
        sets = {k: attach_changes(v, ch, args.horizon) for k, v in sets.items()}
    elif args.loss == "pillar1+2" and args.pillar2 == "future":
        raise SystemExit(f"{change_file} missing: run src/preprocess/fingerprint.py --panel {panel}")
    if args.pillar2 == "window_deltas":                                # Pillar 2 = moves inside the window
        if args.scenario or args.response_weight:
            raise SystemExit("--scenario and --response_weight are defined for the future Pillar 2 target")
        fp2 = fp if args.pillar2_fingerprint in ("same", args.fingerprint) else \
            pd.read_parquet(panel / "targets" / FINGERPRINT[args.pillar2_fingerprint])
        sets = {k: attach_window_deltas(v, fp2, args.t_in, args.pillar2_decay) for k, v in sets.items()}
    if args.pillar1 == "window_mean":                                  # Pillar 1 = structure over the window
        sets = {k: attach_window_mean(v, fp, args.t_in) for k, v in sets.items()}
    quarters = sorted(set(sets["train firms"].meta.quarter) | set(sets["unseen firms"].meta.quarter))
    cutoff = args.cutoff or quarters[-9]
    tr, te = sets["train firms"], sets["unseen firms"]
    data = {"train (in sample)": tr.subset(tr.meta.quarter.le(cutoff).to_numpy()),
            "unseen firms": te.subset(te.meta.quarter.le(cutoff).to_numpy()),
            "out of time": tr.subset(tr.meta.quarter.gt(cutoff).to_numpy()),
            "unseen firms, out of time": te.subset(te.meta.quarter.gt(cutoff).to_numpy())}
    for k, d in data.items():
        print(f"[INFO] {k:26s} {len(d.meta):7,} windows, {d.meta.ticker.nunique():5,} firms, {d.meta.quarter.min()}–{d.meta.quarter.max()}")
    embargo = args.embargo if args.embargo is not None else (
        args.horizon if args.loss == "pillar1+2" and args.pillar2 == "future" else 0)
    last_train = quarters[quarters.index(cutoff) - embargo]
    train_data = data["train (in sample)"].subset(data["train (in sample)"].meta.quarter.le(last_train).to_numpy())
    print(f"[INFO] training on quarters up to {last_train} (cutoff {cutoff}, embargo {embargo})")
    if args.train_frac < 1.0:                                          # data stability: a random share of the firms
        firms = np.sort(train_data.meta.ticker.unique())
        pick = set(firms[np.random.default_rng(args.subsample_seed).random(len(firms)) < args.train_frac])
        train_data = train_data.subset(train_data.meta.ticker.isin(pick).to_numpy())
        print(f"[INFO] --train_frac {args.train_frac}: {len(pick):,} of {len(firms):,} training firms "
              f"(draw {args.subsample_seed}), {len(train_data.meta):,} windows")
    if args.macro_scale == "train":                                    # scaling fitted on the training windows only
        mu, sd = train_data.X_macro.mean(dim=(0, 1)), train_data.X_macro.std(dim=(0, 1)).clamp_min(1e-8)
        for d in list(data.values()) + [train_data]:
            d.X_macro = (d.X_macro - mu) / sd
    F_dim, M_dim = train_data.X_fin.shape[-1], train_data.X_macro.shape[-1]
    if args.scenario:
        if args.loss != "pillar1+2":
            raise SystemExit("--scenario needs --loss pillar1+2")
        scen = scenarios(args, quarters, [q for q in quarters if q <= last_train])
        for d in list(data.values()) + [train_data]:
            attach_scenario(d, scen)
        print(f"[INFO] scenario: {', '.join(args.scenario_cols)} over the next {args.horizon} quarters"
              + (" and the past ones" if args.scenario_past else "")
              + (f", PLACEBO shifted by {args.scenario_shift} quarters" if args.scenario_shift else ""))

    # ---- the two encoders ----
    models = {"macro": SimilarityEncoder(args.t_in, F_dim, M_dim, args.latent_dim, use_macro=True, hidden=args.hidden,
                                         film_init=args.film_init, macro_dropout=args.macro_dropout,
                                         conditioning=args.macro_conditioning, macro_target=args.macro_target),
              "no macro": SimilarityEncoder(args.t_in, F_dim, M_dim, args.latent_dim, use_macro=False, hidden=args.hidden)}
    if args.skip_macro_model:
        models.pop("macro")
    heads = {name: ScenarioHead(scenario_width(args), args.latent_dim) for name in models} if args.scenario else {}
    if args.response_weight and (not args.scenario or args.scenario_past_only):
        raise SystemExit("--response_weight needs --scenario with the realised scenario (not --scenario_past_only)")
    resps = {name: ResponseHead(args.latent_dim, len(args.scenario_cols), train_data.target2.shape[1])
             for name in models} if args.response_weight else {}
    history = {}
    if args.reuse:                                                     # re-score: the original run's encoders
        src = DATA / "out" / "similarity" / args.reuse
        for name, model in models.items():
            model.load_state_dict(torch.load(src / f"encoder_{name.replace(' ', '_')}.pth", map_location="cpu"))
        for name, head in heads.items():
            head.load_state_dict(torch.load(src / f"scenario_{name.replace(' ', '_')}.pth", map_location="cpu"))
        history = original.get("loss_history", {})
        print(f"[INFO] Re-scoring the encoders of {args.reuse} (no training)")
    else:
        for name, model in models.items():
            print(f"[INFO] Training the encoder with {name}")
            history[name] = train(model, train_data, args.epochs, args.learning_rate, args.batch_size, DEVICE,
                                  use_macro=(name == "macro"), seed=args.seed, loss=args.loss, lambda2=args.lambda2,
                                  weight_moves=args.weight_moves, lambda_form=args.lambda_form, head=heads.get(name),
                                  resp=resps.get(name), resp_weight=args.response_weight,
                                  resp_cols=len(args.scenario_cols), pair_form=args.pair_loss, tau=args.tau,
                                  input_mask=args.input_mask)

    for name, model in models.items():
        torch.save(model.state_dict(), out / f"encoder_{name.replace(' ', '_')}.pth")
    for name, head in heads.items():
        torch.save(head.state_dict(), out / f"scenario_{name.replace(' ', '_')}.pth")
    for name, r in resps.items():
        torch.save(r.state_dict(), out / f"response_{name.replace(' ', '_')}.pth")

    # ---- evaluation ----
    growth = revenue_growth(panel)
    others = {}
    for name, f in [("p2", "pillar2_episodes.parquet"), ("p3", "pillar3_comovement.parquet")]:
        if (panel / "targets" / f).exists():
            others[name] = pd.read_parquet(panel / "targets" / f)
    sector = pd.read_parquet(args.panels_dir / "metadata.parquet").drop_duplicates("ticker").set_index("ticker").sector
    pca = PCA(min(args.latent_dim, train_data.target.shape[1]), random_state=args.seed).fit(train_data.target)
    flat = lambda d: d.X_fin.reshape(len(d.X_fin), -1).numpy()                    # the model's input window, flattened
    pca_in = PCA(args.latent_dim, random_state=args.seed).fit(flat(train_data))
    variants = {
        "model, macro real": lambda d: embed(models["macro"], d, DEVICE, "real"),
        "model, macro zero": lambda d: embed(models["macro"], d, DEVICE, "zero"),
        "model, macro shuffled": lambda d: embed(models["macro"], d, DEVICE, "shuffled", seed=args.seed),
        "model trained without macro": lambda d: embed(models["no macro"], d, DEVICE, "none"),
        "raw fingerprint": lambda d: d.target,
        f"PCA of fingerprint ({pca.n_components_})": lambda d: pca.transform(d.target),
        f"PCA of input window ({args.latent_dim})": lambda d: pca_in.transform(flat(d)),
    }
    if args.skip_macro_model:
        variants = {k: v for k, v in variants.items() if not k.startswith("model, macro")}
    if args.pillar2 == "window_deltas":
        # the Pillar 2 target itself (fit: how much structure the moves carry; fit2_future: do past moves predict the
        # next ones?) and the fair baseline for both pillars: PCA of [fingerprint, window deltas], each block scaled to
        # the same total variance on the training windows (the deltas have t_in - 4 times the columns)
        variants["window deltas (Pillar 2 target)"] = lambda d: d.target2
        ok = np.isfinite(train_data.target2).all(axis=1)
        scale = [np.sqrt(train_data.target[ok].var(0).sum()), np.sqrt(train_data.target2[ok].var(0).sum())]
        both = lambda d: np.concatenate([d.target / scale[0], d.target2 / scale[1]], axis=1)
        pca_both = PCA(args.latent_dim, random_state=args.seed).fit(both(train_data)[ok])

        def pca_both_of(d):                                            # NaN rows (unknown deltas) stay NaN: left out
            X = both(d)
            V = np.full((len(X), args.latent_dim), np.nan, np.float32)
            known = np.isfinite(X).all(axis=1)
            V[known] = pca_both.transform(X[known])
            return V
        variants[f"PCA of fingerprint and window deltas ({args.latent_dim})"] = pca_both_of
    for name, head in heads.items():                                   # scenario-reweighted embeddings (Pillar 2 term)
        m, label = ("real", "model") if name == "macro" else ("none", "no macro")
        for sc in ("real", "zero", "shuffled"):
            variants[f"{label}, scenario {sc}"] = (lambda d, mod=models[name], h=head, m=m, sc=sc:
                                                   embed(mod, d, DEVICE, m, seed=args.seed, head=h, scenario=sc))
    if train_data.past_change is not None:                             # Pillar 2 baseline: "firms react like last time"
        variants[f"past change (previous {args.horizon} quarters)"] = lambda d: d.past_change
    rows = []
    for set_name, d in data.items():
        if args.skip_in_sample and set_name == "train (in sample)":
            continue
        for v_name, f in variants.items():
            t0 = time.time()
            r = evaluate(f(d), d, growth=growth, others=others, seed=args.seed)
            rows.append({"set": set_name, "variant": v_name, **r})
            print(f"[INFO] {set_name:26s} {v_name:32s} " + " ".join(f"{k} {v:.3f}" for k, v in r.items()) + f"  ({time.time() - t0:.0f} s)")
        rows.append({"set": set_name, "variant": "SIC industry peers", "peers_r2": sic2_peers_r2(d, growth, sector)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)
    (out / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()}, "cutoff": cutoff,
                                                 "last_train_quarter": last_train,
                                                 "reused_from": args.reuse,
                                                 "universe": f"{args.panels_dir.name}/{args.panel}/tin{args.t_in}",
                                                 "loss_history": history}, indent=2))
    pd.set_option("display.width", 250)
    print(res.set_index(["set", "variant"]).round(3).to_string())
    print(f"[INFO] Results: {out}")

    try:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=f"{args.loss}-{args.inputs}-h{args.hidden}-film{args.film_init}-mw{args.macro_weight_mode}-tin{args.t_in}-z{args.latent_dim}-{args.panel}{'-scen' + str(args.scenario_shift) if args.scenario else ''}{'-p2win' if args.pillar2 == 'window_deltas' else ''}"):
            mlflow.log_params({k: str(v) for k, v in vars(args).items()} | {"cutoff": cutoff})
            for r in rows:
                for k, v in r.items():
                    if k not in ("set", "variant") and pd.notna(v):
                        mlflow.log_metric(f"{r['set']}/{r['variant']}/{k}".replace(" ", "_").replace(",", "").replace("(", "").replace(")", ""), float(v))
            for name, h in history.items():
                for e, loss in enumerate(h):
                    mlflow.log_metric(f"loss_{name.replace(' ', '_')}", loss, step=e)
            mlflow.log_artifacts(str(out))
    except Exception as e:                                       # no server: results are on disk anyway
        print(f"[WARN] MLflow logging skipped: {e}")


if __name__ == "__main__":
    main()
