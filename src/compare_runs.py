"""
The scorecard of the similarity runs (main_similarity.py): one row per trained encoder, always the same columns, on one
evaluation set (default: out of time, the training firms in the quarters after the cutoff).

    cd src && ../xbrl2vec/bin/python compare_runs.py                      # out of time
    cd src && ../xbrl2vec/bin/python compare_runs.py --set "unseen firms, out of time"

Columns:
  run            folder name (timestamp) and its configuration: loss, inputs, encoder (linear / MLP width), FiLM
                 initialisation and macro weighting when not the default, t_in, latent, panel
  model          "macro" (DLinear + FiLM with macro) or "no macro" (same encoder trained without the macro branch)
  Pillar 1 (structure):
  fit            Spearman correlation between embedding cosines and fingerprint cosines        does it learn the target?
  knn10          overlap of the 10 nearest neighbours, embedding vs fingerprint               is the local structure right?
  Pillar 2 (reaction), computed for every run, also those not trained on it:
  fit2           Spearman correlation between embedding cosines and the cosines of the run's Pillar 2 target: the
                 fingerprint changes over the next h quarters (default; who will react alike?) or, for runs with
                 "window deltas" in their config, the year-on-year moves inside the input window (who moved together?)
  knn10_2        overlap of the 10 nearest neighbours, embedding vs the Pillar 2 target
  fit2_future    fit2 against the next-h-quarter changes whatever the run's Pillar 2 target: the column comparable
                 across all runs
  reid4          share of firms whose nearest embedding 4 quarters later is their own          stable identity?
  stability      mean cosine of a firm's embedding in consecutive quarters                     does it jitter?
  p2, p3         Spearman correlation with the Pillar 2 episode / Pillar 3 target similarities
  d_zero, d_shuf (macro model only) fit with the real macro minus fit with the macro zeroed / shuffled across
                 firm-quarters: > 0 means the macro is used, < 0 that it hurts
  d2_zero, d2_shuf  the same for fit2
  Scenario-conditioned runs (--scenario), for the macro and the no-macro model:
  fit2_scen      fit2 of the embeddings reweighted by the realised scenario
  d2_scen_zero   fit2_scen minus fit2 with the scenario set to 0 (head bias only): > 0 means the scenario is used
  d2_scen_shuf   fit2_scen minus fit2 with each quarter given another quarter's scenario: the macro content test
Only runs scored on the fixed evaluation universe (main_similarity.universe) are shown; --all adds the older ones.
References: the raw fingerprint and its PCA (the ceiling for Pillar 1: they are the target) and the past change of the
fingerprint ("firms react like last time", the baseline for Pillar 2), from the latest run of each Pillar 2 target
(with window deltas also the target itself and the PCA of fingerprint and window deltas); the PCA of the model's
input window (the fair baseline: the model's information, no learning), once per kind of input. Written to
data/out/similarity/scorecard.csv (all evaluation sets).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RUNS = Path(__file__).resolve().parents[1] / "data" / "out" / "similarity"
METRICS = ["fit", "knn10", "fit2", "knn10_2", "fit2_future", "reid4", "stability", "p2", "p3"]
DELTAS = ["d_zero", "d_shuf", "d2_zero", "d2_shuf"]
SCEN = ["fit2_scen", "d2_scen_zero", "d2_scen_shuf"]


def config_label(cfg: dict) -> str:
    hidden = int(cfg.get("hidden", 0) or 0)
    extras = [f"FiLM {cfg.get('film_init', 'default')}"]                  # runs before the option: the paper's FiLM
    extras += [f"macro weights {cfg['macro_weight_mode']}" for _ in [0] if cfg.get("macro_weight_mode", "corr") != "corr"]
    extras += [f"macro scale {cfg['macro_scale']}" for _ in [0] if cfg.get("macro_scale", "global") != "global"]
    loss, p2 = cfg.get("loss", "pillar1"), cfg.get("pillar2", "future")
    if loss != "pillar1":
        lam = float(cfg.get("lambda2"))
        label = f"{lam:g}"
        if cfg.get("lambda_form", "additive") == "additive":            # runs before 1 Nov 2026: L1 + λ L2
            label += f" additive ≈ {lam / (1 + lam):.2f}"
        loss += (f" (λ {label}{', weighted' if cfg.get('weight_moves') == 'True' else ''}, "
                 + (f"h {cfg.get('horizon')})" if p2 == "future" else "window deltas)"))
    elif p2 != "future":
        extras.append("fit2 on window deltas")
    if cfg.get("scenario") == "True":
        shift = int(cfg.get("scenario_shift", 0))
        extras.append(("past-macro head" if cfg.get("scenario_past_only") == "True" else "scenario")
                      + (" + past" if cfg.get("scenario_past") == "True" else "")
                      + (f" PLACEBO shift {shift}" if shift else ""))
    if cfg.get("macro_conditioning", "film") != "film":
        extras.append(cfg["macro_conditioning"].replace("_", "-"))
    if cfg.get("scenario_past_lags", "[0]") != "[0]":
        extras.append(f"past lags {cfg['scenario_past_lags']}")
    if cfg.get("weight_moves") == "True":
        extras.append("weighted moves")
    if float(cfg.get("response_weight", 0) or 0):
        extras.append(f"response loss {cfg['response_weight']}")
    if cfg.get("macro_target", "both") != "both":
        extras.append("macro on Pillar 2 only")
    if cfg.get("pillar1", "last") != "last":
        extras.append("Pillar 1 = window mean")
    if cfg.get("pair_loss", "mse") != "mse":
        extras.append(f"neighbourhood loss τ {cfg.get('tau')}")
    if float(cfg.get("input_mask", 0) or 0):
        extras.append(f"input mask {cfg['input_mask']}")
    if float(cfg.get("macro_dropout", 0) or 0):
        extras.append(f"macro dropout {cfg['macro_dropout']}")
    if cfg.get("cutoff") not in (None, "None", "2023Q3"):
        extras.append(f"cutoff {cfg['cutoff']}")
    if cfg.get("seed") not in (None, "42"):
        extras.append(f"seed {cfg['seed']}")
    if cfg.get("reused_from") not in (None, "None"):
        extras.append(f"re-scored {cfg['reused_from']}")
    return ", ".join([loss, cfg.get("inputs", "diff_y"), f"MLP {hidden}" if hidden else "linear", *extras,
                      f"t_in {cfg.get('t_in')}", f"z {cfg.get('latent_dim')}", str(cfg.get("panel")), f"{cfg.get('epochs')} ep"])


def scorecard(runs_dir: Path = RUNS, include_old: bool = False) -> pd.DataFrame:
    """Runs scored on the fixed evaluation universe only, unless include_old (runs from before it existed)."""
    rows = []
    for d in sorted(p for p in runs_dir.iterdir() if (p / "results.csv").exists() and (p / "config.json").exists()):
        res, cfg = pd.read_csv(d / "results.csv"), json.loads((d / "config.json").read_text())
        if "universe" not in cfg and not include_old:
            continue
        pillar2 = cfg.get("pillar2", "future")
        if pillar2 == "future":                                        # fit2 is already the next-h-quarter change
            res["fit2_future"] = res["fit2"]
        res = res.reindex(columns=list(dict.fromkeys(["set", "variant"] + list(res.columns) + METRICS)))
        for s, r in res.groupby("set"):
            v = r.set_index("variant")
            base = {"run": d.name, "config": config_label(cfg), "inputs": cfg.get("inputs", "diff_y"), "set": s,
                    "pillar2": pillar2}
            scen = lambda label: ({"fit2_scen": v.loc[f"{label}, scenario real"].fit2,
                                   "d2_scen_zero": v.loc[f"{label}, scenario real"].fit2 - v.loc[f"{label}, scenario zero"].fit2,
                                   "d2_scen_shuf": v.loc[f"{label}, scenario real"].fit2 - v.loc[f"{label}, scenario shuffled"].fit2}
                                  if f"{label}, scenario real" in v.index else {})
            if "model, macro real" in v.index:
                real, zero, shuf = (v.loc[f"model, macro {k}"] for k in ("real", "zero", "shuffled"))
                rows.append({**base, "model": "macro", **real[METRICS].to_dict(),
                             "d_zero": real.fit - zero.fit, "d_shuf": real.fit - shuf.fit,
                             "d2_zero": real.fit2 - zero.fit2, "d2_shuf": real.fit2 - shuf.fit2, **scen("model")})
            if "model trained without macro" in v.index:
                rows.append({**base, "model": "no macro", **v.loc["model trained without macro"][METRICS].to_dict(),
                             **scen("no macro")})
            for ref in v.index:
                if ref.startswith(("raw fingerprint", "PCA of", "past change", "window deltas")):
                    rows.append({**base, "model": ref, **v.loc[ref][METRICS].to_dict(), "reference": True})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", default="out of time", help="Evaluation set to print.")
    p.add_argument("--all", action="store_true", help="Also the runs from before the fixed evaluation universe.")
    args = p.parse_args()
    sc = scorecard(include_old=args.all)
    if sc.empty:
        raise SystemExit(f"no runs in {RUNS}")
    sc.to_csv(RUNS / "scorecard.csv", index=False)
    s = sc[sc.set == args.set]
    is_ref = s.reference.eq(True) if "reference" in s else pd.Series(False, index=s.index)
    runs, refs = s[~is_ref], s[is_ref]
    last = refs.groupby("pillar2").run.transform("max")               # one reference run per Pillar 2 target
    latest = refs[(refs.run == last) & ~refs.model.str.startswith("PCA of input")]
    pca_in = refs[refs.model.str.startswith("PCA of input")].sort_values("run").drop_duplicates(["inputs", "pillar2"], keep="last")
    pd.set_option("display.width", 280)
    pd.set_option("display.max_colwidth", 70)
    print(f"Scorecard, evaluation set: {args.set}\n")
    print(runs.reindex(columns=["run", "config", "model"] + METRICS + DELTAS + SCEN).round(3)
          .to_string(index=False, na_rep="-"))
    print("\nReferences:")
    out = pd.concat([latest.assign(inputs="-"), pca_in])[["model", "inputs", "pillar2"] + METRICS]
    print(out.round(3).to_string(index=False, na_rep="-"))
    print(f"\nAll sets: {RUNS / 'scorecard.csv'}")


if __name__ == "__main__":
    main()
