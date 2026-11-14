"""
One line per run for the overnight macro search (docs/archive/report_2026-11-01_macro_search.md): the scorecard numbers out
of time (fit, fit2 of z, fit2 of the scenario-reweighted z̃, and the gains over the model trained without macro) and
the G1 placebo test (macro_placebo_eval.py: the macro of another period at inference, 14 shifts).

    cd src && ../xbrl2vec/bin/python score_g1.py <run> [<run> ...]      # runs macro_placebo_eval first if needed

G1 passes when the real macro beats at least 13 of 14 placebos on the Pillar 2 metric (fit2 of z̃ for scenario runs,
fit2 of z otherwise) in both out-of-time sets, and the macro model beats the no-macro model on it by more than 0.007.
For --scenario_past_only runs a second line scores the head-only arm: the no-macro encoder whose head gets the past
macro, against the same head with the macro set to 0.
Output: data/out/similarity/g1_table.csv (all scored runs), printed.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from compare_runs import RUNS, config_label
from macro_placebo_eval import check

SETS = ("out of time", "unseen firms, out of time")


def row(run: str, arm: str = "encoder") -> dict:
    cfg = json.loads((RUNS / run / "config.json").read_text())
    res = pd.read_csv(RUNS / run / "results.csv").set_index(["set", "variant"])
    scen = cfg.get("scenario") == "True"
    plc_file = RUNS / run / "explain" / "08_macro_placebo.csv"
    plc = pd.read_csv(plc_file) if plc_file.exists() else check(run)
    if arm == "head" and "fit2_scen_nomacro" not in plc:
        plc = check(run)
    key = "fit2_scen_nomacro" if arm == "head" else ("fit2_scen" if scen else "fit2")
    out = {"run": run, "config": config_label(cfg) + (" [macro only via head]" if arm == "head" else "")}
    passes = []
    for s in SETS:
        tag = "OOT" if s == "out of time" else "unseen OOT"
        if arm == "head":
            mac, nom = res.loc[(s, "no macro, scenario real")], res.loc[(s, "no macro, scenario zero")]
        else:
            mac = res.loc[(s, "model, scenario real" if scen else "model, macro real")]
            nom = res.loc[(s, "no macro, scenario real" if scen else "model trained without macro")]
        g = plc[plc.set == s]
        real, others = g[g["shift"] == 0].iloc[0], g[g["shift"] != 0]
        beaten = int((others[key] < real[key]).sum())
        gain = mac.fit2 - nom.fit2
        out |= {f"{tag} fit": res.loc[(s, "model, macro real")].fit, f"{tag} P2 metric": mac.fit2,
                f"{tag} gain vs no macro": gain, f"{tag} placebos beaten": f"{beaten}/{len(others)}"}
        passes.append(beaten >= len(others) - 1 and gain > 0.007)
    out["G1"] = "PASS" if all(passes) else "fail"
    return out


def main() -> None:
    rows = []
    for r in sys.argv[1:]:
        rows.append(row(r))
        if json.loads((RUNS / r / "config.json").read_text()).get("scenario_past_only") == "True":
            rows.append(row(r, arm="head"))
    path = RUNS / "g1_table.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    table = pd.concat([old[~old.set_index(["run", "config"]).index.isin(new.set_index(["run", "config"]).index)]
                       if len(old) else old, new])
    table.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
