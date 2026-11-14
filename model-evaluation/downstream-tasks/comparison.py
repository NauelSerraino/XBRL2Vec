"""
Comparison on the downstream tasks T1 and T2: the paper model against three standard alternatives, on the unseen firms
before the cutoff and after it -> comparison.xlsx (sheet "T1 and T2": the means, the best of each row in bold;
"per seed": every value; "legend": the terms and their papers).
  SIC2              industry codes: the peers and the box are the other firms of the same SIC2 (the literature's benchmark)
  raw fingerprint   the 32 ratios of the last quarter, no model
  PCA-32            PCA of the fingerprint and window deltas: the model's linear twin (same two targets, same width 32)
Ten values per method: the model's 10 training seeds (their usefulness.csv); for the raw fingerprint and PCA-32, which
are not trained, 10 seeds of T2's k-means (their T1 has no randomness); SIC2 has no randomness at all.

    xbrl2vec/bin/python model-evaluation/downstream-tasks/comparison.py
"""
import numpy as np
import pandas as pd

from ablation import BLO, HP, MODEL, ORDER, cells, se, write

ue = se.ue
ALTERNATIVES = {"raw fingerprint": ue.RAW_FP, "PCA-32": se.PCA32}


def main() -> None:
    runs = se.runs_of(MODEL, scored=True)
    g, cached = ue.Group(runs[0]), se.baselines_of(runs[0]).reindex(ORDER)
    pick = lambda rows: pd.DataFrame(rows).query("q == 'all'").set_index(ue.CELL).value.reindex(ORDER)
    C = {"model": cells(MODEL)}
    for name, vec in ALTERNATIVES.items():
        V = g.baselines()[vec][0]()
        C[name] = pd.DataFrame([pick(ue.score(g, V, None, vec, seed=s)) for s in range(10)])
        # k-means seed 0 = the scored baselines, up to float noise (PCA varies ~1e-5 with the BLAS threads)
        assert np.allclose(C[name].iloc[0], cached[vec], atol=1e-3), name
    write(C, pd.DataFrame({"model": C["model"].mean(), "SIC2": pick(ue.sic_rows(g)),
                           **{name: C[name].mean() for name in ALTERNATIVES}}), "comparison.xlsx", [
        ("model", "as in ablation.xlsx: 10 training seeds", ""),
        ("SIC2", "2-digit SIC industry codes: the peers and the box are the other firms of the same SIC2; no randomness",
         f"the benchmark of {BLO} and {HP}"),
        ("raw fingerprint", "the 32 ratios of the last quarter, no model; 10 k-means seeds for T2 (T1 has no randomness)", ""),
        ("PCA-32", "PCA (32 components) of the fingerprint and the window deltas, the model's two training targets; no "
         "training; 10 k-means seeds for T2", "")])


if __name__ == "__main__":
    main()
