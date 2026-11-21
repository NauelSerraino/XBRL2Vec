"""
Robustness of the paper model's peers (a firm's 10 nearest firms by cosine, among all firms of the quarter), on the
unseen firms before the cutoff (2015Q4-2022Q4) and after it (2023Q1-2025Q4), means over the quarters -> out/robustness.xlsx:
  masking      share of the 10 peers kept when a random share of the last quarter's 32 ratios is missing (0, as in
               the data), for each of the 10 seeds; at 20 % this is the protocol's T5
  seeds        share of the 10 peers two seeds have in common: mean, min and max over the 45 pairs of seeds
  seed pairs   the same for every pair
  legend       the terms and the papers the measures come from

    xbrl2vec/bin/python model-evaluation/robustness/robustness.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
sys.path.insert(0, str(HERE.parents[1] / "src"))
import usefulness_eval as ue  # noqa: E402

SETS, SHARES = ["unseen firms", "unseen firms, out of time"], [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
WENDLANDT = ("Wendlandt, Kummerfeld & Mihalcea (2018), \"Factors Influencing the Surprising Instability of Word "
             "Embeddings\", NAACL-HLT, 2092-2102")
LEGEND = pd.DataFrame([
    ("peers", "a firm's 10 nearest firms: highest cosine similarity of the embeddings among all 2,115 firms of the quarter", ""),
    ("masking", "share of a firm's 10 peers that stay the same when a random share (\"share missing\") of its last "
     "quarter's 32 ratios is set to missing (0, as missing data enter the model); mean over the unseen firms and the "
     "quarters; one column per seed (42, 1-9) and the mean, min and max over the 10 seeds",
     f"top-10 neighbour overlap as in {WENDLANDT}; masking noise as in Vincent, Larochelle, Bengio & Manzagol (2008), "
     "\"Extracting and Composing Robust Features with Denoising Autoencoders\", ICML, 1096-1103"),
    ("seeds", "share of a firm's 10 peers that two training seeds of the model have in common; mean, min and max over "
     "the 45 pairs of the 10 seeds", f"{WENDLANDT}; see also Antoniak & Mimno (2018), \"Evaluating the Stability of "
     "Embedding-based Word Similarities\", Transactions of the ACL 6, 107-119"),
    ("seed pairs", "the same for every pair of seeds", ""),
    ("scale", "1: identical peers; 0: none in common; random peers would share about 10 / 2,114 ≈ 0.005", ""),
    ("unseen firms", "the 317 firms (15 %) never used in training, quarters 2015Q4-2022Q4 (up to the cutoff)", ""),
    ("unseen firms, out of time", "the same firms after the cutoff, 2023Q1-2025Q4", ""),
    ("model", "the paper model: z 32, input mask 0.2, λ 0.5, signed-log fingerprint, decay 0.8; 10 training seeds", ""),
], columns=["term", "meaning", "source"])


def main() -> None:
    """Peers under masking and across seeds, into out/robustness.xlsx (checked against the scored T5)."""
    runs = ue.runs_of("balanced")
    assert len(runs) == 10, len(runs)
    g = ue.Group(runs[0])
    enc = [g.encoder(r) for r in runs]
    U = [ue._unit(e(g.X)) for e in enc]
    seeds, K = [ue.seed_of(r) for r in runs], g.FP.shape[1]
    mask, pairs = [], []
    for s in SETS:
        for q in g.quarters[s]:
            pool = ue.pool_of(g, q)
            own = np.flatnonzero(g.focal[s][pool])
            nn = [ue.knn(u[pool])[own] for u in U]
            pairs += [{"set": s, "seed a": a, "seed b": b, "kept": ue.overlap(na, nb).mean()}
                      for (a, na), (b, nb) in combinations(zip(seeds, nn), 2)]
            for sh in SHARES:
                holes = ue.random_holes(len(own), K, round(sh * K), seed=1)
                mask += [{"set": s, "share missing": sh, "seed": sd,
                          "kept": ue.overlap(ue.knn_holed(ue._unit(e(g.holed_X(pool[own], holes))), u[pool], own), n).mean()}
                         for sd, e, u, n in zip(seeds, enc, U, nn)]
        print(f"[INFO] {s} done", flush=True)
    m = pd.DataFrame(mask).groupby(["set", "share missing", "seed"]).kept.mean().unstack("seed")[seeds]
    m = pd.concat([m.agg(["mean", "min", "max"], axis=1), m], axis=1)
    p = pd.DataFrame(pairs).groupby(["set", "seed a", "seed b"]).kept.mean()
    t5 = ue.cells_of(runs).rename(index=ue.seed_of)
    for s in SETS:                                   # self-check: 20 % missing gives the scored T5 (other random holes)
        ref = t5[(s, "T5 holes", "10-NN overlap, 20 % of last quarter missing")][seeds]
        assert np.allclose(m.loc[(s, 0.2), seeds], ref, atol=0.03), (s, m.loc[(s, 0.2), seeds].tolist(), ref.tolist())
    OUT.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT / "robustness.xlsx") as xl:
        m.round(4).reset_index().to_excel(xl, sheet_name="masking", index=False)
        p.groupby("set").agg(["mean", "min", "max"]).round(4).reset_index().to_excel(xl, sheet_name="seeds", index=False)
        p.round(4).reset_index().to_excel(xl, sheet_name="seed pairs", index=False)
        LEGEND.to_excel(xl, sheet_name="legend", index=False)
    print(f"[INFO] {OUT / 'robustness.xlsx'}")


if __name__ == "__main__":
    main()
