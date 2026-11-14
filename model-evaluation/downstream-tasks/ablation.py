"""
Ablation on the downstream tasks T1 (the 10 nearest firms) and T2 (the k-means boxes): the paper model against itself
without the input mask and with decay 1 (all moves weigh the same), 10 seeds each, on the unseen firms before the
cutoff (2015Q4-2022Q4) and after it (2023Q1-2025Q4) -> ablation.xlsx (sheet "T1 and T2": the means over the seeds, the
best of each row in bold; "per seed": every value, columns 1-10 = seeds 42, 1-9; "legend": the terms and their papers).
Reads each run's usefulness.csv (scored by src/usefulness_eval.py, where T2 is still keyed "T3 groups").

    xbrl2vec/bin/python model-evaluation/downstream-tasks/ablation.py
"""
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import stability_eval as se  # noqa: E402

MODEL, ABLATIONS = "balanced", {"no mask": "balanced, unmasked", "decay 1": "balanced, decay 1"}
SETS, TASKS, QUANTS = ["unseen firms", "unseen firms, out of time"], ["T1 peers", "T3 groups"], ["size", "revenue growth", "asset growth"]
ORDER = pd.MultiIndex.from_product([SETS, TASKS, QUANTS], names=["set", "task", "quantity"])
BLO = ("Bhojraj, Lee & Oler (2003), \"What's My Line? A Comparison of Industry Classification Schemes for Capital Market "
       "Research\", Journal of Accounting Research 41(5)")
HP = ("Hoberg & Phillips (2016), \"Text-Based Network Industries and Endogenous Product Differentiation\", Journal of "
      "Political Economy 124(5)")
LEGEND = [
    ("T1 peers", "For each unseen firm in a quarter, the mean of the quantity over its 10 nearest firms (highest cosine "
     "similarity of the embeddings among all 2,115 firms of the quarter, never the firm itself; at least 3 with data). "
     "Score: R² (squared correlation) between the firms' own values and their peers' means across the unseen firms of "
     "the quarter, then the mean over the quarters.",
     f"{BLO}; peer mean without the firm itself as in Kaustia & Rantala (2021), \"Common Analysts: Method for Defining "
     "Peer Firms\", Journal of Financial and Quantitative Analysis 56(5)"),
    ("T2 groups", "Each quarter, k-means on the embeddings splits the firms into K boxes, K = the number of SIC2 codes in "
     "the quarter (67): as many boxes as industries. Each firm is compared with the mean of the quantity over the other "
     "members of its box (at least 3 with data); score as in T1.",
     f"{HP}: data-driven industries compared with SIC and NAICS at the same number of industries"),
    ("size", "log total assets", ""),
    ("revenue growth", "log(revenue_t / revenue_t−4), year on year", ""),
    ("asset growth", "log(total assets_t / total assets_t−4), year on year", ""),
    ("quantities", "never a model input (the model sees ratios only); observed values only, winsorised at 1 % and 99 % "
     "per quarter", ""),
    ("unseen firms", "the 317 firms (15 %) never used in training, quarters 2015Q4–2022Q4 (up to the cutoff)", ""),
    ("unseen firms, out of time", "the same firms after the cutoff, 2023Q1–2025Q4", ""),
    ("R²", "0: the peers say nothing about the firm; 1: they match it exactly", ""),
    ("bold", "the highest mean of the row (not a significance test)", ""),
    ("per seed", "every value, columns 1–10 = the 10 seeds", ""),
]


def cells(family: str) -> pd.DataFrame:
    """seed x (set, task, quantity): the T1 and T2 cells of the family's 10 runs, seed 42 first."""
    A = se.cells_of(se.runs_of(family, scored=True))
    assert len(A) == 10, (family, len(A))
    return A.reindex(columns=ORDER)


def write(values: dict[str, pd.DataFrame], means: pd.DataFrame, name: str, rows: list[tuple]) -> None:
    """Sheet "T1 and T2": the means, the best of each row in bold; "per seed": every value of `values`; "legend":
    LEGEND and `rows`. The scorer's "T3 groups" is written as "T2 groups"."""
    t2 = {"T3 groups": "T2 groups"}
    means = means.round(4).rename(index=t2)
    with pd.ExcelWriter(HERE / name) as xl:
        means.reset_index().to_excel(xl, sheet_name="T1 and T2", index=False)
        for row, best in enumerate(means.idxmax(axis=1), start=2):           # header in row 1, values after the cells
            xl.sheets["T1 and T2"].cell(row, ORDER.nlevels + 1 + means.columns.get_loc(best)).font = Font(bold=True)
        pd.concat({k: v.T.set_axis(range(1, 11), axis=1) for k, v in values.items()}, names=["version"]).round(4) \
            .rename(index=t2).reset_index().to_excel(xl, sheet_name="per seed", index=False)
        pd.DataFrame(LEGEND + rows, columns=["term", "meaning", "source"]).to_excel(xl, sheet_name="legend", index=False)
    print(f"[INFO] {HERE / name}")


def main() -> None:
    C = {"model": cells(MODEL), **{name: cells(fam) for name, fam in ABLATIONS.items()}}
    write(C, pd.DataFrame({k: v.mean() for k, v in C.items()}), "ablation.xlsx", [
        ("model", "the paper model: z 32, input mask 0.2, λ 0.5, signed-log fingerprint, decay 0.8; 10 training seeds "
         "(42, 1–9)", ""),
        ("no mask", "the model trained without the input mask", ""),
        ("decay 1", "the model with every move of Pillar 2 weighted the same (no recency weights)", "")])


if __name__ == "__main__":
    main()
