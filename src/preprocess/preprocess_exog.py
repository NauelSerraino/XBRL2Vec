"""
FRED series (one CSV per series, columns observation_date, <ID>) -> exog.parquet: one row per quarter (the value on
the first day of the quarter, where all series have one), levels and quarterly / yearly differences, standardised. A
value missing inside a series is interpolated from its neighbours: October 2025, never published during the US
government shutdown, would otherwise drop 2025Q4 for every firm (windows are joined to the macro by quarter).

    python src/preprocess/preprocess_exog.py --in_dir data/in/download_2026-10-25/fred --out_dir data/in/panels_2026-10-25
"""
import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(root, "..", ".."))
in_dir = os.path.join(project_root, "data", "in", "exogenous")
out_dir = os.path.join(project_root, "data", "in")


def read_all_dfs_in_folder(folder):
    dfs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, f))
            if "observation_date" not in df.columns:
                raise ValueError(f"{f} missing observation_date")
            v = df.columns.drop("observation_date")
            df[v] = df[v].interpolate(limit_area="inside")
            dfs.append(df)

    if not dfs:
        raise ValueError("No CSVs found")

    out = dfs[0]
    for df in dfs[1:]:
        out = out.merge(df, on="observation_date", how="outer")

    return out.sort_values("observation_date")


def create_diffs(df):
    df = df.copy()
    df.sort_values("observation_date", inplace=True)

    for col in df.columns:
        if col not in ["observation_date"]:
            df[f'{col}_DIFF_Q'] = df[col].diff()
            df[f'{col}_DIFF_Y'] = df[col].diff(4)

    return df


def normalize_global(df):
    cols = [c for c in df.columns if c not in ["observation_date"]]
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols]).round(4)
    return df


def preprocess(df):
    df = create_diffs(df)
    df = normalize_global(df)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", default=in_dir, help="Folder of FRED CSVs (default: data/in/exogenous).")
    p.add_argument("--out_dir", default=out_dir, help="Where exog.parquet is written (default: data/in).")
    args = p.parse_args()
    df = read_all_dfs_in_folder(args.in_dir)

    df.dropna(how="any", axis=0, inplace=True)

    mapping = {
        "-01-01": "Q1",
        "-04-01": "Q2",
        "-07-01": "Q3",
        "-10-01": "Q4",
    }
    df["observation_date"] = df["observation_date"].str[:4] + df["observation_date"].str[-6:].map(mapping)

    exog_df = preprocess(df)

    exog_df.to_parquet(os.path.join(args.out_dir, "exog.parquet"))
    print(f"exog.parquet: {len(exog_df)} quarters, {exog_df.observation_date.iloc[0]}..{exog_df.observation_date.iloc[-1]} -> {args.out_dir}")
