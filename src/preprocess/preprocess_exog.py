import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(root, "..", ".."))
in_dir = os.path.join(project_root, "data", "in", "exogenous")
out_dir = os.path.join(project_root, "data", "in")


def read_all_dfs_in_folder(folder):
    dfs = []
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, f))
            if "observation_date" not in df.columns:
                raise ValueError(f"{f} missing observation_date")
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
    df = read_all_dfs_in_folder(in_dir)

    df.dropna(how="any", axis=0, inplace=True)

    mapping = {
        "-01-01": "Q1",
        "-04-01": "Q2",
        "-07-01": "Q3",
        "-10-01": "Q4",
    }
    df["observation_date"] = df["observation_date"].str[:4] + df["observation_date"].str[-6:].map(mapping)

    exog_df = preprocess(df)

    exog_df.to_parquet(os.path.join(out_dir, "exog.parquet"))
