import re
import glob
import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    df = df.drop(columns=['xT', 'yT'], errors='ignore')
    return df


def save_results(df: pd.DataFrame, out_path: str = "attention_summary.csv"):
    df.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")


def load_processed_files(path_pattern="processed/*_processed.parquet", k=None):
    """
    Load and combine up to k processed per-user parquet files.
    Extract participant_id and session_id from filenames.
    """
    files = glob.glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"No files match pattern: {path_pattern}")

    if k is not None:
        files = files[:k]
        print(f"Loading first {k} processed files for testing.")
    else:
        print(f"Loading all {len(files)} processed files.")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        
        match = re.search(r"S_(\d+)_S(\d)_", f)
        if match:
            participant_id = f"S_{match.group(1)}"
            session_id = f"S{match.group(2)}"
        else:
            participant_id = "Unknown"
            session_id = "Unknown"

        df["participant_id"] = participant_id
        df["session_id"] = session_id
        
        if "n" in df.columns:
            df["t_norm"] = (df["n"] - df["n"].min()) / (df["n"].max() - df["n"].min())
        
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(dfs)} files ({len(all_df):,} samples total)")
    print("Columns:", list(all_df.columns))
    return all_df
