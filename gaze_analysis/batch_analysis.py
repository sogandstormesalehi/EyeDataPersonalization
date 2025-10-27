import os
import pandas as pd
import numpy as np
from gaze_analysis.io_utils import load_data
from gaze_analysis.preprocess import compute_velocity
from gaze_analysis.velocity_analysis import adaptive_saccade_threshold
from gaze_analysis.attention_metrics import compute_attention_profile, rolling_entropy

def process_file(path):
    df = load_data(path)
    df = compute_velocity(df)
    saccade_thresh = adaptive_saccade_threshold(df)
    profile = compute_attention_profile(df, saccade_thresh)

    # Map rolling entropy back to timestamps
    times, ents = profile["rolling_entropy_times"], profile["rolling_entropy_values"]
    df['rolling_entropy'] = np.interp(df['n'], times, ents)

    # Add participant/session info
    fname = os.path.basename(path)
    df["file"] = fname
    pid = fname.split("_")[1] 
    df["participant_id"] = pid

    summary = profile["summary"]
    summary["file"] = fname
    summary["participant_id"] = pid

    return df, summary


def analyze_directory(data_dir, pattern="_VID.csv", save_processed=True, out_dir="processed"):
    os.makedirs(out_dir, exist_ok=True)
    all_summaries = []

    for fname in os.listdir(data_dir):
        if not fname.endswith(pattern):
            continue
        path = os.path.join(data_dir, fname)
        try:
            df_proc, summary = process_file(path)
            all_summaries.append(summary)

            if save_processed:
                out_path = os.path.join(out_dir, fname.replace(".csv", "_processed.parquet"))
                df_proc.to_parquet(out_path, index=False, engine="fastparquet")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(out_dir, "attention_summary.csv"), index=False)
    print(f"\nAll summaries saved to {os.path.join(out_dir, 'attention_summary.csv')}")
    return summary_df
