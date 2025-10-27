from scipy.stats import pearsonr
import statsmodels.api as sm
import pandas as pd

def compute_intra_subject_correlation(binned):
    results = []
    for pid, df_p in binned.groupby("participant_id"):
        sessions = df_p["session_id"].unique()
        if len(sessions) < 2:
            continue  
        pivot = df_p.pivot(index="t_bin", columns="session_id", values="rolling_entropy")
        if pivot.shape[1] >= 2:
            s1, s2 = pivot.columns[:2]
            r, _ = pearsonr(pivot[s1].dropna(), pivot[s2].dropna())
            results.append({"participant_id": pid, "intra_corr": r})
    return pd.DataFrame(results)


def regression_user_vs_group(binned):
    results = []
    for pid, df_p in binned.groupby("participant_id"):
        for sess, df_s in df_p.groupby("session_id"):
            # Compute group mean (excluding this participant)
            group_mean = (binned[(binned["session_id"] == sess) & 
                                 (binned["participant_id"] != pid)]
                          .groupby("t_bin")["rolling_entropy"].mean())
            
            merged = pd.merge(df_s, group_mean, on="t_bin", suffixes=("", "_group"))
            if len(merged) < 5:
                continue

            X = sm.add_constant(merged["rolling_entropy_group"])
            y = merged["rolling_entropy"]
            model = sm.OLS(y, X).fit()

            results.append({
                "participant_id": pid,
                "session_id": sess,
                "r2": model.rsquared,
                "slope": model.params["rolling_entropy_group"],
                "intercept": model.params["const"]
            })
    return pd.DataFrame(results)