import numpy as np
from scipy.stats import entropy

def detect_fixations(df, saccade_thresh):
    df['is_saccade'] = df['v_smooth'] > saccade_thresh
    segments, start = [], None
    for i, flag in enumerate(df['is_saccade']):
        if not flag and start is None:
            start = i
        elif flag and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(df) - 1))
    return segments

def fixation_stats(df, fixation_segments):
    durations = [df['n'].iloc[e] - df['n'].iloc[s] for s, e in fixation_segments if e > s]
    total_time = df['n'].iloc[-1] - df['n'].iloc[0]
    focused_time = sum(durations)
    fixation_rate = len(durations) / (total_time / 1000)
    return {
        "Fixation_count": len(durations),
        "Fixation_rate_Hz": fixation_rate,
        "Mean_fixation_duration_ms": np.mean(durations),
        "Median_fixation_duration_ms": np.median(durations),
        "Focused_ratio": focused_time / total_time,
        "Total_time_s": total_time / 1000,
    }

def saccade_stats(df):
    saccade_samples = df[df['is_saccade']]
    total_time = df['n'].iloc[-1] - df['n'].iloc[0]
    saccade_durations = []
    start = None
    for i, flag in enumerate(df['is_saccade']):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            saccade_durations.append(df['n'].iloc[i - 1] - df['n'].iloc[start])
            start = None
    if start is not None:
        saccade_durations.append(df['n'].iloc[-1] - df['n'].iloc[start])
    return {
        "Saccade_count": len(saccade_durations),
        "Saccade_rate_Hz": len(saccade_durations) / (total_time / 1000),
        "Mean_saccade_duration_ms": np.mean(saccade_durations),
        "Mean_velocity_during_saccades": saccade_samples['v_smooth'].mean(),
    }

def spatial_metrics(df, fixation_segments):
    centroids = []
    dispersions = []
    for s, e in fixation_segments:
        fx, fy = df['x'].iloc[s:e], df['y'].iloc[s:e]
        if len(fx) < 5:
            continue
        dx, dy = fx.max() - fx.min(), fy.max() - fy.min()
        dispersions.append(np.sqrt(dx**2 + dy**2))
        centroids.append((fx.mean(), fy.mean()))
    centroid_dispersion = np.sqrt(np.var([c[0] for c in centroids]) +
                                  np.var([c[1] for c in centroids])) if centroids else np.nan
    return {
        "Mean_fixation_dispersion": np.mean(dispersions),
        "Median_fixation_dispersion": np.median(dispersions),
        "Centroid_stability": centroid_dispersion,
    }

def global_entropy(df, bins=40):
    hist, _, _ = np.histogram2d(df['x'], df['y'], bins=bins)
    p = hist.flatten() / hist.sum()
    return entropy(p[p > 0])

def rolling_entropy(df, window_ms=2000, step_ms=500):
    window = int(window_ms / np.mean(np.diff(df['n'])))
    step = int(step_ms / np.mean(np.diff(df['n'])))
    times, ents = [], []
    for start in range(0, len(df) - window, step):
        end = start + window
        seg = df.iloc[start:end]
        hist, _, _ = np.histogram2d(seg['x'], seg['y'], bins=25)
        p = hist.flatten() / hist.sum()
        ents.append(entropy(p[p > 0]))
        times.append(df['n'].iloc[start + window // 2])
    return np.array(times), np.array(ents)

def attention_variability(entropies):
    return {"Attention_variability_index": np.std(entropies)}


def compute_attention_profile(df, saccade_thresh):
    """Compute full set of attention metrics and return dict with results and derived data."""
    fixations = detect_fixations(df, saccade_thresh)
    fix_stats = fixation_stats(df, fixations)
    sac_stats = saccade_stats(df)
    spatial = spatial_metrics(df, fixations)
    glob_ent = global_entropy(df)
    times, ents = rolling_entropy(df)
    variability = attention_variability(ents)
    summary = {
        **fix_stats,
        **sac_stats,
        **spatial,
        "Global_entropy": glob_ent,
        **variability
    }
    return {
        "summary": summary,
        "fixation_segments": fixations,
        "rolling_entropy_times": times,
        "rolling_entropy_values": ents
    }
