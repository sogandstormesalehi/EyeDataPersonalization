import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_velocity_distribution(df, saccade_thresh):
    """Histogram of smoothed velocities and adaptive saccade threshold."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8,4))
    plt.hist(df['v_smooth'].dropna(), bins=100, color='lightsteelblue', alpha=0.8)
    plt.axvline(saccade_thresh, color='red', linestyle='--', linewidth=2,
                label=f"Threshold = {saccade_thresh:.1f} °/s")
    plt.title("Velocity Distribution and Adaptive Saccade Threshold", fontsize=13, weight='bold')
    plt.xlabel("Velocity (°/s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_attention_timeline(df, saccade_thresh):
    """Visualize gaze trace colored by attention state."""
    df['is_saccade'] = df['v_smooth'] > saccade_thresh
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df['n'], df['x'], color='lightgray', linewidth=1.0, alpha=0.6, label='Raw Gaze (x)')
    ax.scatter(df.loc[~df['is_saccade'], 'n'], df.loc[~df['is_saccade'], 'x'],
               c='royalblue', s=6, label='Fixation (Focused)', alpha=0.8)
    ax.scatter(df.loc[df['is_saccade'], 'n'], df.loc[df['is_saccade'], 'x'],
               c='crimson', s=6, label='Saccade (Attention Shift)', alpha=0.8)
    ax.set_title("Gaze-Based Attention Over Time", fontsize=14, weight='bold')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Horizontal Gaze (° dva)")
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    plt.show()

    fig, ax2 = plt.subplots(figsize=(14,4))
    ax2.plot(df['n'], df['v_smooth'], color='dimgray', linewidth=1.2, label='Smoothed Velocity (°/s)')
    ax2.axhline(saccade_thresh, color='red', linestyle='--', linewidth=1.0, label='Saccade Threshold')
    ax2.fill_between(df['n'], 0, df['v_smooth'], where=df['is_saccade'], color='salmon', alpha=0.3)
    ax2.set_title("Smoothed Gaze Velocity with Saccade Threshold", fontsize=13)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Velocity (°/s)")
    ax2.legend()
    plt.tight_layout()
    plt.show()


def plot_fixations(df, fixation_segments):
    """Highlight fixation periods over time."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14,3))
    for s, e in fixation_segments:
        ax.axvspan(df['n'].iloc[s], df['n'].iloc[e], color='royalblue', alpha=0.3)
    ax.plot(df['n'], df['x'], color='gray', linewidth=0.8)
    ax.set_title("Fixation Periods Over Time (Blue = Focused Attention)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Horizontal Gaze (° dva)")
    plt.tight_layout()
    plt.show()


def plot_dispersion_heatmap(dispersion_values, hist, xedges, yedges):
    """Fixation dispersion distribution and gaze heatmap."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(13,5))
    ax[0].hist(dispersion_values, bins=30, color='royalblue', alpha=0.8)
    ax[0].set_title("Fixation Dispersion Distribution")
    ax[0].set_xlabel("Dispersion (° dva)")
    ax[0].set_ylabel("Count")
    ax[1].imshow(hist.T, origin='lower', cmap='plasma',
                 extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
    ax[1].set_title("Gaze Density Heatmap (Attention Map)")
    ax[1].set_xlabel("Horizontal Gaze (° dva)")
    ax[1].set_ylabel("Vertical Gaze (° dva)")
    plt.tight_layout()
    plt.show()


def plot_rolling_entropy(times, entropies):
    """Plot rolling attention entropy timeline."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14,5))
    mean_ent = np.mean(entropies)
    ax.plot(times, entropies, color='darkorange', linewidth=2)
    ax.axhline(mean_ent, color='gray', linestyle='--', alpha=0.7)
    ax.fill_between(times, entropies, mean_ent,
                    where=entropies>mean_ent, color='salmon', alpha=0.3)
    ax.fill_between(times, entropies, mean_ent,
                    where=entropies<=mean_ent, color='royalblue', alpha=0.3)
    ax.set_title("Rolling Attention Entropy (2 s window)", fontsize=14, weight='bold')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Entropy (nats)")
    plt.tight_layout()
    plt.show()



def plot_group_attention_drift(all_df, n_bins=100):
    binned = _bin_attention(all_df, n_bins)
    mean_curve = binned.groupby("t_bin")["rolling_entropy"].mean()
    std_curve  = binned.groupby("t_bin")["rolling_entropy"].std()

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10,5))
    plt.plot(mean_curve.index.astype(float), mean_curve.values,
             color='royalblue', lw=2, label='Mean Entropy')
    plt.fill_between(mean_curve.index.astype(float),
                     mean_curve - std_curve, mean_curve + std_curve,
                     color='royalblue', alpha=0.25, label='±1 SD')
    plt.title("Group Attention Drift Across Participants", fontsize=14, weight='bold')
    plt.xlabel("Normalized Video Time (0–1)")
    plt.ylabel("Rolling Attention Entropy (nats)")
    plt.legend()
    plt.tight_layout()
    plt.show()




def plot_user_heatmap(all_df, n_bins=100):
    """Show heatmap of rolling entropy (attention intensity) for all participants."""
    binned = _bin_attention(all_df, n_bins)
    
    binned["t_bin"] = binned["t_bin"].astype(float).round(5)

    pivot = binned.pivot_table(index='participant_id', columns='t_bin',
                               values='rolling_entropy', aggfunc='mean', observed=True)
    
    sns.set_theme(style="white")
    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={'label': 'Rolling Entropy (nats)'})
    plt.title("Temporal Attention Intensity Heatmap", fontsize=14, weight='bold')
    plt.xlabel("Normalized Video Time (0–1)")
    plt.ylabel("Participant ID")
    plt.tight_layout()
    plt.show()




def _bin_attention(all_df, n_bins=100):
    """Discretize normalized time into bins for smoother group aggregation."""
    bins = np.linspace(0, 1, n_bins + 1)
    labels = (bins[:-1] + bins[1:]) / 2
    all_df = all_df.copy()
    all_df["t_bin"] = pd.cut(all_df["t_norm"], bins=bins, labels=labels, include_lowest=True)

    binned = (
        all_df
        .groupby(["participant_id", "session_id", "t_bin"], observed=True)["rolling_entropy"]
        .mean()
        .reset_index()
    )

    return binned



def plot_group_attention_drift_by_session(all_df, n_bins=100):
    """
    Show group mean ± SD for each session as separate subplots.
    """
    binned = _bin_attention(all_df, n_bins)
    sessions = sorted(binned["session_id"].unique())

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=len(sessions), ncols=1,
        figsize=(10, 4 * len(sessions)), sharex=True, sharey=True
    )

    if len(sessions) == 1:
        axes = [axes]

    for ax, sess in zip(axes, sessions):
        df_s = binned[binned["session_id"] == sess]
        mean_curve = df_s.groupby("t_bin", observed=True)["rolling_entropy"].mean()
        std_curve = df_s.groupby("t_bin", observed=True)["rolling_entropy"].std()

        ax.plot(mean_curve.index.astype(float), mean_curve.values,
                color='royalblue', lw=2, label=f'{sess} Mean')
        ax.fill_between(mean_curve.index.astype(float),
                        mean_curve - std_curve, mean_curve + std_curve,
                        color='royalblue', alpha=0.25)
        ax.set_title(f"Session {sess}: Group Attention Drift", fontsize=13, weight='bold')
        ax.set_ylabel("Rolling Entropy (nats)")
        ax.legend()

    axes[-1].set_xlabel("Normalized Video Time (0–1)")
    plt.suptitle("Group Attention Drift per Session", fontsize=15, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_user_longitudinal(all_df, user_id, n_bins=100):
    """
    Compare one participant's attention drift across sessions as subplots.
    """
    binned = _bin_attention(all_df, n_bins)
    user_df = binned[binned["participant_id"].astype(str).str.contains(str(user_id))]

    if user_df.empty:
        print(f"No data found for user {user_id}")
        return

    sessions = sorted(user_df["session_id"].unique())

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=len(sessions), ncols=1,
        figsize=(10, 4 * len(sessions)), sharex=True, sharey=True
    )

    if len(sessions) == 1:
        axes = [axes]

    for ax, sess in zip(axes, sessions):
        df_s = user_df[user_df["session_id"] == sess]
        ax.plot(df_s["t_bin"].astype(float), df_s["rolling_entropy"],
                color='royalblue', lw=2)
        ax.set_title(f"Session {sess}", fontsize=13, weight='bold')
        ax.set_ylabel("Entropy (nats)")

    axes[-1].set_xlabel("Normalized Video Time (0–1)")
    plt.suptitle(f"User {user_id}: Attention Drift per Session", fontsize=15, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_user_heatmap_by_session(all_df, n_bins=100):
    """Show heatmap of entropy for all participants, separated by session."""
    binned = _bin_attention(all_df, n_bins)
    binned["t_bin"] = binned["t_bin"].astype(float).round(5)

    for sess, df_s in binned.groupby("session_id"):
        pivot = df_s.pivot_table(index='participant_id', columns='t_bin',
                                 values='rolling_entropy', aggfunc='mean', observed=True)
        sns.set_theme(style="white")
        plt.figure(figsize=(12,6))
        sns.heatmap(pivot, cmap="viridis", cbar_kws={'label': 'Rolling Entropy (nats)'})
        plt.title(f"Temporal Attention Intensity Heatmap — {sess}", fontsize=14, weight='bold')
        plt.xlabel("Normalized Video Time (0–1)")
        plt.ylabel("Participant ID")
        plt.tight_layout()
        plt.show()

def plot_user_vs_group(all_df, user_id, n_bins=100):
    """
    Compare a user's attention drift to the group mean for each session
    using separate subplots.
    """
    binned = _bin_attention(all_df, n_bins)
    binned["participant_id"] = binned["participant_id"].astype(str)
    user_id_str = str(user_id)

    match_ids = [uid for uid in binned["participant_id"].unique()
                 if user_id_str in uid or uid.replace("S_", "") == user_id_str]

    if not match_ids:
        print(f"No matching data found for user {user_id}. "
              f"Available IDs: {binned['participant_id'].unique().tolist()}")
        return

    user_df = binned[binned["participant_id"].isin(match_ids)]
    sessions = sorted(binned["session_id"].unique())

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=len(sessions), ncols=1,
        figsize=(10, 4 * len(sessions)), sharex=True, sharey=True
    )

    if len(sessions) == 1:
        axes = [axes]

    for ax, sess in zip(axes, sessions):
        df_group = binned[binned["session_id"] == sess]
        mean_curve = df_group.groupby("t_bin", observed=True)["rolling_entropy"].mean()
        std_curve  = df_group.groupby("t_bin", observed=True)["rolling_entropy"].std()

        user_sess_df = user_df[user_df["session_id"] == sess]

        ax.plot(mean_curve.index.astype(float), mean_curve.values,
                color='slategrey', lw=2, label='Group Mean')
        ax.fill_between(mean_curve.index.astype(float),
                        mean_curve - std_curve, mean_curve + std_curve,
                        color='lightsteelblue', alpha=0.15, label='±1 SD')

        if not user_sess_df.empty:
            ax.plot(user_sess_df["t_bin"].astype(float),
                    user_sess_df["rolling_entropy"],
                    color='mediumturquoise', lw=2.5, label=f'User {user_id}')

        ax.set_title(f"Session {sess}", fontsize=13, weight='bold')
        ax.set_ylabel("Rolling Entropy (nats)")
        ax.legend()

    axes[-1].set_xlabel("Normalized Video Time (0–1)")
    plt.suptitle(f"User {user_id} vs Group — Per-Session Attention Drift", fontsize=15, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_user_uniqueness(reg_df):
    """
    Visualize how each participant's gaze-entropy dynamics differ from the group.

    Parameters
    ----------
    reg_df : pd.DataFrame
        Output of regression_user_vs_group() containing columns:
        ['participant_id', 'session_id', 'r2', 'slope', 'intercept']
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 4))
    sns.barplot(data=reg_df, x="participant_id", y="r2", hue="session_id", palette="coolwarm")
    plt.xticks(rotation=90)
    plt.title("User Uniqueness (R² vs Group Mean) per Session", fontsize=13, weight="bold")
    plt.ylabel("R² (Alignment with Group)")
    plt.xlabel("Participant ID")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.barplot(data=reg_df, x="participant_id", y="slope", hue="session_id", palette="viridis")
    plt.xticks(rotation=90)
    plt.title("Focus Tendency (Slope vs Group Mean) per Session", fontsize=13, weight="bold")
    plt.ylabel("Slope (User / Group)")
    plt.xlabel("Participant ID")
    plt.tight_layout()
    plt.show()
