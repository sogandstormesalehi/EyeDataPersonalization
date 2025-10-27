import numpy as np
import matplotlib.pyplot as plt

def adaptive_saccade_threshold(df):
    base_vel = df['v_smooth'].dropna()
    median_v = np.median(base_vel)
    mad_v = np.median(np.abs(base_vel - median_v))
    thresh = median_v + 6 * mad_v
    return thresh

def plot_velocity_distribution(df, saccade_thresh):
    plt.figure(figsize=(8,4))
    plt.hist(df['v_smooth'].dropna(), bins=100, color='lightsteelblue', alpha=0.8)
    plt.axvline(saccade_thresh, color='red', linestyle='--', linewidth=2)
    plt.title("Velocity Distribution and Adaptive Saccade Threshold")
    plt.xlabel("Velocity (°/s)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
