import numpy as np

def compute_sampling_rate(df, time_col='n'):
    dt = np.diff(df[time_col])
    return 1000 / np.mean(dt)

def compute_velocity(df):
    df['vx'] = np.gradient(df['x'], df['n']) * 1000
    df['vy'] = np.gradient(df['y'], df['n']) * 1000
    df['v'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['v_smooth'] = df['v'].rolling(window=25, center=True).median()
    return df
