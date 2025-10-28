import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from glob import glob


DATA_PATH = "processed/"
SEQ_LEN = 200        # roughly first ~8s at 25 Hz
PRED_LEN = 250       # predict ~10s horizon
DOWNSAMPLE = 10      # from 250 Hz to 25 Hz


class SlidingGazeDataset(Dataset):
    """
    Creates overlapping temporal windows of gaze data for short-term forecasting.

    Each sample = (past window -> next window) for rolling entropy prediction.
    """

    def __init__(self, path_pattern, window_sec=5, pred_sec=5,
                 sample_rate=25, downsample=10, stride=50):
        """
        path_pattern : str, folder or glob for parquet files
        window_sec   : seconds of input context
        pred_sec     : seconds of future to predict
        sample_rate  : effective rate after downsampling
        stride       : number of samples to shift between windows
        """
        self.files = sorted(glob(path_pattern + "/*.parquet"))
        self.window_len = int(window_sec * sample_rate)
        self.pred_len   = int(pred_sec * sample_rate)
        self.downsample = downsample
        self.stride     = stride
        self.samples = self._make_windows()

    def _make_windows(self):
        samples = []
        for f in self.files:
            df = pd.read_parquet(f)[['x', 'y', 'vx', 'vy', 'rolling_entropy']].dropna()
            df = df.iloc[::self.downsample]
            arr = df.values.astype(np.float32)
            for i in range(0, len(arr) - self.window_len - self.pred_len, self.stride):
                x_in  = arr[i : i + self.window_len]
                y_out = arr[i + self.window_len : i + self.window_len + self.pred_len, -1]
                samples.append((x_in, y_out))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)
