from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch

class ERKWindowDataset(Dataset):
    """
    Per-cell sliding window dataset over ERK trajectory data.

    Each sample is a fixed-length window of shape (window_size, n_features)
    drawn from a single cell's (uid) time series, sorted by frame.

    Args:
        df:             Source dataframe with at least `uid`, `frame`, and all
                        columns in feature_columns.
        feature_columns: Ordered list of column names to use as features.
        window_size:    Number of consecutive frames per sample.
        stride:         Step size between window starts (default 1).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        window_size: int,
        stride: int = 1,
    ):
        self.feature_columns = feature_columns
        self.window_size = window_size

        # Build index: list of (array_of_features, start_idx) per valid window
        self._windows: list[tuple[np.ndarray, int]] = []

        for uid, group in df.groupby("uid"):
            # Sort chronologically within each cell
            traj = (
                group.sort_values("frame")[feature_columns]
                .to_numpy(dtype=np.float32)
            )

            # Slide windows; cells shorter than window_size are silently skipped
            for start in range(0, len(traj) - window_size + 1, stride):
                self._windows.append((traj, start))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        traj, start = self._windows[idx]
        window = traj[start : start + self.window_size]  # (window_size, n_features)
        return torch.from_numpy(window)
