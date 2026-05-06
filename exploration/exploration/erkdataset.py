from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import torch


def _qcut_labels(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-bin `values` into up to `n_bins` integer labels.

    Robust to ties (duplicate quantile edges are dropped). Returns (labels, edges).
    """
    labels, edges = pd.qcut(
        values, n_bins, labels=False, duplicates="drop", retbins=True
    )
    labels = np.asarray(labels, dtype=np.int64)
    return labels, np.asarray(edges)


class ERKWindowDataset(Dataset):
    """
    Per-cell sliding window dataset over ERK trajectory data, with optional
    response/stimulus stratification for balanced batch sampling.

    Each sample is a fixed-length window of shape (window_size, n_features)
    drawn from a single cell's (uid) time series, sorted by frame.

    Stratification (when enabled) labels each window by a 2D bucket:
      - response score: peak-to-peak of `response_col` in the window
        (proxy for whether ERK responded)
      - stimulus score: mean of `stim_col` in the window
        (proxy for input dose)
    Each axis is split into quantile bins; the joint label is exposed via
    `self.strata` and a balanced sampler can be obtained with
    `make_balanced_sampler()`.

    Args:
        df:                Source dataframe with `uid`, `frame`, all feature cols,
                           plus `response_col` and `stim_col` if stratifying.
        feature_columns:   Ordered list of column names to use as features.
        window_size:       Number of consecutive frames per sample.
        stride:            Step size between window starts (default 1).
        stratify:          If True, compute per-window strata.
        response_col:      Column scoring window response amplitude.
        stim_col:          Column scoring window stimulus dose.
        n_response_bins:   Quantile bins on response score.
        n_stim_bins:       Quantile bins on stimulus score.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        window_size: int,
        stride: int = 1,
        stratify: bool = True,
        response_col: str = "cnr_mean_norm",
        stim_col: str = "fluence_mJ_cm2",
        n_response_bins: int = 3,
        n_stim_bins: int = 3,
    ):
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.stratify_enabled = stratify
        self.response_col = response_col
        self.stim_col = stim_col

        self._windows: list[tuple[np.ndarray, int]] = []
        self._uids: list = []
        response_scores: list[float] = []
        stim_scores: list[float] = []

        for uid, group in df.groupby("uid"):
            g = group.sort_values("frame")
            traj = g[feature_columns].to_numpy(dtype=np.float32)
            if stratify:
                resp_ts = g[response_col].to_numpy(dtype=np.float64)
                stim_ts = g[stim_col].to_numpy(dtype=np.float64)

            for start in range(0, len(traj) - window_size + 1, stride):
                self._windows.append((traj, start))
                self._uids.append(uid)
                if stratify:
                    w_r = resp_ts[start : start + window_size]
                    w_s = stim_ts[start : start + window_size]
                    response_scores.append(float(np.ptp(w_r)))
                    stim_scores.append(float(np.mean(w_s)))

        if stratify and len(self._windows) > 0:
            self.response_scores = np.asarray(response_scores)
            self.stim_scores = np.asarray(stim_scores)

            r_labels, self.response_edges = _qcut_labels(
                self.response_scores, n_response_bins
            )
            s_labels, self.stim_edges = _qcut_labels(
                self.stim_scores, n_stim_bins
            )
            self.response_bins = r_labels
            self.stim_bins = s_labels
            n_s = int(s_labels.max()) + 1
            self.strata = r_labels * n_s + s_labels
            self.n_strata = int(self.strata.max()) + 1
            self._n_stim_bins_actual = n_s
            self._n_response_bins_actual = int(r_labels.max()) + 1
        else:
            self.strata = None
            self.n_strata = 0

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        traj, start = self._windows[idx]
        window = traj[start : start + self.window_size]
        return torch.from_numpy(window)

    def get_window(self, idx: int) -> np.ndarray:
        """Return the raw window as a numpy array (window_size, n_features)."""
        traj, start = self._windows[idx]
        return traj[start : start + self.window_size]

    def stratum_counts(self) -> np.ndarray:
        """Number of windows per stratum index (length n_strata)."""
        assert self.strata is not None, "Dataset was constructed with stratify=False"
        return np.bincount(self.strata, minlength=self.n_strata)

    def stratum_label(self, s: int) -> str:
        """Human-readable (response_bin, stim_bin) label for stratum index `s`."""
        n_s = self._n_stim_bins_actual
        return f"r{s // n_s}_s{s % n_s}"

    def make_balanced_sampler(
        self,
        num_samples: int | None = None,
        replacement: bool = True,
    ) -> WeightedRandomSampler:
        """Return a sampler that draws windows with weights inversely proportional
        to their stratum size, yielding (in expectation) balanced batches."""
        assert self.strata is not None, "Dataset was constructed with stratify=False"
        counts = self.stratum_counts().astype(np.float64)
        # Avoid /0 on empty strata (shouldn't happen but be safe).
        counts[counts == 0] = 1.0
        weights = 1.0 / counts[self.strata]
        # Normalize so per-window weights sum to 1 (WeightedRandomSampler doesn't
        # require this, but it makes the values interpretable as probabilities).
        weights = weights / weights.sum()
        if num_samples is None:
            num_samples = len(self.strata)
        return WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=replacement,
        )

    def sample_per_stratum(
        self,
        n_per_stratum: int = 10,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Sample up to `n_per_stratum` windows from each stratum and return
        a long-format DataFrame suitable for plotting.

        Columns: stratum (int), stratum_label (str), sample_id (int),
                 window_idx (int), uid, t (0..window_size-1), and one column
                 per feature in `feature_columns`.
        """
        assert self.strata is not None, "Dataset was constructed with stratify=False"
        rng = np.random.default_rng(seed)
        rows = []
        for s in range(self.n_strata):
            idx = np.where(self.strata == s)[0]
            if len(idx) == 0:
                continue
            k = min(n_per_stratum, len(idx))
            picked = rng.choice(idx, size=k, replace=False)
            for sample_id, w_idx in enumerate(picked):
                window = self.get_window(int(w_idx))
                uid = self._uids[int(w_idx)]
                for t in range(self.window_size):
                    row = {
                        "stratum": int(s),
                        "stratum_label": self.stratum_label(int(s)),
                        "sample_id": int(sample_id),
                        "window_idx": int(w_idx),
                        "uid": uid,
                        "t": int(t),
                    }
                    for j, c in enumerate(self.feature_columns):
                        row[c] = float(window[t, j])
                    rows.append(row)
        return pd.DataFrame(rows)
