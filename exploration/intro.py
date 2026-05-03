import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from hastyplot import qplot 
    import numpy as np
    import pandas as pd

    return mo, np, pd, qplot


@app.cell
def _(pd):
    sim = pd.read_parquet('./stochastic_sim_v2_output.parquet')
    real = pd.read_parquet('./dataset.parquet')
    return (real,)


@app.cell
def _(real):
    real
    return


@app.cell
def _(real):
    _g = real.groupby('ramp_pattern_name')
    _g['frame'].max(), _g['uid'].count()
    # _g.apply(lambda _d: _d['frame'].max())
    # _g.apply(lambda _d: _d['uid'].count())
    return


@app.cell
def _(mo, qplot, real):
    _df = real.groupby(['uid', 'ramp_pattern_name'])['frame'].count()
    _df = _df.reset_index()
    mo.ui.altair_chart(
        qplot(data=_df, x='frame', facet_col='ramp_pattern_name')
    )
    return


@app.cell
def _(real):
    real.info()
    return


@app.cell(disabled=True)
def _():
    #class ERKWindowDataset(Dataset):
        # """
        # Per-cell sliding window dataset over ERK trajectory data.

        # Each sample is a fixed-length window of shape (window_size, n_features)
        # drawn from a single cell's (uid) time series, sorted by frame.

        # Args:
        #     df:             Source dataframe with at least `uid`, `frame`, and all
        #                     columns in feature_columns.
        #     feature_columns: Ordered list of column names to use as features.
        #     window_size:    Number of consecutive frames per sample.
        #     stride:         Step size between window starts (default 1).
        # """

        # def __init__(
        #     self,
        #     df: pd.DataFrame,
        #     feature_columns: list[str],
        #     window_size: int,
        #     stride: int = 1,
        # ):
        #     self.feature_columns = feature_columns
        #     self.window_size = window_size

        #     # Build index: list of (array_of_features, start_idx) per valid window
        #     self._windows: list[tuple[np.ndarray, int]] = []

        #     for uid, group in df.groupby("uid"):
        #         # Sort chronologically within each cell
        #         traj = (
        #             group.sort_values("frame")[feature_columns]
        #             .to_numpy(dtype=np.float32)
        #         )

        #         # Slide windows; cells shorter than window_size are silently skipped
        #         for start in range(0, len(traj) - window_size + 1, stride):
        #             self._windows.append((traj, start))

        # def __len__(self) -> int:
        #     return len(self._windows)

        # def __getitem__(self, idx: int) -> torch.Tensor:
        #     traj, start = self._windows[idx]
        #     window = traj[start : start + self.window_size]  # (window_size, n_features)
        #     return torch.from_numpy(window)
    return


@app.cell
def _():
    from erkdataset import ERKWindowDataset

    return (ERKWindowDataset,)


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset, DataLoader

    FEATURE_COLUMNS = """
    fluence_mJ_cm2
    ewma_fast
    cnr_mean_norm
    n_5
    median_cnr_0_9
    """.split()
    FEATURE_COLUMNS
    return (FEATURE_COLUMNS,)


@app.cell
def _(ERKWindowDataset, FEATURE_COLUMNS, real, torch):

    dataset = ERKWindowDataset(
        df=real,
        feature_columns=FEATURE_COLUMNS,
        window_size=50,
        stride=20,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    # Each batch: (64, 50, n_features)
    return (loader,)


app._unparsable_cell(
    r"""
    | Column | Type | Description |
    |---|---|---|
    | `cnr` | float64 | Cytoplasm-to-nucleus ratio: `mean_intensity_C1_ring / mean_intensity_C1_nuc`. |
    | `cnr_median` | float64 | Same ratio using median intensities. |
    | `uid` | string | Unique cell identifier: `ramp_pattern_name + str(fov) + "_" + str(particle)`. Incorporates the ramp pattern so cells from different stimulation patterns are distinguishable. |
    | `frame` | uint32 | Alias for `timestep`. |
    | `cnr_median_norm` | float64 | `cnr_median` divided by the per-cell baseline median of `cnr_median` (over frames `< norm_until_timepoint`). |
    | `cnr_norm` | float64 | `cnr` divided by the per-cell baseline median of `cnr` (over frames `< norm_until_timepoint`). |
    | `median_cnr_0_9` | float64 | Per-cell median of `cnr` over the baseline window (frames 0 to `norm_until_timepoint - 1`). Merged onto every row for that cell. |
    | `energy_uJ` | float64 | Energy per stimulation pulse in microjoules: `P_uW * stim_exposure * 1e-3`. Interpolated from the ND5 calibration table. `0` when `stim_exposure` is `0`. |
    | `fluence_mJ_cm2` | float64 | Fluence (energy dose per unit area) per pulse: `irradiance * stim_exposure * 1e-3` (mJ/cm2). |
    | `energy_per_cell` | float64 | `fluence_mJ_cm2 * area`. Not in physical units since `area` is in pixels. **Warning:** nuclear area changes with ERK perturbation, so this column should not be used for quantitative comparisons. |

    ---

    ## Stimulation Input Features (`add_stim_features`)

    Implemented in `notebooks/experiment/preprocessing.py`. This function augments the DataFrame with 9 per-cell stimulation input channels intended for modeling single-cell ERK responses. Assumes a uniform 1-minute grid (1 frame = 1 minute). Must be called **after** `load_and_clean` (which provides `fluence_mJ_cm2` and `stim`).

    ```python
    from notebooks.experiment.preprocessing import add_stim_features
    df = add_stim_features(df)
    ```

    ### Parameters

    | Parameter | Default | Description |
    |---|---|---|
    | `df` | *(required)* | DataFrame produced by `load_and_clean`. |
    | `window_min` | `5` | Rolling window size in frames (minutes) for `n_5` and `slope_5`. |
    | `ewma_alpha_fast` | `0.5` | Smoothing factor for fast EWMA (half-life ~1 min). |
    | `ewma_alpha_slow` | `0.1` | Smoothing factor for slow EWMA (half-life ~7 min). |

    ### Columns added

    | Column | Type | Description |
    |---|---|---|
    | `u_t` | float64 | Raw pulse amplitude (`fluence_mJ_cm2`). Primary input signal. |
    | `m_t` | int | Activation indicator: 1 if stimulated, 0 otherwise. Disambiguates rest from zero-amplitude frames. |
    | `dt_since_pulse` | float64 | Minutes (frames) since the most recent pulse for this cell. `NaN` before the cell's first pulse. Captures gap dynamics. |
    | `ewma_fast` | float64 | Exponentially weighted moving average of `u_t` with `alpha=0.5`. Short-term effective stimulation level (equivalent to a discretised first-order ODE with fast decay). |
    | `ewma_slow` | float64 | EWMA of `u_t` with `alpha=0.1`. Medium-term accumulation (slow decay). |
    | `n_5` | int | Number of pulses (`m_t == 1`) in the last `window_min` frames. Measures burst density. |
    | `slope_5` | float64 | OLS slope of `u_t` over the last `window_min` frames. Detects ramp-up / ramp-down patterns. |
    | `burst_pos` | int | 1-indexed position within the current consecutive burst of stimulated frames. 0 when the cell is not stimulated. Captures adaptation/facilitation within a burst. |
    | `s_cum` | float64 | Cumulative sum of `u_t` up to and including the current frame. Total light exposure history. |

    ### Design rationale

    - **`u_t`, `m_t`, `dt_since_pulse`** are direct encodings of the stimulation protocol — essentially free and universally useful.
    - **`ewma_fast`, `ewma_slow`** bridge deep learning and classical compartmental modeling: each EWMA is a discretised first-order ODE with a different time constant.
    - **`n_5`, `slope_5`, `burst_pos`** capture higher-level temporal patterns (burst density, ramps, within-burst position).
    - **`s_cum`** provides the model with long-term exposure history.

    ---
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell
def _():

    import torch.nn as nn
    import torch.functional as F
    import torch
    from tqdm import tqdm

    return nn, torch


@app.cell
def _(torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device
    return (device,)


@app.cell
def _(device, mo, nn, np, torch):
    EPOCHS = 2 
    SEQ_LEN = 50




    class SubNetwork(nn.Module):
        def __init__(self, in_dim, out_dim, n_layers=2):
            super(SubNetwork, self).__init__()
            self.n_layers = n_layers 
            self.rnn = nn.GRU(in_dim, out_dim, num_layers=n_layers)
            self.fc =  nn.Linear(out_dim, out_dim)
            self.nl = nn.Sigmoid()

        def forward(self, input, h0=None):
            if h0 is not None:
                o,_ = self.rnn(input, h0)
            else:
                o,_ = self.rnn(input)
            o = self.fc(o)
            return self.nl(o)

    class TimeGAN(nn.Module):
        def __init__(self, data_dim, latent_dim):
            super(TimeGAN, self).__init__()
            self.data_dim, self.latent_dim = data_dim, latent_dim
            self.encoder = SubNetwork(data_dim, latent_dim)#.to(device)
            self.recovery = SubNetwork(latent_dim, data_dim)
            self.generator = SubNetwork(data_dim ,latent_dim)
            self.discriminator = SubNetwork(latent_dim,1)
            self.to(device)
            self.device = device

        def train__(
            self,
            train_dl:torch.utils.data.DataLoader,
            test_dl :torch.utils.data.DataLoader
        ):
            batch_count = len(train_dl)
            self.losses = np.empty((EPOCHS * batch_count , 3))
            optim_encoder =  torch.optim.Adam(params=self.encoder.parameters()) 
            optim_recovery =  torch.optim.Adam(params=self.recovery.parameters()) 
            optim_discriminator =  torch.optim.Adam(params=self.discriminator.parameters()) 
            optim_generator =  torch.optim.Adam(params=self.generator.parameters()) 

            bce_l = nn.BCELoss(reduction='mean')
            mse_l = nn.MSELoss(reduction='mean')

            for e in range(EPOCHS):
                print(f'\n===================== EPOCH{e} =======================')
                for i, x in mo.status.progress_bar( enumerate(train_dl) , show_eta=True, show_rate=True, total=len(train_dl)):
                    print('batch ', i )
                    x = x.to(device)

                    # Load and convert to L 
                    _BS,_sq, _feat_len = x.shape
                    assert _sq == SEQ_LEN
                    enc_x = self.encoder(x)
                    rec_x = self.recovery(enc_x)
                    _generated = self.generate_random(SEQ_LEN * _BS).reshape(_BS, _sq, 3)

                    if i == 0 and e == 0:
                        print('_BS,_sq, _feat_len')
                        print(_BS,_sq, _feat_len)
                        print('enc',enc_x.shape)
                        print('rec',rec_x.shape)
                        print('generated shape',_generated.shape)


                    # Forward pass

                    _disc_data = torch.concat([enc_x, _generated])
                    _idx = torch.concat([torch.ones(_BS), torch.zeros(_BS)]).to(device)

                    _perm = torch.randperm(2*_BS,)
                    _inv_perm = nn.utils.rnn.invert_permutation(_perm)

                    _res = self.discriminator(_disc_data[_perm])

                    _y = torch.repeat_interleave(_idx[_perm], SEQ_LEN).reshape(2*_BS,SEQ_LEN).unsqueeze(2)

                    l_unsupervised = bce_l(_res, _y)

                    l_rec = mse_l(x, rec_x)

                    # supervised loss
                    _normies = torch.empty(_BS)

                    for _bi, _B in enumerate(enc_x): # SEQ_LEN, LATENT
                        real = _B[1:]
                        predicted = torch.empty_like(real)
                        _Bl = list(_B)

                        for _iL, _h0 in enumerate(_Bl[:-1]) :
                            _pred = self.generate_random(1, h0=_h0.unsqueeze(0).repeat(self.generator.n_layers, 1))
                            predicted[_iL] = _pred.unsqueeze(0)

                        _normies[_bi] = torch.linalg.norm(real - predicted,dim=1).sum()

                    _normean = _normies.mean()
                    l_supervised = _normean


                    # entire supervised loss in 3 lines, fully batched

                    # h0    = enc_x[:, 0, :].unsqueeze(0).repeat(self.generator.n_layers, 1, 1)
                    # z     = torch.randn(_BS, _sq - 1, self.data_dim, device=self.device)
                    # h_pred = self.generator(z, h0=h0)
                    # l_sup  = mse_l(h_pred, enc_x[:, 1:, :])


                    # ----------------- Parameter update
                    optim_encoder.zero_grad()
                    optim_generator.zero_grad()
                    optim_recovery.zero_grad()
                    optim_discriminator.zero_grad()

                    l_rec.backward(retain_graph=True)
                    l_unsupervised.backward(retain_graph=True)
                    l_supervised.backward(retain_graph=True)

                    self.losses[e * batch_count + i ] = [_i.detach().item() for _i in [l_rec, l_unsupervised, l_supervised]]

                    optim_encoder.step()
                    optim_generator.step()
                    optim_recovery.step()
                    optim_discriminator.step()
            return self.losses

        def _sample_noise(self, n):
            _z = torch.distributions.Normal(
                torch.zeros((self.data_dim,)),
                torch.ones((self.data_dim,)),
            )
            return _z.sample(
                (n,)
            ).to(device)

        def _rec_loss(self, enc, rec):
            pass

        def generate_random(self,length=None, h0=None):
            '''
            Generate a sequence of `length` using just Generator
            '''
            if length is None:
                length = SEQ_LEN

            _z = self._sample_noise(length)
            _gnrtd = self.generator(_z, h0)
            return _gnrtd

        def generate_from(self, base, n):
            '''
            Encode the user-provided `base` sequence, then
            feed it as a starting hidden state to the generator, and sample.
            '''
            pass

        def predict(self, *args, **kwargs): 
            return generate_from(*args,**kwargs)

    return EPOCHS, TimeGAN


@app.cell
def _(FEATURE_COLUMNS, TimeGAN, loader, torch):
    _tg = TimeGAN(len(FEATURE_COLUMNS), 3)
    _tg = torch.compile(_tg)
    final_losses = _tg.train__(loader, None)
    return (final_losses,)


@app.cell
def _(EPOCHS, df, final_losses, np, pd, qplot):
    _rlosses = final_losses.reshape(EPOCHS, -1, 3).mean(1)
    _rec, _sup, _unsup = _rlosses[:, 0], _rlosses[:, 1], _rlosses[:, 2]
    _t = np.linspace(0, len(_rec))
    _df = pd.DataFrame({"rec": _rec, "sup": _sup, "unsup": _unsup, "t":_t})
    qplot(df, 't', 'rec')
    return


@app.cell
def _(mo, torch):
    _x = torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32).reshape(5,2)
    _y = torch.tensor([5,10,15,20,15])
    _yp = torch.repeat_interleave(_y, 2).reshape(5,2)

    mo.ui.matrix(_x),  mo.ui.matrix(_yp),  mo.ui.matrix(_x - _yp), mo.ui.matrix(torch.linalg.norm(_x - _yp,dim=1))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
