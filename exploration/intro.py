import marimo

__generated_with = "0.23.4"
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
    #sim = pd.read_parquet('./stochastic_sim_v2_output.parquet')
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


@app.cell(hide_code=True)
def _():
    from erkdataset import ERKWindowDataset

    return (ERKWindowDataset,)


@app.cell(hide_code=True)
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
    EPOCHS = 5
    SEQ_LEN = 25 
    dataset = ERKWindowDataset(
        df=real,
        feature_columns=FEATURE_COLUMNS,
        window_size=SEQ_LEN,
        stride=13,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    # Each batch: (64, 50, n_features)
    return EPOCHS, SEQ_LEN, loader


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
def _(EPOCHS, SEQ_LEN, device, nn, np, torch):
    class SubNetwork(nn.Module):
        def __init__(self, in_dim, out_dim, n_layers=2):
            super(SubNetwork, self).__init__()
            self.n_layers = n_layers 
            self.rnn = nn.GRU(in_dim, out_dim, num_layers=n_layers, batch_first=True)
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
            bar = None
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
                for i, x in enumerate(train_dl):
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

                    _h0    = enc_x[:, 0, :].unsqueeze(0).repeat(self.generator.n_layers, 1, 1)
                    _z     = torch.randn(_BS, _sq - 1, self.data_dim, device=self.device)

                    _h_pred = self.generator(_z, h0=_h0) # offender

                    l_supervised  = mse_l(_h_pred, enc_x[:, 1:, :])

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
                    if bar is not None:
                        bar.update(increment=1, title=f"Epoch {e+1} | Batch {i+1}", subtitle=f"rec={l_rec:.4f}, uns={l_unsupervised:.4f}, sup={l_supervised:.4f}")

            return self.losses

        def _sample_noise(self, n):
            _z = torch.distributions.Normal(
                torch.zeros((self.data_dim,)),
                torch.ones((self.data_dim,)),
            )
            return _z.sample(
                (n,)
            ).to(device)

        def generate_random(self,length=None, h0=None):
            '''
            Generate a latent sequence of `length` using just Generator
            '''
            if length is None:
                length = SEQ_LEN

            _z = self._sample_noise(length)
            _gnrtd = self.generator(_z, h0)
            return _gnrtd

        def generate_data(self, length):
            _l = self.generate_random(length)
            _r = self.recovery(_l)
            return _r


    return (TimeGAN,)


@app.cell
def _(EPOCHS, FEATURE_COLUMNS, TimeGAN, loader, mo):
    _tg = TimeGAN(len(FEATURE_COLUMNS), 3)
    #_tg = torch.compile(_tg)
    with mo.status.progress_bar(total=EPOCHS * len(loader)) as _bar:
        final_losses = _tg.train__(loader, _bar)
    return (final_losses,)


@app.cell
def _(EPOCHS, final_losses, pd, qplot):
    #_rlosses = final_losses.reshape(EPOCHS, -1, 3).mean(1)
    _c = ['reconstruction', 'unsupervised', 'supervised']
    batch_count = final_losses.shape[0] // EPOCHS
    _rlosses = final_losses.reshape(EPOCHS, batch_count, 3).mean(1)  # (EPOCHS, 3)
    _df = pd.DataFrame(_rlosses, columns=_c)
    _df.index.name='epoch'
    _df = _df.reset_index()
    _df_long = _df.melt(
        id_vars="epoch",
        value_vars=_c,
        var_name="type",
        value_name="value"
    )
    qplot(_df_long, 'epoch', 'value', facet_col='type')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
