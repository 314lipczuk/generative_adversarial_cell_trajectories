import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from hastyplot import qplot 
    import numpy as np
    import pandas as pd

    return mo, pd, qplot


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
    cnr_mean_norm
    median_cnr_0_9
    """.split()
    FEATURE_COLUMNS
    return (FEATURE_COLUMNS,)


@app.cell
def _(ERKWindowDataset, FEATURE_COLUMNS, real, torch):
    EPOCHS = 100
    SEQ_LEN = 25
    dataset = ERKWindowDataset(
        df=real,
        feature_columns=FEATURE_COLUMNS,
        window_size=SEQ_LEN,
        stride=13,
        stratify=True,
        response_col="cnr_mean_norm",
        stim_col="fluence_mJ_cm2",
        n_response_bins=3,
        n_stim_bins=3,
    )

    balanced_sampler = dataset.make_balanced_sampler()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        sampler=balanced_sampler,
        num_workers=0,
    )
    return EPOCHS, SEQ_LEN, dataset, loader


@app.cell
def _(dataset, pd):
    # Per-stratum window counts. Labels read as r{response_bin}_s{stim_bin},
    # where response_bin is the cnr_mean_norm peak-to-peak quantile and
    # stim_bin is the fluence_mJ_cm2 mean quantile.
    _counts = dataset.stratum_counts()
    _stratum_count_df = pd.DataFrame(
        {
            "stratum": list(range(dataset.n_strata)),
            "stratum_label": [dataset.stratum_label(i) for i in range(dataset.n_strata)],
            "n_windows": _counts,
        }
    )
    _stratum_count_df
    return


@app.cell
def _(dataset):
    # Bin edges actually used (after pd.qcut drops duplicate quantiles for ties).
    {
        "response_edges (cnr_mean_norm ptp)": dataset.response_edges.tolist(),
        "stim_edges (fluence_mJ_cm2 mean)": dataset.stim_edges.tolist(),
    }
    return


@app.cell
def _(dataset):
    strata_samples = dataset.sample_per_stratum(n_per_stratum=10, seed=0)
    return (strata_samples,)


@app.cell
def _(mo, strata_samples):
    import altair as alt

    # cnr_mean_norm — the response axis. Each line is one sampled window;
    # facets are strata, ordered so response amplitude increases left→right
    # within each row and stimulus dose increases row-by-row.
    _resp_chart = (
        alt.Chart(strata_samples)
        .mark_line(opacity=0.55)
        .encode(
            x=alt.X("t:Q", title="frame within window"),
            y=alt.Y("cnr_mean_norm:Q", title="cnr_mean_norm"),
            color=alt.Color("sample_id:N", legend=None),
            detail="window_idx:N",
        )
        .properties(width=160, height=110)
        .facet(facet=alt.Facet("stratum_label:N", title="stratum"), columns=3)
        .properties(title="cnr_mean_norm per stratum (10 samples each)")
    )
    mo.ui.altair_chart(_resp_chart)
    return (alt,)


@app.cell
def _(alt, mo, strata_samples):
    # fluence_mJ_cm2 — the stimulus axis. Useful sanity check that the
    # stim-bin axis really separates dose levels.
    _stim_chart = (
        alt.Chart(strata_samples)
        .mark_line(opacity=0.55)
        .encode(
            x=alt.X("t:Q", title="frame within window"),
            y=alt.Y("fluence_mJ_cm2:Q", title="fluence_mJ_cm2"),
            color=alt.Color("sample_id:N", legend=None),
            detail="window_idx:N",
        )
        .properties(width=160, height=110)
        .facet(facet=alt.Facet("stratum_label:N", title="stratum"), columns=3)
        .properties(title="fluence_mJ_cm2 per stratum (10 samples each)")
    )
    mo.ui.altair_chart(_stim_chart)
    return


@app.cell
def _():

    import torch.nn as nn
    import torch.functional as F
    import torch
    from tqdm import tqdm

    return nn, torch


@app.cell
def _(torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    device
    return (device,)


@app.cell
def _(SEQ_LEN, device, nn, torch):
    class SubNetwork(nn.Module):
        def __init__(
            self, in_dim, out_dim, n_layers=2, final_activation=nn.Sigmoid
        ):
            super().__init__()
            self.n_layers = n_layers
            self.rnn = nn.GRU(
                in_dim, out_dim, num_layers=n_layers, batch_first=True
            )
            self.fc = nn.Linear(out_dim, out_dim)
            self.nl = final_activation()

        def forward(self, input, h0=None):
            if h0 is not None:
                o, _ = self.rnn(input, h0)
            else:
                o, _ = self.rnn(input)
            return self.nl(self.fc(o))


    class TimeGAN(nn.Module):
        def __init__(self, data_dim, latent_dim, seq_len=SEQ_LEN):
            super().__init__()
            self.data_dim = data_dim
            self.latent_dim = latent_dim
            self.seq_len = seq_len
            self.device = device

            self.encoder = SubNetwork(data_dim, latent_dim)
            self.recovery = SubNetwork(
                latent_dim, data_dim, final_activation=nn.Identity
            )
            self.generator = SubNetwork(data_dim, latent_dim)
            self.supervisor = SubNetwork(latent_dim, latent_dim)
            self.discriminator = SubNetwork(latent_dim, 1)
            self.to(device)

        def _sample_noise(self, batch_size, seq_len=None):
            seq_len = seq_len or self.seq_len
            return torch.randn(
                batch_size, seq_len, self.data_dim, device=self.device
            )

        def fit(
            self,
            train_dl,
            epochs_embed=5,
            epochs_supervise=5,
            epochs_joint=5,
            d_skip_threshold=0.15,
            mm_weight=100.0,
            sup_weight=100.0,
            bar=None,
        ):
            bce = nn.BCELoss()
            mse = nn.MSELoss()
            opt_er = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.recovery.parameters())
            )
            opt_s = torch.optim.Adam(self.supervisor.parameters())
            opt_g = torch.optim.Adam(
                list(self.generator.parameters())
                + list(self.supervisor.parameters())
            )
            opt_e_joint = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.recovery.parameters())
            )
            opt_d = torch.optim.Adam(self.discriminator.parameters())
            losses = {"embed": [], "supervise": [], "joint": []}

            for e in range(epochs_embed):
                for x in train_dl:
                    x = x.to(self.device)
                    h = self.encoder(x)
                    x_tilde = self.recovery(h)
                    l_rec = mse(x_tilde, x)
                    opt_er.zero_grad()
                    l_rec.backward()
                    opt_er.step()
                    losses["embed"].append({"rec": l_rec.item()})
                    if bar is not None:
                        bar.update(
                            increment=1,
                            title=f"[1/3 Embed] Ep {e + 1}",
                            subtitle=f"rec={l_rec.item():.4f}",
                        )

            for e in range(epochs_supervise):
                for x in train_dl:
                    x = x.to(self.device)
                    with torch.no_grad():
                        h = self.encoder(x)
                    h_pred = self.supervisor(h[:, :-1, :])
                    l_sup = mse(h_pred, h[:, 1:, :])
                    opt_s.zero_grad()
                    l_sup.backward()
                    opt_s.step()
                    losses["supervise"].append({"sup": l_sup.item()})
                    if bar is not None:
                        bar.update(
                            increment=1,
                            title=f"[2/3 Sup] Ep {e + 1}",
                            subtitle=f"sup={l_sup.item():.4f}",
                        )

            for e in range(epochs_joint):
                for x in train_dl:
                    x = x.to(self.device)
                    bs = x.shape[0]
                    z = self._sample_noise(bs)
                    e_hat = self.generator(z)
                    h_hat = self.supervisor(e_hat)
                    y_fake = self.discriminator(h_hat)
                    l_g_adv = bce(y_fake, torch.ones_like(y_fake))
                    h = self.encoder(x)
                    h_sup_pred = self.supervisor(h[:, :-1, :])
                    l_g_sup = mse(h_sup_pred, h[:, 1:, :])
                    x_hat = self.recovery(h_hat)
                    l_g_v1 = torch.mean(
                        torch.abs(
                            torch.sqrt(x_hat.var(dim=0, unbiased=False) + 1e-6)
                            - torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)
                        )
                    )
                    l_g_v2 = torch.mean(
                        torch.abs(x_hat.mean(dim=0) - x.mean(dim=0))
                    )
                    l_g_mm = l_g_v1 + l_g_v2
                    l_g = (
                        l_g_adv
                        + sup_weight * torch.sqrt(l_g_sup + 1e-8)
                        + mm_weight * l_g_mm
                    )
                    opt_g.zero_grad()
                    l_g.backward()
                    opt_g.step()

                    h = self.encoder(x)
                    x_tilde = self.recovery(h)
                    l_e_rec = mse(x_tilde, x)
                    h_sup_pred = self.supervisor(h[:, :-1, :])
                    l_e_sup = mse(h_sup_pred, h[:, 1:, :])
                    l_e = 10.0 * torch.sqrt(l_e_rec + 1e-8) + 0.1 * l_e_sup
                    opt_e_joint.zero_grad()
                    l_e.backward()
                    opt_e_joint.step()

                    with torch.no_grad():
                        h_real = self.encoder(x)
                        z_d = self._sample_noise(bs)
                        h_hat_d = self.supervisor(self.generator(z_d))
                    y_real = self.discriminator(h_real)
                    y_fake_d = self.discriminator(h_hat_d)
                    l_d = bce(y_real, torch.ones_like(y_real)) + bce(
                        y_fake_d, torch.zeros_like(y_fake_d)
                    )
                    opt_d.zero_grad()
                    _stepped = l_d.item() > d_skip_threshold
                    if _stepped:
                        l_d.backward()
                        opt_d.step()

                    losses["joint"].append(
                        {
                            "rec": l_e_rec.item(),
                            "sup": l_g_sup.item(),
                            "g_adv": l_g_adv.item(),
                            "g_mm": l_g_mm.item(),
                            "d": l_d.item(),
                            "d_stepped": float(_stepped),
                        }
                    )
                    if bar is not None:
                        bar.update(
                            increment=1,
                            title=f"[3/3 Joint] Ep {e + 1}",
                            subtitle=f"rec={l_e_rec.item():.3f} sup={l_g_sup.item():.3f} g={l_g_adv.item():.3f} mm={l_g_mm.item():.3f} d={l_d.item():.3f}{'' if _stepped else ' [skip]'}",
                        )
            self.losses = losses
            return losses

        def generate(self, batch_size, seq_len=None):
            seq_len = seq_len or self.seq_len
            self.eval()
            with torch.no_grad():
                z = self._sample_noise(batch_size, seq_len)
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                x_hat = self.recovery(h_hat)
            return x_hat

    return (TimeGAN,)


@app.cell
def _(EPOCHS, FEATURE_COLUMNS, TimeGAN, loader, mo, torch):
    tg = TimeGAN(len(FEATURE_COLUMNS), 8)
    with mo.status.progress_bar(total=3 * EPOCHS * len(loader)) as _bar:
        final_losses = tg.fit(
            loader,
            epochs_embed=EPOCHS,
            epochs_supervise=EPOCHS,
            epochs_joint=EPOCHS,
            bar=_bar,
        )
    torch.save(tg, 'timeGAN_v2.pt')
    return final_losses, tg


@app.cell
def _(final_losses, pd, qplot):
    _rows = []
    for _stage, _entries in final_losses.items():
        for _i, _ent in enumerate(_entries):
            for _k, _v in _ent.items():
                _rows.append(
                    {"stage": _stage, "iter": _i, "facet": f"{_stage}:{_k}", "value": _v}
                )
    _df_long = pd.DataFrame(_rows)
    qplot(_df_long, "iter", "value", facet_col="facet")
    return


@app.cell(hide_code=True)
def _(FEATURE_COLUMNS, dataset, device, mo, pd, tg):
    import torch as _t
    import altair as _alt

    tg.eval()
    _n = 5
    _real = _t.stack([dataset[i] for i in range(_n)]).to(device)
    with _t.no_grad():
        _gen = tg.generate(_n)

    _rows = []
    for _src, _arr in [
        ("real", _real.cpu().numpy()),
        ("generated", _gen.cpu().numpy()),
    ]:
        for _sid in range(_arr.shape[0]):
            for _ti in range(_arr.shape[1]):
                for _fi, _fn in enumerate(FEATURE_COLUMNS):
                    _rows.append(
                        {
                            "source": _src,
                            "cell": _sid,
                            "feature": _fn,
                            "row": f"cell{_sid} | {_fn}",
                            "t": _ti,
                            "value": float(_arr[_sid, _ti, _fi]),
                        }
                    )
    _df_cmp = pd.DataFrame(_rows)

    # Sort row order: cell-major, feature-minor
    _row_order = [f"cell{c} | {f}" for c in range(_n) for f in FEATURE_COLUMNS]

    _base = (
        _alt.Chart(_df_cmp)
        .mark_line()
        .encode(
            x=_alt.X("t:Q", title="frame"),
            y=_alt.Y("value:Q", title=None),
        )
        .properties(width=180, height=70)
    )

    _chart_cmp = (
        _base.facet(
            row=_alt.Row(
                "row:N",
                sort=_row_order,
                title=None,
                header=_alt.Header(labelAngle=0, labelAlign="left"),
            ),
            column=_alt.Column("source:N", title=None),
        )
        .resolve_scale(y="independent")
        .properties(title="real vs generated, per (cell, feature)")
    )
    mo.ui.altair_chart(_chart_cmp)
    return


@app.cell(hide_code=True)
def _(FEATURE_COLUMNS, SEQ_LEN, dataset, device, mo, pd, tg):
    import torch as _t
    import altair as _alt

    tg.eval()
    _n = 5
    _prefix_len = 12
    _real_s = _t.stack([dataset[i] for i in range(_n)]).to(device)
    with _t.no_grad():
        _h = tg.encoder(_real_s)
        _h_pref = _h[:, :_prefix_len, :]
        _out, _hid = tg.supervisor.rnn(_h_pref)
        _last = tg.supervisor.nl(tg.supervisor.fc(_out[:, -1:, :]))
        _rolled = [_last]
        _cur = _last
        for _ in range(SEQ_LEN - _prefix_len - 1):
            _out, _hid = tg.supervisor.rnn(_cur, _hid)
            _cur = tg.supervisor.nl(tg.supervisor.fc(_out))
            _rolled.append(_cur)
        _h_cont = _t.cat(_rolled, dim=1)
        _h_full = _t.cat([_h_pref, _h_cont], dim=1)
        _x_pred = tg.recovery(_h_full)

    _real_np = _real_s.cpu().numpy()
    _pred_np = _x_pred.cpu().numpy()

    _rows = []
    for _sid in range(_n):
        for _ti in range(SEQ_LEN):
            _region = "prefix" if _ti < _prefix_len else "rollout"
            for _fi, _fn in enumerate(FEATURE_COLUMNS):
                _rows.append(
                    {
                        "sample": _sid,
                        "t": _ti,
                        "feature": _fn,
                        "kind": f"real ({_region})",
                        "value": float(_real_np[_sid, _ti, _fi]),
                    }
                )
                _rows.append(
                    {
                        "sample": _sid,
                        "t": _ti,
                        "feature": _fn,
                        "kind": f"pred ({_region})",
                        "value": float(_pred_np[_sid, _ti, _fi]),
                    }
                )
    _df_sup = pd.DataFrame(_rows)

    _base_sup = (
        _alt.Chart(_df_sup)
        .mark_line(opacity=0.85)
        .encode(
            x=_alt.X("t:Q", title="frame"),
            y=_alt.Y("value:Q"),
            color=_alt.Color(
                "kind:N",
                scale=_alt.Scale(
                    domain=[
                        "real (prefix)",
                        "pred (prefix)",
                        "real (rollout)",
                        "pred (rollout)",
                    ],
                    range=["#1f77b4", "#aec7e8", "#d62728", "#ff9896"],
                ),
            ),
            detail="sample:N",
            strokeDash=_alt.StrokeDash("kind:N"),
        )
        .properties(
            width=200,
            height=130,
            title=f"Supervisor rollout: prefix t<{_prefix_len}",
        )
    )

    _chart_sup = _base_sup.facet(
        column=_alt.Column("feature:N", title=None),
        row=_alt.Row("sample:N", title="sample"),
    ).resolve_scale(y="independent")
    mo.ui.altair_chart(_chart_sup)
    return


@app.cell(hide_code=True)
def changelog(mo):
    mo.md(r"""
    ## Changelog

    ### Architecture & training (vs. starting point)
    - **3-stage training**: embedder+recovery on reconstruction → supervisor on next-step prediction in real latent space → joint adversarial.
    - **Supervisor network added** — separate from generator. Generator emits raw latent; supervisor enforces temporal dynamics. G output is passed through supervisor before D sees it.
    - **Separate G / E / D optimizer steps** with correct adversarial signs: G minimizes `BCE(D(fake),1)`, D minimizes `BCE(D(real),1) + BCE(D(fake),0)`. Earlier code stepped G+D on the same loss with the same sign — broken.
    - **Recovery uses `nn.Identity`** (not sigmoid). Features aren't bounded to [0,1].
    - **Latent dim 8** (up from 3). Latent ≠ bottleneck — it's the space where dynamics are learnable.
    - **Public `tg`** instead of cell-private `_tg` — viz cells reference it directly, no `globals()` hack.

    ### Mode-collapse mitigations
    - **Moment matching** loss: `mean(|var(x_hat) − var(x)|) + mean(|mean(x_hat) − mean(x)|)` per feature, batch-wise. Weighted 100× in `l_g`. Directly punishes low-diversity output.
    - **D-skip threshold**: skip `opt_d.step()` when `l_d < 0.15`. Prevents D from saturating and killing G's gradient.
    - **Tunable weights** as `fit()` args: `mm_weight`, `sup_weight`, `d_skip_threshold`.

    ### Feature reduction
    - Dropped engineered features `ewma_fast` and `n_5`. Kept `fluence_mJ_cm2` (stim), `cnr_mean_norm` (ERK), `median_cnr_0_9` (baseline). Engineered features are deterministic functions of raw signals — G shouldn't waste capacity recomputing them, and D gets free tells if G's recomputation is imperfect.

    ### Visualizations
    - **Fsup**: 5 real vs 5 generated trajectories, faceted per feature.
    - **AXDa**: supervisor rollout — real prefix `t<12`, model-continued `t≥12`. Color separates real vs predicted × prefix vs rollout.

    ### Discussion (not implemented)
    - **Stratum / stim conditioning** — would split the multi-mode problem into per-stim subproblems, ideal for the downstream "stim → ERK" pretraining goal. Decided to first push unconditional with the knobs above; condition on stim only (continuous fluence or one-hot bin) if collapse persists. Conditioning is the inductive bias that matches the downstream task — not a cheat.

    ### Operational notes
    - `EPOCHS = 1` and `num_workers = 0` for fast iteration / no fork crashes. Bump for real runs.
    """)
    return


if __name__ == "__main__":
    app.run()
