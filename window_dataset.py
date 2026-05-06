"""Slice per-cell trajectories into fixed-length sliding windows.

Each input row belongs to a `uid`. Within a uid, rows are ordered by `frame`
(falls back to `timestep`/`time`) and chopped into windows of `--length` with
step `--stride`. Output adds two columns:

  window_id  -- "<uid>__w<k>", unique per window
  window_pos -- 0..length-1 position inside the window

Trailing windows shorter than `length` are dropped unless `--keep-partial`.
"""

import argparse

import numpy as np
import pandas as pd


def _order_col(df: pd.DataFrame) -> str:
    for c in ("frame", "timestep", "time"):
        if c in df.columns:
            return c
    raise ValueError("need one of frame/timestep/time to order trajectories")


def window_trajectories(
    df: pd.DataFrame,
    length: int,
    stride: int,
    keep_partial: bool = False,
) -> pd.DataFrame:
    if length <= 0 or stride <= 0:
        raise ValueError("length and stride must be positive")

    order = _order_col(df)
    df = df.sort_values(["uid", order], kind="stable").reset_index(drop=True)

    pieces = []
    for uid, g in df.groupby("uid", sort=False):
        n = len(g)
        if n < length and not keep_partial:
            continue
        starts = range(0, max(1, n - length + 1) if not keep_partial else n, stride)
        for k, s in enumerate(starts):
            e = min(s + length, n)
            if not keep_partial and e - s < length:
                break
            w = g.iloc[s:e].copy()
            w["window_id"] = f"{uid}__w{k}"
            w["window_pos"] = np.arange(e - s, dtype=np.int32)
            pieces.append(w)

    if not pieces:
        return df.iloc[0:0].assign(window_id=pd.Series(dtype=object),
                                   window_pos=pd.Series(dtype=np.int32))
    return pd.concat(pieces, ignore_index=True)


def summarize(out: pd.DataFrame, length: int) -> None:
    if out.empty:
        print("no windows produced")
        return
    n_win = out["window_id"].nunique()
    sizes = out.groupby("window_id").size()
    print(f"windows: {n_win:,}   rows: {len(out):,}")
    print(f"window length: min={sizes.min()}  max={sizes.max()}  target={length}")
    print(f"unique uids represented: {out['uid'].nunique():,}")
    if "ramp_pattern_name" in out.columns:
        per_pat = out.groupby("ramp_pattern_name")["window_id"].nunique()
        print("windows per pattern:")
        print(per_pat.to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input parquet (must have a uid column)")
    ap.add_argument("--output", required=True, help="output parquet")
    ap.add_argument("--length", type=int, required=True, help="window length (timesteps)")
    ap.add_argument("--stride", type=int, required=True, help="stride between window starts")
    ap.add_argument("--keep-partial", action="store_true",
                    help="keep trailing windows shorter than length (default: drop)")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if "uid" not in df.columns:
        raise SystemExit("input must contain a 'uid' column")

    out = window_trajectories(df, length=args.length, stride=args.stride,
                              keep_partial=args.keep_partial)
    out.to_parquet(args.output, index=False)
    print(f"wrote {args.output}")
    summarize(out, args.length)


if __name__ == "__main__":
    main()
