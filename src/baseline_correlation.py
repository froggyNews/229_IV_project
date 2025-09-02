from __future__ import annotations

"""Baseline historical correlation calculations for 1h ATM slices."""

from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import numpy as np

from feature_engineering import build_iv_panel, DEFAULT_DB_PATH
from surface_correlation import build_surface_feature_matrix
from surface_dataset import build_surface_tensor_dataset
from data_loader_coordinator import load_cores_with_auto_fetch


def compute_baseline_correlations(
    tickers: Sequence[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path | None = None,
    cores: Dict[str, pd.DataFrame] | None = None,
    tolerance: str = "15s",
    surface_mode: str = "atm",
    include_surface: bool = True,
    k_bins: int = 10,
    t_bins: int = 10,
    surface_agg: str = "median",
    include_surface_returns: bool = True,
    forward_steps: int = 1,
    surface_return_method: str = "diff",  # 'diff' | 'log' | 'pct'
) -> dict:
    """Compute historical correlation matrices for IV level and IV returns.

    Returns a dict with keys: 'clip', 'iv_returns', 'surface', 'surface_returns'.
    """
    if cores is None:
        db = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        atm_only = (str(surface_mode).lower() != "full")
        cores = load_cores_with_auto_fetch(list(tickers), start, end, db, atm_only=atm_only)

    panel = build_iv_panel(cores, tolerance=tolerance, agg=surface_agg)
    if panel is None or panel.empty:
        # Surface correlation can still be attempted if cores present
        surface_corr = pd.DataFrame()
        surface_ret_corr = pd.DataFrame()
        if include_surface and cores:
            try:
                feat = build_surface_feature_matrix(
                    cores=cores,
                    tickers=list(tickers),
                    k_bins=k_bins,
                    t_bins=t_bins,
                    agg=surface_agg,
                )
                if not feat.empty and len(feat.index) >= 2:
                    surface_corr = feat.T.corr(min_periods=1)
            except Exception:
                surface_corr = pd.DataFrame()
        if include_surface_returns and cores:
            try:
                dbp = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
                surf_df = build_surface_tensor_dataset(
                    tickers=list(tickers),
                    start=start,
                    end=end,
                    db_path=dbp,
                    k_bins=k_bins,
                    t_bins=t_bins,
                    agg=surface_agg,
                    forward_steps=forward_steps,
                    tolerance=tolerance,
                )
                if not surf_df.empty:
                    feat_cols = [c for c in surf_df.columns if c.startswith("K") and "_T" in c]
                    by_t: dict[str, pd.Series] = {}
                    for t in tickers:
                        s = surf_df[surf_df.get("symbol") == t].sort_values("ts_event")
                        if s.empty:
                            continue
                        surf_mean = s[feat_cols].mean(axis=1, skipna=True)
                        sr = (np.log(surf_mean) - np.log(surf_mean.shift(1))).dropna()
                        by_t[t] = sr
                    if by_t:
                        aligned = pd.concat(by_t, axis=1, join="inner")
                        if aligned.shape[1] >= 2 and aligned.shape[0] >= 2:
                            surface_ret_corr = aligned.corr()
            except Exception:
                surface_ret_corr = pd.DataFrame()
        return {"clip": pd.DataFrame(), "iv_returns": pd.DataFrame(), "surface": surface_corr, "surface_returns": surface_ret_corr}

    def _fast_corr(df: pd.DataFrame) -> pd.DataFrame:
        # Try a fast path using numpy.corrcoef on rows with complete data.
        if df is None or df.empty or df.shape[1] < 2:
            return pd.DataFrame()
        df32 = df.astype(np.float32)
        no_na = df32.dropna()
        if no_na.shape[0] >= 2 and no_na.shape[1] >= 2:
            try:
                C = np.corrcoef(no_na.values, rowvar=False)
                return pd.DataFrame(C, index=no_na.columns, columns=no_na.columns)
            except Exception:
                pass
        # Fallback to pandas pairwise-complete correlation
        try:
            return df.corr()
        except Exception:
            return pd.DataFrame()

    iv_cols = [f"IV_{t}" for t in tickers if f"IV_{t}" in panel.columns]
    ret_cols = [f"IVRET_{t}" for t in tickers if f"IVRET_{t}" in panel.columns]
    if iv_cols:
        print(f"Computing correlations for IV columns: {iv_cols}. sample rows:\n{panel[iv_cols].head()}")
    if ret_cols:
        print(f"Computing correlations for IV return columns: {ret_cols}. sample rows:\n{panel[ret_cols].head()}")

    if len(iv_cols) >= 2:
        corr_iv = _fast_corr(panel[iv_cols])
        clip_corr = corr_iv.rename(index=lambda x: x[3:], columns=lambda x: x[3:])
    else:
        clip_corr = pd.DataFrame()

    if len(ret_cols) >= 2:
        corr_ret = _fast_corr(panel[ret_cols])
        ivret_corr = corr_ret.rename(index=lambda x: x[6:], columns=lambda x: x[6:])
    else:
        ivret_corr = pd.DataFrame()

    # Surface-based correlation across tickers (flattened KxT features)
    surface_corr = pd.DataFrame()
    if include_surface and len(tickers) >= 2:
        print(f"Building surface feature matrix for {len(tickers)} tickers (k_bins={k_bins}, t_bins={t_bins}, agg={surface_agg})...")
        try:
            feat = build_surface_feature_matrix(
                cores=cores,
                tickers=list(tickers),
                k_bins=k_bins,
                t_bins=t_bins,
                agg=surface_agg,
            )
            if not feat.empty and len(feat.index) >= 2:
                surface_corr = feat.T.corr(min_periods=1)
        except Exception:
            surface_corr = pd.DataFrame()

    # Surface-based returns correlation (mean across grid -> configurable diff)
    surface_ret_corr = pd.DataFrame()
    if include_surface_returns and len(tickers) >= 2:
        print(f"Building surface tensor dataset for returns (k_bins={k_bins}, t_bins={t_bins}, agg={surface_agg})...")
        try:
            dbp = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
            surf_df = build_surface_tensor_dataset(
                tickers=list(tickers),
                start=start,
                end=end,
                db_path=dbp,
                k_bins=k_bins,
                t_bins=t_bins,
                agg=surface_agg,
                forward_steps=forward_steps,
                tolerance=tolerance,
            )
            if not surf_df.empty:
                feat_cols = [c for c in surf_df.columns if c.startswith("K") and "_T" in c]
                by_t: dict[str, pd.Series] = {}
                for t in tickers:
                    s = surf_df[surf_df.get("symbol") == t].sort_values("ts_event")
                    if s.empty:
                        continue
                    surf_mean = s[feat_cols].mean(axis=1, skipna=True)
                    method = str(surface_return_method).lower()
                    if method in ("diff", "difference", "delta"):
                        sr = surf_mean.diff().dropna()
                    elif method in ("pct", "percent", "percentage"):
                        with np.errstate(divide='ignore', invalid='ignore'):
                            sr = (surf_mean / surf_mean.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
                    else:
                        sr = (np.log(surf_mean) - np.log(surf_mean.shift(1))).dropna()
                    # Align to minute grid to increase overlap across tickers
                    try:
                        idx = pd.to_datetime(s.loc[sr.index, "ts_event"], utc=True, errors="coerce")
                    except Exception:
                        idx = pd.to_datetime(s.loc[sr.index, "ts_event"], errors="coerce")
                    sr.index = idx
                    sr = sr.groupby(sr.index.floor("1min")).mean()
                    by_t[t] = sr
                if by_t:
                    aligned = pd.concat(by_t, axis=1, join="inner")
                    # Require at least 2 overlapping timestamps across at least 2 tickers
                    if aligned.shape[1] >= 2 and aligned.shape[0] >= 2:
                        surface_ret_corr = aligned.corr(min_periods=1)
        except Exception:
            surface_ret_corr = pd.DataFrame()

    return {"clip": clip_corr, "iv_returns": ivret_corr, "surface": surface_corr, "surface_returns": surface_ret_corr}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute baseline historical correlations (levels, returns, surfaces)."
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--start", type=str, default=None, help="Start timestamp")
    parser.add_argument("--end", type=str, default=None, help="End timestamp")
    parser.add_argument(
        "--db-path", type=str, default=None, help="Path to SQLite DB (defaults to project DB)"
    )
    parser.add_argument("--tolerance", type=str, default="15s", help="Merge tolerance")
    parser.add_argument("--no-surface", action="store_true", help="Disable surface correlation")
    parser.add_argument("--k-bins", type=int, default=10, help="Moneyness grid bins for surfaces")
    parser.add_argument("--t-bins", type=int, default=10, help="Maturity grid bins for surfaces")
    parser.add_argument("--surface-agg", choices=["median", "mean"], default="median", help="Aggregation in grid")
    parser.add_argument("--no-surface-returns", action="store_true", help="Disable surface-returns correlation")
    parser.add_argument("--forward-steps", type=int, default=1, help="Forward steps for surface returns")
    args = parser.parse_args()

    result = compute_baseline_correlations(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        db_path=args.db_path,
        tolerance=args.tolerance,
        include_surface=not args.no_surface,
        k_bins=args.k_bins,
        t_bins=args.t_bins,
        surface_agg=args.surface_agg,
        include_surface_returns=not args.no_surface_returns,
        forward_steps=args.forward_steps,
    )
    if not result["clip"].empty:
        print("IV clip correlation matrix:")
        print(result["clip"])
    else:
        print("IV clip correlation matrix: <not enough data>")
    if not result["iv_returns"].empty:
        print("IV returns correlation matrix:")
        print(result["iv_returns"])
    else:
        print("IV returns correlation matrix: <not enough data>")
    if not args.no_surface:
        if not result.get("surface", pd.DataFrame()).empty:
            print("IV surface correlation matrix:")
            print(result["surface"])
        else:
            print("IV surface correlation matrix: <not enough data>")
    if not args.no_surface_returns:
        if not result.get("surface_returns", pd.DataFrame()).empty:
            print("IV surface return correlation matrix:")
            print(result["surface_returns"])
        else:
            print("IV surface return correlation matrix: <not enough data>")

