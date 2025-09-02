from __future__ import annotations

"""Correlation weights based on flattened IV surface features.

Builds a common (K,T) grid across tickers from core option data, aggregates
IV levels into that grid, flattens each ticker's surface to a feature vector,
and computes ticker-to-ticker correlations. Provides a utility to turn the
correlation column for a given target into non-negative, normalized weights.

Typical use:
    cores = load_cores_with_auto_fetch(tickers, start, end, db_path)
    feats = build_surface_feature_matrix(cores, tickers)
    w = corr_weights_from_matrix(feats, target="QBTS", peers=["IONQ","RGTI","QUBT"]) 
"""

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from feature_engineering import DEFAULT_DB_PATH
from data_loader_coordinator import load_cores_with_auto_fetch


def _compute_moneyness(df: pd.DataFrame) -> pd.Series:
    """Compute relative moneyness (K/S - 1) robustly from available columns."""
    K = pd.to_numeric(df.get("strike_price"), errors="coerce")
    S = pd.to_numeric(df.get("stock_close"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (K / S) - 1.0
    return m.replace([np.inf, -np.inf], np.nan)


def _grid_edges(values: pd.Series, n_bins: int, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Choose grid edges for binning via robust clipping and quantiles."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        # fallback symmetric range
        v = pd.Series(np.linspace(-0.1, 0.1, 100))
    lo = float(np.nanpercentile(v, 5)) if vmin is None else vmin
    hi = float(np.nanpercentile(v, 95)) if vmax is None else vmax
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = (-0.1, 0.1)
    return np.linspace(lo, hi, int(n_bins) + 1)


def build_surface_feature_matrix(
    cores: Dict[str, pd.DataFrame],
    tickers: Sequence[str],
    k_bins: int = 10,
    t_bins: int = 10,
    agg: str = "median",
) -> pd.DataFrame:
    """Build flattened IV-surface feature matrix (rows=tickers, cols=grid cells).

    For each ticker, bins moneyness (K/S-1) and time_to_expiry into a common
    grid across tickers, aggregates `iv_clip` within each cell (median by
    default), then flattens the grid to a vector.
    """
    # Collect global edges to ensure a shared grid
    m_all, t_all = [], []
    for t in tickers:
        df = cores.get(t)
        if df is None or df.empty:
            continue
        m_all.append(_compute_moneyness(df))
        t_all.append(pd.to_numeric(df.get("time_to_expiry"), errors="coerce"))
    m_all = pd.concat(m_all).dropna() if m_all else pd.Series(dtype=float)
    t_all = pd.concat(t_all).dropna() if t_all else pd.Series(dtype=float)

    m_edges = _grid_edges(m_all, k_bins)
    t_edges = _grid_edges(t_all, t_bins, vmin=max(1/365.0, float(t_all.min())) if len(t_all) else 1/365.0,
                          vmax=float(t_all.quantile(0.95)) if len(t_all) else 1.0)

    # Aggregate per ticker
    cols: List[str] = [f"K{i}_T{j}" for j in range(t_bins) for i in range(k_bins)]
    feat_rows: Dict[str, List[float]] = {}

    for t in tickers:
        df = cores.get(t)
        if df is None or df.empty:
            feat_rows[t] = [np.nan] * (k_bins * t_bins)
            continue
        iv = pd.to_numeric(df.get("iv_clip", df.get("iv")), errors="coerce")
        m = _compute_moneyness(df)
        tau = pd.to_numeric(df.get("time_to_expiry"), errors="coerce")
        valid = iv.notna() & m.notna() & tau.notna()
        if not valid.any():
            feat_rows[t] = [np.nan] * (k_bins * t_bins)
            continue
        m_b = pd.cut(m[valid], bins=m_edges, labels=False, include_lowest=True)
        t_b = pd.cut(tau[valid], bins=t_edges, labels=False, include_lowest=True)
        grid = (
            pd.DataFrame({"K": m_b, "T": t_b, "iv": iv[valid]})
            .dropna(subset=["K", "T", "iv"])
            .astype({"K": int, "T": int})
        )
        if grid.empty:
            feat_rows[t] = [np.nan] * (k_bins * t_bins)
            continue
        if agg == "mean":
            piv = grid.pivot_table(index="T", columns="K", values="iv", aggfunc="mean")
        else:
            piv = grid.pivot_table(index="T", columns="K", values="iv", aggfunc="median")
        # Reindex to full grid
        piv = piv.reindex(index=range(t_bins), columns=range(k_bins))
        vec = piv.values.flatten(order="C").astype(float)
        feat_rows[t] = vec.tolist()

    feature_df = pd.DataFrame.from_dict(feat_rows, orient="index", columns=cols)

    # Impute missing by column medians, then by global median, finally zeros
    if not feature_df.empty:
        col_meds = feature_df.median(axis=0, skipna=True)
        feature_df = feature_df.apply(lambda s: s.fillna(col_meds[s.name]))
        if feature_df.isna().any().any():
            feature_df = feature_df.fillna(feature_df.stack().median())
        feature_df = feature_df.fillna(0.0)

    return feature_df


def corr_weights_from_matrix(
    feature_df: pd.DataFrame,
    target: str,
    peers: Sequence[str],
    clip_negative: bool = True,
    power: float = 1.0,
) -> pd.Series:
    """Turn cross-ticker correlations into non-negative, normalized weights.

    - Computes corr = feature_df.T.corr(min_periods=1)
    - Takes the column corresponding to `target`, filtered to `peers`.
    - Optionally clips negatives to 0 and applies a power transform.
    - Normalizes to sum to 1; if all zero/NaN, returns equal weights.
    """
    if feature_df is None or feature_df.empty:
        return pd.Series({p: 1.0 / len(peers) for p in peers})
    corr = feature_df.T.corr(min_periods=1)
    col = corr[target] if target in corr.columns else pd.Series(index=peers, dtype=float)
    w = col.reindex(peers)
    if clip_negative:
        w = w.clip(lower=0.0)
    if power is not None and power != 1.0:
        with np.errstate(invalid="ignore"):
            w = np.power(w, float(power))
    w = w.fillna(0.0)
    s = float(w.sum())
    if s <= 0:
        # fallback: equal weights
        return pd.Series({p: 1.0 / max(1, len(peers)) for p in peers})
    return w / s


def build_features_and_weights(
    tickers: Sequence[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path | None = None,
    k_bins: int = 10,
    t_bins: int = 10,
    agg: str = "median",
    target: str | None = None,
    clip_negative: bool = True,
    power: float = 1.0,
    surface_mode: str = "full",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: load cores, build surface features, and (optionally) weights.

    Returns feature_df and (if target set) a one-row DataFrame of weights for target.
    """
    db = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    atm_only = (str(surface_mode).lower() != "full")
    cores = load_cores_with_auto_fetch(list(tickers), start, end, db, atm_only=atm_only)
    feats = build_surface_feature_matrix(cores, tickers, k_bins=k_bins, t_bins=t_bins, agg=agg)
    if target is None:
        return feats, pd.DataFrame()
    peers = [t for t in tickers if t != target]
    w = corr_weights_from_matrix(feats, target=target, peers=peers, clip_negative=clip_negative, power=power)
    return feats, pd.DataFrame([w], index=[target])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Surface-feature correlations and weights")
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--db", type=str, default=None)
    ap.add_argument("--k-bins", type=int, default=10)
    ap.add_argument("--t-bins", type=int, default=10)
    ap.add_argument("--agg", choices=["median", "mean"], default="median")
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--clip-negative", action="store_true")
    ap.add_argument("--power", type=float, default=1.0)
    args = ap.parse_args()

    feats, weights = build_features_and_weights(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        db_path=args.db,
        k_bins=args.k_bins,
        t_bins=args.t_bins,
        agg=args.agg,
        target=args.target,
        clip_negative=True,
        power=1.0,
    )
    print("Feature matrix (rows=tickers, cols=KxT cells):")
    print(feats.head())
    if args.target:
        print(f"\nWeights for target {args.target}:")
        print(weights)
