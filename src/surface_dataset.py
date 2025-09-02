from __future__ import annotations

"""Build supervised datasets where features are tensors from IV surfaces.

Each example corresponds to (ts_event, ticker). Features are a flattened
(K x T) grid aggregated from all option quotes for that ticker/timestamp.
Targets can reuse existing ATM-based labels (e.g., iv_ret_fwd) for continuity.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sqlite3

import numpy as np
import pandas as pd

from data_loader_coordinator import _calculate_iv as _calc_iv  # reuse IV calc
from data_loader_coordinator import load_cores_with_auto_fetch


def _infer_timeframe_from_db(db_path: Path) -> str:
    s = str(db_path).lower()
    return "1m" if ("1m" in s or s.endswith("_1m.db")) else "1h"


def _read_processed_rows(db_path: Path, ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    tf = _infer_timeframe_from_db(db_path)
    table = f"processed_merged_{tf}"
    with sqlite3.connect(str(db_path)) as conn:
        where = ["ticker = ?"]
        params: List[object] = [ticker]
        if start:
            where.append("ts_event >= ?")
            params.append(pd.Timestamp(start, tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        if end:
            where.append("ts_event <= ?")
            params.append(pd.Timestamp(end, tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        q = (
            f"SELECT ts_event, opt_close, stock_close, opt_volume, stock_volume, "
            f"       expiry_date, option_type, strike_price, time_to_expiry, moneyness "
            f"FROM {table} WHERE {' AND '.join(where)} ORDER BY ts_event"
        )
        try:
            df = pd.read_sql(q, conn, params=params, parse_dates=["ts_event", "expiry_date"])
        except Exception:
            return pd.DataFrame()
    # Compute IV per quote
    if not df.empty:
        df["iv"] = df.apply(
            lambda r: _calc_iv(
                r.get("opt_close", np.nan),
                r.get("stock_close", np.nan),
                r.get("strike_price", np.nan),
                max(float(r.get("time_to_expiry", np.nan)) if pd.notna(r.get("time_to_expiry", np.nan)) else np.nan, 1e-6),
                str(r.get("option_type", "C")),
                0.045,
            ),
            axis=1,
        )
        df["iv_clip"] = pd.to_numeric(df["iv"], errors="coerce").clip(lower=1e-6)
        df = df.dropna(subset=["ts_event", "iv_clip", "strike_price", "time_to_expiry", "stock_close"]).copy()
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df = df.sort_values("ts_event")
    return df


def _grid_edges(series: pd.Series, n_bins: int, low_q: float = 0.05, high_q: float = 0.95, fallback: Tuple[float,float] = (-0.1, 0.1)) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        lo, hi = fallback
    else:
        lo = float(np.nanpercentile(s, low_q*100.0))
        hi = float(np.nanpercentile(s, high_q*100.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = fallback
    return np.linspace(lo, hi, int(n_bins) + 1)


def _flatten_surface_for_timestamp(df: pd.DataFrame, ts: pd.Timestamp, m_edges: np.ndarray, t_edges: np.ndarray, k_bins: int, t_bins: int, agg: str) -> np.ndarray:
    sub = df[df["ts_event"] == ts]
    if sub.empty:
        return np.full(k_bins * t_bins, np.nan)
    K = pd.to_numeric(sub["strike_price"], errors="coerce")
    S = pd.to_numeric(sub["stock_close"], errors="coerce")
    m = (K / S) - 1.0
    tau = pd.to_numeric(sub["time_to_expiry"], errors="coerce")
    iv = pd.to_numeric(sub["iv_clip"], errors="coerce")
    valid = m.notna() & tau.notna() & iv.notna()
    if not valid.any():
        return np.full(k_bins * t_bins, np.nan)
    kb = pd.cut(m[valid], bins=m_edges, labels=False, include_lowest=True)
    tb = pd.cut(tau[valid], bins=t_edges, labels=False, include_lowest=True)
    grid = pd.DataFrame({"K": kb, "T": tb, "iv": iv[valid]}).dropna()
    if grid.empty:
        return np.full(k_bins * t_bins, np.nan)
    if agg == "mean":
        piv = grid.pivot_table(index="T", columns="K", values="iv", aggfunc="mean")
    else:
        piv = grid.pivot_table(index="T", columns="K", values="iv", aggfunc="median")
    piv = piv.reindex(index=range(t_bins), columns=range(k_bins))
    return piv.values.flatten(order="C").astype(float)


def build_surface_tensor_dataset(
    tickers: Sequence[str],
    start: str | None,
    end: str | None,
    db_path: Path | str,
    k_bins: int = 10,
    t_bins: int = 10,
    agg: str = "median",
    forward_steps: int = 1,
    tolerance: str = "30s",
) -> pd.DataFrame:
    """Return pooled dataset with surface-tensor features and ATM-based targets.

    - Features: columns K{i}_T{j} flattened, built per (ticker, ts_event).
    - Target: ATM-based iv_ret_fwd computed from cores, merged by asof.
    """
    dbp = Path(db_path)
    # Read processed rows and compute IVs
    per_ticker_rows: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = _read_processed_rows(dbp, t, start, end)
        if df is None or df.empty:
            continue
        per_ticker_rows[t] = df
    if not per_ticker_rows:
        return pd.DataFrame()

    # Build global grid edges
    all_m = []
    all_tau = []
    for df in per_ticker_rows.values():
        K = pd.to_numeric(df["strike_price"], errors="coerce")
        S = pd.to_numeric(df["stock_close"], errors="coerce")
        all_m.append(((K / S) - 1.0).replace([np.inf, -np.inf], np.nan))
        all_tau.append(pd.to_numeric(df["time_to_expiry"], errors="coerce"))
    m_edges = _grid_edges(pd.concat(all_m), k_bins)
    t_edges = _grid_edges(pd.concat(all_tau), t_bins, low_q=0.01, high_q=0.99, fallback=(1/365.0, 1.5))

    # Build per-ticker feature frames
    frames: List[pd.DataFrame] = []
    for t, df in per_ticker_rows.items():
        ts_list = df["ts_event"].dropna().drop_duplicates().sort_values()
        rows: List[Dict[str, object]] = []
        for ts in ts_list:
            vec = _flatten_surface_for_timestamp(df, ts, m_edges, t_edges, k_bins, t_bins, agg)
            row = {f"K{i}_T{j}": float(vec[j * k_bins + i]) for j in range(t_bins) for i in range(k_bins)}
            row["ts_event"] = pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize("UTC")
            row["symbol"] = t
            rows.append(row)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    feats = pd.concat(frames, ignore_index=True)

    # Compute ATM-based target using existing loader (ensures consistency)
    cores = load_cores_with_auto_fetch(list(tickers), start, end, dbp)
    label_rows: List[pd.DataFrame] = []
    for t in tickers:
        core = cores.get(t)
        if core is None or core.empty or "iv_clip" not in core.columns:
            continue
        s = pd.DataFrame({
            "ts_event": pd.to_datetime(core["ts_event"], utc=True, errors="coerce"),
            "symbol": t,
            "iv_clip": pd.to_numeric(core["iv_clip"], errors="coerce"),
        }).dropna()
        s = s.sort_values("ts_event").reset_index(drop=True)
        log_iv = np.log(s["iv_clip"].astype(float))
        s["iv_ret_fwd"] = log_iv.shift(-forward_steps) - log_iv
        label_rows.append(s[["ts_event", "symbol", "iv_ret_fwd"]])
    labels = pd.concat(label_rows, ignore_index=True) if label_rows else pd.DataFrame()
    if labels.empty:
        return feats

    # Merge labels to features per symbol by asof
    out_frames: List[pd.DataFrame] = []
    tol = pd.Timedelta(tolerance)
    for t in tickers:
        f = feats[feats["symbol"] == t].sort_values("ts_event")
        l = labels[labels["symbol"] == t].sort_values("ts_event")
        if f.empty or l.empty:
            continue
        # Merge both iv_ret_fwd and iv_clip if present in label rows
        merged = pd.merge_asof(f, l.sort_values("ts_event"), on="ts_event", direction="backward", tolerance=tol)
        out_frames.append(merged)
    out = pd.concat(out_frames, ignore_index=True) if out_frames else feats
    return out
