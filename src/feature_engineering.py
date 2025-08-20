# feature_engineering.py  —  compact, time-safe, IV/IVRET-aware
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from greeks import bs_delta, bs_gamma, bs_vega

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
TRADING_MIN_PER_DAY = 390
ANNUAL_MINUTES = 252 * TRADING_MIN_PER_DAY

__all__ = [
    "build_pooled_iv_return_dataset_time_safe",
    "build_iv_return_dataset_time_safe",
    "build_target_peer_dataset",
]

# ------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------
def _valid_core(df: pd.DataFrame) -> bool:
    """Core must be a non-empty DataFrame with required columns."""
    # Check if df is None first
    if df is None:
        return False
    
    return isinstance(df, pd.DataFrame) and not df.empty and {"ts_event", "iv_clip"}.issubset(df.columns)
# ------------------------------------------------------------
# DB I/O + raw IV
# ------------------------------------------------------------
def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    q = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(q, (name,)).fetchone() is not None


def _read_sql(conn: sqlite3.Connection, q: str, params: tuple, parse_dates: List[str]) -> pd.DataFrame:
    return pd.read_sql_query(q, conn, params=params, parse_dates=parse_dates)


def load_atm_from_sqlite(
    ticker: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Load near-ATM rows (or fallback processed rows) for one ticker."""
    with sqlite3.connect(str(db_path)) as conn:
        table = "atm_slices_1m" if _table_exists(conn, "atm_slices_1m") else None
        if table is None:
            table = "processed_merged_1m" if _table_exists(conn, "processed_merged_1m") else "processed_merged"
            if not _table_exists(conn, table):
                raise RuntimeError(f"ATM/processed tables not found in {db_path}")

        clauses, params = ["ticker=?"], [ticker]
        if start is not None:
            if isinstance(start, str):
                start_dt = pd.to_datetime(start)
            else:
                start_dt = start
            clauses.append("ts_event >= ?")
            params.append(start_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        if end is not None:
            if isinstance(end, str):
                end_dt = pd.to_datetime(end)
            else:
                end_dt = end
            clauses.append("ts_event <= ?")
            params.append(end_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))

        where = " AND ".join(clauses)
        q = f"""
        SELECT ts_event, expiry_date, opt_symbol, stock_symbol,
               opt_close, stock_close, opt_volume, stock_volume,
               option_type, strike_price, time_to_expiry, moneyness
        FROM {table}
        WHERE {where}
        ORDER BY ts_event
        """
        df = _read_sql(conn, q, tuple(params), ["ts_event", "expiry_date"])
    return df


# ---- IV from price (kept minimal; you already have a full BS stack elsewhere) ----
from scipy.stats import norm
from scipy.optimize import brentq

def _bs_price(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if cp == "C" else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(price: float, S: float, K: float, T: float, r: float, cp: str,
                lo: float = 1e-6, hi: float = 5.0) -> float:
    if not np.isfinite([price, S, K, T, r]).all() or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    intrinsic = max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    if price <= intrinsic + 1e-10:
        return 1e-6
    f = lambda sig: _bs_price(S, K, T, r, sig, cp) - price
    try:
        return brentq(f, lo, hi, maxiter=100, xtol=1e-8)
    except Exception:
        return np.nan

def compute_iv_column(df: pd.DataFrame, r: float = 0.045) -> pd.DataFrame:
    df = df.copy()
    def _row(row):
        return implied_vol(
            float(row["opt_close"]),
            float(row["stock_close"]),
            float(row["strike_price"]),
            max(float(row["time_to_expiry"]), 1e-6),
            r,
            str(row["option_type"])[0].upper(),
        )
    df["iv"] = df.apply(_row, axis=1)
    return df


# ------------------------------------------------------------
# Core transforms (single source of truth)
# ------------------------------------------------------------
def _encode_option_type(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str[0].map({"P": 0, "C": 1}).astype("float32")


def _atm_core(ticker: str, *, start=None, end=None, r: float = 0.045, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Minimal, typed, time-sorted core for one ticker with a clean 'iv'."""
    df = load_atm_from_sqlite(ticker, start=start, end=end, db_path=db_path)
    df = compute_iv_column(df, r=r)

    keep = [
        "ts_event", "expiry_date", "iv",
        "opt_volume", "stock_close", "stock_volume",
        "time_to_expiry", "strike_price", "option_type",
    ]
    out = df[keep].copy()
    out["symbol"] = ticker
    out["ts_event"] = pd.to_datetime(out["ts_event"], utc=True, errors="coerce")
    out = out.dropna(subset=["iv"]).sort_values("ts_event").reset_index(drop=True)
    out["iv_clip"] = out["iv"].clip(lower=1e-6)
    S = out["stock_close"].astype(float).to_numpy()
    K = out["strike_price"].astype(float).to_numpy()
    T = np.maximum(out["time_to_expiry"].astype(float).to_numpy(), 1e-9)
    sig = out["iv_clip"].astype(float).to_numpy()
    sqrtT = np.sqrt(T)

    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrtT)
    pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    is_call = out["option_type"].astype(str).str.upper().str[0].eq("C").to_numpy()

    out["delta"] = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    out["gamma"] = pdf / (S * sig * sqrtT)
    out["vega"] = S * pdf * sqrtT

    return out
def _build_iv_panel(cores: Dict[str, pd.DataFrame], tolerance: str) -> pd.DataFrame:
    """
    Build a wide, time-safe panel of IV and IVRET for all tickers:
        ts_event, IV_<T1>.., IVRET_<T1>.., IV_<Tn>.., IVRET_<Tn>..
    Skips any invalid/empty cores gracefully.
    """
    tol = pd.Timedelta(tolerance)
    iv_wide: pd.DataFrame | None = None

    for t, df in cores.items():
        if not _valid_core(df):
            print(f"[iv_panel] skip {t}: core is None/empty or missing required cols")
            continue

        # ensure proper types and sorting for merge_asof
        tmp = (
            df[["ts_event", "iv_clip"]]
            .rename(columns={"iv_clip": f"IV_{t}"})
            .copy()
        )
        tmp["ts_event"] = pd.to_datetime(tmp["ts_event"], utc=True, errors="coerce")
        tmp = tmp.dropna(subset=["ts_event", f"IV_{t}"]).sort_values("ts_event")

        # IVRET uses each ticker's own history
        tmp[f"IVRET_{t}"] = np.log(tmp[f"IV_{t}"]) - np.log(tmp[f"IV_{t}"].shift(1))
        tmp = tmp[["ts_event", f"IV_{t}", f"IVRET_{t}"]]

        iv_wide = tmp if iv_wide is None else pd.merge_asof(
            iv_wide.sort_values("ts_event"),
            tmp,
            on="ts_event",
            direction="backward",
            tolerance=tol,
        )

    return iv_wide if iv_wide is not None else pd.DataFrame(columns=["ts_event"])


def _add_simple_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight controls that are stable and non-leaky."""
    X = df.copy()
    X["hour"] = X["ts_event"].dt.hour.astype("int16")
    X["minute"] = X["ts_event"].dt.minute.astype("int16")
    X["day_of_week"] = X["ts_event"].dt.dayofweek.astype("int16")
    X["days_to_expiry"] = (X["time_to_expiry"] * 365.0).astype("float32")
    X["option_type_enc"] = _encode_option_type(X["option_type"])

    # equity context (safe)
    if "stock_close" in X.columns:
        X["logS"] = np.log(X["stock_close"].astype(float))
        X["ret_1m"] = X["logS"].diff()
        rv = X["ret_1m"].rolling(30).std()
        X["rv_30m"] = (rv * np.sqrt(ANNUAL_MINUTES / 30)).shift(1)
        X.drop(columns=["logS"], inplace=True)

    # option flow (safe)
    if "opt_volume" in X.columns:
        X["opt_vol_change_1m"] = (
            X["opt_volume"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        X["opt_vol_roll_15m"] = X["opt_volume"].rolling(15).mean().shift(1)

    return X


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ------------------------------------------------------------
# Public datasets
# ------------------------------------------------------------
def build_iv_return_dataset_time_safe(
    tickers: List[str],
    start=None, end=None, r: float = 0.045,
    forward_steps: int = 15, tolerance: str = "2s",
    db_path: Path | str | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Per-target dict dataset (no pooling). Each value has:
      - labels: iv_ret_fwd and iv_ret_fwd_abs
      - features: IV_SELF, IVRET_SELF, IV_<peer>*, IVRET_<peer>* + simple controls
    """
    dbp = Path(db_path) if db_path else DEFAULT_DB_PATH
    cores_raw = {t: _atm_core(t, start=start, end=end, r=r, db_path=dbp) for t in tickers}
    cores = {t: df for t, df in cores_raw.items() if _valid_core(df)}
    print(cores)
    dropped = None
    if dropped:
        print(f"[iv_dataset] dropped invalid cores: {sorted(dropped)}")

    panel = _build_iv_panel(cores, tolerance=tolerance)

    out: Dict[str, pd.DataFrame] = {}
    for tgt, base in cores.items():
        feats = base.copy()
        feats["iv_ret_fwd"] = np.log(feats["iv_clip"].shift(-forward_steps)) - np.log(feats["iv_clip"])
        feats["iv_ret_fwd_abs"] = feats["iv_ret_fwd"].abs()
        feats = pd.merge_asof(
            feats.sort_values("ts_event"), panel.sort_values("ts_event"),
            on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )

        # rename own columns, identify peers
        if f"IV_{tgt}" in feats.columns:
            feats = feats.rename(columns={f"IV_{tgt}": "IV_SELF"})
        if f"IVRET_{tgt}" in feats.columns:
            feats = feats.rename(columns={f"IVRET_{tgt}": "IVRET_SELF"})
        peer_cols = [c for c in feats.columns if c.startswith("IV_") and c != "IV_SELF"]
        peer_ret_cols = [c for c in feats.columns if c.startswith("IVRET_") and c != "IVRET_SELF"]

        feats = _add_simple_controls(feats)

        cols = (
            ["iv_ret_fwd", "iv_ret_fwd_abs", "IV_SELF", "IVRET_SELF"]
            + peer_cols + peer_ret_cols
            + [
                "opt_volume",
                "time_to_expiry",
                "days_to_expiry",
                "strike_price",
                "option_type_enc",
                "delta",
                "gamma",
                "vega",
                "hour",
                "minute",
                "day_of_week",
            ]
        )
        sub = _numeric(feats[[c for c in cols if c in feats.columns]]).dropna(subset=["iv_ret_fwd"])
        out[tgt] = sub.reset_index(drop=True)

    return out


def build_pooled_iv_return_dataset_time_safe(
    tickers: List[str],
    start=None, end=None, r: float = 0.045,
    forward_steps: int = 1, tolerance: str = "2s",
    db_path: Path | str | None = None,
    cores: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    One pooled frame suitable for XGB eval:
      - columns: iv_ret_fwd, iv_ret_fwd_abs, iv_clip, IV_<ticker>*, IVRET_<ticker>*,
                 controls, and sym_* one-hots
      - time-safe merges; no leakage
    """
    dbp = Path(db_path) if db_path else DEFAULT_DB_PATH
    if cores is None:
        dbp = Path(db_path) if db_path else DEFAULT_DB_PATH
        cores_raw = {t: _atm_core(t, start=start, end=end, r=r, db_path=dbp) for t in tickers}
        cores = {t: df for t, df in cores_raw.items() if _valid_core(df)}
        dropped = set(cores_raw) - set(cores)
        if dropped:
            print(f"[pooled] dropped invalid cores: {sorted(dropped)}")

    dropped = None
    if dropped:
        print(f"[pooled] dropped invalid cores: {sorted(dropped)}")

    panel = _build_iv_panel(cores, tolerance=tolerance)

    frames = []
    for tgt, base in cores.items():   # <-- iterate sanitized cores
        feats = base.copy()
        feats["iv_ret_fwd"] = np.log(feats["iv_clip"].shift(-forward_steps)) - np.log(feats["iv_clip"])
        feats["iv_ret_fwd_abs"] = feats["iv_ret_fwd"].abs()
        feats = pd.merge_asof(
            feats.sort_values("ts_event"), panel.sort_values("ts_event"),
            on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )
        feats["symbol"] = tgt
        feats = _add_simple_controls(feats)

        cols = [
            "iv_ret_fwd", "iv_ret_fwd_abs", "iv_clip",
            "opt_volume", "time_to_expiry", "days_to_expiry", "strike_price",
            "option_type_enc", "delta", "gamma", "vega", "hour", "minute", "day_of_week",
        ] + [c for c in feats.columns if c.startswith("IV_") or c.startswith("IVRET_")] + ["symbol"]

        frames.append(_numeric(feats[cols]).dropna(subset=["iv_ret_fwd"]))

    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if pooled.empty:
        return pooled

    # symbol one-hots with guaranteed columns for all input tickers
    pooled = pd.get_dummies(pooled, columns=["symbol"], prefix="sym", dtype=float)
    for t in tickers:
        col = f"sym_{t}"
        if col not in pooled.columns:
            pooled[col] = 0.0

    # tidy ordering
    front = ["iv_ret_fwd", "iv_ret_fwd_abs", "iv_clip"]
    onehots = [f"sym_{t}" for t in tickers]
    other = [c for c in pooled.columns if c not in front + onehots]
    return pooled[front + other + onehots]

def build_target_peer_dataset(
    target: str,
    tickers: List[str],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    r: float = 0.045,
    forward_steps: int = 1,
    tolerance: str = "2s",
    db_path: Path | str | None = None,
    target_kind: str = "iv_ret",
    cores: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    assert target in tickers, "target must be included in tickers"

    if cores is None:
        dbp = Path(db_path) if db_path else DEFAULT_DB_PATH
        cores_raw = {t: _atm_core(t, start=start, end=end, r=r, db_path=dbp) for t in tickers}
        cores = {t: df for t, df in cores_raw.items() if _valid_core(df)}

    if target not in cores:
        raise ValueError(f"Target {target} produced no valid core (None/empty/missing cols)")

    assert target in tickers, "target must be included in tickers"
    dbp = Path(db_path) if db_path else DEFAULT_DB_PATH

    panel = _build_iv_panel(cores, tolerance=tolerance)
    feats = cores[target].copy()
    feats = pd.merge_asof(
        feats.sort_values("ts_event"),
        panel.sort_values("ts_event"),
        on="ts_event",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )

    # rename self columns
    rename_map = {}
    if f"IV_{target}" in feats.columns:
        rename_map[f"IV_{target}"] = "IV_SELF"
    if f"IVRET_{target}" in feats.columns:
        rename_map[f"IVRET_{target}"] = "IVRET_SELF"
    if rename_map:
        feats = feats.rename(columns=rename_map)

    # compute label
    if target_kind == "iv_ret":
        y = np.log(feats["iv_clip"].shift(-forward_steps)) - np.log(feats["iv_clip"])
    elif target_kind == "iv":
        y = feats["iv_clip"]
    else:
        raise ValueError("target_kind must be 'iv_ret' or 'iv'")

    # add non-leaky controls
    feats = _add_simple_controls(feats)

    # strictly exclude any target-named columns and any stray 'target' column
    feats = feats.drop(
        columns=["target", f"IV_{target}", f"IVRET_{target}"],
        errors="ignore",
    )

    # peers = all IV_/IVRET_ except the self aliases
    peer_cols = [
        c for c in feats.columns
        if c.startswith("IV_") and c not in {"IV_SELF"}
    ]
    peer_ret_cols = [
        c for c in feats.columns
        if c.startswith("IVRET_") and c not in {"IVRET_SELF"}
    ]

    final_cols = (
        ["y", "IV_SELF", "IVRET_SELF"]
        + peer_cols + peer_ret_cols
        + [
            "opt_volume",
            "time_to_expiry",
            "days_to_expiry",
            "strike_price",
            "option_type_enc",
            "delta",
            "gamma",
            "vega",
            "hour",
            "minute",
            "day_of_week",
            "iv_clip",
        ]
    )

    feats["y"] = y
    out = _numeric(feats[[c for c in final_cols if c in feats.columns]]).dropna(subset=["y"])
    return out.reset_index(drop=True)
