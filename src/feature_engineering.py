# xgb_iv.py
from __future__ import annotations
import os, sqlite3, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.metrics import mean_squared_error, r2_score

DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
# --- add near top of file ---
TRADING_MIN_PER_DAY = 390
ANNUAL_MINUTES = 252 * TRADING_MIN_PER_DAY

def _bs_delta(S, K, T, r, sigma, cp):
    if not np.isfinite([S, K, T, r, sigma]).all() or S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return norm.cdf(d1) if cp == "C" else norm.cdf(d1) - 1.0

def _bs_vega(S, K, T, r, sigma):
    if not np.isfinite([S, K, T, r, sigma]).all() or S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return S * norm.pdf(d1) * sqrtT

def _add_features(df: pd.DataFrame, peer_cols: list[str], target_ticker: str,
                  r: float = 0.045, target_mode: str = "iv_ret") -> pd.DataFrame:
    """
    Build time-safe feature block. target_mode in {"iv_ret", "iv"}.
    Returns a frame with column 'target' and only numeric features (+ symbol one-hots).
    """
    X = df.copy()

    # --- basic time context ---
    X["ts_event"] = pd.to_datetime(X["ts_event"], utc=True, errors="coerce")
    X["hour"] = X["ts_event"].dt.hour.astype("int16")
    X["minute"] = X["ts_event"].dt.minute.astype("int16")
    X["day_of_week"] = X["ts_event"].dt.dayofweek.astype("int16")
    X["minute_of_day"] = (X["hour"] * 60 + X["minute"]).astype("int16")
    X["sin_t"] = np.sin(2 * np.pi * X["minute_of_day"] / TRADING_MIN_PER_DAY)
    X["cos_t"] = np.cos(2 * np.pi * X["minute_of_day"] / TRADING_MIN_PER_DAY)

    # --- term / moneyness ---
    X["days_to_expiry"] = (X["time_to_expiry"] * 365.0).astype("float32")
    X["sqrt_T"] = np.sqrt(np.clip(X["time_to_expiry"].astype(float), 1e-8, None))
    X["inv_sqrt_T"] = 1.0 / X["sqrt_T"]
    # Guard against divide-by-zero
    with np.errstate(divide="ignore", invalid="ignore"):
        X["log_moneyness"] = np.log(X["stock_close"].astype(float) / X["strike_price"].astype(float))
    X["log_moneyness"].replace([np.inf, -np.inf], np.nan, inplace=True)
    X["abs_log_moneyness"] = X["log_moneyness"].abs()
    X["option_type_enc"] = _encode_option_type(X["option_type"])

    # --- target IV level and safe lags ---
    X["iv_clip"] = X["iv"].clip(lower=1e-6)
    X["iv_lag1"] = X["iv_clip"].shift(1)
    X["iv_ema5"] = X["iv_clip"].ewm(span=5, adjust=False).mean().shift(1)

    # --- equity price features ---
    if "stock_close" in X.columns:
        X["logS"] = np.log(X["stock_close"].astype(float))
        X["ret_1m"] = X["logS"].diff()
        X["ret_5m"] = X["logS"].diff(5)
        # realized vol (30m window), annualized, shifted to prevent leakage
        rv = X["ret_1m"].rolling(30).std()
        X["rv_30m"] = (rv * np.sqrt(ANNUAL_MINUTES / 30)).shift(1)
        # price z-score over 30m, shifted
        mu30 = X["stock_close"].rolling(30).mean()
        sd30 = X["stock_close"].rolling(30).std()
        X["zS_30m"] = ((X["stock_close"] - mu30) / (sd30 + 1e-9)).shift(1)

    # --- option flow features ---
    if "opt_volume" in X.columns:
        X["opt_vol_change_1m"] = X["opt_volume"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X["opt_vol_roll_15m"] = X["opt_volume"].rolling(15).mean().shift(1)
        mu60 = X["opt_volume"].rolling(60).mean()
        sd60 = X["opt_volume"].rolling(60).std()
        X["opt_vol_z_60m"] = ((X["opt_volume"] - mu60) / (sd60 + 1e-9)).shift(1)

    # --- Greeks (use current iv level; shift to be safe) ---
    def _safe_delta(row):
        return _bs_delta(row.get("stock_close", np.nan),
                         row.get("strike_price", np.nan),
                         row.get("time_to_expiry", np.nan),
                         r, row.get("iv_clip", np.nan),
                         str(row.get("option_type", "C"))[:1].upper())
    def _safe_vega(row):
        return _bs_vega(row.get("stock_close", np.nan),
                        row.get("strike_price", np.nan),
                        row.get("time_to_expiry", np.nan),
                        r, row.get("iv_clip", np.nan))

    X["delta"] = X.apply(_safe_delta, axis=1).shift(1)
    X["vega"]  = X.apply(_safe_vega, axis=1).shift(1)

    # --- peer IV aggregates (only peers, not target column) ---
    peer_cols = [c for c in peer_cols if c in X.columns]
    if peer_cols:
        peer_mean = X[peer_cols].mean(axis=1)
        X["peer_iv_mean"] = peer_mean
        X["peer_iv_std"]  = X[peer_cols].std(axis=1)
        X["iv_spread_to_peer"] = X["iv_clip"] - X["peer_iv_mean"]
        X["peer_mean_lag1"] = X["peer_iv_mean"].shift(1)
        X["peer_mean_ema5"] = peer_mean.ewm(span=5, adjust=False).mean().shift(1)

    # --- choose target ---
    if target_mode == "iv_ret":
        X["target"] = np.log(X["iv_clip"].shift(-1)) - np.log(X["iv_clip"])
    elif target_mode == "iv":
        X["target"] = X["iv_clip"]  # predict level; features are lagged so no trivial leakage
    else:
        raise ValueError("target_mode must be 'iv_ret' or 'iv'")

    # --- symbol one-hot (pooled usage) ---
    if "symbol" in X.columns:
        X = pd.get_dummies(X, columns=["symbol"], prefix="sym", dtype=float)

    # --- finalize numeric features ---
    drop_cols = ["ts_event", "expiry_date", "opt_symbol", "stock_symbol",
                 "iv", "iv_clip", "logS"]  # keep only engineered numerics
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    # Keep only numeric + target
    for col in X.columns:
        if col != "target":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.dropna(subset=["target"]).reset_index(drop=True)

    return X

# ---------- SQL helpers ----------
def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

def _read_sql(conn: sqlite3.Connection, q: str, params: tuple, parse_dates: List[str]) -> pd.DataFrame:
    return pd.read_sql_query(q, conn, params=params, parse_dates=parse_dates)

# ---------- BS & IV ----------
def _bs_price(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S-K,0.0) if cp=="C" else max(K-S,0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) if cp=="C" else K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def implied_vol(price: float, S: float, K: float, T: float, r: float, cp: str,
                lo: float = 1e-6, hi: float = 5.0) -> float:
    if not np.isfinite([price,S,K,T,r]).all() or price<=0 or S<=0 or K<=0 or T<=0:
        return np.nan
    intrinsic = max(S-K,0.0) if cp=="C" else max(K-S,0.0)
    if price <= intrinsic + 1e-10: return 1e-6
    f = lambda sig: _bs_price(S,K,T,r,sig,cp) - price
    try:
        return brentq(f, lo, hi, maxiter=100, xtol=1e-8)
    except Exception:
        return np.nan

def compute_iv_column(df: pd.DataFrame, r: float = 0.045) -> pd.DataFrame:
    df = df.copy()
    def _row(row):
        return implied_vol(float(row["opt_close"]), float(row["stock_close"]),
                           float(row["strike_price"]), max(float(row["time_to_expiry"]),1e-6),
                           r, str(row["option_type"])[0].upper())
    df["iv"] = df.apply(_row, axis=1)
    return df

# ---------- Load ATM rows ----------
def load_atm_from_sqlite(ticker: str,
                         start: pd.Timestamp | None = None,
                         end: pd.Timestamp | None = None,
                         db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        table = "atm_slices_1m" if _table_exists(conn, "atm_slices_1m") else None
        if table is None:
            # fallback to processed_merged_1m (caller can compute ATM later if desired)
            table = "processed_merged_1m" if _table_exists(conn, "processed_merged_1m") else "processed_merged"
            if not _table_exists(conn, table):
                raise RuntimeError(f"ATM/processed tables not found in {db_path}")
        clauses, params = ["ticker=?"], [ticker]
        if start is not None: clauses.append("ts_event >= ?"); params.append(start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        if end   is not None: clauses.append("ts_event <= ?"); params.append(end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        where = " AND ".join(clauses)
        q = f"""SELECT ts_event, expiry_date, opt_symbol, stock_symbol,
                       opt_close, stock_close, opt_volume, stock_volume,
                       option_type, strike_price, time_to_expiry, moneyness
                FROM {table}
                WHERE {where}
                ORDER BY ts_event"""
        return _read_sql(conn, q, tuple(params), ["ts_event","expiry_date"])

def _encode_option_type(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str[0].map({"P":0,"C":1}).astype("float32")

# ---------- Build datasets ----------
def _atm_core(ticker: str, start=None, end=None, r: float = 0.045,
              db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    df = load_atm_from_sqlite(ticker, start=start, end=end, db_path=db_path)
    df = compute_iv_column(df, r=r)
    keep = [
        "ts_event", "expiry_date", "iv",
        "opt_volume", "stock_close", "stock_volume",
        "time_to_expiry", "strike_price", "option_type"
    ]
    out = df[keep].copy()
    out["symbol"] = ticker
    out["ts_event"] = pd.to_datetime(out["ts_event"], utc=True, errors="coerce")
    return out.dropna(subset=["iv"]).sort_values("ts_event").reset_index(drop=True)

def build_iv_return_dataset_time_safe(
    tickers: List[str],
    start=None, end=None, r: float = 0.045,
    forward_steps: int = 1, tolerance: str = "2s",
    db_path: Path | str | None = None
) -> Dict[str, pd.DataFrame]:
    """Per-target datasets (no pooling)."""
    dbp = Path(db_path) if db_path else DEFAULT_DB_PATH
    cores = {t: _atm_core(t, start=start, end=end, r=r, db_path=dbp) for t in tickers}
    iv_wide = None
    for t, df in cores.items():
        tmp = df[["ts_event","iv"]].rename(columns={"iv": f"IV_{t}"}).sort_values("ts_event")
        iv_wide = tmp if iv_wide is None else pd.merge_asof(
            iv_wide, tmp, on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )
    out: Dict[str, pd.DataFrame] = {}
    for tgt, dft in cores.items():
        feats = dft.copy()
        feats["option_type_enc"] = _encode_option_type(feats["option_type"])
        feats["hour"] = feats["ts_event"].dt.hour.astype("int16")
        feats["minute"] = feats["ts_event"].dt.minute.astype("int16")
        feats["day_of_week"] = feats["ts_event"].dt.dayofweek.astype("int16")
        feats["days_to_expiry"] = (feats["time_to_expiry"] * 365.0).astype("float32")
        feats["iv_clip"] = feats["iv"].clip(lower=1e-6)
        feats["iv_ret_fwd"] = np.log(feats["iv_clip"].shift(-forward_steps)) - np.log(feats["iv_clip"])
        panel = pd.merge_asof(
            feats.sort_values("ts_event"), iv_wide.sort_values("ts_event"),
            on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )
        # inside the per-target loop after building `panel`:
        peer_cols = [c for c in panel.columns if c.startswith("IV_") and c != f"IV_{tgt}"]
        ds = _add_features(panel, peer_cols=peer_cols, target_ticker=tgt,
                   r=r, target_mode="iv_ret")   # or "iv"

        num_cols = ["opt_volume","time_to_expiry","strike_price",
                    "option_type_enc"] + peer_cols
        for c in num_cols:
            if c in panel.columns:
                panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(np.float32)
        out[tgt] = panel[["iv_ret_fwd"] + num_cols].dropna(subset=["iv_ret_fwd"]).reset_index(drop=True)
    return out

def build_pooled_iv_return_dataset_time_safe(
    tickers: List[str],
    start=None, end=None, r: float = 0.045,
    forward_steps: int = 1, tolerance: str = "2s",
    db_path: Path | str | None = None
) -> pd.DataFrame:
    """One pooled frame: target iv_ret_fwd + peers IVs + numeric features + symbol one-hot."""
    dbp = Path(db_path) if db_path else DEFAULT_DB_PATH
    cores = {t: _atm_core(t, start=start, end=end, r=r, db_path=dbp) for t in tickers}
    iv_wide = None
    for t, df in cores.items():
        tmp = df[["ts_event","iv"]].rename(columns={"iv": f"IV_{t}"}).sort_values("ts_event")
        iv_wide = tmp if iv_wide is None else pd.merge_asof(
            iv_wide, tmp, on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )
    frames = []
    for tgt, dft in cores.items():
        feats = dft.copy()
        feats["option_type_enc"] = _encode_option_type(feats["option_type"])

        # Add iv_clip if not present
        if "iv_clip" not in feats.columns:
            feats["iv_clip"] = feats["iv"].clip(lower=1e-6)
        
        feats["iv_ret_fwd"] = np.log(feats["iv_clip"].shift(-forward_steps)) - np.log(feats["iv_clip"])
        panel = pd.merge_asof(
            feats.sort_values("ts_event"), iv_wide.sort_values("ts_event"),
            on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )
        
        # Inside the per-target loop after building `panel`:
        # ... inside the per-target loop after you build `panel` ...
        peer_cols = [c for c in panel.columns if c.startswith("IV_") and c != f"IV_{tgt}"]

        cols = [
            "iv_ret_fwd",
            "iv_clip",                    # <— add this so evaluation can use it
            "opt_volume", "time_to_expiry", "strike_price",
            "option_type_enc"
        ] + peer_cols + ["symbol"]

        # strong numeric hygiene
        for c in ["opt_volume","time_to_expiry","days_to_expiry","strike_price"] + peer_cols:
            if c in panel.columns:
                panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(np.float32)

        frames.append(panel[cols].dropna(subset=["iv_ret_fwd"]).reset_index(drop=True))

        # after concatenating frames:
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not pooled.empty:
        pooled = pd.get_dummies(pooled, columns=["symbol"], prefix="sym", dtype=float)

        # ensure one-hot columns exist for ALL tickers you passed in
        for t in tickers:
            col = f"sym_{t}"
            if col not in pooled.columns:
                pooled[col] = 0.0

        # reorder (optional, but tidy)
        front = ["iv_ret_fwd", "iv_clip"]
        onehots = [f"sym_{t}" for t in tickers]
        other = [c for c in pooled.columns if c not in front + onehots]
        pooled = pooled[front + other + onehots]


    return pooled