"""
Cleaned feature engineering with duplicated/unnecessary code removed.

Removed functions now handled by data_loader_coordinator:
- Complex database loading logic (coordinator handles this)
- Individual ticker core loading (coordinator does this better)
- Core validation logic (moved to coordinator)

Kept essential functions:
- Feature engineering logic
- Dataset building functions that other modules depend on
- Core transformation functions
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# Config
DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
ANNUAL_MINUTES = 252 * 390

# What to hide when predicting each target (preserved from original)
HIDE_COLUMNS = {
    "iv_ret_fwd": ["iv_ret_fwd_abs"],
    "iv_ret_fwd_abs": ["iv_ret_fwd"], 
    "iv_clip": ["iv_ret_fwd", "iv_ret_fwd_abs"]
}

# Core features (preserved from original)
CORE_FEATURE_COLS = [
    "opt_volume", "time_to_expiry", "days_to_expiry", "strike_price",
    "option_type_enc", "delta", "gamma", "vega", "hour", "minute", "day_of_week"
]


# ------------------------------------------------------------
# Keep: Core feature engineering functions (other modules depend on these)
# ------------------------------------------------------------

def add_all_features(df: pd.DataFrame, forward_steps: int = 1, r: float = 0.045) -> pd.DataFrame:
    """Centralized feature engineering (preserves all original feature logic)."""
    df = df.copy()
    
    # Forward returns (preserves original single log transform approach)
    log_col = np.log(df["iv_clip"].astype(float))
    fwd = log_col.shift(-forward_steps) - log_col
    df["iv_ret_fwd"] = fwd
    df["iv_ret_fwd_abs"] = fwd.abs()
    
    # Vectorized Greeks (preserves original implementation)
    S = df["stock_close"].astype(float).to_numpy()
    K = df["strike_price"].astype(float).to_numpy()
    T = np.maximum(df["time_to_expiry"].astype(float).to_numpy(), 1e-9)
    sig = df["iv_clip"].astype(float).to_numpy()
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sqrtT)
    pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    is_call = df["option_type"].astype(str).str.upper().str[0].eq("C").to_numpy()
    df["delta"] = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
    df["gamma"] = pdf / (S * sig * sqrtT)
    df["vega"] = S * pdf * sqrtT
    
    # Time features (preserves original dtypes)
    df["hour"] = df["ts_event"].dt.hour.astype("int16")
    df["minute"] = df["ts_event"].dt.minute.astype("int16") 
    df["day_of_week"] = df["ts_event"].dt.dayofweek.astype("int16")
    df["days_to_expiry"] = (df["time_to_expiry"] * 365.0).astype("float32")
    df["option_type_enc"] = (df["option_type"].astype(str).str.upper().str[0]
                            .map({"P": 0, "C": 1}).astype("float32"))
    
    # Equity context (preserves original logic)
    if "stock_close" in df.columns:
        logS = np.log(df["stock_close"].astype(float))
        ret_1m = logS.diff()
        rv = ret_1m.rolling(30).std()
        df["rv_30m"] = (rv * np.sqrt(ANNUAL_MINUTES / 30)).shift(1)
    
    # Option flow (preserves original logic)
    if "opt_volume" in df.columns:
        pct_change = df["opt_volume"].pct_change()
        df["opt_vol_change_1m"] = (pct_change.replace([np.inf, -np.inf], np.nan).fillna(0.0))
        df["opt_vol_roll_15m"] = df["opt_volume"].rolling(15).mean().shift(1)
    
    return df


def build_iv_panel(cores: Dict[str, pd.DataFrame], tolerance: str = "2s") -> pd.DataFrame:
    """Centralized IV panel building (preserves original merge logic)."""
    tol = pd.Timedelta(tolerance)
    iv_wide = None
    
    for ticker, df in cores.items():
        if df is None or df.empty or not {"ts_event", "iv_clip"}.issubset(df.columns):
            continue
            
        tmp = df[["ts_event", "iv_clip"]].rename(columns={"iv_clip": f"IV_{ticker}"}).copy()
        tmp["ts_event"] = pd.to_datetime(tmp["ts_event"], utc=True, errors="coerce")
        tmp = tmp.dropna(subset=["ts_event", f"IV_{ticker}"]).sort_values("ts_event")
        tmp[f"IVRET_{ticker}"] = np.log(tmp[f"IV_{ticker}"]) - np.log(tmp[f"IV_{ticker}"].shift(1))
        tmp = tmp[["ts_event", f"IV_{ticker}", f"IVRET_{ticker}"]]
        
        if iv_wide is None:
            iv_wide = tmp
        else:
            iv_wide = pd.merge_asof(
                iv_wide.sort_values("ts_event"), tmp, on="ts_event",
                direction="backward", tolerance=tol
            )
    
    return iv_wide if iv_wide is not None else pd.DataFrame(columns=["ts_event"])


def finalize_dataset(df: pd.DataFrame, target_col: str, drop_symbol: bool = True) -> pd.DataFrame:
    """Centralized dataset finalization (preserves original cleanup logic)."""
    # Convert to numeric (preserves original approach)
    out = df.copy()
    for c in out.columns:
        if c != "ts_event":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    
    # Drop missing targets
    out = out.dropna(subset=[target_col])
    
    # Hide leaky columns (preserves original hide logic)
    for col in HIDE_COLUMNS.get(target_col, []):
        if col in out.columns:
            out = out.drop(columns=col)
    
    # Drop symbol if requested (for per-ticker datasets)
    if drop_symbol and "symbol" in out.columns:
        out = out.drop(columns=["symbol"])
        
    return out.reset_index(drop=True)


# ------------------------------------------------------------
# Keep: Main dataset building functions (required by other modules)
# ------------------------------------------------------------

def build_pooled_iv_return_dataset_time_safe(
    tickers: Sequence[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    r: float = 0.045,
    forward_steps: int = 1,
    tolerance: str = "2s",
    db_path: Path | str | None = None,
    cores: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Build pooled dataset for forecasting forward IV return."""
    
    # If cores not provided, need to load them (fallback for backward compatibility)
    if cores is None:
        from data_loader_coordinator import load_cores_with_auto_fetch
        cores = load_cores_with_auto_fetch(tickers, start, end, db_path)
    
    # Build panel
    panel = build_iv_panel(cores, tolerance=tolerance)
    
    frames = []
    for ticker in tickers:
        if ticker not in cores:
            continue
            
        # Add features
        feats = add_all_features(cores[ticker], forward_steps=forward_steps, r=r)
        
        # Merge with panel
        feats = pd.merge_asof(
            feats.sort_values("ts_event"), panel.sort_values("ts_event"),
            on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )
        
        # Finalize (keeps symbol column for pooled analysis)
        clean = finalize_dataset(feats, "iv_ret_fwd", drop_symbol=False)
        frames.append(clean)
    
    if not frames:
        return pd.DataFrame()
        
    pooled = pd.concat(frames, ignore_index=True)
    if pooled.empty:
        return pooled
    
    # One-hot encode symbol (preserves original logic)
    pooled = pd.get_dummies(pooled, columns=["symbol"], prefix="sym", dtype=float)
    
    # Ensure all ticker columns exist
    for ticker in tickers:
        col = f"sym_{ticker}"
        if col not in pooled.columns:
            pooled[col] = 0.0
    
    # Column ordering (preserves original order)
    front = ["iv_ret_fwd"]
    if "iv_ret_fwd_abs" in pooled.columns:
        front.append("iv_ret_fwd_abs")
    if "iv_clip" in pooled.columns:
        front.append("iv_clip")
    onehots = [f"sym_{t}" for t in tickers]
    other = [c for c in pooled.columns if c not in front + onehots]
    
    return pooled[front + other + onehots]


def build_iv_return_dataset_time_safe(
    tickers: Sequence[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    r: float = 0.045,
    forward_steps: int = 15,
    tolerance: str = "2s",
    db_path: Path | str | None = None,
    cores: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build per‑ticker datasets for forecasting forward IV return."""
    
    # If cores not provided, need to load them (fallback for backward compatibility)
    if cores is None:
        from data_loader_coordinator import load_cores_with_auto_fetch
        cores = load_cores_with_auto_fetch(tickers, start, end, db_path)
    
    # Build panel
    panel = build_iv_panel(cores, tolerance=tolerance)
    
    datasets = {}
    for ticker in tickers:
        if ticker not in cores:
            continue
            
        # Add features
        feats = add_all_features(cores[ticker], forward_steps=forward_steps, r=r)
        
        # Merge with panel
        feats = pd.merge_asof(
            feats.sort_values("ts_event"), panel.sort_values("ts_event"),
            on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )
        
        # Finalize (removes symbol column for per-ticker analysis)
        datasets[ticker] = finalize_dataset(feats, "iv_ret_fwd", drop_symbol=True)
    
    return datasets


def build_target_peer_dataset(
    target: str,
    tickers: Sequence[str],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    r: float = 0.045,
    forward_steps: int = 1,
    tolerance: str = "2s",
    db_path: Path | str | None = None,
    target_kind: str = "iv_ret",
    cores: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Build dataset for single target vs peers."""
    
    if target not in tickers:
        raise AssertionError("target must be included in tickers")
    
    # If cores not provided, need to load them (fallback for backward compatibility)
    if cores is None:
        from data_loader_coordinator import load_cores_with_auto_fetch
        cores = load_cores_with_auto_fetch(tickers, start, end, db_path)
    
    if target not in cores:
        raise ValueError(f"Target {target} produced no valid core")
    
    # Build panel and add features
    panel = build_iv_panel(cores, tolerance=tolerance)
    feats = add_all_features(cores[target], forward_steps=forward_steps, r=r)
    
    # Merge with panel
    feats = pd.merge_asof(
        feats.sort_values("ts_event"), panel.sort_values("ts_event"),
        on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
    )
    
    # Set target variable (preserves original target_kind logic)
    if target_kind == "iv_ret":
        feats["y"] = feats["iv_ret_fwd"]
        hide_key = "iv_ret_fwd"
    elif target_kind == "iv":
        feats["y"] = feats["iv_clip"]
        hide_key = "iv_clip"
    else:
        raise ValueError("target_kind must be 'iv_ret' or 'iv'")
    
    # Finalize
    return finalize_dataset(feats, "y", drop_symbol=True)





def load_ticker_core(ticker: str, start=None, end=None, r=0.045, db_path=DEFAULT_DB_PATH) -> pd.DataFrame:
    """Centralized ticker data loading with IV calculation."""
    
    with sqlite3.connect(str(db_path)) as conn:
        # Table selection logic (preserves original)
        table = "atm_slices_1m"
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            table = "processed_merged_1m" 
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                table = "processed_merged"
        
        where_clauses, params = ["ticker=?"], [ticker]
        if start:
            where_clauses.append("ts_event >= ?")
            params.append(pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        if end:
            where_clauses.append("ts_event <= ?")
            params.append(pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            
        query = f"""
        SELECT ts_event, expiry_date, opt_symbol, stock_symbol,
               opt_close, stock_close, opt_volume, stock_volume,
               option_type, strike_price, time_to_expiry, moneyness
        FROM {table} WHERE {' AND '.join(where_clauses)}
        ORDER BY ts_event
        """
        
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["ts_event", "expiry_date"])
    
    if df.empty:
        return df
        
    # IV calculation (preserves original BS implementation)
    df["iv"] = df.apply(lambda row: _calculate_iv(
        row["opt_close"], row["stock_close"], row["strike_price"], 
        max(row["time_to_expiry"], 1e-6), row["option_type"], r
    ), axis=1)
    
    # Core cleanup (preserves original column selection and processing)
    keep = ["ts_event", "expiry_date", "iv", "opt_volume", "stock_close", 
            "stock_volume", "time_to_expiry", "strike_price", "option_type"]
    df = df[keep].copy()
    df["symbol"] = ticker
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df.dropna(subset=["iv"]).sort_values("ts_event").reset_index(drop=True)
    df["iv_clip"] = df["iv"].clip(lower=1e-6)
    
    return df



def _calculate_iv(price: float, S: float, K: float, T: float, cp: str, r: float) -> float:
    """Centralized IV calculation (preserves original BS logic)."""
    if not np.isfinite([price, S, K, T, r]).all() or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    
    intrinsic = max(S - K, 0.0) if cp.upper().startswith('C') else max(K - S, 0.0)
    if price <= intrinsic + 1e-10:
        return 1e-6
        
    def bs_price(sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return intrinsic
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        if cp.upper().startswith('C'):
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    try:
        return brentq(lambda sig: bs_price(sig) - price, 1e-6, 5.0, maxiter=100, xtol=1e-8)
    except:
        return np.nan

# Export original names for backward compatibility
__all__ = [
    "build_pooled_iv_return_dataset_time_safe",
    "build_iv_return_dataset_time_safe", 
    "build_target_peer_dataset",
    "add_all_features",
    "build_iv_panel",
    "finalize_dataset"
]
