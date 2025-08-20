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


def _hagan_implied_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    """Approximate Black implied volatility under the SABR model."""
    if F <= 0 or K <= 0 or T <= 0:
        return np.nan

    if np.isclose(F, K):
        term1 = alpha / (F ** (1 - beta))
        term2 = 1 + (
            ((1 - beta) ** 2 / 24) * (alpha ** 2 / (F ** (2 - 2 * beta)))
            + (rho * beta * nu * alpha / (4 * F ** (1 - beta)))
            + ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
        ) * T
        return term1 * term2

    FK_beta = (F * K) ** ((1 - beta) / 2)
    logFK = np.log(F / K)
    z = (nu / alpha) * FK_beta * logFK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    if np.isclose(z, 0):
        x_z = 1  # limit z/x_z -> 1 as z -> 0
    term1 = alpha / (FK_beta * (1 + ((1 - beta) ** 2 / 24) * (logFK ** 2) + ((1 - beta) ** 4 / 1920) * (logFK ** 4)))
    term2 = z / x_z
    term3 = 1 + (
        ((1 - beta) ** 2 / 24) * (alpha ** 2 / (FK_beta ** 2))
        + (rho * beta * nu * alpha / (4 * FK_beta))
        + ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
    ) * T
    return term1 * term2 * term3


def _solve_sabr_alpha(sigma: float, F: float, K: float, T: float, beta: float, rho: float, nu: float) -> float:
    """Calibrate alpha for a single observation using Hagan's formula."""
    if np.any(np.isnan([sigma, F, K, T])) or sigma <= 0:
        return np.nan

    def objective(a: float) -> float:
        return _hagan_implied_vol(F, K, T, a, beta, rho, nu) - sigma

    try:
        return brentq(objective, 1e-6, 5.0, maxiter=100)
    except ValueError:
        return np.nan


def _add_sabr_features(df: pd.DataFrame, beta: float = 0.5) -> pd.DataFrame:
    """Compute simple SABR parameter features and drop raw price/IV columns."""
    F_series = df.get("stock_close")
    K_series = df.get("strike_price")
    T_series = df.get("time_to_expiry")
    sigma_series = df.get("iv_clip")
    if F_series is None or K_series is None or T_series is None or sigma_series is None:
        return df

    F = F_series.astype(float).to_numpy()
    K = K_series.astype(float).to_numpy()
    T = np.maximum(T_series.astype(float).to_numpy(), 1e-9)
    sigma = sigma_series.astype(float).to_numpy()

    # Heuristic estimates for rho and nu
    moneyness = (K / F) - 1.0
    rho = np.tanh(moneyness * 5.0)
    nu_series = (
        df["iv_clip"].astype(float).rolling(30).std() * np.sqrt(ANNUAL_MINUTES / 30)
    ).shift(1)
    nu = nu_series.to_numpy()

    alpha = np.array([
        _solve_sabr_alpha(sig, f, k, t, beta, r, n)
        for sig, f, k, t, r, n in zip(sigma, F, K, T, rho, nu)
    ])

    df["sabr_alpha"] = alpha
    df["sabr_beta"] = beta
    df["sabr_rho"] = rho
    df["sabr_nu"] = nu

    # Remove raw stock price and IV information to hide them
    df = df.drop(columns=[c for c in ["stock_close", "iv_clip"] if c in df.columns])
    return df

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
    # df["hour"] = df["ts_event"].dt.hour.astype("int16")
    # df["minute"] = df["ts_event"].dt.minute.astype("int16") 
    # df["day_of_week"] = df["ts_event"].dt.dayofweek.astype("int16")
    # df["days_to_expiry"] = (df["time_to_expiry"] * 365.0).astype("float32")
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

    # SABR parameters and hide raw price/IV data
    df = _add_sabr_features(df)

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

    # Remove raw stock and IV information entirely
    leak_cols = [c for c in out.columns if c in {"stock_close", "iv_clip"} or c.startswith("IV_") or c.startswith("IVRET_")]
    if leak_cols:
        out = out.drop(columns=leak_cols)
    
    # Drop symbol if requested (for per-ticker datasets)
    if drop_symbol and "symbol" in out.columns:
        out = out.drop(columns=["symbol"])
    
    out = _normalize_numeric_features(out, target_col=target_col, exclude_prefixes=("sym_",), keep_cols=("ts_event",))

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

# add near the imports
from pandas.api.types import is_numeric_dtype

# --- NEW ---
def _normalize_numeric_features(
    df: pd.DataFrame,
    target_col: str,
    exclude_prefixes: Sequence[str] = ("sym_",),
    keep_cols: Sequence[str] = ("ts_event",),
) -> pd.DataFrame:
    """
    Z-score normalize numeric feature columns:
      x_norm = (x - mean) / std
    Skips the target column, time index, and any columns starting with prefixes in exclude_prefixes.
    """
    out = df.copy()
    # pick numeric feature columns
    num_cols: list[str] = []
    for c in out.columns:
        if c == target_col or c in keep_cols:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if is_numeric_dtype(out[c]):
            num_cols.append(c)

    if num_cols:
        means = out[num_cols].mean(axis=0)
        stds = out[num_cols].std(axis=0).replace(0.0, 1.0).fillna(1.0)
        out[num_cols] = (out[num_cols] - means) / stds
        # optional: keep stats for inspection
        out.attrs["norm_means"] = {c: float(means[c]) for c in num_cols}
        out.attrs["norm_stds"]  = {c: float(stds[c])  for c in num_cols}
    return out


# Export original names for backward compatibility
__all__ = [
    "build_pooled_iv_return_dataset_time_safe",
    "build_iv_return_dataset_time_safe", 
    "build_target_peer_dataset",
    "add_all_features",
    "build_iv_panel",
    "finalize_dataset"
]


# ------------------------------------------------------------
# REMOVED FUNCTIONS (now handled by data_loader_coordinator):
# ------------------------------------------------------------
# - load_atm_from_sqlite() -> coordinator handles DB access
# - _atm_core() -> coordinator handles core loading
# - load_ticker_core() -> coordinator handles this better
# - _valid_core() -> coordinator validates cores
# - _table_exists() -> coordinator handles DB logic
# - _read_sql() -> coordinator handles DB logic  
# - compute_iv_column() -> coordinator handles IV calculation
# - implied_vol() -> coordinator handles IV calculation
# - _bs_price() -> coordinator handles IV calculation
# - _encode_option_type() -> moved inline to add_all_features
# - _compute_forward_returns() -> moved inline to add_all_features
# - _merge_panel() -> simplified inline in main functions
# - _add_simple_controls() -> moved inline to add_all_features
# - _numeric() -> moved inline to finalize_dataset
# - build_base_iv_dataset() -> replaced by coordinator + individual functions