import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def trim_unnecessary_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop obviously non-model features and add simple time encodings.
    Designed to take rows read from processed_merged (SQLite).
    """
    df = data.copy()

    # Always coerce to tz-aware UTC
    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df["hour"] = df["ts_event"].dt.hour
        df["minute"] = df["ts_event"].dt.minute
        df["day_of_week"] = df["ts_event"].dt.dayofweek

    # Encode option_type as 0=P, 1=C
    if "option_type" in df.columns:
        df["option_type_encoded"] = df["option_type"].astype("category").cat.codes

    drop_cols = [
        "Unnamed: 0.1", "Unnamed: 0",
        "rtype_x", "publisher_id_x", "instrument_id_x",
        "rtype_y", "publisher_id_y", "instrument_id_y",
        "open_x", "high_x", "low_x", "opt_symbol",
        "open_y", "high_y", "low_y", "stock_symbol",
        "expiry_date", "option_type", "ts_event",
        "stock_close", "opt_close",
    ]
    return df.drop(columns=drop_cols, errors="ignore")


# -------------------------------------------------
# Black–Scholes & IV inversion
# -------------------------------------------------
def _bs_price(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> float:
    """European Black–Scholes price (C=call, P=put)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if cp == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price: float, S: float, K: float, T: float, r: float, cp: str,
                lo: float = 1e-6, hi: float = 5.0) -> float:
    """Invert BS to get sigma via brentq. Returns np.nan if it fails."""
    if not np.all(np.isfinite([price, S, K, T, r])):
        return np.nan
    if S <= 0 or K <= 0 or T <= 0 or price <= 0:
        return np.nan

    intrinsic = max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    if price <= intrinsic + 1e-10:
        return 1e-6  # clamp to tiny vol

    def f(sig):
        return _bs_price(S, K, T, r, sig, cp) - price

    try:
        return brentq(f, lo, hi, maxiter=100, xtol=1e-8)
    except Exception:
        return np.nan


# -------------------------------------------------
# IV computation on processed_merged rows
# -------------------------------------------------
def compute_implied_vol(data: pd.DataFrame, r: float = 0.045) -> pd.DataFrame:
    """
    Adds 'iv' column to rows (expects columns from processed_merged: 
    ['opt_close','stock_close','strike_price','time_to_expiry','option_type']).
    """
    required = ["opt_close", "stock_close", "strike_price", "time_to_expiry", "option_type"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = data.copy()
    for c in ["opt_close", "stock_close", "strike_price", "time_to_expiry"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["option_type"] = df["option_type"].astype(str).str.upper().str[0].replace({"CALL": "C", "PUT": "P"})

    def _row_iv(row):
        return implied_vol(
            price=row["opt_close"],
            S=row["stock_close"],
            K=row["strike_price"],
            T=max(row["time_to_expiry"], 1e-6),
            r=r,
            cp=row["option_type"]
        )

    df["iv"] = df.apply(_row_iv, axis=1)
    df["iv_ok"] = np.isfinite(df["iv"]) & (df["iv"] > 0) & (df["iv"] < 5.0)
    return df


# -------------------------------------------------
# Feature builder
# -------------------------------------------------
def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a lean feature frame for modeling.
    Keeps 'iv' as target candidate and adds simple engineered features.
    """
    df = data.copy()

    if "strike_price" in df.columns and "stock_close" in df.columns:
        df["moneyness"] = df["stock_close"] - df["strike_price"]
        df["abs_moneyness"] = df["moneyness"].abs()
        with np.errstate(divide="ignore", invalid="ignore"):
            df["strike_to_spot"] = df["strike_price"] / df["stock_close"]
            df["log_moneyness"] = np.log(df["stock_close"] / df["strike_price"])

    df = trim_unnecessary_columns(df)

    keep_extra = [c for c in [
        "time_to_expiry", "moneyness", "abs_moneyness",
        "strike_to_spot", "log_moneyness",
        "option_type_encoded", "hour", "minute", "day_of_week", "iv"
    ] if c in df.columns]

    df = df[[c for c in df.columns if c not in keep_extra] + keep_extra]

    essential = [c for c in ["time_to_expiry", "strike_to_spot", "log_moneyness", "iv"] if c in df.columns]
    if essential:
        df = df.dropna(subset=essential)

    return df


# -------------------------------------------------
# Pipeline entry
# -------------------------------------------------
def pre_prediction_full(data: pd.DataFrame, r: float = 0.045) -> pd.DataFrame:
    """
    1) Compute implied vol for each row.
    2) Build engineered features.
    Returns a tidy DataFrame with an 'iv' column usable as target or input.
    """
    with_iv = compute_implied_vol(data, r=r)
    features = create_features(with_iv)
    return features
