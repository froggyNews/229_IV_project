# xgb_iv_transfer.py
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.optimize import brentq
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

DB_PATH = Path("data/iv_data.db")

# -------------------------------------------------
# SQLite loader
# -------------------------------------------------
def load_processed_from_sqlite(
    ticker: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Load processed_merged rows for a ticker from SQLite."""
    with sqlite3.connect(db_path) as conn:
        clauses = ["ticker = ?"]
        params = [ticker]
        if start is not None:
            clauses.append("ts_event >= ?")
            params.append(start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        if end is not None:
            clauses.append("ts_event <= ?")
            params.append(end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        where = " AND ".join(clauses)

        q = f"""
        SELECT ts_event, expiry_date, opt_symbol, stock_symbol,
               opt_close, stock_close, opt_volume, stock_volume,
               option_type, strike_price, time_to_expiry, moneyness
        FROM processed_merged
        WHERE {where}
        ORDER BY ts_event
        """
        df = pd.read_sql_query(
            q, conn, params=params, parse_dates=["ts_event", "expiry_date"]
        )
    return df

# -------------------------------------------------
# Time helpers
# -------------------------------------------------
def _ensure_datetime_utc(df: pd.DataFrame, col: str = "ts_event") -> pd.DataFrame:
    if col in df.columns:
        is_dt = is_datetime64_any_dtype(df[col]) or is_datetime64tz_dtype(df[col])
        if not is_dt:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df

def _encode_option_type(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str[0].map({"P": 0, "C": 1}).astype("float32")

# -------------------------------------------------
# Black–Scholes IV
# -------------------------------------------------
def _bs_price(S, K, T, r, sigma, cp):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    d1
