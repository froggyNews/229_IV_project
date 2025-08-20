"""
Data loader coordinator - helps orchestrate data loading across existing modules.

This module doesn't duplicate functionality but provides a clean interface
to coordinate between feature_engineering.py, fetch_data_sqlite.py, and 
train_peer_effects.py.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, Sequence, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# Import existing functions
from fetch_data_sqlite import fetch_and_save


def _calculate_iv(price: float, S: float, K: float, T: float, cp: str, r: float) -> float:
    """IV calculation (moved here to avoid circular imports)."""
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


def load_ticker_core(ticker: str, start=None, end=None, r=0.045, db_path=None) -> pd.DataFrame:
    """Load ticker core data with IV calculation."""
    
    if db_path is None:
        db_path = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
    
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
        
    # IV calculation
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


class DataCoordinator:
    """Coordinates data loading and ensures consistency across modules."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
        self.api_key = os.getenv("DATABENTO_API_KEY")
        
    def load_cores_with_fetch(
        self, 
        tickers: Sequence[str], 
        start: str, 
        end: str,
        auto_fetch: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load ticker cores, automatically fetching missing data if possible.
        """
        cores = {}
        missing_tickers = []
        
        print(f"Loading cores for {len(tickers)} tickers...")
        
        # First pass: try to load existing data
        for ticker in tickers:
            try:
                # Call the module-level function
                core = load_ticker_core(ticker, start=start, end=end, db_path=self.db_path)
                if not core.empty:
                    cores[ticker] = core
                    print(f"  ✓ {ticker}: {len(core):,} rows")
                else:
                    missing_tickers.append(ticker)
                    print(f"  ✗ {ticker}: no data found")
            except Exception as e:
                print(f"  ✗ {ticker}: error loading ({e})")
                missing_tickers.append(ticker)
        
        # Second pass: fetch missing data if enabled and API key available
        if missing_tickers and auto_fetch and self.api_key:
            print(f"Auto-fetching {len(missing_tickers)} missing tickers...")
            
            start_ts = pd.Timestamp(start, tz="UTC")
            end_ts = pd.Timestamp(end, tz="UTC")
            
            for ticker in missing_tickers:
                try:
                    print(f"  Fetching {ticker}...")
                    fetch_and_save(self.api_key, ticker, start_ts, end_ts, self.db_path, force=False)
                    
                    # Retry loading after fetch - call module-level function
                    core = load_ticker_core(ticker, start=start, end=end, db_path=self.db_path)
                    if not core.empty:
                        cores[ticker] = core
                        print(f"    ✓ Fetched {ticker}: {len(core):,} rows")
                    else:
                        print(f"    ✗ {ticker}: no data even after fetch")
                        
                except Exception as e:
                    print(f"    ✗ {ticker}: fetch failed ({e})")
                    continue
                    
        elif missing_tickers and not self.api_key:
            print("Warning: Missing tickers but no DATABENTO_API_KEY for auto-fetch")
            
        print(f"Final result: {len(cores)}/{len(tickers)} tickers loaded")
        return cores
    
    def validate_cores_for_analysis(
        self, 
        cores: Dict[str, pd.DataFrame], 
        analysis_type: str = "general"
    ) -> Dict[str, pd.DataFrame]:
        """Validate cores are suitable for specific analysis types."""
        valid_cores = {}
        
        for ticker, core in cores.items():
            # Basic validation
            if core is None or core.empty:
                print(f"Skipping {ticker}: empty core")
                continue
                
            if not {"ts_event", "iv_clip"}.issubset(core.columns):
                print(f"Skipping {ticker}: missing required columns")
                continue
                
            # Analysis-specific validation
            if analysis_type == "peer_effects":
                if len(core) < 100:
                    print(f"Skipping {ticker}: insufficient data for peer effects ({len(core)} rows)")
                    continue
                    
            elif analysis_type == "pooled":
                if len(core) < 50:
                    print(f"Skipping {ticker}: insufficient data for pooling ({len(core)} rows)")
                    continue
            
            valid_cores[ticker] = core
            
        return valid_cores
    
    def get_analysis_summary(self, cores: Dict[str, pd.DataFrame]) -> Dict:
        """Get summary statistics for loaded cores."""
        if not cores:
            return {"status": "no_data"}
            
        summary = {
            "n_tickers": len(cores),
            "tickers": list(cores.keys()),
            "total_rows": sum(len(df) for df in cores.values()),
            "date_ranges": {},
            "avg_rows_per_ticker": sum(len(df) for df in cores.values()) // len(cores)
        }
        
        # Get date ranges for each ticker
        for ticker, core in cores.items():
            if not core.empty and "ts_event" in core.columns:
                dates = pd.to_datetime(core["ts_event"])
                summary["date_ranges"][ticker] = {
                    "start": dates.min().strftime("%Y-%m-%d"),
                    "end": dates.max().strftime("%Y-%m-%d"),
                    "rows": len(core)
                }
        
        return summary


# Convenience functions for backward compatibility
def load_cores_with_auto_fetch(
    tickers: Sequence[str], 
    start: str, 
    end: str, 
    db_path: Optional[Path] = None,
    auto_fetch: bool = True
) -> Dict[str, pd.DataFrame]:
    """Convenience function that wraps DataCoordinator for simple usage."""
    coordinator = DataCoordinator(db_path)
    return coordinator.load_cores_with_fetch(tickers, start, end, auto_fetch)


def validate_cores(
    cores: Dict[str, pd.DataFrame], 
    analysis_type: str = "general"
) -> Dict[str, pd.DataFrame]:
    """Convenience function for core validation."""
    coordinator = DataCoordinator()
    return coordinator.validate_cores_for_analysis(cores, analysis_type)
