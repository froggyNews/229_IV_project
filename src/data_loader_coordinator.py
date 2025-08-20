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

# Import existing functions with safe error handling
try:
    from fetch_data_sqlite import fetch_and_save, get_conn, init_schema
except ImportError as e:
    print(f"Warning: Could not import fetch_data_sqlite functions: {e}")
    fetch_and_save = None
    get_conn = None
    init_schema = None


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


def _safe_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Safely check if a table exists in the database."""
    try:
        result = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", 
            (table_name,)
        ).fetchone()
        return result is not None
    except Exception:
        return False


def _safe_data_exists(conn: sqlite3.Connection, table: str, ticker: str, start: str = None, end: str = None) -> bool:
    """Safely check if data exists for a ticker in the given time range."""
    try:
        where_clauses, params = ["ticker=?"], [ticker]
        if start:
            where_clauses.append("ts_event >= ?")
            params.append(pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        if end:
            where_clauses.append("ts_event <= ?")
            params.append(pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            
        query = f"SELECT COUNT(*) FROM {table} WHERE {' AND '.join(where_clauses)}"
        result = conn.execute(query, params).fetchone()
        return result[0] > 0 if result else False
    except Exception:
        return False

def load_ticker_core(ticker: str, start=None, end=None, r=0.045, db_path=None) -> pd.DataFrame:
    """Load ticker core data with IV calculation."""
    
    if db_path is None:
        db_path = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
    
    # Check if database file exists
    if not Path(db_path).exists():
        print(f"Database file does not exist: {db_path}")
        return pd.DataFrame()
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # Try tables in order of preference
            table = None
            for candidate in ["atm_slices_1m", "processed_merged_1m", "processed_merged"]:
                if _safe_table_exists(conn, candidate):
                    if _safe_data_exists(conn, candidate, ticker, start, end):
                        table = candidate
                        break
            
            if table is None:
                print(f"No data found for {ticker} in any table")
                return pd.DataFrame()
            
            # If we have processed_merged_1m but not atm_slices_1m, populate ATM slices
            if table == "processed_merged_1m" and _safe_table_exists(conn, "atm_slices_1m"):
                print(f"Populating ATM slices for {ticker}...")
                _populate_atm_slices(conn, ticker)
                # Check if ATM data is now available
                if _safe_data_exists(conn, "atm_slices_1m", ticker, start, end):
                    table = "atm_slices_1m"
            
            where_clauses, params = ["ticker=?"], [ticker]
            if start:
                where_clauses.append("ts_event >= ?")
                params.append(pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            if end:
                where_clauses.append("ts_event <= ?")
                params.append(pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            
            # Build query based on table
            if table == "atm_slices_1m":
                query = f"""
                SELECT ts_event, expiry_date, opt_symbol, stock_symbol,
                       opt_close, stock_close, opt_volume, stock_volume,
                       option_type, strike_price, time_to_expiry, moneyness
                FROM {table} WHERE {' AND '.join(where_clauses)}
                ORDER BY ts_event
                """
            else:
                # For processed_merged_1m, we need to select ATM options manually
                query = f"""
                SELECT ts_event, expiry_date, opt_symbol, stock_symbol,
                       opt_close, stock_close, opt_volume, stock_volume,
                       option_type, strike_price, time_to_expiry, moneyness
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY ticker, ts_event, expiry_date
                               ORDER BY ABS(strike_price - stock_close)
                           ) as rn
                    FROM {table} 
                    WHERE {' AND '.join(where_clauses)}
                ) ranked
                WHERE rn = 1
                ORDER BY ts_event
                """
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["ts_event", "expiry_date"])
            
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()
    
    if df.empty:
        return df
        
    try:
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
        
    except Exception as e:
        print(f"Error processing IV data for {ticker}: {e}")
        return pd.DataFrame()
    
    return df

def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> list:
    """Get column names for a table."""
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]
    except Exception:
        return []


def _populate_atm_slices(conn: sqlite3.Connection, ticker: str) -> None:
    """Populate ATM slices table from processed_merged_1m data."""
    try:
        # Get column schemas for both tables
        atm_cols = _get_table_columns(conn, "atm_slices_1m")
        processed_cols = _get_table_columns(conn, "processed_merged_1m")
        
        if not atm_cols or not processed_cols:
            print(f"  ✗ Could not get table schemas for {ticker}")
            return
        
        # Find common columns (excluding any extras)
        common_cols = [col for col in atm_cols if col in processed_cols]
        
        if len(common_cols) < 10:  # Sanity check
            print(f"  ✗ Too few common columns ({len(common_cols)}) for {ticker}")
            return
        
        # Build the query with exact column matching
        cols_str = ", ".join(common_cols)
        
        q = f"""
        INSERT OR REPLACE INTO atm_slices_1m ({cols_str})
        SELECT {cols_str}
        FROM (
            SELECT
                {cols_str},
                ROW_NUMBER() OVER (
                  PARTITION BY ticker, ts_event, expiry_date
                  ORDER BY ABS(strike_price - stock_close)
                ) rn
            FROM processed_merged_1m
            WHERE ticker = ?
        )
        WHERE rn = 1;
        """
        
        result = conn.execute(q, (ticker,))
        conn.commit()
        rows_affected = result.rowcount
        print(f"  ✓ Populated ATM slices for {ticker} ({rows_affected} rows)")
        
    except Exception as e:
        print(f"  ✗ Failed to populate ATM slices for {ticker}: {e}")


def load_ticker_core(ticker: str, start=None, end=None, r=0.045, db_path=None) -> pd.DataFrame:
    """Load ticker core data with IV calculation."""
    
    if db_path is None:
        db_path = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
    
    # Check if database file exists
    if not Path(db_path).exists():
        print(f"Database file does not exist: {db_path}")
        return pd.DataFrame()
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # Debug: Check table schemas
            # print(f"DEBUG: atm_slices_1m columns: {_get_table_columns(conn, 'atm_slices_1m')}")
            # print(f"DEBUG: processed_merged_1m columns: {_get_table_columns(conn, 'processed_merged_1m')}")
            
            # Try tables in order of preference
            table = None
            for candidate in ["atm_slices_1m", "processed_merged_1m", "processed_merged"]:
                if _safe_table_exists(conn, candidate):
                    if _safe_data_exists(conn, candidate, ticker, start, end):
                        table = candidate
                        break
            
            if table is None:
                print(f"No data found for {ticker} in any table")
                return pd.DataFrame()
            
            # If we have processed_merged_1m but not atm_slices_1m, populate ATM slices
            if table == "processed_merged_1m" and _safe_table_exists(conn, "atm_slices_1m"):
                # Only populate if ATM table is truly empty for this ticker
                if not _safe_data_exists(conn, "atm_slices_1m", ticker, start, end):
                    _populate_atm_slices(conn, ticker)
                    # Check if ATM data is now available
                    if _safe_data_exists(conn, "atm_slices_1m", ticker, start, end):
                        table = "atm_slices_1m"
            
            where_clauses, params = ["ticker=?"], [ticker]
            if start:
                where_clauses.append("ts_event >= ?")
                params.append(pd.to_datetime(start).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            if end:
                where_clauses.append("ts_event <= ?")
                params.append(pd.to_datetime(end).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
            
            # Get required columns that actually exist in the table
            required_cols = ["ts_event", "expiry_date", "opt_symbol", "stock_symbol",
                           "opt_close", "stock_close", "opt_volume", "stock_volume",
                           "option_type", "strike_price", "time_to_expiry", "moneyness"]
            
            available_cols = _get_table_columns(conn, table)
            select_cols = [col for col in required_cols if col in available_cols]
            
            if len(select_cols) < 8:  # Minimum required columns
                print(f"Insufficient columns in {table} for {ticker}")
                return pd.DataFrame()
            
            cols_str = ", ".join(select_cols)
            
            # Build query based on table
            if table == "atm_slices_1m":
                query = f"""
                SELECT {cols_str}
                FROM {table} WHERE {' AND '.join(where_clauses)}
                ORDER BY ts_event
                """
            else:
                # For processed_merged_1m, we need to select ATM options manually
                query = f"""
                SELECT {cols_str}
                FROM (
                    SELECT {cols_str},
                           ROW_NUMBER() OVER (
                               PARTITION BY ticker, ts_event, expiry_date
                               ORDER BY ABS(strike_price - stock_close)
                           ) as rn
                    FROM {table} 
                    WHERE {' AND '.join(where_clauses)}
                ) ranked
                WHERE rn = 1
                ORDER BY ts_event
                """
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["ts_event", "expiry_date"])
            
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return pd.DataFrame()
    
    if df.empty:
        return df
        
    try:
        # IV calculation
        df["iv"] = df.apply(lambda row: _calculate_iv(
            row["opt_close"], row["stock_close"], row["strike_price"], 
            max(row["time_to_expiry"], 1e-6), row["option_type"], r
        ), axis=1)
        
        # Core cleanup (preserves original column selection and processing)
        available_keep = [col for col in ["ts_event", "expiry_date", "iv", "opt_volume", "stock_close", 
                                        "stock_volume", "time_to_expiry", "strike_price", "option_type"] 
                         if col in df.columns]
        df = df[available_keep].copy()
        df["symbol"] = ticker
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
        df = df.dropna(subset=["iv"]).sort_values("ts_event").reset_index(drop=True)
        df["iv_clip"] = df["iv"].clip(lower=1e-6)
        
    except Exception as e:
        print(f"Error processing IV data for {ticker}: {e}")
        return pd.DataFrame()
    
    return df


class DataCoordinator:
    """Coordinates data loading and ensures consistency across modules."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
        self.api_key = os.getenv("DATABENTO_API_KEY")
        
    def _safe_fetch_data(self, ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
        """Safely attempt to fetch data using fetch_data_sqlite functions."""
        if not self.api_key:
            print(f"    ✗ No API key available for fetching {ticker}")
            return False
            
        if fetch_and_save is None:
            print(f"    ✗ fetch_data_sqlite functions not available for {ticker}")
            return False
            
        try:
            # Initialize database schema if needed
            if not self.db_path.exists():
                print(f"    Creating new database: {self.db_path}")
                if get_conn and init_schema:
                    conn = get_conn(self.db_path)
                    init_schema(conn)
                    conn.close()
                else:
                    print(f"    ✗ Cannot initialize database schema")
                    return False
            
            # Attempt to fetch data (force=True to repopulate ATM slices)
            fetch_and_save(self.api_key, ticker, start_ts, end_ts, self.db_path, force=True)
            return True
            
        except Exception as e:
            print(f"    ✗ Fetch failed for {ticker}: {e}")
            return False
    
    def populate_missing_atm_slices(self, tickers: Sequence[str]) -> None:
        """Populate ATM slices for tickers that have processed data but missing ATM slices."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                for ticker in tickers:
                    # Check if we have processed data but missing ATM data
                    has_processed = _safe_data_exists(conn, "processed_merged_1m", ticker)
                    has_atm = _safe_data_exists(conn, "atm_slices_1m", ticker)
                    
                    if has_processed and not has_atm:
                        print(f"  Populating ATM slices for {ticker}...")
                        _populate_atm_slices(conn, ticker)
        except Exception as e:
            print(f"Error populating ATM slices: {e}")
    
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
        
        # First, try to populate ATM slices from existing processed data
        # Only do this once at the beginning
        self.populate_missing_atm_slices(tickers)
        
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
        if missing_tickers and auto_fetch:
            print(f"Auto-fetching {len(missing_tickers)} missing tickers...")
            
            start_ts = pd.Timestamp(start, tz="UTC")
            end_ts = pd.Timestamp(end, tz="UTC")
            
            for ticker in missing_tickers:
                print(f"  Fetching {ticker}...")
                
                if self._safe_fetch_data(ticker, start_ts, end_ts):
                    # Retry loading after fetch
                    try:
                        core = load_ticker_core(ticker, start=start, end=end, db_path=self.db_path)
                        if not core.empty:
                            cores[ticker] = core
                            print(f"    ✓ Fetched {ticker}: {len(core):,} rows")
                        else:
                            print(f"    ✗ {ticker}: no data even after fetch")
                    except Exception as e:
                        print(f"    ✗ {ticker}: error loading after fetch ({e})")
                else:
                    print(f"    ✗ {ticker}: could not fetch data")
                    
        elif missing_tickers and not auto_fetch:
            print(f"Skipping auto-fetch for {len(missing_tickers)} missing tickers (auto_fetch=False)")
            
        print(f"Final result: {len(cores)}/{len(tickers)} tickers loaded")
        return cores

    # ...rest of the methods remain the same...




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