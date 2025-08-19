# fetch_data_sqlite.py
from __future__ import annotations
import os
from pathlib import Path
import argparse
import sqlite3
from typing import Tuple
import numpy as np
import pandas as pd
import databento as db
from dotenv import load_dotenv

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def get_conn(db_path: Path) -> sqlite3.Connection:
    _ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS opra_1m (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL, symbol TEXT,
        PRIMARY KEY (ticker, ts_event, symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_opra_1m_ts ON opra_1m(ticker, ts_event);

    CREATE TABLE IF NOT EXISTS equity_1m (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL, symbol TEXT,
        PRIMARY KEY (ticker, ts_event)
    );
    CREATE INDEX IF NOT EXISTS idx_equity_1m_ts ON equity_1m(ticker, ts_event);

    CREATE TABLE IF NOT EXISTS equity_1h (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL,
        PRIMARY KEY (ticker, ts_event)
    );
    CREATE INDEX IF NOT EXISTS idx_equity_1h_ts ON equity_1h(ticker, ts_event);

    CREATE TABLE IF NOT EXISTS merged_1m (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        opt_symbol TEXT, stock_symbol TEXT,
        opt_close REAL, stock_close REAL,
        opt_volume REAL, stock_volume REAL,
        PRIMARY KEY (ticker, ts_event, opt_symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_merged_1m_ts ON merged_1m(ticker, ts_event);

    CREATE TABLE IF NOT EXISTS processed_merged_1m (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        opt_symbol TEXT, stock_symbol TEXT,
        opt_close REAL, stock_close REAL,
        opt_volume REAL, stock_volume REAL,
        expiry_date TEXT, option_type TEXT,
        strike_price REAL, time_to_expiry REAL, moneyness REAL,
        PRIMARY KEY (ticker, ts_event, opt_symbol)
    );
    CREATE INDEX IF NOT EXISTS idx_processed_1m_ts  ON processed_merged_1m(ticker, ts_event);
    CREATE INDEX IF NOT EXISTS idx_processed_1m_exp ON processed_merged_1m(ticker, expiry_date);

    CREATE TABLE IF NOT EXISTS atm_slices_1m (
        ticker TEXT NOT NULL,
        ts_event TEXT NOT NULL,
        expiry_date TEXT NOT NULL,
        opt_symbol TEXT, stock_symbol TEXT,
        opt_close REAL, stock_close REAL,
        opt_volume REAL, stock_volume REAL,
        option_type TEXT, strike_price REAL,
        time_to_expiry REAL, moneyness REAL,
        PRIMARY KEY (ticker, ts_event, expiry_date)
    );
    CREATE INDEX IF NOT EXISTS idx_atm_1m_ts  ON atm_slices_1m(ticker, ts_event);
    CREATE INDEX IF NOT EXISTS idx_atm_1m_exp ON atm_slices_1m(ticker, expiry_date);
    """)
    conn.commit()

def _iso_utc(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, utc=True, errors="coerce")
    return s.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def _upsert(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    if df.empty: return
    tmp = f"tmp_{table}"
    df.to_sql(tmp, conn, if_exists="replace", index=False)
    cols = ",".join(df.columns)
    conn.execute(f"INSERT OR IGNORE INTO {table} ({cols}) SELECT {cols} FROM {tmp};")
    conn.execute(f"DROP TABLE {tmp};")
    conn.commit()

def _calculate_sigma_realized(bars: pd.DataFrame, tz: str = "America/New_York") -> float:
    if bars is None or bars.empty: return np.nan
    df = bars[["ts_event","close"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(inplace=True)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df.dropna(subset=["ts_event"], inplace=True)
    df["ts_local"] = df["ts_event"].dt.tz_convert(tz)
    delta = df["ts_event"].sort_values().diff().median()
    is_hourly = pd.notna(delta) and delta >= pd.Timedelta("30min")
    if is_hourly:
        mask = ((df["ts_local"].dt.minute == 0) &
                (df["ts_local"].dt.hour >= 10) &
                (df["ts_local"].dt.hour <= 15))
        min_obs = 3
    else:
        h, m = df["ts_local"].dt.hour, df["ts_local"].dt.minute
        mask = (((h > 9) | ((h == 9) & (m >= 30))) & (h < 16))
        min_obs = 100
    df = df.loc[mask].sort_values("ts_event")
    if df.empty: return np.nan
    df["trade_date"] = df["ts_local"].dt.date
    df["logp"] = np.log(df["close"])
    df["ret"] = df.groupby("trade_date")["logp"].diff()
    rv = df.groupby("trade_date")["ret"].apply(lambda x: np.nansum(x.values**2))
    obs = df.groupby("trade_date")["ret"].apply(lambda x: np.isfinite(x.values).sum())
    rv = rv.loc[obs[obs >= min_obs].index]
    if rv.empty or not np.isfinite(rv).any(): return np.nan
    sig = float(np.sqrt(rv.mean() * 252.0))
    return sig if np.isfinite(sig) else np.nan

def _fetch(API_KEY: str, start: pd.Timestamp, end: pd.Timestamp, ticker: str
           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    client = db.Historical(API_KEY)
    opt_symbol = f"{ticker}.opt"
    opra_1m = client.timeseries.get_range(
        dataset="OPRA.PILLAR", stype_in="parent", symbols=[opt_symbol],
        schema="OHLCV-1m", start=start, end=end
    ).to_df().reset_index()
    eq_1m = client.timeseries.get_range(
        dataset="XNAS.ITCH", symbols=[ticker],
        schema="OHLCV-1m", start=start, end=end
    ).to_df().reset_index()
    eq_1h = client.timeseries.get_range(
        dataset="XNAS.ITCH", symbols=[ticker],
        schema="OHLCV-1H", start=start - pd.DateOffset(years=2), end=end
    ).to_df().reset_index()
    for df in (opra_1m, eq_1m, eq_1h):
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    return opra_1m, eq_1m, eq_1h

def _populate_atm(conn: sqlite3.Connection, ticker: str) -> None:
    q = """
    INSERT OR REPLACE INTO atm_slices_1m (
        ticker, ts_event, expiry_date, opt_symbol, stock_symbol,
        opt_close, stock_close, opt_volume, stock_volume,
        option_type, strike_price, time_to_expiry, moneyness
    )
    SELECT *
    FROM (
        SELECT
            pm.ticker, pm.ts_event, pm.expiry_date, pm.opt_symbol, pm.stock_symbol,
            pm.opt_close, pm.stock_close, pm.opt_volume, pm.stock_volume,
            pm.option_type, pm.strike_price, pm.time_to_expiry, pm.moneyness,
            ROW_NUMBER() OVER (
              PARTITION BY pm.ticker, pm.ts_event, pm.expiry_date
              ORDER BY ABS(pm.strike_price - pm.stock_close)
            ) rn
        FROM processed_merged_1m pm
        WHERE pm.ticker = ?
    )
    WHERE rn = 1;
    """
    conn.execute(q, (ticker,))
    conn.commit()

def preprocess_and_store(API_KEY: str, start: pd.Timestamp, end: pd.Timestamp,
                         ticker: str, conn: sqlite3.Connection, force: bool=False) -> None:
    # skip if we already have this window
    if not force:
        q = """SELECT COUNT(1) FROM processed_merged_1m
               WHERE ticker=? AND ts_event >= ? AND ts_event <= ?"""
        args = (ticker, start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        if conn.execute(q, args).fetchone()[0] > 0:
            print(f"[SKIP] {ticker} already in processed_merged_1m for window.")
            return

    opra_1m, eq_1m, eq_1h = _fetch(API_KEY, start, end, ticker)

    # persist raw
    opra_db = opra_1m[["ts_event","open","high","low","close","volume","symbol"]].copy()
    opra_db.insert(0,"ticker",ticker); opra_db["ts_event"] = _iso_utc(opra_db["ts_event"])
    _upsert(conn, opra_db, "opra_1m")

    eq1m_db = eq_1m[["ts_event","open","high","low","close","volume","symbol"]].copy()
    eq1m_db.insert(0,"ticker",ticker); eq1m_db["ts_event"] = _iso_utc(eq1m_db["ts_event"])
    _upsert(conn, eq1m_db, "equity_1m")

    eq1h_db = eq_1h[["ts_event","open","high","low","close","volume"]].copy()
    eq1h_db.insert(0,"ticker",ticker); eq1h_db["ts_event"] = _iso_utc(eq1h_db["ts_event"])
    _upsert(conn, eq1h_db, "equity_1h")

    # merge (1m), restrict to RTH 14:00–21:00 UTC
    merged = pd.merge(
        opra_1m.rename(columns={"close":"opt_close","volume":"opt_volume","symbol":"opt_symbol"}),
        eq_1m.rename(columns={"close":"stock_close","volume":"stock_volume","symbol":"stock_symbol"}),
        on="ts_event", how="inner"
    )
    merged = merged.copy()
    merged = merged[(merged["ts_event"].dt.hour >= 14) & (merged["ts_event"].dt.hour <= 21)]

    # OCC parse: YYMMDD [C|P] ########
    ex = merged["opt_symbol"].astype(str).str.extract(r"(\d{6})([CP])(\d{8})")
    merged["expiry_date"]  = pd.to_datetime(ex[0], format="%y%m%d", utc=True, errors="coerce")
    merged["option_type"]  = ex[1]
    merged["strike_price"] = pd.to_numeric(ex[2], errors="coerce") / 1000.0
    merged["time_to_expiry"] = ((merged["expiry_date"] - merged["ts_event"]).dt.total_seconds()
                                /(365*24*3600)).clip(lower=0.0)
    merged["moneyness"] = np.where(
        merged["option_type"].eq("C"),
        merged["stock_close"] - merged["strike_price"],
        np.where(merged["option_type"].eq("P"),
                 merged["strike_price"] - merged["stock_close"], np.nan)
    )
    merged = merged.dropna(subset=["expiry_date","strike_price","option_type",
                                   "opt_close","stock_close","time_to_expiry"])

    # persist merged + processed
    m_db = merged[["ts_event","opt_symbol","stock_symbol","opt_close","stock_close","opt_volume","stock_volume"]].copy()
    m_db.insert(0,"ticker",ticker); m_db["ts_event"] = _iso_utc(m_db["ts_event"])
    _upsert(conn, m_db, "merged_1m")

    p_db = merged[["ts_event","opt_symbol","stock_symbol","opt_close","stock_close","opt_volume","stock_volume",
                   "expiry_date","option_type","strike_price","time_to_expiry","moneyness"]].copy()
    p_db.insert(0,"ticker",ticker)
    p_db["ts_event"]   = _iso_utc(p_db["ts_event"])
    p_db["expiry_date"]= _iso_utc(p_db["expiry_date"])
    _upsert(conn, p_db, "processed_merged_1m")

    _populate_atm(conn, ticker)

    sigma = _calculate_sigma_realized(eq_1h)
    print(f"[DONE] {ticker}: rows={len(p_db)}  sigma_annual≈{sigma:.4f}" if np.isfinite(sigma) else
          f"[DONE] {ticker}: rows={len(p_db)}  sigma_annual≈nan")

def fetch_and_save(API_KEY: str, ticker: str, start: pd.Timestamp, end: pd.Timestamp,
                   db_path: Path, force: bool=False) -> Path:
    conn = get_conn(db_path)
    init_schema(conn)
    preprocess_and_store(API_KEY, start, end, ticker, conn, force=force)
    conn.close()
    return db_path

def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    API_KEY = os.getenv("DATABENTO_API_KEY")
    if not API_KEY:
        raise ValueError("Missing DATABENTO_API_KEY")

    start = pd.Timestamp(args.start, tz="UTC")
    end   = pd.Timestamp(args.end, tz="UTC")
    for t in args.tickers:
        print(f"[DL] {t}  {start.date()} → {end.date()}")
        fetch_and_save(API_KEY, t, start, end, db_path=args.db, force=args.force)
    print(f"[OK] SQLite: {args.db.resolve()}")

if __name__ == "__main__":
    main()
