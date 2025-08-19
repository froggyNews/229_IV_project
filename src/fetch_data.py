# fetch_data_sqlite.py  (REPLACE FILE)
from __future__ import annotations
import os
from pathlib import Path
import sqlite3
from typing import Tuple
import pandas as pd
import numpy as np
import databento as db
from dotenv import load_dotenv

# Default if caller doesn't pass a db_path
DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))

# -----------------------------
# SQLite helpers
# -----------------------------
def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def get_conn(db_path: Path) -> sqlite3.Connection:
    _ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS opra_1m (
            ticker TEXT NOT NULL,
            ts_event TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL,
            symbol TEXT,
            PRIMARY KEY (ticker, ts_event, symbol)
        );
        CREATE INDEX IF NOT EXISTS idx_opra_1m_ts ON opra_1m(ticker, ts_event);

        CREATE TABLE IF NOT EXISTS equity_1m (
            ticker TEXT NOT NULL,
            ts_event TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL,
            symbol TEXT,
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
            opt_symbol TEXT,
            stock_symbol TEXT,
            opt_close REAL,
            stock_close REAL,
            opt_volume REAL,
            stock_volume REAL,
            PRIMARY KEY (ticker, ts_event, opt_symbol)
        );
        CREATE INDEX IF NOT EXISTS idx_merged_1m_ts ON merged_1m(ticker, ts_event);

        CREATE TABLE IF NOT EXISTS processed_merged_1m (
            ticker TEXT NOT NULL,
            ts_event TEXT NOT NULL,
            opt_symbol TEXT,
            stock_symbol TEXT,
            opt_close REAL,
            stock_close REAL,
            opt_volume REAL,
            stock_volume REAL,
            expiry_date TEXT,
            option_type TEXT,
            strike_price REAL,
            time_to_expiry REAL,
            moneyness REAL,
            PRIMARY KEY (ticker, ts_event, opt_symbol)
        );
        CREATE INDEX IF NOT EXISTS idx_processed_1m_ts  ON processed_merged_1m(ticker, ts_event);
        CREATE INDEX IF NOT EXISTS idx_processed_1m_exp ON processed_merged_1m(ticker, expiry_date);

        CREATE TABLE IF NOT EXISTS atm_slices_1m (
            ticker TEXT NOT NULL,
            ts_event TEXT NOT NULL,
            expiry_date TEXT NOT NULL,
            opt_symbol TEXT,
            stock_symbol TEXT,
            opt_close REAL,
            stock_close REAL,
            opt_volume REAL,
            stock_volume REAL,
            option_type TEXT,
            strike_price REAL,
            time_to_expiry REAL,
            moneyness REAL,
            PRIMARY KEY (ticker, ts_event, expiry_date)
        );
        CREATE INDEX IF NOT EXISTS idx_atm_1m_ts  ON atm_slices_1m(ticker, ts_event);
        CREATE INDEX IF NOT EXISTS idx_atm_1m_exp ON atm_slices_1m(ticker, expiry_date);
        """
    )
    conn.commit()

def populate_atm_slices(conn: sqlite3.Connection, ticker: str):
    q = """
    INSERT OR REPLACE INTO atm_slices_1m (
        ticker, ts_event, expiry_date, opt_symbol, stock_symbol,
        opt_close, stock_close, opt_volume, stock_volume,
        option_type, strike_price, time_to_expiry, moneyness
    )
    SELECT
        ticker, ts_event, expiry_date, opt_symbol, stock_symbol,
        opt_close, stock_close, opt_volume, stock_volume,
        option_type, strike_price, time_to_expiry, moneyness
    FROM (
        SELECT
            pm.*,
            ROW_NUMBER() OVER (
                PARTITION BY pm.ticker, pm.ts_event, pm.expiry_date
                ORDER BY ABS(pm.strike_price - pm.stock_close) ASC
            ) AS rn
        FROM processed_merged_1m pm
        WHERE pm.ticker = ?
    )
    WHERE rn = 1;
    """
    conn.execute(q, (ticker,))
    conn.commit()

def _iso_utc(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return s.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def _upsert_df(conn: sqlite3.Connection, df: pd.DataFrame, table: str):
    if df.empty:
        return
    tmp = f"tmp_{table}"
    df.to_sql(tmp, conn, if_exists="replace", index=False)
    cols = list(df.columns)
    col_list = ",".join(cols)
    conn.execute(f"INSERT OR IGNORE INTO {table} ({col_list}) SELECT {col_list} FROM {tmp};")
    conn.execute(f"DROP TABLE {tmp};")
    conn.commit()

# -----------------------------
# Databento fetch
# -----------------------------
def get_data_from_databento(API_KEY: str, start_date: pd.Timestamp, end_date: pd.Timestamp, ticker: str
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    client = db.Historical(API_KEY)
    opt_symbol = f"{ticker}.opt"

    opra_1m = client.timeseries.get_range(
        dataset="OPRA.PILLAR",
        stype_in="parent",
        symbols=[opt_symbol],
        schema="OHLCV-1m",
        start=start_date,
        end=end_date,
    ).to_df().reset_index()

    eq_1m = client.timeseries.get_range(
        dataset="XNAS.ITCH",
        symbols=[ticker],
        schema="OHLCV-1m",
        start=start_date,
        end=end_date,
    ).to_df().reset_index()

    eq_1h = client.timeseries.get_range(
        dataset="XNAS.ITCH",
        symbols=[ticker],
        schema="OHLCV-1H",
        start=start_date - pd.DateOffset(years=2),
        end=end_date,
    ).to_df().reset_index()

    for df in (opra_1m, eq_1m, eq_1h):
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")

    return opra_1m, eq_1m, eq_1h

# -----------------------------
# Preprocess & persist to SQLite
# -----------------------------
def calculate_sigma(equity_1h: pd.DataFrame) -> float:
    eq = equity_1h.copy()
    eq["close"] = pd.to_numeric(eq["close"], errors="coerce")
    ret = eq["close"].pct_change()
    hourly_sigma = ret.std()
    ann_sigma = hourly_sigma * np.sqrt(252 * 6.5)  # rough scale
    return float(ann_sigma) if np.isfinite(ann_sigma) else np.nan

def preprocess_and_store(
    API_KEY: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ticker: str,
    conn: sqlite3.Connection,
    r: float = 0.045,
    force: bool = False,
):
    if not force:
        q = """
        SELECT COUNT(1) FROM processed_merged_1m
        WHERE ticker = ?
          AND ts_event >= ?
          AND ts_event <= ?;
        """
        cur = conn.execute(q, (ticker, start_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                    end_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")))
        cnt = cur.fetchone()[0]
        if cnt and cnt > 0:
            print(f"[SKIP] processed_merged_1m already has rows for {ticker} in window.")
            return

    opra_1m, eq_1m, eq_1h = get_data_from_databento(API_KEY, start_date, end_date, ticker)

    # persist raw
    opra_db = opra_1m[["ts_event","open","high","low","close","volume","symbol"]].copy()
    opra_db.insert(0, "ticker", ticker)
    opra_db["ts_event"] = _iso_utc(opra_db["ts_event"])
    _upsert_df(conn, opra_db, "opra_1m")

    equity_1m_db = eq_1m[["ts_event","open","high","low","close","volume","symbol"]].copy()
    equity_1m_db.insert(0, "ticker", ticker)
    equity_1m_db["ts_event"] = _iso_utc(equity_1m_db["ts_event"])
    _upsert_df(conn, equity_1m_db, "equity_1m")

    equity_1h_db = eq_1h[["ts_event","open","high","low","close","volume"]].copy()
    equity_1h_db.insert(0, "ticker", ticker)
    equity_1h_db["ts_event"] = _iso_utc(equity_1h_db["ts_event"])
    _upsert_df(conn, equity_1h_db, "equity_1h")

    # merge @ 1m
    merged = pd.merge(
        opra_1m.rename(columns={"close":"opt_close","volume":"opt_volume","symbol":"opt_symbol"}),
        eq_1m.rename(columns={"close":"stock_close","volume":"stock_volume","symbol":"stock_symbol"}),
        on="ts_event",
        how="inner",
        suffixes=("_opt","_eq"),
    )
    merged = merged.copy()
    merged["ts_event"] = pd.to_datetime(merged["ts_event"], utc=True, errors="coerce")
    merged = merged[(merged["ts_event"].dt.hour >= 14) & (merged["ts_event"].dt.hour <= 21)]

    # OCC parse
    extracted = merged["opt_symbol"].astype(str).str.extract(r"(\d{6})([CP])(\d{8})")
    merged["expiry_date"] = pd.to_datetime(extracted[0], format="%y%m%d", errors="coerce", utc=True)
    merged["option_type"] = extracted[1]
    merged["strike_price"] = pd.to_numeric(extracted[2], errors="coerce") / 1000.0
    merged["time_to_expiry"] = ((merged["expiry_date"] - merged["ts_event"]).dt.total_seconds() / (365*24*3600)).clip(lower=0.0)
    merged["moneyness"] = np.where(
        merged["option_type"].eq("C"),
        merged["stock_close"] - merged["strike_price"],
        np.where(merged["option_type"].eq("P"), merged["strike_price"] - merged["stock_close"], np.nan)
    )
    merged = merged.dropna(subset=["expiry_date","strike_price","option_type","opt_close","stock_close","time_to_expiry"])

    # persist merged_1m + processed_merged_1m
    merged_db = merged[["ts_event","opt_symbol","stock_symbol","opt_close","stock_close","opt_volume","stock_volume"]].copy()
    merged_db.insert(0, "ticker", ticker)
    merged_db["ts_event"] = _iso_utc(merged_db["ts_event"])
    _upsert_df(conn, merged_db, "merged_1m")

    processed_db = merged[[
        "ts_event","opt_symbol","stock_symbol","opt_close","stock_close","opt_volume","stock_volume",
        "expiry_date","option_type","strike_price","time_to_expiry","moneyness"
    ]].copy()
    processed_db.insert(0, "ticker", ticker)
    processed_db["ts_event"] = _iso_utc(processed_db["ts_event"])
    processed_db["expiry_date"] = _iso_utc(processed_db["expiry_date"])
    _upsert_df(conn, processed_db, "processed_merged_1m")

    # ATM slices
    populate_atm_slices(conn, ticker)

    sigma = calculate_sigma(eq_1h)
    sigma_str = f"{sigma:.4f}" if np.isfinite(sigma) else "nan"
    print(f"[DONE] {ticker}: processed_1m={len(processed_db)}  sigma_annual≈{sigma_str}")

# -----------------------------
# Public entry
# -----------------------------
def fetch_and_save(API_KEY: str, ticker: str, start: pd.Timestamp, end: pd.Timestamp,
                   db_path: Path | str | None = None, force: bool = False) -> Path:
    dbp = Path(db_path) if db_path else DEFAULT_DB_PATH
    conn = get_conn(dbp)
    init_schema(conn)
    preprocess_and_store(API_KEY, start, end, ticker, conn, force=force)
    conn.close()
    return dbp

# -----------------------------
# Demo main
# -----------------------------
def main():
    load_dotenv()
    API_KEY = os.getenv("DATABENTO_API_KEY")
    if not API_KEY:
        raise ValueError("Missing DATABENTO_API_KEY")

    tickers = ["QBTS", "IONQ", "QUBT", "RGTI"]
    start_date = pd.Timestamp("2025-01-02", tz="UTC")
    end_date   = pd.Timestamp("2025-01-06", tz="UTC")
    run_db = Path(f"data/iv_data_1m_{start_date:%Y%m%d}_{end_date:%Y%m%d}.db")

    for t in tickers:
        print(f"[DL] {t}  {start_date.date()} → {end_date.date()}")
        fetch_and_save(API_KEY, t, start_date, end_date, db_path=run_db, force=False)

    print(f"SQLite DB ready at: {run_db.resolve()}")

if __name__ == "__main__":
    main()
