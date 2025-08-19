# xgb_iv_transfer.py  (ONLY CHANGES SHOWN)
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import os

DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
# top of file
import os, sqlite3
from pathlib import Path
DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def load_processed_from_sqlite(
    ticker: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    table: str | None = None,
) -> pd.DataFrame:
    if table is None:
        table = "processed_merged_1m"  # prefer fresh, 1m schema

    with sqlite3.connect(db_path) as conn:
        # fallback if this DB only has legacy table
        if not _table_exists(conn, table):
            alt = "processed_merged"
            if _table_exists(conn, alt):
                table = alt
            else:
                raise RuntimeError(f"Neither '{table}' nor '{alt}' exists in {db_path}")

        clauses, params = ["ticker = ?"], [ticker]
        if start is not None:
            clauses.append("ts_event >= ?"); params.append(start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        if end is not None:
            clauses.append("ts_event <= ?"); params.append(end.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        where = " AND ".join(clauses)

        q = f"""
        SELECT ts_event, expiry_date, opt_symbol, stock_symbol,
               opt_close, stock_close, opt_volume, stock_volume,
               option_type, strike_price, time_to_expiry, moneyness
        FROM {table}
        WHERE {where}
        ORDER BY ts_event
        """
        return pd.read_sql_query(q, conn, params=params, parse_dates=["ts_event","expiry_date"])


# ... (BS, implied_vol, pick_atm_per_timestamp unchanged) ...

def _atm_core_sqlite(
    ticker: str,
    start=None,
    end=None,
    r: float = 0.045,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    df = load_processed_from_sqlite(ticker, start=start, end=end, db_path=db_path)
    df = compute_iv_column(pick_atm_per_timestamp(df), r=r)
    if "opt_volume" not in df.columns:
        df["opt_volume"] = np.nan
    keep = ["ts_event","iv","opt_volume","time_to_expiry","strike_price","option_type"]
    out = df[keep].copy()
    out["symbol"] = ticker
    if "ts_event" in out.columns:
        is_dt = is_datetime64_any_dtype(out["ts_event"]) or is_datetime64tz_dtype(out["ts_event"])
        if not is_dt:
            out["ts_event"] = pd.to_datetime(out["ts_event"], utc=True, errors="coerce")
    return out.sort_values("ts_event").reset_index(drop=True)

def build_iv_return_dataset_time_safe(
    tickers: list[str],
    start=None,
    end=None,
    r: float = 0.045,
    forward_steps: int = 1,
    tolerance: str = "2s",
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, pd.DataFrame]:
    cores = {t: _atm_core_sqlite(t, start=start, end=end, r=r, db_path=db_path) for t in tickers}
    iv_wide = None
    for t, df in cores.items():
        tmp = df[["ts_event","iv"]].rename(columns={"iv": f"IV_{t}"}).sort_values("ts_event")
        iv_wide = tmp if iv_wide is None else pd.merge_asof(
            iv_wide, tmp, on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )
    out = {}
    for tgt, dft in cores.items():
        feats = dft.copy().sort_values("ts_event").reset_index(drop=True)
        feats["option_type_enc"] = feats["option_type"].astype(str).str.upper().str[0].map({"P":0,"C":1}).astype("float32")
        feats["hour"] = feats["ts_event"].dt.hour.astype("int16")
        feats["minute"] = feats["ts_event"].dt.minute.astype("int16")
        feats["day_of_week"] = feats["ts_event"].dt.dayofweek.astype("int16")
        feats["iv_clip"] = feats["iv"].clip(lower=1e-6)
        feats["iv_ret_fwd"] = np.log(feats["iv_clip"].shift(-forward_steps)) - np.log(feats["iv_clip"])
        panel = pd.merge_asof(
            feats, iv_wide, on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )
        peer_cols = [c for c in panel.columns if c.startswith("IV_") and c != f"IV_{tgt}"]
        final_cols = ["iv_clip","opt_volume","time_to_expiry","strike_price","option_type_enc","hour","minute","day_of_week"] + peer_cols
        for c in ["opt_volume","time_to_expiry","strike_price"] + peer_cols:
            if c in panel.columns:
                panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(np.float32)
        ds = panel[final_cols].dropna(subset=["iv_clip"]).reset_index(drop=True)
        out[tgt] = ds
    return out

def make_aligned_panel_sqlite(
    src_ticker: str,
    tgt_ticker: str,
    start=None,
    end=None,
    tolerance="2s",
    r=0.045,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    src = compute_iv_column(pick_atm_per_timestamp(load_processed_from_sqlite(src_ticker, start, end, db_path)), r=r)
    tgt = compute_iv_column(pick_atm_per_timestamp(load_processed_from_sqlite(tgt_ticker, start, end, db_path)), r=r)
    src = src[["ts_event","iv"]].rename(columns={"iv": f"{src_ticker}_iv"})
    tgt = tgt[["ts_event","iv"]].rename(columns={"iv": f"{tgt_ticker}_iv"})
    panel = pd.merge_asof(
        src.sort_values("ts_event"), tgt.sort_values("ts_event"),
        on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
    ).sort_values("ts_event").reset_index(drop=True)
    panel[f"{src_ticker}_iv_lag1"] = panel[f"{src_ticker}_iv"].shift(1)
    panel[f"{src_ticker}_iv_lag5"] = panel[f"{src_ticker}_iv"].rolling(5, min_periods=1).mean().shift(1)
    return panel.dropna(subset=[f"{src_ticker}_iv", f"{tgt_ticker}_iv"])

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
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (
        S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        if cp == "C"
        else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    )

def implied_vol(price, S, K, T, r, cp, lo=1e-6, hi=5.0):
    price = float(price)
    intrinsic = max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
    if price <= intrinsic + 1e-10:
        return 1e-6
    def f(sig): return _bs_price(S, K, T, r, sig, cp) - price
    try:
        return brentq(f, lo, hi, maxiter=100, xtol=1e-8)
    except Exception:
        return np.nan

def compute_iv_column(df, r=0.045):
    df = df.copy()
    def _row_iv(row):
        return implied_vol(
            price=row["opt_close"],
            S=row["stock_close"],
            K=row["strike_price"],
            T=max(float(row["time_to_expiry"]), 1e-6),
            r=r,
            cp=row["option_type"]
        )
    df["iv"] = df.apply(_row_iv, axis=1)
    return df

# -------------------------------------------------
# ATM slice
# -------------------------------------------------
def pick_atm_per_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["strike_price", "stock_close", "time_to_expiry", "option_type", "opt_close"])
    df["abs_moneyness"] = (df["strike_price"] - df["stock_close"]).abs()
    idx = df.groupby(["ts_event", "expiry_date"])["abs_moneyness"].idxmin()
    atm = df.loc[idx].copy()
    return atm.drop(columns=["abs_moneyness"], errors="ignore")


# -------------------------------------------------
# XGB training
# -------------------------------------------------
def train_xgb_iv_returns_time_safe(
    ds: pd.DataFrame,
    test_frac: float = 0.2,
    params: dict | None = None,
):
    """Train with chronological split (no lookahead)."""
    ds = ds.reset_index(drop=True)
    n = len(ds)
    if n < 10:
        raise ValueError(f"Too few rows: {n}")

    split_idx = int(n * (1 - test_frac))
    train_df, test_df = ds.iloc[:split_idx], ds.iloc[split_idx:]

    y_tr, X_tr = train_df["iv_clip"], train_df.drop(columns=["iv_clip"])
    y_te, X_te = test_df["iv_clip"], test_df.drop(columns=["iv_clip"])

    if params is None:
        params = dict(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    return model, {
        "RMSE": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "R2": float(r2_score(y_te, y_pred)),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }

def timeseries_cv_score(ds: pd.DataFrame, n_splits: int = 5, params: dict | None = None):
    """K-fold time series CV."""
    ds = ds.reset_index(drop=True)
    y = ds["iv_clip"].values
    X = ds.drop(columns=["iv_clip"]).values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        model = xgb.XGBRegressor(**(params or dict(objective="reg:squarederror")))
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        scores.append({
            "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
            "R2": float(r2_score(y_te, pred)),
        })
    return scores
def build_pooled_iv_return_dataset_time_safe(
    tickers: list[str],
    start=None,
    end=None,
    r: float = 0.045,
    forward_steps: int = 1,
    tolerance: str = "2s",
    db_path: str = 'iv_data.db',
) -> pd.DataFrame:
    """
    Build ONE pooled d
    ataset across all tickers.

    Target:
      iv_ret_fwd = log(IV_{t+Δ}) - log(IV_t), Δ=forward_steps

    Features at time t:
      - target: opt_volume, strike_price, option_type_enc
      - expiry info: days_to_expiry = (expiry_date - ts_event).days (numeric)
      - time encodings: hour, minute, day_of_week
      - symbol one-hot (sym_* columns)
      - peer IVs at t (backward as-of): IV_{peer}

    Returns a single DataFrame with columns:
      ['iv_ret_fwd', 'opt_volume','strike_price','option_type_enc',
       'days_to_expiry','hour','minute','day_of_week', sym_*..., IV_*...]
    """
    # 1) per-ticker ATM cores from SQLite (as earlier)
    cores = {t: _atm_core_sqlite(t, start=start, end=end, r=r) for t in tickers}

    # 2) wide panel of IVs for all tickers (as-of backward)
    iv_wide = None
    for t, df in cores.items():
        tmp = df[["ts_event", "iv"]].rename(columns={"iv": f"IV_{t}"}).sort_values("ts_event")
        iv_wide = tmp if iv_wide is None else pd.merge_asof(
            iv_wide, tmp, on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )

    pooled_frames = []
    for tgt, dft in cores.items():
        feats = dft.copy().sort_values("ts_event").reset_index(drop=True)

        # option type encoding
        feats["option_type_enc"] = _encode_option_type(feats["option_type"])

        # time encodings
        feats["hour"] = feats["ts_event"].dt.hour.astype("int16")
        feats["minute"] = feats["ts_event"].dt.minute.astype("int16")
        feats["day_of_week"] = feats["ts_event"].dt.dayofweek.astype("int16")

        # expiry as numeric: days_to_expiry
        # (we still have time_to_expiry in years in DB, but user asked for 'expiry_date' feature;
        # converting to days gives a usable numeric)
        # To compute days, we need expiry_date. Load minimally from DB for this ticker/time range:
        # (Fast path: derive from time_to_expiry if present)
        if "time_to_expiry" in feats.columns and feats["time_to_expiry"].notna().any():
            feats["days_to_expiry"] = (feats["time_to_expiry"] * 365.0).clip(lower=0).astype("float32")
        else:
            # Fallback (rare): if expiry_date not on feats, reload small slice from processed_merged
            back = load_processed_from_sqlite(tgt, start=start, end=end)[["ts_event","expiry_date"]]
            feats = pd.merge_asof(
                feats.sort_values("ts_event"),
                back.sort_values("ts_event"),
                on="ts_event",
                direction="backward",
                tolerance=pd.Timedelta(tolerance)
            )
            feats["days_to_expiry"] = ((feats["expiry_date"] - feats["ts_event"]).dt.total_seconds() / 86400.0)\
                                        .clip(lower=0).astype("float32")

        # forward target
        feats["iv_clip"] = feats["iv"].clip(lower=1e-6)
        feats["iv_ret_fwd"] = np.log(feats["iv_clip"].shift(-forward_steps)) - np.log(feats["iv_clip"])

        # join peer IVs as-of t (no lookahead)
        panel = pd.merge_asof(
            feats, iv_wide.sort_values("ts_event"),
            on="ts_event", direction="backward", tolerance=pd.Timedelta(tolerance)
        )

        # symbol one-hot (pooled model needs symbol as a feature)
        panel["symbol"] = tgt
        sym_dummies = pd.get_dummies(panel["symbol"], prefix="sym", dtype=np.float32)
        panel = pd.concat([panel, sym_dummies], axis=1)


        # choose columns
        peer_cols = [c for c in panel.columns if c.startswith("IV_") and c != f"IV_{tgt}"]
        final_cols = [
            "iv_clip",
            "opt_volume",
            "strike_price",
            "option_type_enc",
            "days_to_expiry",
        ] + list(sym_dummies.columns) + peer_cols

        # numeric coercions
        for c in ["opt_volume", "strike_price", "days_to_expiry"] + peer_cols:
            if c in panel.columns:
                panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(np.float32)
        panel["option_type_enc"] = panel["option_type_enc"].astype(np.float32)
        panel["hour"] = panel["hour"].astype(np.int16)
        panel["minute"] = panel["minute"].astype(np.int16)
        panel["day_of_week"] = panel["day_of_week"].astype(np.int16)


        panel = panel[final_cols].dropna(subset=["iv_clip"]).reset_index(drop=True)
        pooled_frames.append(panel)

    # 3) stack into a single pooled dataset
    pooled = pd.concat(pooled_frames, axis=0, ignore_index=True)
    # Final numeric-only safety: drop any lingering non-numeric features
    non_num = pooled.drop(columns=["iv_clip"]).select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_num:
        print(f"[WARN] Dropping non-numeric columns: {non_num}")
        pooled = pooled.drop(columns=non_num)

    # safe ordering: features after target
    y = pooled["iv_clip"]
    X = pooled.drop(columns=["iv_clip"])

    # Return single frame (keep target first for convenience)
    return pd.concat([y, X], axis=1)


import json

def train_xgb_iv_returns_time_safe_pooled(
    ds: pd.DataFrame,
    test_frac: float = 0.2,
    params: dict | None = None,
    model_name: str = "iv_pooled",
    metrics_dir: str | Path = "metrics",
    model_dir: str | Path = "models",
):
    """
    Train ONE pooled XGB model on the concatenated dataset and save metrics/model.
    """
    ds = ds.reset_index(drop=True)

    n = len(ds)
    if n < 10:
        raise ValueError(f"Too few rows: {n}")
    split_idx = int(n * (1 - test_frac))

    train_df, test_df = ds.iloc[:split_idx], ds.iloc[split_idx:]
    y_tr, X_tr = train_df["iv_clip"], train_df.drop(columns=["iv_clip"])
    y_te, X_te = test_df["iv_clip"], test_df.drop(columns=["iv_clip"])

    if params is None:
        params = dict(
            objective="reg:squarederror",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
        )

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "R2": float(r2_score(y_te, y_pred)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": int(X_tr.shape[1]),
    }

    # --- Save artifacts ---
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    metrics_path = Path(metrics_dir) / f"{model_name}_metrics.json"
    model_path   = Path(model_dir) / f"{model_name}.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    model.save_model(model_path)

    print(f"[SAVED] metrics → {metrics_path}")
    print(f"[SAVED] model   → {model_path}")

    return model, metrics

# -------------------------------------------------
# Demo main
# -------------------------------------------------
def main():
    tickers = ["QBTS", "IONQ"]
    datasets = build_iv_return_dataset_time_safe(tickers)
    for tgt, df in datasets.items():
        print(f"[IV-RET] {tgt}: {len(df)} rows")
        if len(df) > 50:
            _, metrics = train_xgb_iv_returns_time_safe(df)
            print(f"   RMSE={metrics['RMSE']:.6f}, R²={metrics['R2']:.3f}")

    print("\n[ATM PANEL] QBTS vs IONQ")
    panel = make_aligned_panel_sqlite("QBTS", "IONQ")
    print(panel.head())

if __name__ == "__main__":
    main()
