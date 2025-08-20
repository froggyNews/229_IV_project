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

from greeks import bs_delta, bs_gamma, bs_vega

DEFAULT_DB_PATH = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
# --- add near top of file ---
TRADING_MIN_PER_DAY = 390
ANNUAL_MINUTES = 252 * TRADING_MIN_PER_DAY

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
    # target’s own past IV return (strictly lagged to avoid leakage)
    X["iv_ret_1m"] = (np.log(X["iv_clip"]) - np.log(X["iv_clip"].shift(1))).shift(1)

    # --- peer IV-RET aggregates (optional) ---
    if peer_ret_cols:
        peer_ret_cols = [c for c in peer_ret_cols if c in X.columns]
        if peer_ret_cols:
            pr_mean = X[peer_ret_cols].mean(axis=1)
            X["peer_ivret_mean"] = pr_mean
            X["peer_ivret_std"]  = X[peer_ret_cols].std(axis=1)
    # --- Greeks (use current iv level; shift to be safe) ---
    def _safe_delta(row):
        return bs_delta(
            row.get("stock_close", np.nan),
            row.get("strike_price", np.nan),
            row.get("time_to_expiry", np.nan),
            r,
            row.get("iv_clip", np.nan),
            str(row.get("option_type", "C"))[:1].upper(),
        )

    def _safe_gamma(row):
        return bs_gamma(
            row.get("stock_close", np.nan),
            row.get("strike_price", np.nan),
            row.get("time_to_expiry", np.nan),
            r,
            row.get("iv_clip", np.nan),
        )

    def _safe_vega(row):
        return bs_vega(
            row.get("stock_close", np.nan),
            row.get("strike_price", np.nan),
            row.get("time_to_expiry", np.nan),
            r,
            row.get("iv_clip", np.nan),
        )

    X["delta"] = X.apply(_safe_delta, axis=1).shift(1)
    X["gamma"] = X.apply(_safe_gamma, axis=1).shift(1)
    X["vega"] = X.apply(_safe_vega, axis=1).shift(1)

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
    tol = pd.Timedelta(tolerance)
    for t, df in cores.items():
        tmp = df[["ts_event", "iv"]].sort_values("ts_event").copy()
        tmp["iv_clip_tmp"] = tmp["iv"].clip(lower=1e-6)
        tmp[f"IV_{t}"] = tmp["iv_clip_tmp"]
        tmp[f"IVRET_{t}"] = np.log(tmp["iv_clip_tmp"]) - np.log(tmp["iv_clip_tmp"].shift(1))
        tmp = tmp[["ts_event", f"IV_{t}", f"IVRET_{t}"]]
        iv_wide = tmp if iv_wide is None else pd.merge_asof(
            iv_wide, tmp, on="ts_event", direction="backward", tolerance=tol
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
    # rename target’s own columns
    if f"IV_{tgt}" in panel.columns:
        panel = panel.rename(columns={f"IV_{tgt}": "IV_SELF"})
    if f"IVRET_{tgt}" in panel.columns:
        panel = panel.rename(columns={f"IVRET_{tgt}": "IVRET_SELF"})

    peer_cols = [c for c in panel.columns if c.startswith("IV_") and c != "IV_SELF"]
    peer_ret_cols = [c for c in panel.columns if c.startswith("IVRET_") and c != "IVRET_SELF"]

    # numeric hygiene (now includes IVRET)
    num_cols = ["opt_volume","time_to_expiry","strike_price","option_type_enc","IV_SELF","IVRET_SELF"] \
            + peer_cols + peer_ret_cols
    for c in num_cols:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(np.float32)

    out[tgt] = panel[["iv_ret_fwd"] + [c for c in num_cols if c in panel.columns]] \
                .dropna(subset=["iv_ret_fwd"]).reset_index(drop=True)

            
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
    tol = pd.Timedelta(tolerance)
    for t, df in cores.items():
        tmp = df[["ts_event", "iv"]].sort_values("ts_event").copy()
        tmp["iv_clip_tmp"] = tmp["iv"].clip(lower=1e-6)
        tmp[f"IV_{t}"] = tmp["iv_clip_tmp"]
        tmp[f"IVRET_{t}"] = np.log(tmp["iv_clip_tmp"]) - np.log(tmp["iv_clip_tmp"].shift(1))
        tmp = tmp[["ts_event", f"IV_{t}", f"IVRET_{t}"]]
        iv_wide = tmp if iv_wide is None else pd.merge_asof(
            iv_wide, tmp, on="ts_event", direction="backward", tolerance=tol
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

# -------------------------------------------------
# Spillover dataset (time-safe) and metrics
# -------------------------------------------------
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def build_spillover_dataset_time_safe(
    tickers: List[str],
    start=None, end=None, r: float = 0.045,
    forward_steps: int = 1, tolerance: str = "2s",
    db_path: Path | str | None = None,
    target_mode: str = "iv_ret",   # "iv_ret" or "iv"
) -> pd.DataFrame:
    """
    Build a pooled dataset for spillover measurement.
    Uses _add_features() to create a 'target' column and engineered features, including peer IVs.
    """
    assert target_mode in {"iv_ret", "iv"}
    dbp = Path(db_path) if db_path else DEFAULT_DB_PATH

    # 1) per-ticker ATM cores
    cores = {t: _atm_core(t, start=start, end=end, r=r, db_path=dbp) for t in tickers}

    # 2) wide IV panel for peers (backward as-of to avoid lookahead)
    iv_wide = None
    tol = pd.Timedelta(tolerance)
    for t, df in cores.items():
        tmp = df[["ts_event", "iv"]].sort_values("ts_event").copy()
        tmp["iv_clip_tmp"] = tmp["iv"].clip(lower=1e-6)
        tmp[f"IV_{t}"] = tmp["iv_clip_tmp"]
        tmp[f"IVRET_{t}"] = np.log(tmp["iv_clip_tmp"]) - np.log(tmp["iv_clip_tmp"].shift(1))
        tmp = tmp[["ts_event", f"IV_{t}", f"IVRET_{t}"]]
        iv_wide = tmp if iv_wide is None else pd.merge_asof(
            iv_wide, tmp, on="ts_event", direction="backward", tolerance=tol
        )


    # 3) build features per target, then pool
    frames = []
    for tgt, dft in cores.items():
        feats = dft.copy()
        # ensure iv_clip and fwd return (used inside _add_features)
        feats["iv_clip"] = feats["iv"].clip(lower=1e-6)
        if target_mode == "iv_ret":
            feats["iv_ret_fwd"] = np.log(feats["iv_clip"].shift(-forward_steps)) - np.log(feats["iv_clip"])

        panel = pd.merge_asof(
            feats.sort_values("ts_event"),
            iv_wide.sort_values("ts_event"),
            on="ts_event", direction="backward",
            tolerance=pd.Timedelta(tolerance),
        )
        # peer columns = other tickers' IVs
        peer_cols = [c for c in panel.columns if c.startswith("IV_") and c != f"IV_{tgt}"]
        peer_ret_cols = [c for c in panel.columns if c.startswith("IVRET_") and c != f"IVRET_{tgt}"]

        X = _add_features(panel,
                        peer_cols=peer_cols,
                        target_ticker=tgt,
                        r=r,
                        target_mode=target_mode,
                        peer_ret_cols=peer_ret_cols)

        X["target_symbol"] = tgt
        frames.append(X)

    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # one-hot symbols if present (some flows add sym_ already; this is a safety net)
    if "target_symbol" in pooled.columns and not any(c.startswith("sym_") for c in pooled.columns):
        pooled = pd.get_dummies(pooled, columns=["target_symbol"], prefix="sym", dtype=float)
    return pooled


def _chrono_split(X: pd.DataFrame, y: np.ndarray, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    n = len(X)
    if n < 10:
        raise ValueError(f"Too few rows: {n}")
    split = int(n * (1 - test_frac))
    return X.iloc[:split], X.iloc[split:], y[:split], y[split:]


def _train_xgb(X_tr: pd.DataFrame, y_tr: np.ndarray, params: dict | None = None) -> xgb.XGBRegressor:
    if params is None:
        params = dict(
            objective="reg:squarederror",
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)
    return model


def _ensure_numeric_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(axis=1, how="all")
    return out


def compute_iv_spillover_metrics(
    pooled: pd.DataFrame,
    test_frac: float = 0.2,
    model_params: dict | None = None,
    metrics_dir: Path | str | None = None,
    prefix: str = "iv_spillover",
) -> dict:
    """
    Quantifies IV spillover on a pooled dataset with column 'target':
      - Train 'controls-only' (no peers) and 'full' (controls + peers) XGB on chrono split.
      - Report R2/RMSE/MAE and ΔR2, ΔRMSE (spillover strength).
      - Report peer contribution share using tree SHAP-like contributions.

    Returns a dict with 'control', 'full', 'summary'. Optionally saves CSV/JSON artifacts.
    """
    if "target" not in pooled.columns:
        raise KeyError("Expected 'target' column in pooled dataset. Build it with build_spillover_dataset_time_safe(...).")

    pooled = pooled.reset_index(drop=True)
    y = pooled["target"].astype(float).values
    X = pooled.drop(columns=["target"]).copy()
    X = _ensure_numeric_all(X)

    # identify peer vs control features
    peer_cols = [c for c in X.columns if c.startswith("IV_") or c.startswith("peer_")]
    control_cols = [c for c in X.columns if c not in peer_cols]

    # chronological split
    Xtr_ctrl, Xte_ctrl, ytr, yte = _chrono_split(X[control_cols], y, test_frac)
    Xtr_full, Xte_full, _,   _   = _chrono_split(X,               y, test_frac)

    # train
    m_ctrl = _train_xgb(Xtr_ctrl, ytr, model_params)
    m_full = _train_xgb(Xtr_full, ytr, model_params)

    # eval
    pred_ctrl = m_ctrl.predict(Xte_ctrl)
    pred_full = m_full.predict(Xte_full)

    def _metrics(y_true, y_pred) -> dict:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        return {"RMSE": rmse, "MAE": mae, "R2": r2, "n": int(len(y_true))}

    met_ctrl = _metrics(yte, pred_ctrl)
    met_full = _metrics(yte, pred_full)

    # SHAP-like contributions for full model
    dtest = xgb.DMatrix(Xte_full, feature_names=Xte_full.columns.tolist())
    contribs = m_full.get_booster().predict(dtest, pred_contribs=True)  # [n, n_features+1], last col = bias
    contribs = np.asarray(contribs)

    # peer contribution share per row
    feat_names = Xte_full.columns.tolist()
    peer_idx = [feat_names.index(c) for c in peer_cols if c in feat_names]
    abs_sum   = np.abs(contribs[:, :-1]).sum(axis=1)  # exclude bias
    peer_sum  = np.abs(contribs[:, peer_idx]).sum(axis=1) if peer_idx else np.zeros(len(Xte_full))
    peer_share = np.divide(peer_sum, np.where(abs_sum == 0.0, 1.0, abs_sum))
    peer_share = np.clip(peer_share, 0.0, 1.0)

    summary = {
        "spillover_R2_gain": float(met_full["R2"] - met_ctrl["R2"]),
        "spillover_RMSE_drop": float(met_ctrl["RMSE"] - met_full["RMSE"]),
        "peer_share_mean": float(np.mean(peer_share)),
        "peer_share_median": float(np.median(peer_share)),
        "peer_share_p90": float(np.quantile(peer_share, 0.90)),
        "peer_share_p95": float(np.quantile(peer_share, 0.95)),
        "n_test": met_full["n"],
        "n_features_full": int(Xtr_full.shape[1]),
        "n_features_control": int(Xtr_ctrl.shape[1]),
        "n_peer_features": int(len(peer_cols)),
    }

    out = {"control": met_ctrl, "full": met_full, "summary": summary}

    # optional: persist artifacts
    if metrics_dir is not None:
        mdir = Path(metrics_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        # metrics JSON
        with open(mdir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        # per-row peer share
        pd.DataFrame({
            "peer_share": peer_share,
            "y_true": yte,
            "yhat_full": pred_full,
            "yhat_ctrl": pred_ctrl,
            "resid_full": yte - pred_full,
            "resid_ctrl": yte - pred_ctrl,
        }).to_csv(mdir / f"{prefix}_peer_share.csv", index=False)
        # importances
        def _xgb_imp_df(model: xgb.XGBRegressor) -> pd.DataFrame:
            bst = model.get_booster()
            types = ["gain", "weight", "cover", "total_gain", "total_cover"]
            frames = []
            for t in types:
                d = bst.get_score(importance_type=t)
                if d:
                    frames.append(pd.DataFrame({"feature": list(d.keys()), t: list(d.values())}))
            if not frames:
                return pd.DataFrame(columns=["feature"])
            imp = frames[0]
            for f in frames[1:]:
                imp = imp.merge(f, on="feature", how="outer")
            return imp.fillna(0.0).sort_values("gain", ascending=False)
        _xgb_imp_df(m_ctrl).to_csv(mdir / f"{prefix}_importances_control.csv", index=False)
        _xgb_imp_df(m_full).to_csv(mdir / f"{prefix}_importances_full.csv", index=False)

    return out


def build_target_peer_dataset(
    target: str,
    tickers: List[str],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    r: float = 0.045,
    forward_steps: int = 1,
    tolerance: str = "2s",
    db_path: Path | str | None = None,
    target_kind: str = "iv_ret",   # "iv_ret" (default) or "iv"
) -> pd.DataFrame:
    """
    Build a single-target dataset to study the effect of all peer IVs on the target.

    Rows are aligned to the target's timestamps; peer IVs are merged 'as of' (backward)
    within the provided tolerance.

    Returns columns:
      - y                      (iv_ret_fwd or iv for the target)
      - IV_SELF                (target IV at t)
      - IV_<peer>              (peer IVs at t, as-of backward merge)
      - opt_volume, time_to_expiry, days_to_expiry, strike_price
      - option_type_enc, hour, minute, day_of_week
      - iv_clip                (safeguarded target IV at t)
    """
    assert target in tickers, "target must be included in tickers"
    dbp = Path(db_path) if db_path else DEFAULT_DB_PATH

    # 1) Load per-ticker ATM cores (these must include 'ts_event' and 'iv' at minimum)
    cores: Dict[str, pd.DataFrame] = {
        t: _atm_core(t, start=start, end=end, r=r, db_path=dbp) for t in tickers
    }

    # 2) Start from target rows (carry option/control fields) and add basic features
    feats = cores[target].copy().sort_values("ts_event").reset_index(drop=True)
    feats["option_type_enc"] = _encode_option_type(feats["option_type"])
    feats["hour"] = feats["ts_event"].dt.hour.astype("int16")
    feats["minute"] = feats["ts_event"].dt.minute.astype("int16")
    feats["day_of_week"] = feats["ts_event"].dt.dayofweek.astype("int16")
    feats["days_to_expiry"] = (feats["time_to_expiry"] * 365.0).astype("float32")
    feats["iv_clip"] = feats["iv"].clip(lower=1e-6)

    if target_kind == "iv_ret":
        feats["y"] = (
            np.log(feats["iv_clip"].shift(-forward_steps)) - np.log(feats["iv_clip"])
        )
    elif target_kind == "iv":
        feats["y"] = feats["iv"]
    else:
        raise ValueError("target_kind must be 'iv_ret' or 'iv'")

    # 3) Merge peers' IVs onto the target timeline (as-of backward within tolerance)
    panel = feats.sort_values("ts_event")
    tol = pd.Timedelta(tolerance)

    for t, df in cores.items():
        tmp = (
            df[["ts_event", "iv"]]
            .rename(columns={"iv": f"IV_{t}"})
            .sort_values("ts_event")
        )
        panel = pd.merge_asof(
            panel, tmp, on="ts_event", direction="backward", tolerance=tol
        )

    for t, df in cores.items():
        tmp = (
            df[["ts_event", "iv"]].sort_values("ts_event").copy()
        )
        tmp["iv_clip_tmp"] = tmp["iv"].clip(lower=1e-6)
        tmp[f"IV_{t}"] = tmp["iv_clip_tmp"]
        tmp[f"IVRET_{t}"] = np.log(tmp["iv_clip_tmp"]) - np.log(tmp["iv_clip_tmp"].shift(1))
        tmp = tmp[["ts_event", f"IV_{t}", f"IVRET_{t}"]]
        panel = pd.merge_asof(panel, tmp, on="ts_event", direction="backward", tolerance=tol)

    # rename target’s own
    if f"IV_{target}" in panel.columns:
        panel = panel.rename(columns={f"IV_{target}": "IV_SELF"})
    if f"IVRET_{target}" in panel.columns:
        panel = panel.rename(columns={f"IVRET_{target}": "IVRET_SELF"})

    peer_cols = [c for c in panel.columns if c.startswith("IV_") and c != "IV_SELF"]
    peer_ret_cols = [c for c in panel.columns if c.startswith("IVRET_") and c != "IVRET_SELF"]

    numeric_cols = ["opt_volume","time_to_expiry","days_to_expiry","strike_price","IV_SELF","IVRET_SELF"] \
                + peer_cols + peer_ret_cols
    for c in numeric_cols:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce").astype(np.float32)

    final_cols = (
        ["y", "IV_SELF", "IVRET_SELF"]
        + peer_cols + peer_ret_cols
        + ["opt_volume","time_to_expiry","days_to_expiry","strike_price",
        "option_type_enc","hour","minute","day_of_week","iv_clip"]
    )
    ds = panel[[c for c in final_cols if c in panel.columns]].dropna(subset=["y"]).reset_index(drop=True)
    return ds
