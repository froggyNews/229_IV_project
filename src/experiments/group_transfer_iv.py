from __future__ import annotations
import sys
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from scipy.stats import spearmanr, t
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_object_dtype,
)

# Add the parent directory (src) to Python path for module imports
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering import build_pooled_iv_return_dataset_time_safe
from data_loader_coordinator import load_cores_with_auto_fetch



@dataclass
class ExpConfig:
    groups: Dict[str, List[str]]
    db_path: str
    start: str
    end: str
    forward_steps: int
    output_dir: str
    xgb_params: Dict
    target: str = "iv_ret_fwd"  # default target column
    n_splits: int = 4
    r: float = 0.045
    tolerance: str = "2s"
    cores: Optional[Dict[str, pd.DataFrame]] = None
    auto_fetch: bool = True


def _time_cv_splits(n: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining TimeSeriesSplit wrapper.

    Parameters
    ----------
    n : int
        Number of observations available.
    n_splits : int
        Desired number of splits.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of ``(train_index, test_index)`` tuples.

    Raises
    ------
    ValueError
        If ``n`` is not large enough to accommodate ``n_splits`` folds.
    """

    if n <= n_splits:
        raise ValueError(
            f"Not enough samples ({n}) to create {n_splits} splits for TimeSeriesSplit"
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(np.arange(n)))


def _smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    return np.mean(2.0 * np.abs(y_pred - y_true) / np.where(denom == 0, 1.0, denom))


def _to_utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _diebold_mariano(e1: np.ndarray, e2: np.ndarray, power: int = 2) -> Tuple[float, float]:
    """Simple Diebold-Mariano test for equal predictive accuracy."""
    d = np.abs(e1) ** power - np.abs(e2) ** power
    d_mean = d.mean()
    d_var = d.var(ddof=1)
    if d_var == 0:
        return np.nan, 1.0
    stat = d_mean / math.sqrt(d_var / len(d))
    df = len(d) - 1
    p = 2 * (1 - t.cdf(np.abs(stat), df))
    return stat, p


def _fit_eval_one(X, y, params, splits):
    """Train once with forward-chaining CV; return OOS preds & feature importance."""
    X_proc = X.copy()

    for col in X_proc.columns:
        if is_datetime64_any_dtype(X_proc[col]):
            X_proc[col] = X_proc[col].astype("int64")
        elif is_object_dtype(X_proc[col]):
            X_proc[col] = X_proc[col].astype("category")
    enable_cat = any(
        isinstance(X_proc[c].dtype, pd.CategoricalDtype) for c in X_proc.columns
    )
    if enable_cat:
        params = {**params, "enable_categorical": True}

    model = XGBRegressor(**params)
    oos_idx = np.zeros(len(y), dtype=bool)
    oos_pred = np.full(len(y), np.nan)
    last_fold = None

    for tr, te in splits:
        model.fit(X_proc.iloc[tr], y.iloc[tr])
        yhat = model.predict(X_proc.iloc[te])
        oos_pred[te] = yhat
        oos_idx[te] = True
        last_fold = (tr, te)

    tr, te = last_fold
    perm = permutation_importance(
        model, X_proc.iloc[te], y.iloc[te], n_repeats=10, random_state=42, scoring="r2"
    )
    perm_imp = pd.Series(perm.importances_mean, index=X_proc.columns)

    shap_vals = model.get_booster().predict(
        xgb.DMatrix(X_proc.iloc[te], enable_categorical=enable_cat), pred_contribs=True
    )
    shap_abs_mean = pd.Series(
        np.abs(shap_vals[:, :-1]).mean(axis=0), index=X_proc.columns
    )

    metrics = {
        "r2": r2_score(y[oos_idx], pd.Series(oos_pred[oos_idx], index=y.index[oos_idx])),
        "mae": mean_absolute_error(y[oos_idx], oos_pred[oos_idx]),
        "smape": _smape(y[oos_idx].values, oos_pred[oos_idx]),
    }
    return model, pd.Series(oos_pred, index=y.index), metrics, perm_imp, shap_abs_mean


def _prep_group_frame(df: pd.DataFrame, tickers: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    g = df[df["ticker"].isin(tickers)].copy()
    drop = [c for c in ["asof_date", "expiry", "quote_id", "iv_theo", "future_iv"] if c in g.columns]
    X = g.drop(columns=[target_col] + drop, errors="ignore")
    y = g[target_col]
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    for col in X.columns:
        if is_datetime64_any_dtype(X[col]):
            X[col] = X[col].astype("int64")
        elif is_object_dtype(X[col]):
            X[col] = X[col].astype("category")
    valid = ~X.isna().any(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]


def run_experiment(cfg: ExpConfig):
    """Run the group transfer learning experiment."""
    
    print("🚀 Starting Group Transfer Learning Experiment")
    print(f"📊 Groups: {cfg.groups}")
    print(f"📅 Date range: {cfg.start} to {cfg.end}")

    start_ts = _to_utc(cfg.start)
    end_ts = _to_utc(cfg.end)

    # Get all tickers from all groups
    all_tickers = sum(cfg.groups.values(), [])
    print(f"📈 All tickers: {all_tickers}")

    cores = cfg.cores
    if cores is None:
        cores = load_cores_with_auto_fetch(
            all_tickers, start_ts, end_ts, Path(cfg.db_path), auto_fetch=cfg.auto_fetch
        )

    # Build pooled dataset
    print("🔗 Building pooled dataset...")
    df = build_pooled_iv_return_dataset_time_safe(
        tickers=all_tickers,
        start=start_ts,
        end=end_ts,
        r=cfg.r,
        forward_steps=cfg.forward_steps,
        tolerance=cfg.tolerance,
        db_path=cfg.db_path,
        cores=cores,
    )
    
    if df.empty:
        raise ValueError("No data found for the specified tickers and date range")

    print(f"📊 Dataset shape: {df.shape}")

    # Derive ticker column from one-hot symbol columns if needed
    sym_cols = [c for c in df.columns if c.startswith("sym_")]
    if sym_cols:
        zero_rows = df[sym_cols].sum(axis=1) == 0
        if zero_rows.any():
            print(f"⚠️  Dropping {int(zero_rows.sum())} rows without ticker encoding")
            df = df.loc[~zero_rows]
    if sym_cols and "ticker" not in df.columns:
        df["ticker"] = df[sym_cols].idxmax(axis=1).str.replace("sym_", "")

    if "ticker" not in df.columns:
        raise ValueError("Dataset lacks ticker information")

    print(f"📊 Available tickers: {sorted(df['ticker'].unique().tolist())}")

    # Continue with rest of experiment...

    for col in ["ticker", cfg.target]:
        if col not in df.columns:
            raise ValueError(f"Required column {col} missing from dataset")

    # Prepare data for each group and track feature sets
    group_frames: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    feat_sets: List[set] = []
    for name, tickers in cfg.groups.items():
        X, y = _prep_group_frame(df, tickers, cfg.target)
        if len(y) == 0:
            print(f"⚠️  Skipping {name}: no samples for tickers {tickers}")
            continue
        group_frames[name] = (X, y)
        feat_sets.append(set(X.columns))

    if not group_frames:
        raise ValueError("No groups have data for the specified tickers and date range")

    base_feats = sorted(set.intersection(*feat_sets)) if feat_sets else []
    for k, (X, y) in group_frames.items():
        group_frames[k] = (X[base_feats], y)

    # Filter out groups without enough samples for the desired number of splits
    valid_frames: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for name, (X, y) in group_frames.items():
        if len(y) <= cfg.n_splits:
            print(
                f"⚠️  Skipping {name}: only {len(y)} samples (need > {cfg.n_splits} for CV)"
            )
            continue
        valid_frames[name] = (X, y)

    if not valid_frames:
        raise ValueError("No groups have enough samples for cross-validation")

    group_frames = valid_frames
    splits = {name: _time_cv_splits(len(y), cfg.n_splits) for name, (X, y) in group_frames.items()}

    results = []
    imps = []
    shap_rows = []

    models = {}
    oos_preds = {}
    for name, (X, y) in group_frames.items():
        m, yhat_oos, met, perm, shap_abs = _fit_eval_one(X, y, cfg.xgb_params, splits[name])
        models[name] = m
        oos_preds[name] = (y, yhat_oos)
        results.append({"train_group": name, "test_group": name, **met})
        for f, v in perm.items():
            imps.append({"model": name, "feature": f, "perm_importance": v})
        for f, v in shap_abs.items():
            shap_rows.append({"model": name, "feature": f, "shap_abs_mean": v})

    for tr_name in group_frames:
        Xtr, ytr = group_frames[tr_name]
        m = models[tr_name]
        for te_name in group_frames:
            if te_name == tr_name:
                continue
            Xte, yte = group_frames[te_name]
            yhat = m.predict(Xte)
            met = {
                "r2": r2_score(yte, yhat),
                "mae": mean_absolute_error(yte, yhat),
                "smape": _smape(yte.values, yhat),
            }
            if te_name in oos_preds:
                y_te_oos, yhat_te_oos = oos_preds[te_name]
                ix = yte.index.intersection(yhat_te_oos.index)
                if len(ix) > 50:
                    e_cross = yte.loc[ix].values - m.predict(Xte.loc[ix])
                    e_own = y_te_oos.loc[ix].values - yhat_te_oos.loc[ix].values
                    dm_stat, dm_p = _diebold_mariano(e_cross, e_own)
                    met["dm_stat_vs_own"] = dm_stat
                    met["dm_p_vs_own"] = dm_p
            results.append({"train_group": tr_name, "test_group": te_name, **met})

    fi = pd.DataFrame(imps).pivot_table(index="feature", columns="model", values="perm_importance")
    sh = pd.DataFrame(shap_rows).pivot_table(index="feature", columns="model", values="shap_abs_mean")
    sim_report = {}
    models_list = list(group_frames.keys())
    if len(models_list) >= 2:
        a, b = models_list[0], models_list[1]
        ra = fi[a].rank(ascending=False)
        rb = fi[b].rank(ascending=False)
        rho_fi, p_fi = spearmanr(ra, rb, nan_policy="omit")
        rs = sh[[a, b]].dropna()
        rho_shap, p_shap = spearmanr(rs[a], rs[b], nan_policy="omit")
        sim_report = {
            "perm_rank_spearman": {"pairs": [a, b], "rho": float(rho_fi), "p": float(p_fi)},
            "shap_spearman": {"pairs": [a, b], "rho": float(rho_shap), "p": float(p_shap)},
        }

    res_df = pd.DataFrame(results)
    imp_df = pd.DataFrame(imps)
    shap_df = pd.DataFrame(shap_rows)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


    res_df.to_csv(Path(cfg.output_dir, f"transfer_matrix_{timestamp}.csv"), index=False)
    imp_df.to_csv(Path(cfg.output_dir, f"feature_importance_{timestamp}.csv"), index=False)
    shap_df.to_csv(Path(cfg.output_dir, f"shap_summary_{timestamp}.csv"), index=False)
    with open(Path(cfg.output_dir, f"similarity_report_{timestamp}.json"), "w") as f:
        json.dump(sim_report, f, indent=2)

    return res_df, imp_df, shap_df, sim_report

if __name__ == "__main__":
    # Example usage
    config = ExpConfig(
        groups={
            "group1": ["QUBT", "RGTI"],
            "group2": ["QBTS", "IONQ"]
        },
        db_path="data/iv_data_1h.db",
        start="2025-08-02",
        end="2025-08-06",
        forward_steps=5,
        output_dir="output",
        xgb_params={
            "objective": "reg:squarederror",
            "n_estimators": 350,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42
        }
    )
    run_experiment(config)
