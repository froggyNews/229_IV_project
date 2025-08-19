# src/evaluate_model.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import os 
from feature_engineering import build_pooled_iv_return_dataset_time_safe


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any non-numeric features (xgboost requirement)."""
    non_num = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_num:
        print(f"[WARN] Dropping non-numeric columns: {non_num}")
        df = df.drop(columns=non_num)
    return df


def _align_columns_to_model(model: xgb.XGBRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """Align feature columns to what the model expects (by name)."""
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        try:
            expected = model.get_booster().feature_names
        except Exception:
            expected = None

    if expected.any():
        # add any missing columns as 0, drop extras
        for col in expected:
            if col not in X.columns:
                X[col] = 0.0
        X = X.reindex(columns=expected)
    return X


def _xgb_importances(model: xgb.XGBRegressor) -> pd.DataFrame:
    """Collect XGBoost gain/weight/cover importances."""
    bst = model.get_booster()
    types = ["gain", "weight", "cover", "total_gain", "total_cover"]
    imp = None
    for t in types:
        try:
            s = bst.get_score(importance_type=t)  # dict: feature -> score
        except Exception:
            s = {}
        if not s:
            continue
        df = pd.DataFrame(list(s.items()), columns=["feature", t])
        imp = df if imp is None else imp.merge(df, on="feature", how="outer")
    if imp is None:
        imp = pd.DataFrame(columns=["feature"])
    for t in types:
        if t not in imp.columns:
            imp[t] = 0.0
    # Sort by gain as a sensible default
    imp = imp.fillna(0.0).sort_values("gain", ascending=False).reset_index(drop=True)
    return imp


def evaluate_pooled_model(
    model_path: str | Path,
    tickers: list[str],
    start: str,
    end: str,
    test_frac: float = 0.2,
    forward_steps: int = 1,
    tolerance: str = "2s",
    r: float = 0.045,
    metrics_dir: str | Path = "metrics",
    outputs_prefix: str = "iv_returns_pooled_eval",
    save_predictions: bool = True,
    perm_repeats: int = 5,
    perm_sample: int | None = 5000,
):
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build pooled dataset (target first: 'iv_ret_fwd') ----
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    pooled = build_pooled_iv_return_dataset_time_safe(
        tickers=tickers,
        start=start_ts,
        end=end_ts,
        r=r,
        forward_steps=forward_steps,
        tolerance=tolerance,
        db_path = Path(args.db)
    )
    print(f"[DATA] pooled rows={len(pooled)}, features={pooled.shape[1]-1}")
    # Debugging iv_clip creation

    print(f"Pooled DataFrame columns: {pooled.columns}")
    print(f"Pooled DataFrame shape: {pooled.shape}")

    if "iv_clip" not in pooled.columns:
        raise KeyError("'iv_clip' column is missing in the pooled DataFrame. Debug the dataset creation process.")
        
    y = pooled["iv_clip"].astype(float)
    X = pooled.drop(columns=["iv_clip"])
    X = _ensure_numeric(X)

    # ---- Chronological split (same as training) ----
    n = len(X)
    if n < 10:
        raise ValueError(f"Too few rows for evaluation: {n}")
    split_idx = int(n * (1 - test_frac))
    X_tr, X_te = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_tr, y_te = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    # ---- Load model & align columns ----
# ---- Load model & align columns ----
    model_path = _resolve_model_path(model_path, fallback_dir="models")
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    X_tr = _align_columns_to_model(model, X_tr)
    X_te = _align_columns_to_model(model, X_te)

    # ---- Predict & metrics ----
    y_pred = model.predict(X_te)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "R2": float(r2_score(y_te, y_pred)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": int(X_tr.shape[1]),
        "tickers": tickers,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "forward_steps": forward_steps,
        "tolerance": tolerance,
        "test_frac": test_frac,
    }
    print(f"[METRICS] RMSE={metrics['RMSE']:.6f}  R²={metrics['R2']:.3f}")

    # ---- XGBoost importances ----
    xgb_imp = _xgb_importances(model)

    # ---- Permutation importance (on a sample for speed, optional) ----
    perm_df = pd.DataFrame()
    if perm_repeats and perm_repeats > 0:
        if (perm_sample is not None) and (len(X_te) > perm_sample):
            Xp = X_te.sample(perm_sample, random_state=42)
            yp = y_te.loc[Xp.index]
        else:
            Xp, yp = X_te, y_te
        perm = permutation_importance(
            model, Xp, yp,
            n_repeats=perm_repeats,
            random_state=42,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
        perm_df = pd.DataFrame({
            "feature": Xp.columns,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
        }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    # ---- Save artifacts ----
    metrics_path = metrics_dir / f"{outputs_prefix}_metrics.json"
    xgb_imp_path = metrics_dir / f"{outputs_prefix}_xgb_importances.csv"
    perm_imp_path = metrics_dir / f"{outputs_prefix}_perm_importances.csv"
    preds_path   = metrics_dir / f"{outputs_prefix}_predictions.csv"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    xgb_imp.to_csv(xgb_imp_path, index=False)
    if not perm_df.empty:
        perm_df.to_csv(perm_imp_path, index=False)

    if save_predictions:
        pd.DataFrame({
            "y_true": y_te.values,
            "y_pred": y_pred,
            "resid": y_te.values - y_pred,
        }).to_csv(preds_path, index=False)

    print(f"[SAVED] {metrics_path}")
    print(f"[SAVED] {xgb_imp_path}")
    if not perm_df.empty:
        print(f"[SAVED] {perm_imp_path}")
    if save_predictions:
        print(f"[SAVED] {preds_path}")

from pathlib import Path

def _resolve_model_path(model_path: str | Path, fallback_dir: str | Path = "models") -> Path:
    p = Path(model_path)
    if p.exists():
        return p
    fb = Path(fallback_dir)
    if fb.exists():
        cand = sorted(fb.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if cand:
            print(f"[INFO] Model not found at {p}. Using latest in {fb}: {cand[0].name}")
            return cand[0]
    raise FileNotFoundError(f"Model not found at {p} and no JSON models in {fb}")

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate pooled IV-returns XGBoost model.")
    p.add_argument("--model", required=True, help="Path to saved XGBoost model (JSON).")
    p.add_argument("--tickers", nargs="+", default=["QBTS", "IONQ", "RGTI", "QUBT"])
    p.add_argument("--start", default="2025-01-02")
    p.add_argument("--end",   default="2025-01-06")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--forward-steps", type=int, default=1)
    p.add_argument("--tolerance", default="2s")
    p.add_argument("--r", type=float, default=0.045)
    p.add_argument("--metrics-dir", default="metrics")
    p.add_argument("--prefix", default="iv_returns_pooled_eval")
    p.add_argument("--no-preds", action="store_true", help="Do not save per-row predictions.")
    p.add_argument("--perm-repeats", type=int, default=5, help="Permutation importance repeats (0 to disable).")
    p.add_argument("--perm-sample", type=int, default=5000, help="Cap permutation sample size for speed.")
    p.add_argument("--db", type=str, default=os.getenv("IV_DB_PATH", "data/iv_data_1m.db"),
                    help="Path to the SQLite DB to read from.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_pooled_model(
        model_path=args.model,
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        test_frac=args.test_frac,
        forward_steps=args.forward_steps,
        tolerance=args.tolerance,
        r=args.r,
        metrics_dir=args.metrics_dir,
        outputs_prefix=args.prefix,
        save_predictions=not args.no_preds,
        perm_repeats=args.perm_repeats,
        perm_sample=args.perm_sample,
    )
