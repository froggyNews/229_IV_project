# train_iv_returns.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from feature_engineering import (
    build_pooled_iv_return_dataset_time_safe,
)
# ---------- Train (pooled) ----------
def train_xgb_iv_returns_time_safe_pooled(
    pooled: pd.DataFrame,
    test_frac: float = 0.2,
    params: dict | None = None,
) -> Tuple[xgb.XGBRegressor, dict]:
    pooled = pooled.sort_index().reset_index(drop=True)
    n = len(pooled)
    if n < 100:
        raise ValueError(f"Too few rows: {n}")
    split = int(n * (1 - test_frac))
    train_df, test_df = pooled.iloc[:split], pooled.iloc[split:]
    y_tr = train_df["iv_ret_fwd"].astype(float).values
    y_te = test_df["iv_ret_fwd"].astype(float).values
    X_tr = train_df.drop(columns=["iv_ret_fwd"]).astype(float)
    X_te = test_df.drop(columns=["iv_ret_fwd"]).astype(float)

    if params is None:
        params = dict(objective="reg:squarederror",
                      n_estimators=350, learning_rate=0.05,
                      max_depth=6, subsample=0.9, colsample_bytree=0.9,
                      random_state=42)
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = dict(RMSE=float(np.sqrt(mean_squared_error(y_te, pred))),
                   R2=float(r2_score(y_te, pred)),
                   n_train=int(len(X_tr)), n_test=int(len(X_te)))
    return model, metrics

def save_model_and_metrics(model: xgb.XGBRegressor, metrics: dict,
                           model_path: Path, metrics_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--forward-steps", type=int, default=1)
    ap.add_argument("--tolerance", default="2s")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--metrics-dir", type=Path, default=Path("metrics"))
    ap.add_argument("--prefix", default="iv_returns_pooled")
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end   = pd.Timestamp(args.end, tz="UTC")

    pooled = build_pooled_iv_return_dataset_time_safe(
        tickers=args.tickers, start=start, end=end,
        forward_steps=args.forward_steps, tolerance=args.tolerance,
        db_path=args.db
    )
    if pooled.empty:
        raise RuntimeError("Pooled dataset is empty.")

    model, metrics = train_xgb_iv_returns_time_safe_pooled(
        pooled, test_frac=args.test_frac
    )
    stamp = f"{start:%Y%m%d}_{end:%Y%m%d}"
    model_path   = args.models_dir / f"{args.prefix}_{stamp}.json"
    metrics_path = args.metrics_dir / f"{args.prefix}_{stamp}.json"
    save_model_and_metrics(model, metrics, model_path, metrics_path)

    print(f"[TRAIN] rows={len(pooled)}, features={pooled.shape[1]-1}")
    print(f"[TRAIN] RMSE={metrics['RMSE']:.6f}  R²={metrics['R2']:.3f}")
    print(f"[SAVED] {model_path}")
    print(f"[SAVED] {metrics_path}")

if __name__ == "__main__":
    main()
