from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

from feature_engineering import build_pooled_iv_return_dataset_time_safe


def base_arguments() -> argparse.ArgumentParser:
    """Define base arguments for the script."""
    ap = argparse.ArgumentParser(description="Run models for iv_ret_fwd and iv_clip")
    ap.add_argument("--db", required=True, type=Path, help="Path to the database file.")
    ap.add_argument("--tickers", nargs="+", required=True, help="List of tickers.")
    ap.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    ap.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    ap.add_argument("--test-frac", type=float, default=0.2, help="Fraction of data for testing.")
    ap.add_argument("--forward-steps", type=int, default=1, help="Number of forward steps.")
    ap.add_argument("--tolerance", default="2s", help="Tolerance for time alignment.")
    ap.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("metrics/iv_metrics.json"),
        help="Path to save metrics JSON file.",
    )
    return ap


def train_xgb(
    pooled: pd.DataFrame,
    target: str,
    test_frac: float = 0.2,
    drop_cols: list[str] | None = None,
    params: dict | None = None,
):
    """Train XGBoost model on given target and return metrics."""
    pooled = pooled.sort_index().reset_index(drop=True)
    drop_cols = drop_cols or []

    y = pooled[target].astype(float).values
    features = [c for c in pooled.columns if c not in [target] + drop_cols]
    X = pooled[features].astype(float)

    n = len(X)
    if n < 10:
        raise ValueError(f"Too few rows: {n}")
    split = int(n * (1 - test_frac))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split], y[split:]

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
    pred = model.predict(X_te)
    metrics = dict(
        RMSE=float(np.sqrt(mean_squared_error(y_te, pred))),
        R2=float(r2_score(y_te, pred)),
        n_train=int(len(X_tr)),
        n_test=int(len(X_te)),
    )
    return model, metrics


def main() -> None:
    ap = base_arguments()
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    pooled = build_pooled_iv_return_dataset_time_safe(
        tickers=args.tickers,
        start=start,
        end=end,
        forward_steps=args.forward_steps,
        tolerance=args.tolerance,
        db_path=args.db,
    )
    if pooled.empty:
        raise RuntimeError("Pooled dataset is empty.")

    metrics: dict[str, dict] = {}

    # Train on iv_ret_fwd (iv_clip kept as feature)
    _, m_ret = train_xgb(pooled, "iv_ret_fwd", test_frac=args.test_frac)
    metrics["iv_ret_fwd"] = m_ret

    # Train on iv_clip (drop iv_ret_fwd to avoid leakage)
    _, m_clip = train_xgb(
        pooled,
        "iv_clip",
        test_frac=args.test_frac,
        drop_cols=["iv_ret_fwd"],
    )
    metrics["iv_clip"] = m_clip

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[iv_ret_fwd] RMSE={m_ret['RMSE']:.6f}  R²={m_ret['R2']:.3f}")
    print(f"[iv_clip]    RMSE={m_clip['RMSE']:.6f}  R²={m_clip['R2']:.3f}")
    print(f"[SAVED] {args.metrics_path}")

import argparse

def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description="Run main_runner for iv_ret_fwd and iv_clip")
    ap.add_argument("--db", required=True, type=str, help="Path to the database file.")
    ap.add_argument("--tickers", nargs="+", required=True, help="List of tickers.")
    ap.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    ap.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    ap.add_argument("--test-frac", type=float, default=0.2, help="Fraction of data for testing.")
    ap.add_argument("--forward-steps", type=int, default=1, help="Number of forward steps.")
    ap.add_argument("--tolerance", default="2s", help="Tolerance for time alignment.")
    ap.add_argument(
        "--metrics-path",
        type=str,
        default="metrics/iv_metrics.json",
        help="Path to save metrics JSON file.",
    )
    return ap.parse_args()

def main():
    args = parse_args()
    print(f"Arguments parsed: {args}")
    # Add your main logic here

if __name__ == "__main__":
    main()
# if __name__ == "__main__":
#     def main():
#         args = parse_args()
#         evaluate_pooled_model(
#             model_path=args.model,
#             tickers=args.tickers,
#             start=args.start,
#             end=args.end,
#             test_frac=args.test_frac,
#             forward_steps=args.forward_steps,
#             tolerance=args.tolerance,
#             r=args.r,
#             metrics_dir=args.metrics_dir,
#             outputs_prefix=args.prefix,
#             save_predictions=not args.no_preds,
#             perm_repeats=args.perm_repeats,
#             perm_sample=args.perm_sample,
#         )

#     main()
    