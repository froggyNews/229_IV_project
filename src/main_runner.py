# main_runner.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# optional: pick up IV_DB_PATH etc.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from feature_engineering import build_pooled_iv_return_dataset_time_safe
from model_evaluation import evaluate_pooled_model


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train & evaluate pooled IV models (returns + level).")
    ap.add_argument("--db", type=Path,
                    default=Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db")),
                    help="Path to SQLite DB (defaults to IV_DB_PATH or data/iv_data_1m.db).")
    ap.add_argument("--tickers", nargs="+", required=True, help="Tickers to include.")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC).")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC).")
    ap.add_argument("--test-frac", type=float, default=0.2, help="Test fraction (chronological split).")
    ap.add_argument("--forward-steps", type=int, default=1, help="Forward steps for iv_ret_fwd target.")
    ap.add_argument("--tolerance", default="2s", help="As-of merge tolerance (e.g. '0s','2s').")
    ap.add_argument("--metrics-path", type=Path, default=Path("metrics/iv_metrics.json"),
                    help="Path to write combined metrics JSON.")
    ap.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory to save models.")
    ap.add_argument("--prefix", type=str, default="iv_returns_pooled",
                    help="Filename prefix for saved models/outputs.")
    ap.add_argument("--evaluate-model", type=Path, default=None,
                    help="Optional path to a pre-trained model to evaluate instead of newly trained ones.")
    return ap.parse_args()


# ---------------------------
# Training
# ---------------------------
def train_xgb(
    pooled: pd.DataFrame,
    target: str,
    test_frac: float = 0.2,
    drop_cols: list[str] | None = None,
    params: dict | None = None,
) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train XGBRegressor on pooled dataset, return (model, metrics).
    - pooled: DataFrame containing target + numeric features
    - target: 'iv_ret_fwd' or 'iv_clip'
    """
    drop_cols = drop_cols or []
    if target not in pooled.columns:
        raise KeyError(f"Target '{target}' not found in columns: {list(pooled.columns)}")

    pooled = pooled.sort_index().reset_index(drop=True)

    # target/feature split
    y = pooled[target].astype(float).values
    feat_cols = [c for c in pooled.columns if c not in [target] + drop_cols]
    X = pooled[feat_cols].astype(float)

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
            n_jobs=0,
            tree_method="hist",
        )

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    metrics = dict(
        RMSE=float(np.sqrt(mean_squared_error(y_te, pred))),
        R2=float(r2_score(y_te, pred)),
        n_train=int(len(X_tr)),
        n_test=int(len(X_te)),
        features=feat_cols,
    )
    return model, metrics


# ---------------------------
# Orchestration
# ---------------------------
def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    # Build pooled dataset once
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
    print(f"[POOLED] rows={len(pooled):,}, columns={pooled.shape[1]}")

    # Train iv_ret_fwd
    ret_model, m_ret = train_xgb(
        pooled, target="iv_ret_fwd", test_frac=args.test_frac
    )

    # Train iv_clip (drop iv_ret_fwd to avoid any leakage)
    clip_model, m_clip = train_xgb(
        pooled, target="iv_clip", test_frac=args.test_frac, drop_cols=["iv_ret_fwd"]
    )

    # Save metrics
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump({"iv_ret_fwd": m_ret, "iv_clip": m_clip}, f, indent=2)
    print(f"[METRICS] saved → {args.metrics_path}")

    # Save models
    args.models_dir.mkdir(parents=True, exist_ok=True)
    ret_path = args.models_dir / f"{args.prefix}_ret.json"
    clip_path = args.models_dir / f"{args.prefix}_clip.json"
    ret_model.save_model(ret_path.as_posix())
    clip_model.save_model(clip_path.as_posix())
    print(f"[SAVE] iv_ret_fwd model → {ret_path}")
    print(f"[SAVE] iv_clip    model → {clip_path}")

    # Evaluate
    eval_targets: list[tuple[str, Path]] = []
    if args.evaluate_model is not None:
        eval_targets.append(("custom", args.evaluate_model))
    else:
        eval_targets.append(("iv_ret_fwd", ret_path))
        eval_targets.append(("iv_clip", clip_path))

    for tag, model_path in eval_targets:
        evaluate_pooled_model(
            model_path=model_path,
            tickers=args.tickers,
            start=args.start,
            end=args.end,
            test_frac=args.test_frac,
            forward_steps=args.forward_steps,
            tolerance=args.tolerance,
            metrics_dir=args.metrics_path.parent,
            outputs_prefix=f"{args.prefix}_{tag}",
            save_predictions=False,
            perm_repeats=0,
            db_path=args.db,
        )
        print(f"[EVAL] done → {tag} ({model_path})")


if __name__ == "__main__":
    main()
