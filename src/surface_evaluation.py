from __future__ import annotations

"""Evaluate XGBoost models trained on surface-tensor features.

Builds the same surface-tensor dataset used for training across a time window,
performs a time-respecting train/test split, aligns features to the model, and
computes RMSE and R2. Optionally writes a single JSON report to disk.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

from .surface_dataset import build_surface_tensor_dataset


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    non_num = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_num:
        df = df.drop(columns=non_num)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(float)
    return df


def _align_to_model(model: xgb.XGBRegressor, X: pd.DataFrame) -> pd.DataFrame:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return X
    names = list(names)
    # if names are generic f0,f1,... skip strict alignment
    if names and all(n.startswith("f") and n[1:].isdigit() for n in names):
        return X
    # add missing columns with zeros
    missing = [c for c in names if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    return X.reindex(columns=names)


def evaluate_surface_model(
    model_path: str | Path,
    tickers: List[str],
    start: str,
    end: str,
    test_frac: float = 0.2,
    forward_steps: int = 1,
    tolerance: str = "30s",
    k_bins: int = 10,
    t_bins: int = 10,
    agg: str = "median",
    db_path: str | Path = Path("data/iv_data_1m.db"),
    metrics_dir: str | Path | None = "metrics",
    outputs_prefix: str = "surface_eval",
    target_col: str | None = None,
    save_report: bool = True,
) -> Dict:
    # Build dataset
    df = build_surface_tensor_dataset(
        tickers=tickers,
        start=start,
        end=end,
        db_path=db_path,
        k_bins=k_bins,
        t_bins=t_bins,
        agg=agg,
        forward_steps=forward_steps,
        tolerance=tolerance,
    )
    if df is None or df.empty:
        raise ValueError("Empty surface dataset for evaluation")

    # One-hot symbol and select features
    if "symbol" in df.columns:
        df = pd.get_dummies(df, columns=["symbol"], prefix="sym", dtype=float)
    if target_col is None:
        name = Path(model_path).name.lower()
        if "ret" in name:
            target_col = "iv_ret_fwd"
        elif "clip" in name or "level" in name:
            target_col = "iv_clip"
        else:
            raise ValueError("Cannot infer target_col from model name; pass target_col explicitly")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not in dataset")

    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[c for c in df.columns if c == target_col or c == "ts_event"]).copy()
    X = _ensure_numeric(X)

    # Split by time order
    n = len(X)
    if n < 10:
        raise ValueError(f"Too few rows for evaluation: {n}")
    split = int(n * (1 - test_frac))
    X_tr, X_te = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_tr, y_te = y.iloc[:split].copy(), y.iloc[split:].copy()

    # Load model
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    # Align features
    X_tr = _align_to_model(model, X_tr)
    X_te = _align_to_model(model, X_te)

    # Predict and metrics
    y_pred = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    r2 = float(r2_score(y_te, y_pred))

    metrics = {
        "RMSE": rmse,
        "R2": r2,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features": int(X_tr.shape[1]),
        "tickers": list(tickers),
        "start": pd.Timestamp(start, tz="UTC").isoformat(),
        "end": pd.Timestamp(end, tz="UTC").isoformat(),
        "forward_steps": forward_steps,
        "tolerance": tolerance,
        "test_frac": test_frac,
        "model_path": str(model_path),
    }

    evaluation = {"metrics": metrics}

    if save_report and metrics_dir is not None:
        md = Path(metrics_dir)
        md.mkdir(parents=True, exist_ok=True)
        out = md / f"{outputs_prefix}_evaluation.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2)
        evaluation["eval_path"] = str(out)

    return evaluation

