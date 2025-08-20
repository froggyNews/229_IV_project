"""
Utility to evaluate saved XGBoost models on pooled IV return data.

Loads a previously trained XGBoost model and evaluates it on the pooled data
produced by `build_pooled_iv_return_dataset_time_safe`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

from feature_engineering import build_pooled_iv_return_dataset_time_safe


# -----------------------------
# Utilities
# -----------------------------

def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-numeric columns; cast bool -> float. (XGBoost wants numeric)."""
    # Drop truly non-numeric (object, category, etc.)
    non_num = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_num:
        print(f"[WARN] Dropping non-numeric columns: {non_num}")
        df = df.drop(columns=non_num)

    # Cast bool -> float for safety
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(float)

    return df


def _expected_feature_names(model):
    """Retrieve the expected feature names from the model."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return []
    return list(names)


def _looks_like_generic_xgb_names(names: Iterable[str]) -> bool:
    """True if names are like ['f0', 'f1', ...]."""
    names = list(names)
    if not names:
        return False
    return all(n.startswith("f") and n[1:].isdigit() for n in names)


def _align_columns_to_model(model: xgb.XGBRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """
    Align feature columns to those used by the model.
    - If the model has meaningful feature names, add any missing columns (0.0) and reorder.
    - If the model only has generic 'f0' names, DO NOT reindex (we would
      likely break alignment). In that case we leave X as-is and print a note.
    """
    expected = _expected_feature_names(model)
    if not expected:
        # Nothing to align to
        return X

    if _looks_like_generic_xgb_names(expected):
        print("[WARN] Model feature names look generic (f0,f1,...). "
              "Skipping strict reindex to avoid misalignment.")
        return X

    # Add missing columns with zeros
    missing = [c for c in expected if c not in X.columns]
    if missing:
        print(f"[INFO] Adding {len(missing)} missing columns present in model but not in data.")
        for col in missing:
            X[col] = 0.0

    # Reindex to model order; silently drop extras not in the model
    X = X.reindex(columns=list(expected))
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
    imp = imp.fillna(0.0).sort_values("gain", ascending=False).reset_index(drop=True)
    return imp


def _resolve_model_path(model_path: str | Path, fallback_dir: str | Path = "models") -> Path:
    """Resolve `model_path` or fall back to the newest model in `fallback_dir`."""
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


# -----------------------------
# Core evaluation
# -----------------------------

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
    db_path: str | Path = Path("data/iv_data_1m.db"),
    target_col: str | None = None,               # <-- NEW
) -> None:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    pooled = build_pooled_iv_return_dataset_time_safe(
        tickers=tickers,
        start=start_ts,
        end=end_ts,
        r=r,
        forward_steps=forward_steps,
        tolerance=tolerance,
        db_path=Path(db_path),
    )
    # --- choose the right target ---
    if target_col is None:
        name = Path(model_path).name.lower()
        if "ret" in name:
            target_col = "iv_ret_fwd"
        elif "clip" in name or "level" in name:
            target_col = "iv_clip"
        else:
            raise ValueError("target_col not set and cannot infer from model name.")

    if target_col not in pooled.columns:
        raise KeyError(f"Target column '{target_col}' not in pooled columns.")

    # y and X (drop the other target to avoid leakage)
    y = pooled[target_col].astype(float)
    drop_cols = [target_col]
    if target_col == "iv_clip" and "iv_ret_fwd" in pooled.columns:
        drop_cols.append("iv_ret_fwd")   # prevent peeking at t+1 if someone left it in X
    X = pooled.drop(columns=[c for c in drop_cols if c in pooled.columns])
    X = _ensure_numeric(X)               # your existing helper


    n = len(X)
    if n < 10:
        raise ValueError(f"Too few rows for evaluation: {n}")

    # Time-respecting split (assumes pooled is time-ordered)
    split_idx = int(n * (1 - test_frac))
    X_tr, X_te = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_tr, y_te = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    # Load model
    model_path = _resolve_model_path(model_path, fallback_dir="models")
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    # Align columns to model expectations
    X_tr = _align_columns_to_model(model, X_tr)
    X_te = _align_columns_to_model(model, X_te)

    # Predict and score
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
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "forward_steps": forward_steps,
        "tolerance": tolerance,
        "test_frac": test_frac,
        "model_path": str(model_path),
    }
    print(f"[METRICS] RMSE={rmse:.6f}  R²={r2:.3f}")

    # Importances: built-in XGB
    xgb_imp = _xgb_importances(model)

    # Permutation importance on (optionally) a sample of test set
    perm_df = pd.DataFrame()
    if perm_repeats and perm_repeats > 0:
        if (perm_sample is not None) and (len(X_te) > perm_sample):
            Xp = X_te.sample(perm_sample, random_state=42)
            yp = y_te.loc[Xp.index]
        else:
            Xp, yp = X_te, y_te
        perm = permutation_importance(
            model,
            Xp,
            yp,
            n_repeats=perm_repeats,
            random_state=42,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
        perm_df = pd.DataFrame(
            {
                "feature": Xp.columns,
                "perm_importance_mean": perm.importances_mean,
                "perm_importance_std": perm.importances_std,
            }
        ).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    # Outputs
    metrics_path = metrics_dir / f"{outputs_prefix}_metrics.json"
    xgb_imp_path = metrics_dir / f"{outputs_prefix}_xgb_importances.csv"
    perm_imp_path = metrics_dir / f"{outputs_prefix}_perm_importances.csv"
    preds_path = metrics_dir / f"{outputs_prefix}_predictions.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    xgb_imp.to_csv(xgb_imp_path, index=False)
    if not perm_df.empty:
        perm_df.to_csv(perm_imp_path, index=False)

    # Per-symbol effects via SHAP-like contributions
    save_symbol_effects(
        model=model,
        X_test=X_te,
        metrics_dir=metrics_dir,
        prefix=outputs_prefix,
    )

    if save_predictions:
        pd.DataFrame(
            {
                "y_true": y_te.values,
                "y_pred": y_pred,
                "resid": y_te.values - y_pred,
            }
        ).to_csv(preds_path, index=False)

    print(f"[SAVED] {metrics_path}")
    print(f"[SAVED] {xgb_imp_path}")
    if not perm_df.empty:
        print(f"[SAVED] {perm_imp_path}")
    if save_predictions:
        print(f"[SAVED] {preds_path}")


# -----------------------------
# Symbol effects & SHAP-ish saves
# -----------------------------

def save_symbol_effects(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    metrics_dir: str | Path,
    prefix: str,
) -> None:
    """
    Save per-feature SHAP-like average contributions and a sym_* summary.

    Uses XGBoost's pred_contribs=True to get SHAP + bias.
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    booster = model.get_booster()
    dm = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist())
    contribs = booster.predict(dm, pred_contribs=True)  # SHAP + bias col
    contribs_df = pd.DataFrame(contribs, columns=list(X_test.columns) + ["bias"])

    # Average contribution per feature (sorted)
    feat_avg = contribs_df.drop(columns=["bias"], errors="ignore").mean().sort_values(ascending=False)
    feat_avg_df = feat_avg.to_frame("avg_contrib").reset_index().rename(columns={"index": "feature"})
    feat_avg_df.to_csv(metrics_dir / f"{prefix}_feature_shap_avg.csv", index=False)

    # Roll-up: symbol one-hots "sym_*"
    sym_cols = [c for c in X_test.columns if c.startswith("sym_")]
    if sym_cols:
        sym_avg = contribs_df[sym_cols].mean().sort_values(ascending=False)
        sym_df = sym_avg.to_frame("avg_shap").reset_index().rename(columns={"index": "feature"})
        sym_df["ticker"] = sym_df["feature"].str.replace(r"^sym_", "", regex=True)
        sym_df["direction"] = np.where(sym_df["avg_shap"] >= 0, "↑", "↓")
        sym_df.to_csv(metrics_dir / f"{prefix}_sym_shap_avg.csv", index=False)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pooled IV-returns XGBoost model.")
    p.add_argument("--model", required=True, help="Path to saved XGBoost model (JSON).")
    p.add_argument("--tickers", nargs="+", default=["QBTS", "IONQ", "RGTI", "QUBT"])
    p.add_argument("--start", default="2025-01-02")
    p.add_argument("--end", default="2025-01-06")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--forward-steps", type=int, default=1)
    p.add_argument("--tolerance", default="2s")
    p.add_argument("--r", type=float, default=0.045)
    p.add_argument("--metrics-dir", default="metrics")
    p.add_argument("--prefix", default="iv_returns_pooled_eval")
    p.add_argument("--no-preds", action="store_true", help="Do not save per-row predictions.")
    p.add_argument("--perm-repeats", type=int, default=5, help="Permutation importance repeats (0 to disable).")
    p.add_argument("--perm-sample", type=int, default=5000, help="Cap permutation sample size for speed.")
    p.add_argument(
        "--db",
        type=str,
        default=os.getenv("IV_DB_PATH", "data/iv_data_1m.db"),
        help="Path to the SQLite DB to read from.",
    )
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
        db_path=args.db,
    )
