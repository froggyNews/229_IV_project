# src/train_peer_effects.py — programmatic peer-effects trainer (no CLI, hardened)
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engineering import build_target_peer_dataset


@dataclass
class PeerEffectsConfig:
    target: str
    tickers: Sequence[str]
    start: str | None = None                 # "YYYY-MM-DD" (UTC) or ISO ts
    end: str | None = None
    db_path: str | Path | None = None
    target_kind: Literal["iv_ret","iv"] = "iv_ret"
    test_frac: float = 0.2
    forward_steps: int = 15
    tolerance: str = "2s"
    metrics_dir: Path = Path("metrics")
    prefix: str = "peer_effects"
    xgb_params: Optional[Dict[str, Any]] = None

    # NEW: control self-features and robustness
    include_self: Literal["keep","drop","lag1"] = "keep"
    winsorize_y_q: Optional[float] = None         # e.g., 0.005
    winsorize_peer_ret_q: Optional[float] = None  # e.g., 0.005 (applies to IVRET_* cols)
    save_report: bool = False


def _chrono_split(X: pd.DataFrame, y: pd.Series, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n = len(X)
    if n < 10:
        raise ValueError(f"Too few rows: {n}")
    split = int(n * (1 - test_frac))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def _winsorize_series(s: pd.Series, q: float) -> pd.Series:
    if q is None or q <= 0:
        return s
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


def train_peer_effects(cfg: PeerEffectsConfig) -> Dict[str, Any]:
    """Train a per-target XGB model and emit an evaluation dictionary.

    When ``cfg.save_report`` is true the evaluation is also written to
    ``cfg.metrics_dir`` using ``cfg.prefix`` and ``cfg.target`` to build the
    filename.

    """
    start_ts = pd.Timestamp(cfg.start, tz="UTC") if cfg.start else None
    end_ts   = pd.Timestamp(cfg.end,   tz="UTC") if cfg.end   else None

    ds = build_target_peer_dataset(
        target=cfg.target,
        tickers=list(cfg.tickers),
        start=start_ts,
        end=end_ts,
        forward_steps=cfg.forward_steps,
        tolerance=cfg.tolerance,
        db_path=cfg.db_path,
        target_kind=cfg.target_kind,
    )
    if len(ds) < 50:
        msg = f"[{cfg.target}] too few rows: {len(ds)}"
        print(msg)
        return {"status": "too_few_rows", "rows": len(ds), "message": msg}

    # --- target & features ---
    y = ds["y"].astype(float).copy()
    X = ds.drop(columns=["y"]).copy()

    # Prevent leakage for target_kind="iv": drop contemporaneous iv level
    if cfg.target_kind == "iv":
        for c in ("iv_clip", "IV_SELF"):
            if c in X.columns:
                X.drop(columns=c, inplace=True)

    # Handle self-features for iv_ret (or iv if you want to be strict)
    if cfg.include_self == "drop":
        for c in ("IV_SELF", "IVRET_SELF"):
            if c in X.columns:
                X.drop(columns=c, inplace=True)
    elif cfg.include_self == "lag1":
        if "IV_SELF" in X:      X["IV_SELF_L1"] = X["IV_SELF"].shift(1)
        if "IVRET_SELF" in X:   X["IVRET_SELF_L1"] = X["IVRET_SELF"].shift(1)
        X.drop(columns=[c for c in ("IV_SELF","IVRET_SELF") if c in X.columns], inplace=True, errors="ignore")
        # drop first row(s) created by lag
        mask = ~X.isna().any(axis=1)
        y = y[mask]
        X = X[mask]

    # Optional robustness: winsorize y and peer returns
    if cfg.winsorize_y_q:
        y = _winsorize_series(y, cfg.winsorize_y_q)
    if cfg.winsorize_peer_ret_q:
        ivret_cols = [c for c in X.columns if c.startswith("IVRET_")]
        for c in ivret_cols:
            X[c] = _winsorize_series(pd.to_numeric(X[c], errors="coerce"), cfg.winsorize_peer_ret_q)

    # Numeric cast (XGBoost handles remaining NaNs)
    X = X.astype(float)

    # Chronological split
    X_tr, X_te, y_tr, y_te = _chrono_split(X, y, cfg.test_frac)

    params = cfg.xgb_params or dict(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    rmse = float(np.sqrt(np.mean((pred - y_te.values) ** 2)))
    denom = float(np.sum((y_te.values - float(y_te.values.mean())) ** 2))
    r2 = float(1.0 - (float(np.sum((pred - y_te.values) ** 2)) / denom)) if denom > 0 else 0.0

    out_dir = Path(cfg.metrics_dir)
    if cfg.save_report:
        out_dir.mkdir(parents=True, exist_ok=True)


    # Metrics dictionary
    metrics = dict(
        target=cfg.target,
        target_kind=cfg.target_kind,
        rows_train=int(len(X_tr)),
        rows_test=int(len(X_te)),
        rmse=rmse,
        r2=r2,
        features=list(X.columns),
        include_self=cfg.include_self,
        winsorize_y_q=cfg.winsorize_y_q,
        winsorize_peer_ret_q=cfg.winsorize_peer_ret_q,
    )
    print(f"[{cfg.target}] RMSE={rmse:.4f} R²={r2:.3f} rows={len(ds)}")

    # Importances
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp_df = pd.DataFrame({"feature": list(gain.keys()), "gain": list(gain.values())}).sort_values("gain", ascending=False)

    # SHAP-like (pred_contribs)
    dm = xgb.DMatrix(X_te, feature_names=X.columns.tolist())
    shap_contribs = booster.predict(dm, pred_contribs=True)
    shap_df = pd.DataFrame(shap_contribs, columns=list(X.columns) + ["bias"])

    # Peer-only columns (exclude self aliases if present)
    peer_cols = [c for c in X.columns if (c.startswith("IV_") or c.startswith("IVRET_"))]
    peer_cols = [c for c in peer_cols if c not in ("IV_SELF", "IVRET_SELF", "IV_SELF_L1", "IVRET_SELF_L1")]

    peer_abs_df = pd.DataFrame(columns=["feature", "mean_abs_shap"])
    peer_signed_df = pd.DataFrame(columns=["feature", "mean_signed_shap"])
    if peer_cols:
        peer_abs = shap_df[peer_cols].abs().mean().sort_values(ascending=False).to_frame("mean_abs_shap")
        peer_signed = shap_df[peer_cols].mean().sort_values(ascending=False).to_frame("mean_signed_shap")
        peer_abs_df = peer_abs.reset_index().rename(columns={"index": "feature"})
        peer_signed_df = peer_signed.reset_index().rename(columns={"index": "feature"})

    evaluation = {
        "metrics": metrics,
        "xgb_importances": imp_df.to_dict(orient="records"),
        "peer_effect_abs_shap": peer_abs_df.to_dict(orient="records"),
        "peer_effect_signed_shap": peer_signed_df.to_dict(orient="records"),
    }
    result = {
        "status": "ok",
        "rmse": rmse,
        "r2": r2,
        "evaluation": evaluation,
    }
    if cfg.save_report:
        eval_path = out_dir / f"{cfg.prefix}_{cfg.target}_evaluation.json"
        eval_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        result["evaluation_path"] = str(eval_path)

    return result
