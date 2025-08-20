# src/train_peer_effects.py
from __future__ import annotations
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engineering import build_target_peer_dataset

def train_eval_target(
    target: str,
    tickers: list[str],
    start: str | None,
    end: str | None,
    db_path: str | None,
    target_kind: str = "iv_ret",
    test_frac: float = 0.2,
    forward_steps: int = 1,
    tolerance: str = "2s",
    metrics_dir: str = "metrics",
    prefix: str = "peer_effects",
    params: dict | None = None,
):
    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts   = pd.Timestamp(end, tz="UTC") if end else None

    ds = build_target_peer_dataset(
        target=target,
        tickers=tickers,
        start=start_ts,
        end=end_ts,
        forward_steps=forward_steps,
        tolerance=tolerance,
        db_path=db_path,
        target_kind=target_kind,
    )
    if len(ds) < 50:
        print(f"[{target}] too few rows: {len(ds)}")
        return

    y = ds["y"].astype(float)
    X = ds.drop(columns=["y"]).astype(float)

    # chronological split
    split = int(len(X) * (1 - test_frac))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    if params is None:
        params = dict(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            tree_method="hist",
        )

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    rmse = float(np.sqrt(((pred - y_te.values) ** 2).mean()))
    r2   = float(1 - ((pred - y_te.values) ** 2).sum() / ((y_te.values - y_te.values.mean()) ** 2).sum())

    out_dir = Path(metrics_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = dict(
        target=target, target_kind=target_kind,
        rows_train=len(X_tr), rows_test=len(X_te),
        rmse=rmse, r2=r2,
        features=list(X.columns),
    )
    (out_dir / f"{prefix}_{target}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{target}] RMSE={rmse:.4f} R²={r2:.3f}  rows={len(ds)}")

    # Save gain-based importance
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp_df = pd.DataFrame({"feature": list(gain.keys()), "gain": list(gain.values())}).sort_values("gain", ascending=False)
    imp_df.to_csv(out_dir / f"{prefix}_{target}_xgb_importance.csv", index=False)

    # SHAP contributions to quantify **peer effects**
    dm = xgb.DMatrix(X_te, feature_names=X.columns.tolist())
    shap_contribs = booster.predict(dm, pred_contribs=True)  # includes bias as last column
    shap_df = pd.DataFrame(shap_contribs, columns=list(X.columns) + ["bias"])

    peer_cols = [c for c in X.columns if c.startswith("IV_")]
    if "IV_SELF" in X.columns:
        peer_cols = [c for c in peer_cols if c != "IV_SELF"]  # focus on *peer* effects

    # Average absolute SHAP per peer (strong, sign-agnostic “effect size”)
    peer_abs = shap_df[peer_cols].abs().mean().sort_values(ascending=False).to_frame("mean_abs_shap")
    # Signed average (directional tilt)
    peer_signed = shap_df[peer_cols].mean().sort_values(ascending=False).to_frame("mean_signed_shap")

    peer_abs.reset_index().rename(columns={"index":"feature"}).to_csv(
        out_dir / f"{prefix}_{target}_peer_effect_abs_shap.csv", index=False
    )
    peer_signed.reset_index().rename(columns={"index":"feature"}).to_csv(
        out_dir / f"{prefix}_{target}_peer_effect_signed_shap.csv", index=False
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--db")
    ap.add_argument("--target-kind", choices=["iv_ret","iv"], default="iv_ret")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--forward-steps", type=int, default=1)
    ap.add_argument("--tolerance", default="2s")
    ap.add_argument("--metrics-dir", default="metrics")
    ap.add_argument("--prefix", default="peer_effects")
    args = ap.parse_args()

    train_eval_target(
        target=args.target,
        tickers=args.tickers,
        start=args.start, end=args.end,
        db_path=args.db,
        target_kind=args.target_kind,
        test_frac=args.test_frac,
        forward_steps=args.forward_steps,
        tolerance=args.tolerance,
        metrics_dir=args.metrics_dir,
        prefix=args.prefix,
    )

if __name__ == "__main__":
    main()
