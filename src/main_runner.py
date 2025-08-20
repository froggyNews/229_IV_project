# main_runner.py — no-CLI, programmatic runner with optional peer-effects
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any

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
from train_peer_effects import PeerEffectsConfig, train_peer_effects


@dataclass
class RunConfig:
    # data
    db: Path = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
    tickers: Sequence[str] = field(default_factory=list)  # required non-empty
    start: str = "2025-01-02"  # UTC date string or ISO timestamp string
    end: str = "2025-01-06"    # UTC date string or ISO timestamp string
    forward_steps: int = 1
    tolerance: str = "2s"
    test_frac: float = 0.2

    # outputs
    metrics_path: Path = Path("metrics/iv_metrics.json")
    models_dir: Path = Path("models")
    prefix: str = "iv_returns_pooled"

    # training
    xgb_params: Optional[Dict[str, Any]] = None

    # evaluation
    evaluate_trained: bool = True                # evaluate models we train below
    evaluate_model_path: Optional[Path] = None   # or evaluate a pre-trained single model

    # --- NEW: peer-effects training ---
    peer_targets: Sequence[str] = field(default_factory=list)  # e.g., ["AAPL", "MSFT"]; empty -> skip
    peer_target_kind: str = "iv_ret"                           # "iv_ret" or "iv"
    peer_prefix: str = "peer_effects"                          # file prefix for peer-effects artifacts
    peer_target_kinds: Optional[Sequence[str]] = None

def _train_xgb(
    pooled: pd.DataFrame,
    target: str,
    test_frac: float = 0.2,
    drop_cols: Sequence[str] | None = None,
    params: dict | None = None,
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """Train XGBRegressor on pooled dataset, return (model, metrics)."""
    drop_cols = list(drop_cols or [])
    if target not in pooled.columns:
        raise KeyError(f"Target '{target}' not in columns: {list(pooled.columns)}")

    pooled = pooled.sort_index().reset_index(drop=True)
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
            n_jobs=-1,
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


def run(cfg: RunConfig) -> Dict[str, Any]:
    """Build pooled dataset, train/evaluate models, and optionally run peer-effects for targets."""
    if not cfg.tickers:
        raise ValueError("RunConfig.tickers must be non-empty.")

    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")

    # Build pooled dataset once
    pooled = build_pooled_iv_return_dataset_time_safe(
        tickers=list(cfg.tickers),
        start=start,
        end=end,
        forward_steps=cfg.forward_steps,
        tolerance=cfg.tolerance,
        db_path=cfg.db,
    )
    if pooled.empty:
        raise RuntimeError("Pooled dataset is empty.")
    print(f"[POOLED] rows={len(pooled):,}, columns={pooled.shape[1]}")

    # Train iv_ret_fwd
    ret_model, m_ret = _train_xgb(
        pooled, target="iv_ret_fwd", test_frac=cfg.test_frac, params=cfg.xgb_params
    )

    # Train iv_clip (drop iv_ret_fwd to avoid any leakage)
    clip_model, m_clip = _train_xgb(
        pooled, target="iv_clip", test_frac=cfg.test_frac, drop_cols=["iv_ret_fwd"], params=cfg.xgb_params
    )

    # Save aggregate metrics for pooled models
    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = {"iv_ret_fwd": m_ret, "iv_clip": m_clip}
    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[METRICS] saved → {cfg.metrics_path}")

    # Save models
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    ret_path = cfg.models_dir / f"{cfg.prefix}_ret.json"
    clip_path = cfg.models_dir / f"{cfg.prefix}_clip.json"
    ret_model.save_model(ret_path.as_posix())
    clip_model.save_model(clip_path.as_posix())
    print(f"[SAVE] iv_ret_fwd model → {ret_path}")
    print(f"[SAVE] iv_clip    model → {clip_path}")

    # Evaluate pooled models (and/or a provided model path)
    eval_targets: list[tuple[str, Path]] = []
    if cfg.evaluate_trained:
        eval_targets.append(("iv_ret_fwd", ret_path))
        eval_targets.append(("iv_clip", clip_path))
    if cfg.evaluate_model_path is not None:
        eval_targets.append(("custom", cfg.evaluate_model_path))
    
    for tag, model_path in eval_targets:
        evaluate_pooled_model(
            model_path=model_path,
            tickers=list(cfg.tickers),
            start=cfg.start,
            end=cfg.end,
            test_frac=cfg.test_frac,
            forward_steps=cfg.forward_steps,
            tolerance=cfg.tolerance,
            metrics_dir=cfg.metrics_path.parent,
            outputs_prefix=f"{cfg.prefix}_{tag}",
            save_predictions=False,
            perm_repeats=0,
            db_path=cfg.db,
            target_col=("iv_ret_fwd" if tag == "iv_ret_fwd" or "ret" in str(model_path).lower() else "iv_clip"),
        )

        print(f"[EVAL] done → {tag} ({model_path})")

    # --- Peer-effects per target (optional) ---
    if cfg.peer_targets:
        kinds = list(cfg.peer_target_kinds) if cfg.peer_target_kinds else [cfg.peer_target_kind]
        for tgt in cfg.peer_targets:
            for kind in kinds:
                pe_cfg = PeerEffectsConfig(
                    target=tgt,
                    tickers=list(cfg.tickers),
                    start=cfg.start,
                    end=cfg.end,
                    db_path=str(cfg.db),
                    target_kind=kind,                         # <-- run both
                    test_frac=cfg.test_frac,
                    forward_steps=cfg.forward_steps,
                    tolerance=cfg.tolerance,
                    metrics_dir=cfg.metrics_path.parent,
                    prefix=f"{cfg.peer_prefix}_{kind}",       # separate outputs
                    xgb_params=cfg.xgb_params,
                )
                res = train_peer_effects(pe_cfg)
                print(f"[PEER] {tgt}:{kind} → {res.get('status','ok')}")

    return all_metrics


# Example programmatic usage
if __name__ == "__main__":
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    tickers = ["QUBT", "QBTS", "RGTI", "IONQ"]

    CFG = RunConfig(
        tickers=tickers,
        start="2025-08-02",
        end="2025-08-06",
        db=Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db")),
        prefix=f"iv_{timestamp}_pooled",              # fixed f-string
        evaluate_trained=True,
        evaluate_model_path=None,

        # run peer effects for ALL tickers, for BOTH iv_ret and iv (level)
        peer_targets=tickers,                          # fixed undefined name
        peer_target_kinds=("iv_ret", "iv"),            # ← both kinds in one pass
        peer_prefix=f"peer_effects_{timestamp}",
    )
    run(CFG)
