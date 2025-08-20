# train_iv_returns.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from typing import Tuple
import numpy as np
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

