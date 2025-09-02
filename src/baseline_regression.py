from __future__ import annotations

"""Baseline pairwise/market-factor regressions for IV levels and IV returns.

This module provides a lightweight baseline: for each ticker, regress its
IV return series on the cross-sectional mean of peer IV returns (a simple
"market" factor). Optionally, do the same for IV levels.

Outputs per ticker: intercept, beta, R^2, observations.
"""

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from feature_engineering import build_iv_panel, DEFAULT_DB_PATH
from data_loader_coordinator import load_cores_with_auto_fetch


def _simple_ols(y: pd.Series, x: pd.Series) -> Dict[str, float]:
    """Compute OLS intercept, slope, and R^2 for y ~ a + b*x.

    Handles NaNs by dropping aligned missing values.
    """
    df = pd.concat({"y": y, "x": x}, axis=1).dropna()
    if len(df) < 3:
        return {"intercept": np.nan, "beta": np.nan, "r2": np.nan, "n": float(len(df))}

    x_vals = df["x"].values
    y_vals = df["y"].values

    x_mean = x_vals.mean()
    y_mean = y_vals.mean()
    x_center = x_vals - x_mean
    y_center = y_vals - y_mean

    ss_x = np.dot(x_center, x_center)
    if ss_x == 0:
        return {"intercept": float(y_mean), "beta": 0.0, "r2": 0.0, "n": float(len(df))}

    beta = float(np.dot(x_center, y_center) / ss_x)
    intercept = float(y_mean - beta * x_mean)
    y_hat = intercept + beta * x_vals
    ss_tot = float(np.dot(y_center, y_center))
    ss_res = float(np.dot(y_vals - y_hat, y_vals - y_hat))
    r2 = 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    return {"intercept": intercept, "beta": beta, "r2": r2, "n": float(len(df))}


def compute_baseline_regression(
    tickers: Sequence[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path | None = None,
    cores: Dict[str, pd.DataFrame] | None = None,
    tolerance: str = "15s",
    include_levels: bool = True,
) -> dict:
    """Compute simple market-factor regressions for IV returns and levels.

    For each ticker t, regress IVRET_t on the cross-sectional mean of IVRET of
    the other provided tickers. If ``include_levels`` is True, repeat for IV_t.

    Returns a dictionary with keys ``"iv_returns"`` and (optionally)
    ``"iv_levels"`` mapping ticker -> metrics dict.
    """
    if cores is None:
        db = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        cores = load_cores_with_auto_fetch(list(tickers), start, end, db)

    panel = build_iv_panel(cores, tolerance=tolerance)
    if panel is None or panel.empty:
        return {"iv_returns": {}, "iv_levels": {} if include_levels else {}}

    # IV returns regressions
    ret_cols = [f"IVRET_{t}" for t in tickers if f"IVRET_{t}" in panel.columns]
    results_ret: Dict[str, Dict[str, float]] = {}
    if len(ret_cols) >= 2:
        ivret_df = panel[ret_cols]
        for t in tickers:
            col = f"IVRET_{t}"
            if col not in ivret_df.columns:
                continue
            others = [c for c in ivret_df.columns if c != col]
            if not others:
                continue
            market = ivret_df[others].mean(axis=1)
            metrics = _simple_ols(ivret_df[col], market)
            results_ret[t] = metrics

    # IV level regressions (optional)
    results_lvl: Dict[str, Dict[str, float]] = {}
    if include_levels:
        lvl_cols = [f"IV_{t}" for t in tickers if f"IV_{t}" in panel.columns]
        if len(lvl_cols) >= 2:
            iv_df = panel[lvl_cols]
            for t in tickers:
                col = f"IV_{t}"
                if col not in iv_df.columns:
                    continue
                others = [c for c in iv_df.columns if c != col]
                if not others:
                    continue
                market = iv_df[others].mean(axis=1)
                metrics = _simple_ols(iv_df[col], market)
                results_lvl[t] = metrics

    return {"iv_returns": results_ret, "iv_levels": results_lvl}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute baseline market-factor regressions for IV returns/levels."
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--start", type=str, default=None, help="Start timestamp")
    parser.add_argument("--end", type=str, default=None, help="End timestamp")
    parser.add_argument(
        "--db-path", type=str, default=None, help="Path to SQLite DB (defaults to project DB)"
    )
    parser.add_argument("--tolerance", type=str, default="15s", help="Merge tolerance")
    parser.add_argument("--no-levels", action="store_true", help="Skip IV level regressions")
    args = parser.parse_args()

    res = compute_baseline_regression(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        db_path=args.db_path,
        tolerance=args.tolerance,
        include_levels=not args.no_levels,
    )

    print("IV returns regression metrics (ticker -> beta, R^2, n):")
    for t, m in res.get("iv_returns", {}).items():
        print(f"  {t}: beta={m.get('beta', np.nan):.4f} R2={m.get('r2', np.nan):.3f} n={int(m.get('n', 0))}")
    if not args.no_levels:
        print("\nIV level regression metrics (ticker -> beta, R^2, n):")
        for t, m in res.get("iv_levels", {}).items():
            print(f"  {t}: beta={m.get('beta', np.nan):.4f} R2={m.get('r2', np.nan):.3f} n={int(m.get('n', 0))}")

