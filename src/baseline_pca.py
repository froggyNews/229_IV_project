from __future__ import annotations

"""Baseline PCA for IV levels and IV returns.

Computes principal components on the cross-section of tickers for:
 - IV returns (recommended default)
 - IV levels (optional)

Returns explained variance ratios and component loadings for top-N components.
"""

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from feature_engineering import build_iv_panel, DEFAULT_DB_PATH
from data_loader_coordinator import load_cores_with_auto_fetch


def _pca_summary(X: pd.DataFrame, n_components: int) -> Dict:
    """Run PCA on standardized data and return loadings and explained variance."""
    if X is None or X.empty or X.shape[1] < 2:
        return {"explained_variance_ratio": [], "components": {}, "n_samples": 0}

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X.values)
    n = min(n_components, X.shape[1])
    pca = PCA(n_components=n, svd_solver="full", random_state=42)
    scores = pca.fit_transform(X_std)

    # Map component loadings to column names
    loadings = {}
    cols = list(X.columns)
    for i in range(n):
        comp_name = f"PC{i+1}"
        load = {col: float(pca.components_[i, j]) for j, col in enumerate(cols)}
        loadings[comp_name] = load

    return {
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "components": loadings,
        "n_samples": int(X.shape[0]),
    }


def compute_baseline_pca(
    tickers: Sequence[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path | None = None,
    cores: Dict[str, pd.DataFrame] | None = None,
    tolerance: str = "2s",
    n_components: int = 3,
    include_levels: bool = False,
    surface_mode: str = "atm",   # "atm" or "full"
    surface_agg: str = "median", # passed to build_iv_panel
) -> dict:
    """Compute PCA on IV return panel (and optionally IV level panel)."""
    if cores is None:
        db = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        atm_only = (str(surface_mode).lower() != "full")
        cores = load_cores_with_auto_fetch(list(tickers), start, end, db, atm_only=atm_only)
    panel = build_iv_panel(cores, tolerance=tolerance, agg=surface_agg)
    if panel is None or panel.empty:
        return {"iv_returns": {}, "iv_levels": {} if include_levels else {}}

    # Build matrices
    ret_cols = [f"IVRET_{t}" for t in tickers if f"IVRET_{t}" in panel.columns]
    lvl_cols = [f"IV_{t}" for t in tickers if f"IV_{t}" in panel.columns]

    results = {}
    if len(ret_cols) >= 2:
        X_ret = panel[ret_cols].dropna()
        # Rename without prefixes in outputs
        X_ret.columns = [c[6:] for c in X_ret.columns]
        results["iv_returns"] = _pca_summary(X_ret, n_components)
    else:
        results["iv_returns"] = {"explained_variance_ratio": [], "components": {}, "n_samples": 0}

    if include_levels:
        if len(lvl_cols) >= 2:
            X_lvl = panel[lvl_cols].dropna()
            X_lvl.columns = [c[3:] for c in X_lvl.columns]
            results["iv_levels"] = _pca_summary(X_lvl, n_components)
        else:
            results["iv_levels"] = {"explained_variance_ratio": [], "components": {}, "n_samples": 0}

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute PCA on IV return/level panels.")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--start", type=str, default=None, help="Start timestamp")
    parser.add_argument("--end", type=str, default=None, help="End timestamp")
    parser.add_argument("--db-path", type=str, default=None, help="Path to SQLite DB")
    parser.add_argument("--tolerance", type=str, default="2s", help="Merge tolerance")
    parser.add_argument("--n-components", type=int, default=3, help="Number of principal components")
    parser.add_argument("--include-levels", action="store_true", help="Also run PCA on IV levels")
    parser.add_argument("--surface-mode", choices=["atm", "full"], default="atm", help="ATM-only or full surface")
    parser.add_argument("--surface-agg", choices=["median", "mean"], default="median", help="Aggregate across surface")
    args = parser.parse_args()

    res = compute_baseline_pca(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        db_path=args.db_path,
        tolerance=args.tolerance,
        n_components=args.n_components,
        include_levels=args.include_levels,
        surface_mode=args.surface_mode,
        surface_agg=args.surface_agg,
    )

    evr = res.get("iv_returns", {}).get("explained_variance_ratio", [])
    print("IV returns PCA explained variance ratio:")
    print([round(v, 4) for v in evr])
    if args.include_levels:
        evr_l = res.get("iv_levels", {}).get("explained_variance_ratio", [])
        print("\nIV levels PCA explained variance ratio:")
        print([round(v, 4) for v in evr_l])
