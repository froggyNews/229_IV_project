from __future__ import annotations

"""Baseline historical correlation calculations for 1h ATM slices."""

from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd

from feature_engineering import build_iv_panel, DEFAULT_DB_PATH
from data_loader_coordinator import load_cores_with_auto_fetch


def compute_baseline_correlations(
    tickers: Sequence[str],
    start: str | None = None,
    end: str | None = None,
    db_path: str | Path | None = None,
    cores: Dict[str, pd.DataFrame] | None = None,
    tolerance: str = "2s",
    surface_mode: str = "atm",  # "atm" or "full"
    surface_agg: str = "median",  # passed to build_iv_panel
) -> dict:
    """Compute historical correlation matrices for IV level and IV returns.

    Parameters
    ----------
    tickers : Sequence[str]
        Tickers to include in the analysis.
    start, end : str | None
        Optional start and end timestamps for data loading. These are
        forwarded to :func:`load_cores_with_auto_fetch` when ``cores`` is not
        provided.
    db_path : str | Path | None
        Path to the SQLite database containing 1h ATM slices. Defaults to the
        project's ``DEFAULT_DB_PATH`` when ``None``.
    cores : dict | None
        Optional mapping of ``ticker -> DataFrame``. If provided, data is taken
        from this mapping instead of loading from disk. Primarily intended for
        testing.
    tolerance : str
        Merge tolerance passed to :func:`build_iv_panel`.

    Returns
    -------
    dict
        Dictionary with two keys:
        ``"clip"`` – correlation matrix of IV levels.
        ``"iv_returns"`` – correlation matrix of IV return series.
    """
    if cores is None:
        db = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        atm_only = (str(surface_mode).lower() != "full")
        cores = load_cores_with_auto_fetch(list(tickers), start, end, db, atm_only=atm_only)

    panel = build_iv_panel(cores, tolerance=tolerance, agg=surface_agg)
    if panel is None or panel.empty:
        return {"clip": pd.DataFrame(), "iv_returns": pd.DataFrame()}

    iv_cols = [f"IV_{t}" for t in tickers if f"IV_{t}" in panel.columns]
    ret_cols = [f"IVRET_{t}" for t in tickers if f"IVRET_{t}" in panel.columns]
    print(f"Computing correlations for IV columns: {iv_cols}. sample rows:\n{panel[iv_cols].head()}")
    print(f"Computing correlations for IV return columns: {ret_cols}. sample rows:\n{panel[ret_cols].head()}")
    clip_corr = (
        panel[iv_cols].corr().rename(index=lambda x: x[3:], columns=lambda x: x[3:])
        if len(iv_cols) >= 2
        else pd.DataFrame()
    )
    ivret_corr = (
        panel[ret_cols].corr().rename(index=lambda x: x[6:], columns=lambda x: x[6:])
        if len(ret_cols) >= 2
        else pd.DataFrame()
    )

    return {"clip": clip_corr, "iv_returns": ivret_corr}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute baseline historical correlations for 1h ATM slices."
    )
    parser.add_argument("tickers", nargs="+", help="Ticker symbols")
    parser.add_argument("--start", type=str, default=None, help="Start timestamp")
    parser.add_argument("--end", type=str, default=None, help="End timestamp")
    parser.add_argument(
        "--db-path", type=str, default=None, help="Path to SQLite DB (defaults to project DB)"
    )
    parser.add_argument("--tolerance", type=str, default="2s", help="Merge tolerance")
    args = parser.parse_args()

    result = compute_baseline_correlations(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        db_path=args.db_path,
        tolerance=args.tolerance,
    )
    if not result["clip"].empty:
        print("IV clip correlation matrix:")
        print(result["clip"])
    else:
        print("IV clip correlation matrix: <not enough data>")
    if not result["iv_returns"].empty:
        print("IV returns correlation matrix:")
        print(result["iv_returns"])
    else:
        print("IV returns correlation matrix: <not enough data>")
