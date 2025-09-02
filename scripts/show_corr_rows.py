from __future__ import annotations

"""Quick helper to print the first two rows used for baseline correlations.

Usage (PowerShell):
  python scripts/show_corr_rows.py --tickers QBTS IONQ RGTI QUBT \
      --start 2025-06-02 --end 2025-08-06 --db data/iv_data_1m.db --tolerance 30s

This prints the first two aligned rows of the IV/IVRET panel that
`baseline_correlation.py` uses to compute correlation matrices.
"""

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
from sys import path
path.append(str(Path(__file__).parent.parent / "src"))  # Local imports via src

# Local imports via src/
from feature_engineering import build_iv_panel, DEFAULT_DB_PATH
from data_loader_coordinator import load_cores_with_auto_fetch


def main() -> None:
    p = argparse.ArgumentParser(description="Show two panel rows used for correlations")
    p.add_argument("--tickers", nargs="+", required=True, help="Tickers to include")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--db", type=str, default=None, help="Path to SQLite DB (defaults to project setting)")
    p.add_argument("--tolerance", type=str, default="15s", help="Merge tolerance (e.g., 15s, 30s)")
    args = p.parse_args()

    db_path = Path(args.db) if args.db else DEFAULT_DB_PATH
    cores = load_cores_with_auto_fetch(list(args.tickers), args.start, args.end, db_path)
    panel = build_iv_panel(cores, tolerance=args.tolerance)
    if panel is None or panel.empty:
        print("<no data: panel is empty>")
        return

    cols: list[str] = ["ts_event"]
    iv_cols = [f"IV_{t}" for t in args.tickers if f"IV_{t}" in panel.columns]
    ret_cols = [f"IVRET_{t}" for t in args.tickers if f"IVRET_{t}" in panel.columns]
    cols.extend(iv_cols + ret_cols)
    sub = panel[cols].head(2).copy()
    # Pretty-print timestamps
    if "ts_event" in sub.columns:
        sub["ts_event"] = pd.to_datetime(sub["ts_event"]).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()

