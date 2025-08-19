# iv_transfer_main.py  (ONLY CHANGES SHOWN)
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv

from fetch_data import fetch_and_save
from xgb_iv import (
    build_iv_return_dataset_time_safe,
    train_xgb_iv_returns_time_safe_pooled,   # if you’re using pooled trainer
    make_aligned_panel_sqlite,
)

def main():
    load_dotenv()
    API_KEY = os.getenv("DATABENTO_API_KEY")
    if not API_KEY:
        raise ValueError("Missing DATABENTO_API_KEY")

    tickers = ["QBTS","IONQ","RGTI","QUBT"]
    start = pd.Timestamp("2025-01-02", tz="UTC")
    end   = pd.Timestamp("2025-01-06", tz="UTC")

    # >>>>>>> isolate this run in its own DB <<<<<<<
    run_db = Path(f"data/iv_data_1m_{start:%Y%m%d}_{end:%Y%m%d}.db")
    os.environ["IV_DB_PATH"] = str(run_db)  # optional: let other modules pick it up

    # 1) download to this DB
    for t in tickers:
        print(f"\n[DL] {t} for {start.date()} → {end.date()} …")
        fetch_and_save(API_KEY, t, start, end, db_path=run_db, force=False)

    # 2) build datasets from this *same* DB
    datasets = build_iv_return_dataset_time_safe(
        tickers=tickers, start=start, end=end, db_path=run_db
    )

    # 3) (example) per-target training, or pooled if you have it
    for tgt, df in datasets.items():
        if len(df) < 100:
            print(f"[IV-RET] {tgt}: too few rows ({len(df)}), skipping.")
            continue
        # your per-target trainer here...

    # cross-ticker panel (also reads from run_db)
    print("\n[ATM PANEL] QBTS vs IONQ")
    panel = make_aligned_panel_sqlite("QBTS","IONQ", start=start, end=end, db_path=run_db)
    print(panel.head())

if __name__ == "__main__":
    main()
