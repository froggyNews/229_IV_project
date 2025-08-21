#!/usr/bin/env python3
"""Test date filtering in data loader."""

import sys
sys.path.append('src')

from data_loader_coordinator import load_ticker_core
from pathlib import Path

# Test date filtering
print("Testing date filtering...")
print("=" * 50)

# Test with a specific date range
start_date = "2025-08-05"
end_date = "2025-08-06"
ticker = "QUBT"

print(f"Loading {ticker} data for {start_date} to {end_date}")

core = load_ticker_core(
    ticker=ticker,
    start=start_date,
    end=end_date,
    db_path=Path("data/iv_data_1m.db")
)

if not core.empty:
    print(f"Loaded {len(core):,} rows")
    if 'ts_event' in core.columns:
        min_date = core['ts_event'].min()
        max_date = core['ts_event'].max()
        print(f"Date range in data: {min_date} to {max_date}")
        
        # Check if dates are within requested range
        import pandas as pd
        start_pd = pd.to_datetime(start_date, utc=True)
        end_pd = pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        print(f"Requested range: {start_pd} to {end_pd}")
        
        if min_date >= start_pd:
            print("✓ Start date filtering working correctly")
        else:
            print("✗ Start date filtering NOT working - data before start date found")
            
        if max_date <= end_pd:
            print("✓ End date filtering working correctly")
        else:
            print("✗ End date filtering NOT working - data after end date found")
    else:
        print("No ts_event column found")
else:
    print("No data loaded")
