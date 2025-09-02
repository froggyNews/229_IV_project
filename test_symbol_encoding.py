#!/usr/bin/env python3
"""Test that pooled dataset retains correct ticker one-hot encoding."""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
from feature_engineering import build_pooled_iv_return_dataset_time_safe


def make_core(symbol):
    dates = pd.date_range('2025-01-01', periods=5, freq='T', tz='UTC')
    data = {
        'ts_event': dates,
        'iv_clip': np.linspace(0.1, 0.5, 5),
        'strike_price': 10.0,
        'time_to_expiry': 1/365,
        'stock_close': 10.0,
        'option_type': ['C'] * 5,
        'opt_volume': 100,
        'symbol': [symbol] * 5,
    }
    return pd.DataFrame(data)

cores = {sym: make_core(sym) for sym in ["AAA", "BBB"]}

df = build_pooled_iv_return_dataset_time_safe(
    tickers=["AAA", "BBB"],
    start=None,
    end=None,
    r=0.045,
    forward_steps=1,
    tolerance="15s",
    cores=cores,
)

sym_cols = [c for c in df.columns if c.startswith('sym_')]
assert sym_cols == ['sym_AAA', 'sym_BBB'], f"Unexpected sym columns: {sym_cols}"

# Each row should have exactly one ticker indicator equal to 1
row_sums = df[sym_cols].sum(axis=1)
assert (row_sums == 1).all(), "Each row should have exactly one active ticker indicator"

print("✅ Ticker encoding preserved correctly")
