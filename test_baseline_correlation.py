#!/usr/bin/env python3
"""Test baseline historical correlation computation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from baseline_correlation import compute_baseline_correlations


def make_core(symbol: str, values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(values), freq="T", tz="UTC")
    data = {
        "ts_event": dates,
        "iv_clip": values,
        "strike_price": 10.0,
        "time_to_expiry": 1 / 365,
        "stock_close": 10.0,
        "option_type": ["C"] * len(values),
        "opt_volume": 100,
        "symbol": [symbol] * len(values),
    }
    return pd.DataFrame(data)


cores = {
    "AAA": make_core("AAA", [0.1, 0.2, 0.4, 0.2, 0.5]),
    "BBB": make_core("BBB", [0.2, 0.4, 0.8, 0.4, 1.0]),
}

result = compute_baseline_correlations(["AAA", "BBB"], cores=cores)
clip_corr = result["clip"]
ret_corr = result["iv_returns"]

assert np.isclose(clip_corr.loc["AAA", "BBB"], 1.0)
assert np.isclose(ret_corr.loc["AAA", "BBB"], 1.0)

print("✅ baseline correlation computed correctly")
