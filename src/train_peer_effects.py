"""
Simplified peer effects training - clear and focused.

The goal: For a target ticker (e.g., AAPL), predict its IV changes using:
1. Peer tickers' current IV levels and returns
2. Optionally, the target's own lagged features (to avoid leakage)
3. Market/time control features

This helps understand: "How much does MSFT's IV movement predict AAPL's future IV?"
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from feature_engineering import build_target_peer_dataset


@dataclass 
class PeerConfig:
    """Simplified configuration for peer effects analysis."""
    target: str                           # Target ticker to predict
    tickers: Sequence[str]               # All tickers (includes target + peers)
    start: str                           # Start date
    end: str                             # End date
    db_path: Optional[Path] = None       # Database path
    
    # Model settings
    # target_kind options: "iv_ret"/"iv_ret_fwd" (forward return),
    # "iv_ret_fwd_abs" (absolute forward return), or "iv" (levels)
    target_kind: str = "iv_ret"
    forward_steps: int = 15              # How many steps ahead to predict
    test_frac: float = 0.2               # Test set fraction
    tolerance: float = "2s"
    # Peer effects settings
    include_self_lag: bool = True        # Include target's own lagged features
    exclude_contemporaneous: bool = True  # Exclude same-time target features (avoid leakage)
    
    # Output
    output_dir: Path = Path("peer_analysis")
    save_details: bool = False           # Save detailed analysis


def example_peer_analysis():
    """Example of how to run peer effects analysis."""
    from data_loader_coordinator import load_ticker_core
    
    # Configuration
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    targets = ["AAPL", "MSFT"]  # Analyze peer effects for these
    
    # Load data (or use from main pipeline)
    cores = {}
    for ticker in tickers:
        cores[ticker] = load_ticker_core(ticker, start="2025-01-01", end="2025-01-31")
    
    # Run analysis
    results = run_multi_target_analysis(
        targets=targets,
        tickers=tickers,
        start="2025-01-01", 
        end="2025-01-31",
        cores=cores,
        target_kind="iv_ret",  # Predict IV returns
        forward_steps=15,      # 15-minute ahead prediction
        include_self_lag=True, # Use target's own lagged features
        save_details=True      # Save detailed results
    )
    
    return results


if __name__ == "__main__":
    results = example_peer_analysis()
    print("Analysis complete!")