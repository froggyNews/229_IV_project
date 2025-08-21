"""
Compare pooled vs isolated IV return forecasting models.
"""

import sys
import os
from pathlib import Path

# Add the parent directory (src) to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
project_root = src_dir.parent

# Add src directory to path for imports
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Now import the required modules
try:
    from feature_engineering import (
        build_pooled_iv_return_dataset_time_safe,
        build_iv_return_dataset_time_safe,
    )
    from data_loader_coordinator import load_cores_with_auto_fetch
    print("✓ Successfully imported feature_engineering and data_loader_coordinator")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    raise

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from typing import Dict, Any, List
import warnings

# Configure warnings and plotting
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

def compare_pooled_vs_isolated(
    tickers: List[str] = ["QUBT", "QBTS", "RGTI", "IONQ"],
    start: str = "2025-08-02",
    end: str = "2025-08-06",
    forward_steps: int = 15,
    test_frac: float = 0.2,
    r: float = 0.045,
    tolerance: str = "2s",
    db_path: str = "data/iv_data_1m.db",
    save_results: bool = True,
    output_dir: str = "outputs/experiments"
) -> Dict[str, Any]:
    """
    Compare pooled vs isolated model performance.
    
    Returns:
        Dictionary containing comparison results and metrics
    """
    
    print("=== Pooled vs Isolated IV Return Forecasting Comparison ===")
    print(f"Tickers: {tickers}")
    print(f"Date range: {start} to {end}")
    print(f"Forward steps: {forward_steps}")
    
    # Load data cores
    print("\n1. Loading data cores...")
    cores = load_cores_with_auto_fetch(
        tickers=tickers,
        start=start,
        end=end,
        db_path=Path(db_path),
        auto_fetch=True
    )
    
    if not cores:
        raise ValueError("No cores loaded successfully")
    
    print(f"   Loaded cores for {len(cores)} tickers")
    for ticker, core in cores.items():
        print(f"   {ticker}: {len(core):,} rows")
    
    # Build pooled dataset
    print("\n2. Building pooled dataset...")
    pooled_data = build_pooled_iv_return_dataset_time_safe(
        tickers=tickers,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        r=r,
        forward_steps=forward_steps,
        tolerance=tolerance,
        db_path=Path(db_path),
        cores=cores
    )
    
    if pooled_data.empty:
        raise ValueError("Pooled dataset is empty")
    
    print(f"   Pooled dataset: {len(pooled_data):,} rows, {pooled_data.shape[1]} columns")
    
    # Build isolated datasets
    print("\n3. Building isolated datasets...")
    isolated_data = build_iv_return_dataset_time_safe(
        tickers=tickers,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        r=r,
        forward_steps=forward_steps,
        tolerance=tolerance,
        db_path=Path(db_path),
        cores=cores
    )
    
    print(f"   Built isolated datasets for {len(isolated_data)} tickers")
    for ticker, data in isolated_data.items():
        print(f"   {ticker}: {len(data):,} rows, {data.shape[1]} columns")
    
    # Train models and compare (would need to import training functions)
    print("\n4. Training models...")
    print("   (Model training functionality would be implemented here)")
    
    # Placeholder results
    results = {
        "experiment_info": {
            "tickers": tickers,
            "date_range": f"{start} to {end}",
            "forward_steps": forward_steps,
            "test_frac": test_frac,
            "timestamp": datetime.now().isoformat()
        },
        "data_summary": {
            "pooled": {
                "rows": len(pooled_data),
                "columns": pooled_data.shape[1]
            },
            "isolated": {
                ticker: {"rows": len(data), "columns": data.shape[1]}
                for ticker, data in isolated_data.items()
            }
        },
        "cores_summary": {
            ticker: len(core) for ticker, core in cores.items()
        }
    }
    
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"pooled_vs_isolated_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n5. Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    try:
        results = compare_pooled_vs_isolated()
        print("\n✓ Experiment completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()