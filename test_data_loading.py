#!/usr/bin/env python3
"""Test script to verify data loading functionality."""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from data_loader_coordinator import DataCoordinator, load_cores_with_auto_fetch
    print("✅ Successfully imported DataCoordinator")
    
    # Test instantiation
    coordinator = DataCoordinator()
    print(f"✅ Successfully created DataCoordinator instance with db_path: {coordinator.db_path}")
    
    # Test method existence
    if hasattr(coordinator, 'load_cores_with_fetch'):
        print("✅ load_cores_with_fetch method exists")
    else:
        print("❌ load_cores_with_fetch method missing")
        
    if hasattr(coordinator, 'validate_cores_for_analysis'):
        print("✅ validate_cores_for_analysis method exists")
    else:
        print("❌ validate_cores_for_analysis method missing")
        
    print("\n🧪 Testing basic data loading...")
    
    # Test basic data loading (should handle missing data gracefully)
    test_tickers = ["AAPL"]
    test_start = "2025-01-01"
    test_end = "2025-01-02"
    
    cores = coordinator.load_cores_with_fetch(
        tickers=test_tickers,
        start=test_start,
        end=test_end,
        auto_fetch=False  # Don't fetch, just test the logic
    )
    
    print(f"✅ load_cores_with_fetch completed successfully")
    print(f"   Returned {len(cores)} cores")
    
    # Test validation
    validated = coordinator.validate_cores_for_analysis(cores)
    print(f"✅ validate_cores_for_analysis completed successfully")
    print(f"   Validated {len(validated)} cores")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
