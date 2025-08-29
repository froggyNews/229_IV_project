#!/usr/bin/env python3
"""
Test script for enhanced auto-fetch functionality.

This script demonstrates the new automatic data fetching capabilities
integrated into the peer group analysis workflow.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peer_group_analyzer import PeerGroupAnalyzer, PeerGroupConfig
from fetch_data_sqlite import auto_fetch_missing_data, ensure_data_availability
import pandas as pd


def test_basic_auto_fetch():
    """Test basic auto-fetch functionality."""
    
    print("🧪 Testing Basic Auto-Fetch Functionality")
    print("=" * 50)
    
    # Define test parameters
    tickers = ["ASTS", "SATS", "VZ", "T"]
    start = pd.Timestamp("2025-08-02", tz="UTC")
    end = pd.Timestamp("2025-08-06", tz="UTC")
    db_path = Path("data/iv_data_1h.db")
    
    try:
        # Test the auto_fetch_missing_data function
        print(f"📥 Testing auto-fetch for {len(tickers)} tickers...")
        results = auto_fetch_missing_data(tickers, start, end, db_path)
        
        print(f"\n📊 Fetch Results:")
        print(f"  ✅ Fetched: {len(results['fetched'])} tickers")
        print(f"  ⏭️  Skipped: {len(results['skipped'])} tickers")
        print(f"  ❌ Failed: {len(results['failed'])} tickers")
        
        if results['fetched']:
            print(f"  📥 Newly fetched: {', '.join(results['fetched'])}")
        if results['skipped']:
            print(f"  ⏭️  Already available: {', '.join(results['skipped'])}")
        if results['failed']:
            print(f"  ❌ Failed to fetch: {', '.join(results['failed'])}")
        
        return len(results['failed']) == 0
        
    except Exception as e:
        print(f"❌ Basic auto-fetch test failed: {e}")
        return False


def test_ensure_data_availability():
    """Test the ensure_data_availability function."""
    
    print("\n🧪 Testing Data Availability Assurance")
    print("=" * 50)
    
    tickers = ["ASTS", "SATS", "VZ", "T"]
    start = pd.Timestamp("2025-08-02", tz="UTC")
    end = pd.Timestamp("2025-08-06", tz="UTC")
    db_path = Path("data/iv_data_1h.db")
    
    try:
        print(f"🔍 Ensuring data availability for {len(tickers)} tickers...")
        success = ensure_data_availability(tickers, start, end, db_path, auto_fetch=True)
        
        if success:
            print("✅ All data successfully ensured to be available")
        else:
            print("⚠️  Some data could not be ensured - check logs above")
        
        return success
        
    except Exception as e:
        print(f"❌ Data availability test failed: {e}")
        return False


def test_peer_group_auto_fetch():
    """Test auto-fetch integration in peer group analysis."""
    
    print("\n🧪 Testing Peer Group Analysis with Auto-Fetch")
    print("=" * 50)
    
    try:
        # Create a configuration that includes some potentially missing tickers
        config = PeerGroupConfig(
            groups={
                "satellite": ["ASTS", "SATS"],
                "telecom": ["VZ", "T"],
                "test_group": ["ASTS", "VZ"]  # Mixed group for testing
            },
            start="2025-08-02",
            end="2025-08-06",
            auto_fetch=True,  # Enable auto-fetch
            save_detailed_results=False,  # Don't save files during test
            debug=True
        )
        
        # Create analyzer
        analyzer = PeerGroupAnalyzer(config)
        
        # Test just the data loading part
        print("📊 Testing data loading with auto-fetch...")
        analyzer.load_data()
        
        if analyzer.cores:
            print(f"✅ Successfully loaded data for {len(analyzer.cores)} tickers")
            
            # Quick validation
            total_rows = sum(len(df) for df in analyzer.cores.values())
            print(f"📊 Total rows loaded: {total_rows:,}")
            
            return True
        else:
            print("❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Peer group auto-fetch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantum_crypto_groups():
    """Test with quantum and crypto groups from the updated example."""
    
    print("\n🧪 Testing Extended Groups (Quantum & Crypto)")
    print("=" * 50)
    
    try:
        config = PeerGroupConfig(
            groups={
                "satellite": ["ASTS", "SATS"],
                "telecom": ["VZ", "T"],
                "quantum": ["QUBT", "QBTS", "RGTI", "IONQ"],
                "crypto_miners": ["MARA", "WULF", "IREN"],
            },
            start="2025-08-02",
            end="2025-08-06",
            auto_fetch=True,
            save_detailed_results=False,
            debug=True
        )
        
        analyzer = PeerGroupAnalyzer(config)
        
        # Test data loading for extended groups
        print("📊 Testing data loading for extended ticker groups...")
        analyzer.load_data()
        
        if analyzer.cores:
            print(f"✅ Loaded data for {len(analyzer.cores)} out of {len(analyzer.all_tickers)} requested tickers")
            
            # Show which groups have sufficient data
            for group_name, group_tickers in config.groups.items():
                available = [t for t in group_tickers if t in analyzer.cores]
                print(f"  {group_name}: {len(available)}/{len(group_tickers)} tickers available")
            
            return len(analyzer.cores) > 0
        else:
            print("❌ No data loaded for extended groups")
            return False
            
    except Exception as e:
        print(f"❌ Extended groups test failed: {e}")
        return False


def main():
    """Run all auto-fetch tests."""
    
    print("🚀 AUTO-FETCH FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        ("Basic Auto-Fetch", test_basic_auto_fetch),
        ("Data Availability Assurance", test_ensure_data_availability),
        ("Peer Group Integration", test_peer_group_auto_fetch),
        ("Extended Groups", test_quantum_crypto_groups),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        try:
            success = test_func()
            results[test_name] = "✅ PASSED" if success else "❌ FAILED"
        except Exception as e:
            print(f"💥 Test '{test_name}' crashed: {e}")
            results[test_name] = "💥 CRASHED"
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        print(f"  {result} {test_name}")
    
    # Overall result
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Auto-fetch functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
