#!/usr/bin/env python3
"""
Example script demonstrating peer group analysis functionality.

This script shows how to use the comprehensive peer group analyzer to:
1. Define multiple peer groups
2. Analyze intra-group correlations and peer effects
3. Analyze inter-group relationships
4. Generate organized reports

Usage:
    python examples/peer_group_example.py

Or with custom groups:
    python src/main_runner.py --enable-peer-group-analysis \
        --peer-groups "satellite:ASTS,SATS" "telecom:VZ,T" \
        --tickers ASTS SATS VZ T \
        --start 2025-08-02 --end 2025-08-06
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peer_group_analyzer import PeerGroupAnalyzer, PeerGroupConfig
from main_runner import RunConfig, run_pipeline


def example_standalone_peer_group_analysis():
    """Example of running peer group analysis standalone."""
    
    print("🚀 Running Standalone Peer Group Analysis Example")
    
    # Define peer groups
    groups = {
        "satellite": ["ASTS", "SATS"],
        "telecom": ["VZ", "T"],
        "tech_adjacent": ["ASTS", "VZ"],  # Mixed group for comparison
        "all_combined": ["ASTS", "SATS", "VZ", "T"]
    }
    
    # Create configuration
    config = PeerGroupConfig(
        groups=groups,
        start="2025-08-02",
        end="2025-08-06",
        target_kinds=["iv_ret", "iv"],
        forward_steps=15,
        test_frac=0.2,
        tolerance="15s",
        r=0.045,
        output_dir=Path("outputs/peer_group_examples"),
        save_detailed_results=True,
        debug=True
    )
    
    # Run analysis
    try:
        analyzer = PeerGroupAnalyzer(config)
        results = analyzer.run_full_analysis()
        
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Results summary:")
        print(f"  - Groups analyzed: {len(config.groups)}")
        print(f"  - Tickers involved: {len(analyzer.all_tickers)}")
        print(f"  - Output directory: {analyzer.output_dir}")
        
        # Print some key findings
        if "intra_correlations" in results:
            print(f"\n📈 Intra-group correlation summary:")
            for group_name, data in results["intra_correlations"].items():
                if "error" not in data:
                    iv_mean = data.get("iv_correlations", {}).get("mean", "N/A")
                    ret_mean = data.get("iv_return_correlations", {}).get("mean", "N/A")
                    print(f"  {group_name}: IV corr={iv_mean:.3f}, Return corr={ret_mean:.3f}")
        
        if "inter_correlations" in results:
            print(f"\n🔄 Inter-group correlation summary:")
            for pair_name, data in results["inter_correlations"].items():
                if "error" not in data:
                    iv_mean = data.get("iv_cross_correlations", {}).get("mean", "N/A")
                    ret_mean = data.get("iv_return_cross_correlations", {}).get("mean", "N/A")
                    print(f"  {pair_name}: IV cross-corr={iv_mean:.3f}, Return cross-corr={ret_mean:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise


def example_integrated_with_main_runner():
    """Example of running peer group analysis integrated with main runner."""
    
    print("\n🚀 Running Integrated Peer Group Analysis Example")
    
    # Create configuration that includes peer group analysis
    config = RunConfig(
        tickers=["ASTS", "VZ", "T", "SATS"],
        start="2025-08-02",
        end="2025-08-06",
        forward_steps=15,
        test_frac=0.2,
        tolerance="15s",
        r=0.045,
        
        # Enable baseline correlations
        compute_baseline_correlations=True,
        
        # Enable peer group analysis
        enable_peer_group_analysis=True,
        peer_groups={
            "satellite": ["ASTS", "SATS"],
            "telecom": ["VZ", "T"],
            "mixed": ["ASTS", "VZ", "T", "SATS"]
        },
        
        # Peer effects settings
        peer_targets=["ASTS", "VZ", "T", "SATS"],
        peer_target_kinds=["iv_ret", "iv"],
        
        # Output settings
        output_dir=Path("outputs/integrated_example"),
        debug=True
    )
    
    try:
        # Run complete pipeline
        results = run_pipeline(config)
        
        print("\n✅ Integrated analysis completed successfully!")
        print(f"📊 Pipeline results include:")
        for key in results.keys():
            print(f"  - {key}")
        
        # Print peer group analysis summary
        if "peer_group_results" in results and results["peer_group_results"]:
            peer_results = results["peer_group_results"]
            if "metadata" in peer_results:
                groups = peer_results["metadata"].get("groups", {})
                print(f"\n🎯 Peer group analysis covered {len(groups)} groups:")
                for group_name, tickers in groups.items():
                    print(f"  {group_name}: {', '.join(tickers)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Integrated analysis failed: {e}")
        raise


def main():
    """Run both examples."""
    
    print("=" * 60)
    print("PEER GROUP ANALYSIS EXAMPLES")
    print("=" * 60)
    
    # Example 1: Standalone peer group analysis
    try:
        standalone_results = example_standalone_peer_group_analysis()
        print(f"\n✅ Standalone example completed")
    except Exception as e:
        print(f"\n❌ Standalone example failed: {e}")
        standalone_results = None
    
    print("\n" + "=" * 60)
    
    # Example 2: Integrated with main runner
    try:
        integrated_results = example_integrated_with_main_runner()
        print(f"\n✅ Integrated example completed")
    except Exception as e:
        print(f"\n❌ Integrated example failed: {e}")
        integrated_results = None
    
    print("\n" + "=" * 60)
    print("EXAMPLES SUMMARY")
    print("=" * 60)
    
    if standalone_results:
        print("✅ Standalone peer group analysis: SUCCESS")
    else:
        print("❌ Standalone peer group analysis: FAILED")
    
    if integrated_results:
        print("✅ Integrated pipeline analysis: SUCCESS")
    else:
        print("❌ Integrated pipeline analysis: FAILED")
    
    print(f"\n📁 Check output directories for detailed results:")
    print(f"  - outputs/peer_group_examples/")
    print(f"  - outputs/integrated_example/")


if __name__ == "__main__":
    main()
