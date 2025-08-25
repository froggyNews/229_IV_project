#!/usr/bin/env python3
"""
Command-line interface for viewing peer group analysis results.

This script provides easy command-line access to peer group analysis summaries
with various viewing options and output formats.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from results_summary_viewer import (
    PeerGroupSummaryViewer, 
    view_latest_results, 
    quick_summary,
    detailed_group_report
)


def main():
    parser = argparse.ArgumentParser(
        description="View peer group analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # View latest results summary
  %(prog)s --detailed quantum       # Detailed view of quantum group
  %(prog)s --file results.json      # View specific results file
  %(prog)s --export --save-report   # Export data and save report
  %(prog)s --list-groups             # List available groups
        """
    )
    
    # Input options
    parser.add_argument(
        "--file", "-f", 
        type=Path,
        help="Specific results file to analyze (default: latest)"
    )
    
    parser.add_argument(
        "--results-dir", 
        type=Path, 
        default=Path("outputs/peer_groups"),
        help="Directory containing results (default: outputs/peer_groups)"
    )
    
    # View options
    parser.add_argument(
        "--detailed", "-d", 
        type=str,
        help="Show detailed analysis for specific group"
    )
    
    parser.add_argument(
        "--list-groups", "-l",
        action="store_true",
        help="List all available groups"
    )
    
    parser.add_argument(
        "--correlations-only", "-c",
        action="store_true",
        help="Show only correlation analysis"
    )
    
    parser.add_argument(
        "--peer-effects-only", "-p",
        action="store_true",
        help="Show only peer effects analysis"
    )
    
    # Export options
    parser.add_argument(
        "--export", "-e",
        action="store_true",
        help="Export data to CSV files"
    )
    
    parser.add_argument(
        "--save-report", "-s",
        action="store_true",
        help="Save summary report to file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for exports (default: same as results)"
    )
    
    # Display options
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact display format"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress status messages"
    )
    
    args = parser.parse_args()
    
    try:
        # Load results
        if not args.quiet:
            print("🔍 Loading peer group analysis results...")
        
        if args.file:
            if not args.file.exists():
                print(f"❌ Results file not found: {args.file}")
                sys.exit(1)
            viewer = PeerGroupSummaryViewer(args.file)
        else:
            viewer = view_latest_results(args.results_dir)
        
        if not args.quiet:
            print(f"✅ Loaded results from: {viewer.results_path}")
        
        # Handle list groups
        if args.list_groups:
            groups = viewer.metadata.get("groups", {})
            print(f"\n📋 Available Groups ({len(groups)} total):")
            for group_name, tickers in groups.items():
                print(f"  {group_name}: {', '.join(tickers)} ({len(tickers)} tickers)")
            return
        
        # Handle detailed group view
        if args.detailed:
            if not args.quiet:
                print(f"📊 Showing detailed analysis for: {args.detailed}")
            viewer.print_detailed_group_analysis(args.detailed)
            return
        
        # Handle specific analysis types
        if args.correlations_only:
            print("🔗 CORRELATION ANALYSIS")
            print("=" * 50)
            viewer._print_correlation_summary()
            return
        
        if args.peer_effects_only:
            print("🎯 PEER EFFECTS ANALYSIS")
            print("=" * 50)
            viewer._print_peer_effects_summary()
            return
        
        # Default: show executive summary
        if args.compact:
            # Compact version - just key numbers
            print_compact_summary(viewer)
        else:
            viewer.print_executive_summary()
        
        # Handle exports
        if args.export:
            export_data(viewer, args.output_dir or viewer.results_path.parent, args.quiet)
        
        if args.save_report:
            report_path = viewer.save_summary_report(
                args.output_dir / "summary_report.txt" if args.output_dir else None
            )
            if not args.quiet:
                print(f"📄 Summary report saved to: {report_path}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        if not args.quiet:
            print("\nTroubleshooting:")
            print("1. Ensure peer group analysis has been run")
            print("2. Check that results directory exists")
            print("3. Use --file to specify exact results file")
        sys.exit(1)


def print_compact_summary(viewer: PeerGroupSummaryViewer):
    """Print a compact summary of results."""
    metadata = viewer.metadata
    groups = metadata.get("groups", {})
    
    print("📊 PEER GROUP ANALYSIS - COMPACT SUMMARY")
    print("-" * 45)
    
    # Basic stats
    print(f"Period: {metadata.get('config', {}).get('start', 'N/A')} to {metadata.get('config', {}).get('end', 'N/A')}")
    print(f"Groups: {len(groups)}, Tickers: {metadata.get('total_tickers', 'N/A')}")
    
    # Correlation highlights
    intra_corr = viewer.results.get("intra_correlations", {})
    correlations = []
    
    for group_name, data in intra_corr.items():
        if "error" in data:
            continue
        iv_corr = data.get("iv_correlations", {}).get("mean", None)
        if iv_corr is not None:
            correlations.append((group_name, iv_corr))
    
    if correlations:
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\nTop Correlations:")
        for group, corr in correlations[:3]:
            print(f"  {group}: {corr:+.3f}")
    
    # Peer effects highlights
    intra_effects = viewer.results.get("intra_peer_effects", {})
    best_models = []
    
    for group_name, data in intra_effects.items():
        if "error" in data:
            continue
        results = data.get("results", {})
        for target_kind, analysis in results.items():
            if "error" in analysis:
                continue
            r2 = analysis.get("avg_r2", None)
            if r2 is not None:
                best_models.append((f"{group_name}_{target_kind}", r2))
    
    if best_models:
        best_models.sort(key=lambda x: x[1], reverse=True)
        print(f"\nBest Models (R²):")
        for model, r2 in best_models[:3]:
            print(f"  {model}: {r2:.3f}")


def export_data(viewer: PeerGroupSummaryViewer, output_dir: Path, quiet: bool = False):
    """Export analysis data to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Export correlations
        corr_df = viewer.get_correlation_dataframe()
        corr_path = output_dir / "correlations_summary.csv"
        corr_df.to_csv(corr_path, index=False)
        if not quiet:
            print(f"📊 Correlations exported to: {corr_path}")
        
        # Export peer effects
        effects_df = viewer.get_peer_effects_dataframe()
        effects_path = output_dir / "peer_effects_summary.csv"
        effects_df.to_csv(effects_path, index=False)
        if not quiet:
            print(f"🎯 Peer effects exported to: {effects_path}")
        
        # Export metadata
        metadata_path = output_dir / "analysis_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(viewer.metadata, f, indent=2, default=str)
        if not quiet:
            print(f"📋 Metadata exported to: {metadata_path}")
    
    except Exception as e:
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    main()
