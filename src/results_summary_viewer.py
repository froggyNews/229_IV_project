"""
Results Summary Viewer

This module provides clean, user-friendly ways to view high-level summaries 
of peer group analysis results, including correlations, peer effects, and 
statistical insights.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class PeerGroupSummaryViewer:
    """
    A comprehensive viewer for peer group analysis results.
    
    Provides clean, high-level summaries and visualizations of:
    - Intra-group correlations
    - Inter-group relationships
    - Peer effects analysis
    - Statistical significance
    """
    
    def __init__(self, results_path: Path):
        """
        Initialize the summary viewer.
        
        Parameters
        ----------
        results_path : Path
            Path to the peer_group_analysis.json file
        """
        self.results_path = Path(results_path)
        self.results = self._load_results()
        self.metadata = self.results.get("metadata", {})
        
    def _load_results(self) -> Dict[str, Any]:
        """Load results from JSON file."""
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Could not load results from {self.results_path}: {e}")
    
    def print_executive_summary(self) -> None:
        """Print a high-level executive summary."""
        print("=" * 80)
        print("PEER GROUP ANALYSIS - EXECUTIVE SUMMARY")
        print("=" * 80)
        
        # Basic info
        config = self.metadata.get("config", {})
        groups = self.metadata.get("groups", {})
        
        print(f"📅 Analysis Period: {config.get('start', 'N/A')} to {config.get('end', 'N/A')}")
        print(f"🏢 Total Groups: {len(groups)}")
        print(f"📊 Total Tickers: {self.metadata.get('total_tickers', 'N/A')}")
        print(f"⏰ Analysis Timestamp: {config.get('timestamp', 'N/A')}")
        
        print(f"\n📋 Groups Analyzed:")
        for group_name, tickers in groups.items():
            print(f"  {group_name}: {', '.join(tickers)} ({len(tickers)} tickers)")
        
        # Correlation summary
        self._print_correlation_summary()
        
        # Peer effects summary
        self._print_peer_effects_summary()
        
        # Key insights
        self._print_key_insights()
    
    def _print_correlation_summary(self) -> None:
        """Print correlation analysis summary."""
        print(f"\n🔗 CORRELATION ANALYSIS SUMMARY")
        print("-" * 50)
        
        intra_corr = self.results.get("intra_correlations", {})
        inter_corr = self.results.get("inter_correlations", {})
        
        # Intra-group correlations
        print("Within-Group Correlations:")
        for group_name, data in intra_corr.items():
            if "error" in data:
                print(f"  {group_name}: ❌ {data['error']}")
                continue
                
            iv_corr = data.get("iv_correlations", {}).get("mean", np.nan)
            ret_corr = data.get("iv_return_correlations", {}).get("mean", np.nan)
            n_tickers = data.get("n_tickers", 0)
            
            if not np.isnan(iv_corr):
                iv_str = f"{iv_corr:.3f}"
            else:
                iv_str = "N/A"
            
            if not np.isnan(ret_corr):
                ret_str = f"{ret_corr:.3f}"
            else:
                ret_str = "N/A"
            
            print(f"  {group_name} ({n_tickers} tickers): IV={iv_str}, Returns={ret_str}")
        
        # Inter-group correlations
        if inter_corr:
            print("\nBetween-Group Correlations:")
            for pair_name, data in inter_corr.items():
                if "error" in data:
                    print(f"  {pair_name}: ❌ {data['error']}")
                    continue
                    
                iv_corr = data.get("iv_cross_correlations", {}).get("mean", np.nan)
                ret_corr = data.get("iv_return_cross_correlations", {}).get("mean", np.nan)
                
                if not np.isnan(iv_corr):
                    iv_str = f"{iv_corr:.3f}"
                else:
                    iv_str = "N/A"
                
                if not np.isnan(ret_corr):
                    ret_str = f"{ret_corr:.3f}"
                else:
                    ret_str = "N/A"
                
                print(f"  {pair_name}: IV={iv_str}, Returns={ret_str}")
    
    def _print_peer_effects_summary(self) -> None:
        """Print peer effects analysis summary."""
        print(f"\n🎯 PEER EFFECTS ANALYSIS SUMMARY")
        print("-" * 50)
        
        intra_effects = self.results.get("intra_peer_effects", {})
        inter_effects = self.results.get("inter_peer_effects", {})
        
        # Intra-group peer effects
        print("Within-Group Peer Effects:")
        for group_name, data in intra_effects.items():
            if "error" in data:
                print(f"  {group_name}: ❌ {data['error']}")
                continue
            
            results = data.get("results", {})
            tickers = data.get("tickers", [])
            
            print(f"  {group_name} ({len(tickers)} tickers):")
            
            for target_kind, analysis in results.items():
                if "error" in analysis:
                    print(f"    {target_kind}: ❌ {analysis['error']}")
                    continue
                
                avg_r2 = analysis.get("avg_r2", np.nan)
                avg_rmse = analysis.get("avg_rmse", np.nan)
                successful = analysis.get("successful_targets", 0)
                total = analysis.get("target_count", 0)
                
                if not np.isnan(avg_r2):
                    r2_str = f"{avg_r2:.3f}"
                else:
                    r2_str = "N/A"
                
                if not np.isnan(avg_rmse):
                    rmse_str = f"{avg_rmse:.4f}"
                else:
                    rmse_str = "N/A"
                
                print(f"    {target_kind}: R²={r2_str}, RMSE={rmse_str} ({successful}/{total} successful)")
        
        # Inter-group peer effects
        if inter_effects:
            print("\nBetween-Group Peer Effects:")
            for pair_name, data in inter_effects.items():
                if "error" in data:
                    print(f"  {pair_name}: ❌ {data['error']}")
                    continue
                
                target_group = data.get("target_group", "")
                peer_group = data.get("peer_group", "")
                results = data.get("results", {})
                
                print(f"  {target_group} ← {peer_group}:")
                
                for target_kind, analysis in results.items():
                    cross_effects = analysis.get("cross_effect_summary", {})
                    target_tickers = analysis.get("target_tickers", [])
                    
                    if cross_effects:
                        # Show top cross-group effect
                        top_peer = list(cross_effects.keys())[0] if cross_effects else "None"
                        top_effect = cross_effects.get(top_peer, {}).get("mean_effect", 0)
                        print(f"    {target_kind}: Top effect from {top_peer} = {top_effect:.4f}")
                    else:
                        print(f"    {target_kind}: No significant cross effects")
    
    def _print_key_insights(self) -> None:
        """Print key insights and findings."""
        print(f"\n💡 KEY INSIGHTS")
        print("-" * 50)
        
        insights = []
        
        # Analyze correlations
        intra_corr = self.results.get("intra_correlations", {})
        
        # Find strongest intra-group correlations
        strongest_group = None
        strongest_corr = -1
        
        for group_name, data in intra_corr.items():
            if "error" in data:
                continue
            iv_corr = data.get("iv_correlations", {}).get("mean", np.nan)
            if not np.isnan(iv_corr) and abs(iv_corr) > abs(strongest_corr):
                strongest_corr = iv_corr
                strongest_group = group_name
        
        if strongest_group:
            insights.append(f"Strongest intra-group correlation: {strongest_group} ({strongest_corr:.3f})")
        
        # Find groups with negative correlations
        negative_groups = []
        for group_name, data in intra_corr.items():
            if "error" in data:
                continue
            iv_corr = data.get("iv_correlations", {}).get("mean", np.nan)
            if not np.isnan(iv_corr) and iv_corr < -0.01:  # Threshold for "meaningful" negative
                negative_groups.append(f"{group_name} ({iv_corr:.3f})")
        
        if negative_groups:
            insights.append(f"Groups with negative correlations: {', '.join(negative_groups)}")
        
        # Analyze peer effects performance
        intra_effects = self.results.get("intra_peer_effects", {})
        best_performing_group = None
        best_r2 = -1
        
        for group_name, data in intra_effects.items():
            if "error" in data:
                continue
            results = data.get("results", {})
            for target_kind, analysis in results.items():
                if "error" in analysis:
                    continue
                avg_r2 = analysis.get("avg_r2", np.nan)
                if not np.isnan(avg_r2) and avg_r2 > best_r2:
                    best_r2 = avg_r2
                    best_performing_group = f"{group_name} ({target_kind})"
        
        if best_performing_group:
            insights.append(f"Best peer effects model: {best_performing_group} (R²={best_r2:.3f})")
        
        # Check for cross-group effects
        inter_effects = self.results.get("inter_peer_effects", {})
        significant_cross_effects = []
        
        for pair_name, data in inter_effects.items():
            if "error" in data:
                continue
            results = data.get("results", {})
            for target_kind, analysis in results.items():
                cross_effects = analysis.get("cross_effect_summary", {})
                for peer, effect_data in cross_effects.items():
                    mean_effect = effect_data.get("mean_effect", 0)
                    if abs(mean_effect) > 0.01:  # Threshold for significant effect
                        significant_cross_effects.append(f"{pair_name} via {peer} ({mean_effect:.3f})")
        
        if significant_cross_effects:
            insights.append(f"Significant cross-group effects: {len(significant_cross_effects)} found")
        
        # Print insights
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        if not insights:
            print("No significant patterns identified in the current analysis.")
    
    def get_correlation_dataframe(self) -> pd.DataFrame:
        """Create a summary DataFrame of all correlations."""
        rows = []
        
        # Intra-group correlations
        intra_corr = self.results.get("intra_correlations", {})
        for group_name, data in intra_corr.items():
            if "error" in data:
                continue
                
            iv_corr = data.get("iv_correlations", {}).get("mean", np.nan)
            iv_std = data.get("iv_correlations", {}).get("std", np.nan)
            ret_corr = data.get("iv_return_correlations", {}).get("mean", np.nan)
            ret_std = data.get("iv_return_correlations", {}).get("std", np.nan)
            n_tickers = data.get("n_tickers", 0)
            
            rows.append({
                "Group_1": group_name,
                "Group_2": group_name,
                "Relationship": "Intra-group",
                "IV_Correlation_Mean": iv_corr,
                "IV_Correlation_Std": iv_std,
                "Return_Correlation_Mean": ret_corr,
                "Return_Correlation_Std": ret_std,
                "N_Tickers": n_tickers
            })
        
        # Inter-group correlations
        inter_corr = self.results.get("inter_correlations", {})
        for pair_name, data in inter_corr.items():
            if "error" in data:
                continue
                
            group1 = data.get("group1", "")
            group2 = data.get("group2", "")
            
            iv_corr = data.get("iv_cross_correlations", {}).get("mean", np.nan)
            iv_std = data.get("iv_cross_correlations", {}).get("std", np.nan)
            ret_corr = data.get("iv_return_cross_correlations", {}).get("mean", np.nan)
            ret_std = data.get("iv_return_cross_correlations", {}).get("std", np.nan)
            
            n_tickers = len(data.get("group1_tickers", [])) + len(data.get("group2_tickers", []))
            
            rows.append({
                "Group_1": group1,
                "Group_2": group2,
                "Relationship": "Inter-group",
                "IV_Correlation_Mean": iv_corr,
                "IV_Correlation_Std": iv_std,
                "Return_Correlation_Mean": ret_corr,
                "Return_Correlation_Std": ret_std,
                "N_Tickers": n_tickers
            })
        
        return pd.DataFrame(rows)
    
    def get_peer_effects_dataframe(self) -> pd.DataFrame:
        """Create a summary DataFrame of peer effects results."""
        rows = []
        
        # Intra-group peer effects
        intra_effects = self.results.get("intra_peer_effects", {})
        for group_name, data in intra_effects.items():
            if "error" in data:
                continue
                
            results = data.get("results", {})
            tickers = data.get("tickers", [])
            
            for target_kind, analysis in results.items():
                if "error" in analysis:
                    continue
                
                rows.append({
                    "Group": group_name,
                    "Target_Kind": target_kind,
                    "Analysis_Type": "Intra-group",
                    "Avg_R2": analysis.get("avg_r2", np.nan),
                    "Avg_RMSE": analysis.get("avg_rmse", np.nan),
                    "Successful_Targets": analysis.get("successful_targets", 0),
                    "Total_Targets": analysis.get("target_count", 0),
                    "N_Tickers": len(tickers)
                })
        
        # Inter-group peer effects (simplified)
        inter_effects = self.results.get("inter_peer_effects", {})
        for pair_name, data in inter_effects.items():
            if "error" in data:
                continue
                
            target_group = data.get("target_group", "")
            peer_group = data.get("peer_group", "")
            results = data.get("results", {})
            
            for target_kind, analysis in results.items():
                cross_effects = analysis.get("cross_effect_summary", {})
                
                # Get average cross effect
                cross_effects_values = [eff["mean_effect"] for eff in cross_effects.values()]
                avg_cross_effect = np.mean(cross_effects_values) if cross_effects_values else np.nan
                
                rows.append({
                    "Group": f"{target_group} ← {peer_group}",
                    "Target_Kind": target_kind,
                    "Analysis_Type": "Inter-group",
                    "Avg_R2": np.nan,  # Not directly comparable
                    "Avg_RMSE": np.nan,  # Not directly comparable
                    "Successful_Targets": len(analysis.get("target_tickers", [])),
                    "Total_Targets": len(analysis.get("target_tickers", [])),
                    "Avg_Cross_Effect": avg_cross_effect
                })
        
        return pd.DataFrame(rows)
    
    def print_detailed_group_analysis(self, group_name: str) -> None:
        """Print detailed analysis for a specific group."""
        print(f"\n{'=' * 60}")
        print(f"DETAILED ANALYSIS: {group_name.upper()}")
        print(f"{'=' * 60}")
        
        # Group composition
        groups = self.metadata.get("groups", {})
        if group_name not in groups:
            print(f"❌ Group '{group_name}' not found in analysis.")
            return
        
        tickers = groups[group_name]
        print(f"📊 Tickers: {', '.join(tickers)} ({len(tickers)} total)")
        
        # Intra-group correlations
        intra_corr = self.results.get("intra_correlations", {}).get(group_name, {})
        if "error" not in intra_corr:
            print(f"\n🔗 Intra-Group Correlations:")
            
            iv_corr_data = intra_corr.get("iv_correlations", {})
            ret_corr_data = intra_corr.get("iv_return_correlations", {})
            
            iv_mean = iv_corr_data.get('mean', np.nan)
            iv_std = iv_corr_data.get('std', np.nan)
            ret_mean = ret_corr_data.get('mean', np.nan)
            ret_std = ret_corr_data.get('std', np.nan)
            
            iv_mean_str = f"{iv_mean:.3f}" if not np.isnan(iv_mean) else "N/A"
            iv_std_str = f"{iv_std:.3f}" if not np.isnan(iv_std) else "N/A"
            ret_mean_str = f"{ret_mean:.3f}" if not np.isnan(ret_mean) else "N/A"
            ret_std_str = f"{ret_std:.3f}" if not np.isnan(ret_std) else "N/A"
            
            print(f"  IV Correlations: mean={iv_mean_str}, std={iv_std_str}")
            print(f"  Return Correlations: mean={ret_mean_str}, std={ret_std_str}")
            
            # Detailed correlation matrix
            iv_matrix = iv_corr_data.get("matrix", {})
            if iv_matrix:
                print(f"\n  Detailed IV Correlation Matrix:")
                for ticker1, row in iv_matrix.items():
                    for ticker2, corr in row.items():
                        if ticker1 != ticker2:  # Skip diagonal
                            print(f"    {ticker1} ↔ {ticker2}: {corr:.3f}")
        
        # Peer effects
        intra_effects = self.results.get("intra_peer_effects", {}).get(group_name, {})
        if "error" not in intra_effects:
            print(f"\n🎯 Peer Effects Analysis:")
            
            results = intra_effects.get("results", {})
            for target_kind, analysis in results.items():
                if "error" in analysis:
                    continue
                
                print(f"\n  {target_kind.upper()} Analysis:")
                
                avg_r2 = analysis.get('avg_r2', np.nan)
                avg_rmse = analysis.get('avg_rmse', np.nan)
                
                r2_str = f"{avg_r2:.3f}" if not np.isnan(avg_r2) else "N/A"
                rmse_str = f"{avg_rmse:.4f}" if not np.isnan(avg_rmse) else "N/A"
                
                print(f"    Average R²: {r2_str}")
                print(f"    Average RMSE: {rmse_str}")
                print(f"    Success Rate: {analysis.get('successful_targets', 0)}/{analysis.get('target_count', 0)}")
                
                # Top peer effects
                peer_effects = analysis.get("peer_effect_summary", {})
                if peer_effects:
                    print(f"    Top Peer Effects:")
                    for i, (peer, effect_data) in enumerate(list(peer_effects.items())[:3], 1):
                        effect = effect_data.get("mean_effect", 0)
                        appearances = effect_data.get("appearances", 0)
                        print(f"      {i}. {peer}: {effect:.4f} (appears {appearances} times)")
        
        # Cross-group effects involving this group
        self._print_cross_group_effects_for_group(group_name)
    
    def _print_cross_group_effects_for_group(self, group_name: str) -> None:
        """Print cross-group effects involving a specific group."""
        inter_effects = self.results.get("inter_peer_effects", {})
        
        # Effects ON this group
        incoming_effects = []
        outgoing_effects = []
        
        for pair_name, data in inter_effects.items():
            if "error" in data:
                continue
                
            target_group = data.get("target_group", "")
            peer_group = data.get("peer_group", "")
            
            if target_group == group_name:
                incoming_effects.append((peer_group, data))
            elif peer_group == group_name:
                outgoing_effects.append((target_group, data))
        
        if incoming_effects:
            print(f"\n🔽 Effects ON {group_name} from other groups:")
            for peer_group, data in incoming_effects:
                results = data.get("results", {})
                for target_kind, analysis in results.items():
                    cross_effects = analysis.get("cross_effect_summary", {})
                    if cross_effects:
                        top_effect = list(cross_effects.values())[0].get("mean_effect", 0)
                        print(f"  From {peer_group} ({target_kind}): {top_effect:.4f}")
        
        if outgoing_effects:
            print(f"\n🔼 Effects FROM {group_name} to other groups:")
            for target_group, data in outgoing_effects:
                # This is more complex as we need to find effects from our group's tickers
                results = data.get("results", {})
                our_tickers = self.metadata.get("groups", {}).get(group_name, [])
                
                for target_kind, analysis in results.items():
                    cross_effects = analysis.get("cross_effect_summary", {})
                    our_effects = {ticker: effect for ticker, effect in cross_effects.items() 
                                 if ticker in our_tickers}
                    
                    if our_effects:
                        avg_effect = np.mean([eff["mean_effect"] for eff in our_effects.values()])
                        print(f"  To {target_group} ({target_kind}): {avg_effect:.4f}")
    
    def save_summary_report(self, output_path: Optional[Path] = None) -> Path:
        """Save a comprehensive summary report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.results_path.parent / f"summary_report_{timestamp}.txt"
        
        # Capture print output
        import io
        from contextlib import redirect_stdout
        
        output = io.StringIO()
        
        with redirect_stdout(output):
            self.print_executive_summary()
            
            # Add detailed analysis for each group
            groups = self.metadata.get("groups", {})
            for group_name in groups.keys():
                self.print_detailed_group_analysis(group_name)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(output.getvalue())
        
        print(f"📄 Summary report saved to: {output_path}")
        return output_path


# Convenience functions
def view_latest_results(results_dir: Path = Path("outputs/peer_groups")) -> PeerGroupSummaryViewer:
    """Load and view the latest peer group analysis results."""
    # Find the most recent results directory
    if not results_dir.exists():
        raise ValueError(f"Results directory not found: {results_dir}")
    
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No results found in {results_dir}")
    
    latest_dir = max(subdirs, key=lambda d: d.name)
    results_file = latest_dir / "peer_group_analysis.json"
    
    if not results_file.exists():
        raise ValueError(f"Results file not found: {results_file}")
    
    return PeerGroupSummaryViewer(results_file)


def quick_summary(results_path: Optional[Path] = None) -> None:
    """Print a quick summary of the latest or specified results."""
    if results_path is None:
        viewer = view_latest_results()
    else:
        viewer = PeerGroupSummaryViewer(results_path)
    
    viewer.print_executive_summary()


def detailed_group_report(group_name: str, results_path: Optional[Path] = None) -> None:
    """Print a detailed report for a specific group."""
    if results_path is None:
        viewer = view_latest_results()
    else:
        viewer = PeerGroupSummaryViewer(results_path)
    
    viewer.print_detailed_group_analysis(group_name)


if __name__ == "__main__":
    # Example usage
    try:
        print("🔍 Loading latest peer group analysis results...")
        viewer = view_latest_results()
        viewer.print_executive_summary()
        
        # Save summary report
        report_path = viewer.save_summary_report()
        print(f"\n📄 Detailed report saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTo use this module:")
        print("1. Ensure you have peer group analysis results")
        print("2. Run: python src/results_summary_viewer.py")
        print("3. Or use: quick_summary() function in Python")
