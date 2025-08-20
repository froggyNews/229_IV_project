"""
Data loader coordinator - helps orchestrate data loading across existing modules.

This module doesn't duplicate functionality but provides a clean interface
to coordinate between feature_engineering.py, fetch_data_sqlite.py, and 
train_peer_effects.py.
"""

import os
from pathlib import Path
from typing import Dict, Sequence, Optional

import pandas as pd

# Import existing functions
from fetch_data_sqlite import fetch_and_save
from feature_engineering import load_ticker_core

class DataCoordinator:
    """Coordinates data loading and ensures consistency across modules."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
        self.api_key = os.getenv("DATABENTO_API_KEY")
        
    def load_cores_with_fetch(
        self, 
        tickers: Sequence[str], 
        start: str, 
        end: str,
        auto_fetch: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load ticker cores, automatically fetching missing data if possible.
        
        This is the main coordination function that:
        1. Tries to load existing data using feature_engineering.load_ticker_core
        2. If data is missing and auto_fetch=True, uses fetch_data_sqlite.fetch_and_save
        3. Returns a clean cores dict for use across all modules
        """
        cores = {}
        missing_tickers = []
        
        print(f"Loading cores for {len(tickers)} tickers...")
        
        # First pass: try to load existing data
        for ticker in tickers:
            try:
                core = load_ticker_core(ticker, start=start, end=end, db_path=self.db_path)
                if not core.empty:
                    cores[ticker] = core
                    print(f"  ✓ {ticker}: {len(core):,} rows")
                else:
                    missing_tickers.append(ticker)
                    print(f"  ✗ {ticker}: no data found")
            except Exception as e:
                print(f"  ✗ {ticker}: error loading ({e})")
                missing_tickers.append(ticker)
        
        # Second pass: fetch missing data if enabled and API key available
        if missing_tickers and auto_fetch and self.api_key:
            print(f"Auto-fetching {len(missing_tickers)} missing tickers...")
            
            start_ts = pd.Timestamp(start, tz="UTC")
            end_ts = pd.Timestamp(end, tz="UTC")
            
            for ticker in missing_tickers:
                try:
                    print(f"  Fetching {ticker}...")
                    fetch_and_save(self.api_key, ticker, start_ts, end_ts, self.db_path, force=False)
                    
                    # Retry loading after fetch
                    core = load_ticker_core(ticker, start=start, end=end, db_path=self.db_path)
                    if not core.empty:
                        cores[ticker] = core
                        print(f"    ✓ Fetched {ticker}: {len(core):,} rows")
                    else:
                        print(f"    ✗ {ticker}: no data even after fetch")
                        
                except Exception as e:
                    print(f"    ✗ {ticker}: fetch failed ({e})")
                    continue
                    
        elif missing_tickers and not self.api_key:
            print("Warning: Missing tickers but no DATABENTO_API_KEY for auto-fetch")
            
        print(f"Final result: {len(cores)}/{len(tickers)} tickers loaded")
        return cores
    
    def validate_cores_for_analysis(
        self, 
        cores: Dict[str, pd.DataFrame], 
        analysis_type: str = "general"
    ) -> Dict[str, pd.DataFrame]:
        """
        Validate cores are suitable for specific analysis types.
        
        This ensures consistent validation logic across modules.
        """
        valid_cores = {}
        
        for ticker, core in cores.items():
            # Basic validation (from feature_engineering._valid_core logic)
            if core is None or core.empty:
                print(f"Skipping {ticker}: empty core")
                continue
                
            if not {"ts_event", "iv_clip"}.issubset(core.columns):
                print(f"Skipping {ticker}: missing required columns")
                continue
                
            # Analysis-specific validation
            if analysis_type == "peer_effects":
                # Need sufficient data for train/test split
                if len(core) < 100:
                    print(f"Skipping {ticker}: insufficient data for peer effects ({len(core)} rows)")
                    continue
                    
            elif analysis_type == "pooled":
                # Need at least some data to contribute to pool
                if len(core) < 50:
                    print(f"Skipping {ticker}: insufficient data for pooling ({len(core)} rows)")
                    continue
            
            valid_cores[ticker] = core
            
        return valid_cores
    
    def get_analysis_summary(self, cores: Dict[str, pd.DataFrame]) -> Dict:
        """Get summary statistics for loaded cores."""
        if not cores:
            return {"status": "no_data"}
            
        summary = {
            "n_tickers": len(cores),
            "tickers": list(cores.keys()),
            "total_rows": sum(len(df) for df in cores.values()),
            "date_ranges": {},
            "avg_rows_per_ticker": sum(len(df) for df in cores.values()) // len(cores)
        }
        
        # Get date ranges for each ticker
        for ticker, core in cores.items():
            if not core.empty and "ts_event" in core.columns:
                dates = pd.to_datetime(core["ts_event"])
                summary["date_ranges"][ticker] = {
                    "start": dates.min().strftime("%Y-%m-%d"),
                    "end": dates.max().strftime("%Y-%m-%d"),
                    "rows": len(core)
                }
        
        return summary


# Convenience functions for backward compatibility
def load_cores_with_auto_fetch(
    tickers: Sequence[str], 
    start: str, 
    end: str, 
    db_path: Optional[Path] = None,
    auto_fetch: bool = True
) -> Dict[str, pd.DataFrame]:
    """Convenience function that wraps DataCoordinator for simple usage."""
    coordinator = DataCoordinator(db_path)
    return coordinator.load_cores_with_fetch(tickers, start, end, auto_fetch)


def validate_cores(
    cores: Dict[str, pd.DataFrame], 
    analysis_type: str = "general"
) -> Dict[str, pd.DataFrame]:
    """Convenience function for core validation."""
    coordinator = DataCoordinator()
    return coordinator.validate_cores_for_analysis(cores, analysis_type)


# Integration helpers for existing modules
class AnalysisConfig:
    """Shared configuration that works across all analysis types."""
    
    def __init__(
        self,
        tickers: Sequence[str],
        start: str,
        end: str,
        db_path: Optional[Path] = None,
        forward_steps: int = 15,
        test_frac: float = 0.2,
        tolerance: str = "2s"
    ):
        self.tickers = list(tickers)
        self.start = start
        self.end = end
        self.db_path = db_path or Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
        self.forward_steps = forward_steps
        self.test_frac = test_frac
        self.tolerance = tolerance
        
        # Load cores once for all analyses
        self.coordinator = DataCoordinator(self.db_path)
        self.cores = self.coordinator.load_cores_with_fetch(
            self.tickers, self.start, self.end, auto_fetch=True
        )
        
    def get_pooled_cores(self) -> Dict[str, pd.DataFrame]:
        """Get cores validated for pooled analysis."""
        return self.coordinator.validate_cores_for_analysis(self.cores, "pooled")
        
    def get_peer_cores(self) -> Dict[str, pd.DataFrame]:
        """Get cores validated for peer effects analysis.""" 
        return self.coordinator.validate_cores_for_analysis(self.cores, "peer_effects")
    
    def summary(self) -> Dict:
        """Get summary of loaded data."""
        return self.coordinator.get_analysis_summary(self.cores)


# Example integration with existing modules
def run_integrated_analysis(config: AnalysisConfig):
    """Example of how to use the coordinator with existing modules."""
    
    print("=== Integrated Analysis ===")
    print(f"Config: {config.summary()}")
    
    # 1. Pooled analysis using existing feature_engineering functions
    from feature_engineering import build_pooled_iv_return_dataset_time_safe
    
    pooled_cores = config.get_pooled_cores()
    if pooled_cores:
        print(f"\nRunning pooled analysis with {len(pooled_cores)} tickers...")
        pooled_data = build_pooled_iv_return_dataset_time_safe(
            tickers=list(pooled_cores.keys()),
            start=config.start,
            end=config.end,
            forward_steps=config.forward_steps,
            tolerance=config.tolerance,
            db_path=config.db_path,
            cores=pooled_cores  # Pass pre-loaded cores
        )
        print(f"Pooled dataset: {len(pooled_data):,} rows")
    
    # 2. Peer effects using existing train_peer_effects functions
    from train_peer_effects import PeerEffectsConfig, run_peer_effects
    from feature_engineering import build_target_peer_dataset
    
    peer_cores = config.get_peer_cores()
    if len(peer_cores) >= 2:  # Need at least target + 1 peer
        print(f"\nRunning peer effects with {len(peer_cores)} tickers...")
        
        # Example: analyze first ticker vs others
        target = list(peer_cores.keys())[0]
        
        # Use existing PeerEffectsConfig with pre-loaded cores
        peer_config = PeerEffectsConfig(
            target=target,
            tickers=list(peer_cores.keys()),
            start=config.start,
            end=config.end,
            db_path=str(config.db_path),
            forward_steps=config.forward_steps,
            test_frac=config.test_frac
        )
        
        # Build dataset using existing function with cores
        dataset = build_target_peer_dataset(
            target=target,
            tickers=list(peer_cores.keys()),
            start=config.start,
            end=config.end,
            forward_steps=config.forward_steps,
            db_path=config.db_path,
            cores=peer_cores  # Pass pre-loaded cores
        )
        
        if not dataset.empty:
            result = run_peer_effects(peer_config, dataset)
            print(f"Peer effects for {target}: {result.get('status', 'completed')}")


if __name__ == "__main__":
    # Example usage
    config = AnalysisConfig(
        tickers=["QUBT", "QBTS", "RGTI", "IONQ"],
        start="2025-08-02",
        end="2025-08-06"
    )
    
    run_integrated_analysis(config)