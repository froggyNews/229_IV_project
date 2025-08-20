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
    target_kind: str = "iv_ret"          # "iv_ret" (returns) or "iv" (levels)
    forward_steps: int = 15              # How many steps ahead to predict
    test_frac: float = 0.2               # Test set fraction
    tolerance: float = 1e-6              # Tolerance for numerical stability
    # Peer effects settings
    include_self_lag: bool = True        # Include target's own lagged features
    exclude_contemporaneous: bool = True  # Exclude same-time target features (avoid leakage)
    
    # Output
    output_dir: Path = Path("peer_analysis")
    save_details: bool = False           # Save detailed analysis


def prepare_peer_dataset(cfg: PeerConfig, cores: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build and clean dataset for peer effects analysis."""
    
    print(f"Building peer dataset for target: {cfg.target}")
    
    # Build base dataset
    dataset = build_target_peer_dataset(
        target=cfg.target,
        tickers=cfg.tickers,
        start=cfg.start,
        end=cfg.end,
        forward_steps=cfg.forward_steps,
        db_path=cfg.db_path,
        target_kind=cfg.target_kind,
        cores=cores
    )
    
    if dataset.empty:
        raise ValueError(f"No data for target {cfg.target}")
    
    print(f"  Base dataset: {len(dataset):,} rows, {dataset.shape[1]} columns")
    
    # Identify column types for clarity
    target_col = "y"
    
    # Self features (target's own IV data)
    self_iv_cols = [c for c in dataset.columns if c == f"IV_{cfg.target}"]
    self_ret_cols = [c for c in dataset.columns if c == f"IVRET_{cfg.target}"]
    
    # Peer features (other tickers' IV data) 
    peer_iv_cols = [c for c in dataset.columns 
                    if c.startswith("IV_") and c != f"IV_{cfg.target}"]
    peer_ret_cols = [c for c in dataset.columns 
                     if c.startswith("IVRET_") and c != f"IVRET_{cfg.target}"]
    
    # Control features (time, Greeks, etc.)
    control_cols = [c for c in dataset.columns 
                    if not c.startswith(("IV_", "IVRET_")) and c != target_col]
    
    print(f"  Self IV features: {len(self_iv_cols)}")
    print(f"  Self return features: {len(self_ret_cols)}")  
    print(f"  Peer IV features: {len(peer_iv_cols)}")
    print(f"  Peer return features: {len(peer_ret_cols)}")
    print(f"  Control features: {len(control_cols)}")
    
    # Handle self-features to avoid leakage
    feature_cols = peer_iv_cols + peer_ret_cols + control_cols
    
    if cfg.exclude_contemporaneous:
        # Don't use target's current IV/returns (would be leakage)
        print("  Excluding contemporaneous self features (avoiding leakage)")
    else:
        # Include current self features (may be leaky but sometimes useful)
        feature_cols.extend(self_iv_cols + self_ret_cols)
        print("  Including contemporaneous self features")
    
    if cfg.include_self_lag:
        # Add lagged self features (safe from leakage)
        for col in self_iv_cols + self_ret_cols:
            lag_col = f"{col}_lag1"
            dataset[lag_col] = dataset[col].shift(1)
            feature_cols.append(lag_col)
        print("  Added lagged self features")
    
    # Select final columns
    final_cols = [target_col] + feature_cols
    clean_dataset = dataset[final_cols].dropna()
    
    print(f"  Final dataset: {len(clean_dataset):,} rows, {len(feature_cols)} features")
    
    return clean_dataset


def train_peer_model(dataset: pd.DataFrame, test_frac: float) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """Train XGBoost model for peer effects."""
    
    # Prepare data
    y = dataset["y"].astype(float)
    X = dataset.drop(columns=["y", "ts_event"]).astype(float)
    
    # Chronological split (important for time series)
    n = len(dataset)
    split_idx = int(n * (1 - test_frac))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"  Training: {len(X_train):,} samples")
    print(f"  Testing: {len(X_test):,} samples")
    
    # Train model with sensible defaults
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_pred - y_test.values) ** 2))
    r2 = 1 - np.sum((y_pred - y_test.values) ** 2) / np.sum((y_test.values - y_test.mean()) ** 2)
    
    metrics = {
        "rmse": float(rmse),
        "r2": float(r2),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(X.columns)
    }
    
    return model, metrics


def analyze_peer_effects(model: xgb.XGBRegressor, feature_names: Sequence[str]) -> Dict[str, Any]:
    """Analyze which peers have the strongest effects."""
    
    # Get feature importance
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    # Categorize features for analysis
    peer_effects = {}
    control_effects = {}
    self_effects = {}
    
    for _, row in feature_imp.iterrows():
        feat = row["feature"]
        imp = row["importance"]
        
        if feat.startswith("IV_") or feat.startswith("IVRET_"):
            if "lag1" in feat:
                self_effects[feat] = imp
            else:
                # Extract ticker name from feature
                ticker = feat.split("_", 1)[1]
                if ticker not in peer_effects:
                    peer_effects[ticker] = 0
                peer_effects[ticker] += imp
        else:
            control_effects[feat] = imp
    
    # Sort by total effect
    peer_effects = dict(sorted(peer_effects.items(), key=lambda x: x[1], reverse=True))
    
    return {
        "peer_rankings": peer_effects,
        "self_lag_effects": self_effects,
        "control_effects": control_effects,
        "detailed_features": feature_imp.head(20).to_dict("records")
    }


def run_peer_analysis(cfg: PeerConfig, cores: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Complete peer effects analysis for one target."""
    
    print(f"\n=== Peer Effects Analysis: {cfg.target} ===")
    
    try:
        # 1. Prepare dataset
        dataset = prepare_peer_dataset(cfg, cores)
        
        # 2. Train model
        model, metrics = train_peer_model(dataset, cfg.test_frac)
        
        print(f"  Model Performance: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.4f}")
        
        # 3. Analyze peer effects
        feature_names = [c for c in dataset.columns if c != "y"]
        analysis = analyze_peer_effects(model, feature_names)
        
        # 4. Report findings
        print(f"  Top peer effects:")
        for ticker, effect in list(analysis["peer_rankings"].items())[:5]:
            print(f"    {ticker}: {effect:.4f}")
        
        # 5. Save results
        results = {
            "target": cfg.target,
            "target_kind": cfg.target_kind,
            "config": {
                "forward_steps": cfg.forward_steps,
                "include_self_lag": cfg.include_self_lag,
                "exclude_contemporaneous": cfg.exclude_contemporaneous
            },
            "performance": metrics,
            "peer_analysis": analysis,
            "status": "success"
        }
        
        if cfg.save_details:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            output_file = cfg.output_dir / f"{cfg.target}_peer_effects.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Detailed results saved: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"  Error in peer analysis: {e}")
        return {
            "target": cfg.target,
            "status": "error", 
            "error": str(e)
        }


def run_multi_target_analysis(
    targets: Sequence[str],
    tickers: Sequence[str], 
    start: str,
    end: str,
    cores: Dict[str, pd.DataFrame],
    **kwargs
) -> Dict[str, Any]:
    """Run peer effects analysis for multiple targets."""
    
    print(f"\n=== Multi-Target Peer Effects Analysis ===")
    print(f"Targets: {targets}")
    print(f"All tickers: {tickers}")
    print(f"Date range: {start} to {end}")
    
    results = {}
    
    for target in targets:
        if target not in cores:
            print(f"\nSkipping {target}: no data available")
            results[target] = {"status": "no_data"}
            continue
            
        cfg = PeerConfig(
            target=target,
            tickers=tickers,
            start=start,
            end=end,
            **kwargs
        )
        
        results[target] = run_peer_analysis(cfg, cores)
    
    # Summary
    successful = [t for t, r in results.items() if r.get("status") == "success"]
    print(f"\n=== Summary ===")
    print(f"Successful analyses: {len(successful)}/{len(targets)}")
    
    if successful:
        print("\nBest performing models:")
        perf = [(t, results[t]["performance"]["r2"]) for t in successful]
        for target, r2 in sorted(perf, key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {target}: R² = {r2:.3f}")
    
    return results


# Example usage function
def example_peer_analysis():
    """Example of how to run peer effects analysis."""
    from feature_engineering import load_ticker_core
    
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