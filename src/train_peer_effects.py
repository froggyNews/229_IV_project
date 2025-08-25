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
        raise ValueError(f"No data found for target {cfg.target}")
    
    print(f"  Base dataset: {len(dataset):,} rows, {dataset.shape[1]} columns")
    
    # Only drop rows where target is missing - be less aggressive
    target_col = "y"
    if target_col in dataset.columns:
        initial_rows = len(dataset)
        dataset = dataset.dropna(subset=[target_col])
        dropped = initial_rows - len(dataset)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with missing target")
        
        # If still empty, check why
        if len(dataset) == 0:
            print("  ERROR: All target values are NaN!")
            print(f"  Target column '{target_col}' statistics:")
            print(f"  - Total values: {len(dataset)}")
            return pd.DataFrame()
    
    # Don't be too aggressive with other cleaning
    print(f"  Final dataset: {len(dataset):,} rows, {dataset.shape[1]} features")
    
    # Add lagged features if requested
    if cfg.include_self_lag and len(dataset) > 0:
        # Include target's own lagged features
        if "ts_event" in dataset.columns:
            dataset = dataset.sort_values("ts_event").reset_index(drop=True)
        
        # Create lag feature safely
        if "ticker" in dataset.columns:
            dataset["y_lag1"] = dataset.groupby("ticker")[target_col].shift(1)
        else:
            dataset["y_lag1"] = dataset[target_col].shift(15)
            
        # Only drop if we still have sufficient data
        before_lag = len(dataset)
        dataset = dataset.dropna(subset=["y_lag1"])
        after_lag = len(dataset)
        print(f"  Added self-lag feature: {after_lag:,} rows (dropped {before_lag - after_lag} for lag)")
    if cfg.exclude_contemporaneous:
        for c in ("IV_SELF", "IVRET_SELF", "iv_clip"):
            if c in dataset.columns:
                dataset.drop(columns=c, inplace=True)
        return dataset
# before: def train_peer_model(dataset: pd.DataFrame, test_frac: float) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
from typing import List, Tuple, Dict, Any
from pandas.api.types import is_datetime64_any_dtype

def train_peer_model(dataset: pd.DataFrame, test_frac: float) -> Tuple[xgb.XGBRegressor, Dict[str, Any], List[str]]:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty - cannot train model")

    # Build feature matrix from non-datetime columns only
    # (don’t guess later—lock the list now and reuse it for analysis)
    feature_cols = [c for c in dataset.columns if c != "y" and not is_datetime64_any_dtype(dataset[c])]
    X = dataset[feature_cols].copy()
    y = pd.to_numeric(dataset["y"], errors="coerce")

    # numeric coercion
    X = X.apply(pd.to_numeric, errors="coerce")

    # keep rows with valid y AND at least one non-NaN feature
    valid_idx = y.notna() & X.notna().any(axis=1)
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)

    # final safety
    n = len(X)
    if n < 10:
        raise ValueError(f"Insufficient data: only {n} samples available")

    split_idx = max(1, int(n * (1 - test_frac)))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=min(300, max(50, len(X_train))),
        learning_rate=0.1,
        max_depth=min(6, max(3, int(np.log2(len(X_train))))),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    m = min(len(y_pred), len(y_test))
    y_pred, y_test_vals = y_pred[:m], y_test.iloc[:m].values

    rmse = float(np.sqrt(np.mean((y_pred - y_test_vals) ** 2))) if m else float("nan")
    ss_tot = float(np.sum((y_test_vals - float(np.mean(y_test_vals))) ** 2)) if m else 0.0
    r2 = float(1 - np.sum((y_pred - y_test_vals) ** 2) / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        "rmse": None if np.isnan(rmse) else rmse,
        "r2": r2 if m else None,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
    }
    # return the exact feature order used to train
    return model, metrics, feature_cols

def analyze_peer_effects(model: xgb.XGBRegressor, feature_names: Sequence[str]) -> Dict[str, Any]:
    importances = np.asarray(model.feature_importances_)
    names = list(feature_names)

    # harden against any mismatch
    k = min(len(importances), len(names))
    importances = importances[:k]
    names = names[:k]

    feature_imp = pd.DataFrame({"feature": names, "importance": importances}) \
                    .sort_values("importance", ascending=False)

    peer_effects, control_effects, self_effects = {}, {}, {}
    for _, row in feature_imp.iterrows():
        feat = row["feature"]; imp = float(row["importance"])
        if feat in ("IV_SELF", "IVRET_SELF", "IV_SELF_L1", "IVRET_SELF_L1", "y_lag1"):
            self_effects[feat] = self_effects.get(feat, 0.0) + imp
        elif feat.startswith(("IV_", "IVRET_")):
            ticker = feat.split("_", 1)[1]
            peer_effects[ticker] = peer_effects.get(ticker, 0.0) + imp
        else:
            control_effects[feat] = control_effects.get(feat, 0.0) + imp

    peer_effects = dict(sorted(peer_effects.items(), key=lambda x: x[1], reverse=True))
    return {
        "peer_rankings": peer_effects,
        "self_lag_effects": self_effects,
        "control_effects": control_effects,
        "detailed_features": feature_imp.head(20).to_dict("records"),
    }

def run_peer_analysis(cfg: PeerConfig, cores: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Complete peer effects analysis for one target."""
    
    print(f"\n=== Peer Effects Analysis: {cfg.target} ===")
    
    try:
        # 1. Prepare dataset
        dataset = prepare_peer_dataset(cfg, cores)
        
        # Check if dataset is empty
        if dataset.empty:
            print(f"  No valid data available for {cfg.target}")
            return {
                "target": cfg.target,
                "target_kind": cfg.target_kind,
                "status": "no_data",
                "error": "Empty dataset after cleaning"
            }
        
        # 2. Train model
        # in run_peer_analysis(...)
        model, metrics, feat_cols_used = train_peer_model(dataset, cfg.test_frac)

        print(f"  Model Performance: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.4f}")

        analysis = analyze_peer_effects(model, feat_cols_used)

        # Check if model training was successful
        if metrics.get('r2') is None or metrics.get('rmse') is None:
            print(f"  Model training failed for {cfg.target}")
            return {
                "target": cfg.target,
                "target_kind": cfg.target_kind,
                "status": "training_failed",
                "error": "Model metrics are None"
            }
        
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

            # Avoid overwriting existing JSON files by finding a unique filename
            if output_file.exists():
                base = output_file.stem
                suffix = output_file.suffix
                counter = 1
                while True:
                    candidate = output_file.with_name(f"{base}_{counter}{suffix}")
                    if not candidate.exists():
                        output_file = candidate
                        break
                    counter += 1

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
    targets: List[str],
    tickers: List[str], 
    start: str,
    end: str,
    cores: Dict[str, pd.DataFrame],
    target_kind: str = "iv_ret",
    forward_steps: int = 1,
    test_frac: float = 0.2,
    tolerance: str = "2s",
    db_path: Path = None,
    include_self_lag: bool = True,
    exclude_contemporaneous: bool = True,
    save_details: bool = False,
    debug: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run peer effects analysis for multiple targets."""
    
    if debug:
        print(f"DEBUG: Running multi-target analysis for {len(targets)} targets")
        print(f"DEBUG: Available cores: {list(cores.keys())}")
    
    print(f"\n=== Multi-Target Peer Effects Analysis ===")
    print(f"Targets: {targets}")
    print(f"All tickers: {tickers}")
    print(f"Date range: {start} to {end}")
    
    results = {}
    successful_analyses = 0
    
    for target in targets:
        if target not in cores or cores[target] is None or cores[target].empty:
            print(f"\nSkipping {target}: no data available")
            results[target] = {"status": "no_data", "error": "No core data available"}
            continue
            
        try:
            print(f"\n=== Peer Effects Analysis: {target} ===")
            
            if debug:
                print(f"DEBUG: Analyzing {target} with target_kind={target_kind}")
            
            # Create configuration (don't pass debug to PeerConfig)
            cfg = PeerConfig(
                target=target,
                tickers=tickers,
                start=start,
                end=end,
                target_kind=target_kind,
                forward_steps=forward_steps,
                test_frac=test_frac,
                tolerance=tolerance,
                db_path=db_path,
            )
            
            # Prepare dataset (pass debug to prepare_peer_dataset instead)
            dataset = prepare_peer_dataset(cfg, cores, debug=debug)
            
            if dataset.empty:
                print(f"  No valid data for {target}")
                results[target] = {"status": "no_valid_data", "error": "Dataset is empty after preparation"}
                continue
                
            # Train model
            model, metrics, feature_names = train_peer_model(dataset, test_frac)
            
            results[target] = {
                "status": "success",
                "metrics": metrics,
                "feature_names": feature_names,
                "config": {
                    "target_kind": target_kind,
                    "forward_steps": forward_steps,
                    "test_frac": test_frac,
                    "n_samples": len(dataset),
                    "n_features": len(feature_names)
                }
            }
            
            successful_analyses += 1
            print(f"  ✓ {target}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.6f}")
            
        except Exception as e:
            print(f"  ✗ Error analyzing {target}: {e}")
            results[target] = {"status": "error", "error": str(e)}
            if debug:
                import traceback
                print(f"DEBUG: Full traceback for {target}:")
                traceback.print_exc()
    
    print(f"\n=== Summary ===")
    print(f"Successful analyses: {successful_analyses}/{len(targets)}")
    
    return results


def prepare_peer_dataset(cfg: PeerConfig, cores: Dict[str, pd.DataFrame], debug: bool = False) -> pd.DataFrame:
    """Build and clean dataset for peer effects analysis."""
    
    if debug:
        print(f"DEBUG: Preparing peer dataset for {cfg.target}")
        print(f"DEBUG: Available cores: {list(cores.keys())}")
    
    print(f"Building peer dataset for target: {cfg.target}")
    
    # Build base dataset (pass debug to build_target_peer_dataset)
    dataset = build_target_peer_dataset(
        target=cfg.target,
        tickers=cfg.tickers,
        start=cfg.start,
        end=cfg.end,
        forward_steps=cfg.forward_steps,
        db_path=cfg.db_path,
        target_kind=cfg.target_kind,
        cores=cores,
        debug=debug,
    )
    
    if dataset.empty:
        if debug:
            print(f"DEBUG: build_target_peer_dataset returned empty DataFrame for {cfg.target}")
        raise ValueError(f"No data found for target {cfg.target}")
    
    print(f"  Base dataset: {len(dataset):,} rows, {dataset.shape[1]} columns")
    
    # DEBUG: Check what happens at each step
    if debug:
        print(f"  DEBUG: Before cleaning - {len(dataset)} rows")
    
    # Check for missing target values
    target_col = "y"
    before_dropna = len(dataset)
    dataset = dataset.dropna(subset=[target_col])
    after_dropna = len(dataset)
    if debug:
        print(f"  DEBUG: After dropping NaN targets - {after_dropna} rows (dropped {before_dropna - after_dropna})")
    
    # Replace infinite values with NaN but don't drop rows yet
    before_inf = len(dataset)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    # After replacing infinities, ensure target column is still valid
    dataset = dataset.dropna(subset=[target_col])
    after_inf = len(dataset)
    if debug:
        print(
            f"  DEBUG: After handling infinities - {after_inf} rows (dropped {before_inf - after_inf})"
        )
    
    # Check datetime exclusion
    before_datetime = len(dataset)
    datetime_cols = dataset.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        if debug:
            print(f"  DEBUG: Found datetime columns: {datetime_cols.tolist()}")
        dataset = dataset.drop(columns=datetime_cols)
    after_datetime = len(dataset)
    if debug:
        print(f"  DEBUG: After dropping datetime cols - {after_datetime} rows")
    
    # Final check - ensure we have data
    if len(dataset) == 0:
        print("  ERROR: Dataset became empty after cleaning!")
        print("  This suggests the target variable 'y' has all NaN/inf values")
        return pd.DataFrame()
    
    # Identify column types for clarity
    target_col = "y"
    
    # Self features (target's own IV data)
    self_iv_cols = [c for c in dataset.columns if c == f"IV_{cfg.target}"]
    self_ret_cols = [c for c in dataset.columns if c == f"RET_{cfg.target}"]
    peer_iv_cols = [c for c in dataset.columns if c.startswith("IV_") and c != f"IV_{cfg.target}"]
    peer_ret_cols = [c for c in dataset.columns if c.startswith("RET_") and c != f"RET_{cfg.target}"]
    control_cols = [c for c in dataset.columns if not any(c.startswith(p) for p in ["IV_", "RET_"]) and c != target_col]
    
    print(f"  Self IV features: {len(self_iv_cols)}")
    print(f"  Self return features: {len(self_ret_cols)}")
    print(f"  Peer IV features: {len(peer_iv_cols)}")
    print(f"  Peer return features: {len(peer_ret_cols)}")
    print(f"  Control features: {len(control_cols)}")
    
    # Exclude contemporaneous self features to avoid leakage
    contemporaneous_self = self_iv_cols + self_ret_cols
    if contemporaneous_self:
        print("  Excluding contemporaneous self features (avoiding leakage)")
        dataset = dataset.drop(columns=contemporaneous_self)
    
    # Add lagged self features if available
    print("  Added lagged self features")
    
    print(f"  Final dataset: {len(dataset):,} rows, {dataset.shape[1]} features")
    
    return dataset

# Example usage function
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