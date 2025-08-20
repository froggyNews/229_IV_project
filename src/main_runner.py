

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Optional environment loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Import centralized functions
from feature_engineering import (
    build_pooled_iv_return_dataset_time_safe,
    build_target_peer_dataset,
)
from data_loader_coordinator import load_cores_with_auto_fetch
from train_peer_effects import run_multi_target_analysis

@dataclass
class RunConfig:
    """Simplified configuration with sensible defaults."""
    # Core data settings
    tickers: Sequence[str] = field(default_factory=list)
    start: str = "2025-01-02"
    end: str = "2025-01-15"
    db_path: Path = Path(os.getenv("IV_DB_PATH", "data/iv_data_1m.db"))
    
    # Model settings
    forward_steps: int = 15
    test_frac: float = 0.2
    tolerance: str = "2s"

    # Output settings
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S"))
    output_dir: Path = Path("outputs")

    # Optional settings
    xgb_params: Optional[Dict[str, Any]] = None
    peer_targets: Sequence[str] = field(default_factory=list)
    peer_target_kinds: Sequence[str] = field(default_factory=lambda: ["iv_ret"])
    drop_zero_iv_ret: bool = False


def get_default_xgb_params() -> Dict[str, Any]:
    """Get default XGBoost parameters."""
    return {
        "objective": "reg:squarederror",
        "n_estimators": 350,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def train_model(data: pd.DataFrame, target: str, test_frac: float, 
                drop_cols: Sequence[str] = None, params: Dict = None) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """Train XGBoost model and return model + metrics."""
    
    if target not in data.columns:
        raise KeyError(f"Target '{target}' not found in data")
    
    # Prepare data - exclude datetime and non-numeric columns
    drop_cols = list(drop_cols or [])
    
    # Always exclude datetime columns and other non-feature columns
    exclude_cols = [target] + drop_cols + ["ts_event", "expiry_date", "symbol"]
    
    # Get numeric columns only
    feature_cols = []
    for col in data.columns:
        if col not in exclude_cols:
            # Check if column can be converted to numeric
            try:
                pd.to_numeric(data[col], errors='raise')
                feature_cols.append(col)
            except (ValueError, TypeError):
                print(f"Excluding non-numeric column: {col}")
                continue
    
    if not feature_cols:
        raise ValueError("No valid feature columns found")
    
    print(f"Using {len(feature_cols)} features for training")
    
    X = data[feature_cols].astype(float)
    y = data[target].astype(float).values
    
    # Train/test split
    n = len(X)
    split_idx = int(n * (1 - test_frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model_params = params or get_default_xgb_params()
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": feature_cols
    }
    
    return model, metrics


def train_pooled_models(cfg: RunConfig) -> Dict[str, Any]:
    """Train pooled models for all targets."""
    
    print(f"Loading data for tickers: {cfg.tickers}")
    
    # Use coordinator for clean data loading
    cores = load_cores_with_auto_fetch(
        tickers=cfg.tickers,
        start=cfg.start,
        end=cfg.end,
        db_path=cfg.db_path,
        auto_fetch=True,
        drop_zero_iv_ret=cfg.drop_zero_iv_ret
    )
    
    # Build pooled dataset
    pooled = build_pooled_iv_return_dataset_time_safe(
        tickers=list(cores.keys()),
        start=cfg.start,
        end=cfg.end,
        forward_steps=cfg.forward_steps,
        tolerance=cfg.tolerance,
        db_path=cfg.db_path,
        cores=cores
    )
    
    if pooled.empty:
        raise ValueError("Pooled dataset is empty")
    
    print(f"Pooled dataset: {len(pooled):,} rows, {pooled.shape[1]} columns")
    
    # Train models for different targets
    results = {}
    models = {}
    
    # IV return model
    if "iv_ret_fwd" in pooled.columns:
        print("Training IV return model...")
        model, metrics = train_model(
            pooled, "iv_ret_fwd", cfg.test_frac, 
            drop_cols=["iv_ret_fwd_abs"], params=cfg.xgb_params
        )
        results["iv_ret_fwd"] = metrics
        models["iv_ret_fwd"] = model
    
    # Absolute IV return model  
    if "iv_ret_fwd_abs" in pooled.columns:
        print("Training absolute IV return model...")
        model, metrics = train_model(
            pooled, "iv_ret_fwd_abs", cfg.test_frac,
            drop_cols=["iv_ret_fwd"], params=cfg.xgb_params
        )
        results["iv_ret_fwd_abs"] = metrics
        models["iv_ret_fwd_abs"] = model
    
    # IV level model
    if "iv_clip" in pooled.columns:
        print("Training IV level model...")
        model, metrics = train_model(
            pooled, "iv_clip", cfg.test_frac,
            drop_cols=["iv_ret_fwd", "iv_ret_fwd_abs"], params=cfg.xgb_params
        )
        results["iv_clip"] = metrics  
        models["iv_clip"] = model
    
    return {"metrics": results, "models": models, "cores": cores}


def train_peer_effects(cfg: RunConfig, cores: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Train peer effects models using updated train_peer_effects module."""
    
    if not cfg.peer_targets:
        return {}
    
    print(f"\n=== Peer Effects Analysis ===") 
    print(f"Analyzing {len(cfg.peer_target_kinds)} targets: {cfg.peer_targets}")
    
    all_results = {}
    
    for target_kind in cfg.peer_target_kinds:
        print(f"\nAnalyzing target kind: {target_kind}")
        
        # Use the updated run_multi_target_analysis function
        results = run_multi_target_analysis(
            targets=cfg.peer_targets,
            tickers=list(cfg.tickers),  # Use all configured tickers
            start=cfg.start,
            end=cfg.end,
            cores=cores,
            target_kind=target_kind,
            forward_steps=cfg.forward_steps,
            test_frac=cfg.test_frac,
            tolerance=cfg.tolerance,
            db_path=cfg.db_path,
            include_self_lag=True,
            exclude_contemporaneous=True,
            save_details=True  # Don't save individual files, we'll aggregate
        )
        
        # Store results by target_kind
        for target, result in results.items():
            key = f"{target}_{target_kind}"
            all_results[key] = result
    
    return results


def save_results(cfg: RunConfig, pooled_results: Dict, peer_results: Dict) -> Path:
    """Save all results to JSON file."""
    
    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare metrics (exclude models from JSON)
    output = {
        "config": {
            "tickers": list(cfg.tickers),
            "start": cfg.start,
            "end": cfg.end,
            "forward_steps": cfg.forward_steps,
            "test_frac": cfg.test_frac,
            "timestamp": cfg.timestamp,
            "db_path": str(cfg.db_path),
            "output_dir": str(cfg.output_dir),
            "xgb_params": cfg.xgb_params if cfg.xgb_params is not None else get_default_xgb_params(),
            "peer_targets": list(cfg.peer_targets),
            "peer_target_kinds": list(cfg.peer_target_kinds),
            "tolerance": cfg.tolerance,
            "drop_zero_iv_ret": cfg.drop_zero_iv_ret,
        },
        "pooled_models": pooled_results[1],
        "peer_effects": peer_results,
    }
    
    # Save metrics
    metrics_path = cfg.output_dir / f"metrics_{cfg.timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {metrics_path}")
    return metrics_path


def save_models(cfg: RunConfig, pooled_results: Dict, peer_results: Dict) -> Dict[str, Path]:
    """Save trained models to disk."""
    
    models_dir = cfg.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # Save pooled models
    for name, model in pooled_results.get("models", {}).items():
        path = models_dir / f"pooled_{name}_{cfg.timestamp}.json"
        model.save_model(str(path))
        saved_paths[f"pooled_{name}"] = path
        print(f"Saved pooled model: {path}")
    
    # Note: The updated train_peer_effects module focuses on analysis rather than model persistence
    # Peer effects models are not saved in the updated version as they are used for analysis only
    
    return saved_paths


def run_pipeline(cfg: RunConfig) -> Dict[str, Any]:
    """Run complete training pipeline."""
    
    if not cfg.tickers:
        raise ValueError("No tickers specified")
    
    print(f"Starting pipeline with {len(cfg.tickers)} tickers")
    print(f"Date range: {cfg.start} to {cfg.end}")
    
    # Train pooled models
    pooled_results = train_pooled_models(cfg)
    
    # Train peer effects models  
    peer_results = train_peer_effects(cfg, pooled_results["cores"])
    
    # Save results
    metrics_path = save_results(cfg, pooled_results, peer_results)
    model_paths = save_models(cfg, pooled_results, peer_results)
    
    return {
        "pooled_results": pooled_results,
        "peer_results": peer_results,
        "metrics_path": metrics_path,
        "model_paths": model_paths
    }


# Example usage
if __name__ == "__main__":
    
    # Simple configuration
    config = RunConfig(
        tickers=["QUBT", "QBTS", "RGTI", "IONQ"],
        start="2025-06-02",
        end="2025-08-06",
        forward_steps=15,
        test_frac=0.2,
        peer_targets=["QUBT", "QBTS", "RGTI", "IONQ"],  # Subset for peer effects
        peer_target_kinds=["iv_ret", "iv_ret_fwd", "iv_ret_fwd_abs", "iv"],  # Both return and level
        drop_zero_iv_ret=True,
    )
    
    # Run pipeline
    try:
        results = run_pipeline(config)
        print("\nPipeline completed successfully!")
        print(f"Models trained: {len(results['model_paths'])}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise