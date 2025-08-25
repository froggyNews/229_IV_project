import argparse
import csv
import json
import os
import sys
from datetime import datetime
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

# Try to import model evaluation
try:
    from model_evaluation import evaluate_pooled_model
    MODEL_EVALUATION_AVAILABLE = True
except ImportError:
    print("Warning: model_evaluation module not found")
    MODEL_EVALUATION_AVAILABLE = False

@dataclass
class RunConfig:
    """Configuration for the main pipeline run."""
    
    # Core settings
    tickers: Sequence[str] = field(default_factory=lambda: ["QUBT", "QBTS", "RGTI", "IONQ"])
    start: str = "2025-08-02"
    end: str = "2025-08-06"
    
    # Model settings
    forward_steps: int = 15
    test_frac: float = 0.2
    tolerance: str = "2s"
    r: float = 0.045
    
    # Peer effects settings
    peer_targets: Sequence[str] = field(default_factory=lambda: ["QUBT", "QBTS"])
    peer_target_kinds: Sequence[str] = field(default_factory=lambda: ["iv_ret", "iv"])
    
    # Paths
    db_path: Path = field(default_factory=lambda: Path("data/iv_data_1m.db"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    
    # Runtime settings
    auto_fetch: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    debug: bool = False  # Add debug flag
    
    # Data settings
    drop_zero_iv_ret: bool = True
    # Optional settings
    xgb_params: Optional[Dict[str, Any]] = None


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


def train_pooled_models(cfg: RunConfig, cores: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Train pooled models for IV returns and levels."""
    
    print("\n=== Training Pooled Models ===")
    
    if cfg.debug:
        print("DEBUG: Debug mode enabled - will save data snapshots")
        print(f"DEBUG: Using preloaded cores for tickers: {list(cores.keys())}")
    
    # Load pooled dataset with cores
    pooled = build_pooled_iv_return_dataset_time_safe(
        tickers=list(cfg.tickers),
        start=pd.Timestamp(cfg.start, tz="UTC"),
        end=pd.Timestamp(cfg.end, tz="UTC"),
        r=cfg.r,
        forward_steps=cfg.forward_steps,
        tolerance=cfg.tolerance,
        db_path=cfg.db_path,
        cores=cores,  # Pass the loaded cores
        debug=cfg.debug,
    )
    
    if pooled.empty:
        raise ValueError("Pooled dataset is empty")
    
    print(f"Pooled dataset: {len(pooled):,} rows, {pooled.shape[1]} columns")
    
    models = {}
    metrics = {}
    evaluations = {}  # Store detailed evaluations
    
    # Train IV return model
    print("Training IV return model...")
    model_ret, metrics_ret = train_model(
        data=pooled,
        target="iv_ret_fwd",
        test_frac=cfg.test_frac,
        drop_cols=["iv_ret_fwd_abs", "core_iv_ret_fwd_abs", "iv_clip"],  # Avoid leakage
        params=get_default_xgb_params()
    )
    models["iv_ret_fwd"] = model_ret
    metrics["iv_ret_fwd"] = metrics_ret
    
    # Train IV level model if available
    if "iv_clip" in pooled.columns:
        print("Training IV level model...")
        model_level, metrics_level = train_model(
            data=pooled,
            target="iv_clip",
            test_frac=cfg.test_frac,
            drop_cols=["iv_ret_fwd", "iv_ret_fwd_abs", "core_iv_ret_fwd_abs"],  # Avoid leakage
            params=get_default_xgb_params()
        )
        models["iv_clip"] = model_level
        metrics["iv_clip"] = metrics_level
    
    return {
        "models": models,
        "metrics": metrics,
        "dataset_info": {
            "n_rows": len(pooled),
            "n_features": pooled.shape[1],
            "tickers": list(cfg.tickers),
            "date_range": f"{cfg.start} to {cfg.end}"
        }
    }


def train_peer_effects(cfg: RunConfig, cores: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Train peer effects models using updated train_peer_effects module."""
    
    if not cfg.peer_targets:
        return {}
    
    print(f"\n=== Peer Effects Analysis ===") 
    print(f"Analyzing {len(cfg.peer_target_kinds)} targets: {cfg.peer_targets}")
    
    if cfg.debug:
        print("DEBUG: Debug mode enabled for peer effects")
    
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
            save_details=False,  # Don't save individual files, we'll aggregate
            debug=cfg.debug,  # Pass debug flag
        )

        # Store results by target_kind
        for target, result in results.items():
            key = f"{target}_{target_kind}"
            all_results[key] = result

    return all_results


def save_results(
    cfg: RunConfig,
    pooled_results: Dict,
    peer_results: Dict,
    eval_results: Dict,
) -> Path:
    """Save all results to disk in a structured format."""

    # Ensure base output directory exists
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Create run-specific directory using timestamp for easy comparison
    run_dir = cfg.output_dir / cfg.timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

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
        "pooled_models": pooled_results.get("metrics", {}),
        "peer_effects": peer_results,
        "evaluation": eval_results,
    }

    # Save metrics for this run
    metrics_path = run_dir / "results.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    # Build a flat summary for easy comparison across runs
    summary = {
        "timestamp": cfg.timestamp,
        "start": cfg.start,
        "end": cfg.end,
        "tickers": ";".join(cfg.tickers),
        "forward_steps": cfg.forward_steps,
        "test_frac": cfg.test_frac,
    }

    for model_name, metrics in pooled_results.get("metrics", {}).items():
        summary[f"{model_name}_rmse"] = metrics.get("rmse")
        summary[f"{model_name}_r2"] = metrics.get("r2")

    summary_path = cfg.output_dir / "run_summary.csv"
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    print(f"Results saved to: {metrics_path}")
    print(f"Run summary updated: {summary_path}")
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
    
    # Load cores once at the start
    print("\n=== Loading Data Cores ===")
    print(f"Requested date range: {cfg.start} to {cfg.end}")
    
    cores = load_cores_with_auto_fetch(
        tickers=list(cfg.tickers),
        start=cfg.start,
        end=cfg.end,
        db_path=cfg.db_path,
        auto_fetch=cfg.auto_fetch
    )
    
    if not cores:
        raise ValueError("No cores loaded successfully")
    
    if cfg.debug:
        print(f"DEBUG: Loaded cores for tickers: {list(cores.keys())}")
        
    # Debug: Show actual date ranges in loaded data
    print("Actual data ranges loaded:")
    for ticker, core in cores.items():
        if not core.empty and 'ts_event' in core.columns:
            min_date = core['ts_event'].min()
            max_date = core['ts_event'].max()
            print(f"  {ticker}: {min_date.strftime('%Y-%m-%d %H:%M')} to {max_date.strftime('%Y-%m-%d %H:%M')} ({len(core):,} rows)")
        else:
            print(f"  {ticker}: No valid data or missing ts_event column")
    
    # Train pooled models
    pooled_results = train_pooled_models(cfg, cores)
    
    # Train peer effects models
    peer_results = train_peer_effects(cfg, cores)

    # Save models first so evaluation can load them
    model_paths = save_models(cfg, pooled_results, peer_results)

    # Evaluate each saved pooled model
    eval_results = {}
    if MODEL_EVALUATION_AVAILABLE and model_paths:
        print("\n=== Model Evaluation (NIDEL-style) ===")
        for name, path in model_paths.items():
            try:
                print(f"Evaluating {name} model...")
                evaluation = evaluate_pooled_model(
                    model_path=path,
                    tickers=list(cfg.tickers),
                    start=cfg.start,
                    end=cfg.end,
                    test_frac=cfg.test_frac,
                    forward_steps=cfg.forward_steps,
                    tolerance=cfg.tolerance,
                    r=cfg.r,
                    metrics_dir=cfg.output_dir / "evaluations",
                    outputs_prefix=f"{path.stem}_eval",
                    save_predictions=True,
                    perm_repeats=5,
                    perm_sample=5000,
                    db_path=cfg.db_path,
                    target_col=name.replace("pooled_", ""),  # Extract target name
                    save_report=True,
                )
                eval_results[name] = evaluation
                print(f"  ✓ {name}: RMSE={evaluation['metrics']['RMSE']:.6f}, R²={evaluation['metrics']['R2']:.3f}")
            except Exception as e:
                print(f"  ✗ Failed to evaluate {name}: {e}")
                eval_results[name] = {"error": str(e)}

    # Save aggregated results including evaluation
    metrics_path = save_results(cfg, pooled_results, peer_results, eval_results)

    return {
        "pooled_results": pooled_results,
        "peer_results": peer_results,
        "evaluation": eval_results,
        "metrics_path": metrics_path,
        "model_paths": model_paths,
    }


def parse_arguments() -> RunConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(description="Run IV return forecasting pipeline")
    
    # Data settings
    parser.add_argument("--tickers", nargs="+", default=["QUBT", "QBTS", "RGTI", "IONQ"],
                        help="List of tickers to analyze")
    parser.add_argument("--start", default="2025-06-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-08-06", help="End date (YYYY-MM-DD)")
    
    # Model settings
    parser.add_argument("--forward-steps", type=int, default=15, 
                        help="Number of forward steps for prediction")
    parser.add_argument("--test-frac", type=float, default=0.2, 
                        help="Fraction of data to use for testing")
    parser.add_argument("--tolerance", default="2s", help="Time tolerance for merging")
    parser.add_argument("--r", type=float, default=0.045, help="Risk-free rate")
    
    # Peer effects settings
    parser.add_argument("--peer-targets", nargs="+", default=["QUBT", "QBTS"],
                        help="Target tickers for peer effects analysis")
    parser.add_argument("--peer-target-kinds", nargs="+", default=["iv_ret", "iv"],
                        help="Types of targets for peer effects")
    
    # Paths
    parser.add_argument("--db-path", type=Path, default=Path("data/iv_data_1m.db"),
                        help="Path to database file")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Output directory for results")
    
    # Runtime settings
    parser.add_argument("--no-auto-fetch", action="store_true", 
                        help="Disable automatic data fetching")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug mode with data snapshots")
    
    args = parser.parse_args()
    
    return RunConfig(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        forward_steps=args.forward_steps,
        test_frac=args.test_frac,
        tolerance=args.tolerance,
        r=args.r,
        peer_targets=args.peer_targets,
        peer_target_kinds=args.peer_target_kinds,
        db_path=args.db_path,
        output_dir=args.output_dir,
        auto_fetch=not args.no_auto_fetch,
        drop_zero_iv_ret=True,  
        debug=args.debug,
    )


# Example usage
if __name__ == "__main__":
    
    # Parse command line arguments
    config = parse_arguments()
    
    # Run pipeline
    try:
        results = run_pipeline(config)
        print(f"\nPipeline completed successfully!")
        
        if config.debug:
            print(f"\nDEBUG: Check debug_snapshots/ directory for data snapshots")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        if config.debug:
            import traceback
            print("\nDEBUG: Full traceback:")
            traceback.print_exc()
        sys.exit(1)
