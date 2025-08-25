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
from baseline_correlation import compute_baseline_correlations
from peer_group_analyzer import PeerGroupAnalyzer, PeerGroupConfig

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
    tickers: Sequence[str] = field(default_factory=lambda: ["ASTS", "VZ", "T", "SATS"])
    start: str = "2025-08-02"
    end: str = "2025-08-06"

    # Model settings
    forward_steps: int = 15
    test_frac: float = 0.2
    tolerance: str = "2s"
    r: float = 0.045

    # Peer effects settings
    peer_targets: Sequence[str] = field(default_factory=lambda: ["ASTS", "VZ", "T", "SATS"])
    peer_target_kinds: Sequence[str] = field(default_factory=lambda: ["iv_ret", "iv"])

    # Paths
    db_path: Path = field(default_factory=lambda: Path("data/iv_data_1m.db"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # Runtime settings
    auto_fetch: bool = True
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    debug: bool = False

    # Data settings
    drop_zero_iv_ret: bool = True

    # Optional settings
    xgb_params: Optional[Dict[str, Any]] = None

    # NEW: never save per-row predictions/targets
    save_per_row: bool = False
    
    # Baseline correlation settings
    compute_baseline_correlations: bool = True
    
    # Peer group analysis settings
    enable_peer_group_analysis: bool = False
    peer_groups: Optional[Dict[str, Sequence[str]]] = None


def _strip_per_row_payloads(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove any per-row predictions/targets if present (defensive).
    We drop keys named 'predictions' or any list/dict that looks like rowwise dumps.
    """
    if not isinstance(obj, dict):
        return obj
    clean = {}
    for k, v in obj.items():
        if k.lower() in {"preds", "predictions", "rows", "per_row", "y_true", "y_pred", "residuals"}:
            # skip
            continue
        if isinstance(v, dict):
            clean[k] = _strip_per_row_payloads(v)
        elif isinstance(v, list):
            # keep lists if they are clearly not rowwise dicts
            if v and isinstance(v[0], dict) and {"y_true", "y_pred"}.intersection(v[0].keys()):
                continue
            clean[k] = v
        else:
            clean[k] = v
    return clean


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
                drop_cols: Sequence[str] = None, params: Dict = None, 
                analysis_type: str = "default") -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """Train XGBoost model and return model + metrics."""
    
    if target not in data.columns:
        raise KeyError(f"Target '{target}' not found in data")
    
    # Import feature configuration
    try:
        from analysis_feature_config import get_features_for_analysis_type, filter_features_by_importance
        use_smart_features = True
    except ImportError:
        print("Warning: analysis_feature_config not available, using basic feature selection")
        use_smart_features = False
    
    # Prepare data - exclude datetime and non-numeric columns
    drop_cols = list(drop_cols or [])
    
    if use_smart_features:
        # Use smart feature selection based on analysis type
        print(f"Using smart feature selection for analysis type: {analysis_type}")
        
        # Get all available columns
        available_cols = list(data.columns)
        
        # Get appropriate features for this analysis type
        feature_cols = get_features_for_analysis_type(
            analysis_type=analysis_type,
            available_columns=available_cols,
            target_column=target,
            include_peer_features=True,
            include_time_features=True,
            include_advanced_features=True
        )
        
        # Filter out manually specified drop columns
        feature_cols = [col for col in feature_cols if col not in drop_cols]
        
        # Prioritize features by importance for this analysis
        feature_cols = filter_features_by_importance(feature_cols, analysis_type)
        
        # Ensure features are numeric
        numeric_features = []
        for col in feature_cols:
            if col in data.columns:
                try:
                    pd.to_numeric(data[col], errors='raise')
                    numeric_features.append(col)
                except (ValueError, TypeError):
                    print(f"Excluding non-numeric feature: {col}")
                    continue
        
        feature_cols = numeric_features
        
    else:
        # Fallback to basic feature selection
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
    
    print(f"Using {len(feature_cols)} features for training ({analysis_type})")
    
    # Print feature breakdown for debugging
    if use_smart_features and len(feature_cols) > 0:
        from analysis_feature_config import AnalysisFeatureConfig
        config = AnalysisFeatureConfig()
        
        greeks = [f for f in feature_cols if f in config.GREEKS_FEATURES]
        vol_features = [f for f in feature_cols if f in config.VOLATILITY_FEATURES]
        panel_features = [f for f in feature_cols if f.startswith(('panel_', 'IV_', 'IVRET_'))]
        time_features = [f for f in feature_cols if f in config.TIME_FEATURES]
        
        print(f"  Greeks: {len(greeks)}, Volatility: {len(vol_features)}, "
              f"Panel: {len(panel_features)}, Time: {len(time_features)}")
    
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
        params=get_default_xgb_params(),
        analysis_type="pooled_iv_returns"
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
            params=get_default_xgb_params(),
            analysis_type="iv_level_modeling"
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


def compute_baseline_correlations_wrapper(cfg: RunConfig, cores: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Compute baseline historical correlations using preloaded cores."""
    
    if not cfg.compute_baseline_correlations:
        return {}
    
    print("\n=== Baseline Correlation Analysis ===")
    print(f"Computing historical correlations for {len(cfg.tickers)} tickers")
    
    if cfg.debug:
        print("DEBUG: Debug mode enabled for baseline correlations")
    
    try:
        correlations = compute_baseline_correlations(
            tickers=list(cfg.tickers),
            start=cfg.start,
            end=cfg.end,
            db_path=cfg.db_path,
            cores=cores,  # Use preloaded cores
            tolerance=cfg.tolerance,
        )
        
        # Convert correlation matrices to dictionaries for JSON serialization
        result = {
            "iv_levels": {
                "correlation_matrix": correlations["clip"].to_dict() if not correlations["clip"].empty else {},
                "description": "Historical correlation matrix of IV levels",
                "n_tickers": len(correlations["clip"].columns) if not correlations["clip"].empty else 0,
            },
            "iv_returns": {
                "correlation_matrix": correlations["iv_returns"].to_dict() if not correlations["iv_returns"].empty else {},
                "description": "Historical correlation matrix of IV returns",
                "n_tickers": len(correlations["iv_returns"].columns) if not correlations["iv_returns"].empty else 0,
            }
        }
        
        # Print summary
        if not correlations["clip"].empty:
            print(f"  ✓ IV levels: {len(correlations['clip'].columns)} tickers")
            # Print mean correlation excluding diagonal
            iv_corr_values = correlations["clip"].values
            mask = ~np.eye(iv_corr_values.shape[0], dtype=bool)
            mean_iv_corr = np.mean(iv_corr_values[mask]) if mask.any() else 0
            print(f"    Mean off-diagonal correlation: {mean_iv_corr:.3f}")
        else:
            print("  ✗ IV levels: No correlation data available")
            
        if not correlations["iv_returns"].empty:
            print(f"  ✓ IV returns: {len(correlations['iv_returns'].columns)} tickers")
            # Print mean correlation excluding diagonal
            ret_corr_values = correlations["iv_returns"].values
            mask = ~np.eye(ret_corr_values.shape[0], dtype=bool)
            mean_ret_corr = np.mean(ret_corr_values[mask]) if mask.any() else 0
            print(f"    Mean off-diagonal correlation: {mean_ret_corr:.3f}")
        else:
            print("  ✗ IV returns: No correlation data available")
            
        return result
        
    except Exception as e:
        print(f"  ✗ Failed to compute baseline correlations: {e}")
        if cfg.debug:
            import traceback
            traceback.print_exc()
        return {"error": str(e)}


def run_peer_group_analysis_wrapper(cfg: RunConfig, cores: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Run comprehensive peer group analysis using the PeerGroupAnalyzer."""
    
    if not cfg.enable_peer_group_analysis:
        return {}
    
    print("\n=== Comprehensive Peer Group Analysis ===")
    
    try:
        # Use provided peer groups or create default groups
        if cfg.peer_groups:
            groups = {name: list(tickers) for name, tickers in cfg.peer_groups.items()}
        else:
            # Create default groups based on available tickers
            all_tickers = list(cfg.tickers)
            if len(all_tickers) >= 4:
                # Try to create meaningful groups if we have 4+ tickers
                mid = len(all_tickers) // 2
                groups = {
                    "group_a": all_tickers[:mid],
                    "group_b": all_tickers[mid:],
                    "all_combined": all_tickers
                }
            else:
                # Single group for small ticker sets
                groups = {"all_tickers": all_tickers}
        
        # Create peer group configuration
        peer_config = PeerGroupConfig(
            groups=groups,
            start=cfg.start,
            end=cfg.end,
            target_kinds=list(cfg.peer_target_kinds),
            forward_steps=cfg.forward_steps,
            test_frac=cfg.test_frac,
            tolerance=cfg.tolerance,
            r=cfg.r,
            db_path=cfg.db_path,
            auto_fetch=cfg.auto_fetch,
            output_dir=cfg.output_dir / "peer_groups",
            timestamp=cfg.timestamp,
            save_detailed_results=True,
            debug=cfg.debug
        )
        
        # Initialize analyzer with preloaded cores
        analyzer = PeerGroupAnalyzer(peer_config)
        analyzer.cores = cores  # Use the already loaded cores
        analyzer.all_tickers = sorted(set().union(*groups.values()))
        
        # Run analysis (skip data loading since we have cores)
        print(f"Running analysis for {len(groups)} peer groups")
        
        results = {}
        results["intra_correlations"] = analyzer.compute_intra_group_correlations()
        results["inter_correlations"] = analyzer.compute_inter_group_correlations()
        results["intra_peer_effects"] = analyzer.compute_intra_group_peer_effects()
        results["inter_peer_effects"] = analyzer.compute_inter_group_peer_effects()
        results["statistical_tests"] = analyzer.run_statistical_tests()
        
        # Add metadata
        results["metadata"] = {
            "config": analyzer._config_to_dict(),
            "timestamp": cfg.timestamp,
            "all_tickers": analyzer.all_tickers,
            "groups": groups,
            "analysis_summary": analyzer._generate_analysis_summary()
        }
        
        # Set the results in analyzer for saving
        analyzer.results = results
        
        if cfg.debug or True:  # Always save peer group results
            analyzer.save_results()
        
        print(f"✅ Peer group analysis completed for {len(groups)} groups")
        return results
        
    except Exception as e:
        print(f"✗ Peer group analysis failed: {e}")
        if cfg.debug:
            import traceback
            traceback.print_exc()
        return {"error": str(e)}


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
    baseline_results: Dict,
    peer_group_results: Dict,
) -> Path:
    """Save all results to disk in a structured format."""

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.output_dir / cfg.timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # If any subtrees slipped in per-row payloads, strip them now
    safe_eval = _strip_per_row_payloads(eval_results)

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
            "save_per_row": cfg.save_per_row,
        },
        "pooled_models": pooled_results.get("metrics", {}),
        "peer_effects": peer_results,
        "baseline_correlations": baseline_results,
        "peer_group_analysis": peer_group_results,
        "evaluation": safe_eval,  # <--- use sanitized view
    }

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
    
    # Compute baseline correlations
    baseline_results = compute_baseline_correlations_wrapper(cfg, cores)
    
    # Run comprehensive peer group analysis
    peer_group_results = run_peer_group_analysis_wrapper(cfg, cores)
    
    # Train pooled models
    pooled_results = train_pooled_models(cfg, cores)
    
    # Train peer effects models
    peer_results = train_peer_effects(cfg, cores)

    # Save models first so evaluation can load them
    model_paths = save_models(cfg, pooled_results, peer_results)

    # Evaluate each saved pooled model
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
                    # CRITICAL: do not write rowwise predictions/targets
                    save_predictions=False,  # <--- changed from True
                    perm_repeats=5,
                    perm_sample=5000,
                    db_path=cfg.db_path,
                    target_col=name.replace("pooled_", ""),
                    save_report=True,
                )
                # Defensive: strip any accidental per-row dumps before we persist
                evaluation = _strip_per_row_payloads(evaluation)

                eval_results[name] = evaluation
                print(f"  ✓ {name}: RMSE={evaluation['metrics']['RMSE']:.6f}, R²={evaluation['metrics']['R2']:.3f}")
            except Exception as e:
                print(f"  ✗ Failed to evaluate {name}: {e}")
                eval_results[name] = {"error": str(e)}


    # Save aggregated results including evaluation
    metrics_path = save_results(cfg, pooled_results, peer_results, eval_results, baseline_results, peer_group_results)

    return {
        "baseline_results": baseline_results,
        "peer_group_results": peer_group_results,
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
    parser.add_argument("--tickers", nargs="+", default=["ASTS", "VZ", "T", "SATS"],
                        help="List of tickers to analyze")
    parser.add_argument("--start", default="2025-08-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-08-06", help="End date (YYYY-MM-DD)")
    
    # Model settings
    parser.add_argument("--forward-steps", type=int, default=15, 
                        help="Number of forward steps for prediction")
    parser.add_argument("--test-frac", type=float, default=0.2, 
                        help="Fraction of data to use for testing")
    parser.add_argument("--tolerance", default="2s", help="Time tolerance for merging")
    parser.add_argument("--r", type=float, default=0.045, help="Risk-free rate")
    
    # Peer effects settings
    parser.add_argument("--peer-targets", nargs="+", default=["ASTS", "VZ", "T", "SATS"],
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
    
    # Baseline correlation settings
    parser.add_argument("--no-baseline-correlations", action="store_true",
                        help="Disable baseline correlation computation")
    
    # Peer group analysis settings
    parser.add_argument("--enable-peer-group-analysis", action="store_true",
                        help="Enable comprehensive peer group analysis")
    parser.add_argument("--peer-groups", type=str, nargs="+", 
                        help="Define peer groups as 'group_name:ticker1,ticker2,...'")
    
    # Parse groups in format: "satellite:ASTS,SATS telecom:VZ,T"
    
    args = parser.parse_args()
    
    # Parse peer groups if provided
    peer_groups = None
    if args.peer_groups:
        peer_groups = {}
        for group_def in args.peer_groups:
            if ':' in group_def:
                group_name, tickers_str = group_def.split(':', 1)
                tickers = [t.strip() for t in tickers_str.split(',')]
                peer_groups[group_name] = tickers
            else:
                print(f"Warning: Invalid peer group format '{group_def}'. Expected 'name:ticker1,ticker2'")
    
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
        compute_baseline_correlations=not args.no_baseline_correlations,
        enable_peer_group_analysis=args.enable_peer_group_analysis,
        peer_groups=peer_groups,
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
