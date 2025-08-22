import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Dict, Any
import json
from datetime import datetime

import pandas as pd

# Add the parent directory (src) to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
project_root = src_dir.parent

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from feature_engineering import (
    build_pooled_iv_return_dataset_time_safe,
    build_iv_return_dataset_time_safe,
)
from data_loader_coordinator import load_cores_with_auto_fetch
from train_iv_returns import train_xgb_iv_returns_time_safe_pooled
from train_peer_effects import PeerConfig, run_peer_analysis
from model_evaluation import evaluate_pooled_model


@dataclass
class ExperimentConfig:
    """Configuration for pooled vs isolated IV-return experiments."""
    # Core settings
    tickers: Sequence[str] = field(
        default_factory=lambda: ["QUBT", "QBTS", "RGTI", "IONQ"]
    )
    start: str = "2025-06-02"
    end: str = "2025-08-06"

    # Model settings
    forward_steps: int = 15
    test_frac: float = 0.2
    tolerance: str = "2s"
    r: float = 0.045

    # Data location
    db_path: Path = Path("data/iv_data_1m.db")
    
    # Saving settings
    save_models: bool = True
    save_reports: bool = True
    save_predictions: bool = True
    save_experiment_summary: bool = True
    perm_repeats: int = 10


def setup_experiment_directories(cfg: ExperimentConfig) -> Dict[str, Path]:
    """Create directory structure for experiment outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"iv_returns_experiment_{timestamp}"
    
    base_dir = project_root / "experiments" / experiment_name
    
    directories = {
        "base": base_dir,
        "models": base_dir / "models",
        "metrics": base_dir / "metrics", 
        "reports": base_dir / "reports",
        "predictions": base_dir / "predictions",
        "config": base_dir / "config"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    if cfg.save_experiment_summary:
        config_dict = {
            "tickers": list(cfg.tickers),
            "start": cfg.start,
            "end": cfg.end,
            "forward_steps": cfg.forward_steps,
            "test_frac": cfg.test_frac,
            "tolerance": cfg.tolerance,
            "r": cfg.r,
            "db_path": str(cfg.db_path),
            "timestamp": timestamp,
            "save_models": cfg.save_models,
            "save_reports": cfg.save_reports,
            "save_predictions": cfg.save_predictions,
            "perm_repeats": cfg.perm_repeats
        }
        
        with open(directories["config"] / "experiment_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
    
    return directories


def load_datasets(cfg: ExperimentConfig, auto_fetch: bool = True) -> Dict[str, pd.DataFrame]:
    """Load pooled and per-ticker datasets for the experiment."""

    cores = None
    if auto_fetch:
        cores = load_cores_with_auto_fetch(
            cfg.tickers, cfg.start, cfg.end, cfg.db_path
        )

    pooled = build_pooled_iv_return_dataset_time_safe(
        tickers=list(cfg.tickers),
        start=pd.Timestamp(cfg.start, tz="UTC"),
        end=pd.Timestamp(cfg.end, tz="UTC"),
        r=cfg.r,
        forward_steps=cfg.forward_steps,
        tolerance=cfg.tolerance,
        db_path=cfg.db_path,
        cores=cores,
    )

    isolated = build_iv_return_dataset_time_safe(
        tickers=list(cfg.tickers),
        start=pd.Timestamp(cfg.start, tz="UTC"),
        end=pd.Timestamp(cfg.end, tz="UTC"),
        r=cfg.r,
        forward_steps=cfg.forward_steps,
        tolerance=cfg.tolerance,
        db_path=cfg.db_path,
        cores=cores,
    )

    return {"pooled": pooled, "isolated": isolated}


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Train isolated, pooled, and peer models with proper saving.

    Parameters
    ----------
    cfg:
        Experiment settings describing tickers, date range and model params.

    Returns
    -------
    Dict[str, Any]
        Metrics for pooled, isolated and peer-effect models.
    """

    # Setup experiment directories
    directories = setup_experiment_directories(cfg)
    
    # Build datasets (this will auto-fetch cores if necessary)
    datasets = load_datasets(cfg, auto_fetch=True)
    pooled_df: pd.DataFrame = datasets["pooled"]
    isolated_dfs: Dict[str, pd.DataFrame] = datasets["isolated"]

    results: Dict[str, Any] = {"isolated": {}, "peer": {}, "experiment_info": {}}
    results["experiment_info"]["directories"] = {k: str(v) for k, v in directories.items()}

    # Train and evaluate one model per ticker (isolated)
    print("Training isolated models...")
    for ticker, df in isolated_dfs.items():
        print(f"  Training isolated model for {ticker}...")
        
        model, train_metrics = train_xgb_iv_returns_time_safe_pooled(
            df, test_frac=cfg.test_frac
        )
        
        # Save model
        model_path = None
        if cfg.save_models:
            model_path = directories["models"] / f"isolated_{ticker}.json"
            model.save_model(model_path)
        
        # Evaluate model
        evaluation = evaluate_pooled_model(
            model_path=model_path if cfg.save_models else model,
            tickers=[ticker],
            start=cfg.start,
            end=cfg.end,
            test_frac=cfg.test_frac,
            forward_steps=cfg.forward_steps,
            tolerance=cfg.tolerance,
            r=cfg.r,
            db_path=cfg.db_path,
            target_col="iv_ret_fwd",
            metrics_dir=directories["metrics"] if cfg.save_reports else None,
            save_report=cfg.save_reports,
            save_predictions=cfg.save_predictions,
            perm_repeats=cfg.perm_repeats,
        )
        
        # Save additional model-specific reports
        if cfg.save_reports:
            report_path = directories["reports"] / f"isolated_{ticker}_report.json"
            isolated_report = {
                "ticker": ticker,
                "model_type": "isolated",
                "train_metrics": train_metrics,
                "evaluation_metrics": evaluation["metrics"],
                "model_path": str(model_path) if model_path else None,
                "dataset_info": {
                    "total_samples": len(df),
                    "features": list(df.columns),
                    "date_range": [df.index.min(), df.index.max()]
                }
            }
            with open(report_path, "w") as f:
                json.dump(isolated_report, f, indent=2, default=str)
        
        results["isolated"][ticker] = evaluation["metrics"]

    # Train pooled model on combined dataset and evaluate
    print("Training pooled model...")
    pooled_model, pooled_train_metrics = train_xgb_iv_returns_time_safe_pooled(
        pooled_df, test_frac=cfg.test_frac
    )
    
    # Save pooled model
    pooled_model_path = None
    if cfg.save_models:
        pooled_model_path = directories["models"] / "pooled.json"
        pooled_model.save_model(pooled_model_path)
    
    # Evaluate pooled model
    pooled_eval = evaluate_pooled_model(
        model_path=pooled_model_path if cfg.save_models else pooled_model,
        tickers=list(cfg.tickers),
        start=cfg.start,
        end=cfg.end,
        test_frac=cfg.test_frac,
        forward_steps=cfg.forward_steps,
        tolerance=cfg.tolerance,
        r=cfg.r,
        db_path=cfg.db_path,
        target_col="iv_ret_fwd",
        metrics_dir=directories["metrics"] if cfg.save_reports else None,
        save_report=cfg.save_reports,
        save_predictions=cfg.save_predictions,
        perm_repeats=cfg.perm_repeats,
    )
    
    # Save pooled model report
    if cfg.save_reports:
        pooled_report_path = directories["reports"] / "pooled_model_report.json"
        pooled_report = {
            "model_type": "pooled",
            "tickers": list(cfg.tickers),
            "train_metrics": pooled_train_metrics,
            "evaluation_metrics": pooled_eval["metrics"],
            "model_path": str(pooled_model_path) if pooled_model_path else None,
            "dataset_info": {
                "total_samples": len(pooled_df),
                "features": list(pooled_df.columns),
                "date_range": [pooled_df.index.min(), pooled_df.index.max()]
            }
        }
        with open(pooled_report_path, "w") as f:
            json.dump(pooled_report, f, indent=2, default=str)
    
    results["pooled"] = pooled_eval["metrics"]

    # Peer-effects models (one model per target ticker)
    print("Training peer-effects models...")
    cores = load_cores_with_auto_fetch(cfg.tickers, cfg.start, cfg.end, cfg.db_path)
    for target in cfg.tickers:
        print(f"  Training peer model for {target}...")
        
        peer_cfg = PeerConfig(
            target=target,
            tickers=list(cfg.tickers),
            start=cfg.start,
            end=cfg.end,
            db_path=cfg.db_path,
            forward_steps=cfg.forward_steps,
            test_frac=cfg.test_frac,
        )
        peer_result = run_peer_analysis(peer_cfg, cores)
        
        # Save peer model reports
        if cfg.save_reports and peer_result:
            peer_report_path = directories["reports"] / f"peer_{target}_report.json"
            peer_report = {
                "target_ticker": target,
                "model_type": "peer_effects",
                "peer_tickers": [t for t in cfg.tickers if t != target],
                "results": peer_result
            }
            with open(peer_report_path, "w") as f:
                json.dump(peer_report, f, indent=2, default=str)
        
        results["peer"][target] = peer_result.get("performance", {})

    # Save experiment summary
    if cfg.save_experiment_summary:
        summary_path = directories["base"] / "experiment_summary.json"
        summary = {
            "experiment_type": "pooled_vs_isolated_iv_returns",
            "config": {
                "tickers": list(cfg.tickers),
                "date_range": [cfg.start, cfg.end],
                "forward_steps": cfg.forward_steps,
                "test_frac": cfg.test_frac
            },
            "results_summary": {
                "isolated_models": len(results["isolated"]),
                "pooled_model": "trained" if "pooled" in results else "failed",
                "peer_models": len(results["peer"])
            },
            "directories": results["experiment_info"]["directories"],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nExperiment completed. Results saved to: {directories['base']}")
        print(f"Summary saved to: {summary_path}")

    return results


if __name__ == "__main__":
    cfg = ExperimentConfig(
        # Enable saving
        save_models=True,
        save_reports=True, 
        save_predictions=True,
        save_experiment_summary=True,
        perm_repeats=10
    )
    
    metrics = run_experiment(cfg)
    print("\nFinal metrics summary:")
    print(json.dumps(metrics, indent=2, default=str))