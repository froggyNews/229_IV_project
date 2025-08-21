import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Dict, Any

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
    start: str = "2025-08-02"
    end: str = "2025-08-06"

    # Model settings
    forward_steps: int = 15
    test_frac: float = 0.2
    tolerance: str = "2s"
    r: float = 0.045

    # Data location
    db_path: Path = Path("data/iv_data_1m.db")


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
    """Train isolated, pooled, and peer models.

    Parameters
    ----------
    cfg:
        Experiment settings describing tickers, date range and model params.

    Returns
    -------
    Dict[str, Any]
        Metrics for pooled, isolated and peer-effect models.
    """

    # Build datasets (this will auto-fetch cores if necessary)
    datasets = load_datasets(cfg, auto_fetch=True)
    pooled_df: pd.DataFrame = datasets["pooled"]
    isolated_dfs: Dict[str, pd.DataFrame] = datasets["isolated"]

    results: Dict[str, Any] = {"isolated": {}, "peer": {}}

    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    # Train and evaluate one model per ticker (isolated)
    for ticker, df in isolated_dfs.items():
        model, _ = train_xgb_iv_returns_time_safe_pooled(
            df, test_frac=cfg.test_frac
        )
        model_path = model_dir / f"isolated_{ticker}.json"
        model.save_model(model_path)
        evaluation = evaluate_pooled_model(
            model_path=model_path,
            tickers=[ticker],
            start=cfg.start,
            end=cfg.end,
            test_frac=cfg.test_frac,
            forward_steps=cfg.forward_steps,
            tolerance=cfg.tolerance,
            r=cfg.r,
            db_path=cfg.db_path,
            target_col="iv_ret_fwd",
            metrics_dir=None,
            save_report=False,
            save_predictions=False,
            perm_repeats=0,
        )
        results["isolated"][ticker] = evaluation["metrics"]

    # Train pooled model on combined dataset and evaluate
    pooled_model, _ = train_xgb_iv_returns_time_safe_pooled(
        pooled_df, test_frac=cfg.test_frac
    )
    pooled_model_path = model_dir / "pooled.json"
    pooled_model.save_model(pooled_model_path)
    pooled_eval = evaluate_pooled_model(
        model_path=pooled_model_path,
        tickers=list(cfg.tickers),
        start=cfg.start,
        end=cfg.end,
        test_frac=cfg.test_frac,
        forward_steps=cfg.forward_steps,
        tolerance=cfg.tolerance,
        r=cfg.r,
        db_path=cfg.db_path,
        target_col="iv_ret_fwd",
        metrics_dir=None,
        save_report=False,
        save_predictions=False,
        perm_repeats=0,
    )
    results["pooled"] = pooled_eval["metrics"]

    # Peer-effects models (one model per target ticker)
    cores = load_cores_with_auto_fetch(cfg.tickers, cfg.start, cfg.end, cfg.db_path)
    for target in cfg.tickers:
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
        results["peer"][target] = peer_result.get("performance", {})

    return results


if __name__ == "__main__":
    cfg = ExperimentConfig()
    metrics = run_experiment(cfg)
    print(metrics)
