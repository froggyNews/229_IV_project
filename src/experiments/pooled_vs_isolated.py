from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Dict, Any
import pandas as pd

from feature_engineering import (
    build_pooled_iv_return_dataset_time_safe,
    build_iv_return_dataset_time_safe,
)
from data_loader_coordinator import load_cores_with_auto_fetch
from train_iv_returns import train_xgb_iv_returns_time_safe_pooled
from train_peer_effects import (
    PeerConfig,
    prepare_peer_dataset,
    train_peer_model,
)


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
    """Load pooled and per-ticker datasets for the experiment.

    Parameters
    ----------
    cfg : ExperimentConfig
        Configuration with tickers, dates and model settings.
    auto_fetch : bool, optional
        If True, automatically fetch core data when missing.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing two keys:
        ``"pooled"`` -> pooled dataset for all tickers
        ``"isolated"`` -> dict of per-ticker datasets
    """

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

    # Train one model per ticker (isolated)
    for ticker, df in isolated_dfs.items():
        _, metrics = train_xgb_iv_returns_time_safe_pooled(
            df, test_frac=cfg.test_frac
        )
        results["isolated"][ticker] = metrics

    # Train pooled model on combined dataset
    _, pooled_metrics = train_xgb_iv_returns_time_safe_pooled(
        pooled_df, test_frac=cfg.test_frac
    )
    results["pooled"] = pooled_metrics

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
        peer_ds = prepare_peer_dataset(peer_cfg, cores)
        _, peer_metrics, _ = train_peer_model(peer_ds, peer_cfg.test_frac)
        results["peer"][target] = peer_metrics

    return results


if __name__ == "__main__":
    cfg = ExperimentConfig()
    metrics = run_experiment(cfg)
    print(metrics)
