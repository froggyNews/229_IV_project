from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Dict
import pandas as pd

from feature_engineering import (
    build_pooled_iv_return_dataset_time_safe,
    build_iv_return_dataset_time_safe,
)
from data_loader_coordinator import load_cores_with_auto_fetch


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
