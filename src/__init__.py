"""Correlated Options Pricing Project - Source Package"""

__version__ = "0.1.0"

from .baseline_correlation import compute_baseline_correlations
from .baseline_regression import compute_baseline_regression
from .baseline_pca import compute_baseline_pca

__all__ = [
    "compute_baseline_correlations",
    "compute_baseline_regression",
    "compute_baseline_pca",
]
