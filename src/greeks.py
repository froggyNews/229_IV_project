from __future__ import annotations
import numpy as np
from scipy.stats import norm

# Helper to check valid inputs

def _valid(S: float, K: float, T: float, r: float, sigma: float) -> bool:
    return np.isfinite([S, K, T, r, sigma]).all() and S > 0 and K > 0 and T > 0 and sigma > 0


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> float:
    if not _valid(S, K, T, r, sigma):
        return np.nan
    d1 = _d1(S, K, T, r, sigma)
    return norm.cdf(d1) if str(cp).upper().startswith("C") else norm.cdf(d1) - 1.0


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if not _valid(S, K, T, r, sigma):
        return np.nan
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if not _valid(S, K, T, r, sigma):
        return np.nan
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)
