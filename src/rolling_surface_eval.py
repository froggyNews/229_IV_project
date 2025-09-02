from __future__ import annotations

"""
Rolling surface reconstruction and evaluation on the full vol surface.

Implements three peer-weight methods learned on the surface (not slices):
 - Corr-based composite (non-negative, normalized)
 - PCA-market weights (PC1 loadings, non-negative, normalized)
 - PCA-regression (non-negative ridge via NNLS with Tikhonov augmentation)

All methods use a rolling estimation window [t-W+1, ..., t] and evaluate
reconstruction accuracy at t+1. Surface vectors are built by binning the
surface along days-to-expiry and moneyness, and taking per-bin medians.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls

from data_loader_coordinator import load_cores_with_auto_fetch


@dataclass
class SurfaceGrid:
    expiry_bins: Sequence[float] = (0, 7, 30, 90, 180, 365)
    moneyness_bins: Sequence[float] = (-0.3, -0.15, -0.07, 0.0, 0.07, 0.15, 0.3)

    def column_labels(self) -> List[str]:
        cols: List[str] = []
        for ei in range(len(self.expiry_bins) - 1):
            for mi in range(len(self.moneyness_bins) - 1):
                cols.append(f"E{ei}_M{mi}")
        return cols


def _bin_surface(df: pd.DataFrame, grid: SurfaceGrid) -> pd.DataFrame:
    """Pivot a ticker's rows into per-timestamp surface vectors on a fixed grid.

    Returns a DataFrame indexed by ts_event with columns like 'E<i>_M<j>'.
    Each cell is the median iv_clip within that (expiry, moneyness) bin.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["ts_event"] = pd.to_datetime(tmp["ts_event"], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=["ts_event", "iv_clip"]).sort_values("ts_event")

    # Ensure features exist
    if "time_to_expiry" not in tmp.columns:
        return pd.DataFrame()
    if "moneyness" not in tmp.columns:
        # If moneyness missing, try to estimate from strike and stock price
        if {"strike_price", "stock_close"}.issubset(tmp.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                tmp["moneyness"] = np.clip((tmp["strike_price"] / tmp["stock_close"]) - 1.0, -2.0, 2.0)
        else:
            return pd.DataFrame()

    tmp["days_to_expiry"] = pd.to_numeric(tmp["time_to_expiry"], errors="coerce") * 365.0
    tmp["moneyness"] = pd.to_numeric(tmp["moneyness"], errors="coerce")

    # Bin along the grid
    ebin = pd.cut(tmp["days_to_expiry"], grid.expiry_bins, right=False, labels=False)
    mbin = pd.cut(tmp["moneyness"], grid.moneyness_bins, right=False, labels=False)
    tmp = tmp.assign(_ebin=ebin, _mbin=mbin)
    tmp = tmp.dropna(subset=["_ebin", "_mbin"])  # rows outside bins

    # Aggregate median iv per cell per timestamp
    piv = (
        tmp.groupby(["ts_event", "_ebin", "_mbin"], as_index=False)["iv_clip"].median()
        .assign(col=lambda d: "E" + d["_ebin"].astype(int).astype(str) + "_M" + d["_mbin"].astype(int).astype(str))
        .pivot(index="ts_event", columns="col", values="iv_clip")
        .sort_index()
    )

    # Reindex to full grid columns
    cols = SurfaceGrid(grid.expiry_bins, grid.moneyness_bins).column_labels()
    piv = piv.reindex(columns=cols)
    piv.columns.name = None
    return piv


def build_surface_vectors(
    cores: Dict[str, pd.DataFrame], grid: Optional[SurfaceGrid] = None
) -> Dict[str, pd.DataFrame]:
    """Vectorize each ticker's surface on a common grid."""
    grid = grid or SurfaceGrid()
    out: Dict[str, pd.DataFrame] = {}
    for tkr, df in cores.items():
        mat = _bin_surface(df, grid)
        if mat is not None and not mat.empty:
            out[tkr] = mat
    return out


def _stack_time_cells(mat: pd.DataFrame, window_idx: np.ndarray) -> np.ndarray:
    """Flatten a window of rows (time) × columns (cells) to a 1D vector."""
    sub = mat.iloc[window_idx]
    return sub.values.reshape(-1)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return 0.0
    va, vb = a[mask], b[mask]
    if np.allclose(va, va.mean()) or np.allclose(vb, vb.mean()):
        return 0.0
    return float(np.corrcoef(va, vb)[0, 1])


def _normalize_weights_nonneg(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / (s + eps)


def _corr_weights(
    mats: Dict[str, pd.DataFrame], target: str, window_idx: np.ndarray, peers: List[str]
) -> np.ndarray:
    tgt_vec = _stack_time_cells(mats[target], window_idx)
    ws: List[float] = []
    for p in peers:
        pv = _stack_time_cells(mats[p], window_idx)
        r = _pearson_corr(tgt_vec, pv)
        ws.append(max(r, 0.0))
    return _normalize_weights_nonneg(np.array(ws))


def _pca_market_weights(
    mats: Dict[str, pd.DataFrame], target: str, window_idx: np.ndarray, peers: List[str]
) -> np.ndarray:
    # Build matrix with rows = time×cells, cols = peers
    cols = []
    for p in peers:
        cols.append(_stack_time_cells(mats[p], window_idx))
    A = np.column_stack(cols)
    # Standardize columns
    A = A.astype(float)
    mask = np.isfinite(A)
    # Replace missing with column means (computed on finite entries)
    for j in range(A.shape[1]):
        col = A[:, j]
        finite = np.isfinite(col)
        mu = col[finite].mean() if finite.any() else 0.0
        col[~finite] = mu
        A[:, j] = col
    scaler = StandardScaler(with_mean=True, with_std=True)
    A_std = scaler.fit_transform(A)
    pca = PCA(n_components=1, svd_solver="full", random_state=42)
    pca.fit(A_std)
    u = pca.components_[0]  # length = n_peers
    return _normalize_weights_nonneg(np.maximum(u, 0.0))


def _pca_regression_weights(
    mats: Dict[str, pd.DataFrame], target: str, window_idx: np.ndarray, peers: List[str], ridge: float = 1e-3
) -> np.ndarray:
    # Design matrix A: rows = time×cells, cols = peers
    cols = []
    for p in peers:
        cols.append(_stack_time_cells(mats[p], window_idx))
    A = np.column_stack(cols).astype(float)
    # Response b: flattened target
    b = _stack_time_cells(mats[target], window_idx).astype(float)
    # Replace NaNs with column means (A) and mean(b)
    for j in range(A.shape[1]):
        col = A[:, j]
        finite = np.isfinite(col)
        mu = col[finite].mean() if finite.any() else 0.0
        col[~finite] = mu
        A[:, j] = col
    b[~np.isfinite(b)] = np.nanmean(b)

    # Ridge via Tikhonov augmentation with NNLS positivity
    if ridge > 0:
        A_aug = np.vstack([A, np.sqrt(ridge) * np.eye(A.shape[1])])
        b_aug = np.concatenate([b, np.zeros(A.shape[1])])
    else:
        A_aug, b_aug = A, b

    w, _ = nnls(A_aug, b_aug)
    return _normalize_weights_nonneg(w)


def _reconstruct_at(
    mats: Dict[str, pd.DataFrame], time_pos: int, peers: List[str], weights: np.ndarray
) -> Tuple[pd.Series, List[str]]:
    """Reconstruct target surface at a single time position using peer weights.

    Returns reconstructed Series indexed by cell labels and the aligned columns list.
    """
    cols = mats[peers[0]].columns
    # Stack peer rows at time_pos
    peer_rows = []
    for p in peers:
        row = mats[p].iloc[time_pos]
        peer_rows.append(row.to_numpy())
    M = np.vstack(peer_rows)  # shape = (n_peers, n_cells)
    # Replace NaNs by column means across peers
    for j in range(M.shape[1]):
        col = M[:, j]
        finite = np.isfinite(col)
        mu = col[finite].mean() if finite.any() else 0.0
        col[~finite] = mu
        M[:, j] = col
    xhat = weights @ M  # (n_cells,)
    return pd.Series(xhat, index=cols), list(cols)


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    a = y_true.to_numpy(dtype=float)
    b = y_pred.reindex(y_true.index).to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return {"rmse": np.nan, "r2": np.nan, "cos": np.nan}
    a, b = a[mask], b[mask]
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    sst = float(np.sum((a - np.mean(a)) ** 2))
    sse = float(np.sum((a - b) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1e-12 else np.nan
    # Cosine similarity
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    cos = float((a @ b) / (na * nb)) if na > 0 and nb > 0 else np.nan
    return {"rmse": rmse, "r2": r2, "cos": cos}


def rolling_surface_evaluation(
    tickers: Sequence[str],
    target: str,
    start: str,
    end: str,
    db_path: Path | str,
    window: int,
    surface_mode: str = "full",
    grid: Optional[SurfaceGrid] = None,
    tolerance: str = "2s",
    ridge_lambda: float = 1e-3,
    include_weights: bool = False,
) -> pd.DataFrame:
    """Run rolling evaluation for Corr, PCA-market, and PCA-reg composites.

    Returns a DataFrame indexed by evaluation timestamp with columns like:
    ['rmse_corr','r2_corr','cos_corr','rmse_pca','r2_pca','cos_pca',
     'rmse_pcareg','r2_pcareg','cos_pcareg']
    """
    atm_only = (str(surface_mode).lower() != "full")
    cores = load_cores_with_auto_fetch(list(tickers), start, end, Path(db_path), atm_only=atm_only)
    grid = grid or SurfaceGrid()

    mats = build_surface_vectors(cores, grid)
    if target not in mats or any(p not in mats for p in tickers if p != target):
        return pd.DataFrame()

    peers = [p for p in tickers if p != target]
    # Align timelines: use target's timestamps and merge-asof peers to it
    tgt = mats[target].copy()
    # Reindex peers to target timeline with backward fill within tolerance
    for p in peers:
        df = mats[p].copy().reset_index()
        df = df.rename(columns={"ts_event": "_ts"})
        df_sorted = df.sort_values("_ts")
        # Build a single asof index merge by concatenating values columns back after asof on a key
        aligned = []
        for col in df.columns:
            if col == "_ts":
                continue
            left = pd.DataFrame({"ts_event": tgt.index})
            right = df_sorted[["_ts", col]].rename(columns={"_ts": "ts_event"})
            merged = pd.merge_asof(
                left.sort_values("ts_event"),
                right.sort_values("ts_event"),
                on="ts_event",
                direction="backward",
                tolerance=pd.Timedelta(tolerance),
            )
            aligned.append(merged.set_index("ts_event")[col])
        mats[p] = pd.concat(aligned, axis=1)
        mats[p].columns = mats[target].columns  # ensure same column order
        mats[p] = mats[p].loc[tgt.index]

    n = len(tgt)
    if n <= window + 1:
        return pd.DataFrame()

    records = []
    index = []
    for i in range(window - 1, n - 1):
        win_idx = np.arange(i - window + 1, i + 1)
        eval_pos = i + 1

        # Corr-based weights
        w_corr = _corr_weights(mats, target, win_idx, peers)
        xhat_corr, cols = _reconstruct_at(mats, eval_pos, peers, w_corr)
        y_true = tgt.iloc[eval_pos]
        m_corr = _metrics(y_true, xhat_corr)

        # PCA-market weights
        w_pca = _pca_market_weights(mats, target, win_idx, peers)
        xhat_pca, _ = _reconstruct_at(mats, eval_pos, peers, w_pca)
        m_pca = _metrics(y_true, xhat_pca)

        # PCA-regression weights
        w_reg = _pca_regression_weights(mats, target, win_idx, peers, ridge=ridge_lambda)
        xhat_reg, _ = _reconstruct_at(mats, eval_pos, peers, w_reg)
        m_reg = _metrics(y_true, xhat_reg)

        rec = {
            "rmse_corr": m_corr["rmse"], "r2_corr": m_corr["r2"], "cos_corr": m_corr["cos"],
            "rmse_pca": m_pca["rmse"], "r2_pca": m_pca["r2"], "cos_pca": m_pca["cos"],
            "rmse_pcareg": m_reg["rmse"], "r2_pcareg": m_reg["r2"], "cos_pcareg": m_reg["cos"],
        }
        if include_weights:
            for j, p in enumerate(peers):
                rec[f"w_corr_{p}"] = float(w_corr[j])
                rec[f"w_pca_{p}"] = float(w_pca[j])
                rec[f"w_pcareg_{p}"] = float(w_reg[j])
        records.append(rec)
        index.append(tgt.index[eval_pos])

    out = pd.DataFrame.from_records(records, index=pd.Index(index, name="ts_event"))
    return out


__all__ = [
    "SurfaceGrid",
    "build_surface_vectors",
    "rolling_surface_evaluation",
]
