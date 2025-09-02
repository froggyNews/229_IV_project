from __future__ import annotations

"""
Generate figures for Slides 10–13 without Jupyter.

Outputs under `plots/`:
 - slide10_iv_level_corr_heatmap.png
 - slide10_iv_return_corr_heatmap.png
 - slide10_iv_surface_corr_heatmap.png
 - slide10_iv_surface_return_corr_heatmap.png
 - slide10_rolling_corr_<TARGET>.png (optional)
 - baseline_regression_ivret_beta.png | baseline_regression_ivret_r2.png
 - baseline_regression_iv_returns.csv | baseline_regression_iv_levels.csv
 - baseline_pca_ivret_scree.png | baseline_pca_ivret_PC1_loadings.png | *_loadings.csv | *_evr.csv
 - slide12_importance_levels.png
 - slide13_importance_returns.png
 - evaluation-derived tables: *_xgb_importances.csv, *_permutation_importances.csv, *_shap_feature_avg.csv, *_shap_symbol_avg.csv
 - *_csv (optional with --save-csv for correlations)

Usage (example):
  python scripts/run_plots.py --tickers QBTS IONQ RGTI QUBT --start 2025-01-02 --end 2025-08-27 \
      --db data/iv_data_1h.db --rolling-target QBTS --eval-dir outputs/evaluations

Env override:
  IV_DB_PATH: path to SQLite DB (fallback if --db not given)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
from matplotlib import ticker as mticker
from cycler import cycler
import seaborn as sns
import re
import xgboost as xgb

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Robust stdout/stderr encoding on Windows consoles (avoid emoji crashes)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure project root and src/ on path to import local modules
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Prefer absolute imports via src/
from src.baseline_correlation import compute_baseline_correlations
from src.baseline_regression import compute_baseline_regression
from src.baseline_pca import compute_baseline_pca
from src.data_loader_coordinator import load_cores_with_auto_fetch
from src.surface_correlation import build_features_and_weights
from src.feature_engineering import (
    build_iv_panel,
    DEFAULT_DB_PATH,
    build_pooled_iv_return_dataset_time_safe,
)
from src.rolling_surface_eval import rolling_surface_evaluation

# ----------------------------
# Global style: clean blues
# ----------------------------
plt.rcParams.update({
    "figure.dpi": 200,           # higher preview DPI; export DPI is set at save time
    "savefig.dpi": 200,
    "figure.autolayout": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "white"})


def _filter_weekdays_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Keep Mon-Fri only to avoid weekend gaps in visuals."""
    try:
        return idx[idx.dayofweek < 5]
    except Exception:
        return idx

def _filter_trading_hours_index(
    idx: pd.DatetimeIndex,
    market_tz: str = "America/New_York",
    start_local: str = "09:30",
    end_local: str = "16:00",
) -> pd.DatetimeIndex:
    """Filter to regular trading hours (09:30–16:00 local market time)."""
    try:
        base = idx
        if not isinstance(idx, pd.DatetimeIndex):
            return idx
        if base.tz is None:
            aware = base.tz_localize("UTC")
        else:
            aware = base
        local = aware.tz_convert(market_tz)
        pos = local.indexer_between_time(start_local, end_local, include_start=True, include_end=True)
        return base.take(pos)
    except Exception:
        return idx

def _break_lines_at_day_boundaries(
    s: pd.Series, market_tz: str = "America/New_York"
) -> pd.Series:
    """Set first tick of each local day to NaN to break lines across days."""
    try:
        idx = s.index
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
            return s
        aware = idx.tz_localize("UTC") if idx.tz is None else idx
        local = aware.tz_convert(market_tz)
        day = pd.Series(local.date, index=s.index)
        first_of_day = day != day.shift(1)
        out = s.copy()
        out[first_of_day] = np.nan
        return out
    except Exception:
        return s


def _ensure_plots_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _blue_seq_cmap():
    return plt.cm.Blues


def _blue_div_cmap():
    # We display |corr| in blue and annotate with signed values.
    return plt.cm.Blues


def _blue_line_cycler(n: int = 10):
    cmap = _blue_seq_cmap()
    colors = [cmap(i) for i in np.linspace(0.35, 0.95, n)]
    return cycler(color=colors)


_SAVE_OPTS: dict[str, object] = {"dpi": 200, "formats": ["png"], "transparent": False}


def _add_footer_stamp(title_extras: str = ""):
    """Add a subtle bottom-right stamp with timestamp and context."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    txt = f"Generated {ts}  {title_extras}".strip()
    plt.gcf().text(0.99, 0.01, txt, ha="right", va="bottom", fontsize=8, color="#516b91", alpha=0.85)


def _save(figpath: Path, title_extras: str = ""):
    _add_footer_stamp(title_extras)
    dpi = int(_SAVE_OPTS.get("dpi", 240))
    formats = list(_SAVE_OPTS.get("formats", ["png"]))
    transparent = bool(_SAVE_OPTS.get("transparent", False))
    for fmt in formats:
        out = figpath.with_suffix(f".{fmt}")
        plt.savefig(out, bbox_inches="tight", dpi=dpi, transparent=transparent)
        print(f"[SAVED] {out}")
    plt.close()


def _save_heatmap(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    start: str,
    end: str,
    note: str = "",
) -> None:
    if df is None or df.empty:
        print(f"[WARN] {title}: not enough data; skipping heatmap {out_path.name}")
        return

    # Emphasize magnitude with a blue map; annotate signed values to show direction.
    data = df.copy()
    mag = data.abs()

    plt.figure(figsize=(9.0, 7.2))
    ax = sns.heatmap(
        mag,
        annot=data.round(2),
        fmt="",
        cmap=_blue_div_cmap(),
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Correlation |ρ|"},
        linewidths=0.6,
        linecolor="white",
        square=True,
    )
    ax.set_title(f"{title}\n{start} → {end}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.4)

    _save(out_path, note)


def _aggregate_importances_by_ticker(
    feature_gain: dict[str, float], tickers: Sequence[str]
) -> pd.DataFrame:
    if not feature_gain:
        return pd.DataFrame(columns=["ticker", "agg_gain", "gain_pct"])
    rows = [{"feature": k, "gain": float(v)} for k, v in feature_gain.items()]
    imp_df = pd.DataFrame(rows)

    def which_ticker(feat: str) -> str | None:
        for t in tickers:
            if t in feat:
                return t
        return None

    imp_df["ticker"] = imp_df["feature"].apply(which_ticker)
    agg = (
        imp_df.groupby("ticker", dropna=True)["gain"]
        .sum()
        .sort_values(ascending=False)
    )
    out = agg.reset_index().rename(columns={"gain": "agg_gain"})
    total = float(out["agg_gain"].sum()) if len(out) else 0.0
    out["gain_pct"] = (out["agg_gain"] / total * 100.0) if total > 0 else 0.0
    return out


def _train_and_importances(
    pooled: pd.DataFrame,
    target: str,
    tickers: Sequence[str],
    test_frac: float = 0.2,
) -> Tuple[xgb.XGBRegressor, float, pd.DataFrame]:
    if pooled is None or pooled.empty:
        raise ValueError("Empty pooled dataset")
    if target not in pooled.columns:
        raise KeyError(f"Missing target column: {target}")

    y = pd.to_numeric(pooled[target], errors="coerce")
    X = pooled.drop(columns=[target]).copy()

    # Minimal leakage control
    leak_cols: list[str] = []
    if target == "iv_clip":
        leak_cols += [c for c in ["iv_ret_fwd", "iv_ret_fwd_abs", "core_iv_ret_fwd_abs"] if c in X.columns]
    elif target == "iv_ret_fwd":
        leak_cols += [c for c in ["iv_ret_fwd_abs", "core_iv_ret_fwd_abs"] if c in X.columns]
    if leak_cols:
        X.drop(columns=leak_cols, inplace=True, errors="ignore")

    X = X.select_dtypes(include=["number", "bool"]).astype(float)

    n = len(X)
    if n < 100:
        raise ValueError(f"Too few rows to train: {n}")

    split = int(n * (1 - test_frac))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse = float(np.sqrt(((y_te - y_pred) ** 2).mean()))  # simple holdout RMSE

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")  # dict
    agg_df = _aggregate_importances_by_ticker(gain, tickers)
    return model, rmse, agg_df


def _plot_importance_bars(
    agg_df: pd.DataFrame,
    title: str,
    out_path: Path,
    start: str,
    end: str,
    note: str = "",
) -> None:
    if agg_df is None or agg_df.empty:
        print(f"[WARN] No importances to plot for {title}")
        return

    agg_df = agg_df.sort_values("agg_gain", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    cmap = _blue_seq_cmap()
    norm = Normalize(vmin=0.0, vmax=float(agg_df["agg_gain"].max()) if len(agg_df) else 1.0)
    colors = [cmap(0.35 + 0.6 * norm(v)) for v in agg_df["agg_gain"].to_numpy()]
    bars = ax.bar(
        agg_df["ticker"],
        agg_df["agg_gain"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_title(f"{title}\n{start} → {end}")
    ax.set_ylabel("Aggregate gain")
    ax.set_xlabel("")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    # Annotate each bar with raw gain and % share
    for rect, raw, pct in zip(bars, agg_df["agg_gain"], agg_df["gain_pct"]):
        ax.annotate(
            f"{raw:.2f}  ({pct:.1f}%)",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Secondary axis: cumulative % of gain (Pareto cue)
    cum_pct = agg_df["gain_pct"].cumsum()
    ax2 = ax.twinx()
    ax2.plot(
        np.arange(len(cum_pct)),
        cum_pct,
        marker="o",
        linewidth=1.2,
        alpha=0.8,
        color=plt.cm.Blues(0.6),
    )
    ax2.set_ylabel("Cumulative % of gain", rotation=270, labelpad=12)
    ax2.set_ylim(0, 105)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax2.grid(False)

    _save(out_path, note)


def _resolve_db_path(arg_db: str | None) -> Path:
    # Priority: CLI arg > IV_DB_PATH env (trimmed) > DEFAULT_DB_PATH
    if arg_db:
        return Path(arg_db)
    env_db = os.getenv("IV_DB_PATH")
    if env_db:
        p = Path(env_db.strip().strip('"').strip("'"))
        if str(p).strip():
            return p
    return Path(DEFAULT_DB_PATH)


def _load_xgb_gain_importance(model_path: Path) -> pd.DataFrame:
    """Load an XGBoost JSON model and return gain-based importances."""
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    bst = model.get_booster()
    gain = bst.get_score(importance_type="gain") or {}
    df = pd.DataFrame(
        [(k, float(v)) for k, v in gain.items()], columns=["feature", "gain"]
    ).sort_values("gain", ascending=False)
    return df


def _plot_surface_importance_heatmap(imp_df: pd.DataFrame, out_path: Path, title: str):
    """Aggregate feature gains back to KxT grid and plot a heatmap.

    Expects features named like K{i}_T{j}. Unknown patterns are ignored.
    """
    if imp_df is None or imp_df.empty:
        print(f"[WARN] No importance data to plot for {title}")
        return
    # Parse indices
    rows = []
    for _, r in imp_df.iterrows():
        f = str(r["feature"]) if "feature" in r else str(r.get(0, ""))
        m = re.match(r"^K(\d+)_T(\d+)$", f)
        if not m:
            continue
        i = int(m.group(1))
        j = int(m.group(2))
        rows.append((i, j, float(r["gain"])) )
    if not rows:
        print(f"[WARN] No K/T features parsed for {title}")
        return
    df = pd.DataFrame(rows, columns=["K", "T", "gain"]).groupby(["T", "K"]).sum().reset_index()
    k_max = int(df["K"].max())
    t_max = int(df["T"].max())
    grid = df.pivot(index="T", columns="K", values="gain").reindex(
        index=range(t_max + 1), columns=range(k_max + 1)
    ).fillna(0.0)

    plt.figure(figsize=(9.0, 6.6))
    ax = sns.heatmap(
        grid,
        cmap=_blue_seq_cmap(),
        cbar_kws={"label": "Gain"},
        linewidths=0.4,
        linecolor="white",
        square=False,
    )
    ax.set_title(title)
    ax.set_xlabel("Moneyness bin K")
    ax.set_ylabel("Maturity bin T")
    _save(out_path)


def _save_regression_tables_and_plots(
    tickers: Sequence[str],
    start: str,
    end: str,
    db_path: Path,
    tolerance: str,
    plots_dir: Path,
) -> None:
    """Run baseline regression and save CSV + simple bar charts."""
    try:
        res = compute_baseline_regression(
            tickers=tickers,
            start=start,
            end=end,
            db_path=db_path,
            tolerance=tolerance,
        )
        ivr = pd.DataFrame.from_dict(res.get("iv_returns", {}), orient="index")
        ivr.index.name = "ticker"
        if not ivr.empty:
            ivr.reset_index(inplace=True)
            ivr.to_csv(plots_dir / "baseline_regression_iv_returns.csv", index=False)

            # Betas
            plt.figure(figsize=(7.2, 4.2))
            ax = plt.gca()
            ax.bar(ivr["ticker"], ivr["beta"], color=plt.cm.Blues(0.6))
            ax.set_title(f"Baseline Regression Betas (IVRET vs market)\n{start} – {end}")
            ax.set_ylabel("Beta")
            ax.set_xlabel("")
            plt.xticks(rotation=0)
            _save(plots_dir / "baseline_regression_ivret_beta.png")

            # R^2
            plt.figure(figsize=(7.2, 4.2))
            ax = plt.gca()
            ax.bar(ivr["ticker"], ivr["r2"], color=plt.cm.Blues(0.45))
            ax.set_title(f"Baseline Regression R² (IVRET vs market)\n{start} – {end}")
            ax.set_ylabel("R²")
            ax.set_xlabel("")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=0)
            _save(plots_dir / "baseline_regression_ivret_r2.png")

        lvl = pd.DataFrame.from_dict(res.get("iv_levels", {}), orient="index")
        if not lvl.empty:
            lvl.index.name = "ticker"
            lvl.reset_index(inplace=True)
            lvl.to_csv(plots_dir / "baseline_regression_iv_levels.csv", index=False)

    except Exception as e:
        print(f"[WARN] Baseline regression failed: {e}")


def _save_pca_tables_and_plots(
    tickers: Sequence[str],
    start: str,
    end: str,
    db_path: Path,
    tolerance: str,
    plots_dir: Path,
    n_components: int = 3,
    surface_mode: str = "atm",
    surface_agg: str = "median",
) -> None:
    """Run baseline PCA and save EVR + loadings (tables + plots)."""
    try:
        res = compute_baseline_pca(
            tickers=tickers,
            start=start,
            end=end,
            db_path=db_path,
            tolerance=tolerance,
            n_components=n_components,
            include_levels=False,
            surface_mode=surface_mode,
            surface_agg=surface_agg,
        )
        ivr = res.get("iv_returns", {})
        evr = ivr.get("explained_variance_ratio", [])
        comps = ivr.get("components", {})

        if evr:
            pd.DataFrame({
                "component": [f"PC{i+1}" for i in range(len(evr))],
                "explained_var_ratio": evr,
            }).to_csv(plots_dir / "baseline_pca_ivret_evr.csv", index=False)

            # Scree plot
            plt.figure(figsize=(6.8, 4.0))
            ax = plt.gca()
            xs = np.arange(1, len(evr) + 1)
            ax.plot(xs, evr, marker="o", color=plt.cm.Blues(0.65))
            ax.set_title(f"PCA Scree (IV Returns)\n{start} – {end}")
            ax.set_xlabel("Component")
            ax.set_ylabel("Explained variance ratio")
            ax.set_xticks(xs)
            ax.set_ylim(0, max(0.01, max(evr) * 1.15))
            _save(plots_dir / "baseline_pca_ivret_scree.png")

        if comps:
            for name, loadings in comps.items():
                df = pd.DataFrame(list(loadings.items()), columns=["ticker", "loading"]).sort_values(
                    "loading", ascending=False
                )
                df.to_csv(plots_dir / f"baseline_pca_ivret_{name}_loadings.csv", index=False)
            if any(pc in comps for pc in ("PC1", "PC2", "PC3")):
                out = plot_pca_loadings_panel(
                    comps=comps,
                    start=start,
                    end=end,
                    plots_dir=plots_dir,
                    pcs=("PC1", "PC2", "PC3"),  # will auto-skip missing
                    sort_by="PC1",
                    top_n=None,                 # or e.g. 20
                    annotate=False,
                    fname="baseline_pca_ivret_pc123_loadings.png",
                )
                print("Saved:", out)

    except Exception as e:
        print(f"[WARN] Baseline PCA failed: {e}")


def _export_eval_tables(eval_paths: list[Path], out_dir: Path) -> None:
    import json
    for p in eval_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                js = json.load(f)
            stem = p.stem
            if js.get("xgb_importances"):
                pd.DataFrame(js["xgb_importances"]).to_csv(out_dir / f"{stem}_xgb_importances.csv", index=False)
            if js.get("permutation_importances"):
                pd.DataFrame(js["permutation_importances"]).to_csv(
                    out_dir / f"{stem}_permutation_importances.csv", index=False
                )
            if js.get("feature_shap_avg"):
                pd.DataFrame(js["feature_shap_avg"]).to_csv(out_dir / f"{stem}_shap_feature_avg.csv", index=False)
            if js.get("sym_shap_avg"):
                pd.DataFrame(js["sym_shap_avg"]).to_csv(out_dir / f"{stem}_shap_symbol_avg.csv", index=False)
            print(f"[SAVED] tables for {p}")
        except Exception as e:
            print(f"[WARN] Failed to export tables for {p}: {e}")

import math
from pathlib import Path
from typing import Dict, Mapping, Iterable, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_pca_loadings_panel(
    comps: Mapping[str, Mapping[str, float]],
    start: str,
    end: str,
    plots_dir: Path,
    pcs: Iterable[str] = ("PC1", "PC2", "PC3"),
    sort_by: str = "PC1",               # which PC to use for ticker sort (falls back to abs if missing)
    top_n: int | None = None,           # limit number of tickers (e.g., 20) or None for all
    annotate: bool = False,             # add numeric labels to bars
    fname: str = "pca_ivret_pc123_loadings.png",
) -> Path:
    """
    Create a panel of horizontal bar charts for PCA loadings (PC1–PC3).
    - comps: dict like {"PC1": {"IONQ": 0.4, ...}, "PC2": {...}, ...}
    """
    # Gather available PCs in order
    pcs = [pc for pc in pcs if pc in comps and isinstance(comps[pc], Mapping) and len(comps[pc]) > 0]
    if not pcs:
        raise ValueError("No requested principal components found in `comps`.")

    # Build tidy dataframe of loadings
    recs: List[Tuple[str, str, float]] = []
    all_tickers: set[str] = set()
    for pc in pcs:
        for t, v in comps[pc].items():
            recs.append((pc, t, float(v)))
            all_tickers.add(t)
    df = pd.DataFrame(recs, columns=["pc", "ticker", "loading"])

    # Choose sort reference
    sort_ref = sort_by if sort_by in pcs else pcs[0]
    # Compute sort order using absolute loading on the reference PC (fallback to 0 for missing)
    ref = (df[df["pc"] == sort_ref]
           .set_index("ticker")["loading"]
           .reindex(sorted(all_tickers)))
    ref_abs = ref.abs().fillna(0.0).sort_values(ascending=False)
    tickers_sorted = list(ref_abs.index)
    if top_n is not None:
        tickers_sorted = tickers_sorted[:int(top_n)]

    # Filter df to tickers in view and pivot-wide for consistent axis limits
    df = df[df["ticker"].isin(tickers_sorted)]
    # Find symmetric x-limits across all shown PCs for a clean visual scale
    lim = float(np.nanmax(np.abs(df["loading"].values))) if not df.empty else 1.0
    lim = max(lim, 1e-6)

    # Figure layout
    n = len(pcs)
    fig_h = max(2.8, 0.35 * len(tickers_sorted))  # scale height with number of tickers
    fig_w = 9.5
    fig, axes = plt.subplots(
        nrows=1, ncols=n, figsize=(fig_w, fig_h), sharey=True, constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    # Colormap for sign (no external libs)
    # positive -> one color, negative -> another
    pos_color = plt.cm.Blues(0.65)
    neg_color = plt.cm.Oranges(0.65)

    # Draw each PC subplot
    for ax, pc in zip(axes, pcs):
        sub = (df[df["pc"] == pc]
               .set_index("ticker")
               .reindex(tickers_sorted))
        y = np.arange(len(tickers_sorted))

        # Split +/-
        vals = sub["loading"].fillna(0.0).to_numpy()
        colors = [pos_color if v >= 0 else neg_color for v in vals]

        ax.barh(y, vals, height=0.6, color=colors, edgecolor="none")
        ax.axvline(0, color="0.3", lw=1, alpha=0.8)                  # zero line
        ax.grid(axis="x", color="0.9", lw=0.8)                       # light x-grid
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        ax.set_title(pc, fontsize=12, pad=8)
        ax.set_xlabel("Loading")

        if ax is axes[0]:
            ax.set_yticks(y, tickers_sorted, fontsize=10)
        else:
            ax.set_yticks([])

        # Optional numeric annotations
        if annotate:
            for yi, v in zip(y, vals):
                ax.text(
                    v + (0.01 * np.sign(v) if v != 0 else 0.01),
                    yi,
                    f"{v:.2f}",
                    va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=9,
                    color="0.25",
                )

    # Super title
    fig.suptitle(f"PCA Loadings (IV Returns): {start} – {end}", fontsize=13, y=1.02)
    # Subtle caption
    fig.text(0.01, 0.005, "Bars colored by sign (blue = +, orange = −). Tick ordering by |{}|.".format(sort_ref),
             fontsize=9, color="0.35")

    plots_dir.mkdir(parents=True, exist_ok=True)
    outpath = plots_dir / fname
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath

def run(args: argparse.Namespace) -> None:
    plots_dir = _ensure_plots_dir(Path(args.plots_dir))

    # Configure export quality/options
    try:
        fmts = [f.strip() for f in str(getattr(args, "export_formats", "png")).split(",") if f.strip()]
    except Exception:
        fmts = ["png"]
    _SAVE_OPTS["dpi"] = int(getattr(args, "export_dpi", 300))
    _SAVE_OPTS["formats"] = fmts
    _SAVE_OPTS["transparent"] = bool(getattr(args, "transparent", False))
    plt.rcParams["savefig.dpi"] = int(_SAVE_OPTS["dpi"])  # ensure consistent DPI

    # Resolve DB path robustly
    db_path = _resolve_db_path(args.db)

    # Determine timeframe and adjust defaults
    timeframe = (args.timeframe or ("1m" if str(db_path).endswith("1m.db") else "1h")).lower()
    forward_steps = args.forward_steps
    tolerance = args.tolerance
    rolling_window = args.rolling_window
    if timeframe == "1m":
        if forward_steps == 15:
            forward_steps = 60
        if tolerance == "15s":
            tolerance = "30s"
        if rolling_window is None:
            rolling_window = 23400  # ~60 trading days at 1m
    else:
        if rolling_window is None:
            rolling_window = 390  # ~60 trading days at 1h

    # If timeframe indicates 1m but DB looks hourly, switch to 1m DB
    if timeframe == "1m" and str(db_path).endswith("iv_data_1h.db"):
        db_path = Path("data/iv_data_1m.db")

    print(f"Tickers: {args.tickers}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Database: {db_path} | timeframe={timeframe} | tol={tolerance} | fwd={forward_steps}")

    # ---------- Slide 10 heatmaps ----------
    try:
        corrs = compute_baseline_correlations(
            tickers=args.tickers,
            start=args.start,
            end=args.end,
            db_path=db_path,
            tolerance=tolerance,
            surface_mode=getattr(args, "surface_mode", "atm"),
            surface_agg=getattr(args, "surface_agg", "median"),
            k_bins=int(getattr(args, "surface_k_bins", 10)),
            t_bins=int(getattr(args, "surface_t_bins", 10)),
            include_surface=not getattr(args, "no_surface", False),
            include_surface_returns=not getattr(args, "no_surface_returns", False),
            forward_steps=forward_steps,
            surface_return_method=str(getattr(args, "surface_return_method", "diff")),
        )
        clip_corr = corrs.get("clip", pd.DataFrame())
        ivret_corr = corrs.get("iv_returns", pd.DataFrame())
        surface_corr = corrs.get("surface", pd.DataFrame())
        surface_ret_corr = corrs.get("surface_returns", pd.DataFrame())

        _save_heatmap(
            clip_corr,
            "Historical IV Level Correlations",
            plots_dir / "slide10_iv_level_corr_heatmap.png",
            args.start,
            args.end,
            note=f"timeframe={timeframe}",
        )
        _save_heatmap(
            ivret_corr,
            "Historical IV Return Correlations",
            plots_dir / "slide10_iv_return_corr_heatmap.png",
            args.start,
            args.end,
            note=f"timeframe={timeframe}",
        )
        _save_heatmap(
            surface_corr,
            "Historical IV Surface Correlations",
            plots_dir / "slide10_iv_surface_corr_heatmap.png",
            args.start,
            args.end,
            note=f"timeframe={timeframe}",
        )
        _save_heatmap(
            surface_ret_corr,
            "Historical IV Surface Return Correlations",
            plots_dir / "slide10_iv_surface_return_corr_heatmap.png",
            args.start,
            args.end,
            note=f"timeframe={timeframe}",
        )

        if args.save_csv:
            if not clip_corr.empty:
                clip_corr.to_csv(plots_dir / "slide10_iv_level_corr.csv")
            if not ivret_corr.empty:
                ivret_corr.to_csv(plots_dir / "slide10_iv_return_corr.csv")
            if not surface_corr.empty:
                surface_corr.to_csv(plots_dir / "slide10_iv_surface_corr.csv")
            if not surface_ret_corr.empty:
                surface_ret_corr.to_csv(plots_dir / "slide10_iv_surface_return_corr.csv")
        # Optional: surface-based weights for a target
        if getattr(args, "surface_weights_target", None):
            try:
                feats, weights = build_features_and_weights(
                    tickers=args.tickers,
                    start=args.start,
                    end=args.end,
                    db_path=db_path,
                    k_bins=int(getattr(args, "surface_k_bins", 10)),
                    t_bins=int(getattr(args, "surface_t_bins", 10)),
                    agg=str(getattr(args, "surface_agg", "median")),
                    surface_mode=str(getattr(args, "surface_mode", "full")),
                    target=args.surface_weights_target,
                    clip_negative=True,
                    power=1.0,
                )
                if not weights.empty:
                    w = weights.iloc[0].dropna()
                    weights.to_csv(plots_dir / f"baseline_surface_weights_{args.surface_weights_target}.csv")
                    plt.figure(figsize=(8.6, 4.6))
                    ax = plt.gca()
                    ax.bar(w.index, w.values, color=plt.cm.Blues(0.6))
                    ax.set_title(
                        f"Surface Corr-Derived Weights for {args.surface_weights_target}\n{args.start} to {args.end}"
                    )
                    ax.set_ylabel("Weight")
                    ax.set_ylim(0, max(0.01, float(w.max()) * 1.15))
                    plt.xticks(rotation=0)
                    _save(plots_dir / f"baseline_surface_weights_{args.surface_weights_target}.png")
            except Exception as e2:
                print(f"[WARN] Surface weight computation failed: {e2}")
    except Exception as e:
        print(f"[WARN] Slide 10 heatmaps failed: {e}")

    # Baseline regression and PCA outputs
    if not getattr(args, "skip_regression", False):
        _save_regression_tables_and_plots(
            args.tickers, args.start, args.end, db_path, tolerance, plots_dir
        )
    if not getattr(args, "skip_pca", False):
        _save_pca_tables_and_plots(
            args.tickers,
            args.start,
            args.end,
            db_path,
            tolerance,
            plots_dir,
            n_components=3,
            surface_mode=getattr(args, "surface_mode", "atm"),
            surface_agg=getattr(args, "surface_agg", "median"),
        )

    # ---------- Slide 10 rolling correlations (optional) ----------
    if args.rolling_target:
        try:
            cores = load_cores_with_auto_fetch(
                list(args.tickers), args.start, args.end, db_path,
                atm_only=(args.surface_mode != "full")
            )
            panel = build_iv_panel(cores, tolerance=tolerance, agg=getattr(args, "surface_agg", "median")) if cores else None
            if panel is None or panel.empty:
                print("[WARN] Panel empty — skipping rolling correlations")
            else:
                win = rolling_window
                tgt = args.rolling_target
                tgt_col = f"IVRET_{tgt}"
                if tgt_col not in panel.columns:
                    print(f"[WARN] Missing target return column: {tgt_col}")
                else:
                    plt.figure(figsize=(10.5, 6.0))
                    ax = plt.gca()
                    ax.set_prop_cycle(_blue_line_cycler(n=len(args.tickers)))
                    for peer in args.tickers:
                        if peer == tgt:
                            continue
                        peer_col = f"IVRET_{peer}"
                        if peer_col not in panel.columns:
                            continue
                        s = (
                            panel[tgt_col]
                            .rolling(win, min_periods=max(5, win // 4))
                            .corr(panel[peer_col])
                            .dropna()
                        )
                        # Filter plotting to weekdays
                        if len(s) > 0:
                            try:
                                s = s.loc[_filter_weekdays_index(s.index)]
                                s = s.loc[_filter_trading_hours_index(s.index)]
                            except Exception:
                                pass
                            s = _break_lines_at_day_boundaries(s)
                            s.rename(peer, inplace=True)
                            ax.plot(s.index, s.values, label=peer, linewidth=1.4, alpha=0.95)

                    ax.axhline(0, color="#3b4d66", lw=0.8, linestyle="--", alpha=0.9)
                    ax.set_ylim(-1.05, 1.05)
                    ax.set_title(f"Rolling {win} bars — Historical IV Return Corr vs {tgt}\n{args.start} → {args.end}")
                    ax.set_ylabel("Correlation")
                    ax.set_xlabel("")
                    ax.legend(title="Peer", ncols=3, frameon=False, loc="upper left")

                    # time-aware x-axis
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
                    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
                    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

                    out = plots_dir / f"slide10_rolling_corr_{tgt}.png"
                    _save(out, f"timeframe={timeframe}, tol={tolerance}")

                # Also plot rolling correlations for IV levels (not returns)
                lvl_tgt_col = f"IV_{tgt}"
                if lvl_tgt_col not in panel.columns:
                    print(f"[WARN] Missing target level column: {lvl_tgt_col}")
                else:
                    plt.figure(figsize=(10.5, 6.0))
                    ax = plt.gca()
                    ax.set_prop_cycle(_blue_line_cycler(n=len(args.tickers)))
                    for peer in args.tickers:
                        if peer == tgt:
                            continue
                        peer_col = f"IV_{peer}"
                        if peer_col not in panel.columns:
                            continue
                        s = (
                            panel[lvl_tgt_col]
                            .rolling(win, min_periods=max(5, win // 4))
                            .corr(panel[peer_col])
                            .dropna()
                        )
                        # Filter plotting to weekdays
                        if len(s) > 0:
                            try:
                                s = s.loc[_filter_weekdays_index(s.index)]
                                s = s.loc[_filter_trading_hours_index(s.index)]
                            except Exception:
                                pass
                            s = _break_lines_at_day_boundaries(s)
                            s.rename(peer, inplace=True)
                            ax.plot(s.index, s.values, label=peer, linewidth=1.4, alpha=0.95)

                    ax.axhline(0, color="#3b4d66", lw=0.8, linestyle="--", alpha=0.9)
                    ax.set_ylim(-1.05, 1.05)
                    ax.set_title(f"Rolling {win} bars — Historical IV Level Corr vs {tgt}\n{args.start} → {args.end}")
                    ax.set_ylabel("Correlation")
                    ax.set_xlabel("")
                    ax.legend(title="Peer", ncols=3, frameon=False, loc="upper left")

                    # time-aware x-axis
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
                    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
                    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

                    out_levels = plots_dir / f"slide10_rolling_corr_levels_{tgt}.png"
                    _save(out_levels, f"timeframe={timeframe}, tol={tolerance}")
        except Exception as e:
            print(f"[WARN] Rolling correlations failed: {e}")

    # ---------- Rolling surface weights (optional) ----------
    if getattr(args, "surface_weights_target", None):
        try:
            tgt = args.surface_weights_target
            win = rolling_window
            eval_df = rolling_surface_evaluation(
                tickers=args.tickers,
                target=tgt,
                start=args.start,
                end=args.end,
                db_path=db_path,
                window=win,
                surface_mode=getattr(args, "surface_mode", "atm" if timeframe == "1h" else "full"),
                tolerance=tolerance,
                ridge_lambda=1e-3,
                include_weights=True,
            )
            if eval_df is None or eval_df.empty:
                print("[WARN] Rolling surface evaluation returned no rows")
            else:
                method = getattr(args, "surface_weights_method", "corr")
                prefix = {
                    "corr": "w_corr_",
                    "pca": "w_pca_",
                    "pcareg": "w_pcareg_",
                }.get(method, "w_corr_")
                weight_cols = [c for c in eval_df.columns if c.startswith(prefix)]
                if not weight_cols:
                    print(f"[WARN] No weight columns for method '{method}'")
                else:
                    # Choose top-k peers by average weight
                    k = int(getattr(args, "surface_weights_topk", 5))
                    means = eval_df[weight_cols].mean().sort_values(ascending=False)
                    top_cols = list(means.index[:k])

                    # Weekday-only subset for plotting
                    try:
                        idx_weekdays = _filter_weekdays_index(eval_df.index)
                        idx_hours = _filter_trading_hours_index(idx_weekdays)
                        plot_df = eval_df.loc[idx_hours, top_cols]
                    except Exception:
                        plot_df = eval_df[top_cols]

                    plt.figure(figsize=(10.5, 6.0))
                    ax = plt.gca()
                    ax.set_prop_cycle(_blue_line_cycler(n=len(top_cols)))
                    for col in top_cols:
                        label = col.replace(prefix, "")
                        s = _break_lines_at_day_boundaries(plot_df[col])
                        ax.plot(s.index, s.values, label=label, linewidth=1.4, alpha=0.95)
                    ax.set_title(f"Rolling {win} bars — {method.upper()} Weights vs {tgt}\n{args.start} → {args.end}")
                    ax.set_ylabel("Weight")
                    ax.set_xlabel("")
                    ax.set_ylim(0, 1.05)
                    ax.legend(title="Peer", ncols=3, frameon=False, loc="upper left")
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
                    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
                    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

                    out = plots_dir / f"rolling_surface_weights_{method}_{tgt}.png"
                    _save(out, f"timeframe={timeframe}, tol={tolerance}")

                if args.save_csv:
                    eval_df.to_csv(plots_dir / f"rolling_surface_eval_{tgt}.csv")
        except Exception as e:
            print(f"[WARN] Rolling surface weights failed: {e}")

    # ---------- Slides 12–13 pooled XGB importances ----------
    if not args.skip_pooled:
        try:
            start_ts = pd.Timestamp(args.start, tz="UTC")
            end_ts = pd.Timestamp(args.end, tz="UTC")
            pooled = build_pooled_iv_return_dataset_time_safe(
                tickers=args.tickers,
                start=start_ts,
                end=end_ts,
                forward_steps=forward_steps,
                tolerance=tolerance,
                db_path=db_path,
            )
            print(f"Pooled dataset: {len(pooled):,} rows, {pooled.shape[1]} columns")

            # Slide 12: levels
            try:
                _, rmse_lvl, agg_lvl = _train_and_importances(
                    pooled.copy(), "iv_clip", args.tickers, test_frac=args.test_frac
                )
                print(f"Levels RMSE: {rmse_lvl:.6f}")
                _plot_importance_bars(
                    agg_lvl,
                    "Slide 12: Historical IV Level — Per-Ticker Importance (gain)",
                    plots_dir / "slide12_importance_levels.png",
                    args.start,
                    args.end,
                    note=f"timeframe={timeframe}",
                )
                if args.save_csv and not agg_lvl.empty:
                    agg_lvl.to_csv(plots_dir / "slide12_importance_levels.csv", index=False)
            except Exception as e:
                print(f"[WARN] Slide 12 training/plot failed: {e}")

            # Slide 13: returns
            try:
                _, rmse_ret, agg_ret = _train_and_importances(
                    pooled.copy(), "iv_ret_fwd", args.tickers, test_frac=args.test_frac
                )
                print(f"Returns RMSE: {rmse_ret:.6f}")
                _plot_importance_bars(
                    agg_ret,
                    "Slide 13: Historical IV Return — Per-Ticker Importance (gain)",
                    plots_dir / "slide13_importance_returns.png",
                    args.start,
                    args.end,
                    note=f"timeframe={timeframe}",
                )
                if args.save_csv and not agg_ret.empty:
                    agg_ret.to_csv(plots_dir / "slide13_importance_returns.csv", index=False)
            except Exception as e:
                print(f"[WARN] Slide 13 training/plot failed: {e}")

        except Exception as e:
            print(f"[WARN] Building pooled dataset failed: {e}")

    # Export evaluation tables (feature importance, permutation, SHAP)
    eval_paths: list[Path] = []
    # Priority 1: explicit files provided
    if getattr(args, "eval_files", None):
        eval_paths.extend([Path(p) for p in args.eval_files])
    else:
        # Priority 2: latest file in eval_dir if requested
        if getattr(args, "eval_dir_latest", False):
            d = Path(getattr(args, "eval_dir", "outputs/evaluations"))
            if d.exists():
                files = sorted(d.glob("*_evaluation.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if files:
                    eval_paths.append(files[0])
        # Priority 3: all files in eval_dir if explicitly requested
        elif getattr(args, "export_all_evals", False):
            d = Path(getattr(args, "eval_dir", "outputs/evaluations"))
            if d.exists():
                eval_paths.extend(sorted(d.glob("*_evaluation.json")))
    if eval_paths:
        print(f"Exporting tables for {len(eval_paths)} evaluation file(s):")
        for p in eval_paths:
            print(" -", p)
        _export_eval_tables(eval_paths, plots_dir)

    # Surface model importances (if provided)
    if getattr(args, "surface_models", None):
        for mp in args.surface_models:
            try:
                model_path = Path(mp)
                imp = _load_xgb_gain_importance(model_path)
                if imp.empty:
                    print(f"[WARN] No gain importances in {model_path}")
                    continue
                # Save raw table
                imp.to_csv(plots_dir / f"surface_importances_{model_path.stem}.csv", index=False)
                # Top-N bar
                topn = int(getattr(args, "surface_top_n", 20))
                top = imp.head(topn)
                plt.figure(figsize=(9.5, 5.0))
                ax = plt.gca()
                ax.bar(top["feature"], top["gain"], color=plt.cm.Blues(0.6))
                ax.set_title(f"Surface Model Feature Gain (Top {topn})\n{model_path.name}")
                ax.set_ylabel("Gain")
                ax.set_xlabel("")
                plt.xticks(rotation=90)
                _save(plots_dir / f"surface_importances_top{topn}_{model_path.stem}.png")
                # KxT heatmap
                _plot_surface_importance_heatmap(
                    imp,
                    plots_dir / f"surface_importances_grid_{model_path.stem}.png",
                    title=f"Surface Model Gain Heatmap (K×T)\n{model_path.name}",
                )
            except Exception as e:
                print(f"[WARN] Surface model importance failed for {mp}: {e}")

    # Average surface per ticker (over the selected window)
    if getattr(args, "plot_surfaces", False):
        try:
            feats, _ = build_features_and_weights(
                tickers=args.tickers,
                start=args.start,
                end=args.end,
                db_path=db_path,
                k_bins=int(getattr(args, "surface_k_bins", 10)),
                t_bins=int(getattr(args, "surface_t_bins", 10)),
                agg=str(getattr(args, "surface_agg", "median")),
                surface_mode=str(getattr(args, "surface_mode", "full")),
                target=None,
            )
            if feats is not None and not feats.empty:
                # parse K and T indices
                kt = [c for c in feats.columns if c.startswith("K") and "_T" in c]
                # infer dims
                max_k = 0
                max_t = 0
                import re
                for c in kt:
                    m = re.match(r"^K(\d+)_T(\d+)$", c)
                    if m:
                        max_k = max(max_k, int(m.group(1)))
                        max_t = max(max_t, int(m.group(2)))
                for ticker in feats.index:
                    row = feats.loc[ticker]
                    data = {}
                    for c in kt:
                        m = re.match(r"^K(\d+)_T(\d+)$", c)
                        if not m:
                            continue
                        i = int(m.group(1)); j = int(m.group(2))
                        data[(j, i)] = float(row[c])
                    # build grid
                    grid = np.full((max_t + 1, max_k + 1), np.nan)
                    for (j, i), val in data.items():
                        grid[j, i] = val
                    plt.figure(figsize=(9.0, 6.6))
                    ax = sns.heatmap(
                        pd.DataFrame(grid),
                        cmap=_blue_seq_cmap(),
                        cbar_kws={"label": "IV level"},
                        linewidths=0.4,
                        linecolor="white",
                        square=False,
                    )
                    ax.set_title(f"Average IV Surface (K×T) — {ticker}\n{args.start} to {args.end}")
                    ax.set_xlabel("Moneyness bin K")
                    ax.set_ylabel("Maturity bin T")
                    _save(plots_dir / f"avg_surface_{ticker}.png")
        except Exception as e:
            print(f"[WARN] Average surface plotting failed: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate plots for Slides 10–13 (1h/1m-aware)")
    p.add_argument(
        "--tickers",
        nargs="+",
        default=["QBTS", "IONQ", "RGTI", "QUBT"],
        help="Tickers to include",
    )
    p.add_argument("--start", default="2025-06-02")
    p.add_argument("--end", default="2025-08-06")
    p.add_argument("--db", default=None, help="Path to SQLite DB (overrides IV_DB_PATH)")
    p.add_argument("--timeframe", choices=["1h", "1m"], default="1m", help="Data timeframe to set sensible defaults")
    p.add_argument("--plots-dir", default="plots")
    p.add_argument("--export-dpi", type=int, default=300, help="Export DPI for saved figures")
    p.add_argument("--export-formats", default="png", help="Comma-separated formats (e.g., 'png,svg')")
    p.add_argument("--transparent", action="store_true", help="Transparent background for exports")
    p.add_argument(
        "--surface-agg",
        choices=["median", "mean"],
        default="median",
        help="Aggregate IV across surface per timestamp (median or mean)",
    )
    p.add_argument(
        "--surface-mode",
        choices=["atm", "full"],
        default="atm",
        help="Use ATM-by-expiry only (atm) or full surface (all strikes+expiries)",
    )
    p.add_argument("--no-surface", action="store_true", help="Disable surface correlation heatmap")
    p.add_argument("--no-surface-returns", action="store_true", help="Disable surface returns correlation heatmap")
    p.add_argument(
        "--surface-k-bins",
        type=int,
        default=10,
        help="Number of moneyness bins for surface grids",
    )
    p.add_argument(
        "--surface-t-bins",
        type=int,
        default=10,
        help="Number of maturity bins for surface grids",
    )
    p.add_argument(
        "--surface-weights-target",
        default=None,
        help="Ticker to visualize rolling peer weights for surface reconstruction",
    )
    p.add_argument(
        "--surface-weights-method",
        choices=["corr", "pca", "pcareg"],
        default="corr",
        help="Weighting method to visualize (corr, pca, pcareg)",
    )
    p.add_argument(
        "--surface-weights-topk",
        type=int,
        default=5,
        help="Top-k peers by average weight to plot",
    )
    p.add_argument(
        "--surface-return-method",
        choices=["diff", "log", "pct"],
        default="diff",
        help="How to compute surface 'returns': diff (s_t - s_{t-1}), log (ln s_t - ln s_{t-1}), or pct ((s_t/s_{t-1})-1)",
    )
    p.add_argument("--skip-regression", action="store_true", help="Skip baseline regression outputs")
    p.add_argument("--skip-pca", action="store_true", help="Skip baseline PCA outputs")
    p.add_argument(
        "--rolling-target",
        default=None,
        help="Ticker for rolling correlation line plot (omit to skip)",
    )
    p.add_argument(
        "--rolling-window",
        type=int,
        default=None,
        help="Rolling window size in bars (auto: 390 for 1h, 23400 for 1m)",
    )
    p.add_argument("--skip-pooled", action="store_true", help="Skip pooled XGB analysis")
    p.add_argument("--forward-steps", type=int, default=15)
    p.add_argument("--tolerance", default="15s")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--save-csv", action="store_true", help="Also save CSVs of outputs")
    p.add_argument("--eval-files", nargs="*", help="Evaluation JSON file(s) to export tables from")
    p.add_argument("--eval-dir", default="outputs/evaluations", help="Directory to scan for *_evaluation.json")
    p.add_argument("--eval-dir-latest", action="store_true", help="Export tables only for the most recent *_evaluation.json in --eval-dir")
    p.add_argument("--export-all-evals", action="store_true", help="Export tables for all *_evaluation.json in --eval-dir")
    # Surface model importances
    p.add_argument("--surface-models", nargs="*", help="Path(s) to surface-trained model JSON files")
    p.add_argument("--surface-top-n", type=int, default=20, help="Top-N features for bar chart")
    # Average surface plots per ticker
    p.add_argument("--plot-surfaces", action="store_true", help="Plot average (K×T) surface per ticker")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
