from __future__ import annotations

"""
Generate figures for Slides 10–13 without Jupyter.

Outputs under `plots/`:
 - slide10_iv_level_corr_heatmap.png
 - slide10_iv_return_corr_heatmap.png
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
import xgboost as xgb

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
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
from src.feature_engineering import (
    build_iv_panel,
    DEFAULT_DB_PATH,
    build_pooled_iv_return_dataset_time_safe,
)

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
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "white"})


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


_SAVE_OPTS: dict[str, object] = {"dpi": 240, "formats": ["png"], "transparent": False}


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
            if "PC1" in comps:
                df = pd.DataFrame(list(comps["PC1"].items()), columns=["ticker", "loading"]).sort_values(
                    "loading", ascending=False
                )
                plt.figure(figsize=(7.2, 4.2))
                ax = plt.gca()
                ax.bar(df["ticker"], df["loading"], color=plt.cm.Blues(0.55))
                ax.set_title(f"PCA PC1 Loadings (IV Returns)\n{start} – {end}")
                ax.set_ylabel("Loading")
                ax.set_xlabel("")
                plt.xticks(rotation=0)
                _save(plots_dir / "baseline_pca_ivret_pc1_loadings.png")

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
        if tolerance == "2s":
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
        )
        clip_corr = corrs.get("clip", pd.DataFrame())
        ivret_corr = corrs.get("iv_returns", pd.DataFrame())

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

        if args.save_csv:
            if not clip_corr.empty:
                clip_corr.to_csv(plots_dir / "slide10_iv_level_corr.csv")
            if not ivret_corr.empty:
                ivret_corr.to_csv(plots_dir / "slide10_iv_return_corr.csv")
    except Exception as e:
        print(f"[WARN] Slide 10 heatmaps failed: {e}")

    # Baseline regression and PCA outputs
    if not getattr(args, "skip_regression", False):
        _save_regression_tables_and_plots(
            args.tickers, args.start, args.end, db_path, tolerance, plots_dir
        )
    if not getattr(args, "skip_pca", False):
        _save_pca_tables_and_plots(
            args.tickers, args.start, args.end, db_path, tolerance, plots_dir, n_components=3
        )

    # ---------- Slide 10 rolling correlations (optional) ----------
    if args.rolling_target:
        try:
            cores = load_cores_with_auto_fetch(
                list(args.tickers), args.start, args.end, db_path
            )
            panel = build_iv_panel(cores, tolerance=tolerance) if cores else None
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
                        if len(s) > 0:
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
        except Exception as e:
            print(f"[WARN] Rolling correlations failed: {e}")

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
    if getattr(args, "eval_files", None):
        eval_paths.extend([Path(p) for p in args.eval_files])
    if getattr(args, "eval_dir", None):
        d = Path(args.eval_dir)
        if d.exists():
            eval_paths.extend(sorted(d.glob("*_evaluation.json")))
    if eval_paths:
        _export_eval_tables(eval_paths, plots_dir)


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
    p.add_argument("--tolerance", default="2s")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--save-csv", action="store_true", help="Also save CSVs of outputs")
    p.add_argument("--eval-files", nargs="*", help="Evaluation JSON file(s) to export tables from")
    p.add_argument("--eval-dir", default="outputs/evaluations", help="Directory to scan for *_evaluation.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
