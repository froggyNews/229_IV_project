# analysis/peer_group_analysis.py
from __future__ import annotations

import json
import itertools
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import pandas as pd

# Project infra
from data_loader_coordinator import load_cores_with_auto_fetch
from feature_engineering import build_iv_panel
from train_peer_effects import run_peer_analysis, PeerConfig


# -----------------------
# Config
# -----------------------

@dataclass
class PeerGroupConfig:
    """Configuration for peer group analysis (drop-in compatible)."""
    groups: Dict[str, List[str]] = field(default_factory=dict)  # group_name -> [tickers]
    start: str = "2025-08-02"
    end: str = "2025-08-06"

    # Analysis settings
    target_kinds: List[str] = field(default_factory=lambda: ["iv_ret", "iv"])
    forward_steps: int = 15
    test_frac: float = 0.2
    tolerance: str = "15s"
    r: float = 0.045

    # Data
    db_path: Path = field(default_factory=lambda: Path("data/iv_data_1h.db"))
    auto_fetch: bool = True

    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs/peer_groups"))
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_detailed_results: bool = True
    debug: bool = False

    # Sensible defaults if not provided
    def __post_init__(self):
        if not self.groups:
            self.groups = {
                "satellite": ["ASTS", "SATS"],
                "telecom": ["VZ", "T"],
                "mixed": ["ASTS", "VZ", "T", "SATS"],
            }


# -----------------------
# Analyzer
# -----------------------

class PeerGroupAnalyzer:
    def __init__(self, config: PeerGroupConfig):
        self.config = config
        self.cores: Optional[Dict[str, pd.DataFrame]] = None
        self.all_tickers: List[str] = []
        self.results: Dict[str, Any] = {}
        self._validate_config()
        self._setup_dirs()

    # ----- setup & validation -----

    def _validate_config(self) -> None:
        if not self.config.groups:
            raise ValueError("At least one peer group must be defined")

        uniq = set()
        for tks in self.config.groups.values():
            uniq.update(tks)
        self.all_tickers = sorted(uniq)
        if len(self.all_tickers) < 2:
            raise ValueError("At least 2 unique tickers required")

        for name, tks in self.config.groups.items():
            if len(tks) < 2:
                warnings.warn(
                    f"Group '{name}' has only {len(tks)} ticker(s). "
                    "Intra-group correlation/peer effects will be limited."
                )

    def _setup_dirs(self) -> None:
        self.outdir = self.config.output_dir / self.config.timestamp
        for sub in ["correlations", "intra_group", "inter_group", "peer_effects", "statistical_tests"]:
            (self.outdir / sub).mkdir(parents=True, exist_ok=True)

    # ----- data loading -----

    def load_data(self) -> None:
        print(f"\n=== Loading cores for {len(self.all_tickers)} tickers ===")
        print(f"Date range: {self.config.start} → {self.config.end}")
        self.cores = load_cores_with_auto_fetch(  # robust loader + optional fetch
            tickers=self.all_tickers,
            start=self.config.start,
            end=self.config.end,
            db_path=self.config.db_path,
            auto_fetch=self.config.auto_fetch,
        )
        if not self.cores:
            raise ValueError("No cores loaded")

        for t in self.all_tickers:
            df = self.cores.get(t)
            if df is not None and not df.empty and "ts_event" in df.columns:
                print(f"  ✓ {t}: {len(df):,} rows "
                      f"{pd.to_datetime(df.ts_event.min()).strftime('%Y-%m-%d %H:%M')} → "
                      f"{pd.to_datetime(df.ts_event.max()).strftime('%Y-%m-%d %H:%M')}")
            else:
                print(f"  ✗ {t}: no data")

    # ----- correlations (time-aligned via panel) -----

    def _panel_subset(self, tickers: List[str]) -> pd.DataFrame:
        # Build a unified IV/IVRET panel for requested tickers
        sub = {t: self.cores[t] for t in tickers if t in self.cores and not self.cores[t].empty}
        panel = build_iv_panel(sub, tolerance=self.config.tolerance)
        return panel if panel is not None else pd.DataFrame(columns=["ts_event"])

    @staticmethod
    def _corr_of(matrix: pd.DataFrame) -> pd.DataFrame:
        # compute correlation over numeric columns only
        if matrix.empty:
            return pd.DataFrame()
        X = matrix.select_dtypes(include=[np.number]).dropna()
        if X.shape[1] < 2 or X.empty:
            return pd.DataFrame()
        return pd.DataFrame(np.corrcoef(X.values.T), index=X.columns, columns=X.columns)

    @staticmethod
    def _summary_from_corr(C: pd.DataFrame) -> Dict[str, float]:
        if C is None or C.empty or C.shape[0] < 2:
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
        mask = ~np.eye(C.shape[0], dtype=bool)
        vals = C.values[mask]
        return {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }

    def compute_intra_group_correlations(self) -> Dict[str, Any]:
        print("\n=== Intra‑group correlations (IVRET) ===")
        out: Dict[str, Any] = {}
        for gname, tks in self.config.groups.items():
            if len(tks) < 2:
                out[gname] = {"error": "insufficient tickers"}
                continue
            panel = self._panel_subset(tks)
            # restrict to IVRET_* columns for innovation‑style correlations
            cols = [c for c in panel.columns if c.startswith("IVRET_")]
            M = panel[cols].dropna(how="all").dropna(axis=0) if cols else pd.DataFrame()
            C = self._corr_of(M)
            stats = self._summary_from_corr(C)
            out[gname] = {
                "tickers": tks,
                "ret_corr_matrix": C.to_dict() if not C.empty else {},
                **stats,
            }
            print(f"  {gname}: mean={stats['mean']:.3f} std={stats['std']:.3f}")
        return out

    def compute_inter_group_correlations(self) -> Dict[str, Any]:
        print("\n=== Inter‑group correlations (IVRET cross blocks) ===")
        names = list(self.config.groups.keys())
        out: Dict[str, Any] = {}

        for a, b in itertools.combinations(names, 2):
            ta = [t for t in self.config.groups[a] if t in self.cores]
            tb = [t for t in self.config.groups[b] if t in self.cores]
            panel = self._panel_subset(ta + tb)
            cols = [c for c in panel.columns if c.startswith("IVRET_")]
            M = panel[cols].dropna(how="all").dropna(axis=0) if cols else pd.DataFrame()
            C = self._corr_of(M)
            xvals = []
            if not C.empty:
                # map columns back to tickers
                def tick(col: str) -> str:
                    return col.replace("IVRET_", "")
                a_cols = [c for c in C.columns if tick(c) in ta]
                b_cols = [c for c in C.columns if tick(c) in tb]
                for ca in a_cols:
                    for cb in b_cols:
                        if ca in C.index and cb in C.columns:
                            xvals.append(C.loc[ca, cb])
            if xvals:
                vec = np.array(xvals, float)
                stats = {
                    "mean": float(np.nanmean(vec)),
                    "std": float(np.nanstd(vec)),
                    "min": float(np.nanmin(vec)),
                    "max": float(np.nanmax(vec)),
                }
            else:
                stats = {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
            out[f"{a}_vs_{b}"] = {"group1": a, "group2": b, **stats}
            print(f"  {a} ↔ {b}: cross‑mean={stats['mean']:.3f}")
        return out

    # ----- peer effects -----

    def _run_peer_one(self, target: str, tickers: List[str], kind: str) -> Dict[str, Any]:
        cfg = PeerConfig(
            target=target,
            tickers=tickers,
            start=self.config.start,
            end=self.config.end,
            db_path=self.config.db_path,
            target_kind=kind,
            forward_steps=self.config.forward_steps,
            test_frac=self.config.test_frac,
        )
        r = run_peer_analysis(cfg, self.cores)
        # Keep only summary‑level outputs
        return {
            "status": r.get("status", "unknown"),
            "performance": r.get("performance", {}),
            "peer_rankings": (r.get("peer_analysis", {}) or {}).get("peer_rankings", {}),
            "self_lag_effects": (r.get("peer_analysis", {}) or {}).get("self_lag_effects", {}),
        }

    def compute_intra_group_peer_effects(self) -> Dict[str, Any]:
        print("\n=== Intra‑group peer effects ===")
        results: Dict[str, Any] = {}
        for gname, tks in self.config.groups.items():
            if len(tks) < 2:
                results[gname] = {"error": "insufficient tickers"}
                continue
            block: Dict[str, Any] = {}
            for kind in self.config.target_kinds:
                per_target = {}
                for tgt in tks:
                    if tgt not in self.cores or self.cores[tgt].empty:
                        per_target[tgt] = {"status": "no_data"}
                        continue
                    per_target[tgt] = self._run_peer_one(tgt, tks, kind)
                block[kind] = self._aggregate_peer_rankings(per_target)
            results[gname] = block
            print(f"  {gname}: done")
        return results

    def compute_inter_group_peer_effects(self) -> Dict[str, Any]:
        print("\n=== Inter‑group peer effects ===")
        out: Dict[str, Any] = {}
        names = list(self.config.groups.keys())
        for target_group in names:
            targets = [t for t in self.config.groups[target_group] if t in self.cores]
            if not targets:
                continue
            for peer_group in names:
                if peer_group == target_group:
                    continue
                peers = [t for t in self.config.groups[peer_group] if t in self.cores]
                if not peers:
                    continue

                block: Dict[str, Any] = {}
                for kind in self.config.target_kinds:
                    per_target = {}
                    collector: Dict[str, List[float]] = {}
                    feature_pool = list(set(targets + peers))
                    for tgt in targets:
                        r = self._run_peer_one(tgt, feature_pool, kind)
                        # keep only peers coming from the other group
                        cross = {p: float(v) for p, v in r.get("peer_rankings", {}).items() if p in peers}
                        per_target[tgt] = cross
                        for p, v in cross.items():
                            collector.setdefault(p, []).append(v)
                    # aggregate cross‑group effects
                    agg = {
                        p: {
                            "mean_effect": float(np.mean(vals)),
                            "std_effect": float(np.std(vals)),
                            "appearances": int(len(vals)),
                        }
                        for p, vals in collector.items()
                    }
                    block[kind] = dict(sorted(agg.items(), key=lambda kv: kv[1]["mean_effect"], reverse=True))
                key = f"{target_group}_from_{peer_group}"
                out[key] = {
                    "target_group": target_group,
                    "peer_group": peer_group,
                    "results": block,
                }
                print(f"  {target_group} ← {peer_group}: done")
        return out

    # ----- stats (lightweight placeholders) -----

    def run_statistical_tests(self) -> Dict[str, Any]:
        # Placeholders: simple Fisher‑z like summaries; you can extend as needed
        tests = {"correlation_tests": {}, "peer_effect_tests": {}}
        intra = self.results.get("intra_correlations", {})
        for g, data in intra.items():
            try:
                C = pd.DataFrame(data.get("ret_corr_matrix", {}))
                if C.shape[0] >= 2:
                    mask = ~np.eye(C.shape[0], dtype=bool)
                    z = np.arctanh(np.clip(C.values[mask], -0.999999, 0.999999))
                    tests["correlation_tests"][g] = {"z_mean": float(np.nanmean(z))}
            except Exception:
                tests["correlation_tests"][g] = {"z_mean": np.nan}
        # Peer tests: mean>0 one‑sample t‑like summary (non‑inferential placeholder)
        intra_pe = self.results.get("intra_peer_effects", {})
        for g, kinds in intra_pe.items():
            tests["peer_effect_tests"][g] = {}
            for kind, block in kinds.items():
                vals = [d["mean_effect"] for d in (block or {}).get("summary", {}).values()]
                tests["peer_effect_tests"][g][kind] = {"avg_effect": float(np.mean(vals)) if vals else np.nan}
        return tests

    # ----- aggregation & IO -----

    @staticmethod
    def _aggregate_peer_rankings(per_target: Dict[str, Any]) -> Dict[str, Any]:
        bag: Dict[str, List[float]] = {}
        per_target_clean = {}
        for tgt, obj in per_target.items():
            ranks = obj.get("peer_rankings", {}) if isinstance(obj, dict) else {}
            per_target_clean[tgt] = ranks
            for p, v in ranks.items():
                bag.setdefault(p, []).append(float(v))
        agg = {
            p: {
                "mean_effect": float(np.mean(vals)),
                "std_effect": float(np.std(vals)),
                "appearances": int(len(vals)),
            }
            for p, vals in bag.items()
        }
        agg_sorted = dict(sorted(agg.items(), key=lambda kv: kv[1]["mean_effect"], reverse=True))
        return {"targets": per_target_clean, "summary": agg_sorted}

    def run_full_analysis(self) -> Dict[str, Any]:
        print("🚀 Peer Group Analysis")
        self.load_data()
        self.results["intra_correlations"] = self.compute_intra_group_correlations()
        self.results["inter_correlations"] = self.compute_inter_group_correlations()
        self.results["intra_peer_effects"] = self.compute_intra_group_peer_effects()
        self.results["inter_peer_effects"] = self.compute_inter_group_peer_effects()
        self.results["statistical_tests"] = self.run_statistical_tests()
        self.results["metadata"] = {
            "config": self._config_dict(),
            "timestamp": self.config.timestamp,
            "groups": self.config.groups,
            "tickers": self.all_tickers,
        }
        if self.config.save_detailed_results:
            self._save_all()
        print("✅ Completed")
        return self.results

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "groups": self.config.groups,
            "start": self.config.start,
            "end": self.config.end,
            "target_kinds": self.config.target_kinds,
            "forward_steps": self.config.forward_steps,
            "test_frac": self.config.test_frac,
            "tolerance": self.config.tolerance,
            "r": self.config.r,
            "db_path": str(self.config.db_path),
            "auto_fetch": self.config.auto_fetch,
            "timestamp": self.config.timestamp,
            "debug": self.config.debug,
        }

    def _save_all(self) -> None:
        base = self.outdir
        (base / "correlations").mkdir(exist_ok=True)
        (base / "peer_effects").mkdir(exist_ok=True)
        (base / "statistical_tests").mkdir(exist_ok=True)

        with open(base / "peer_group_analysis.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # split saves (summary‑only)
        if "intra_correlations" in self.results:
            with open(base / "correlations" / "intra.json", "w") as f:
                json.dump(self.results["intra_correlations"], f, indent=2, default=str)
        if "inter_correlations" in self.results:
            with open(base / "correlations" / "inter.json", "w") as f:
                json.dump(self.results["inter_correlations"], f, indent=2, default=str)
        if "intra_peer_effects" in self.results:
            with open(base / "peer_effects" / "intra.json", "w") as f:
                json.dump(self.results["intra_peer_effects"], f, indent=2, default=str)
        if "inter_peer_effects" in self.results:
            with open(base / "peer_effects" / "inter.json", "w") as f:
                json.dump(self.results["inter_peer_effects"], f, indent=2, default=str)
        if "statistical_tests" in self.results:
            with open(base / "statistical_tests" / "tests.json", "w") as f:
                json.dump(self.results["statistical_tests"], f, indent=2, default=str)


# -----------------------
# Entrypoint
# -----------------------

def run_peer_group_analysis(config: PeerGroupConfig) -> Dict[str, Any]:
    return PeerGroupAnalyzer(config).run_full_analysis()


if __name__ == "__main__":
    cfg = PeerGroupConfig(
    groups = {
        "solar": ["FSLR", "SOL", "ENPH", "SEDG"],
        "quantum": ["QBTS", "IONQ", "SKYT", "RGTI", "QUBT"],
        "satellite": ["ASTS", "GSAT", "SATS"],
        "crypto_miners": ["MARA", "RIOT", "CIFR", "IREN", "WULF"],
        "airlines": ["AAL", "CPA", "DAL", "JBLU", "JETS"],
    },
        start="2025-08-02",
        end="2025-08-06",
        debug=True,
    )
    db_path = Path("data/iv_data_1h.db")
    res = run_peer_group_analysis(cfg)
    print("Keys:", list(res.keys()))
