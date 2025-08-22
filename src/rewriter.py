#!/usr/bin/env python3
"""
rewriter.py — one-shot project patcher

What it does (idempotent):
  1) train_peer_effects.py
     - Set PeerConfig.tolerance to str default "2s"
     - Remove duplicate/old prepare_peer_dataset() definitions
     - Ensure the canonical prepare_peer_dataset(cfg, cores, debug=False) exists
  2) group_transfer_iv.py
     - Define a timestamp string and use it instead of undefined `ts`
     - Ensure output dir exists before writing CSV/JSON
  3) model_evaluation.py
     - Allow evaluate_pooled_model(model_path=...) to accept an XGBRegressor
       instance OR a path; only resolve/load when given a path.
  4) (No-op safe) Prints a summary of applied changes and writes .bak backups.

Usage:
  python rewriter.py               # apply changes
  python rewriter.py --dry-run     # show planned edits only
  python rewriter.py --root src    # if files live under ./src
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import Tuple

# --------- utils ---------
def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        bak.write_text(read(p), encoding="utf-8")

def write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")

def replace_once(s: str, pattern: str, repl: str, flags=re.DOTALL) -> Tuple[str, bool]:
    new, n = re.subn(pattern, repl, s, flags=flags)
    return new, bool(n)

def find_project_file(root: Path, filename: str) -> Path | None:
    # search breadth-first for the file
    for p in [root] + [d for d in root.rglob("*") if d.is_dir() and len(d.parts) - len(root.parts) <= 3]:
        cand = p / filename
        if cand.exists():
            return cand
    return None

def report(ok: bool, msg: str):
    print(("✓ " if ok else "• ") + msg)

# --------- patchers ---------
def patch_train_peer_effects(path: Path, dry: bool) -> bool:
    """Unify tolerance type, de-dupe prepare_peer_dataset, ensure canonical version."""
    src = read(path)
    orig = src

    # 1) PeerConfig.tolerance -> str default "2s"
    # Accept either 'tolerance: float = 1e-6' or 'tolerance: ... = ...'
    src, ch1 = replace_once(
        src,
        r"(tolerance\s*:\s*[^\n=]+=\s*)([^\n]+)",
        r'\1"2s"',
    )

    # 2) Remove an earlier duplicate definition of prepare_peer_dataset(...) if present.
    # We keep the version that has signature `def prepare_peer_dataset(cfg, cores, debug: bool = False)`
    # Heuristic: delete any prepare_peer_dataset that does NOT include ", debug:"
    pat_dupe = r"""
        ^def\sprepare_peer_dataset\(
            \s*cfg\s*:\s*PeerConfig\s*,\s*cores\s*:\s*Dict\[str,\s*pd\.DataFrame\]\s*
        \)\s*->\s*pd\.DataFrame\s*:\s*\n  # no debug kw
        (?:\s.*\n)+?                      # function body
        (?=^def|\Z)
    """
    src2, ch2 = replace_once(src, pat_dupe, "", flags=re.DOTALL | re.MULTILINE | re.VERBOSE)

    # 3) Ensure canonical prepare_peer_dataset has debug kw (if not, add it)
    src3 = src2
    if not re.search(r"def\sprepare_peer_dataset\(\s*cfg:.*cores:.*debug:\s*bool\s*=\s*False\)", src2, re.DOTALL):
        # Try to retrofit the remaining prepare to include debug flag in signature and pass-through to build_*
        src3, ch3a = replace_once(
            src2,
            r"(def\sprepare_peer_dataset\(\s*cfg:\s*PeerConfig\s*,\s*cores:\s*Dict\[str,\s*pd\.DataFrame\]\s*)(\))",
            r"\1, debug: bool = False\2",
        )
        # Add debug=debug when calling build_target_peer_dataset(...)
        src3, ch3b = replace_once(
            src3,
            r"(build_target_peer_dataset\([\s\S]*?cores=cores)(\s*[,)]\s*)",
            r"\1, debug=debug\2",
        )
        ch3 = ch3a or ch3b
    else:
        ch3 = False

    changed = ch1 or ch2 or ch3
    if changed and not dry:
        write_backup(path)
        write(path, src3)
    report(changed, f"train_peer_effects.py patched ({'dry-run' if dry else 'written'})")
    return changed

def patch_group_transfer_iv(path: Path, dry: bool) -> bool:
    """Define timestamp, ensure dir, replace undefined ts variable in writes."""
    src = read(path)
    orig = src

    # Ensure output dir mkdir + timestamp creation right before writing section.
    # Add a safe block near the end, but we also replace any f"..._{ts}..." with timestamp.
    # a) introduce timestamp if not present
    if "timestamp = pd.Timestamp.utcnow().strftime(" not in src:
        # import pandas as pd might not be at top; ensure pd exists (file already imports pandas as pd)
        insert_point = len(src)
        # Add nothing globally; we will add inline before writes.
        pass

    # b) replace filenames that reference {ts} -> {timestamp}
    src1, ch1 = replace_once(src, r"\{ts\}", "{timestamp}")

    # c) ensure directory exists & introduce timestamp before writes
    # Try to find where res_df.to_csv is; insert prelude above it if missing.
    if "timestamp = pd.Timestamp.utcnow().strftime(" not in src1:
        prelude = (
            "    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)\n"
            "    timestamp = pd.Timestamp.utcnow().strftime(\"%Y%m%d_%H%M%S\")\n"
        )
        src1, ch2 = replace_once(
            src1,
            r"(\n\s*res_df\.to_csv\(.*\)\s*,?\s*index=False\))",
            lambda m: "\n" + prelude + m.group(0),
            flags=re.DOTALL,
        )
    else:
        ch2 = False

    changed = ch1 or ch2
    if changed and not dry:
        write_backup(path)
        write(path, src1)
    report(changed, f"group_transfer_iv.py patched ({'dry-run' if dry else 'written'})")
    return changed

def patch_model_evaluation(path: Path, dry: bool) -> bool:
    """Allow evaluate_pooled_model to accept an xgboost model instance OR a path."""
    src = read(path)
    orig = src

    # Add isinstance check and skip loading if a live model is passed.
    if "isinstance(model_path, xgb.XGBRegressor)" in src:
        changed = False
    else:
        # Insert helper import check is already there: 'import xgboost as xgb'
        # Replace the block that *always* resolves & loads the model
        src2, ch1 = replace_once(
            src,
            r"""
            #\s*Load\s*model[\s\S]*?
            model\s*=\s*xgb\.XGBRegressor\(\)\s*\n\s*model\.load_model\(str\(model_path\)\)
            """,
            (
                "# Load model (path OR live XGBRegressor)\n"
                "    if isinstance(model_path, xgb.XGBRegressor):\n"
                "        model = model_path\n"
                "    else:\n"
                "        model_path = _resolve_model_path(model_path, fallback_dir=\"models\")\n"
                "        model = xgb.XGBRegressor()\n"
                "        model.load_model(str(model_path))"
            ),
            flags=re.DOTALL,
        )
        changed = ch1

        if changed and not dry:
            write_backup(path)
            write(path, src2)

    report(changed, f"model_evaluation.py patched ({'dry-run' if dry else 'written'})")
    return changed

# --------- main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."), help="project root (where the files live)")
    ap.add_argument("--dry-run", action="store_true", help="show planned edits only")
    args = ap.parse_args()

    root = args.root.resolve()

    files = {
        "train_peer_effects.py": patch_train_peer_effects,
        "group_transfer_iv.py": patch_group_transfer_iv,
        "model_evaluation.py": patch_model_evaluation,
    }

    any_changed = False
    missing = []
    for fname, fn in files.items():
        path = find_project_file(root, fname)
        if not path:
            missing.append(fname)
            report(False, f"{fname} not found under {root}")
            continue
        try:
            changed = fn(path, args.dry_run)
            any_changed = any_changed or changed
        except Exception as e:
            print(f"✗ Error patching {fname}: {e}", file=sys.stderr)

    print("\nSummary:")
    if missing:
        print("  Missing files:", ", ".join(missing))
    print(f"  Changes applied: {'yes' if any_changed else 'no (already up-to-date or dry-run)'}")
    if not args.dry_run and any_changed:
        print("  Backups written alongside originals (*.bak)")

if __name__ == "__main__":
    main()
