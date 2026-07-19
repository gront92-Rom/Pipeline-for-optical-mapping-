#!/usr/bin/env python3
"""compare_samples.py — Build a wide-format comparison CSV from one or many samples.

Reads `results/<sample>/must/metrics.json` for each sample and emits a single
CSV (and a markdown table) with one row per sample and one column per metric.

The metrics.json layout in this pipeline is FLAT — every agent writes its keys
at the top level (not nested by agent). To make columns human-readable we apply
a heuristic prefix based on the key name:

    loader__fps         (anything about loading / fps / n_frames / dye)
    mask__coverage      (coverage / snr / solidity / n_holes / compactness)
    peak__n_peaks       (peaks / threshold_frac / sigma_temporal / ...)
    activation__method  (method / tat_* / valid_coverage / n_active_pixels)
    apd__apd80_median   (apd* / cat* / level / inverted / preprocessing_owner)
    cv__cv_median       (cv_median / cv_mean / anisotropy / valid_fraction / verdict)
    alternans__phenotype (alternans_phenotype / AC_* / concordance / spectral_purity)
    sideline__threshold  (sideline_threshold / cleaning_* / etc.)
    other__<key>        (anything not matched above)

Usage:
    python3 compare_samples.py 004A 005A 006B
    python3 compare_samples.py --results-dir results --output compare.csv
    python3 compare_samples.py --results-dir results --all
    python3 compare_samples.py --all --format md
    python3 compare_samples.py --all --raw          # no heuristic prefix
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------- heuristic prefix routing ----------
# Order matters: first match wins. Keep names short and stable.
_PREFIX_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(fps|n_frames|height|width|dye|recording_mode|stim_hz|elapsed_s|loader_mode|sample_id|method|preprocessing_owner|inverted|falling_edge|parabolic_interp|hot_pixel_percentile|sigma_spatial|sigma_temporal|min_amp|threshold_frac|min_distance_factor|drop_first|min_agreement|frame_tolerance|soft_assignment_sigma|min_quality|n_regions|n_beats_selected|n_beats)$"), "loader"),
    (re.compile(r"^(coverage|solidity|n_holes|compactness|snr|level|s4_op|png|qcd|qcf)$"), "mask"),
    (re.compile(r"^(n_peaks|n_active_pixels|tat_|valid_coverage)$"), "activation"),
    (re.compile(r"^(apd|cat_|ca_)", re.IGNORECASE), "apd"),
    (re.compile(r"^(cv_|anisotropy|valid_pixels|total_pixels|valid_fraction|cv_min|cv_max|qc_threshold|verdict)$"), "cv"),
    (re.compile(r"^(alternans_|^ac_|^concordance|^spectral_purity|^poincare)"), "alternans"),
    (re.compile(r"^(sideline|cleaning_)"), "sideline"),
]


def assign_prefix(key: str) -> str:
    for pat, prefix in _PREFIX_RULES:
        if pat.search(key):
            return f"{prefix}__{key}"
    return f"other__{key}"


# ---------- I/O ----------
def load_metrics(sample_dir: Path) -> dict[str, Any]:
    p = sample_dir / "must" / "metrics.json"
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def flatten(metrics: dict[str, Any], sample_id: str, use_prefix: bool = True) -> dict[str, str]:
    row: dict[str, str] = {"sample_id": sample_id}
    for k, v in metrics.items():
        if isinstance(v, (dict, list)):
            continue  # nested / arrays — skip
        col = assign_prefix(k) if use_prefix else k
        row[col] = "" if v is None else str(v)
    return row


def discover_samples(results_dir: Path) -> list[str]:
    out: list[str] = []
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "must" / "metrics.json").exists():
            out.append(sub.name)
    return out


def _column_order(rows: list[dict[str, str]], preferred: list[str]) -> list[str]:
    cols = ["sample_id"]
    seen: set[str] = set()
    # Preferred order first (only if present)
    for c in preferred:
        if c == "sample_id":
            continue
        if any(c in r for r in rows):
            cols.append(c)
            seen.add(c)
    # Then the rest in insertion order
    for r in rows:
        for k in r:
            if k != "sample_id" and k not in seen:
                seen.add(k)
                cols.append(k)
    return cols


_PREFERRED_COLS = [
    "loader__fps",
    "loader__n_frames",
    "loader__dye",
    "loader__recording_mode",
    "loader__stim_hz",
    "loader__elapsed_s",
    "mask__coverage",
    "mask__solidity",
    "mask__compactness",
    "activation__n_active_pixels",
    "activation__valid_coverage",
    "apd__apd80_median_ms",
    "apd__apd80_iqr_ms",
    "apd__apd50_median_ms",
    "apd__cat80_median_ms",
    "cv__cv_median_m_per_s",
    "cv__cv_mean_m_per_s",
    "cv__anisotropy",
    "cv__valid_fraction",
    "cv__verdict",
    "alternans__alternans_phenotype",
    "alternans__AC_95th_ms",
    "alternans__concordance_index",
    "alternans__spectral_purity",
]


def write_csv(rows: list[dict[str, str]], path: Path, preferred: list[str]) -> None:
    if not rows:
        return
    cols = _column_order(rows, preferred)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}  ({len(rows)} rows × {len(cols)} columns)")


def write_markdown(rows: list[dict[str, str]], path: Path, preferred: list[str]) -> None:
    if not rows:
        return
    cols = _column_order(rows, preferred)
    with path.open("w") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
    print(f"Wrote {path}  ({len(rows)} rows × {len(cols)} columns)")


# ---------- main ----------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build comparison CSV/markdown from per-sample metrics.json"
    )
    parser.add_argument("samples", nargs="*", help="Sample IDs (e.g. 004A 005A 006B)")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing <sample>/must/metrics.json (default: results)")
    parser.add_argument("--all", action="store_true",
                        help="Auto-discover all samples under --results-dir")
    parser.add_argument("--output", default="comparison.csv",
                        help="Output CSV path (default: comparison.csv)")
    parser.add_argument("--format", choices=["csv", "md", "both"], default="both")
    parser.add_argument("--raw", action="store_true",
                        help="Disable heuristic prefix routing (use raw key names)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"ERROR: {results_dir} is not a directory", file=sys.stderr)
        return 2

    if args.all:
        samples = discover_samples(results_dir)
        if not samples:
            print(f"No samples found under {results_dir}", file=sys.stderr)
            return 2
    else:
        samples = args.samples
        if not samples:
            parser.error("Provide sample IDs or use --all")

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for sid in samples:
        sd = results_dir / sid
        if not (sd / "must" / "metrics.json").exists():
            missing.append(sid)
            continue
        rows.append(flatten(load_metrics(sd), sid, use_prefix=not args.raw))

    if missing:
        print(f"WARNING: missing metrics.json for: {', '.join(missing)}", file=sys.stderr)

    if not rows:
        print("No usable samples — nothing to write.", file=sys.stderr)
        return 1

    out_csv = Path(args.output)
    preferred = [] if args.raw else _PREFERRED_COLS
    if args.format in ("csv", "both"):
        write_csv(rows, out_csv, preferred)
    if args.format in ("md", "both"):
        write_markdown(rows, out_csv.with_suffix(".md"), preferred)

    return 0


if __name__ == "__main__":
    sys.exit(main())