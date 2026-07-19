#!/usr/bin/env python3
"""ingest_samples.py — Ingest raw MiCAM recordings into pipeline-ready layout.

The pipeline expects `data/<sample_id>/*.rsh|.gsh|.rsd|.gsd|.bvx|.npy`.
This helper takes raw files from a "messy" source directory and arranges them
into the right structure. It does NOT copy data — it produces (a) the target
directory layout and (b) shell commands you can paste to actually copy.

Why not auto-copy? Because (1) recordings are large and a dry-run is safer,
(2) copy decisions for partitioned .rsd recordings are non-trivial (must
include ALL chunks + .gsd), and (3) the user might want to verify the plan
first.

Usage:
    # 1. Dry-run plan from a source directory (recommended first)
    python3 ingest_samples.py plan /mnt/d/micam_dump --target data/

    # 2. Apply the plan (prints shell commands to run)
    python3 ingest_samples.py plan /mnt/d/micam_dump --target data/ --apply

    # 3. Inventory what's already in data/
    python3 ingest_samples.py inventory data/

    # 4. Generate a batch list (for `for s in $(cat batch.txt); do ...`)
    python3 ingest_samples.py batch data/ > batch.txt

    # 5. Auto-discover and ingest from /mnt/wslg/distro/tmp/
    python3 ingest_samples.py plan /mnt/wslg/distro/tmp --target data/ \\
        --pattern 'sham_*'  # optional glob to filter

Grouping rules (sample_id detection):
    1. If parent dir matches one of the patterns below, the dir name IS the sample.
       Patterns: sham_NNNx, mTACc-nola-*-NNNx, mSHAM-*-NNNx, NNNx (3 digits + A/B)
    2. Otherwise, sample_id is parsed from the filename:
       - Strip prefixes: 'recording_', date stamps like 2026-05-08-
       - Strip suffixes: .rsh, .gsh, .rsd, .gsd, .bvx, .npy
       - Take the LAST 4 chars if they look like 'NNNx' (NNN = digits, x = A|B|C)
       - Else: hash the filename and use first 8 hex chars as sample_id
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# ---------- sample_id detection ----------
KNOWN_DIR_PATTERNS = [
    re.compile(r"^(sham|mTAC[a-zA-Z-]*|mSHAM|wt|ko)[\-_]?(\d{3}[A-Z])$", re.IGNORECASE),
    re.compile(r"^(\d{3}[A-Z])$"),
    re.compile(r"^\d{4}-\d{2}-\d{2}-.+-(\d{3}[A-Z])_fresh$", re.IGNORECASE),  # existing data/ subdirs
]

SUFFIXES = (".rsh", ".gsh", ".rsd", ".gsd", ".bvx", ".npy")

# Filename patterns like "(0)", "(1)", " (0) " are RSD chunk indices — strip
CHUNK_SUFFIX_RE = re.compile(r"\s*\(\d+\)\s*$")


def detect_sample_id_from_dir(dirname: str) -> str | None:
    """If the directory name looks like a sample id, return it."""
    for pat in KNOWN_DIR_PATTERNS:
        m = pat.match(dirname)
        if m:
            # last group is the sample id (e.g. "004A")
            return m.group(m.lastindex) if m.lastindex else m.group(0)
    return None


def detect_sample_id_from_filename(filename: str) -> str:
    """Parse sample id from filename. Fallback to hash."""
    stem = filename
    for suf in SUFFIXES:
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break
    stem = CHUNK_SUFFIX_RE.sub("", stem)  # remove " (0)" etc.
    # Try to find NNNx at the end (3 digits + 1 letter)
    m = re.search(r"(\d{3}[A-Z])$", stem, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Try to find NNNx after last separator
    m = re.search(r"[-_](\d{3}[A-Z])(?:_|$)", stem, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: hash
    return "S" + hashlib.md5(filename.encode()).hexdigest()[:7]


def is_recording_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUFFIXES


# ---------- ingestion ----------
def find_recordings(root: Path, pattern: str | None) -> dict[str, list[Path]]:
    """Walk `root` and group recordings by sample_id.

    Returns: {sample_id: [list of files]}
    """
    grouped: dict[str, list[Path]] = defaultdict(list)

    # First pass: check if root itself is a single sample dir
    sid = detect_sample_id_from_dir(root.name)
    if sid:
        try:
            for f in sorted(root.iterdir()):
                if is_recording_file(f):
                    grouped[sid].append(f)
        except PermissionError:
            pass
        if grouped:
            return dict(grouped)

    # Otherwise, walk all subdirs
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if pattern and not re.search(pattern, sub.name):
            continue
        # skip permission-denied dirs (e.g. snap-private-tmp)
        try:
            children = list(sub.iterdir())
        except PermissionError:
            continue
        # Try dir name as sample id first
        sid = detect_sample_id_from_dir(sub.name)
        if sid is None:
            # Otherwise derive from filenames inside
            for f in sorted(children):
                if is_recording_file(f):
                    grouped[detect_sample_id_from_filename(f.name)].append(f)
            continue
        # Known dir pattern → use that name
        for f in sorted(children):
            if is_recording_file(f):
                grouped[sid].append(f)

    return dict(grouped)


def validate_partitioned(files: list[Path]) -> tuple[bool, str]:
    """For .rsd recordings, check we have .rsh + .gsd + ALL .rsd chunks.

    Returns (is_complete, warning_message).
    """
    has_rsh = any(f.suffix.lower() == ".rsh" for f in files)
    has_gsd = any(f.suffix.lower() == ".gsd" for f in files)
    chunks = sorted(f for f in files if f.suffix.lower() == ".rsd")
    if not chunks:
        return True, ""  # not partitioned
    if not has_rsh:
        return False, "missing .rsh companion"
    if not has_gsd:
        return False, "missing .gsd companion (required for partitioned)"
    # Check chunk indices are contiguous from (0)
    indices = []
    for c in chunks:
        m = re.search(r"\((\d+)\)", c.stem)
        if m:
            indices.append(int(m.group(1)))
    if indices != list(range(len(chunks))):
        return False, f"non-contiguous chunk indices: {indices}"
    return True, ""


def print_plan(grouped: dict[str, list[Path]], target_root: Path, apply: bool) -> int:
    print(f"# Plan: {len(grouped)} samples → {target_root}/")
    print(f"# Apply mode: {'YES' if apply else 'NO (dry-run, copy commands shown)'}")
    print()

    total_files = 0
    warnings = 0
    for sid, files in sorted(grouped.items()):
        target_dir = target_root / sid
        print(f"## {sid}  ({len(files)} files)")
        target_dir.mkdir(parents=True, exist_ok=True) if apply else None

        ok, warn = validate_partitioned(files)
        if not ok:
            print(f"  ⚠️  PARTITIONED INCOMPLETE: {warn}")
            warnings += 1
        elif warn:
            print(f"  ⚠️  {warn}")

        for f in files:
            target = target_dir / f.name
            total_files += 1
            if apply:
                # Use cp via shell to handle symlinks + speed
                os.system(f'cp -n "{f}" "{target}"')
                print(f"  copied   {f.name}")
            else:
                cmd = f'cp -n "{f}" "{target}"'
                print(f"  {cmd}")
        print()

    print(f"# Total: {total_files} files across {len(grouped)} samples")
    if warnings:
        print(f"# ⚠️  {warnings} samples have incomplete partitioned recordings")
    return 0 if warnings == 0 else 1


def print_inventory(data_root: Path) -> int:
    if not data_root.is_dir():
        print(f"ERROR: {data_root} not a directory", file=sys.stderr)
        return 2

    samples = []
    for sub in sorted(data_root.iterdir()):
        if not sub.is_dir():
            continue
        files = [f for f in sub.iterdir() if is_recording_file(f)]
        if not files:
            continue
        ok, warn = validate_partitioned(files)
        samples.append((sub.name, files, ok, warn))

    print(f"# Inventory: {len(samples)} samples in {data_root}/")
    print()
    print(f"{'sample_id':<25} {'files':>6}  {'size_mb':>9}  status")
    print("-" * 70)
    for sid, files, ok, warn in samples:
        size = sum(f.stat().st_size for f in files) / (1024 * 1024)
        status = "⚠️  " + warn if warn else ("✓" if ok else "⚠️")
        print(f"{sid:<25} {len(files):>6}  {size:>9.1f}  {status}")
    return 0


def print_batch(data_root: Path) -> int:
    if not data_root.is_dir():
        return 2
    for sub in sorted(data_root.iterdir()):
        if not sub.is_dir():
            continue
        files = [f for f in sub.iterdir() if is_recording_file(f)]
        if files:
            print(sub.name)
    return 0


# ---------- main ----------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest raw MiCAM recordings into cardiac_pipeline_v3 data/ layout"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Plan ingestion from a source directory")
    p_plan.add_argument("source", help="Source directory with raw .rsh/.gsh/.rsd/.gsd files")
    p_plan.add_argument("--target", default="data/", help="Target data/ dir (default: data/)")
    p_plan.add_argument("--pattern", default=None, help="Regex filter for subdir names (e.g. 'sham_')")
    p_plan.add_argument("--apply", action="store_true", help="Actually copy files (default: dry-run)")

    p_inv = sub.add_parser("inventory", help="List what is already in data/")
    p_inv.add_argument("data_root", default="data/", nargs="?")

    p_batch = sub.add_parser("batch", help="Print sample IDs one per line (for shell loops)")
    p_batch.add_argument("data_root", default="data/", nargs="?")

    args = parser.parse_args()
    target = Path(args.target if hasattr(args, "target") else "data/")

    if args.cmd == "plan":
        src = Path(args.source)
        if not src.is_dir():
            print(f"ERROR: {src} not a directory", file=sys.stderr)
            return 2
        grouped = find_recordings(src, args.pattern)
        if not grouped:
            print(f"# No recordings found under {src}", file=sys.stderr)
            return 1
        return print_plan(grouped, target, args.apply)

    if args.cmd == "inventory":
        return print_inventory(Path(args.data_root))

    if args.cmd == "batch":
        return print_batch(Path(args.data_root))

    return 0


if __name__ == "__main__":
    sys.exit(main())