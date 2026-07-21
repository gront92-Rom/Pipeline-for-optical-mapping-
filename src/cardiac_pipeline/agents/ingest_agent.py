"""
cardiac_pipeline.agents.ingest_agent — distribute raw MiCAM files by sample_id + treatment.

Reads a flat directory of .rsh/.rsd/.gsd/.gsh/.rsm files and moves/copies them
into per-sample/treatment folders under data/<SAMPLE_ID>/<TREATMENT>/.

Treatment extraction: regex matches patterns like -bs2, -bleb, -iso, -ca, -stretch.
Sample ID extraction: regex matches patterns like -001A, _055B.

Also writes per-sample metadata.json with the full treatment list and the
pipeline mode each treatment will use.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# Files we care about
INGEST_EXTENSIONS = {".rsh", ".rsd", ".gsd", ".gsh", ".rsm"}

# Sample ID: 3 digits + single uppercase letter, preceded by - or _
SAMPLE_ID_PATTERN = re.compile(r"[_-](\d{3}[A-Z])(?:\(\d+\))?")

# Treatment patterns: ordered by specificity (first match wins)
TREATMENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"-bs\d+\b"),     "bsl"),      # -bs1, -bs2, -bs10
    (re.compile(r"-bsl\b"),       "bsl"),      # -bsl
    (re.compile(r"-bleb\b"),      "bleb"),     # blebbistatin
    (re.compile(r"-iso\d*\b"),    "iso"),      # -iso, -iso1, -iso10
    (re.compile(r"-ca\b"),        "ca"),       # -ca
    (re.compile(r"-caff\b"),      "ca"),       # -caff
    (re.compile(r"-caffeine\b"),  "ca"),       # -caffeine
    (re.compile(r"-stretch\b"),   "stretch"),  # mechanical stretch
    (re.compile(r"-veh\b"),       "veh"),      # vehicle control
]

# Pipeline mode per treatment (which stages to run)
TREATMENT_PIPELINE_MODES: dict[str, list[str]] = {
    "bsl":     ["loader", "mask", "peak_detector", "activation", "apd", "conduction", "alternans", "cleaning", "report"],
    "bleb":    ["loader", "mask", "peak_detector", "apd", "cleaning", "report"],  # CaT only
    "iso":     ["loader", "mask", "peak_detector", "activation", "apd", "conduction", "alternans", "cleaning", "report"],
    "ca":      ["loader", "mask", "peak_detector", "apd", "cleaning", "report"],
    "stretch": ["loader", "mask", "cleaning", "report"],
    "veh":     ["loader", "mask", "peak_detector", "activation", "apd", "conduction", "alternans", "cleaning", "report"],
    "unknown": ["loader", "mask", "peak_detector", "activation", "apd", "conduction", "alternans", "cleaning", "report"],
}


def extract_sample_id(filename: str) -> str | None:
    """Extract sample ID (e.g. '001A') from filename."""
    m = SAMPLE_ID_PATTERN.search(filename)
    return m.group(1) if m else None


def extract_treatment(filename: str) -> str | None:
    """Extract treatment (e.g. 'bsl', 'iso') from filename."""
    for pattern, name in TREATMENT_PATTERNS:
        if pattern.search(filename):
            return name
    return None


@dataclass
class IngestSummary:
    """Result of an ingest run."""
    n_files_moved: int = 0
    n_files_skipped: int = 0
    n_samples: int = 0
    n_treatments: int = 0
    samples: dict[str, list[str]] = field(default_factory=dict)
    skipped_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_files_moved": self.n_files_moved,
            "n_files_skipped": self.n_files_skipped,
            "n_samples": self.n_samples,
            "n_treatments": self.n_treatments,
            "samples": self.samples,
            "skipped_files": self.skipped_files[:20],
        }


class IngestAgent:
    """Distributes raw files from a flat directory into per-sample/treatment folders."""

    def run(
        self,
        source_dir: str,
        out_dir: str = "data",
        *,
        copy: bool = False,
        force: bool = False,
        dry_run: bool = False,
    ) -> IngestSummary:
        """Run ingest.

        Args:
            source_dir: Flat directory with raw files.
            out_dir: Output root (default: data/).
            copy: Copy instead of move.
            force: Overwrite existing files.
            dry_run: Plan only, don't actually move/copy.

        Returns:
            IngestSummary with counts and per-sample breakdown.
        """
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source dir not found: {source}")

        out_root = Path(out_dir)
        if not dry_run:
            out_root.mkdir(parents=True, exist_ok=True)

        # Bucket files by (sample_id, treatment)
        buckets: dict[tuple[str, str], list[Path]] = defaultdict(list)
        skipped: list[str] = []

        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in INGEST_EXTENSIONS:
                continue
            sid = extract_sample_id(path.name)
            if sid is None:
                skipped.append(path.name)
                continue
            treatment = extract_treatment(path.name) or "unknown"
            buckets[(sid, treatment)].append(path)

        summary = IngestSummary()

        for (sid, treatment), files in sorted(buckets.items()):
            dest_dir = out_root / sid / treatment
            sample_metadata_path = out_root / sid / "metadata.json"

            for src in files:
                dst = dest_dir / src.name
                if dst.exists() and not force:
                    summary.n_files_skipped += 1
                    continue
                if dry_run:
                    action = "COPY" if copy else "MOVE"
                    print(f"  [DRY] {action} {src.name} -> {sid}/{treatment}/")
                else:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    if copy:
                        shutil.copy2(src, dst)
                    else:
                        shutil.move(str(src), str(dst))
                summary.n_files_moved += 1

            summary.samples.setdefault(sid, []).append(treatment)

            if not dry_run:
                _update_sample_metadata(sample_metadata_path, sid, treatment, files)

        summary.n_samples = len(summary.samples)
        summary.n_treatments = sum(len(t) for t in summary.samples.values())
        summary.skipped_files = skipped
        return summary


def _update_sample_metadata(path: Path, sample_id: str, treatment: str, files: list[Path]) -> None:
    """Write or update per-sample metadata.json with treatment info."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    else:
        meta = {}

    meta.setdefault("sample_id", sample_id)
    meta.setdefault("treatments", {})
    meta["treatments"].setdefault(treatment, {"files": [], "ingested_at": ""})
    meta["treatments"][treatment]["files"] = sorted(f.name for f in files)
    meta["treatments"][treatment]["ingested_at"] = datetime.now(timezone.utc).isoformat()
    meta["last_updated"] = datetime.now(timezone.utc).isoformat()

    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
