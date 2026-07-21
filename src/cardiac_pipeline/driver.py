"""
cardiac_pipeline.driver — pipeline orchestrator.

The 'brain' of cardiac-pipeline. Reads top-to-bottom as a plan:
  1. Build a PipelineResult.
  2. For each stage in STAGES: run or skip, record result, write manifest.
  3. Finalise: write summary, return.

All magic strings live in constants.py; stage registry in stages.py;
dataclasses in models.py. This file contains only orchestration logic.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cardiac_pipeline import constants as C
from cardiac_pipeline.models import (
    PipelineResult,
    StageResult,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_SKIPPED,
)
from cardiac_pipeline.stages import STAGES, Stage


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

def manifest_path(sample_id: str, stage_id: str, results_root: Path = Path(C.RESULTS_ROOT_DEFAULT)) -> Path:
    """Per-stage manifest: results/<sample>/must/stage_<id>.json."""
    return results_root / sample_id / C.MUST_SUBDIR / C.manifest_filename(stage_id)


def summary_path(sample_id: str, results_root: Path = Path(C.RESULTS_ROOT_DEFAULT)) -> Path:
    """Aggregated run-summary: results/<sample>/summary.json."""
    return results_root / sample_id / C.SUMMARY_FILENAME


# -----------------------------------------------------------------------------
# Input auto-detection
# -----------------------------------------------------------------------------

def auto_detect_input(sample_id: str, data_root: Path = Path(C.DATA_ROOT_DEFAULT)) -> Optional[str]:
    """Find the raw input file for a sample. Returns path as string or None.

    Search priority: any file in data/<sample>/ matching INPUT_FILE_EXTENSIONS.
    Returns the first one found.
    """
    sample_dir = data_root / sample_id
    if not sample_dir.is_dir():
        return None
    for ext in C.INPUT_FILE_EXTENSIONS:
        matches = sorted(sample_dir.glob(f"*{ext}"))
        if matches:
            return str(matches[0])
    return None


# -----------------------------------------------------------------------------
# Manifest I/O (atomic writes)
# -----------------------------------------------------------------------------

def write_atomic(path: Path, payload: dict) -> None:
    """Atomic write: tmp file + replace. Safe on POSIX and Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def is_cached(sample_id: str, stage_id: str, results_root: Path = Path(C.RESULTS_ROOT_DEFAULT)) -> bool:
    """True if a manifest exists (and is readable) for this stage."""
    p = manifest_path(sample_id, stage_id, results_root)
    return p.is_file() and p.stat().st_size > 0


# -----------------------------------------------------------------------------
# Stage execution
# -----------------------------------------------------------------------------

def run_one_stage(
    stage: Stage,
    sample_id: str,
    cfg,
    input_file: Optional[str],
    force: bool,
    debug: bool,
    results_root: Path,
) -> StageResult:
    """Execute one stage. Returns a StageResult. Never raises — exceptions are captured."""
    # Cache check
    if not force and is_cached(sample_id, stage.name, results_root):
        return StageResult.skipped(stage.name, manifest_path(sample_id, stage.name, results_root))

    # Build agent + kwargs
    agent = stage.agent_cls(sample_id, config=cfg)
    kwargs: dict = {"force": True}
    if stage.requires_input and input_file:
        kwargs["input_path"] = input_file
    if debug:
        kwargs["debug"] = True

    # Run
    t0 = time.time()
    try:
        agent.run(**kwargs)
    except Exception as exc:
        elapsed = time.time() - t0
        return StageResult.failed(stage.name, elapsed, exc)

    elapsed = time.time() - t0
    mpath = manifest_path(sample_id, stage.name, results_root)
    write_atomic(mpath, {
        "stage": stage.name,
        "status": STATUS_OK,
        "elapsed_s": round(elapsed, 2),
        "finished_at": _now_iso(),
    })
    return StageResult.succeed(stage.name, elapsed, mpath)


# -----------------------------------------------------------------------------
# Pipeline driver
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_pipeline(
    sample_id: str,
    *,
    force: bool = False,
    input_file: Optional[str] = None,
    debug: bool = False,
    results_root: str = C.RESULTS_ROOT_DEFAULT,
) -> PipelineResult:
    """Run the full pipeline for one sample. Returns a PipelineResult.

    This is the only public entry point. The body is intentionally linear:
    set up → loop stages → finalise.
    """
    results_root = Path(results_root)
    input_file = input_file or auto_detect_input(sample_id)
    if not input_file:
        raise FileNotFoundError(
            f"No input file found for sample {sample_id!r} in {C.DATA_ROOT_DEFAULT}/{sample_id}/. "
            f"Pass --input <path> or place a .rsh/.gsh/.rsd/.gsd file there."
        )

    cfg = _build_config(sample_id, results_root)

    result = PipelineResult(
        sample_id=sample_id,
        input_file=input_file,
        started_at=_now_iso(),
    )
    t_start = time.time()

    for stage in STAGES:
        sr = run_one_stage(stage, sample_id, cfg, input_file, force, debug, results_root)
        result.record(sr)
        if sr.status == STATUS_ERROR:
            break  # halt on first error

    result.finished_at = _now_iso()
    result.total_elapsed_s = time.time() - t_start
    result.ok = result.failed_stage == "" and all(
        s.status in (STATUS_OK, STATUS_SKIPPED) for s in result.stages.values()
    )

    write_atomic(summary_path(sample_id, results_root), result.to_dict())
    return result


# -----------------------------------------------------------------------------
# Config (lightweight wrapper around env vars used by agents)
# -----------------------------------------------------------------------------

def _build_config(sample_id: str, results_root: Path):
    """Build the agent config object. Mirrors the env vars set by run_cardiac.sh.

    PipelineConfig loads from config/default.yaml and exposes results_root,
    data_root, pixel_size_mm, plus nested per-agent config dicts.
    """
    from cardiac_pipeline.base_agent import PipelineConfig
    cfg = PipelineConfig()  # loads config/default.yaml
    # Override results_root per-call (CLI --results flag)
    cfg.results_root = results_root
    cfg.data_root = Path(C.DATA_ROOT_DEFAULT)
    return cfg
