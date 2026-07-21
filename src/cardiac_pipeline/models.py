"""
cardiac_pipeline.models — plain dataclasses for pipeline results.

Kept separate from driver.py so that other modules (tests, CLI, future
batch runner) can import them without pulling in driver logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Status enum-like strings (kept as plain strings for JSON serialisation simplicity).
STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_REJECT = "REJECT"
STATUS_ERROR = "ERROR"
STATUS_SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class StageResult:
    """Result of running (or skipping) one stage.

    Immutable: written once when the stage finishes and never mutated.
    Serialises cleanly to JSON for manifest files.
    """

    name: str
    status: str                                  # one of STATUS_* above
    elapsed_s: float = 0.0
    error: str = ""
    manifest_path: Optional[Path] = None         # set when manifest was written

    @classmethod
    def skipped(cls, name: str, manifest_path: Path) -> "StageResult":
        return cls(name=name, status=STATUS_SKIPPED, manifest_path=manifest_path)

    @classmethod
    def succeed(cls, name: str, elapsed_s: float, manifest_path: Path) -> "StageResult":
        return cls(name=name, status=STATUS_OK, elapsed_s=elapsed_s, manifest_path=manifest_path)

    @classmethod
    def failed(cls, name: str, elapsed_s: float, exc: Exception) -> "StageResult":
        return cls(name=name, status=STATUS_ERROR, elapsed_s=elapsed_s, error=str(exc))


@dataclass
class PipelineResult:
    """Aggregated result of running the pipeline for one sample.

    Aggregates per-stage results plus run-level metadata. Mutable on purpose
    because the orchestrator populates it incrementally as stages run.
    """

    sample_id: str
    input_file: str
    started_at: str
    finished_at: str = ""
    total_elapsed_s: float = 0.0
    ok: bool = False
    failed_stage: str = ""
    stages: dict[str, StageResult] = field(default_factory=dict)

    def record(self, stage_result: StageResult) -> None:
        self.stages[stage_result.name] = stage_result
        if stage_result.status == STATUS_ERROR:
            self.failed_stage = stage_result.name
            self.ok = False

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def n_ok(self) -> int:
        return sum(1 for s in self.stages.values() if s.status == STATUS_OK)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "input_file": self.input_file,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_s": round(self.total_elapsed_s, 2),
            "ok": self.ok,
            "failed_stage": self.failed_stage,
            "stages": {
                k: {
                    "status": v.status,
                    "elapsed_s": round(v.elapsed_s, 2),
                    "error": v.error,
                    "manifest": str(v.manifest_path) if v.manifest_path else None,
                }
                for k, v in self.stages.items()
            },
        }
