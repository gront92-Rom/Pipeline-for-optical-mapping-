"""
cardiac_pipeline.stages — declarative registry of pipeline stages.

Single source of truth for what stages exist, in what order, and what each
stage does. Imported by driver.run_pipeline(); also powers `list-agents`.

Why a separate module:
  - The stage list is the most important knowledge in the system.
  - Keeping it separate makes driver.py a thin orchestrator, not a registry.
  - Stage description powers `cardiac-pipeline status` without extra logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from cardiac_pipeline.agents.activation_agent import ActivationAgent
from cardiac_pipeline.agents.alternans_agent import AlternansAgent
from cardiac_pipeline.agents.apd_agent import APDAgent
from cardiac_pipeline.agents.cleaning_agent import CleaningAgent
from cardiac_pipeline.agents.conduction_agent import ConductionAgent
from cardiac_pipeline.agents.loader_agent import LoaderAgent
from cardiac_pipeline.agents.mask_agent import MaskAgent
from cardiac_pipeline.agents.peak_detector_agent import PeakDetectorAgent
from cardiac_pipeline.agents.report_agent import ReportAgent


@dataclass(frozen=True)
class Stage:
    """One pipeline stage.

    Attributes:
        agent_cls: The agent class to instantiate and run.
        description: Short human-readable description (used in status/help).
        requires_input: True only for loader — needs the input file path.
    """

    name: str
    agent_cls: Type
    description: str = ""
    requires_input: bool = False


# The pipeline. Order matters: downstream agents depend on upstream outputs.
# Note: agents internally call ensure_dependencies() — order is "expected",
# not "strictly required" — but keep it topologically meaningful for UX.
STAGES: list[Stage] = [
    Stage("loader",         LoaderAgent,        "Load raw video + preprocessing",           requires_input=True),
    Stage("mask",           MaskAgent,          "Tissue segmentation"),
    Stage("peak_detector",  PeakDetectorAgent,  "Beat detection (multi-trace v3.7)"),
    Stage("activation",     ActivationAgent,    "Activation time map (TAT)"),
    Stage("apd",            APDAgent,           "APD30/50/80 maps + per-beat 3D stack"),
    Stage("conduction",     ConductionAgent,    "Conduction velocity CVL/CVT"),
    Stage("alternans",      AlternansAgent,     "Alternans detection (concordance / AC95 / FFT)"),
    Stage("cleaning",       CleaningAgent,      "Drop intermediate .npy artefacts"),
    Stage("report",         ReportAgent,        "Summary report (Markdown)"),
]


def get_stage(name: str) -> Stage:
    """Look up a stage by its short name. Raises KeyError if unknown."""
    for s in STAGES:
        if s.name == name:
            return s
    raise KeyError(f"Unknown stage: {name!r}. Known: {[s.name for s in STAGES]}")
