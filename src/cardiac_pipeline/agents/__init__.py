"""
Pipeline agents — each agent handles one stage of the cardiac optical mapping pipeline.

Stage map:
    Stage 0:  LoaderAgent          — raw data loading, metadata extraction
    Stage 0b: SidelineAgent        — long-file interception (>= 4096 frames)
    Stage 1:  MaskAgent            — tissue mask extraction
    Stage 2:  PeakDetectorAgent    — preprocessing + beat detection
    Stage 3:  ActivationAgent      — activation time maps
    Stage 4:  ConductionAgent      — conduction velocity maps (CV)
    Stage 5:  APDAgent             — APD/CaT maps + per-beat 3D stack
    Stage 6:  AlternansAgent       — alternans detection (concordance, Poincaré, FFT)

Agents:
    loader_agent:        LoaderAgent — data loading, metadata extraction, preprocessing (Stage 0)
    sideline_agent:      SidelineAgent — long-file interception, trace extraction + guide (Stage 0b)
    mask_agent:          MaskAgent — tissue mask extraction (Stage 1)
    peak_detector_agent: PeakDetectorAgent — preprocessing + beat detection (Stage 2)
    activation_agent:    ActivationAgent — activation time maps (Stage 3)
    conduction_agent:    ConductionAgent — conduction velocity maps (Stage 4)
    apd_agent:           APDAgent — APD/CaT maps + per-beat 3D stack (Stage 5)
    alternans_agent:     AlternansAgent — alternans detection (Stage 6)

SidelineAgent contract:
    run() returns {"status": "pass", ...} for short files (< 4096 frames).
    run() returns {"status": "sideline_isolated", ...} for long files.
    The orchestrator (optical_pipeline_worker) MUST check the returned status
    and abort the main pipeline stages when status == "sideline_isolated".
"""

# SidelineAgent v2 — functional module (no class export)
# from cardiac_pipeline.agents.sideline_agent import SidelineAgent  # v1, removed

__all__ = []
