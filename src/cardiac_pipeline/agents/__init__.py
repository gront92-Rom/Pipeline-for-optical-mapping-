"""
Pipeline agents — each agent handles one stage of the cardiac optical mapping pipeline.

Stage map:
    Stage 1:  LoaderAgent          — raw data loading, metadata extraction
    Stage 2:  MaskAgent            — tissue mask extraction
    Stage 3:  PeakDetectorAgent    — preprocessing + beat detection
    Stage 4:  ActivationAgent      — activation time maps
    Stage 5:  ConductionAgent      — conduction velocity maps (CV)
    Stage 6:  APDAgent             — APD/CaT maps + per-beat 3D stack
    Stage 7:  AlternansAgent       — alternans detection (concordance, Poincaré, FFT)

Agents:
    mask_agent:          MaskAgent — tissue mask extraction (Stage 2)
    peak_detector_agent: PeakDetectorAgent — preprocessing + beat detection (Stage 3)
    activation_agent:    ActivationAgent — activation time maps (Stage 4)
    conduction_agent:    ConductionAgent — conduction velocity maps (Stage 5)
    apd_agent:           APDAgent — APD/CaT maps + per-beat 3D stack (Stage 6)
    alternans_agent:     AlternansAgent — alternans detection (Stage 7)
"""
