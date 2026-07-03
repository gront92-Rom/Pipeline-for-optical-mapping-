"""
Pipeline agents — each agent handles one stage of the cardiac optical mapping pipeline.

Agents:
    mask_agent:          MaskAgent — tissue mask extraction (Stage 2)
    peak_detector_agent: PeakDetectorAgent — preprocessing + beat detection (Stage 3)
    activation_agent:    ActivationAgent — activation time maps (Stage 4)
    apd_agent:           APDAgent — APD/CaT maps + per-beat 3D stack (Stage 4)
    alternans_agent:     AlternansAgent — alternans detection (Stage 5)
    conduction_agent:    ConductionAgent — conduction velocity maps (Stage CV)
"""
