"""
Pipeline agents — each agent handles one stage of the cardiac optical mapping pipeline.

Agents:
    mask_agent:          MaskAgent — tissue mask extraction (Stage 2)
    peak_detector_agent: PeakDetectorAgent — preprocessing + beat detection (Stage 3)
    activation_agent:    ActivationAgent — activation time maps (Stage 4)
    conduction_agent:    ConductionAgent — conduction velocity maps (Stage 5)
"""
