"""
Utility modules for the cardiac optical mapping pipeline.

Modules:
    metadata_extractor: Extract metadata from .bvx/.rsh/.gsh files
    preprocess: Video preprocessing (spatial smooth, temporal lowpass, inversion, ASLS)
    cv_estimators: CV calculation kernels (hybrid structure tensor + polynomial Bayly)
    signal: APD/CaT math kernels (ROI pooling, upstroke, repolarization, corner ROI, QC)
"""
