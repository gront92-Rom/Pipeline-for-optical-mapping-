"""Standalone v3.7 multi-trace run on 004A."""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(name)s [%(levelname)s] %(message)s')

from cardiac_pipeline.agents.peak_detector_agent import PeakDetectorAgent
from cardiac_pipeline.base_agent import PipelineConfig

cfg = PipelineConfig({
    "results_root": "results",
    "peak_detector": {
        # v3.6
        "threshold_frac":      0.5,
        "sigma_temporal":      3.0,
        "min_distance_factor": 0.6,
        "drop_first":          False,
        "min_peaks":           3,
        # v3.7 multi-trace
        "n_regions":           3,
        "min_region_pixels":   50,
        "min_agreement":       2,
        "frame_tolerance":     10,
        "soft_assignment_sigma": 20.0,
        "min_quality":         0.66,
        "n_beats_select":      3,
    },
})

agent = PeakDetectorAgent("004A", config=cfg)
result = agent.run(force=True)
print("=== RESULT ===")
print(result)