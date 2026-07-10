"""Standalone v3.7 APDAgent run on 004A."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cardiac_pipeline.agents.apd_agent import APDAgent
from cardiac_pipeline.base_agent import PipelineConfig

cfg = PipelineConfig({
    "results_root": "results",
    "apd": {
        "levels": [30, 50, 80],
        "hot_pixel_percentile": 50,
        "min_amp_abs": 100.0,
        "min_amp_noise_mult": 3.0,
    },
})

agent = APDAgent("004A", config=cfg)
result = agent.run(force=True)
print("=== RESULT ===")
print(result)