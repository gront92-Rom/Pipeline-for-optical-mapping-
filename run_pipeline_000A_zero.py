#!/usr/bin/env python3
import sys, json, traceback
sys.path.insert(0, "src")

from cardiac_pipeline.agents.loader_agent import LoaderAgent
from cardiac_pipeline.agents.mask_agent import MaskAgent
from cardiac_pipeline.agents.peak_detector_agent import PeakDetectorAgent
from cardiac_pipeline.agents.activation_agent import ActivationAgent
from cardiac_pipeline.agents.apd_agent import APDAgent
from cardiac_pipeline.agents.conduction_agent import ConductionAgent
from cardiac_pipeline.base_agent import PipelineConfig

sample_id = "000A_zero"
input_file = "/tmp/sham_000A/2026-05-08-mSHAM-bsl-4Hz-0508-000A.rsh"
results_root = "/home/rymedv/.openclaw/workspace-lab/cardiac_pipeline_v3/results"

cfg = PipelineConfig({"results_root": results_root, "peak_detector": {"n_regions": 3}})

stages = [
    ("Loader", LoaderAgent(sample_id, config=cfg), {"input_path": input_file, "force": True}),
    ("Mask", MaskAgent(sample_id, config=cfg), {"force": True}),
    ("PeakDetector", PeakDetectorAgent(sample_id, config=cfg), {"force": True}),
    ("Activation", ActivationAgent(sample_id, config=cfg), {"force": True}),
    ("APD", APDAgent(sample_id, config=cfg), {"force": True}),
    ("Conduction", ConductionAgent(sample_id, config=cfg), {"force": True}),
]

results = {}
for name, agent, kwargs in stages:
    print(f"\n=== {name} ===", flush=True)
    try:
        result = agent.run(**kwargs)
        print(json.dumps(result, indent=2, default=str), flush=True)
        results[name] = result
    except Exception as e:
        print(f"STAGE {name} FAILED: {e}", flush=True)
        traceback.print_exc()
        results[name] = {"error": str(e)}
        break

print("\n=== PIPELINE COMPLETE ===", flush=True)
print(json.dumps(results, indent=2, default=str), flush=True)

# Also dump metrics.json if present
import os
metrics_path = os.path.join(results_root, sample_id, "must", "metrics.json")
print(f"\n--- metrics.json ({metrics_path}) ---", flush=True)
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        print(f.read(), flush=True)
else:
    print("no metrics.json found", flush=True)

# List output dirs
for kind in ("must", "debug"):
    d = os.path.join(results_root, sample_id, kind)
    print(f"\n--- ls {d} ---", flush=True)
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            print(fn, flush=True)
    else:
        print("(missing)", flush=True)