#!/usr/bin/env python3
"""
Smoke tests for LoaderAgent (Stage 1).
Runs without optimap or real MiCAM files — uses synthetic .npy input.
"""
import json
import sys
import tempfile
import shutil
import logging
from pathlib import Path

import numpy as np

# Make sure the package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cardiac_pipeline.base_agent import PipelineConfig
from cardiac_pipeline.agents.loader_agent import LoaderAgent

logging.basicConfig(level=logging.WARNING)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
        FAIL += 1


def make_synthetic_npy(tmp_dir: Path, shape=(200, 48, 48), fps=500.0) -> Path:
    """Create a synthetic .npy video file."""
    rng = np.random.default_rng(42)
    video = rng.random(shape).astype(np.float32)
    npy_path = tmp_dir / "test_video.npy"
    np.save(npy_path, video)
    return npy_path


def run_tests():
    tmp = Path(tempfile.mkdtemp(prefix="loader_smoke_"))
    try:
        # ----------------------------------------------------------------
        # Test 1: Basic full-mode run with .npy input
        # ----------------------------------------------------------------
        print("\n[Test 1] Full-mode run with synthetic .npy")
        results_dir = tmp / "results_t1"
        cfg = PipelineConfig({
            "results_root": str(results_dir),
            "loader": {
                "crop_left": 4,
                "crop_right": 4,
                "sideline_threshold": 4096,
                "default_fps": 500.0,
                "default_dye": "A",
            },
            "preprocess": {
                "spatial_sigma": 1.0,
                "lp_cutoff_activation_hz": 80.0,
                "lp_cutoff_apd_hz": 150.0,
                "chunk_size": 8192,
                "overlap": 64,
            },
        })
        npy_path = make_synthetic_npy(tmp, shape=(200, 48, 48))
        agent = LoaderAgent("smoke_001", config=cfg)
        result = agent.run(input_path=npy_path, force=True)

        check("status == success", result["status"] == "success")
        check("shape[0] == 200 frames", result["shape"][0] == 200)
        check("width cropped (48 - 4 - 4 = 40)", result["shape"][2] == 40,
              f"got {result['shape'][2]}")
        check("fps returned", result["fps"] == 500.0)

        must_dir = results_dir / "smoke_001" / "must"
        debug_dir = results_dir / "smoke_001" / "debug"

        check("raw_video.npy saved",       (must_dir / "raw_video.npy").exists())
        check("metadata.json saved",       (must_dir / "metadata.json").exists())
        check("preproc_video.npy saved",   (debug_dir / "preproc_video.npy").exists())
        check("preproc_video_apd.npy saved", (debug_dir / "preproc_video_apd.npy").exists())

        # Verify metadata content
        with open(must_dir / "metadata.json") as f:
            meta = json.load(f)
        check("metadata has fps",       meta.get("fps") == 500.0)
        check("metadata has sample_id", meta.get("sample_id") == "smoke_001")

        # Verify shapes of preprocessed videos
        act_video = np.load(debug_dir / "preproc_video.npy")
        apd_video = np.load(debug_dir / "preproc_video_apd.npy")
        check("preproc_video shape matches", act_video.shape == (200, 48, 40),
              f"got {act_video.shape}")
        check("preproc_video_apd shape matches", apd_video.shape == (200, 48, 40),
              f"got {apd_video.shape}")
        check("preproc_video dtype float32", act_video.dtype == np.float32)
        check("preproc_video_apd dtype float32", apd_video.dtype == np.float32)

        # ----------------------------------------------------------------
        # Test 2: Sideline mode (>= sideline_threshold frames)
        # ----------------------------------------------------------------
        print("\n[Test 2] Sideline-mode (long recording)")
        results_dir2 = tmp / "results_t2"
        cfg2 = PipelineConfig({
            "results_root": str(results_dir2),
            "loader": {
                "crop_left": 0,
                "crop_right": 0,
                "sideline_threshold": 100,   # Low threshold to trigger sideline
                "default_fps": 500.0,
                "default_dye": "A",
            },
            "preprocess": {
                "spatial_sigma": 1.0,
                "lp_cutoff_activation_hz": 80.0,
                "lp_cutoff_apd_hz": 150.0,
                "chunk_size": 8192,
                "overlap": 64,
            },
        })
        npy_path2 = make_synthetic_npy(tmp, shape=(200, 32, 32))  # 200 >= 100 → sideline
        agent2 = LoaderAgent("smoke_002", config=cfg2)
        result2 = agent2.run(input_path=npy_path2, force=True)

        check("sideline status", result2["status"] == "sideline")
        check("n_frames reported", result2["n_frames"] == 200)

        debug_dir2 = results_dir2 / "smoke_002" / "debug"
        check("sideline_trace.npy saved", (debug_dir2 / "sideline_trace.npy").exists())
        # preproc_video should NOT be saved in sideline mode
        check("preproc_video NOT saved in sideline", not (debug_dir2 / "preproc_video.npy").exists())

        trace = np.load(debug_dir2 / "sideline_trace.npy")
        check("trace length == n_frames", len(trace) == 200)

        # ----------------------------------------------------------------
        # Test 3: Skip if already processed (cache)
        # ----------------------------------------------------------------
        print("\n[Test 3] Cache skip (force=False)")
        result3 = agent.run(input_path=npy_path, force=False)
        check("status == skipped", result3["status"] == "skipped")

        # ----------------------------------------------------------------
        # Test 4: Missing file → FileNotFoundError
        # ----------------------------------------------------------------
        print("\n[Test 4] Missing input file → raises FileNotFoundError")
        results_dir4 = tmp / "results_t4"
        cfg4 = PipelineConfig({
            "results_root": str(results_dir4),
            "loader": {"default_fps": 500.0, "default_dye": "A"},
        })
        agent4 = LoaderAgent("smoke_004", config=cfg4)
        raised = False
        try:
            agent4.run(input_path="/nonexistent/file.npy", force=True)
        except FileNotFoundError:
            raised = True
        check("FileNotFoundError raised for missing file", raised)

        # ----------------------------------------------------------------
        # Test 5: crop_left + crop_right >= W → no crash, warning issued
        # ----------------------------------------------------------------
        print("\n[Test 5] Oversized crop → no crash")
        results_dir5 = tmp / "results_t5"
        cfg5 = PipelineConfig({
            "results_root": str(results_dir5),
            "loader": {
                "crop_left": 100,
                "crop_right": 100,
                "sideline_threshold": 4096,
                "default_fps": 500.0,
                "default_dye": "A",
            },
            "preprocess": {
                "spatial_sigma": 1.0,
                "lp_cutoff_activation_hz": 80.0,
                "lp_cutoff_apd_hz": 150.0,
                "chunk_size": 8192,
                "overlap": 64,
            },
        })
        (tmp / "t5").mkdir(parents=True, exist_ok=True)
        npy_path5 = make_synthetic_npy(tmp / "t5", shape=(50, 16, 16))
        agent5 = LoaderAgent("smoke_005", config=cfg5)
        result5 = agent5.run(input_path=npy_path5, force=True)
        check("no crash on oversized crop", result5["status"] == "success")
        check("width unchanged on oversized crop", result5["shape"][2] == 16)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Summary
    total = PASS + FAIL
    print(f"\n{'='*50}")
    print(f"LoaderAgent smoke tests: {PASS}/{total} PASSED")
    if FAIL:
        print(f"  {FAIL} FAILED")
        sys.exit(1)
    else:
        print("  All tests passed!")


if __name__ == "__main__":
    run_tests()
