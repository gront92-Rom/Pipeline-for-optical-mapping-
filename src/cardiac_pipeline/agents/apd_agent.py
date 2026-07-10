"""
apd_agent.py — Stage 4: Action Potential Duration (APD) map.

v3.7 spec (2026-07-09, developer review):

  - Uses PeakDet v3.7 outputs: selected_peaks, peaks_per_region, weights
  - Soft-weighted per-region APD (weighted average, NOT median)
  - Hot mask by percentile (default 50 = top 50% std pixels)
  - Active pixel loop: iterate only over hot_mask pixels
  - Dynamic min_amp: max(100, 3 * sigma_noise)
  - Outlier rejection: APD > max(BCL*0.9, 300ms) or APD < 10ms
  - Single-pass detection for all levels (30/50/80)

Outputs (per run):

  must/:
    - apd30_map.npy     (H, W) — median APD30 over selected beats
    - apd50_map.npy     (H, W)
    - apd80_map.npy     (H, W)
    - apd_report.json   — summary statistics

  debug/:
    - apd_4d.npy        (n_levels, H, W, n_beats) — raw per-beat APD
    - hot_mask.npy      (H, W) bool — pixels included in analysis
    - min_amp.npy       float — dynamic threshold used
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig
from cardiac_pipeline.utils.apd_detector import (
    DEFAULT_LEVELS,
    DEFAULT_HOT_PIXEL_PERCENTILE,
    DEFAULT_MIN_AMP_ABS,
    DEFAULT_MIN_AMP_NOISE_MULT,
    compute_hot_mask,
    compute_per_pixel_min_amp,
    detect_all_apd_levels_pixel,
    estimate_noise_sigma,
    reject_apd_outliers,
)


class APDAgent(BaseAgent):
    """Stage 4: Per-pixel APD30/50/80 detection."""

    DEPENDS_ON = []  # explicitly set in pipeline runner
    REQUIRED_INPUTS = [
        "preproc_video.npy",
        "mask.npy",
        "peaks.npy",
        "selected_peaks.npy",
        "peaks_per_region.npy",
        "weights.npy",
    ]

    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None):
        super().__init__(sample_id, config)

        # Read APD config
        apd_cfg = getattr(self.config, "apd", None)
        if not apd_cfg:
            apd_cfg = {}
        # Back-compat: nested access for dict
        if hasattr(apd_cfg, "levels"):
            self.levels = list(apd_cfg.levels)
        else:
            self.levels = apd_cfg.get("levels", DEFAULT_LEVELS)

        if hasattr(apd_cfg, "hot_pixel_percentile"):
            self.hot_pixel_percentile = int(apd_cfg.hot_pixel_percentile)
        else:
            self.hot_pixel_percentile = int(apd_cfg.get(
                "hot_pixel_percentile", DEFAULT_HOT_PIXEL_PERCENTILE))

        if hasattr(apd_cfg, "min_amp_abs"):
            self.min_amp_abs = float(apd_cfg.min_amp_abs)
        else:
            self.min_amp_abs = float(apd_cfg.get(
                "min_amp_abs", DEFAULT_MIN_AMP_ABS))

        if hasattr(apd_cfg, "min_amp_noise_mult"):
            self.min_amp_noise_mult = float(apd_cfg.min_amp_noise_mult)
        else:
            self.min_amp_noise_mult = float(apd_cfg.get(
                "min_amp_noise_mult", DEFAULT_MIN_AMP_NOISE_MULT))

        # Detected levels must be sorted (for output naming consistency)
        self.levels = sorted(self.levels)

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """Run Stage 4: APD map computation."""
        t0 = time.time()

        # === Lazy dependencies ===
        self.ensure_dependencies(force=force)

        # === 1. Load required inputs ===
        self.logger.info(f"[APD] Loading inputs from {self.must_dir}")
        preproc_video = np.load(self.must_dir / "preproc_video.npy")  # (T, H, W)
        mask = np.load(self.must_dir / "mask.npy").astype(bool)  # (H, W)

        with open(self.must_dir / "metadata.json") as f:
            metadata = json.load(f)
        fps = float(metadata["fps"])
        stim_hz = float(metadata.get("stim_hz", 5.86))

        # PeakDet v3.7 outputs
        selected_peaks = np.load(self.must_dir / "selected_peaks.npy")  # (n_beats,)
        selected_peaks = selected_peaks[selected_peaks >= 0]  # strip padding
        peaks_per_region = np.load(self.must_dir / "peaks_per_region.npy")  # (n_regions, max_beats)
        weights = np.load(self.must_dir / "weights.npy")  # (H, W, n_regions)

        T, H, W = preproc_video.shape
        n_regions = peaks_per_region.shape[0]
        n_beats = len(selected_peaks)

        self.logger.info(
            f"[APD] preproc=({T},{H},{W}), mask_coverage={mask.mean():.3f}, "
            f"fps={fps}, stim_hz={stim_hz}, n_beats={n_beats}, n_regions={n_regions}"
        )

        # === 2. Compute hot mask ===
        hot_mask = compute_hot_mask(preproc_video, mask, self.hot_pixel_percentile)
        active_coords = np.argwhere(hot_mask)  # (N_active, 2) — [[y, x], ...]
        n_active = len(active_coords)
        self.logger.info(
            f"[APD] Hot mask: {n_active} active pixels "
            f"(percentile={self.hot_pixel_percentile}, "
            f"{n_active/mask.sum()*100:.1f}% of masked)"
        )

        if n_active == 0:
            raise ValueError("[APD] No active pixels — hot mask is empty")

        # === 3. Estimate noise sigma ===
        sigma_noise = estimate_noise_sigma(preproc_video, mask, fps)
        min_amp = compute_per_pixel_min_amp(
            preproc_video, mask, sigma_noise,
            abs_floor=self.min_amp_abs,
            noise_multiplier=self.min_amp_noise_mult,
        )
        self.logger.info(
            f"[APD] sigma_noise={sigma_noise:.2f}, min_amp={min_amp:.2f} "
            f"(abs_floor={self.min_amp_abs}, noise_mult={self.min_amp_noise_mult})"
        )

        # === 4. Compute BCL ===
        if len(selected_peaks) > 1:
            dt_ms = np.diff(selected_peaks) / fps * 1000.0
            bcl_ms = float(np.median(dt_ms))
        else:
            bcl_ms = 1000.0 / stim_hz if stim_hz > 0 else 167.0
        self.logger.info(f"[APD] BCL={bcl_ms:.1f}ms (outlier threshold={max(bcl_ms*0.9, 300.0):.1f}ms)")

        # === 5. Main loop: iterate active pixels ===
        # 4D array: (n_levels, H, W, n_beats)
        apd_4d = np.full((len(self.levels), H, W, n_beats), np.nan, dtype=np.float32)

        for pi, (h, w) in enumerate(active_coords):
            if pi % 500 == 0 and pi > 0:
                self.logger.debug(f"[APD]   processed {pi}/{n_active} pixels")

            for beat_i, peak_idx in enumerate(selected_peaks):
                # Collect per-region APD for this pixel/beat
                apd_per_region_by_level: Dict[int, List[float]] = {lv: [] for lv in self.levels}
                weight_per_region: List[float] = []

                for r in range(n_regions):
                    peaks_r = peaks_per_region[r]
                    if beat_i + 1 >= len(peaks_r):
                        continue

                    peak_start_r = int(peaks_r[beat_i])
                    peak_end_r = int(peaks_r[beat_i + 1])
                    if peak_start_r < 0 or peak_end_r < 0:
                        continue

                    apd_values = detect_all_apd_levels_pixel(
                        preproc_video, int(h), int(w),
                        peak_start_r, peak_end_r, fps,
                        levels=self.levels, min_amp=min_amp,
                    )

                    # Filter outliers using BCL rule
                    apd_filtered = reject_apd_outliers(apd_values, bcl_ms)

                    # Accumulate per level
                    for lv in self.levels:
                        if lv in apd_filtered:
                            apd_per_region_by_level[lv].append(apd_filtered[lv])
                    weight_per_region.append(float(weights[h, w, r]))

                # === Weighted average per level (FIX from developer review) ===
                if weight_per_region:
                    w_arr = np.array(weight_per_region)
                    if w_arr.sum() > 0:
                        w_norm = w_arr / w_arr.sum()
                    else:
                        continue
                    for lv_i, lv in enumerate(self.levels):
                        vals = apd_per_region_by_level[lv]
                        if len(vals) == 0:
                            continue
                        if len(vals) == 1:
                            apd_4d[lv_i, h, w, beat_i] = vals[0]
                        else:
                            # np.average with normalized weights
                            vals_arr = np.array(vals[:len(w_norm)])
                            apd_4d[lv_i, h, w, beat_i] = float(
                                np.average(vals_arr, weights=w_norm[:len(vals_arr)])
                            )

        # === 6. Median over beats → spatial APD maps ===
        apd_maps = {}
        for lv_i, lv in enumerate(self.levels):
            apd_maps[lv] = np.nanmedian(apd_4d[lv_i], axis=2)

        # === 7. Save artifacts ===
        for lv, apd_map in apd_maps.items():
            self.save_must(apd_map.astype(np.float32), f"apd{lv}_map.npy")

        self.save_debug(apd_4d, "apd_4d.npy")
        self.save_debug(hot_mask, "hot_mask.npy")
        self.save_debug(np.array([min_amp], dtype=np.float32), "min_amp.npy")

        # Compute summary statistics
        report = {
            "sample_id": self.sample_id,
            "fps": fps,
            "stim_hz": stim_hz,
            "bcl_ms": bcl_ms,
            "n_levels": len(self.levels),
            "levels": self.levels,
            "n_beats": n_beats,
            "n_regions": n_regions,
            "n_active_pixels": n_active,
            "hot_pixel_percentile": self.hot_pixel_percentile,
            "sigma_noise": sigma_noise,
            "min_amp": min_amp,
            "min_amp_abs_floor": self.min_amp_abs,
            "min_amp_noise_mult": self.min_amp_noise_mult,
            "apd_outlier_threshold_ms": max(bcl_ms * 0.9, 300.0),
            "apd_min_threshold_ms": 10.0,
            "elapsed_s": time.time() - t0,
        }

        for lv, apd_map in apd_maps.items():
            valid = apd_map[hot_mask & np.isfinite(apd_map)]
            if len(valid) > 0:
                report[f"apd{lv}_median_ms"] = float(np.median(valid))
                report[f"apd{lv}_iqr_ms"] = [
                    float(np.percentile(valid, 25)),
                    float(np.percentile(valid, 75)),
                ]
                report[f"apd{lv}_n_valid"] = int(len(valid))
                report[f"apd{lv}_n_detected"] = int((hot_mask & np.isfinite(apd_map)).sum())
            else:
                report[f"apd{lv}_median_ms"] = None
                report[f"apd{lv}_iqr_ms"] = None
                report[f"apd{lv}_n_valid"] = 0
                report[f"apd{lv}_n_detected"] = 0

        self.save_must(report, "apd_report.json")

        self.logger.info(
            f"[APD] Done in {time.time()-t0:.2f}s. "
            f"APD80 median={report.get('apd80_median_ms', 'NA')}ms, "
            f"valid={report.get('apd80_n_valid', 0)}/{n_active}"
        )

        return {
            "status": "success",
            "n_active_pixels": n_active,
            "apd_maps_paths": {str(lv): str(self.must_dir / f"apd{lv}_map.npy")
                               for lv in self.levels},
            "metrics": report,
        }


# === CLI entry point ===
if __name__ == "__main__":
    import argparse
    try:
        from omegaconf import OmegaConf  # noqa: F401
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="APDAgent standalone (v3.7)")
    parser.add_argument("sample_id", help="Sample ID (e.g. 005A)")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--hot-pixel-percentile", type=int, default=50)
    args = parser.parse_args()

    cfg = PipelineConfig({
        "results_root": args.results_root,
        "apd": {
            "levels": [30, 50, 80],
            "hot_pixel_percentile": args.hot_pixel_percentile,
            "min_amp_abs": 100.0,
            "min_amp_noise_mult": 3.0,
        },
    })
    agent = APDAgent(args.sample_id, config=cfg)
    result = agent.run()
    print(result)