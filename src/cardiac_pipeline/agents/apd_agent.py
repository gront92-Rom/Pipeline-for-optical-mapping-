"""
apd_agent.py — Stage 4: Action Potential Duration (APD) map.

v3.8 (2026-07-10, rewrite):
  - Uses peaks.npy (ALL consensus peaks) — NOT selected_peaks (top-N subset)
  - Single-region mode: no peaks_per_region, no weights, no soft assignment
  - Beat window = peak[i] .. peak[i+1] (last beat extended by BCL)
  - Hot mask by percentile (default 50 = top 50% std pixels)
  - Dynamic min_amp: max(abs_floor, noise_mult * sigma_noise)
  - Outlier rejection: APD > max(BCL*0.9, 300ms) or APD < 10ms
  - Single-pass detection for all levels (30/50/80)

Outputs (per run):

  must/:
    - apd30_map.npy     (H, W) — median APD30 over all beats
    - apd50_map.npy     (H, W)
    - apd80_map.npy     (H, W)
    - apd_report.json   — summary statistics
    - apd_per_beat_3d.npz — (H, W, N_beats) APD80 stack for AlternansAgent

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
    """Stage 4: Per-pixel APD30/50/80 detection using ALL consensus peaks."""

    DEPENDS_ON = []  # set by pipeline runner or lazy-dep
    REQUIRED_INPUTS = [
        "preproc_video.npy",
        "mask.npy",
        "peaks.npy",
    ]

    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None):
        super().__init__(sample_id, config)

        apd_cfg = getattr(self.config, "apd", None)
        if not apd_cfg:
            apd_cfg = {}

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

        self.levels = sorted(self.levels)

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """Run Stage 4: APD map computation over ALL consensus peaks."""
        t0 = time.time()

        # === Lazy dependencies ===
        self.ensure_dependencies(force=force)

        # === 1. Load inputs ===
        self.logger.info(f"[APD] Loading inputs from {self.must_dir}")
        preproc_video = np.load(self.must_dir / "preproc_video.npy")  # (T, H, W)
        mask = np.load(self.must_dir / "mask.npy").astype(bool)       # (H, W)

        with open(self.must_dir / "metadata.json") as f:
            metadata = json.load(f)
        fps = float(metadata["fps"])
        stim_hz = float(metadata.get("stim_hz", 5.86))

        # ALL consensus peaks (NOT selected_peaks top-N subset)
        peaks = np.load(self.must_dir / "peaks.npy")
        peaks = peaks[peaks >= 0]  # strip any padding
        n_beats = len(peaks)

        T, H, W = preproc_video.shape

        self.logger.info(
            f"[APD] preproc=({T},{H},{W}), mask_coverage={mask.mean():.3f}, "
            f"fps={fps}, stim_hz={stim_hz}, n_beats={n_beats}"
        )

        if n_beats < 2:
            raise ValueError(
                f"[APD] Need >= 2 peaks for APD, got {n_beats}. "
                f"Sample {self.sample_id} needs manual review."
            )

        # === 2. Compute beat windows (peak[i], peak[i+1]) ===
        # Last beat: extend by BCL frames (no next peak)
        if n_beats > 1:
            dt_frames = np.diff(peaks)
            bcl_frames = float(np.median(dt_frames))
        else:
            bcl_frames = fps / stim_hz if stim_hz > 0 else 167.0

        bcl_ms = bcl_frames / fps * 1000.0

        beat_starts = peaks[:-1].copy()
        beat_ends = peaks[1:].copy()
        # Append last beat window: last peak + BCL frames
        last_end = int(peaks[-1] + bcl_frames)
        if last_end > T:
            last_end = T
        beat_starts = np.append(beat_starts, peaks[-1])
        beat_ends = np.append(beat_ends, last_end)

        n_windows = len(beat_starts)
        self.logger.info(
            f"[APD] BCL={bcl_ms:.1f}ms, {n_windows} beat windows "
            f"(outlier threshold={max(bcl_ms*0.9, 300.0):.1f}ms)"
        )

        # === 3. Hot mask ===
        hot_mask = compute_hot_mask(preproc_video, mask, self.hot_pixel_percentile)
        active_coords = np.argwhere(hot_mask)  # (N_active, 2)
        n_active = len(active_coords)
        self.logger.info(
            f"[APD] Hot mask: {n_active} active pixels "
            f"(percentile={self.hot_pixel_percentile}, "
            f"{n_active/mask.sum()*100:.1f}% of masked)"
        )

        if n_active == 0:
            raise ValueError("[APD] No active pixels — hot mask is empty")

        # === 4. Noise estimation ===
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

        # === 5. Main loop: per-pixel, per-beat APD ===
        # 4D: (n_levels, H, W, n_windows)
        apd_4d = np.full((len(self.levels), H, W, n_windows), np.nan, dtype=np.float32)

        for pi, (h, w) in enumerate(active_coords):
            if pi % 500 == 0 and pi > 0:
                self.logger.debug(f"[APD]   processed {pi}/{n_active} pixels")

            for bi in range(n_windows):
                start_f = int(beat_starts[bi])
                end_f = int(beat_ends[bi])

                if start_f < 0 or end_f <= start_f:
                    continue

                apd_values = detect_all_apd_levels_pixel(
                    preproc_video, int(h), int(w),
                    start_f, end_f, fps,
                    levels=self.levels, min_amp=min_amp,
                )

                # Outlier rejection
                apd_filtered = reject_apd_outliers(apd_values, bcl_ms)

                for lv_i, lv in enumerate(self.levels):
                    if lv in apd_filtered:
                        apd_4d[lv_i, h, w, bi] = apd_filtered[lv]

        # === 6. Median over beats → spatial maps ===
        apd_maps = {}
        for lv_i, lv in enumerate(self.levels):
            apd_maps[lv] = np.nanmedian(apd_4d[lv_i], axis=2)

        # === 7. Save artifacts ===
        for lv, apd_map in apd_maps.items():
            self.save_must(apd_map.astype(np.float32), f"apd{lv}_map.npy")

        self.save_debug(apd_4d, "apd_4d.npy")
        self.save_debug(hot_mask, "hot_mask.npy")
        self.save_debug(np.array([min_amp], dtype=np.float32), "min_amp.npy")

        # apd_per_beat_3d.npz for AlternansAgent
        dye = metadata.get("dye") or metadata.get("recording_mode") or "A"
        d = str(dye).upper().strip()
        metric = "CaT" if d in ("B", "CALCIUM", "CAT", "CA") else "APD"
        apd80_idx = self.levels.index(80) if 80 in self.levels else -1
        if apd80_idx >= 0:
            apd80_3d = apd_4d[apd80_idx]  # (H, W, n_windows)
            np.savez_compressed(
                self.must_dir / "apd_per_beat_3d.npz",
                apd80=apd80_3d.astype(np.float32),
                metric=metric,
                n_beats=n_windows,
            )
            self.logger.info(
                f"[MUST] Saved: apd_per_beat_3d.npz "
                f"(apd80 {apd80_3d.shape}, metric={metric})"
            )

        # === 8. Report ===
        report = {
            "sample_id": self.sample_id,
            "fps": fps,
            "stim_hz": stim_hz,
            "bcl_ms": bcl_ms,
            "n_levels": len(self.levels),
            "levels": self.levels,
            "n_beats": n_beats,
            "n_windows": n_windows,
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
            "n_beats": n_beats,
            "n_windows": n_windows,
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

    parser = argparse.ArgumentParser(description="APDAgent v3.8 — all peaks")
    parser.add_argument("sample_id", help="Sample ID (e.g. 005A)")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--hot-pixel-percentile", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="Force rerun")
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
    result = agent.run(force=args.force)
    print(result)