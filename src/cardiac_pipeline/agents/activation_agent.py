"""
activation_agent.py — Stage 5: Activation Time (TAT) map (v3.9 spec).

Final logic (Roman 2026-07-10):
  - Global onsets from mean trace: rolling baseline + 50% threshold crossing
    (done by peak_detection.py → selected_peaks)
  - Per-pixel activation: same 50% threshold crossing per pixel
    within [onset, next_onset] window.
  - Light spatial Gaussian (σ=1px) on each frame.
  - Optional: parabolic interpolation for subframe precision.
  - No SavGol derivative per pixel. Single method.

  Outputs (must/):
    - activation_map.npy: (H, W) — TAT map in ms (median across beats)
    - per_beat_activation.npy: (n_beats, H, W) — per-beat TAT
    - activation_report.json: summary

  Outputs (debug/):
    - tat_per_region.npy: (n_beats, n_regions, H, W) — per-region TAT (debug)
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from cardiac_pipeline.utils.activation_detector import (
        detect_tat_map_local,
        combine_regions_soft,
        consensus_tat_methods,
    )
    ACTIVATION_DETECTOR_AVAILABLE = True
except ImportError:
    ACTIVATION_DETECTOR_AVAILABLE = False

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig


class ActivationAgent(BaseAgent):
    """Stage 5: Activation Time (TAT) map — 50% crossing per pixel (v3.9)."""

    DEPENDS_ON: list = []
    REQUIRED_INPUTS: list = [
        "preproc_video.npy",
        "mask.npy",
        "selected_peaks.npy",
        "peaks_per_region.npy",
        "weights.npy",
        "region_masks.npy",
    ]

    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None):
        super().__init__(sample_id, config)

        if not ACTIVATION_DETECTOR_AVAILABLE:
            raise ImportError(
                "activation_detector.py not found. "
                "Need: src/cardiac_pipeline/utils/activation_detector.py"
            )

        act_cfg = getattr(self.config, 'activation', {}) or {}
        self.min_amp = float(act_cfg.get('min_amp', 10.0))
        self.sigma_spatial = float(act_cfg.get('sigma_spatial', 1.0))
        self.falling_edge = bool(act_cfg.get('falling_edge', False))
        self.parabolic_interp = bool(act_cfg.get('parabolic_interp', False))
        self.hot_pixel_percentile = int(act_cfg.get('hot_pixel_percentile', 50))

        # Kept for API compatibility but unused (single method now)
        self.methods = ['threshold_50pct']
        self.agreement_threshold_ms = float(act_cfg.get('agreement_threshold_ms', 30.0))
        self.use_method_consensus = False

        self.metadata: Dict[str, Any] = {}

    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = self.get_path("metadata.json", kind="must")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        return self.metadata

    def _get_fps(self) -> float:
        fps = self.metadata.get("fps") or self.metadata.get("fps_hz")
        if fps is None:
            raise ValueError("fps not in metadata.json")
        return float(fps)

    def _get_stim_hz(self) -> float:
        stim = self.metadata.get("stim_hz") or self.metadata.get("pacing_hz")
        return float(stim) if stim else 10.0

    def _compute_hot_mask(self, preproc: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Top-N% std pixels within mask (matches APD v3.7 logic)."""
        T = preproc.shape[0]
        pixel_std = preproc.reshape(T, -1).std(axis=0).reshape(mask.shape)
        masked_std = pixel_std[mask]
        if len(masked_std) == 0:
            return np.zeros_like(mask, dtype=bool)
        thr = np.percentile(masked_std, self.hot_pixel_percentile)
        return mask & (pixel_std >= thr)

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """Main entry point."""
        if not force and self.exists("activation_map.npy"):
            self.logger.info("activation_map.npy exists, skipping")
            return {"status": "skipped"}

        t0 = time.perf_counter()

        # --- Lazy: ensure PeakDetector ran ---
        from cardiac_pipeline.agents.peak_detector_agent import PeakDetectorAgent
        self.DEPENDS_ON = [PeakDetectorAgent]
        self.ensure_dependencies(force=force)

        # --- Load inputs ---
        self._load_metadata()
        fps = self._get_fps()
        stim_hz = self._get_stim_hz()

        preproc = np.load(self.get_path("preproc_video.npy", kind="must"))
        mask = np.load(self.get_path("mask.npy", kind="must")).astype(bool)
        selected_peaks = np.load(self.get_path("selected_peaks.npy", kind="must"))
        selected_peaks = selected_peaks[selected_peaks >= 0]
        peaks_per_region = np.load(self.get_path("peaks_per_region.npy", kind="must"))
        weights = np.load(self.get_path("weights.npy", kind="must"))  # (H, W, n_regions)

        n_regions = weights.shape[2]
        n_beats = len(selected_peaks)

        self.logger.info(
            f"Loaded: preproc={preproc.shape}, mask_cov={mask.mean():.3f}, "
            f"n_regions={n_regions}, n_beats={n_beats}, fps={fps}"
        )

        if n_beats < 2:
            raise ValueError(
                f"Too few selected peaks ({n_beats}). Need ≥ 2 beats for activation map."
            )

        # --- Use full mask (no hot_mask subsetting) ---
        active_pixels = np.argwhere(mask)
        self.logger.info(
            f"Active pixels: {mask.sum()} (full mask, no hot_mask)"
        )

        # --- For each beat: per-region TAT → soft-weighted combine ---
        per_beat_tat = []
        per_beat_tat_per_region = []

        for beat_i in range(n_beats):
            peak_idx = int(selected_peaks[beat_i])
            per_region_tat = []
            for r in range(n_regions):
                peaks_r = peaks_per_region[r]
                peaks_r_valid = peaks_r[peaks_r >= 0]
                if len(peaks_r_valid) == 0:
                    per_region_tat.append(np.full(mask.shape, np.nan, dtype=np.float32))
                    continue
                if beat_i < len(peaks_r_valid):
                    region_peak = int(peaks_r_valid[beat_i])
                else:
                    region_peak = int(peaks_r_valid[-1])
                # Window end: next peak in this region
                if beat_i + 1 < len(peaks_r_valid):
                    region_peak_end = int(peaks_r_valid[beat_i + 1])
                else:
                    region_peak_end = region_peak + int(fps / max(stim_hz, 0.1))

                # 50% threshold crossing per pixel (v3.9)
                tat_map = detect_tat_map_local(
                    preproc, mask, region_peak, region_peak_end, fps,
                    method='threshold_50pct',
                    min_amp=self.min_amp,
                    sigma_spatial=self.sigma_spatial,
                    falling_edge=self.falling_edge,
                    parabolic_interp=self.parabolic_interp,
                )
                per_region_tat.append(tat_map)

            # Soft-weighted combine across regions
            tat_combined = combine_regions_soft(
                per_region_tat, weights, active_pixels
            )
            per_beat_tat.append(tat_combined)
            per_beat_tat_per_region.append(per_region_tat)

        # --- Median across beats ---
        per_beat_stack = np.stack(per_beat_tat, axis=0)  # (n_beats, H, W)
        # Align each beat to its own earliest = 0
        for i in range(n_beats):
            valid = per_beat_stack[i][np.isfinite(per_beat_stack[i])]
            if len(valid) > 0:
                per_beat_stack[i] = per_beat_stack[i] - np.nanmin(valid)
        # Median across beats
        activation_map = np.nanmedian(per_beat_stack, axis=0)
        # Shift so min = 0
        valid_mask_vals = activation_map[np.isfinite(activation_map) & mask]
        if len(valid_mask_vals) > 0:
            activation_map = activation_map - np.nanmin(valid_mask_vals)

        elapsed = time.perf_counter() - t0

        # --- Save artifacts ---
        self.save_must(activation_map, "activation_map.npy")
        if per_beat_tat:
            self.save_must(per_beat_stack, "per_beat_activation.npy")

        # Debug: per-region TAT
        try:
            tat_per_region_arr = np.stack(
                [np.stack(pr, axis=0) for pr in per_beat_tat_per_region],
                axis=0
            )
            self.save_debug(tat_per_region_arr, "tat_per_region.npy")
        except Exception as e:
            self.logger.warning(f"Could not save tat_per_region: {e}")

        # --- Report ---
        valid_tat = activation_map[np.isfinite(activation_map) & mask]
        tat_max = float(np.nanmax(valid_tat)) if len(valid_tat) > 0 else 0.0
        tat_std = float(np.nanstd(valid_tat)) if len(valid_tat) > 0 else 0.0
        valid_cov = float(len(valid_tat) / mask.sum()) if mask.sum() > 0 else 0.0

        report = {
            "sample_id": self.sample_id,
            "fps": fps,
            "stim_hz": stim_hz,
            "n_beats": n_beats,
            "n_regions": n_regions,
            "method": "threshold_50pct",
            "min_amp": self.min_amp,
            "sigma_spatial": self.sigma_spatial,
            "falling_edge": self.falling_edge,
            "parabolic_interp": self.parabolic_interp,
            "hot_pixel_percentile": self.hot_pixel_percentile,
            "n_active_pixels": int(mask.sum()),
            "tat_max_ms": round(tat_max, 2),
            "tat_std_ms": round(tat_std, 2),
            "valid_coverage": round(valid_cov, 4),
            "png": "activation_map.png",
            "elapsed_s": round(elapsed, 3),
        }
        self.save_must(report, "activation_report.json")

        self._save_png(activation_map, mask)

        self._log_metrics(report)
        self.logger.info(
            f"Done in {elapsed:.2f}s. TAT max={tat_max:.1f}ms, "
            f"valid={valid_cov:.1%}"
        )

        return {
            "status": "success",
            "activation_map_path": "results/{}/must/activation_map.npy".format(self.sample_id),
            "metrics": report,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _save_png(self, activation_map: np.ndarray, mask: np.ndarray) -> None:
        """Save activation map as PNG (jet colormap, masked outside tissue)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Mask non-tissue pixels → NaN for display
            display = activation_map.astype(float).copy()
            display[~mask] = np.nan

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(display, cmap="jet", interpolation="nearest")
            ax.set_title("Activation Time Map", fontsize=13, weight="bold")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("ms", fontsize=11)
            ax.axis("off")

            plt.tight_layout()
            path = self.must_dir / "activation_map.png"
            plt.savefig(path, dpi=150)
            plt.close()
            self.logger.info(f"[MUST] Saved: activation_map.png")
        except Exception as e:
            self.logger.warning(f"activation_map.png skipped: {e}")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ActivationAgent v3.9 — Stage 5")
    parser.add_argument("sample_id", help="Sample ID (e.g. 004A)")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig({"results_root": args.results_root})
    agent = ActivationAgent(args.sample_id, config=cfg)
    result = agent.run(force=args.force)
    print(json.dumps(result, indent=2, default=str))