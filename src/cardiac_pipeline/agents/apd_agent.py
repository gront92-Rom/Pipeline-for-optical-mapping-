"""
apd_agent.py — Stage 4: Action Potential Duration (APD) map.

v3.8 (2026-07-10, rewrite):
  - Uses peaks.npy (ALL consensus peaks) — NOT selected_peaks (top-N subset)
  - Single-region mode: no peaks_per_region, no weights, no soft assignment
  - Beat window = peak[i] .. peak[i+1] (last beat extended by BCL)
  - Full tissue mask (mask.npy) — NO hot-mask filtering
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
    - mask.npy          (H, W) bool — full tissue mask used in analysis
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
    DEFAULT_MIN_AMP_ABS,
    DEFAULT_MIN_AMP_NOISE_MULT,
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

        # === 3. Full tissue mask (no hot-mask filtering) ===
        active_coords = np.argwhere(mask)  # (N_active, 2)
        n_active = len(active_coords)
        self.logger.info(
            f"[APD] Full mask: {n_active} active pixels "
            f"({n_active/mask.size*100:.1f}% of frame)"
        )

        if n_active == 0:
            raise ValueError("[APD] No active pixels — mask is empty")

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
        self.save_debug(mask, "mask.npy")  # full mask used (not hot_mask)
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
            "sigma_noise": sigma_noise,
            "min_amp": min_amp,
            "min_amp_abs_floor": self.min_amp_abs,
            "min_amp_noise_mult": self.min_amp_noise_mult,
            "apd_outlier_threshold_ms": max(bcl_ms * 0.9, 300.0),
            "apd_min_threshold_ms": 10.0,
            "elapsed_s": time.time() - t0,
        }

        for lv, apd_map in apd_maps.items():
            valid = apd_map[mask & np.isfinite(apd_map)]
            if len(valid) > 0:
                report[f"apd{lv}_median_ms"] = float(np.median(valid))
                report[f"apd{lv}_iqr_ms"] = [
                    float(np.percentile(valid, 25)),
                    float(np.percentile(valid, 75)),
                ]
                report[f"apd{lv}_n_valid"] = int(len(valid))
                report[f"apd{lv}_n_detected"] = int((mask & np.isfinite(apd_map)).sum())
            else:
                report[f"apd{lv}_median_ms"] = None
                report[f"apd{lv}_iqr_ms"] = None
                report[f"apd{lv}_n_valid"] = 0
                report[f"apd{lv}_n_detected"] = 0

        self.save_must(report, "apd_report.json")

        # PNG visualization (must/apd_maps.png)
        self._save_png(apd_maps, mask)

        # === 9. 4-point APD extraction (traces + per-point values) ===
        point_results = self._extract_4point_apd(
            preproc_video, mask, peaks, beat_starts, beat_ends,
            fps, bcl_ms, n_windows,
        )
        report["points_4"] = point_results
        # Re-save report with 4-point data
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

    # ------------------------------------------------------------------
    # 4-point APD extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_tissue_points(mean_frame: np.ndarray, mask: np.ndarray,
                            n_points: int = 4,
                            edge_distance: int = 10) -> list:
        """Pick N points ~edge_distance px from the mask boundary.

        Strategy: erode mask by edge_distance → border band = mask AND NOT eroded.
        Divide border band into a 2×2 spatial grid (quadrants), pick brightest
        pixel in each quadrant. Falls back to full-mask grid if border band
        is too small.
        """
        from scipy.ndimage import binary_erosion
        from skimage.morphology import disk

        eroded = binary_erosion(mask, disk(edge_distance))
        border = mask & ~eroded

        ys, xs = np.where(border)
        used_border = len(ys) >= n_points * 4
        if not used_border:
            # Border band too thin — fall back to full mask
            ys, xs = np.where(mask)

        if len(ys) == 0:
            return [(mean_frame.shape[0] // 2, mean_frame.shape[1] // 2)]

        # Use |intensity| for brightness (handles inverted VSD preproc)
        weights = np.abs(mean_frame[ys, xs]).astype(float)

        # 4 angular sectors from centroid (handles elongated/irregular masks)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        angles = np.arctan2(ys - cy, xs - cx)  # -π..π
        # Sector 0: [-π/4, π/4) = right, 1: [π/4, 3π/4) = bottom, etc.
        sector = ((angles + np.pi / 4) // (np.pi / 2) % 4).astype(int)

        points = []
        for si in range(4):
            mask_s = sector == si
            if not mask_s.any():
                continue
            wq = np.where(mask_s, weights, -np.inf)
            # Score = brightness * min_distance_to_existing_points
            order = np.argsort(wq)[::-1]
            best_r, best_c, best_score = None, None, -1
            for bi in order:
                if wq[bi] == -np.inf:
                    break
                r, c = int(ys[bi]), int(xs[bi])
                if not points:
                    score = wq[bi]
                else:
                    min_d = min(((r - pr)**2 + (c - pc)**2)**0.5 for pr, pc in points)
                    score = wq[bi] * min_d
                if score > best_score:
                    best_score = score
                    best_r, best_c = r, c
                    if points and min(((best_r - pr)**2 + (best_c - pc)**2)**0.5 for pr, pc in points) >= 15:
                        break
            if best_r is not None:
                points.append((best_r, best_c))

        # Fill missing from global brightest
        w_fill = weights.copy()
        while len(points) < n_points and w_fill.max() >= 0:
            brightest = int(np.argmax(w_fill))
            r, c = int(ys[brightest]), int(xs[brightest])
            if not any(abs(r - pr) < 10 and abs(c - pc) < 10 for pr, pc in points):
                points.append((r, c))
            w_fill[brightest] = -np.inf

        return points[:n_points]

    @staticmethod
    def _measure_apd_point(trace: np.ndarray, peak_idx: int,
                           end_idx: int, fps: float,
                           levels: list) -> dict:
        """Measure APD at given thresholds for a single beat window.
        APD = time from peak to X% repolarization crossing."""
        dt_ms = 1000.0 / fps
        seg = trace[peak_idx:end_idx]
        if len(seg) < 3:
            return {f"APD{lv}": None for lv in levels}

        amp = seg[0] - seg.min()
        if amp <= 0:
            return {f"APD{lv}": None for lv in levels}

        results = {}
        for lv in levels:
            target = seg[0] - lv / 100.0 * amp
            for fi in range(1, len(seg)):
                if seg[fi] <= target and seg[fi - 1] > target:
                    frac = (seg[fi - 1] - target) / (seg[fi - 1] - seg[fi] + 1e-12)
                    results[f"APD{lv}"] = round((fi - 1 + frac) * dt_ms, 1)
                    break
            else:
                results[f"APD{lv}"] = None
        return results

    def _extract_4point_apd(self, preproc: np.ndarray, mask: np.ndarray,
                            peaks: np.ndarray, beat_starts: np.ndarray,
                            beat_ends: np.ndarray, fps: float,
                            bcl_ms: float, n_windows: int) -> list:
        """Extract APD at 4 tissue points with traces."""
        # Use preproc_video for point selection (mean frame)
        mean_frame = preproc.mean(axis=0)
        points = self._pick_tissue_points(mean_frame, mask, n_points=4)
        self.logger.info(f"[APD] 4-point: {points}")

        # Normalize traces: ΔF/F per pixel (p10 baseline)
        f0 = np.percentile(preproc, axis=0, q=10)
        f0 = np.where(f0 < 1, 1, f0)
        norm = (preproc.astype(np.float64) - f0) / f0

        point_results = []
        for pi, (r, c) in enumerate(points):
            trace_norm = norm[:, r, c]
            trace_raw = preproc[:, r, c].astype(float)

            # Per-beat APD
            per_beat_apds = []
            for bi in range(n_windows):
                start_f = int(beat_starts[bi])
                end_f = int(beat_ends[bi])
                if start_f < 0 or end_f <= start_f:
                    per_beat_apds.append({f"APD{lv}": None for lv in self.levels})
                    continue
                apd = self._measure_apd_point(
                    trace_norm, start_f, end_f, fps, self.levels)
                per_beat_apds.append(apd)

            # Median APD across beats
            median_apd = {}
            for lv in self.levels:
                vals = [pb[f"APD{lv}"] for pb in per_beat_apds
                        if pb[f"APD{lv}"] is not None]
                if vals:
                    median_apd[f"APD{lv}"] = round(float(np.median(vals)), 1)
                    median_apd[f"APD{lv}_std"] = round(float(np.std(vals)), 1)
                    median_apd[f"APD{lv}_n"] = len(vals)
                else:
                    median_apd[f"APD{lv}"] = None
                    median_apd[f"APD{lv}_std"] = None
                    median_apd[f"APD{lv}_n"] = 0

            point_results.append({
                "point": pi,
                "row": r, "col": c,
                "per_beat": per_beat_apds,
                "median": median_apd,
            })

        # Save 4-point traces PNG
        self._save_4point_png(norm, preproc, points, point_results, mask, fps, n_windows, beat_starts)

        self.logger.info(f"[MUST] Saved: apd_4points_traces.png")
        return point_results

    def _save_4point_png(self, norm: np.ndarray, raw: np.ndarray,
                        points: list, point_results: list,
                        mask: np.ndarray, fps: float,
                        n_windows: int, beat_starts: np.ndarray):
        """Save 4-panel trace figure + points map + APD bars."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            T = norm.shape[0]
            t_ms = np.arange(T) * 1000.0 / fps
            n_pts = len(points)
            colors = plt.cm.Set2(np.linspace(0, 1, max(n_pts, 2)))

            # --- Panel 1: Normalized traces (4 points, overlaid) ---
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))

            # Traces
            ax = axes[0, 0]
            for pi, (r, c) in enumerate(points):
                trace = norm[:, r, c]
                ax.plot(t_ms, trace, linewidth=0.6, color=colors[pi],
                        label=f"P{pi} ({r},{c})")
                # Mark beat starts
            for bs in beat_starts:
                ax.axvline(bs * 1000.0 / fps, color='red', linewidth=0.3,
                           alpha=0.3, linestyle='--')
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("ΔF/F")
            ax.set_title("Normalized traces (4 points)")
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.2)

            # Raw traces
            ax = axes[0, 1]
            for pi, (r, c) in enumerate(points):
                trace = raw[:, r, c].astype(float)
                ax.plot(t_ms, trace, linewidth=0.6, color=colors[pi],
                        label=f"P{pi}")
            for bs in beat_starts:
                ax.axvline(bs * 1000.0 / fps, color='red', linewidth=0.3,
                           alpha=0.3, linestyle='--')
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Raw signal")
            ax.set_title("Raw traces (4 points)")
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.2)

            # Points on mean frame
            ax = axes[1, 0]
            mean_frame = raw.mean(axis=0)
            masked_frame = mean_frame.copy().astype(float)
            masked_frame[~mask] = np.nan
            im = ax.imshow(masked_frame, cmap='gray', aspect='auto')
            for pi, (r, c) in enumerate(points):
                ax.plot(c, r, 'o', markersize=10, markeredgecolor=colors[pi],
                        markerfacecolor='none', markeredgewidth=2.5)
                ax.annotate(f"P{pi}", (c, r), textcoords="offset points",
                            xytext=(5, 5), color=colors[pi], fontsize=12,
                            fontweight='bold')
            ax.set_title("Sampling points on mean frame")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.grid(True, alpha=0.2)

            # APD bar chart
            ax = axes[1, 1]
            labels = [f"P{pi}" for pi in range(n_pts)]
            x = np.arange(n_pts)
            width = 0.2
            for i, lv in enumerate(self.levels):
                vals = [pr["median"].get(f"APD{lv}", 0) or 0 for pr in point_results]
                ax.bar(x + i * width, vals, width, label=f'APD{lv}', alpha=0.85)
            ax.set_xlabel("Point")
            ax.set_ylabel("APD (ms)")
            ax.set_title("APD at 4 points (median over beats)")
            ax.set_xticks(x + width * (len(self.levels) - 1) / 2)
            ax.set_xticklabels(labels)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2, axis='y')

            plt.suptitle(f"APD 4-Point Extraction ({n_windows} beats)", fontsize=13)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            path = self.must_dir / "apd_4points_traces.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.warning(f"4-point PNG skipped: {e}")


    # ------------------------------------------------------------------
    # PNG visualization
    # ------------------------------------------------------------------

    def _save_png(self, apd_maps: Dict[int, np.ndarray], mask: np.ndarray):
        """Save APD30/50/80 maps as a 3-panel PNG (must/apd_maps.png)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            panels = [
                (apd_maps.get(30), "APD30"),
                (apd_maps.get(50), "APD50"),
                (apd_maps.get(80), "APD80"),
            ]

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            for ax, (amap, title) in zip(axes, panels):
                if amap is None:
                    ax.set_title(f"{title} (missing)")
                    ax.axis("off")
                    continue
                masked = amap.astype(float).copy()
                masked[~mask] = np.nan
                im = ax.imshow(masked, cmap="hot")
                ax.set_title(title, fontsize=11)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("ms")

            plt.tight_layout()
            path = self.must_dir / "apd_maps.png"
            plt.savefig(path, dpi=150)
            plt.close()
            self.logger.info(f"[MUST] Saved: apd_maps.png")
        except Exception as e:
            self.logger.warning(f"APD maps PNG skipped: {e}")


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
    parser.add_argument("--force", action="store_true", help="Force rerun")
    args = parser.parse_args()

    cfg = PipelineConfig({
        "results_root": args.results_root,
        "apd": {
            "levels": [30, 50, 80],
            "min_amp_abs": 100.0,
            "min_amp_noise_mult": 3.0,
        },
    })
    agent = APDAgent(args.sample_id, config=cfg)
    result = agent.run(force=args.force)
    print(result)