#!/usr/bin/env python3
"""
PeakDetectorAgent — Stage 2.5: Beat Detection

CONTRACT (v3 ARCHITECTURE, 2026-07-09 — VARIANT A):
  - Preprocessing (LPF, inversion, spatial smooth) is the SOLE responsibility of LoaderAgent.
  - PeakDetectorAgent is a CONSUMER: reads preproc_video.npy from must/, never recomputes.
  - Single source of truth: one preprocess, downstream stages read the same signal.

Inputs (lazy):
  - must/preproc_video.npy   (from LoaderAgent — 80 Hz LPF, inverted, ready for detection)
  - must/mask.npy            (from MaskAgent)
  - must/metadata.json       (from LoaderAgent — fps, dye, recording_mode, stim_hz)

Outputs:
  - must/peaks.npy                  — frame indices of beats
  - must/peak_detection_meta.json   — fps, stim_hz, n_peaks, params
  - debug/mean_trace.npy            — smoothed mean tissue trace
  - debug/preproc_stats.json        — min/max/mean/std of preproc_video (for QA)

Architecture:
  - Inherits from BaseAgent (must/debug contracts, sample_id paths, OmegaConf)
  - Lazy-calls LoaderAgent + MaskAgent via ensure_dependencies
  - Config-driven via config.peak_detector section (only detection params, not preprocess)

Исправления при интеграции (2026-07-02):
  - Удалён собственный BaseAgent-стаб → cardiac_pipeline.base_agent
  - Удалён sys.path.insert → пакетные импорты
  - save_intermediate() → save_debug() (BaseAgent API, preproc_video в debug/)
  - save_must(sample_id, dict) → save_must(arr, filename) (BaseAgent API)
  - _get_fps_and_stim: fps fallback 1000 → raise ValueError (F1 fix)
  - fps=0 / fps<0 → raise ValueError (AG1 fix)
  - n_peaks < 3 → ValueError (AG2 fix, не тихий return False)
  - print() → self.logger
  - peak_detector секция добавлена в config/default.yaml
  - stim_hz читается из metadata.json (не только из config)

Рефакторинг 2026-07-09 (VARIANT A — Loader owns preprocessing):
  - Удалён вызов preprocess_video() — PeakDetector больше НЕ фильтрует/инвертирует данные
  - Удалён импорт preprocess_video, should_invert
  - Прямой контракт: must/preproc_video.npy должен существовать (иначе ValueError)
  - DEPENDS_ON = [LoaderAgent, MaskAgent] — lazy-цепочка даёт preproc_video.npy
  - REQUIRED_INPUTS обновлён: ["preproc_video.npy", "mask.npy", "metadata.json"]
  - Удалён self.sigma, self.lp_cutoff, self.chunk_size (больше не нужны агенту)
  - Оставлены только detection params: threshold_frac, sigma_temporal,
    min_distance_factor, drop_first, min_peaks
    (v3.6 spec, 2026-07-09: mean_tissue агрегируется в агенте, не в detect_beats)
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig

try:
    from cardiac_pipeline.utils.peak_detection import detect_beats
    from cardiac_pipeline.utils.region_selector import select_regions_grid
    from cardiac_pipeline.utils.voting import consensus_peaks, select_top_beats
    from cardiac_pipeline.utils.soft_assignment import compute_soft_weights
    PEAK_DETECTION_AVAILABLE = True
except ImportError:
    detect_beats = None
    PEAK_DETECTION_AVAILABLE = False


class PeakDetectorAgent(BaseAgent):
    """
    PeakDetectorAgent — beat detection stage (v3 Variant A).

    CONSUMER contract: reads preproc_video.npy from must/, never recomputes preprocessing.
    LoaderAgent is the single source of truth for spatial/temporal filtering + inversion.

    Inputs (lazy via DEPENDS_ON = [LoaderAgent, MaskAgent]):
      - must/preproc_video.npy   (from LoaderAgent — 80 Hz LPF + inversion)
      - must/mask.npy            (from MaskAgent)
      - must/metadata.json       (from LoaderAgent — fps, dye, recording_mode, stim_hz)

    Outputs:
      - must/peaks.npy
      - must/peak_detection_meta.json
      - debug/mean_trace.npy
      - debug/preproc_stats.json
    """

    DEPENDS_ON: list = []  # [LoaderAgent, MaskAgent] — установлен ниже (lazy import)
    REQUIRED_INPUTS: list = ["preproc_video.npy", "mask.npy", "metadata.json"]

    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None):
        super().__init__(sample_id, config)

        # Detection params only (preprocessing is LoaderAgent's job)
        # v3.6 spec (2026-07-09): threshold_frac=0.5, min_distance_factor=0.6,
        # drop_first=False. prominence_frac deprecated.
        pd_cfg = getattr(self.config, 'peak_detector', {}) or {}

        # Accept both new (threshold_frac) and old (prominence_frac) names
        # for back-compat, but prefer new.
        if 'threshold_frac' in pd_cfg:
            self.threshold_frac = float(pd_cfg.get('threshold_frac'))
        else:
            self.threshold_frac = float(pd_cfg.get('prominence_frac', 0.5))
        self.sigma_temporal      = float(pd_cfg.get('sigma_temporal', 3.0))
        self.min_distance_factor = float(pd_cfg.get('min_distance_factor', 0.6))
        self.drop_first          = bool(pd_cfg.get('drop_first', False))
        self.min_peaks           = int(pd_cfg.get('min_peaks', 3))

        # v3.7 multi-trace params
        self.n_regions              = int(pd_cfg.get('n_regions', 1))
        self.min_region_pixels      = int(pd_cfg.get('min_region_pixels', 50))
        self.min_agreement          = int(pd_cfg.get('min_agreement', 2))
        self.frame_tolerance        = int(pd_cfg.get('frame_tolerance', 10))
        self.soft_assignment_sigma  = float(pd_cfg.get('soft_assignment_sigma', 20.0))
        self.min_quality            = float(pd_cfg.get('min_quality', 0.66))
        self.n_beats_select         = int(pd_cfg.get('n_beats_select', 3))

        self.metadata: Dict[str, Any] = {}

    # ==================== HELPERS ====================

    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = self.get_path("metadata.json", kind="must")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self.logger.warning("metadata.json not found")
        return self.metadata

    def _get_fps(self) -> float:
        """Read fps from metadata.json. Raises if missing or invalid (AG1 fix)."""
        # Try both key names for compatibility
        fps = self.metadata.get("fps") or self.metadata.get("fps_hz")
        if fps is None:
            raise ValueError(
                "fps отсутствует в metadata.json. "
                "LoaderAgent должен сохранить его заранее."
            )
        fps = float(fps)
        if fps <= 0:
            raise ValueError(
                f"fps некорректен (fps={fps}): ожидается положительное число."
            )
        return fps

    def _get_stim_hz(self) -> float:
        """
        stim_hz priority: metadata.stim_hz_effective > metadata.stim_hz >
                          metadata.pacing_hz > config.stim_hz > 10.0 fallback.

        NOTE (2026-07-09): Some .rsh metadata has stim_hz=500.0 (which is
        actually fps). stim_hz_effective is the post-pacing-corrected value
        (e.g. 5.86 for 6Hz pacing) and must be preferred.
        """
        eff = self.metadata.get("stim_hz_effective")
        if eff is not None and float(eff) > 0:
            return float(eff)
        stim = self.metadata.get("stim_hz") or self.metadata.get("pacing_hz")
        if stim is not None and float(stim) > 0:
            return float(stim)
        cfg_stim = getattr(self.config, 'stim_hz', None)
        if cfg_stim is not None and float(cfg_stim) > 0:
            return float(cfg_stim)
        self.logger.warning(
            "stim_hz not found in metadata or config — using 10.0 Hz default"
        )
        return 10.0

    def _load_mask(self) -> np.ndarray:
        """Load mask.npy, run MaskAgent if missing."""
        if not self.exists("mask.npy"):
            self.logger.info("mask.npy not found — running MaskAgent")
            from cardiac_pipeline.agents.mask_agent import MaskAgent
            MaskAgent(self.sample_id, self.config).run()
        return self.load_must("mask.npy").astype(bool)

    def _load_preproc_video(self) -> np.ndarray:
        """Load preproc_video.npy from must/ (created by LoaderAgent).

        Variant A contract: PeakDetector is a CONSUMER, not a producer, of preprocessed data.
        If the file is missing, raise immediately — do NOT recompute preprocessing.
        """
        if not self.exists("preproc_video.npy"):
            raise FileNotFoundError(
                f"must/preproc_video.npy не найден для {self.sample_id}. "
                f"LoaderAgent должен быть запущен первым (Variant A: Loader owns preprocessing). "
                f"Запустите LoaderAgent явно: python -m cardiac_pipeline.agents.loader_agent {self.sample_id}"
            )
        return self.load_must("preproc_video.npy")

    # ==================== v3.6 LEGACY: single-trace ====================

    def _detect_beats_single_trace(
        self,
        preproc_video: np.ndarray,
        mask: np.ndarray,
        T: int, H: int, W: int,
        fps: float,
        stim_hz: float,
    ) -> Dict[str, Any]:
        """v3.6 legacy mode: one global mean_tissue trace, detect_beats once."""
        n_mask = int(mask.sum())
        mean_tissue = preproc_video.reshape(T, H * W)[:, mask.ravel()].mean(axis=1)
        self.logger.info(
            f"[single-trace] mean_tissue shape: {mean_tissue.shape}, "
            f"range=[{mean_tissue.min():.1f}, {mean_tissue.max():.1f}], "
            f"mask_n={n_mask}"
        )

        self.logger.info(
            f"[single-trace] Detecting beats (stim_hz={stim_hz}, "
            f"threshold_frac={self.threshold_frac}, "
            f"sigma_temporal={self.sigma_temporal}, "
            f"min_distance_factor={self.min_distance_factor}, "
            f"drop_first={self.drop_first})"
        )
        peaks, smoothed = detect_beats(
            mean_tissue,
            fps=fps,
            stim_hz=stim_hz,
            sigma_temporal=self.sigma_temporal,
            threshold_frac=self.threshold_frac,
            min_distance_factor=self.min_distance_factor,
            drop_first=self.drop_first,
        )

        # Synthetic multi-trace-compatible fields (n_regions=1).
        # ActivationAgent requires peaks_per_region, weights, region_masks
        # even in single-trace mode. Build them here so downstream stages
        # don't need to special-case n_regions=1.
        region_masks_arr = mask[None, ...].astype(np.uint8)  # (1, H, W)
        ys, xs = np.where(mask)
        region_centers = np.array([[ys.mean(), xs.mean()]], dtype=np.float32)
        peaks_per_region = peaks[None, :].astype(np.int64)   # (1, n_peaks)
        weights = np.ones((H, W, 1), dtype=np.float32)       # (H, W, 1)
        weights[~mask] = 0.0

        return {
            "peaks_global":       peaks.astype(np.int64),
            "smoothed":           smoothed,
            "mean_tissue":        mean_tissue,
            # In single-trace mode, ALL peaks are consensus (1/1 agreement).
            # selected_peaks = all peaks, selected_indices = 0..N-1
            "selected_peaks":     peaks.astype(np.int64),
            "selected_indices":   np.arange(len(peaks), dtype=np.int64),
            # Synthetic multi-trace fields (n_regions=1)
            "peaks_per_region":   peaks_per_region,
            "region_centers":     region_centers,
            "region_masks":       region_masks_arr,
            "region_quality":     np.array([1.0], dtype=np.float32),
            "weights":            weights,
            "consensus_peaks":    peaks.astype(np.int64),
            "consensus_agreement": np.ones(len(peaks), dtype=np.float32),
            "traces_per_region":  mean_tissue[None, ...].astype(np.float32),
        }

    # ==================== v3.7 MULTI-TRACE ====================

    def _detect_beats_multi_trace(
        self,
        preproc_video: np.ndarray,
        mask: np.ndarray,
        T: int, H: int, W: int,
        fps: float,
        stim_hz: float,
    ) -> Dict[str, Any]:
        """v3.7 multi-trace mode.

        1. Compute pixel_std to score regions
        2. Select n_regions=3 cells from 3x3 mask_grid (best n by mean_std)
        3. For each region: mean_tissue -> detect_beats -> peaks[r]
        4. Voting: consensus_peaks = peaks agreed by >= min_agreement regions
        5. Soft assignment: weights[h, w, r] = exp(-dist^2 / 2 sigma^2)
        6. select_top_beats: top-3 consensus peaks by agreement quality
        """
        n_mask = int(mask.sum())
        self.logger.info(
            f"[multi-trace v3.7] Starting with n_regions={self.n_regions}, "
            f"min_agreement={self.min_agreement}, "
            f"frame_tolerance={self.frame_tolerance}, "
            f"soft_sigma={self.soft_assignment_sigma}"
        )

        # 1. Compute pixel_std for region scoring
        pixel_std = preproc_video.reshape(T, H * W).std(axis=0).reshape(H, W)
        self.logger.info(
            f"[multi-trace] pixel_std range: "
            f"[{pixel_std[mask].min():.1f}, {pixel_std[mask].max():.1f}], "
            f"median={np.median(pixel_std[mask]):.1f}"
        )

        # 2. Select n regions via mask_grid
        try:
            region_masks, region_centers, _region_info = select_regions_grid(
                mask, pixel_std, n=self.n_regions, min_region_pixels=self.min_region_pixels
            )
        except ValueError as e:
            # If region selection fails (mask too small), fall back to single trace
            self.logger.warning(
                f"[multi-trace] Region selection failed ({e}); "
                f"falling back to single-trace mode"
            )
            return self._detect_beats_single_trace(
                preproc_video, mask, T, H, W, fps, stim_hz
            )

        n_actual = len(region_centers)
        self.logger.info(
            f"[multi-trace] Selected {n_actual} regions: "
            f"{[(int(c[0]), int(c[1])) for c in region_centers]}"
        )

        # 3. Per-region peak detection
        peaks_per_region: List[np.ndarray] = []
        traces_per_region: List[np.ndarray] = []
        region_quality: List[float] = []

        for r in range(n_actual):
            region_mask = region_masks[r]
            n_pix = int(region_mask.sum())
            mean_tissue_r = preproc_video.reshape(T, H * W)[:, region_mask.ravel()].mean(axis=1)
            quality = float(pixel_std[region_mask].mean())

            self.logger.info(
                f"[multi-trace] Region {r}: center=({region_centers[r][0]}, {region_centers[r][1]}), "
                f"n_pix={n_pix}, quality={quality:.1f}, "
                f"trace_range=[{mean_tissue_r.min():.1f}, {mean_tissue_r.max():.1f}]"
            )

            try:
                peaks_r, smoothed_r = detect_beats(
                    mean_tissue_r,
                    fps=fps,
                    stim_hz=stim_hz,
                    sigma_temporal=self.sigma_temporal,
                    threshold_frac=self.threshold_frac,
                    min_distance_factor=self.min_distance_factor,
                    drop_first=self.drop_first,
                )
            except (ValueError, RuntimeError) as e:
                self.logger.warning(
                    f"[multi-trace] Region {r}: detect_beats failed ({e}); "
                    f"using empty peaks"
                )
                peaks_r = np.array([], dtype=np.int64)
                smoothed_r = mean_tissue_r

            self.logger.info(
                f"[multi-trace] Region {r}: detected {len(peaks_r)} peaks"
            )

            peaks_per_region.append(peaks_r.astype(np.int64))
            traces_per_region.append(smoothed_r)
            region_quality.append(quality)

        # 4. Voting: consensus peaks
        try:
            consensus, agreement = consensus_peaks(
                peaks_per_region,
                n_regions=n_actual,
                min_agreement=self.min_agreement,
                frame_tolerance=self.frame_tolerance,
            )
        except Exception as e:
            self.logger.warning(
                f"[multi-trace] consensus_peaks failed ({e}); "
                f"using union of all peaks"
            )
            # Union fallback
            all_p = sorted(set(p for ps in peaks_per_region for p in ps))
            consensus = np.array(all_p, dtype=np.int64)
            agreement = np.ones(len(consensus), dtype=np.float32)

        self.logger.info(
            f"[multi-trace] Consensus: {len(consensus)} peaks from "
            f"n_actual={n_actual} regions, "
            f"agreement mean={agreement.mean():.2f}, "
            f"min={agreement.min():.2f}, max={agreement.max():.2f}"
        )

        # 5. Soft assignment weights
        weights = compute_soft_weights(
            region_centers, (H, W), sigma=self.soft_assignment_sigma
        )
        self.logger.info(
            f"[multi-trace] Soft weights shape: {weights.shape}, "
            f"sum-to-1 check: {weights.sum(axis=2).mean():.4f}"
        )

        # 6. Select ALL consensus beats (not just top-N) for downstream stages.
        # n_beats_select limits APD quality ranking internally, but selected_peaks
        # must carry all valid beats so AlternansAgent (Stage 7) has enough data.
        n_consensus = len(consensus)
        selected_peaks, selected_indices = select_top_beats(
            consensus,
            agreement,
            n_beats=n_consensus,  # ALL consensus beats, not just top-N
            min_quality=self.min_quality,
            sort_by="temporal",    # preserve temporal order
        )
        # Pad selected_peaks to n_consensus length for shape stability
        if len(selected_peaks) < n_consensus:
            pad = np.full(n_consensus - len(selected_peaks), -1, dtype=np.int64)
            selected_peaks = np.concatenate([selected_peaks, pad])
            pad_idx = np.full(n_consensus - len(selected_indices), -1, dtype=np.int64)
            selected_indices = np.concatenate([selected_indices, pad_idx])

        self.logger.info(
            f"[multi-trace] Selected {int((selected_peaks >= 0).sum())} beats: "
            f"{selected_peaks[selected_peaks >= 0].tolist()} "
            f"(indices: {selected_indices[selected_indices >= 0].tolist()})"
        )

        # Build padded peaks_per_region for .npy storage
        max_beats = max(len(p) for p in peaks_per_region) if peaks_per_region else 0
        peaks_per_region_padded = np.full((n_actual, max_beats), -1, dtype=np.int64)
        for r, p in enumerate(peaks_per_region):
            if len(p) > 0:
                peaks_per_region_padded[r, :len(p)] = p

        # Build region_masks array
        region_masks_arr = np.stack(region_masks, axis=0).astype(np.uint8)

        return {
            "peaks_global": consensus,                              # consensus peaks (saved as peaks.npy)
            "peaks_per_region": peaks_per_region_padded,
            "region_centers": np.array(region_centers, dtype=np.int64),
            "region_masks": region_masks_arr,
            "region_quality": np.array(region_quality, dtype=np.float32),
            "weights": weights,
            "consensus_peaks": consensus,
            "consensus_agreement": agreement,
            "selected_peaks": selected_peaks.astype(np.int64),
            "selected_indices": selected_indices.astype(np.int64),
            "traces_per_region": np.array(traces_per_region, dtype=np.float32),
            "smoothed": np.array(traces_per_region).mean(axis=0) if traces_per_region else None,
            "mean_tissue": None,
        }

    # ==================== RUN ====================

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Main entry point (Variant A — Consumer contract).

        1. Skip if peaks.npy exists and force=False
        2. Lazy: ensure preproc_video.npy + mask.npy + metadata.json exist
           (runs LoaderAgent + MaskAgent if missing)
        3. Read preproc_video.npy (NO recomputation)
        4. Detect beats
        5. Gate: raise if n_peaks < min_peaks
        6. Save artifacts
        """
        if not force and self.exists("peaks.npy"):
            self.logger.info("peaks.npy already exists, skipping (use force=True to rerun)")
            return {"status": "skipped"}

        t0 = time.perf_counter()

        # --- Lazy: ensure Loader + Mask produced all REQUIRED_INPUTS ---
        # NOTE (Variant A, 2026-07-09): ensure_dependencies auto-runs upstream agents
        # if REQUIRED_INPUTS are missing. This is a CONVENIENCE for dev workflows.
        # In production, run LoaderAgent explicitly first to lock in preprocessing params
        # (no silent Loader re-run with default stim_hz=NaN).
        from cardiac_pipeline.agents.loader_agent import LoaderAgent
        from cardiac_pipeline.agents.mask_agent import MaskAgent
        self.DEPENDS_ON = [LoaderAgent, MaskAgent]
        self.ensure_dependencies(force=force)

        # --- 1. Load metadata ---
        self._load_metadata()

        fps     = self._get_fps()
        stim_hz = self._get_stim_hz()
        invert  = bool(self.metadata.get("recording_mode", "").lower() in
                       ("voltage", "vsd", "ap") or
                       str(self.metadata.get("dye", "")).upper().startswith("A"))
        self.logger.info(f"FPS={fps}, stim_hz={stim_hz}, inverted={invert}")

        # --- 2. Load preprocessed video (Variant A: no recompute) ---
        preproc_video = self._load_preproc_video()
        mask          = self._load_mask()
        self.logger.info(f"Preproc video shape: {preproc_video.shape}, "
                         f"mask coverage: {mask.mean():.3f}, "
                         f"range=[{preproc_video.min():.1f}, {preproc_video.max():.1f}]")

        # --- 3. Multi-trace OR single-trace peak detection (v3.7) ---
        T, H, W = preproc_video.shape
        n_mask = int(mask.sum())
        if n_mask == 0:
            raise ValueError(
                f"Empty mask for sample {self.sample_id}: 0 pixels in tissue. "
                f"Mask QC failed."
            )

        if self.n_regions > 1:
            # v3.7 multi-trace mode
            result = self._detect_beats_multi_trace(
                preproc_video, mask, T, H, W, fps, stim_hz
            )
        else:
            # v3.6 legacy single-trace mode
            result = self._detect_beats_single_trace(
                preproc_video, mask, T, H, W, fps, stim_hz
            )

        peaks_global = result["peaks_global"]
        n_peaks = int(len(peaks_global))
        self.logger.info(f"Detected {n_peaks} consensus peaks: {peaks_global.tolist()}")

        # --- 4. Gating (AG2 fix: no silent pass) ---
        if n_peaks < self.min_peaks:
            raise ValueError(
                f"Слишком мало consensus peaks: {n_peaks} (требуется минимум {self.min_peaks}). "
                f"Sample {self.sample_id} требует ручной проверки."
            )

        # --- 5. Save artifacts ---
        self.save_must(peaks_global, "peaks.npy")

        # selected_peaks: ALL consensus beats (both single and multi-trace modes).
        # AlternansAgent (Stage 7) needs all beats — n_beats_select does NOT limit this.
        if "selected_peaks" in result:
            self.save_must(result["selected_peaks"], "selected_peaks.npy")
        if "selected_indices" in result:
            self.save_must(result["selected_indices"], "selected_indices.npy")

        peak_meta = {
            "sample_id":              self.sample_id,
            "fps":                    fps,
            "stim_hz":                stim_hz,
            "n_peaks":                n_peaks,
            "threshold_frac":         self.threshold_frac,
            "sigma_temporal":         self.sigma_temporal,
            "min_distance_factor":    self.min_distance_factor,
            "drop_first":             self.drop_first,
            "inverted":               invert,
            "preprocessing_owner":    "LoaderAgent",   # Variant A marker
            "n_regions":              self.n_regions,  # v3.7
        }

        # v3.7 multi-trace outputs (also saved in single-trace mode since
        # _detect_beats_single_trace now returns synthetic n_regions=1 fields)
        self.save_must(result["peaks_per_region"], "peaks_per_region.npy")
        self.save_must(result["region_centers"], "region_centers.npy")
        self.save_must(result["region_masks"], "region_masks.npy")
        self.save_must(result["region_quality"], "region_quality.npy")
        self.save_must(result["weights"], "weights.npy")
        self.save_must(result["consensus_peaks"], "consensus_peaks.npy")
        self.save_must(result["consensus_agreement"], "consensus_agreement.npy")
        self.save_must(result["selected_peaks"], "selected_peaks.npy")
        self.save_must(result["selected_indices"], "selected_indices.npy")
        self.save_must(result["traces_per_region"], "traces_per_region.npy")

        peak_meta.update({
            "n_beats_selected":      int((result["selected_peaks"] >= 0).sum()),
            "min_agreement":         self.min_agreement,
            "frame_tolerance":       self.frame_tolerance,
            "soft_assignment_sigma": self.soft_assignment_sigma,
            "min_quality":           self.min_quality,
        })

        self.save_must(peak_meta, "peak_detection_meta.json")

        # Save 1D smoothed trace (debug). Note: this is the smoothed mean
        # tissue, NOT the preproc_video aggregation (use mean_tissue itself
        # for raw aggregation).
        if "smoothed" in result:
            self.save_debug(result["smoothed"], "mean_trace.npy")
        if "mean_tissue" in result:
            self.save_debug(result["mean_tissue"], "mean_tissue_raw.npy")
        self.save_debug({
            "min":  float(preproc_video.min()),
            "max":  float(preproc_video.max()),
            "mean": float(preproc_video.mean()),
            "std":  float(preproc_video.std()),
        }, "preproc_stats.json")

        elapsed = time.perf_counter() - t0
        self.logger.info(f"Finished in {elapsed:.2f}s")

        metrics = {**peak_meta, "elapsed_s": round(elapsed, 3)}
        self._log_metrics(metrics)

        return {
            "status":  "success",
            "n_peaks": n_peaks,
            "peaks_path": str(self.get_path("peaks.npy")),
            "metrics": metrics,
        }


# ---------------------------------------------------------------------------
# Standalone CLI (development / debugging)
# ---------------------------------------------------------------------------
# NOTE (Variant A, 2026-07-09): PeakDetector CLI expects preproc_video.npy from LoaderAgent.
# Run LoaderAgent first:
#   python -m cardiac_pipeline.agents.loader_agent <sample_id>
#   python -m cardiac_pipeline.agents.mask_agent    <sample_id>
#   python -m cardiac_pipeline.agents.peak_detector_agent <sample_id>

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="PeakDetectorAgent standalone (Variant A: consumer)")
    parser.add_argument("sample_id", help="Sample ID (e.g. 005A)")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()

    cfg = PipelineConfig()
    cfg.results_root = Path(args.results_root)
    cfg.peak_detector = {
        "threshold_frac":      0.5,
        "sigma_temporal":      3.0,
        "min_distance_factor": 0.6,
        "drop_first":          False,
        "min_peaks":           3,
    }

    agent = PeakDetectorAgent(args.sample_id, config=cfg)
    result = agent.run()
    print(result)
