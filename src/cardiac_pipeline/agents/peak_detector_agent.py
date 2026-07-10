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
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig

try:
    from cardiac_pipeline.utils.peak_detection import detect_beats
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

        # --- 3. Aggregate mean_tissue (1D) and detect beats ---
        # CRITICAL: preproc_video is (T, H, W), mask is (H, W) bool.
        # preproc_video[:, mask] broadcasts to (T, 100, 100) — WRONG.
        # Correct: reshape to (T, H*W) and apply mask.ravel() → (T, n_mask).
        # mean(axis=1) over masked pixels only → (T,).
        T, H, W = preproc_video.shape
        n_mask = int(mask.sum())
        if n_mask == 0:
            raise ValueError(
                f"Empty mask for sample {self.sample_id}: 0 pixels in tissue. "
                f"Mask QC failed."
            )
        mean_tissue = preproc_video.reshape(T, H * W)[:, mask.ravel()].mean(axis=1)
        self.logger.info(f"mean_tissue shape: {mean_tissue.shape}, "
                         f"range=[{mean_tissue.min():.1f}, {mean_tissue.max():.1f}], "
                         f"mask_n={n_mask}")

        if not PEAK_DETECTION_AVAILABLE:
            raise ImportError(
                "cardiac_pipeline.utils.peak_detection not available. "
                "Cannot detect beats."
            )

        self.logger.info(f"Detecting beats (stim_hz={stim_hz}, "
                         f"threshold_frac={self.threshold_frac}, "
                         f"sigma_temporal={self.sigma_temporal}, "
                         f"min_distance_factor={self.min_distance_factor}, "
                         f"drop_first={self.drop_first})")
        peaks, smoothed = detect_beats(
            mean_tissue,
            fps=fps,
            stim_hz=stim_hz,
            sigma_temporal=self.sigma_temporal,
            threshold_frac=self.threshold_frac,
            min_distance_factor=self.min_distance_factor,
            drop_first=self.drop_first,
        )
        n_peaks = int(len(peaks))
        self.logger.info(f"Detected {n_peaks} peaks: {peaks.tolist()}")

        # --- 4. Gating (AG2 fix: no silent pass) ---
        if n_peaks < self.min_peaks:
            raise ValueError(
                f"Слишком мало пиков: {n_peaks} (требуется минимум {self.min_peaks}). "
                f"Sample {self.sample_id} требует ручной проверки."
            )

        # --- 5. Save artifacts ---
        self.save_must(peaks, "peaks.npy")

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
        }
        self.save_must(peak_meta, "peak_detection_meta.json")

        # Save 1D smoothed trace (debug). Note: this is the smoothed mean
        # tissue, NOT the preproc_video aggregation (use mean_tissue itself
        # for raw aggregation).
        self.save_debug(smoothed, "mean_trace.npy")
        self.save_debug(mean_tissue, "mean_tissue_raw.npy")
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

    cfg = PipelineConfig({
        "results_root": args.results_root,
        "peak_detector": {
            "threshold_frac":      0.5,
            "sigma_temporal":      3.0,
            "min_distance_factor": 0.6,
            "drop_first":          False,
            "min_peaks":           3,
        },
    })

    agent = PeakDetectorAgent(args.sample_id, config=cfg)
    result = agent.run()
    print(result)
