#!/usr/bin/env python3
"""
PeakDetectorAgent — Stage: Preprocessing + Beat Detection

Single source of truth for:
  - Spatial + temporal preprocessing (smooth + Butterworth LP)
  - Dye-aware signal inversion (VSD A vs Calcium B)
  - Robust beat/peak detection

Produces:
  - MUST:  peaks.npy                  — frame indices of beats
  - MUST:  peak_detection_meta.json   — fps, stim_hz, n_peaks, params
  - DEBUG: mean_trace.npy             — mean masked signal (for validation)
  - DEBUG: preproc_stats.json         — min/max/mean/std of preproc_video
  - INTERMEDIATE (debug/): preproc_video.npy — smoothed, filtered, inverted

Architecture:
  - Inherits from BaseAgent (must/debug contracts, sample_id paths, OmegaConf)
  - Lazy-calls LoaderAgent + MaskAgent if needed
  - Config-driven via config.peak_detector section

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
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig

try:
    from cardiac_pipeline.utils.preprocess import preprocess_video, should_invert
    PREPROCESS_AVAILABLE = True
except ImportError:
    preprocess_video = None
    should_invert = None
    PREPROCESS_AVAILABLE = False

try:
    from cardiac_pipeline.utils.peak_detection import detect_beats
    PEAK_DETECTION_AVAILABLE = True
except ImportError:
    detect_beats = None
    PEAK_DETECTION_AVAILABLE = False


class PeakDetectorAgent(BaseAgent):
    """
    PeakDetectorAgent — unified preprocessing + beat detection stage.

    Inputs (lazy):
      - must/raw_video.npy  (from LoaderAgent)
      - must/mask.npy          (from MaskAgent)
      - must/metadata.json     (from LoaderAgent)

    Outputs:
      - must/peaks.npy
      - must/peak_detection_meta.json
      - debug/mean_trace.npy
      - debug/preproc_stats.json
      - debug/preproc_video.npy   (heavy intermediate, can be deleted after run)
    """

    DEPENDS_ON: list = []  # [LoaderAgent, MaskAgent] — установлен ниже (lazy import)
    REQUIRED_INPUTS: list = ["mask.npy", "metadata.json"]

    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None):
        super().__init__(sample_id, config)

        # Config section: config.peak_detector (falls back to config.preprocess)
        pd_cfg = getattr(self.config, 'peak_detector', {}) or {}
        pre_cfg = getattr(self.config, 'preprocess', {}) or {}

        self.sigma          = float(pd_cfg.get('spatial_sigma',   pre_cfg.get('spatial_sigma',   2.0)))
        self.lp_cutoff      = float(pd_cfg.get('lp_cutoff_hz',    pre_cfg.get('temporal_cutoff_hz', 80.0)))
        self.prominence_frac = float(pd_cfg.get('prominence_frac', 0.3))
        self.chunk_size     = int(pd_cfg.get('chunk_size',         pre_cfg.get('chunk_size',      8192)))
        self.min_peaks      = int(pd_cfg.get('min_peaks',          3))

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
        stim_hz: приоритет metadata.json > config.stim_hz > 10.0 (documented fallback).
        """
        stim = self.metadata.get("stim_hz") or self.metadata.get("pacing_hz")
        if stim is not None:
            return float(stim)
        cfg_stim = getattr(self.config, 'stim_hz', None)
        if cfg_stim is not None:
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

    def _load_video(self) -> np.ndarray:
        """Load raw_video.npy, run LoaderAgent if missing."""
        if not self.exists("raw_video.npy"):
            self.logger.info("raw_video.npy not found — running LoaderAgent")
            from cardiac_pipeline.agents.loader_agent import LoaderAgent
            LoaderAgent(self.sample_id, self.config).run()
        return self.load_must("raw_video.npy")

    # ==================== RUN ====================

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Main entry point.

        1. Skip if peaks.npy exists and force=False
        2. Load metadata, video, mask (lazy upstream)
        3. Preprocess (spatial + temporal filter + inversion)
        4. Detect beats
        5. Gate: raise if n_peaks < min_peaks (AG2 fix)
        6. Save artifacts
        """
        if not force and self.exists("peaks.npy"):
            self.logger.info("peaks.npy already exists, skipping (use force=True to rerun)")
            return {"status": "skipped"}

        t0 = time.perf_counter()

        # --- Lazy: запускаем Loader + Mask если их выходы отсутствуют ---
        from cardiac_pipeline.agents.loader_agent import LoaderAgent
        from cardiac_pipeline.agents.mask_agent import MaskAgent
        self.DEPENDS_ON = [LoaderAgent, MaskAgent]
        self.ensure_dependencies(force=force)

        # --- 1. Load metadata ---
        self._load_metadata()

        fps     = self._get_fps()
        stim_hz = self._get_stim_hz()
        self.logger.info(f"FPS={fps}, stim_hz={stim_hz}")

        # --- 2. Load inputs ---
        raw_video = self._load_video()
        mask      = self._load_mask()
        self.logger.info(f"Video shape: {raw_video.shape}, mask coverage: {mask.mean():.3f}")

        # --- 3. Preprocessing ---
        if not PREPROCESS_AVAILABLE:
            raise ImportError(
                "cardiac_pipeline.utils.preprocess not available. "
                "Cannot run PeakDetectorAgent."
            )

        self.logger.info(
            f"Preprocessing: sigma={self.sigma}, lp_cutoff={self.lp_cutoff} Hz, "
            f"chunk_size={self.chunk_size}"
        )

        invert = should_invert(
            sample_name=self.sample_id,
            dye=self.metadata.get("dye"),
            recording_mode=self.metadata.get("recording_mode"),
        )
        self.logger.info(f"Inversion: {invert}")

        preproc_video = preprocess_video(
            raw_video,
            mask=mask,
            fps=fps,
            sigma=self.sigma,
            lp_cutoff=self.lp_cutoff,
            chunk_size=self.chunk_size,
            invert=invert,
            sample_name=self.sample_id,
            dye=self.metadata.get("dye"),
            recording_mode=self.metadata.get("recording_mode"),
            do_normalize=False,
        )

        # Save preprocessed video as debug intermediate (heavy, can be deleted)
        self.save_debug(preproc_video, "preproc_video.npy")
        self.logger.info("Saved debug/preproc_video.npy")

        # --- 4. Beat detection ---
        if not PEAK_DETECTION_AVAILABLE:
            raise ImportError(
                "cardiac_pipeline.utils.peak_detection not available. "
                "Cannot detect beats."
            )

        self.logger.info(f"Detecting beats (stim_hz={stim_hz}, prominence_frac={self.prominence_frac})")
        peaks, mean_trace = detect_beats(
            preproc_video,
            mask,
            fps=fps,
            stim_hz=stim_hz,
            prominence_frac=self.prominence_frac,
        )
        n_peaks = int(len(peaks))
        self.logger.info(f"Detected {n_peaks} peaks: {peaks.tolist()}")

        # --- 5. Gating (AG2 fix: no silent pass) ---
        if n_peaks < self.min_peaks:
            raise ValueError(
                f"Слишком мало пиков: {n_peaks} (требуется минимум {self.min_peaks}). "
                f"Sample {self.sample_id} требует ручной проверки."
            )

        # --- 6. Save artifacts ---
        self.save_must(peaks, "peaks.npy")

        peak_meta = {
            "sample_id":        self.sample_id,
            "fps":              fps,
            "stim_hz":          stim_hz,
            "n_peaks":          n_peaks,
            "prominence_frac":  self.prominence_frac,
            "spatial_sigma":    self.sigma,
            "lp_cutoff_hz":     self.lp_cutoff,
            "inverted":         bool(invert),
        }
        self.save_must(peak_meta, "peak_detection_meta.json")

        self.save_debug(mean_trace, "mean_trace.npy")
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

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="PeakDetectorAgent standalone")
    parser.add_argument("sample_id", help="Sample ID (e.g. 005A)")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()

    cfg = PipelineConfig({
        "results_root": args.results_root,
        "peak_detector": {
            "spatial_sigma":   2.0,
            "lp_cutoff_hz":    80.0,
            "prominence_frac": 0.3,
            "chunk_size":      8192,
            "min_peaks":       3,
        },
    })

    agent = PeakDetectorAgent(args.sample_id, config=cfg)
    result = agent.run()
    print(result)
