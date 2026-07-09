#!/usr/bin/env python3
"""
ActivationAgent — Activation Time Map Agent (Consumer of PeakDetectorAgent)

Calculates per-pixel activation times using a method cascade:
  1. threshold_50pct  — 50% upstroke crossing
  2. derivative_max   — max dV/dt
  3. vectorized_interp — vectorized threshold interpolation

Key design (July 2026):
  - Takes preproc_video + peaks + mask from PeakDetectorAgent
  - Calculates up to 5 per-beat activation maps (skip first beat)
  - Saves median map + per-beat maps in MUST
  - If quality is low (WARN) → retries with larger analysis window
  - REJECT → raises ValueError (strict gating)

Inputs (lazy):
  - debug/preproc_video.npy  (from PeakDetectorAgent)
  - must/peaks.npy           (from PeakDetectorAgent)
  - must/mask.npy            (from MaskAgent)
  - must/metadata.json       (from LoaderAgent)

Outputs:
  - must/activation_map.npy         — median activation time map (ms)
  - must/per_beat_activation.npy    — stacked per-beat maps (up to 5)
  - must/activation_report.json     — verdict, metrics, params
  - debug/activation_debug.json     — per-method attempts

Исправления при интеграции (2026-07-02):
  - Удалён inline BaseAgent-стаб → cardiac_pipeline.base_agent
  - process(sample_name) → run(force=False) (BaseAgent API)
  - load_must(sample_name, file) → self.load_must(file) (BaseAgent API)
  - save_must(sample_name, file, data) → self.save_must(data, file) (BaseAgent API)
  - save_debug(sample_name, file, data) → self.save_debug(data, file) (BaseAgent API)
  - invoke_agent() → direct import + .run() (lazy upstream)
  - fps fallback 1000.0 → raise ValueError (AG1 fix)
  - reject() → raise ValueError (strict gating)
  - preproc_video.npy: loaded from debug/ (where PeakDetector saves it)
  - logger → self.logger (BaseAgent convention)
  - window_offset читается из config.activation
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig


# ===========================================================================
# Core activation calculation functions
# ===========================================================================

def activation_threshold_50pct(
    data_inv: np.ndarray,
    mask: np.ndarray,
    peaks: np.ndarray,
    fps: float,
    stim_hz: float = 10.0,
    window_offset: int = 50,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Per-pixel activation via 50% upstroke crossing."""
    H, W = mask.shape
    n_peaks = len(peaks)
    per_beat: List[np.ndarray] = []
    inner_peaks = peaks[1:1 + min(5, n_peaks - 2)] if n_peaks > 2 else peaks[:min(5, n_peaks)]

    for pk in inner_peaks:
        ws = max(0, int(pk) - window_offset)
        we = int(pk) + 30
        am = np.full((H, W), np.nan)
        for py, px in np.argwhere(mask):
            seg = data_inv[ws:we, py, px]
            if len(seg) < 5:
                continue
            bl_local_end = int(pk) - ws
            bl_local_start = max(0, bl_local_end - 20)
            bl = seg[bl_local_start:bl_local_end].mean() if bl_local_end > bl_local_start else seg[0]
            pk_local = int(seg.argmax())
            amp = seg[pk_local] - bl
            if amp < 0.1:
                continue
            thr_50 = bl + 0.5 * amp
            for j in range(pk_local, 0, -1):
                if seg[j] < thr_50:
                    am[py, px] = (ws + j + 1) / fps * 1000.0
                    break
        # Shift to earliest = 0
        if mask.any() and np.any(np.isfinite(am[mask])):
            am = am - np.nanmin(am[mask])
        per_beat.append(am)

    if len(per_beat) == 0:
        return np.full((H, W), np.nan), per_beat

    median_map = np.nanmedian(per_beat, axis=0)
    if mask.any() and np.any(np.isfinite(median_map[mask])):
        median_map = median_map - np.nanmin(median_map[mask])
    return median_map, per_beat


def activation_derivative_max(
    data_inv: np.ndarray,
    mask: np.ndarray,
    peaks: np.ndarray,
    fps: float,
    stim_hz: float = 10.0,
    window_offset: int = 50,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Per-pixel activation via max |dV/dt|."""
    H, W = mask.shape
    n_peaks = len(peaks)
    per_beat: List[np.ndarray] = []
    inner_peaks = peaks[1:1 + min(5, n_peaks - 2)] if n_peaks > 2 else peaks[:min(5, n_peaks)]

    for pk in inner_peaks:
        ws = max(0, int(pk) - window_offset)
        we = int(pk) + 30
        am = np.full((H, W), np.nan)
        for py, px in np.argwhere(mask):
            tr = data_inv[ws:we, py, px]
            if len(tr) < 3:
                continue
            dt = np.abs(np.diff(tr))
            act_local = int(np.argmax(dt))
            am[py, px] = (act_local - (int(pk) - ws)) / fps * 1000.0
        if mask.any() and np.any(np.isfinite(am[mask])):
            am = am - np.nanmin(am[mask])
        per_beat.append(am)

    if len(per_beat) == 0:
        return np.full((H, W), np.nan), per_beat
    median_map = np.nanmedian(per_beat, axis=0)
    if mask.any() and np.any(np.isfinite(median_map[mask])):
        median_map = median_map - np.nanmin(median_map[mask])
    return median_map, per_beat


def activation_vectorized_interp(
    data_inv: np.ndarray,
    mask: np.ndarray,
    peaks: np.ndarray,
    fps: float,
    stim_hz: float = 10.0,
    window_offset: int = 50,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Vectorized 50% threshold interpolation (faster for large masks)."""
    H, W = mask.shape
    n_peaks = len(peaks)
    per_beat: List[np.ndarray] = []
    inner_peaks = peaks[1:1 + min(5, n_peaks - 2)] if n_peaks > 2 else peaks[:min(5, n_peaks)]
    dt_ms = 1000.0 / fps

    for pk in inner_peaks:
        ws = max(0, int(pk) - window_offset)
        we = int(pk) + 30
        seg = data_inv[ws:we]
        if len(seg) < 3:
            per_beat.append(np.full((H, W), np.nan))
            continue

        pk_local = min(int(pk) - ws, len(seg) - 1)
        search_before = seg[:pk_local + 1]
        baseline = search_before.min(axis=0)
        peak_val = seg.max(axis=0)
        amplitude = peak_val - baseline
        threshold = baseline + 0.5 * amplitude

        act_sub = np.full((H, W), np.nan)
        for y in range(H):
            for x in range(W):
                if not mask[y, x]:
                    continue
                bl_idx = int(np.argmin(search_before[:, y, x]))
                thr = threshold[y, x]
                found = -1
                for i in range(bl_idx + 1, pk_local + 1):
                    if seg[i, y, x] >= thr:
                        found = i
                        break
                if found > 0:
                    act_sub[y, x] = found

        act_ms = (act_sub - pk_local) * dt_ms
        act_ms[~mask] = np.nan
        if mask.any() and np.any(np.isfinite(act_ms[mask])):
            act_ms = act_ms - np.nanmin(act_ms[mask])
        per_beat.append(act_ms)

    if len(per_beat) == 0:
        return np.full((H, W), np.nan), per_beat
    median_map = np.nanmedian(per_beat, axis=0)
    if mask.any() and np.any(np.isfinite(median_map[mask])):
        median_map = median_map - np.nanmin(median_map[mask])
    return median_map, per_beat


def judge_activation(
    act_map: np.ndarray,
    mask: np.ndarray,
    per_beat: List[np.ndarray],
    stim_hz: float,
    fps: float,
    beat_pass_rate_reject: float = 0.3,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Judge activation map quality.
    Returns (verdict, reason, metrics).
    verdict: PASS / WARN / REJECT
    """
    if act_map is None or not mask.any():
        return "REJECT", "empty map", {}

    if not np.any(np.isfinite(act_map[mask])):
        return "REJECT", "all NaN in mask", {}

    metrics: Dict[str, Any] = {}
    tat = float(np.nanmax(act_map[mask]) - np.nanmin(act_map[mask]))
    metrics["tat_ms"] = round(tat, 1)
    metrics["tat_std"] = round(float(np.nanstd(act_map[mask])), 1)

    valid_px = int(np.sum(np.isfinite(act_map[mask])))
    metrics["valid_coverage"] = round(float(valid_px / mask.sum()), 4)
    metrics["beats_used"] = len(per_beat)

    beat_tats: List[float] = []
    BCL = 1000.0 / stim_hz if stim_hz > 0 else 100.0
    for am in per_beat:
        if am is not None and np.any(np.isfinite(am[mask])):
            bt = float(np.nanmax(am[mask]) - np.nanmin(am[mask]))
            beat_tats.append(bt)

    metrics["beat_tats"] = [round(t, 1) for t in beat_tats]
    metrics["beat_pass_rate"] = 0.0
    if beat_tats:
        passed = sum(1 for t in beat_tats if t < 0.5 * BCL)
        metrics["beat_pass_rate"] = round(float(passed / len(beat_tats)), 3)

    if tat <= 0:
        return "REJECT", f"TAT={metrics['tat_ms']} <= 0", metrics
    if metrics["beat_pass_rate"] < beat_pass_rate_reject:
        return "REJECT", f"beat_pass_rate={metrics['beat_pass_rate']} < {beat_pass_rate_reject}", metrics
    if metrics["valid_coverage"] < 0.5:
        return "WARN", f"valid_coverage={metrics['valid_coverage']} < 0.5", metrics

    return "PASS", "OK", metrics


# ===========================================================================
# ActivationAgent
# ===========================================================================

class ActivationAgent(BaseAgent):
    """
    ActivationAgent — calculates activation time maps from preprocessed data.

    Uses a method cascade (threshold_50pct → derivative_max → vectorized_interp)
    and retries with larger window on WARN.
    """

    DEPENDS_ON: list = []  # [PeakDetectorAgent] — установлен ниже (lazy import)
    REQUIRED_INPUTS: list = ["peaks.npy", "mask.npy"]

    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None):
        super().__init__(sample_id, config)

        act_cfg = getattr(self.config, 'activation', {}) or {}
        self.window_offset       = int(act_cfg.get('window_offset', 50))
        self.retry_window_offset = int(act_cfg.get('retry_window_offset', 80))
        self.beat_pass_rate_reject = float(act_cfg.get('beat_pass_rate_reject', 0.3))
        self.min_peaks           = int(act_cfg.get('min_peaks', 3))

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
        fps = self.metadata.get("fps") or self.metadata.get("fps_hz")
        if fps is None:
            raise ValueError(
                "fps отсутствует в metadata.json. "
                "LoaderAgent должен сохранить его заранее."
            )
        fps = float(fps)
        if fps <= 0:
            raise ValueError(f"fps некорректен (fps={fps})")
        return fps

    def _get_stim_hz(self) -> float:
        stim = self.metadata.get("stim_hz") or self.metadata.get("pacing_hz")
        if stim is not None:
            return float(stim)
        self.logger.warning("stim_hz not found in metadata — using 10.0 Hz")
        return 10.0

    def _load_preproc_video(self) -> np.ndarray:
        """Load preproc_video.npy from debug/ (where PeakDetector saves it)."""
        path = self.get_path("preproc_video.npy", kind="debug")
        if not path.exists():
            raise FileNotFoundError(
                f"preproc_video.npy not found at {path}. "
                "PeakDetectorAgent should have produced it."
            )
        return np.load(path)

    # ==================== ACTIVATION CALCULATION ====================

    def _calculate_activation(
        self,
        data_inv: np.ndarray,
        mask: np.ndarray,
        peaks: np.ndarray,
        fps: float,
        stim_hz: float,
        window_offset: int = 50,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Try methods in cascade order. Return best result."""
        methods = [
            ("threshold_50pct",   activation_threshold_50pct),
            ("derivative_max",    activation_derivative_max),
            ("vectorized_interp", activation_vectorized_interp),
        ]

        best_median: Optional[np.ndarray] = None
        best_per_beat: List[np.ndarray] = []
        best_verdict = "REJECT"

        for name, fn in methods:
            try:
                median_map, per_beat = fn(
                    data_inv, mask, peaks,
                    fps=fps, stim_hz=stim_hz, window_offset=window_offset,
                )
                verdict, _, _ = judge_activation(
                    median_map, mask, per_beat, stim_hz, fps,
                    beat_pass_rate_reject=self.beat_pass_rate_reject,
                )
                self.logger.info(f"  Method {name}: {verdict}")
                if verdict == "PASS":
                    return median_map, per_beat
                if verdict == "WARN" and best_verdict != "PASS":
                    best_median, best_per_beat, best_verdict = median_map, per_beat, verdict
            except Exception as e:
                self.logger.warning(f"  Method {name} failed: {e}")
                continue

        if best_median is not None:
            return best_median, best_per_beat

        # All methods failed
        H, W = mask.shape
        return np.full((H, W), np.nan), []

    # ==================== RUN ====================

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Main entry point.

        1. Skip if activation_map.npy exists and force=False
        2. Ensure PeakDetectorAgent ran (lazy upstream)
        3. Load preproc_video, peaks, mask, metadata
        4. First pass: method cascade with default window
        5. Retry with larger window if WARN + low coverage
        6. REJECT → raise ValueError (strict gating)
        7. Save artifacts
        """
        if not force and self.exists("activation_map.npy"):
            self.logger.info("activation_map.npy exists, skipping")
            return {"status": "skipped"}

        t0 = time.perf_counter()

        # --- Lazy: запускаем PeakDetector (→ Loader → Mask) если выходы отсутствуют ---
        from cardiac_pipeline.agents.peak_detector_agent import PeakDetectorAgent
        self.DEPENDS_ON = [PeakDetectorAgent]
        self.ensure_dependencies(force=force)

        # --- 2. Load metadata ---
        self._load_metadata()

        fps     = self._get_fps()
        stim_hz = self._get_stim_hz()

        # --- 3. Load inputs ---
        preproc_video = self._load_preproc_video()
        peaks = self.load_must("peaks.npy")
        mask  = self.load_must("mask.npy").astype(bool)

        self.logger.info(
            f"Loaded: video={preproc_video.shape}, peaks={len(peaks)}, "
            f"mask_cov={mask.mean():.3f}, fps={fps}, stim={stim_hz}"
        )

        if len(peaks) < self.min_peaks:
            raise ValueError(
                f"Too few peaks ({len(peaks)}). Minimum {self.min_peaks} required."
            )

        # --- 4. First pass ---
        self.logger.info(f"First pass (window_offset={self.window_offset})")
        median_map, per_beat = self._calculate_activation(
            preproc_video, mask, peaks, fps, stim_hz,
            window_offset=self.window_offset,
        )
        verdict, reason, metrics = judge_activation(
            median_map, mask, per_beat, stim_hz, fps,
            beat_pass_rate_reject=self.beat_pass_rate_reject,
        )
        self.logger.info(f"First pass verdict: {verdict} ({reason})")

        # --- 5. Retry with larger window if WARN ---
        if verdict == "WARN" and metrics.get("valid_coverage", 1.0) < 0.5:
            self.logger.info(
                f"Low coverage. Retrying with window_offset={self.retry_window_offset}"
            )
            median_map2, per_beat2 = self._calculate_activation(
                preproc_video, mask, peaks, fps, stim_hz,
                window_offset=self.retry_window_offset,
            )
            verdict2, reason2, metrics2 = judge_activation(
                median_map2, mask, per_beat2, stim_hz, fps,
                beat_pass_rate_reject=self.beat_pass_rate_reject,
            )
            if verdict2 == "PASS" or (
                verdict2 == "WARN" and
                metrics2.get("valid_coverage", 0) > metrics.get("valid_coverage", 0)
            ):
                median_map, per_beat = median_map2, per_beat2
                verdict, reason, metrics = verdict2, reason2, metrics2
                self.logger.info(f"Retry improved: {verdict}")

        # --- 6. Strict gating ---
        if verdict == "REJECT":
            raise ValueError(
                f"Activation map rejected: {reason}. "
                f"Sample {self.sample_id} requires manual review."
            )

        # --- 7. Save artifacts ---
        self.save_must(median_map, "activation_map.npy")

        if per_beat:
            per_beat_stack = np.stack(per_beat[:5], axis=0) if len(per_beat) > 1 else per_beat[0]
            self.save_must(per_beat_stack, "per_beat_activation.npy")

        elapsed = round(time.perf_counter() - t0, 2)
        report = {
            "sample_id":    self.sample_id,
            "fps":          fps,
            "stim_hz":      stim_hz,
            "n_peaks_used": len(per_beat),
            "verdict":      verdict,
            "reason":       reason,
            "metrics":      metrics,
            "window_offset": self.window_offset,
            "elapsed_s":    elapsed,
        }
        self.save_must(report, "activation_report.json")

        # Debug: per-method attempt log
        self.save_debug({
            "final_verdict": verdict,
            "final_reason":  reason,
            "metrics":       metrics,
        }, "activation_debug.json")

        self._log_metrics({**metrics, "verdict": verdict, "elapsed_s": elapsed})
        self.logger.info(f"Finished in {elapsed}s — {verdict}")

        return {
            "status":  "success",
            "verdict": verdict,
            "metrics": metrics,
        }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ActivationAgent standalone")
    parser.add_argument("sample_id", help="Sample ID (e.g. 005A)")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()

    cfg = PipelineConfig({"results_root": args.results_root})
    agent = ActivationAgent(args.sample_id, config=cfg)
    result = agent.run()
    print(result)
