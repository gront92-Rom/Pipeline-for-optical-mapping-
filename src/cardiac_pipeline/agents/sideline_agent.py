#!/usr/bin/env python3
"""
sideline_agent.py — Модуль-перехватчик для обработки длинных файлов (≥4096 кадров).
Активируется после загрузки (LoaderAgent). Если файл длинный, извлекает
центральный трейс 3х3, фильтрует его и генерирует текстовый гайд для исследователя,
затем сигнализирует о необходимости прервать основной пайплайн.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.sparse import diags, eye as speye
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig
from cardiac_pipeline.utils.preprocess import temporal_lowpass, should_invert

logger = logging.getLogger(__name__)

def generate_sideline_guide_text(filename: str, frames: int, fps: float, data_filename: str) -> str:
    duration_s = frames / fps if fps else 0
    return f"""========================================================================
                      САЙДЛАЙН-ГАЙД ДЛЯ ДЛИННОГО ФАЙЛА
========================================================================
Файл: {filename}
Параметры: {frames} кадров | FPS: {fps} Гц | Длительность: {duration_s:.2f} сек.

Данный файл автоматически исключен из стандартного пространственного 
пайплайна (Stage 1-7), так как его длина превышает лимит в 4096 кадров. 
Стандартный анализ на таких объемах неоптимален и подвержен шумам.

Центральный трейс 3х3 успешно извлечен, усреднен и отфильтрован (Butterworth 4-го порядка, 80 Гц).
Файл с данными трейса: {data_filename}

ЧТО ДЕЛАТЬ С ЭТИМ ФАЙЛОМ ДАЛЬШЕ? (Варианты для анализа в Jupyter / обсуждения):

1. АНАЛИЗ ФОТОБЛИЧИНГА И ДРИФТА БАЗОВОЙ ЛИНИИ (Baseline Drift)
   Сигнал в длинных файлах часто "плывет" вниз из-за выгорания красителя.
   -> Рекомендация: Загрузите трейс и примените алгоритм Asymmetric Least Squares 
      из модуля `asls_baseline` (уже импортирован в воркере), чтобы выровнять тренд.

2. СПЕКТРАЛЬНЫЙ АНАЛИЗ АЛЬТЕРНАНСОВ (Alternans Map)
   Если этот файл записывался при высокой частоте стимуляции для вызова альтернансов:
   -> Рекомендация: Натравите на `filtered_trace` метод FFT (scipy.fft.rfft). 
      Ищите появление выраженного пика на частоте (0.5 * Частота_Стимуляции_Гц).

3. ОЦЕНКА ДЕГРАДАЦИИ ВО ВРЕМЕНИ (Усталость ткани / Токсичность)
   -> Рекомендация: Запустите пиковый детектор по всей длине сигнала. Постройте 
      зависимости:
      а) Амплитуда каждого пика (Beat Amplitude) от времени.
      б) Длительность потенциала действия (APD) от времени.
      Посмотрите, падает ли амплитуда или затягивается ли APD к концу записи.

Для кастомной обработки загрузите данные в Python:
>>> import numpy as np
>>> data = np.load("{data_filename}")
>>> filtered_signal = data['filtered_trace']
========================================================================
"""

class SidelineAgent(BaseAgent):
    """
    SidelineAgent: Проверяет длину видео, извлекает и фильтрует центральный трейс 
    для длинных файлов (>=4096 кадров), генерирует инструкцию для ручного анализа.
    """

    DEPENDS_ON: list = []  # [LoaderAgent] — установлен ниже (lazy import)
    REQUIRED_INPUTS: list = ["raw_video.npy", "metadata.json"]

    def __init__(
        self,
        sample_id: str,
        config: Optional[PipelineConfig] = None,
    ):
        super().__init__(sample_id, config)
        self.frame_limit = 4096
        # Берем частоту среза для фильтра из конфига, по умолчанию 80 Гц (как в гайде)
        self.cutoff_hz = 80.0 
        
    def _load_metadata(self) -> Dict[str, Any]:
        return self.load_must("metadata.json")

    def _load_video(self) -> np.ndarray:
        return self.load_must("raw_video.npy")

    def _maybe_invert(self, trace: np.ndarray, dye: Optional[str]) -> np.ndarray:
        """Invert trace so AP peaks point up (positive deflection)."""
        if dye and dye.upper().startswith("A"):
            # VSD: AP is downward deflection in raw signal → invert
            return -trace
        # Calcium (B) already upward; voltage if unsure also invert for safety
        return -trace if dye is None else trace

    def _extract_center_trace(self, video: np.ndarray) -> np.ndarray:
        T, H, W = video.shape
        cy, cx = H // 2, W // 2
        y0, y1 = max(0, cy - 1), min(H, cy + 2)
        x0, x1 = max(0, cx - 1), min(W, cx + 2)
        return np.mean(video[:, y0:y1, x0:x1], axis=(1, 2)).astype(np.float32)

    def _filter_trace(self, trace: np.ndarray, fps: float) -> np.ndarray:
        trace_3d = trace[:, np.newaxis, np.newaxis]
        filtered_3d = temporal_lowpass(trace_3d, mask=None, fps=fps, cutoff=self.cutoff_hz)
        return filtered_3d.squeeze()

    def _asls_baseline(self, y: np.ndarray, lam: float = 1e7, p: float = 0.01,
                       niter: int = 3) -> np.ndarray:
        """Asymmetric Least Squares baseline (Eilers & Boelens).

        Parameters
        ----------
        y : 1D signal (already polarity-corrected, AP peaks point up)
        lam : smoothness (1e7 for 1D optical traces @ 500-1000 fps)
        p : asymmetry (0.01 = baseline hugs lower envelope)
        niter : 3 is sufficient for convergence on optical traces

        Returns
        -------
        baseline : 1D array, same length as y
        """
        L = len(y)
        # Second-order difference operator D (L-2, L)
        D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L)).tocsr()
        # Penalty matrix: lam * D.T @ D
        P = (D.T @ D) * lam
        w = np.ones(L)
        baseline = np.zeros(L)
        for _ in range(niter):
            W = sp.diags(w, 0, shape=(L, L)).tocsc()
            Z = W + P
            baseline = spsolve(Z, w * y)
            w = p * (y > baseline) + (1 - p) * (y <= baseline)
        return baseline

    def _detect_peaks(self, trace: np.ndarray, fps: float, stim_hz: Optional[float] = None) -> np.ndarray:
        """Simple adaptive peak detector on sideline trace.

        Uses stim_hz from metadata if available; otherwise assumes 16 Hz
        (midpoint of typical 10-20 Hz pacing range for long recordings).
        """
        from scipy.signal import find_peaks
        # Effective pacing frequency: metadata, else 16 Hz default
        eff_hz = stim_hz if (stim_hz and stim_hz > 0) else 16.0
        min_dist = int(0.6 * fps / eff_hz)
        # Prominence: robust estimate based on trace IQR
        q25, q75 = np.percentile(trace, [25, 75])
        iqr = q75 - q25
        prominence = max(iqr * 0.5, np.std(trace) * 0.3)
        peaks, _ = find_peaks(trace, distance=min_dist, prominence=prominence)
        return peaks

    def _compute_rr_stats(self, peaks: np.ndarray, fps: float) -> Dict[str, Any]:
        if len(peaks) < 2:
            return {"n_peaks": len(peaks), "regular": False, "reason": "too few peaks"}
        rr = np.diff(peaks) / fps * 1000.0  # ms
        rr_mean = float(np.mean(rr))
        rr_std = float(np.std(rr))
        rr_cv = rr_std / rr_mean if rr_mean > 0 else float("inf")
        # Regularity: fraction of intervals within ±15% of median
        median_rr = float(np.median(rr))
        within_band = np.sum(np.abs(rr - median_rr) <= 0.15 * median_rr)
        regularity_score = float(within_band / len(rr))
        is_regular = regularity_score >= 0.8 and rr_cv < 0.2
        return {
            "n_peaks": len(peaks),
            "rr_mean_ms": round(rr_mean, 2),
            "rr_std_ms": round(rr_std, 2),
            "rr_cv": round(rr_cv, 4),
            "regularity_score": round(regularity_score, 4),
            "is_regular": bool(is_regular),
        }

    def _compute_dominant_freq(self, trace: np.ndarray, fps: float) -> float:
        from scipy.fft import rfft, rfftfreq
        n = len(trace)
        freqs = rfftfreq(n, d=1.0 / fps)
        spectrum = np.abs(rfft(trace - np.mean(trace)))
        # Ignore DC; pick peak in 0.5–30 Hz physiologic range
        valid = (freqs > 0.5) & (freqs < 30.0)
        if not np.any(valid):
            return 0.0
        idx = np.argmax(spectrum[valid])
        return float(freqs[valid][idx])

    def _segment_trace(self, trace: np.ndarray, peaks: np.ndarray, fps: float) -> List[Dict[str, Any]]:
        """Split trace into beat-to-beat or fixed windows and tag regularity."""
        segments = []
        if len(peaks) < 2:
            return segments
        for i in range(len(peaks) - 1):
            start = int(peaks[i])
            end = int(peaks[i + 1])
            interval_ms = (end - start) / fps * 1000.0
            segments.append({
                "start_frame": start,
                "end_frame": end,
                "interval_ms": round(interval_ms, 2),
                "regular": None,  # will be filled by caller
            })
        # Fill regularity: compare each interval to median
        intervals = [s["interval_ms"] for s in segments]
        med = float(np.median(intervals))
        for s in segments:
            s["regular"] = bool(abs(s["interval_ms"] - med) <= 0.15 * med)
        return segments

    def _save_trace_png(self, trace: np.ndarray, peaks: np.ndarray, segments: List[Dict[str, Any]], fps: float, dye: str) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t_ms = np.arange(len(trace)) / fps * 1000.0
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t_ms, trace, lw=0.7, label="trace")
        ax.plot(t_ms[peaks], trace[peaks], "r.", markersize=6, label=f"peaks={len(peaks)}")
        # shade regular (green) vs irregular (red) segments
        for seg in segments:
            color = "green" if seg["regular"] else "red"
            ax.axvspan(seg["start_frame"] / fps * 1000.0, seg["end_frame"] / fps * 1000.0,
                       color=color, alpha=0.08)
        ax.set_xlabel("time [ms]")
        ax.set_ylabel("signal")
        ax.set_title(f"Sideline trace — {self.sample_id} | dye={dye} | regular shaded green")
        ax.legend()
        png_path = self.get_path("sideline_trace.png", kind="must")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return png_path

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Запускает проверку файла. 
        Возвращает:
            status: "sideline_isolated" если файл длинный, "pass" если файл обычный.
        """
        self.logger.info("Running SidelineAgent check...")

        # --- Lazy: запускаем Loader если raw_video.npy отсутствует ---
        from cardiac_pipeline.agents.loader_agent import LoaderAgent
        self.DEPENDS_ON = [LoaderAgent]
        self.ensure_dependencies(force=force)

        try:
            video = self._load_video()
            metadata = self._load_metadata()
        except FileNotFoundError as e:
            self.logger.error(f"Required files missing: {e}")
            raise

        nt, H, W = video.shape
        fps = metadata.get("fps", 0.0)
        stim_hz = metadata.get("stim_hz") or metadata.get("stim_hz_effective")
        dye = metadata.get("dye")

        if nt < self.frame_limit:
            self.logger.info(f"Video length ({nt} frames) is within limits. Passing to main pipeline.")
            return {
                "status": "pass",
                "frames": nt
            }

        self.logger.warning(f"[SIDELINE DETECTION] Long file detected: {nt} frames (threshold >= {self.frame_limit}).")
        self.logger.warning("[SIDELINE DETECTION] Standard spatial pipeline will be skipped.")

        # 1. Extract central 3x3 trace
        mean_trace = self._extract_center_trace(video)

        # 2. Invert so AP peaks point up
        oriented_trace = self._maybe_invert(mean_trace, dye)

        # 3. ASLS baseline correction (removes photobleaching drift)
        baseline = self._asls_baseline(oriented_trace, lam=1e5, p=0.01, niter=3)
        bc_trace = oriented_trace - baseline
        wander = float(baseline.max() - baseline.min())
        self.logger.info(f"[SIDELINE] ASLS baseline: wander={wander:.1f} a.u.")

        # 4. Filter (Butterworth 80 Hz on baseline-corrected trace)
        filtered_trace = self._filter_trace(bc_trace, fps)

        # 4. Detect peaks
        peaks = self._detect_peaks(filtered_trace, fps, stim_hz)

        # 5. Regularity / FFT
        rr_stats = self._compute_rr_stats(peaks, fps)
        dominant_freq = self._compute_dominant_freq(filtered_trace, fps)

        # 6. Segment
        segments = self._segment_trace(filtered_trace, peaks, fps)

        # 7. Save artifacts
        sideline_data = {
            "raw_mean_trace": mean_trace,
            "oriented_trace": oriented_trace,
            "asls_baseline": baseline,
            "bc_trace": bc_trace,
            "filtered_trace": filtered_trace,
            "peaks": peaks,
        }
        data_path = self.get_path("sideline_trace.npz", kind="must")
        np.savez_compressed(data_path, **sideline_data)
        self.logger.info(f"[MUST] Saved: {data_path.name}")

        png_path = self._save_trace_png(filtered_trace, peaks, segments, fps, dye or "?")

        sideline_metrics = {
            "frames": nt,
            "fps": fps,
            "dye": dye,
            "stim_hz": stim_hz,
            "dominant_freq_hz": round(dominant_freq, 2),
            "sideline_activated": True,
            "asls_wander": round(wander, 2),
            **rr_stats,
        }
        self.save_must(sideline_metrics, "sideline_metrics.json")
        self.save_must(segments, "sideline_segments.json")

        # 8. Human-in-the-loop request: ask user to pick segment(s)
        request = {
            "sample_id": self.sample_id,
            "status": "awaiting_user_decision",
            "message": (
                f"Long recording ({nt} frames). Standard pipeline skipped. "
                f"Detected {len(peaks)} peaks, dominant freq={dominant_freq:.2f} Hz. "
                f"See sideline_trace.png and sideline_segments.json. "
                "Please select segment(s) to analyze, or run full-file custom analysis."
            ),
            "options": [
                {"id": "auto_regular", "label": "Analyze the longest regular segment automatically"},
                {"id": "all_regular", "label": "Analyze all regular segments"},
                {"id": "custom", "label": "Provide custom frame range(s) in sideline_decision.json"},
                {"id": "skip", "label": "Skip this sample"},
            ],
            "segments": segments,
        }
        req_path = self.save_must(request, "sideline_decision_request.json")

        self.logger.info(f"[SIDELINE] Trace saved: {data_path.name}")
        self.logger.info(f"[SIDELINE] PNG saved: {png_path.name}")
        self.logger.info(f"[SIDELINE] Decision request: {req_path.name}")

        return {
            "status": "sideline",
            "frames": nt,
            "n_peaks": len(peaks),
            "dominant_freq_hz": round(dominant_freq, 2),
            "is_regular": rr_stats.get("is_regular", False),
            "data_path": str(data_path),
            "png_path": str(png_path),
            "request_path": str(req_path),
            "metrics": sideline_metrics,
            "segments": segments,
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SidelineAgent standalone")
    parser.add_argument("sample_id", help="Sample ID")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()

    cfg = PipelineConfig()
    cfg.results_root = Path(args.results_root)
    agent = SidelineAgent(args.sample_id, config=cfg)
    print(agent.run())
