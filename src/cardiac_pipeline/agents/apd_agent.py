#!/usr/bin/env python3
"""
apd_agent.py — Stage 6: Расчёт длительности потенциала действия / CaT (APD/CaT Map).
Версия v1 (2026-07-02).

Архитектура:
  Наследует BaseAgent. Читает параметры из PipelineConfig (config/default.yaml).
  Все пути — через self.get_path() / BaseAgent API.
  Вся математика вынесена в utils/signal.py.

Входные данные (lazy — запускает ActivationAgent если нужно):
  - debug/preproc_video_apd.npy  — видео, препроцессированное с target_stage="apd" (150 Гц LPF)
    Если отсутствует — использует debug/preproc_video.npy от PeakDetectorAgent.
  - must/peaks.npy               — глобальные пики биений (от PeakDetectorAgent)
  - must/mask.npy                — маска ткани (от MaskAgent)
  - must/metadata.json           — fps, dye (от LoaderAgent)

Выходные данные:
  MUST:
    - apd30_map.npy / cat30_map.npy   — медианная карта APD30/CaT30 (мс)
    - apd50_map.npy / cat50_map.npy   — медианная карта APD50/CaT50 (мс)
    - apd80_map.npy / cat80_map.npy   — медианная карта APD80/CaT80 (мс)
    - apd_per_beat_3d.npz             — 3D стек (H, W, N_beats) для Stage 7 (Альтернанс)
    - apd_report.json                 — вердикт, метрики, параметры
  DEBUG:
    - apd30_map.png / apd50_map.png / apd80_map.png — PNG-карты
    - apd_traces.png                  — диагностические трейсы 4 угловых ROI
    - apd_debug.json                  — детали QC

Коды возврата (CLI-режим):
  0 = SUCCESS
  1 = CRASH (исключение инфраструктуры)
  2 = REJECT (QC не пройден)

Исправления относительно исходного apd_agent.py:
  - from utils_apd import ... → from cardiac_pipeline.utils.signal import ...
  - fps берётся из metadata.json через _get_fps() (не из --fps аргумента)
  - dye берётся из metadata.json через _get_dye() (не из metadata.get("dye", "A"))
  - Параметры (min_amplitude, qc_min_coverage) берутся из PipelineConfig / config.yaml
  - Все пути через BaseAgent API (must_dir / debug_dir)
  - validate_apd_semantics перенесена в utils/signal.py
  - Диапазоны VSD/CaT берутся из конфига (apd_min_ms, apd_max_ms)
  - Добавлен lazy-механизм: проверяет наличие preproc_video_apd.npy
  - Добавлен _ensure_upstream() для PeakDetectorAgent
  - Вердикт REJECT → raise ValueError (соответствует BaseAgent-контракту)
  - PNG-визуализация сохраняется в debug/ (не в must/)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig
from cardiac_pipeline.utils.signal import (
    masked_spatial_pool,
    find_upstroke_start,
    find_repol_crossing_with_fallback,
    get_4_corners_snapped,
    validate_apd_semantics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# APDAgent
# ---------------------------------------------------------------------------

class APDAgent(BaseAgent):
    """
    Stage 6: Расчёт карт APD/CaT и 3D стека биений.

    Потребляет препроцессированное видео + пики от PeakDetectorAgent.
    Генерирует карты APD30/50/80 (или CaT30/50/80) и 3D стек для Stage 7.
    """

    DEPENDS_ON: list = []  # [PeakDetectorAgent] — установлен ниже (lazy import)
    REQUIRED_INPUTS: list = ["peaks.npy", "mask.npy"]

    def __init__(
        self,
        sample_id: str,
        config: Optional[PipelineConfig] = None,
    ):
        super().__init__(sample_id, config)

        apd_cfg = self.config.apd if isinstance(self.config.apd, dict) else {}

        # Параметры из конфига
        self.min_amplitude:   float = float(apd_cfg.get("min_amplitude",   0.001))
        self.qc_min_coverage: float = float(apd_cfg.get("qc_min_coverage", 0.25))
        self.roi_pool_size:   int   = int(apd_cfg.get("roi_pool_size",     3))
        self.apd_min_ms:      float = float(apd_cfg.get("apd_min_ms",      5.0))
        self.apd_max_ms:      float = float(apd_cfg.get("apd_max_ms",      500.0))

        # Метаданные (заполняются в run())
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = self.get_path("metadata.json", kind="must")
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
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
                "Запустите LoaderAgent для извлечения метаданных из .gsh/.rsh."
            )
        fps = float(fps)
        if fps <= 0:
            raise ValueError(f"fps некорректен (fps={fps})")
        return fps

    def _get_dye(self) -> str:
        """Возвращает 'A' (VSD/вольтаж) или 'B' (Ca²⁺). Обязательно из метаданных."""
        dye = self.metadata.get("dye") or self.metadata.get("recording_mode")
        if dye is None:
            self.logger.warning(
                "dye не найден в metadata.json — используется дефолт 'A' (VSD). "
                "Убедитесь, что LoaderAgent корректно парсит имя файла."
            )
            return "A"
        # Нормализация: "voltage"/"vsd"/"ap" → "A", "calcium"/"cat"/"ca" → "B"
        d = str(dye).upper().strip()
        if d in ("A", "VOLTAGE", "VSD", "AP"):
            return "A"
        if d in ("B", "CALCIUM", "CAT", "CA"):
            return "B"
        self.logger.warning(f"Неизвестный dye='{dye}', используется 'A' (VSD)")
        return "A"

    def _load_preproc_video(self) -> np.ndarray:
        """
        Загружает препроцессированное видео для APD (150 Гц LPF).

        Приоритет:
          1. debug/preproc_video_apd.npy  — специально препроцессированное для APD
          2. debug/preproc_video.npy      — стандартное от PeakDetectorAgent (80 Гц)
             В этом случае выдаёт WARNING: для APD рекомендуется 150 Гц LPF.
        """
        apd_path = self.get_path("preproc_video_apd.npy", kind="debug")
        if apd_path.exists():
            self.logger.info(f"Загружаю preproc_video_apd.npy (150 Гц LPF)")
            return np.load(apd_path)

        std_path = self.get_path("preproc_video.npy", kind="debug")
        if std_path.exists():
            self.logger.warning(
                "preproc_video_apd.npy не найден — использую preproc_video.npy (80 Гц LPF). "
                "Для точного APD рекомендуется запустить preprocess_video(..., target_stage='apd') "
                "и сохранить результат как debug/preproc_video_apd.npy."
            )
            return np.load(std_path)

        raise FileNotFoundError(
            f"Не найдено ни preproc_video_apd.npy, ни preproc_video.npy в {self.debug_dir}. "
            "Запустите PeakDetectorAgent."
        )

    # ------------------------------------------------------------------
    # Главный метод
    # ------------------------------------------------------------------

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Запускает расчёт APD/CaT карт.

        Порядок:
          1. Lazy-проверка upstream (PeakDetectorAgent)
          2. Загрузка метаданных → fps, dye
          3. Загрузка входных данных (видео, пики, маска)
          4. 3×3 ROI-пулинг под маской
          5. Отброс крайних биений (первое и последнее)
          6. Попиксельный цикл: апстрок + реполяризация × 3 уровня
          7. Медианные 2D карты + триангуляция
          8. Угловые ROI-кластеры (вариация биений)
          9. Семантическая валидация + QC-гейтинг
          10. Сохранение артефактов (npy, npz, json, png)
        """
        # Lazy-проверка: если карты уже есть — пропускаем
        if not force and self.exists("apd_report.json"):
            self.logger.info("apd_report.json exists, skipping")
            return {"status": "skipped"}

        t0 = time.perf_counter()

        # --- Lazy: запускаем PeakDetector (→ Loader → Mask) если выходы отсутствуют ---
        from cardiac_pipeline.agents.peak_detector_agent import PeakDetectorAgent
        self.DEPENDS_ON = [PeakDetectorAgent]
        self.ensure_dependencies(force=force)

        # --- 2. Метаданные ---
        self._load_metadata()

        fps = self._get_fps()
        dye = self._get_dye()
        metric = "APD" if dye == "A" else "CaT"

        # --- 3. Загрузка входных данных ---
        preproc_video = self._load_preproc_video()
        all_peaks = self.load_must("peaks.npy")
        mask = self.load_must("mask.npy").astype(bool)
        T, H, W = preproc_video.shape

        self.logger.info(
            f"APDAgent: video={preproc_video.shape}, peaks={len(all_peaks)}, "
            f"mask_cov={mask.mean():.3f}, fps={fps}, dye={dye} ({metric})"
        )

        # --- 4. 3×3 ROI-пулинг ---
        self.logger.info(f"3×3 ROI spatial pooling (mask-aware)...")
        video_roi = masked_spatial_pool(preproc_video, mask, size=self.roi_pool_size)

        # --- 5. Отброс крайних биений ---
        if len(all_peaks) >= 3:
            valid_peaks = all_peaks[1:-1]
            self.logger.info(
                f"Биений: {len(all_peaks)}, отброшены первое и последнее. "
                f"В анализе: {len(valid_peaks)}"
            )
        else:
            valid_peaks = all_peaks
            self.logger.warning(
                f"Критически мало биений ({len(all_peaks)}). Используем все."
            )

        n_beats = len(valid_peaks)

        # --- 6. Попиксельный цикл ---
        apd80_3d = np.full((H, W, n_beats), np.nan, dtype=np.float32)
        apd50_3d = np.full((H, W, n_beats), np.nan, dtype=np.float32)
        apd30_3d = np.full((H, W, n_beats), np.nan, dtype=np.float32)
        amp_3d   = np.full((H, W, n_beats), np.nan, dtype=np.float32)

        valid_pixels_count = 0
        fallback_used_count = 0

        for y in range(H):
            for x in range(W):
                if not mask[y, x]:
                    continue

                trace = video_roi[:, y, x]
                pixel_has_valid_beat = False

                for bi, pk in enumerate(valid_peaks):
                    pk = int(pk)
                    # Граница биения по полному массиву пиков
                    global_bi = int(np.where(all_peaks == pk)[0][0])
                    next_pk = int(all_peaks[global_bi + 1]) if global_bi + 1 < len(all_peaks) else T

                    seg = trace[pk:next_pk]
                    if len(seg) < 3:
                        continue

                    # Локальный пик сокращения
                    local_peak_idx = pk + int(np.argmax(seg))
                    amp = float(trace[local_peak_idx])

                    if amp < self.min_amplitude:
                        continue

                    up_idx, _, _ = find_upstroke_start(trace, local_peak_idx, amp, fps)
                    if up_idx is None:
                        continue

                    amp_3d[y, x, bi] = amp
                    pixel_has_valid_beat = True

                    # Реполяризация для трёх уровней
                    for threshold, arr3d in [(30, apd30_3d), (50, apd50_3d), (80, apd80_3d)]:
                        cross, found, status = find_repol_crossing_with_fallback(
                            trace, local_peak_idx, amp, fps,
                            threshold=threshold,
                            next_peak_idx=next_pk,
                            total_frames=T,
                        )
                        if found and cross is not None:
                            val_ms = (cross - up_idx) / fps * 1000.0
                            # Физиологический коридор
                            if self.apd_min_ms <= val_ms <= self.apd_max_ms:
                                arr3d[y, x, bi] = val_ms
                            if "fallback" in status:
                                fallback_used_count += 1

                if pixel_has_valid_beat:
                    valid_pixels_count += 1

        # --- 7. Медианные 2D карты ---
        apd80_map = np.nanmedian(apd80_3d, axis=2)
        apd50_map = np.nanmedian(apd50_3d, axis=2)
        apd30_map = np.nanmedian(apd30_3d, axis=2)
        triangulation_map = apd80_map - apd30_map

        # --- 8. Угловые ROI-кластеры ---
        corners = get_4_corners_snapped(mask, padding=10)
        corner_stats: List[Dict] = []
        for c in corners:
            cy, cx = c["y"], c["x"]
            beats_80 = apd80_3d[cy, cx, :]
            valid_b  = beats_80[np.isfinite(beats_80)]
            if len(valid_b) > 0:
                corner_stats.append({
                    "label":               c["label"],
                    "coords":              [cy, cx],
                    "APD80_median":        round(float(np.median(valid_b)), 2),
                    "APD80_std":           round(float(np.std(valid_b)), 2),
                    "APD80_iqr":           round(float(np.percentile(valid_b, 75) - np.percentile(valid_b, 25)), 2),
                    "Triangulation_median": round(float(triangulation_map[cy, cx]), 2)
                                           if np.isfinite(triangulation_map[cy, cx]) else None,
                })
            else:
                corner_stats.append({"label": c["label"], "status": "No valid beats captured"})

        # --- 9. Метрики и QC ---
        total_mask = int(np.sum(mask))
        acceptance = valid_pixels_count / total_mask if total_mask > 0 else 0.0

        valid_apd80_pixels = apd80_map[mask & np.isfinite(apd80_map)]
        apd80_spatial_med  = float(np.median(valid_apd80_pixels)) if len(valid_apd80_pixels) > 0 else float("nan")
        apd50_spatial_med  = float(np.nanmedian(apd50_map[mask])) if valid_pixels_count > 0 else float("nan")
        apd30_spatial_med  = float(np.nanmedian(apd30_map[mask])) if valid_pixels_count > 0 else float("nan")

        spatial_dispersion = (
            float(np.percentile(valid_apd80_pixels, 95) - np.percentile(valid_apd80_pixels, 5))
            if len(valid_apd80_pixels) > 0 else float("nan")
        )

        verdict, reason = validate_apd_semantics(
            apd80_spatial_med, apd30_spatial_med, dye,
            vsd_apd80_range=(self.apd_min_ms, self.apd_max_ms),
            cat_apd80_range=(self.apd_min_ms, self.apd_max_ms),
        )

        elapsed = round(time.perf_counter() - t0, 2)

        report = {
            "sample_id":               self.sample_id,
            "status":                  "SUCCESS",
            "metric":                  metric,
            "fps_used":                fps,
            "dye":                     dye,
            "valid_beats_used":        n_beats,
            "acceptance_rate":         round(acceptance, 4),
            "fallback_recoveries":     fallback_used_count,
            "spatial_stats": {
                f"{metric.lower()}80_median":             round(apd80_spatial_med, 1),
                f"{metric.lower()}50_median":             round(apd50_spatial_med, 1),
                f"{metric.lower()}30_median":             round(apd30_spatial_med, 1),
                "spatial_dispersion_repol_95_5":          round(spatial_dispersion, 1),
                "triangulation_tissue_median":            round(float(np.nanmedian(triangulation_map[mask])), 1),
            },
            "clusters_variation_analysis": corner_stats,
            "semantic_verdict":        verdict,
            "reason":                  reason,
            "elapsed_s":               elapsed,
        }

        # QC-гейтинг
        if acceptance < self.qc_min_coverage or verdict == "FAIL":
            report["status"] = "REJECT"
            report["reason"] = (
                reason if verdict == "FAIL"
                else f"Low valid tissue coverage ({acceptance:.1%} < {self.qc_min_coverage:.0%})"
            )
            self.save_must(report, "apd_report.json")
            self.logger.error(f"QC REJECT: {report['reason']}")
            raise ValueError(
                f"APD map rejected: {report['reason']}. "
                f"Sample {self.sample_id} requires manual review."
            )

        # --- 10. Сохранение артефактов ---
        m = metric.lower()
        self.save_must(apd30_map, f"{m}30_map.npy")
        self.save_must(apd50_map, f"{m}50_map.npy")
        self.save_must(apd80_map, f"{m}80_map.npy")

        # 3D стек для Stage 7 (Альтернанс)
        npz_path = self.must_dir / "apd_per_beat_3d.npz"
        np.savez_compressed(
            npz_path,
            apd80=apd80_3d, apd50=apd50_3d, apd30=apd30_3d, amp=amp_3d,
            n_beats=n_beats, metric=metric,
        )
        self.logger.info(f"[MUST] Saved: apd_per_beat_3d.npz")

        self.save_must(report, "apd_report.json")

        # Debug: PNG-карты и диагностические трейсы
        self._save_png_maps(apd30_map, apd50_map, apd80_map, mask, metric)
        self._save_diagnostic_traces(video_roi, valid_peaks, all_peaks, corners, fps, metric, T)

        self.save_debug({
            "acceptance_rate":     acceptance,
            "fallback_recoveries": fallback_used_count,
            "semantic_verdict":    verdict,
            "reason":              reason,
            "corner_stats":        corner_stats,
        }, "apd_debug.json")

        self._log_metrics({
            f"{m}80_median":   apd80_spatial_med,
            f"{m}50_median":   apd50_spatial_med,
            "acceptance_rate": acceptance,
            "verdict":         verdict,
            "elapsed_s":       elapsed,
        })

        self.logger.info(
            f"APDAgent done in {elapsed}s — {verdict}. "
            f"Coverage: {acceptance:.1%}, {metric}80 median: {apd80_spatial_med:.1f} ms"
        )

        return {
            "status":          "success",
            "verdict":         verdict,
            "acceptance_rate": acceptance,
            "metrics":         report["spatial_stats"],
        }

    # ------------------------------------------------------------------
    # Вспомогательные методы визуализации
    # ------------------------------------------------------------------

    def _save_png_maps(
        self,
        apd30_map: np.ndarray,
        apd50_map: np.ndarray,
        apd80_map: np.ndarray,
        mask: np.ndarray,
        metric: str,
    ):
        """Сохраняет PNG-карты APD30/50/80 в debug/."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            m = metric.lower()
            for lvl, data in [("30", apd30_map), ("50", apd50_map), ("80", apd80_map)]:
                plt.figure(figsize=(6, 5))
                masked_data = np.ma.masked_where(~mask, data)
                plt.imshow(masked_data, cmap="jet")
                plt.colorbar(label="ms")
                plt.title(f"Spatial Map: {metric}{lvl}")
                plt.tight_layout()
                png_path = self.debug_dir / f"{m}{lvl}_map.png"
                plt.savefig(png_path, dpi=150)
                plt.close()
                self.logger.info(f"[DEBUG] Saved: {png_path.name}")

        except Exception as e:
            self.logger.warning(f"PNG map generation skipped: {e}")

    def _save_diagnostic_traces(
        self,
        video_roi: np.ndarray,
        valid_peaks: np.ndarray,
        all_peaks: np.ndarray,
        corners: list,
        fps: float,
        metric: str,
        T: int,
    ):
        """Сохраняет диагностические трейсы 4 угловых ROI с разметкой ориентиров в debug/."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(13, 10))
            axes = axes.flatten()

            for idx, c in enumerate(corners):
                cy, cx = c["y"], c["x"]
                trace = video_roi[:, cy, cx]
                ax = axes[idx]
                ax.plot(trace, "k-", label="3×3 ROI Trace", alpha=0.6, linewidth=1.2)

                for b_idx, pk in enumerate(valid_peaks):
                    pk = int(pk)
                    global_bi = int(np.where(all_peaks == pk)[0][0])
                    next_pk = int(all_peaks[global_bi + 1]) if global_bi + 1 < len(all_peaks) else T
                    seg = trace[pk:next_pk]
                    if len(seg) < 3:
                        continue

                    l_peak = pk + int(np.argmax(seg))
                    l_amp  = float(trace[l_peak])
                    up_idx, _, _ = find_upstroke_start(trace, l_peak, l_amp, fps)
                    cross80, found80, _ = find_repol_crossing_with_fallback(
                        trace, l_peak, l_amp, fps,
                        threshold=80, next_peak_idx=next_pk, total_frames=T,
                    )

                    is_first = (b_idx == 0)
                    if up_idx is not None:
                        ax.axvline(up_idx, color="g", linestyle=":", alpha=0.4)
                        ax.plot(up_idx, trace[int(up_idx)], "go", markersize=5,
                                label="Start (Upstroke)" if is_first else "")
                    ax.plot(l_peak, trace[l_peak], "ro", markersize=5,
                            label="Max (Peak)" if is_first else "")
                    if found80 and cross80 is not None:
                        ax.axvline(cross80, color="b", linestyle=":", alpha=0.4)
                        ax.plot(cross80, trace[int(cross80)], "bo", markersize=5,
                                label="End (Repol80)" if is_first else "")

                ax.set_title(f"Cluster ROI: {c['label']} ({cy}, {cx})", fontsize=10, weight="bold")
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, alpha=0.2)

            plt.suptitle(
                f"Diagnostic Traces: {metric} Action Potential Landmark Points",
                fontsize=13, weight="bold",
            )
            plt.tight_layout()
            png_path = self.debug_dir / "apd_traces.png"
            plt.savefig(png_path, dpi=150)
            plt.close()
            self.logger.info(f"[DEBUG] Saved: apd_traces.png")

        except Exception as e:
            self.logger.warning(f"Diagnostic trace plot skipped: {e}")


# ---------------------------------------------------------------------------
# Standalone CLI (sys.exit 0/1/2 для оркестратора)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="APDAgent standalone — Stage 6")
    parser.add_argument("sample_id", help="Sample ID (e.g. 005A)")
    parser.add_argument("--results-root", default="results", help="Results root directory")
    parser.add_argument("--force", action="store_true", help="Force recompute even if output exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = PipelineConfig({"results_root": args.results_root})
    agent = APDAgent(args.sample_id, config=cfg)

    try:
        result = agent.run(force=args.force)
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0)
    except ValueError as e:
        # REJECT — QC не пройден
        logger.error(f"REJECT: {e}")
        sys.exit(2)
    except Exception as e:
        # CRASH — инфраструктурная ошибка
        logger.exception(f"CRASH: {e}")
        sys.exit(1)
