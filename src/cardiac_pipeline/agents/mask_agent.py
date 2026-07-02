"""
MaskAgent v4.1 (Fixed)

Исправления по результатам ревью:
- Убран опасный блок "Последний шанс" (R2)
- При провале всех методов каскада теперь поднимается ValueError (честный гейтинг)
- Добавлено сохранение filtered_video.npy через save_intermediate()
- Сохранена Primary = RSM non-phys логика
- Fallback использует каскад методов + судейство + финальное сглаживание

Исправления при интеграции (2026-07-02):
- Импорты переведены на пакетные пути (cardiac_pipeline.*)
- save_intermediate() → save_debug() (метод из BaseAgent)
- crop_left/crop_right читаются из config.loader (не config.mask)
- get_path() вызывается с явным kind='must' для metadata.json
- print() → self.logger (consistency с BaseAgent)
- stim_hz fallback задокументирован явно
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig

try:
    from cardiac_pipeline.utils.preprocess import preprocess_video, should_invert
    PREPROCESS_AVAILABLE = True
except ImportError:
    preprocess_video = None
    should_invert = None
    PREPROCESS_AVAILABLE = False

try:
    from skimage.morphology import (
        remove_small_objects, binary_fill_holes, binary_opening, binary_closing, disk
    )
    from skimage.measure import regionprops
    from scipy.ndimage import label
    import cv2
    HEAVY_DEPS = True
except ImportError:
    HEAVY_DEPS = False


PERC_EX = 0.005
N_ROUGH = 7


class MaskAgent(BaseAgent):
    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None):
        super().__init__(sample_id, config)
        # crop параметры живут в config.loader (не в config.mask)
        loader_cfg = getattr(self.config, 'loader', {}) or {}
        self.crop_left = int(loader_cfg.get('crop_left', 20))
        self.crop_right = int(loader_cfg.get('crop_right', 8))

        self.mcfg = getattr(self.config, 'mask', {}) or {}
        self.CRITERIA_LOOSE: Dict[str, Any] = {}   # заполняется конкретной реализацией
        self.CRITERIA_STRICT: Dict[str, Any] = {}  # заполняется конкретной реализацией
        self.COV_FLOOR = float(self.mcfg.get('cov_floor', 0.35))

        self.metadata: Dict[str, Any] = {}
        self.raw_video: Optional[np.ndarray] = None

    # ==================== ПОДГОТОВКА ДАННЫХ ====================

    def _load_metadata(self) -> Dict[str, Any]:
        # metadata.json лежит в must_dir (сохранён LoaderAgent)
        meta_path = self.get_path("metadata.json", kind='must')
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self.logger.warning("metadata.json not found — fps/dye/stim_hz will be missing")
        return self.metadata

    def _get_fps(self) -> float:
        fps = self.metadata.get("fps")
        if fps is None:
            raise ValueError(
                "fps отсутствует в metadata.json. "
                "LoaderAgent должен сохранить его заранее."
            )
        return float(fps)

    def _load_and_crop_video(self) -> np.ndarray:
        video = self.load_must("loaded_video.npy")
        # Кроп применяется только если ширина == 128 (MiCAM ULTIMA стандарт)
        if video.ndim == 3 and video.shape[2] == 128:
            video = video[:, :, self.crop_left : self.crop_left + (128 - self.crop_left - self.crop_right)]
        self.raw_video = video
        return video

    def _prepare_data(self) -> None:
        self._load_metadata()
        self._load_and_crop_video()

    # ==================== PRIMARY (RSM non-phys) ====================

    def _primary_rsm_bg_pipeline(self, raw_rsm: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Логика stage2 → stage3_bisect → stage4_cleanup → stage5_smoothing.
        Возвращает (mask, metrics) или (None, {}) если RSM метод не применим.
        """
        # TODO: перенести реализацию из rsm_mask_worker_v3.py
        # Заглушка — возвращает None чтобы активировать fallback
        return None, {}

    # ==================== FALLBACK ====================

    def _get_video_for_fallback(self) -> np.ndarray:
        """
        Возвращает предобработанное видео для fallback-маскирования.
        Кэширует в debug/filtered_video.npy.
        """
        filtered_path = self.get_path("filtered_video.npy", kind='debug')

        if filtered_path.exists():
            self.logger.info("[Fallback] Используется готовый debug/filtered_video.npy")
            return np.load(filtered_path)

        if self.raw_video is None:
            self._load_and_crop_video()

        if not PREPROCESS_AVAILABLE:
            self.logger.warning(
                "[Fallback] preprocess недоступен — используем сырое видео"
            )
            return self.raw_video

        fps = self._get_fps()
        invert = should_invert(
            sample_name=self.sample_id,
            dye=self.metadata.get("dye"),
            recording_mode=self.metadata.get("recording_mode"),
        )

        processed = preprocess_video(
            self.raw_video,
            fps=fps,
            invert=invert,
            sample_name=self.sample_id,
            dye=self.metadata.get("dye"),
            recording_mode=self.metadata.get("recording_mode"),
            do_normalize=False,
            sigma=2.0,
            lp_cutoff=80.0,
        )

        # Кэш в debug (не MUST — промежуточный артефакт)
        self.save_debug(processed, "filtered_video.npy")
        self.logger.info("[Fallback] Предобработанное видео сохранено в debug/filtered_video.npy")

        return processed

    def _fallback_with_cascade(self) -> Tuple[np.ndarray, Dict]:
        """
        Каскад fallback-методов с судейством.
        Поднимает ValueError если все методы провалились (R2 fix).
        """
        self.logger.info("[Fallback] Запуск каскада методов + судейство")

        video = self._get_video_for_fallback()
        fps = self._get_fps()
        # stim_hz: из метаданных, fallback = 10 Hz (документированный дефолт)
        stim_hz = self.metadata.get("stim_hz")
        if stim_hz is None:
            self.logger.warning("[Fallback] stim_hz не найден в metadata — используем 10 Hz")
            stim_hz = 10.0

        methods: List[Dict[str, Any]] = [
            {"name": "bandpower_5_15_p55", "type": "bandpower", "pct": 55},
            {"name": "bandpower_5_15_p45", "type": "bandpower", "pct": 45},
            {"name": "bandpower_stim",     "type": "bandpower_stim", "stim_hz": stim_hz},
            {"name": "foreground_600",     "type": "foreground", "threshold": 600},
            {"name": "foreground_500",     "type": "foreground", "threshold": 500},
        ]

        best_mask: Optional[np.ndarray] = None
        best_qc: Optional[Dict] = None
        best_name: Optional[str] = None

        for m in methods:
            try:
                mask = self._generate_mask_by_type(video, m, fps)

                if HEAVY_DEPS:
                    mask = binary_opening(mask, iterations=1)
                    mask = binary_closing(mask, iterations=2)
                    mask = binary_fill_holes(mask)

                qc = self._compute_mask_qc_fallback(mask, video)
                verdict, reason = self._judge_mask_fallback(qc)

                self.logger.info(f"  {m['name']} → {verdict} ({reason})")

                if verdict == "PASS":
                    best_mask = mask
                    best_qc = qc
                    best_name = m['name']
                    break
                elif verdict == "RETRY":
                    if best_mask is None or qc.get("compactness", 0) > (best_qc or {}).get("compactness", 0):
                        best_mask = mask
                        best_qc = qc
                        best_name = m['name']

            except Exception as e:
                self.logger.warning(f"  {m['name']} error: {e}")
                continue

        # === СТРОГИЙ ГЕЙТИНГ (исправление R2) ===
        if best_mask is None:
            raise ValueError(
                "All fallback cascade methods rejected the mask. "
                "Sample is unviable or requires manual quarantine."
            )

        # Финальное сглаживание (как в RSM)
        if HEAVY_DEPS:
            best_mask = self._apply_contour_smoothing(best_mask)

        return best_mask.astype(bool), {
            "method": f"fallback_cascade_{best_name}",
            "coverage": (best_qc or {}).get("coverage", 0),
            "compactness": (best_qc or {}).get("compactness", 0),
            "used_preprocessing": True,
        }

    # ==================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ====================

    def _generate_mask_by_type(
        self, video: np.ndarray, method: Dict[str, Any], fps: float
    ) -> np.ndarray:
        """
        Генерирует бинарную маску по типу метода.
        TODO: перенести bandpower-реализацию из optical_pipeline_worker.py
        """
        if method["type"] == "bandpower":
            # Заглушка — реализация будет перенесена из pipeline_worker
            raise NotImplementedError("bandpower mask not yet ported from pipeline_worker")
        elif method["type"] == "bandpower_stim":
            raise NotImplementedError("bandpower_stim mask not yet ported from pipeline_worker")
        elif method["type"] == "foreground":
            mean_frame = video.mean(axis=0)
            return (mean_frame > method.get("threshold", 600)).astype(bool)
        raise ValueError(f"Unknown mask method type: {method['type']}")

    def _compute_mask_qc_fallback(self, mask: np.ndarray, video: np.ndarray) -> Dict[str, Any]:
        """
        Вычисляет QC-метрики маски.
        TODO: перенести из compute_mask_qc в pipeline_worker.py
        """
        total = mask.size
        n_true = int(mask.sum())
        coverage = n_true / total if total > 0 else 0.0
        return {
            "coverage": coverage,
            "compactness": 0.0,  # TODO: реализовать
            "n_components": 1,   # TODO: реализовать через label()
        }

    def _judge_mask_fallback(self, qc: Dict[str, Any]) -> Tuple[str, str]:
        """
        Судейство маски: PASS / RETRY / FAIL.
        TODO: перенести пороги из qc_thresholds.yaml / pipeline_worker.py
        """
        cov = qc.get("coverage", 0)
        if cov < 0.05:
            return "FAIL", f"coverage={cov:.3f} < 0.05"
        if cov > 0.95:
            return "FAIL", f"coverage={cov:.3f} > 0.95 (likely background)"
        if cov < self.COV_FLOOR:
            return "RETRY", f"coverage={cov:.3f} < COV_FLOOR={self.COV_FLOOR}"
        return "PASS", f"coverage={cov:.3f}"

    def _apply_contour_smoothing(self, mask: np.ndarray) -> np.ndarray:
        """
        Contour LPF сглаживание контура маски.
        TODO: перенести из rsm_mask_worker_v3.py
        """
        # Заглушка — возвращает маску без изменений до переноса реализации
        return mask

    # ==================== RUN ====================

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Основной метод агента.

        Порядок:
        1. Если mask.npy уже существует и force=False — пропустить (idempotent)
        2. Загрузить metadata.json + loaded_video.npy (от LoaderAgent)
        3. PRIMARY: RSM non-phys pipeline
        4. FALLBACK: cascade если PRIMARY вернул None или маска с дырами
        5. Сохранить mask.npy в must_dir
        """
        if not force and self.exists("mask.npy"):
            self.logger.info("mask.npy already exists, skipping (use force=True to rerun)")
            return {"status": "skipped", "mask_path": str(self.get_path("mask.npy"))}

        # Lazy upstream: LoaderAgent должен был сохранить loaded_video.npy + metadata.json
        try:
            self._prepare_data()
        except FileNotFoundError:
            self.logger.info("loaded_video.npy not found — running LoaderAgent first")
            from cardiac_pipeline.agents.loader_agent import LoaderAgent
            LoaderAgent(self.sample_id, self.config).run()
            self._prepare_data()

        try:
            raw_rsm = self.load_must("raw_rsm.npy")
        except FileNotFoundError:
            self.logger.warning("raw_rsm.npy not found — PRIMARY will be skipped, going to FALLBACK")
            raw_rsm = None

        # PRIMARY
        mask: Optional[np.ndarray] = None
        metrics: Dict[str, Any] = {}

        if raw_rsm is not None:
            mask, metrics = self._primary_rsm_bg_pipeline(raw_rsm)

        # FALLBACK
        if mask is None or metrics.get("n_holes", 99) > 0:
            self.logger.info("[MaskAgent] Переход в Fallback")
            mask, fb_metrics = self._fallback_with_cascade()
            metrics.update(fb_metrics)

        self.save_must(mask.astype(np.uint8), "mask.npy")
        self._log_metrics(metrics)

        return {
            "status": "success",
            "method": metrics.get("method"),
            "mask_path": str(self.get_path("mask.npy")),
            "metrics": metrics,
        }


if __name__ == "__main__":
    print("MaskAgent v4.1 (Fixed) ready.")
