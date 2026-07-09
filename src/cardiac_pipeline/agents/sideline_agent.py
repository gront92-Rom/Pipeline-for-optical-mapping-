#!/usr/bin/env python3
"""
sideline_agent.py — Модуль-перехватчик для обработки длинных файлов (≥4096 кадров).
Активируется после загрузки (LoaderAgent). Если файл длинный, извлекает
центральный трейс 3х3, фильтрует его и генерирует текстовый гайд для исследователя,
затем сигнализирует о необходимости прервать основной пайплайн.
"""

import json
import logging
from typing import Any, Dict, Optional
import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig
from cardiac_pipeline.utils.preprocess import temporal_lowpass

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
        return self.load_must("video.npy")

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

        if nt < self.frame_limit:
            self.logger.info(f"Video length ({nt} frames) is within limits. Passing to main pipeline.")
            return {
                "status": "pass",
                "frames": nt
            }

        self.logger.warning(f"[SIDELINE DETECTION] Обнаружен длинный файл: {nt} кадров (порог >= {self.frame_limit}).")
        self.logger.warning("[SIDELINE DETECTION] Переключение в режим сайдлайн-анализа. Основной пайплайн должен быть заморожен.")

        # 2. Выделяем центральный трейс 3х3 и усредняем его в 1D
        center_h, center_w = H // 2, W // 2
        trace_3x3 = video[:, max(0, center_h - 1):min(H, center_h + 2), 
                             max(0, center_w - 1):min(W, center_w + 2)]
        mean_trace = np.mean(trace_3x3, axis=(1, 2))

        # 3. Фильтрация трейса (Butterworth) через временный 3D-View
        # temporal_lowpass ожидает 3D массив
        trace_3d = mean_trace[:, np.newaxis, np.newaxis]
        filtered_3d = temporal_lowpass(trace_3d, mask=None, fps=fps, cutoff=self.cutoff_hz)
        filtered_trace = filtered_3d.squeeze()

        # 4. Сохранение артефактов
        # Сохраняем данные трейса в must_dir
        sideline_data = {
            "filtered_trace": filtered_trace,
            "raw_mean_trace": mean_trace
        }
        data_path = self.save_must(sideline_data, "sideline_trace.npz")
        
        # 5. Генерация текстового гайда
        guide_text = generate_sideline_guide_text(
            filename=f"{self.sample_id}.rsh", # Используем sample_id как имя
            frames=nt,
            fps=fps,
            data_filename=data_path.name
        )
        
        guide_path = self.get_path("SIDELINE_GUIDE.txt", kind="must")
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(guide_text)
            
        self.logger.info(f"[SIDELINE] Данные трейса сохранены в: {data_path.name}")
        self.logger.info(f"[SIDELINE] Инструкция по дальнейшим шагам сгенерирована в: {guide_path.name}")

        metrics = {
            "frames": nt,
            "fps": fps,
            "sideline_activated": True
        }
        self._log_metrics(metrics)

        return {
            "status": "sideline_isolated",
            "frames": nt,
            "data_path": str(data_path),
            "guide_path": str(guide_path),
            "metrics": metrics
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SidelineAgent standalone")
    parser.add_argument("sample_id", help="Sample ID")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()
    
    cfg = PipelineConfig({"results_root": args.results_root})
    agent = SidelineAgent(args.sample_id, config=cfg)
    print(agent.run())
