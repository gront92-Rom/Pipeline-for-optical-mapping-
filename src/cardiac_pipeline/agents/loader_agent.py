#!/usr/bin/env python3
"""
loader_agent.py — Stage 1: Загрузка данных, извлечение метаданных, предобработка.
Версия v2 (2026-07-02).

Архитектура:
  Наследует BaseAgent. Читает параметры из PipelineConfig (config/default.yaml).
  Делегирует парсинг — utils/metadata_extractor.py.
  Делегирует предобработку — utils/preprocess.py v6.

Входные данные:
  - Путь к файлу данных (.bvx / .rsh / .gsh / .npy) или директории с ними.
    Передаётся в run(input_path=...).

Выходные данные:
  MUST:
    - raw_video.npy          — сырое видео float32 (T, H, W) после кропа
    - preproc_video.npy      — препроцессированное видео (activation, 80 Гц)
    - preproc_video_apd.npy  — препроцессированное видео (apd, 150 Гц)
    - metadata.json          — полные метаданные (fps, dye, pixel_size_mm, ...)
  DEBUG (только при sideline-режиме):
    - sideline_trace.npy     — усреднённый трейс центрального ROI (длинные записи)

Режимы:
  "full"     — стандартная обработка (< sideline_threshold кадров)
  "sideline" — только трейс центрального ROI (>= sideline_threshold кадров)

Коды возврата (CLI-режим):
  0 = SUCCESS
  1 = CRASH (исключение инфраструктуры)
  2 = REJECT (файл не найден / метаданные неполные)

Исправления относительно исходного loader_agent.py:
  - from base_agent import → from cardiac_pipeline.base_agent import
  - from preprocess_v5 import → from cardiac_pipeline.utils.preprocess import
  - from metadata_extractor_v3 import → from cardiac_pipeline.utils.metadata_extractor import
  - save_meta() → save_must() (BaseAgent не имеет save_meta)
  - save_must("loaded_video.npy", ...) → save_must(data, "preproc_video.npy")
    (имя согласовано с PeakDetectorAgent, который ищет preproc_video.npy)
  - Добавлена двойная предобработка: activation (80 Гц) + apd (150 Гц)
  - Добавлен кроп (crop_left / crop_right из конфига)
  - Параметры из PipelineConfig (не из argparse)
  - REJECT → raise ValueError (BaseAgent-контракт)
  - sideline_threshold из конфига (не хардкод 4096)
  - Добавлен raw_video.npy (сырые данные до предобработки)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig
from cardiac_pipeline.utils.metadata_extractor import extract_micam_metadata
from cardiac_pipeline.utils.preprocess import preprocess_video

logger = logging.getLogger(__name__)

# Форматы, которые можно загрузить напрямую через numpy (без optimap)
_NPY_EXTENSIONS = {".npy", ".npz"}


# ---------------------------------------------------------------------------
# LoaderAgent
# ---------------------------------------------------------------------------

class LoaderAgent(BaseAgent):
    """
    Stage 1: Загрузка сырых данных MiCAM ULTIMA, извлечение метаданных,
    двойная предобработка (activation 80 Гц + APD 150 Гц).

    Потребляет: файл данных (.bvx/.rsh/.gsh/.npy) или директорию.
    Производит: raw_video.npy, preproc_video.npy, preproc_video_apd.npy, metadata.json.
    """

    DEPENDS_ON: list = []           # Stage 1 — ни от кого не зависит
    REQUIRED_INPUTS: list = []       # вход — внешний файл (input_path)

    def __init__(
        self,
        sample_id: str,
        config: Optional[PipelineConfig] = None,
    ):
        super().__init__(sample_id, config)

        loader_cfg = self.config.loader if isinstance(self.config.loader, dict) else {}
        preproc_cfg = self.config.preprocess if isinstance(self.config.preprocess, dict) else {}

        # Кроп (пиксели, убираем артефакты краёв матрицы)
        self.crop_left:  int = int(loader_cfg.get("crop_left",  20))
        self.crop_right: int = int(loader_cfg.get("crop_right",  8))

        # Порог sideline-режима (кадры)
        self.sideline_threshold: int = int(loader_cfg.get("sideline_threshold", 4096))

        # Параметры предобработки
        self.spatial_sigma:    float = float(preproc_cfg.get("spatial_sigma",    2.0))
        self.chunk_size:       int   = int(preproc_cfg.get("chunk_size",        8192))
        self.overlap:          int   = int(preproc_cfg.get("overlap",            256))
        self.lp_cutoff_act:    float = float(preproc_cfg.get("lp_cutoff_activation_hz", 80.0))
        self.lp_cutoff_apd:    float = float(preproc_cfg.get("lp_cutoff_apd_hz",       150.0))
        self.asls_lam:         float = float(preproc_cfg.get("asls_lam",         1e8))
        self.asls_p:           float = float(preproc_cfg.get("asls_p",           0.01))
        self.asls_niter:       int   = int(preproc_cfg.get("asls_niter",         3))

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _load_video(self, input_path: Path) -> np.ndarray:
        """
        Загружает видео из файла.

        Поддерживаемые форматы:
          .npy / .npz — numpy напрямую
          .bvx / .rsh / .gsh — через optimap (MiCAM ULTIMA)
        """
        ext = input_path.suffix.lower()

        if ext in _NPY_EXTENSIONS:
            self.logger.info(f"Загружаю numpy: {input_path.name}")
            data = np.load(str(input_path), allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                # Берём первый массив из npz
                key = list(data.keys())[0]
                return data[key].astype(np.float32)
            return data.astype(np.float32)

        # Попытка загрузить через optimap
        try:
            import optimap as om
            self.logger.info(f"Загружаю через optimap: {input_path.name}")
            return om.load_video(str(input_path)).astype(np.float32)
        except ImportError:
            raise ImportError(
                "optimap не установлен. Для загрузки .bvx/.rsh/.gsh файлов "
                "выполните: pip install opticalmapping"
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки {input_path}: {e}") from e

    def _crop_video(self, video: np.ndarray) -> np.ndarray:
        """Убирает артефактные пиксели по краям матрицы (crop_left / crop_right)."""
        _, H, W = video.shape
        cl, cr = self.crop_left, self.crop_right
        if cl + cr >= W:
            self.logger.warning(
                f"crop_left={cl} + crop_right={cr} >= W={W} — кроп пропущен"
            )
            return video
        cropped = video[:, :, cl: W - cr] if cr > 0 else video[:, :, cl:]
        self.logger.info(f"Кроп: W {W} → {cropped.shape[2]} (left={cl}, right={cr})")
        return cropped

    def _extract_metadata(self, input_path: Path) -> Dict[str, Any]:
        """
        Извлекает метаданные из .bvx/.rsh/.gsh или строит минимальный словарь для .npy.
        Всегда сохраняет metadata.json в must/.
        """
        ext = input_path.suffix.lower()

        if ext in _NPY_EXTENSIONS:
            # Для .npy файлов метаданные недоступны — строим минимальный словарь
            self.logger.warning(
                "Входной файл .npy — метаданные недоступны. "
                "fps и dye будут взяты из конфига или останутся None."
            )
            meta = {
                "sample_id":     self.sample_id,
                "source_file":   str(input_path),
                "fps":           self.config.loader.get("default_fps") if isinstance(self.config.loader, dict) else None,
                "dye":           self.config.loader.get("default_dye") if isinstance(self.config.loader, dict) else None,
                "pixel_size_mm": self.config.pixel_size_mm,
                "source":        "npy_fallback",
            }
        else:
            # Полное извлечение из MiCAM-файлов
            meta = extract_micam_metadata(
                input_path.parent,
                base_name=input_path.stem,
                write_json=False,  # Мы сами сохраним через save_must
            )
            meta["sample_id"] = self.sample_id
            meta["source_file"] = str(input_path)

        # Всегда записываем в must/metadata.json
        self.save_must(meta, "metadata.json")
        return meta

    def _handle_sideline(self, video: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sideline-режим для длинных записей (>= sideline_threshold кадров).
        Сохраняет только усреднённый трейс центрального 3×3 ROI.
        """
        T, H, W = video.shape
        cy, cx = H // 2, W // 2
        y0, y1 = max(0, cy - 1), min(H, cy + 2)
        x0, x1 = max(0, cx - 1), min(W, cx + 2)
        trace = np.mean(video[:, y0:y1, x0:x1], axis=(1, 2)).astype(np.float32)
        self.save_debug(trace, "sideline_trace.npy")
        self.logger.warning(
            f"Sideline-режим: {T} кадров >= порога {self.sideline_threshold}. "
            "Сохранён только центральный трейс."
        )
        return {
            "status":   "sideline",
            "n_frames": T,
            "shape":    list(video.shape),
        }

    # ------------------------------------------------------------------
    # Auto-discovery входного файла
    # ------------------------------------------------------------------

    # Расширения файлов данных, которые LoaderAgent умеет загружать
    DATA_EXTENSIONS = (".rsh", ".bvx", ".npy")

    def _discover_input_file(self) -> Path:
        """
        Автоматический поиск входного файла для sample_id.
        
        Порядок поиска:
          1. results_root/<sample_id>/raw/  (*.rsh, *.bvx, *.npy)
          2. data_root/<sample_id>/        (*.rsh, *.bvx, *.npy)
          3. data_root/                    (любой файл с sample_id в имени)
        
        Возвращает Path к найденному файлу.
        Raise ValueError если ничего не найдено.
        """
        sample = self.sample_id
        data_root = Path(self.config.data_root)
        results_root = Path(self.config.results_root)
        
        # 1. results_root/<sample>/raw/
        raw_dir = results_root / sample / "raw"
        if raw_dir.exists():
            for ext in self.DATA_EXTENSIONS:
                candidates = sorted(raw_dir.glob(f"*{ext}"))
                if candidates:
                    self.logger.info(f"Auto-discovery: {candidates[0]} (from {raw_dir})")
                    return candidates[0]
        
        # 2. data_root/<sample>/
        sample_dir = data_root / sample
        if sample_dir.exists():
            for ext in self.DATA_EXTENSIONS:
                candidates = sorted(sample_dir.glob(f"*{ext}"))
                if candidates:
                    self.logger.info(f"Auto-discovery: {candidates[0]} (from {sample_dir})")
                    return candidates[0]
        
        # 3. data_root/ — ищем файл с sample_id в имени
        if data_root.exists():
            for ext in self.DATA_EXTENSIONS:
                # Точное совпадение по sample_id в имени файла
                pattern = f"*{sample}*{ext}"
                candidates = sorted(data_root.rglob(pattern))
                if candidates:
                    # Фильтруем: sample_id как отдельный токен (не подстрока другого ID)
                    # Например, "004A" не должен матчить "004AB"
                    filtered = [
                        c for c in candidates
                        if self._sample_id_matches(c.stem, sample)
                    ]
                    if filtered:
                        self.logger.info(f"Auto-discovery: {filtered[0]} (from data_root rglob)")
                        return filtered[0]
        
        raise ValueError(
            f"Auto-discovery: файл для sample '{sample}' не найден. "
            f"Искал в:\n"
            f"  1. {raw_dir}/  (*.rsh, *.bvx, *.npy)\n"
            f"  2. {sample_dir}/  (*.rsh, *.bvx, *.npy)\n"
            f"  3. {data_root}/  (rglob '*{sample}*')\n"
            f"Передайте input_path явно в run()."
        )
    
    @staticmethod
    def _sample_id_matches(stem: str, sample_id: str) -> bool:
        """
        Проверяет, что sample_id присутствует в имени файла как отдельный токен.
        '004A' матчит '2026-05-08-mSHAM-bs2-6Hz-0508-004A', но не '004AB'.
        """
        # Разбиваем по разделителям и проверяем точное совпадение
        import re
        tokens = re.split(r'[-_.\s]+', stem.upper())
        return sample_id.upper() in tokens

    # Companion-расширения MiCAM ULTIMA
    COMPANION_EXTENSIONS = (".rsh", ".gsh", ".rsm", ".gsd", ".rsd", ".bvx")

    def _discover_companion_files(self, input_path: Path) -> dict:
        """
        Находит все companion-файлы MiCAM ULTIMA рядом с input_path.
        
        Возвращает dict: {".rsh": "/path/to/file.rsh", ".gsh": "...", ...}
        Использует stem (имя без расширения) для поиска.
        """
        stem = input_path.stem
        parent = input_path.parent
        
        companions = {}
        for ext in self.COMPANION_EXTENSIONS:
            # Точное совпадение по stem
            candidate = parent / (stem + ext)
            if candidate.exists():
                companions[ext] = str(candidate)
                continue
            # Без скобок-суффиксов (например, "(0)" в .rsd)
            if ext == ".rsd":
                rsd_candidates = sorted(parent.glob(f"{stem}*{ext}"))
                if rsd_candidates:
                    companions[ext] = [str(c) for c in rsd_candidates]
        
        # Гарантируем: .rsh всегда есть (это основной файл)
        if input_path.suffix == ".rsh":
            companions[".rsh"] = str(input_path)
        elif ".rsh" not in companions:
            # Если input — .bvx/.npy, ищем .rsh с тем же stem
            rsh_candidate = parent / (stem + ".rsh")
            if rsh_candidate.exists():
                companions[".rsh"] = str(rsh_candidate)
        
        return companions

    # ------------------------------------------------------------------
    # Главный метод
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: Optional[Union[str, Path]] = None,
        force: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Запускает полный цикл загрузки и предобработки.

        Порядок:
          1. Проверка кэша (если preproc_video.npy уже есть — пропускаем)
          2. Разрешение пути к входному файлу
          3. Извлечение метаданных → metadata.json
          4. Загрузка видео + кроп
          5. Сохранение raw_video.npy
          6. Sideline-гейтинг (длинные записи)
          7. Предобработка activation (80 Гц) → preproc_video.npy
          8. Предобработка APD (150 Гц) → preproc_video_apd.npy
          9. Сохранение метрик

        Параметры:
            input_path — путь к файлу данных или директории.
                         Если None — ищет файл в стандартном месте
                         (results_root/<sample_id>/raw/).
            force      — перезаписать даже если выходные файлы существуют.
        """
        if not force and self.exists("preproc_video.npy", kind="debug"):
            self.logger.info("preproc_video.npy exists, skipping LoaderAgent")
            return {"status": "skipped"}

        t0 = time.perf_counter()

        # --- 2. Разрешение пути (auto-discovery) ---
        if input_path is None:
            input_path = self._discover_input_file()

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Входной файл не найден: {input_path}")

        # --- 3. Метаданные ---
        self.logger.info(f"Извлекаю метаданные из {input_path}")
        metadata = self._extract_metadata(input_path)

        fps = metadata.get("fps")
        dye = metadata.get("dye")
        recording_mode = metadata.get("recording_mode")

        if fps is None:
            raise ValueError(
                "fps не найден в метаданных. Убедитесь, что .rsh/.gsh файл "
                "присутствует рядом с .bvx, или укажите default_fps в config/loader."
            )
        fps = float(fps)

        # Если recording_mode не извлечён из header — выводим из dye
        if recording_mode is None and dye is not None:
            d = dye.upper().strip()
            if d.startswith("A"):
                recording_mode = "voltage"
            elif d.startswith("B"):
                recording_mode = "calcium"
        metadata["recording_mode"] = recording_mode

        # --- 4. Загрузка видео + кроп ---
        self.logger.info(f"Загружаю видео: {input_path.name}")
        video = self._load_video(input_path)
        video = self._crop_video(video)
        T, H, W = video.shape
        self.logger.info(f"Видео загружено: {video.shape}, fps={fps}, dye={dye}, mode={recording_mode}")

        # --- 4b. Доминантная частота сигнала (FFT) ---
        try:
            from cardiac_pipeline.utils.metadata_extractor import compute_dominant_freq
            stim_hz_effective = compute_dominant_freq(video, fps=fps)
            metadata["stim_hz_effective"] = round(stim_hz_effective, 2)
            self.logger.info(f"Доминантная частота сигнала: {stim_hz_effective:.2f} Hz")
        except Exception as e:
            self.logger.warning(f"compute_dominant_freq failed: {e}")
            metadata["stim_hz_effective"] = None

        # --- 4b.2. Сохранение путей ко всем companion-файлам в metadata ---
        companion_files = self._discover_companion_files(input_path)
        metadata["companion_files"] = companion_files
        self.logger.info(f"Companion files: {list(companion_files.keys())}")

        # Сохраняем обновлённые метаданные
        self.save_must(metadata, "metadata.json")

        # --- 5. Сохранение сырого видео ---
        self.save_must(video, "raw_video.npy")

        # --- 5b. Загрузка .rsm (background frame) для MaskAgent PRIMARY ---
        rsm_path = input_path.with_suffix(".rsm")
        if not rsm_path.exists():
            # Try alternate: replace .rsh with .rsm in the same directory
            rsm_path = input_path.parent / (input_path.stem + ".rsm")
        if rsm_path.exists():
            try:
                rsm_raw = np.fromfile(str(rsm_path), dtype=np.uint16)
                orig_W = 128  # MiCAM ULTIMA sensor width (before crop)
                expected = H * orig_W
                if rsm_raw.size >= expected:
                    rsm_frame = rsm_raw[:expected].reshape(H, orig_W)
                    # Apply same crop as video
                    if self.crop_left + self.crop_right > 0:
                        rsm_frame = rsm_frame[:, self.crop_left:orig_W - self.crop_right]
                    rsm_3d = rsm_frame[np.newaxis, :, :].astype(np.float32)  # (1, H, W_crop)
                    self.save_must(rsm_3d, "raw_rsm.npy")
                    self.logger.info(f"raw_rsm.npy сохранён: {rsm_3d.shape} из {rsm_path.name}")
                else:
                    self.logger.warning(f"rsm size {rsm_raw.size} < expected {expected} — пропускаю")
            except Exception as e:
                self.logger.warning(f"Не удалось загрузить .rsm ({e}) — MaskAgent будет использовать fallback")
        else:
            self.logger.info(f".rsm не найден рядом с {input_path.name} — MaskAgent будет использовать fallback")

        # --- 6. Sideline-гейтинг ---
        if T >= self.sideline_threshold:
            result = self._handle_sideline(video, metadata)
            elapsed = round(time.perf_counter() - t0, 2)
            result["elapsed_s"] = elapsed
            self._log_metrics({"loader_mode": "sideline", "n_frames": T, "elapsed_s": elapsed})
            return result

        # --- 7. Предобработка activation (80 Гц) → must/preproc_video.npy ---
        # Variant A (2026-07-09): preproc_video.npy — MUST (не debug).
        # Single source of truth для PeakDetector/Activation/APD/Alternans.
        self.logger.info(f"Предобработка activation (LPF={self.lp_cutoff_act} Гц) → must/preproc_video.npy")
        preproc_act = preprocess_video(
            video=video,
            fps=fps,
            mask=None,              # Маска ещё не готова (MaskAgent идёт после)
            target_stage="activation",
            lp_cutoff=self.lp_cutoff_act,
            sigma=self.spatial_sigma,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            sample_name=self.sample_id,             # ← FIX: явный sample_name для should_invert()
            dye=dye,
            recording_mode=recording_mode,           # ← из metadata, не из dye
            do_asls=False,          # ASLS требует маску — откладываем до MaskAgent
            do_normalize=True,
        )
        self.save_must(preproc_act, "preproc_video.npy")

        # --- 8. Предобработка APD (150 Гц) → must/preproc_video_apd.npy ---
        # Variant A: тоже MUST (используется APDAgent — отдельная ветка с более мягким LPF)
        self.logger.info(f"Предобработка APD (LPF={self.lp_cutoff_apd} Гц) → must/preproc_video_apd.npy")
        preproc_apd = preprocess_video(
            video=video,
            fps=fps,
            mask=None,
            target_stage="apd",
            lp_cutoff=self.lp_cutoff_apd,
            sigma=self.spatial_sigma,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            sample_name=self.sample_id,             # ← FIX: явный sample_name для should_invert()
            dye=dye,
            recording_mode=recording_mode,           # ← из metadata, не из dye
            do_asls=False,
            do_normalize=True,
        )
        self.save_must(preproc_apd, "preproc_video_apd.npy")

        # --- 9. Метрики ---
        elapsed = round(time.perf_counter() - t0, 2)
        self._log_metrics({
            "loader_mode":   "full",
            "n_frames":      T,
            "height":        H,
            "width":         W,
            "fps":           fps,
            "dye":           dye,
            "recording_mode": recording_mode,
            "stim_hz":       metadata.get("stim_hz"),
            "stim_hz_effective": metadata.get("stim_hz_effective"),
            "elapsed_s":     elapsed,
        })

        self.logger.info(
            f"LoaderAgent done in {elapsed}s — "
            f"video={video.shape}, fps={fps}, dye={dye}, mode={recording_mode}, "
            f"stim_hz_eff={metadata.get('stim_hz_effective')}"
        )

        return {
            "status":  "success",
            "shape":   list(video.shape),
            "fps":     fps,
            "dye":     dye,
            "elapsed_s": elapsed,
        }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="LoaderAgent standalone — Stage 1")
    parser.add_argument("sample_id",   help="Sample ID (e.g. 005A)")
    parser.add_argument("input_path",  help="Path to .bvx/.rsh/.gsh/.npy data file")
    parser.add_argument("--results-root", default="results", help="Results root directory")
    parser.add_argument("--force", action="store_true", help="Force recompute")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg   = PipelineConfig({"results_root": args.results_root})
    agent = LoaderAgent(args.sample_id, config=cfg)

    try:
        result = agent.run(input_path=args.input_path, force=args.force)
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"REJECT: {e}")
        sys.exit(2)
    except Exception as e:
        logger.exception(f"CRASH: {e}")
        sys.exit(1)
