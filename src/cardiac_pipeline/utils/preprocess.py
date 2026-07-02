#!/usr/bin/env python3
"""
preprocess.py — Единый канонический модуль предобработки оптических данных.
Версия v6 (2026-07-02).

Изменения v6 относительно v5:
  - Добавлен параметр target_stage ("activation" | "apd") для авто-подбора LPF:
      activation → 80 Гц  (жёсткий срез, давит шум перед поиском апстрока)
      apd        → 150 Гц (мягкий срез, сохраняет морфологию хвоста реполяризации)
  - Добавлена ASLS-коррекция базовой линии (Eilers & Boelens 2005):
      do_asls=True применяет коррекцию пиксель-за-пикселем строго под маской.
  - КРИТИЧЕСКИ ВАЖНЫЙ порядок операций исправлен:
      Spatial → LPF → Invert → ASLS → Normalize
      Инверсия обязана быть ДО ASLS: алгоритм ASLS с p=0.01 ищет нижнюю огибающую
      (ожидает пики вверх). Для VSD сырой сигнал идёт вниз — без инверсии ASLS
      исказит пики вместо того, чтобы убрать дрейф.
  - should_invert() теперь возвращает bool (не Optional[bool]):
      при неопределённом красителе возвращает True (дефолт VSD/A) с WARNING.
  - Сохранены из v5: chunk-логика sosfiltfilt, mask-aware фильтрация, PP7-фикс.

Функции:
  should_invert()                — определяет необходимость инверсии (VSD vs CaT)
  spatial_smooth()               — Gaussian по осям H, W
  temporal_lowpass()             — Butterworth 4-го порядка, zero-phase, с чанкованием
  asls_baseline_correct_trace()  — ASLS для одного трейса (1D)
  asls_baseline_correct()        — ASLS для всего видео под маской
  normalize_traces()             — ΔF/F нормализация
  preprocess_video()             — главный оркестратор
"""

import logging
import re
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Авто-подбор частоты среза по стадии пайплайна
# ---------------------------------------------------------------------------
_LP_CUTOFF_BY_STAGE = {
    "activation": 80.0,   # Жёсткий срез — давим шум перед поиском апстрока
    "apd":        150.0,  # Мягкий срез — сохраняем морфологию реполяризации
}


# ---------------------------------------------------------------------------
# Определение инверсии
# ---------------------------------------------------------------------------

def _parse_sample_id(name: str) -> Optional[str]:
    m = re.search(r'(?<![0-9])(\d{3,4}[AB])(?:[_.\-]|$)', name, re.IGNORECASE)
    return m.group(1).upper() if m else None


def should_invert(
    sample_name: Optional[str] = None,
    dye: Optional[str] = None,
    recording_mode: Optional[str] = None,
) -> bool:
    """
    Определяет, нужна ли инверсия сигнала (True для VSD/вольтажа).

    Приоритет: recording_mode > dye > sample_name > дефолт True (VSD).
    В отличие от v5, возвращает bool (не Optional[bool]):
    при неопределённом источнике выдаёт WARNING и возвращает True.
    """
    if recording_mode is not None:
        rm = recording_mode.lower().strip()
        if rm in ("voltage", "vsd", "ap"):
            return True
        if rm in ("calcium", "cat", "ca"):
            return False

    if dye is not None:
        d = dye.upper().strip()
        if d in ("A", "VOLTAGE", "VSD"):
            return True
        if d in ("B", "CALCIUM", "CAT"):
            return False

    if sample_name is not None:
        sid = _parse_sample_id(sample_name) or sample_name
        token = sid.upper().split("_")[0]
        if token.endswith("A"):
            return True
        if token.endswith("B"):
            return False

    logger.warning(
        "should_invert: не удалось определить тип красителя по recording_mode/dye/sample_name. "
        "Применяется дефолт True (VSD/A). Передайте dye='A'/'B' или recording_mode для явного контроля."
    )
    return True  # Дефолт: VSD (краситель A)


# ---------------------------------------------------------------------------
# Пространственное сглаживание
# ---------------------------------------------------------------------------

def spatial_smooth(
    video: np.ndarray,
    sigma: float = 2.0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Пространственная фильтрация Гауссом только по осям H и W.
    sigma=(0, s, s) гарантирует нулевое сглаживание по оси времени.
    """
    video = np.asarray(video, dtype=np.float32)
    smooth = gaussian_filter(video, sigma=(0, sigma, sigma))
    if mask is not None:
        smooth[:, ~mask] = 0.0
    return smooth


# ---------------------------------------------------------------------------
# Временная фильтрация (Butterworth, zero-phase, с чанкованием)
# ---------------------------------------------------------------------------

def temporal_lowpass(
    video: np.ndarray,
    fps: float,
    cutoff: float,
    mask: Optional[np.ndarray] = None,
    chunk_size: int = 8192,
    overlap: int = 256,
) -> np.ndarray:
    """
    Zero-phase временная фильтрация Баттерворта 4-го порядка (sosfiltfilt).

    - Если n_frames <= chunk_size: фильтрует весь массив сразу (рекомендуемый путь).
    - При чанковании использует overlap + корректная индексация (PP7-фикс из v5).
    - cutoff автоматически ограничивается до (nyq - 1) для защиты от Найквиста.
    """
    if fps <= 0:
        raise ValueError("fps должен быть передан явно из metadata_extractor")

    video = np.asarray(video, dtype=np.float32)
    n_frames = video.shape[0]

    nyq = 0.5 * fps
    wn = min(cutoff / nyq, 0.99)  # Защита от частоты Найквиста
    sos = butter(4, wn, btype="low", output="sos")

    # Чистый путь — без чанкования
    if n_frames <= chunk_size:
        filtered = sosfiltfilt(sos, video, axis=0)
        if mask is not None:
            filtered[:, ~mask] = 0.0
        return filtered.astype(np.float32)

    # Чанкование с overlap (PP7-фикс: корректная индексация средних чанков)
    result = np.empty_like(video)
    step = chunk_size - overlap
    n_chunks = int(np.ceil(n_frames / step))

    for i in range(n_chunks):
        start = i * step
        end = min(start + chunk_size, n_frames)
        chunk = video[start:end]
        filtered_chunk = sosfiltfilt(sos, chunk, axis=0)

        if i == 0:
            keep = min(step, end - start)
            result[start : start + keep] = filtered_chunk[:keep]
        elif i == n_chunks - 1:
            result[start:end] = filtered_chunk
        else:
            keep_start = overlap // 2
            keep_end = chunk_size - overlap // 2
            keep_len = keep_end - keep_start
            dest_start = start + keep_start
            result[dest_start : dest_start + keep_len] = filtered_chunk[keep_start:keep_end]

    if mask is not None:
        result[:, ~mask] = 0.0

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# ASLS — коррекция базовой линии
# ---------------------------------------------------------------------------

def asls_baseline_correct_trace(
    y: np.ndarray,
    lam: float = 1e8,
    p: float = 0.01,
    niter: int = 3,
) -> np.ndarray:
    """
    Алгоритм ASLS (Asymmetric Least Squares Smoothing, Eilers & Boelens 2005)
    для одномерного сигнала.

    Уничтожает медленный дрейф изолинии (фотообесцвечивание, движение ткани).
    Параметр p=0.01 означает асимметрию: алгоритм ищет НИЖНЮЮ огибающую,
    поэтому сигнал обязан быть инвертирован (пики смотрят вверх) до вызова.

    Параметры:
        lam   — сглаженность базовой линии (больше = плавнее)
        p     — асимметрия (0.01 = нижняя огибающая)
        niter — число итераций (3 достаточно для большинства трейсов)
    """
    L = len(y)
    # Second-order difference matrix: shape (L-2, L)
    # Using eye/diff approach for correct (L, L) penalisation matrix
    from scipy.sparse import eye as speye
    E = speye(L, format="csc")
    D2 = (E[2:] - 2 * E[1:-1] + E[:-2]).tocsr()  # shape (L-2, L)
    DTD = lam * D2.T.dot(D2)                       # shape (L, L)

    w = np.ones(L, dtype=np.float64)
    z = y.copy().astype(np.float64)
    y64 = y.astype(np.float64)

    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L), format="csr")
        Z = W + DTD
        z = spsolve(Z, w * y64)
        w = p * (y64 > z) + (1 - p) * (y64 < z)

    return (y64 - z).astype(np.float32)


def asls_baseline_correct(
    video: np.ndarray,
    mask: np.ndarray,
    lam: float = 1e8,
    p: float = 0.01,
    niter: int = 3,
) -> np.ndarray:
    """
    Применяет ASLS-коррекцию ко всему видео пиксель-за-пикселем строго под маской.

    Требует mask — без неё обработка всего кадра займёт слишком много памяти и времени.
    Работает в float64 для численной стабильности ASLS, возвращает float32.
    """
    video_64 = video.astype(np.float64)
    ys, xs = np.where(mask)
    n_pixels = len(ys)

    logger.info(f"ASLS: обрабатываю {n_pixels} пикселей под маской...")
    for idx, (y, x) in enumerate(zip(ys, xs)):
        video_64[:, y, x] = asls_baseline_correct_trace(
            video_64[:, y, x], lam=lam, p=p, niter=niter
        )
        if (idx + 1) % 5000 == 0:
            logger.info(f"ASLS: {idx + 1}/{n_pixels} пикселей готово")

    return video_64.astype(np.float32)


# ---------------------------------------------------------------------------
# Нормализация
# ---------------------------------------------------------------------------

def normalize_traces(
    video: np.ndarray,
    method: str = "percentile",
    q: float = 10.0,
) -> np.ndarray:
    """
    Нормализация ΔF/F относительно базовой линии.

    method="percentile": F0 = percentile(q) по оси времени (рекомендуется)
    method="min":        F0 = min по оси времени
    """
    if method == "percentile":
        f0 = np.percentile(video, axis=0, q=q)
    elif method == "min":
        f0 = video.min(axis=0)
    else:
        raise ValueError(f"normalize_traces: неизвестный method='{method}'. Используй 'percentile' или 'min'.")

    f0 = np.where(np.abs(f0) < 1e-6, 1e-6, f0)  # Защита от деления на ноль
    return ((video.astype(np.float32) - f0) / f0).astype(np.float32)


# ---------------------------------------------------------------------------
# Главный оркестратор
# ---------------------------------------------------------------------------

def preprocess_video(
    video: np.ndarray,
    fps: float,
    mask: Optional[np.ndarray] = None,
    target_stage: str = "activation",
    lp_cutoff: Optional[float] = None,
    sigma: float = 2.0,
    chunk_size: int = 8192,
    overlap: int = 256,
    invert: Optional[bool] = None,
    sample_name: Optional[str] = None,
    dye: Optional[str] = None,
    recording_mode: Optional[str] = None,
    do_asls: bool = False,
    asls_lam: float = 1e8,
    asls_p: float = 0.01,
    asls_niter: int = 3,
    do_normalize: bool = False,
    normalize_method: str = "percentile",
    normalize_q: float = 10.0,
) -> np.ndarray:
    """
    Главный оркестратор предобработки оптических данных. Порядок операций критически важен.

    Порядок (v6):
        1. Spatial smooth (Gaussian, только H×W)
        2. Temporal LPF (Butterworth, zero-phase)
        3. Invert (СТРОГО ДО ASLS — пики должны смотреть вверх для ASLS)
        4. ASLS baseline correction (под маской)
        5. Normalize (ΔF/F, опционально)

    Параметры:
        video         — входное видео (T, H, W), любой числовой тип
        fps           — частота съёмки (Гц), обязательно из metadata_extractor
        mask          — булева маска ткани (H, W); обязательна при do_asls=True
        target_stage  — "activation" (80 Гц) или "apd" (150 Гц); игнорируется если lp_cutoff задан явно
        lp_cutoff     — явная частота среза (Гц); если None — берётся из target_stage
        sigma         — sigma Гауссова сглаживания (пикселей); 0 = отключить
        chunk_size    — размер чанка для LPF (кадров); 8192 = без чанкования для большинства записей
        overlap       — перекрытие чанков (кадров)
        invert        — True/False явно; None = авто-определение через should_invert()
        sample_name   — имя образца для авто-определения инверсии (напр. "005A")
        dye           — тип красителя ("A"/"B") для авто-определения инверсии
        recording_mode — "voltage"/"calcium" для авто-определения инверсии
        do_asls       — применять ASLS-коррекцию базовой линии (требует mask)
        asls_lam      — сглаженность ASLS (1e8 = плавная базовая линия)
        asls_p        — асимметрия ASLS (0.01 = нижняя огибающая)
        asls_niter    — число итераций ASLS
        do_normalize  — применять ΔF/F нормализацию
        normalize_method — метод нормализации ("percentile" | "min")
        normalize_q   — перцентиль для F0 при method="percentile"

    Возвращает:
        np.ndarray float32, форма (T, H, W)
    """
    if fps <= 0:
        raise ValueError("fps должен быть получен из metadata_extractor и быть > 0!")

    # --- 1. Авто-подбор частоты среза ---
    if lp_cutoff is None:
        stage_key = target_stage.lower()
        if stage_key not in _LP_CUTOFF_BY_STAGE:
            raise ValueError(
                f"target_stage='{target_stage}' неизвестен. "
                f"Допустимые значения: {list(_LP_CUTOFF_BY_STAGE.keys())}"
            )
        lp_cutoff = _LP_CUTOFF_BY_STAGE[stage_key]
        logger.info(f"preprocess_video: target_stage='{target_stage}' → lp_cutoff={lp_cutoff} Гц")

    # --- 2. Пространственное сглаживание ---
    if sigma > 0:
        video = spatial_smooth(video, sigma=sigma, mask=mask)
    else:
        video = np.asarray(video, dtype=np.float32)

    # --- 3. Временной LPF ---
    if lp_cutoff > 0:
        video = temporal_lowpass(
            video,
            fps=fps,
            cutoff=lp_cutoff,
            mask=mask,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    # --- 4. Инверсия (СТРОГО ДО ASLS) ---
    if invert is None:
        invert = should_invert(
            sample_name=sample_name,
            dye=dye,
            recording_mode=recording_mode,
        )
    if invert:
        video = -video
        logger.info("preprocess_video: инверсия применена (VSD/краситель A)")

    # --- 5. ASLS коррекция базовой линии ---
    if do_asls:
        if mask is None:
            raise ValueError(
                "do_asls=True требует передачи mask. "
                "Без маски ASLS обработает все пиксели — это займёт слишком много памяти и времени."
            )
        logger.info(f"preprocess_video: запуск ASLS (lam={asls_lam}, p={asls_p}, niter={asls_niter})")
        video = asls_baseline_correct(video, mask=mask, lam=asls_lam, p=asls_p, niter=asls_niter)

    # --- 6. Нормализация (опционально) ---
    if do_normalize:
        video = normalize_traces(video, method=normalize_method, q=normalize_q)

    return video.astype(np.float32)
