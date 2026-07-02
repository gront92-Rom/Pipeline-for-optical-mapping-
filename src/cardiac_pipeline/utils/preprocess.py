#!/usr/bin/env python3
"""
preprocess_v5.py — Стабильная версия предобработки (исправлен PP7)

Основные изменения v5:
- Исправлена индексация при записи средних чанков (больше нет сдвига на overlap//2)
- Увеличен дефолтный chunk_size до 8192 (большинство записей идут без чанкования)
- Оставлен zero-phase (sosfiltfilt)
- Чанкование теперь более консервативное и безопасное
"""

from typing import Optional
import numpy as np
import re
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfiltfilt


def _parse_sample_id(name: str) -> Optional[str]:
    m = re.search(r'(?<![0-9])(\d{3,4}[AB])(?:[_.\-]|$)', name, re.IGNORECASE)
    return m.group(1).upper() if m else None


def should_invert(
    sample_name: Optional[str] = None,
    dye: Optional[str] = None,
    recording_mode: Optional[str] = None
) -> Optional[bool]:
    if recording_mode is not None:
        rm = recording_mode.lower().strip()
        if rm in ("voltage", "vsd", "ap"):
            return True
        if rm in ("calcium", "cat", "ca"):
            return False
        return None

    if dye is not None:
        d = dye.upper().strip()
        if d in ("A", "VOLTAGE", "VSD"):
            return True
        if d in ("B", "CALCIUM", "CAT"):
            return False
        return None

    if sample_name:
        sid = _parse_sample_id(sample_name) or sample_name
        token = sid.upper().split("_")[0]
        if token.endswith("A"):
            return True
        if token.endswith("B"):
            return False
    return None


def spatial_smooth(
    video: np.ndarray,
    mask: Optional[np.ndarray] = None,
    sigma: float = 2.0
) -> np.ndarray:
    """Только Gaussian (vectorized)."""
    video = np.asarray(video, dtype=np.float32)
    # sigma=(0, s, s) applies spatial-only smoothing: 0 along time axis
    smooth = gaussian_filter(video, sigma=(0, sigma, sigma))
    if mask is not None:
        smooth[:, ~mask] = 0.0
    return smooth


def temporal_lowpass(
    video: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fps: Optional[float] = None,
    cutoff: float = 80.0,
    chunk_size: int = 8192,
    overlap: int = 256
) -> np.ndarray:
    """
    Zero-phase временная фильтрация (sosfiltfilt).

    - Если n_frames <= chunk_size: фильтрует весь массив сразу (рекомендуемый путь).
    - При чанковании используется overlap + корректная индексация (исправлен PP7).
    """
    if fps is None or fps <= 0:
        raise ValueError("fps должен быть передан явно из metadata_extractor")

    video = np.asarray(video, dtype=np.float32)
    n_frames = video.shape[0]

    nyq = 0.5 * fps
    if cutoff >= nyq:
        cutoff = nyq - 1.0

    sos = butter(4, cutoff / nyq, btype="low", output="sos")

    # Чистый путь — без чанкования
    if n_frames <= chunk_size:
        filtered = sosfiltfilt(sos, video, axis=0)
        if mask is not None:
            filtered[:, ~mask] = 0.0
        return filtered

    # Чанкование с overlap (исправленная версия)
    result = np.empty_like(video)
    step = chunk_size - overlap
    n_chunks = int(np.ceil(n_frames / step))

    for i in range(n_chunks):
        start = i * step
        end = min(start + chunk_size, n_frames)
        chunk = video[start:end]
        filtered_chunk = sosfiltfilt(sos, chunk, axis=0)

        if i == 0:
            # Первый чанк — берём до step
            keep = min(step, end - start)
            result[start : start + keep] = filtered_chunk[:keep]
        elif i == n_chunks - 1:
            # Последний чанк — берём целиком
            result[start:end] = filtered_chunk
        else:
            # Средние чанки — корректная индексация (исправлен сдвиг)
            keep_start = overlap // 2
            keep_end = chunk_size - overlap // 2
            keep_len = keep_end - keep_start

            dest_start = start + keep_start
            result[dest_start : dest_start + keep_len] = filtered_chunk[keep_start:keep_end]

    if mask is not None:
        result[:, ~mask] = 0.0

    return result


def normalize_traces(video: np.ndarray, method: str = "percentile", q: float = 10.0) -> np.ndarray:
    if method == "percentile":
        f0 = np.percentile(video, axis=0, q=q)
    elif method == "min":
        f0 = video.min(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    f0 = np.where(f0 < 1, 1, f0)
    return (video.astype(np.float32) - f0) / f0


def preprocess_video(
    video: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fps: Optional[float] = None,
    sigma: float = 2.0,
    lp_cutoff: float = 80.0,
    chunk_size: int = 8192,
    overlap: int = 256,
    invert: Optional[bool] = None,
    sample_name: Optional[str] = None,
    dye: Optional[str] = None,
    recording_mode: Optional[str] = None,
    do_normalize: bool = False
) -> np.ndarray:
    """
    Полный пайплайн предобработки v5.

    Рекомендация: используй chunk_size=8192 или выше.
    Для большинства записей будет использован чистый путь без чанкования.
    """
    if fps is None or fps <= 0:
        raise ValueError("fps должен быть получен из metadata_extractor!")

    video = spatial_smooth(video, mask=mask, sigma=sigma)

    video = temporal_lowpass(
        video,
        mask=mask,
        fps=fps,
        cutoff=lp_cutoff,
        chunk_size=chunk_size,
        overlap=overlap
    )

    if invert is None:
        inv = should_invert(sample_name=sample_name, dye=dye, recording_mode=recording_mode)
        if inv is None:
            print("⚠️  WARNING: Не удалось определить необходимость инверсии. "
                  "Укажите dye='A'/'B' или recording_mode. Инверсия НЕ применена.")
            inv = False
        invert = inv

    if invert:
        video = -video

    if do_normalize:
        video = normalize_traces(video)

    return video.astype(np.float32)
