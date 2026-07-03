#!/usr/bin/env python3
"""
signal.py — Математическое ядро расчёта APD/CaT (Stage 4).
Версия v1 (2026-07-02).

Функции:
  masked_spatial_pool()               — векторизованный 3×3 ROI-пулинг под маской
  find_upstroke_start()               — поиск начала апстрока назад от пика
  find_repol_crossing_with_fallback() — поиск реполяризации вперёд с адаптивным fallback
  get_4_corners_snapped()             — 4 угловых ROI-точки, примагниченных к ткани
  validate_apd_semantics()            — семантическая валидация физиологических диапазонов

Все функции — чистая математика без I/O и без хардкода fps.
Вызываются из APDAgent (agents/apd_agent.py).

Соответствие исходному utils_apd.py:
  - masked_spatial_pool        ← masked_spatial_pool (без изменений)
  - find_upstroke_start        ← find_upstroke_start (без изменений)
  - find_repol_crossing_with_fallback ← find_repol_crossing_with_fallback (без изменений)
  - get_4_corners_snapped      ← get_4_corners_snapped (без изменений)
  - validate_apd_semantics     ← validate_apd_semantics (перенесена из apd_agent.py)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Пространственный ROI-пулинг
# ---------------------------------------------------------------------------

def masked_spatial_pool(
    video: np.ndarray,
    mask: np.ndarray,
    size: int = 3,
) -> np.ndarray:
    """
    Векторизованное 3×3 ROI-усреднение для каждого пикселя кадра.

    Усредняет ТОЛЬКО пиксели внутри скользящего окна, принадлежащие маске ткани.
    Работает в сотни раз быстрее вложенных циклов.

    Параметры:
        video — (T, H, W) float32, препроцессированное видео
        mask  — (H, W) bool, маска ткани
        size  — размер квадратного окна (пикселей), по умолчанию 3

    Возвращает:
        (T, H, W) float32 — сглаженное видео (вне маски = 0)
    """
    # Обнуляем фон, чтобы он не вносил искажений в сумму
    video_masked = video * mask[np.newaxis, :, :]

    # uniform_filter: spatial-only (размер 1 по оси времени, size по H и W)
    smoothed_video = uniform_filter(video_masked, size=(1, size, size), mode='constant')
    smoothed_mask  = uniform_filter(mask.astype(float), size=(size, size), mode='constant')

    # Исключаем деление на ноль для пикселей вне ткани
    valid_mask = smoothed_mask > 0
    roi_video = np.zeros_like(video)
    roi_video[:, valid_mask] = smoothed_video[:, valid_mask] / smoothed_mask[valid_mask]

    return roi_video


# ---------------------------------------------------------------------------
# Поиск апстрока
# ---------------------------------------------------------------------------

def find_upstroke_start(
    trace: np.ndarray,
    peak: int,
    amp: float,
    fps: float,
    primary_frac: float = 0.30,
    fallback_frac: float = 0.50,
    search_back_ms: float = 50.0,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Поиск начала апстрока назад во времени от пика с суб-кадровой интерполяцией.

    Алгоритм:
        1. Ищем первое пересечение уровня primary_frac × amp при движении назад от peak.
        2. Если не найдено — повторяем с fallback_frac.
        3. Суб-кадровая интерполяция линейным методом.

    Параметры:
        trace           — 1D трейс пикселя
        peak            — индекс локального пика (кадр)
        amp             — амплитуда пика
        fps             — частота съёмки (Гц)
        primary_frac    — первичный уровень поиска (доля амплитуды)
        fallback_frac   — запасной уровень поиска
        search_back_ms  — окно поиска назад (мс)

    Возвращает:
        (upstroke_idx, frac_used, method_tag)
        upstroke_idx — суб-кадровый индекс начала апстрока (или None)
        frac_used    — использованная доля амплитуды (или None)
        method_tag   — строка-метка метода
    """
    if amp <= 0:
        return None, None, "no_amp"

    ws = max(0, peak - int(search_back_ms * fps / 1000))

    for frac in [primary_frac, fallback_frac]:
        level = frac * amp
        for j in range(peak, ws, -1):
            if trace[j] < level:
                if j + 1 < len(trace):
                    denom = trace[j + 1] - trace[j]
                    f = (level - trace[j]) / (denom if denom != 0 else 1e-12)
                    return j + f, frac, f"upstroke_{int(frac * 100)}pct"
                return float(j), frac, f"upstroke_{int(frac * 100)}pct"

    return None, None, "no_crossing"


# ---------------------------------------------------------------------------
# Поиск реполяризации с адаптивным fallback
# ---------------------------------------------------------------------------

def find_repol_crossing_with_fallback(
    trace: np.ndarray,
    peak_idx: int,
    amp: float,
    fps: float,
    threshold: int = 80,
    next_peak_idx: Optional[int] = None,
    total_frames: Optional[int] = None,
) -> Tuple[Optional[float], bool, str]:
    """
    Поиск точки реполяризации вперёд от пика с адаптивным расширением окна.

    Алгоритм:
        1. Стандартный поиск: ищем пересечение уровня (1 - threshold/100) × amp
           в диапазоне [peak_idx+1, next_peak_idx).
        2. Fallback: если не найдено — расширяем окно на 30% длины биения
           (помогает при затяжном плато CaT).
        3. Суб-кадровая интерполяция линейным методом.

    Параметры:
        trace         — 1D трейс пикселя
        peak_idx      — индекс локального пика (кадр)
        amp           — амплитуда пика
        fps           — частота съёмки (не используется напрямую, передаётся для совместимости)
        threshold     — уровень реполяризации (%, напр. 80 = APD80)
        next_peak_idx — индекс следующего глобального пика (граница биения)
        total_frames  — общее число кадров (T)

    Возвращает:
        (repol_idx, found, method_tag)
        repol_idx  — суб-кадровый индекс реполяризации (или None)
        found      — True если пересечение найдено
        method_tag — "standard" | "fallback_extended" | "not_found" | "no_amp"
    """
    if amp <= 0:
        return None, False, "no_amp"

    level = amp * (1.0 - threshold / 100.0)
    end_idx = next_peak_idx if next_peak_idx is not None else total_frames

    # 1. Стандартный поиск в рамках текущего биения
    for i in range(peak_idx + 1, end_idx):
        if trace[i] <= level:
            if i > peak_idx + 1:
                denom = trace[i] - trace[i - 1]
                f = (level - trace[i - 1]) / (denom if denom != 0 else 1e-12)
                return (i - 1) + f, True, "standard"
            return float(i), True, "standard"

    # 2. Fallback: динамическое удлинение окна
    if next_peak_idx is not None and total_frames is not None:
        beat_interval = next_peak_idx - peak_idx
        extended_end = min(total_frames, next_peak_idx + int(beat_interval * 0.30))
        for i in range(end_idx, extended_end):
            if trace[i] <= level:
                denom = trace[i] - trace[i - 1]
                f = (level - trace[i - 1]) / (denom if denom != 0 else 1e-12)
                return (i - 1) + f, True, "fallback_extended"

    return None, False, "not_found"


# ---------------------------------------------------------------------------
# Угловые ROI-точки
# ---------------------------------------------------------------------------

def get_4_corners_snapped(
    mask: np.ndarray,
    padding: int = 10,
) -> List[Dict[str, Any]]:
    """
    Находит 4 экстремальные угловые точки маски с отступом padding пикселей от BBox.
    Каждая точка «примагничивается» к ближайшему живому пикселю под маской.

    Параметры:
        mask    — (H, W) bool, маска ткани
        padding — отступ от краёв BBox (пикселей)

    Возвращает:
        Список из 4 словарей: {"label": str, "y": int, "x": int}
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    targets = [
        ("Top-Left",     y_min + padding, x_min + padding),
        ("Top-Right",    y_min + padding, x_max - padding),
        ("Bottom-Left",  y_max - padding, x_min + padding),
        ("Bottom-Right", y_max - padding, x_max - padding),
    ]

    corners = []
    for name, yt, xt in targets:
        dists = (ys - yt) ** 2 + (xs - xt) ** 2
        best_idx = int(np.argmin(dists))
        corners.append({
            "label": name,
            "y": int(ys[best_idx]),
            "x": int(xs[best_idx]),
        })
    return corners


# ---------------------------------------------------------------------------
# Семантическая валидация APD
# ---------------------------------------------------------------------------

def validate_apd_semantics(
    apd80_med: float,
    apd30_med: float,
    dye: str,
    vsd_apd80_range: Tuple[float, float] = (20.0, 300.0),
    cat_apd80_range: Tuple[float, float] = (30.0, 500.0),
) -> Tuple[str, str]:
    """
    Семантический контроль физиологических диапазонов (крысиный желудочек).

    Правила:
        1. apd80_med и apd30_med должны быть конечными и > 0.
        2. APD80 должен попадать в физиологический диапазон для данного красителя.
        3. APD80 / APD30 >= 1.0 (морфологически обязательно).

    Параметры:
        apd80_med       — медианный APD80 по ткани (мс)
        apd30_med       — медианный APD30 по ткани (мс)
        dye             — "A" (VSD/вольтаж) или "B" (Ca²⁺)
        vsd_apd80_range — допустимый диапазон APD80 для VSD (мс)
        cat_apd80_range — допустимый диапазон CaT для Ca²⁺ (мс)

    Возвращает:
        (verdict, reason) — "PASS"/"FAIL" и строка-объяснение
    """
    if not np.isfinite(apd80_med) or not np.isfinite(apd30_med) or apd30_med <= 0:
        return "FAIL", "Non-finite values or zero baseline detected"

    ratio = apd80_med / apd30_med

    if dye == "A":
        lo, hi = vsd_apd80_range
        if not (lo <= apd80_med <= hi):
            return "FAIL", f"VSD APD80 out of biological boundaries: {apd80_med:.1f} ms (expected {lo}–{hi})"
    elif dye == "B":
        lo, hi = cat_apd80_range
        if not (lo <= apd80_med <= hi):
            return "FAIL", f"CaT Duration out of biological boundaries: {apd80_med:.1f} ms (expected {lo}–{hi})"

    if ratio < 1.0:
        return "FAIL", f"Physiologically impossible morphology ratio (APD80/APD30 = {ratio:.2f} < 1.0)"

    return "PASS", "Passed semantic validation rules"
