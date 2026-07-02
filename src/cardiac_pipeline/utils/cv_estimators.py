#!/usr/bin/env python3
"""
cv_estimators.py — Математическое ядро расчёта скорости проведения (CV).

Два метода, оба принимают activation_map (мс), mask (bool), pixel_size_mm (мм/пиксель):

  compute_hybrid_structure_tensor(activation_map, mask, pixel_size_mm)
      Прямой градиент карты активации: CV = 1 / |∇T|.
      Быстро, без сглаживания — точен при высоком SNR карты.

  compute_polynomial_bayly(activation_map, mask, pixel_size_mm, window_size=5)
      Градиент по Гаусс-сглаженной поверхности (аппроксимация метода Бейли).
      Устойчив к шуму, рекомендуется как основной метод.

Единицы:
  - activation_map : мс (время активации)
  - pixel_size_mm  : мм/пиксель
  - CV на выходе   : мм/мс = м/с  (физиологически: 0.05–2.0 м/с для миокарда)

Ограничения физиологического диапазона передаются снаружи через cv_min / cv_max
(берутся из config.conduction.cv_min_m_per_s / cv_max_m_per_s).

Исправления относительно кора (2026-07-02):
  - Единицы: pixel_size_mm в знаменателе → CV в мм/мс = м/с (не произвольных единицах)
  - cv_map clip по cv_min/cv_max вынесен в параметры (не хардкод 2000)
  - np.warnings (deprecated) заменён на warnings.catch_warnings
  - Добавлены возвращаемые значения: cv_map, angles, coherence_map (для debug)
  - Добавлена функция estimate_cv_stats() для агрегации по маске
"""

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Физиологические константы (fallback — основные берутся из конфига агента)
# ---------------------------------------------------------------------------
_CV_MIN_DEFAULT = 0.05   # м/с
_CV_MAX_DEFAULT = 2.0    # м/с


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _safe_gradient(
    activation_map: np.ndarray,
    mask: np.ndarray,
    pixel_size_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет пространственный градиент карты активации.

    Перед расчётом NaN-пиксели вне маски заменяются на nanmean,
    чтобы np.gradient не распространял NaN на соседей.
    После расчёта градиент вне маски обнуляется.

    Возвращает (dy, dx) в единицах мс/мм.
    """
    # Заполняем NaN вне маски средним по маске (не влияет на результат внутри)
    fill_val = float(np.nanmean(activation_map[mask])) if mask.any() else 0.0
    filled = np.where(mask, activation_map, fill_val)
    filled = np.where(np.isfinite(filled), filled, fill_val)

    # Градиент в мс/пиксель
    dy_px, dx_px = np.gradient(filled)

    # Перевод в мс/мм: делим на pixel_size_mm
    dy = dy_px / pixel_size_mm
    dx = dx_px / pixel_size_mm

    # Обнуляем вне маски
    dy = np.where(mask, dy, 0.0)
    dx = np.where(mask, dx, 0.0)

    return dy, dx


def _cv_from_gradient(
    dy: np.ndarray,
    dx: np.ndarray,
    mask: np.ndarray,
    cv_min: float,
    cv_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CV = 1 / |∇T|  (мм/мс = м/с).

    Возвращает (cv_map, angles).
    Пиксели вне физиологического диапазона → NaN.
    """
    epsilon = 1e-9
    grad_mag = np.sqrt(dx ** 2 + dy ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        cv_map = np.where(grad_mag > epsilon, 1.0 / grad_mag, np.nan)

    # Физиологический клип
    cv_map = np.where(
        mask & np.isfinite(cv_map) & (cv_map >= cv_min) & (cv_map <= cv_max),
        cv_map,
        np.nan,
    )

    angles = np.arctan2(dy, dx)
    angles = np.where(mask, angles, np.nan)

    return cv_map, angles


# ---------------------------------------------------------------------------
# Метод 1: Прямой градиент (структурный тензор / hybrid)
# ---------------------------------------------------------------------------

def compute_hybrid_structure_tensor(
    activation_map: np.ndarray,
    mask: np.ndarray,
    pixel_size_mm: float,
    cv_min: float = _CV_MIN_DEFAULT,
    cv_max: float = _CV_MAX_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Скорость проведения через прямой градиент карты активации.

    Параметры
    ----------
    activation_map : np.ndarray (H, W)
        Карта времён активации в мс. NaN вне маски допустимы.
    mask : np.ndarray (H, W), bool
        Маска ткани.
    pixel_size_mm : float
        Размер пикселя в мм (из metadata.json).
    cv_min, cv_max : float
        Физиологические границы CV в м/с (= мм/мс).

    Возвращает
    ----------
    cv_map : np.ndarray (H, W)
        Карта CV в м/с. NaN вне маски и за пределами диапазона.
    angles : np.ndarray (H, W)
        Направление вектора скорости (рад).
    coherence_map : np.ndarray (H, W)
        Псевдо-когерентность: 1 / (1 + |∇²T|·σ²) — для debug.
    """
    if pixel_size_mm <= 0:
        raise ValueError(f"pixel_size_mm должен быть > 0, получено {pixel_size_mm}")
    if not mask.any():
        H, W = activation_map.shape
        nan_map = np.full((H, W), np.nan)
        return nan_map, nan_map.copy(), nan_map.copy()

    dy, dx = _safe_gradient(activation_map, mask, pixel_size_mm)
    cv_map, angles = _cv_from_gradient(dy, dx, mask, cv_min, cv_max)

    # Псевдо-когерентность (для debug/визуализации)
    grad_mag = np.sqrt(dx ** 2 + dy ** 2)
    coherence_map = np.where(mask, 1.0 / (1.0 + grad_mag), np.nan)

    return cv_map, angles, coherence_map


# ---------------------------------------------------------------------------
# Метод 2: Полиномиальный (Гаусс-сглаженный градиент, аппроксимация Бейли)
# ---------------------------------------------------------------------------

def compute_polynomial_bayly(
    activation_map: np.ndarray,
    mask: np.ndarray,
    pixel_size_mm: float,
    window_size: float = 5.0,
    cv_min: float = _CV_MIN_DEFAULT,
    cv_max: float = _CV_MAX_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Скорость проведения через Гаусс-сглаженный градиент (метод Бейли).

    Сглаживание снижает влияние шума карты активации.
    sigma = window_size / 3.0 (правило трёх сигм).

    Параметры
    ----------
    activation_map : np.ndarray (H, W)
        Карта времён активации в мс.
    mask : np.ndarray (H, W), bool
    pixel_size_mm : float
    window_size : float
        Размер окна сглаживания в пикселях (sigma = window_size / 3).
    cv_min, cv_max : float

    Возвращает
    ----------
    cv_map, angles, coherence_map — аналогично compute_hybrid_structure_tensor.
    """
    if pixel_size_mm <= 0:
        raise ValueError(f"pixel_size_mm должен быть > 0, получено {pixel_size_mm}")
    if not mask.any():
        H, W = activation_map.shape
        nan_map = np.full((H, W), np.nan)
        return nan_map, nan_map.copy(), nan_map.copy()

    sigma = max(window_size / 3.0, 0.5)

    # Заполняем NaN перед сглаживанием
    fill_val = float(np.nanmean(activation_map[mask]))
    filled = np.where(mask & np.isfinite(activation_map), activation_map, fill_val)

    smoothed = gaussian_filter(filled, sigma=sigma)

    # Градиент сглаженной поверхности
    dy_px, dx_px = np.gradient(smoothed)
    dy = np.where(mask, dy_px / pixel_size_mm, 0.0)
    dx = np.where(mask, dx_px / pixel_size_mm, 0.0)

    cv_map, angles = _cv_from_gradient(dy, dx, mask, cv_min, cv_max)

    # Когерентность: отношение сглаженного к исходному градиенту
    dy_raw, dx_raw = _safe_gradient(activation_map, mask, pixel_size_mm)
    raw_mag = np.sqrt(dx_raw ** 2 + dy_raw ** 2)
    smooth_mag = np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence_map = np.where(
            mask & (raw_mag > 1e-9),
            np.clip(smooth_mag / (raw_mag + 1e-9), 0.0, 1.0),
            np.nan,
        )

    return cv_map, angles, coherence_map


# ---------------------------------------------------------------------------
# Агрегация статистики по маске
# ---------------------------------------------------------------------------

def estimate_cv_stats(
    cv_map: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """
    Рассчитывает сводную статистику CV по маске ткани.

    Возвращает словарь:
      cv_median_m_per_s, cv_mean_m_per_s, cv_sd_m_per_s,
      cv_p25_m_per_s, cv_p75_m_per_s,
      valid_pixels, total_pixels, valid_fraction
    """
    if not mask.any():
        return {
            "cv_median_m_per_s": None,
            "cv_mean_m_per_s":   None,
            "cv_sd_m_per_s":     None,
            "cv_p25_m_per_s":    None,
            "cv_p75_m_per_s":    None,
            "valid_pixels":      0,
            "total_pixels":      int(mask.sum()),
            "valid_fraction":    0.0,
        }

    vals = cv_map[mask]
    valid = vals[np.isfinite(vals)]
    total = int(mask.sum())
    n_valid = int(len(valid))

    if n_valid == 0:
        return {
            "cv_median_m_per_s": None,
            "cv_mean_m_per_s":   None,
            "cv_sd_m_per_s":     None,
            "cv_p25_m_per_s":    None,
            "cv_p75_m_per_s":    None,
            "valid_pixels":      0,
            "total_pixels":      total,
            "valid_fraction":    0.0,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p25, p50, p75 = float(np.percentile(valid, 25)), float(np.percentile(valid, 50)), float(np.percentile(valid, 75))

    return {
        "cv_median_m_per_s": round(p50, 4),
        "cv_mean_m_per_s":   round(float(np.mean(valid)), 4),
        "cv_sd_m_per_s":     round(float(np.std(valid)), 4),
        "cv_p25_m_per_s":    round(p25, 4),
        "cv_p75_m_per_s":    round(p75, 4),
        "valid_pixels":      n_valid,
        "total_pixels":      total,
        "valid_fraction":    round(n_valid / total, 4),
    }
