#!/usr/bin/env python3
"""
alternans.py — Математическое ядро анализа альтернанса (Stage 7).
Версия v1 (2026-07-02).

Функции:
  compute_spatial_alternans()    — карты амплитуды (мс, %) и фазы (+1/-1)
  compute_concordance_map()      — векторизованный индекс конкордантности (2D conv)
  compute_temporal_spectrum()    — FFT-спектр временного ряда, spectral_purity
  compute_poincare_correlation() — корреляция Пирсона beat_N vs beat_N+1

Все функции — чистая математика без I/O.
Вызываются из AlternansAgent (agents/alternans_agent.py).

Соответствие исходному utils_alternans.py:
  - compute_spatial_alternans    ← compute_spatial_alternans (без изменений)
  - compute_concordance_map      ← compute_concordance_map (без изменений)
  - compute_temporal_spectrum    ← compute_temporal_spectrum (без изменений)
  - compute_poincare_correlation ← inline-код из alternans_agent.py (вынесен сюда)
"""

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.signal import convolve2d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Пространственный альтернанс: амплитуда и фаза
# ---------------------------------------------------------------------------

def compute_spatial_alternans(
    apd3d: np.ndarray,
    mask: np.ndarray,
    sign_floor_ms: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Вычисляет пространственные карты амплитуды и фазы альтернанса.

    Алгоритм:
        1. Разности между соседними биениями: diffs[..., i] = APD[i+1] - APD[i]
        2. AC_ms  = mean(|diffs| / 2) по биениям — амплитуда в мс (пик-к-впадине / 2)
        3. AC_pct = mean(|diffs| / mean_pair × 100) — амплитуда в % от средней длит.
        4. phase_map = sign(mean(diffs)):
             +1 → Long-Short паттерн (первое биение длиннее)
             -1 → Short-Long паттерн
           Если mean(|diffs|) < sign_floor_ms → фаза NaN (шумовой пиксель)

    Параметры:
        apd3d        — (H, W, N_beats) карты длительностей по биениям
        mask         — (H, W) bool, маска ткани
        sign_floor_ms — порог ниже которого фаза считается неопределённой (мс)

    Возвращает:
        ac_ms      — (H, W) float32, амплитуда альтернанса в мс
        ac_pct     — (H, W) float32, амплитуда в % от средней длит.
        phase_map  — (H, W) float32, фаза (+1 / -1 / NaN)
    """
    # Разности соседних биений: shape (H, W, N_beats-1)
    diffs      = apd3d[..., 1:] - apd3d[..., :-1]
    mean_pairs = (apd3d[..., 1:] + apd3d[..., :-1]) / 2.0

    # Амплитуда в мс
    ac_ms = np.nanmean(np.abs(diffs) / 2.0, axis=-1).astype(np.float32)

    # Амплитуда в %
    with np.errstate(divide="ignore", invalid="ignore"):
        ac_pct = np.nanmean(
            (np.abs(diffs) / mean_pairs) * 100.0, axis=-1
        ).astype(np.float32)

    # Карта фазы
    phase_map = np.sign(np.nanmean(diffs, axis=-1)).astype(np.float32)

    # Зануляем шумовые пиксели и фон
    mean_abs_diff = np.nanmean(np.abs(diffs), axis=-1)
    phase_map[mean_abs_diff < sign_floor_ms] = np.nan
    phase_map[~mask] = np.nan

    return ac_ms, ac_pct, phase_map


# ---------------------------------------------------------------------------
# Индекс конкордантности (согласие фаз по 8 соседям)
# ---------------------------------------------------------------------------

def compute_concordance_map(
    phase_map: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Векторизованный расчёт индекса конкордантности по 8 соседям (2D свёртка).

    Интерпретация:
        1.0 — полный синхрон (все соседи в одной фазе) → безопасно
        ~0.5 — дискордантный альтернанс (соседи в противофазе) → опасно

    Алгоритм:
        Для каждого пикселя в фазе +1: concordance = count(+1 соседей) / count(valid соседей)
        Для каждого пикселя в фазе -1: concordance = count(-1 соседей) / count(valid соседей)

    Параметры:
        phase_map — (H, W) float32, фаза (+1 / -1 / NaN)
        mask      — (H, W) bool, маска ткани

    Возвращает:
        (H, W) float32, индекс конкордантности [0, 1] или NaN вне маски
    """
    kernel = np.ones((3, 3), dtype=float)

    phase_clean  = np.where(mask & np.isfinite(phase_map), phase_map, 0.0)
    pos          = (phase_clean == 1).astype(float)
    neg          = (phase_clean == -1).astype(float)
    valid_pixels = (mask & np.isfinite(phase_map)).astype(float)

    pos_count   = convolve2d(pos,          kernel, mode="same", boundary="fill", fillvalue=0)
    neg_count   = convolve2d(neg,          kernel, mode="same", boundary="fill", fillvalue=0)
    total_valid = convolve2d(valid_pixels, kernel, mode="same", boundary="fill", fillvalue=0)

    concordance = np.full_like(phase_map, np.nan, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        concordance[pos == 1] = (pos_count[pos == 1] / total_valid[pos == 1]).astype(np.float32)
        concordance[neg == 1] = (neg_count[neg == 1] / total_valid[neg == 1]).astype(np.float32)

    concordance[~mask] = np.nan
    return concordance


# ---------------------------------------------------------------------------
# FFT-спектр временного ряда
# ---------------------------------------------------------------------------

def compute_temporal_spectrum(
    temporal_series: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Спектральный Фурье-анализ временного ряда разностей биений.
    Ищет пик на частоте Найквиста (0.5 цикла/удар) — сигнатура альтернанса.

    Алгоритм:
        1. Zero-padding до ближайшей степени 2 (лучшее разрешение FFT)
        2. De-meaning (удаление постоянной составляющей)
        3. rfft → мощностной спектр
        4. spectral_purity = P(Nyquist) / P(total)

    Параметры:
        temporal_series — 1D массив разностей APD по биениям

    Возвращает:
        (spec, freqs, spectral_purity)
        spec            — мощностной спектр (или None если < 2 точек)
        freqs           — частоты (cycles/beat)
        spectral_purity — доля мощности на частоте Найквиста [0, 1]
    """
    n = len(temporal_series)
    if n < 2:
        return None, None, 0.0

    # Zero-padding до степени 2
    n_pad  = 2 ** int(np.ceil(np.log2(max(n, 2))))
    padded = np.zeros(n_pad)
    padded[:n] = temporal_series - np.mean(temporal_series)

    spec  = np.abs(np.fft.rfft(padded)) ** 2
    freqs = np.fft.rfftfreq(n_pad, d=1.0)

    purity = float(spec[-1] / np.sum(spec)) if np.sum(spec) > 0 else 0.0

    return spec, freqs, purity


# ---------------------------------------------------------------------------
# Корреляция Пирсона для диаграммы Пуанкаре
# ---------------------------------------------------------------------------

def compute_poincare_correlation(tissue_mean_apd: np.ndarray) -> float:
    """
    Корреляция Пирсона beat_N vs beat_N+1 (диаграмма Пуанкаре).

    Интерпретация:
        ≈ -1.0 — идеальный стабильный альтернанс (длинный↔короткий)
        ≈  0.0 — случайная вариация (нет паттерна)
        ≈ +1.0 — монотонный тренд (не альтернанс)

    Параметры:
        tissue_mean_apd — 1D массив медианных APD по биениям (N_beats,)

    Возвращает:
        float, корреляция Пирсона или 0.0 если < 2 пар
    """
    beat_n  = tissue_mean_apd[:-1]
    beat_n1 = tissue_mean_apd[1:]

    if len(beat_n) < 2:
        return 0.0

    valid = np.isfinite(beat_n) & np.isfinite(beat_n1)
    if valid.sum() < 2:
        return 0.0

    return float(np.corrcoef(beat_n[valid], beat_n1[valid])[0, 1])
