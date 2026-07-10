#!/usr/bin/env python3
"""
cv_estimators.py — Математическое ядро расчёта скорости проведения (CV).

Методы (оба принимают activation_map (мс), mask (bool), pixel_size_mm (мм/пиксель)):

  compute_gradient_angular(activation_map, mask, pixel_size_mm)    ← PRIMARY
      Gradient + angular histogram → CVL, CVT, anisotropy, fiber angle.
      Порог |∇T| > grad_threshold выкидывает плоские пиксели.
      Работает на картах с малым TAT (ступенчатые карты).

  compute_structure_tensor(activation_map, mask, pixel_size_mm)   ← FALLBACK
      Structure tensor eigenvectors → CVL, CVT, anisotropy, fiber angle, coherence.
      Требует плавную карту (большой TAT). На ступенчатых картах — NaN/∞.

  compute_polynomial_bayly(activation_map, mask, pixel_size_mm)   ← legacy
      Гаусс-сглаженный градиент.

Единицы:
  - activation_map : мс
  - pixel_size_mm  : мм/пиксель
  - CV на выходе   : мм/мс = м/с

Возвращает единый dict для каждого метода:
  cv_map, cvl_map, cvt_map, anisotropy_map, fiber_angle_map,
  coherence_map, vx, vy,
  cvl_m_s, cvt_m_s, anisotropy_ratio, fiber_angle_deg, fiber_coherence,
  n_valid, cv_vs_angle (для gradient), n_sources
"""

import logging
import warnings
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    sobel,
    binary_erosion,
    minimum_filter,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Физиологические константы
# ---------------------------------------------------------------------------
_CV_MIN_DEFAULT = 0.05   # м/с
_CV_MAX_DEFAULT = 2.0    # м/с
_GRAD_THRESHOLD_DEFAULT = 0.5  # ms/mm — порог |∇T| для gradient method


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def _safe_gradient(
    activation_map: np.ndarray,
    mask: np.ndarray,
    pixel_size_mm: float,
) -> tuple:
    """Градиент карты активации. Возвращает (dy, dx) в мс/мм."""
    fill_val = float(np.nanmean(activation_map[mask])) if mask.any() else 0.0
    filled = np.where(mask, activation_map, fill_val)
    filled = np.where(np.isfinite(filled), filled, fill_val)
    dy_px, dx_px = np.gradient(filled)
    dy = dy_px / pixel_size_mm
    dx = dx_px / pixel_size_mm
    dy = np.where(mask, dy, 0.0)
    dx = np.where(mask, dx, 0.0)
    return dy, dx


def _cv_from_gradient(
    dy: np.ndarray,
    dx: np.ndarray,
    mask: np.ndarray,
    cv_min: float,
    cv_max: float,
) -> tuple:
    """CV = 1/|∇T|. Возвращает (cv_map, angles, vx, vy)."""
    epsilon = 1e-9
    grad_mag = np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_map = np.where(grad_mag > epsilon, 1.0 / grad_mag, np.nan)
    valid = mask & np.isfinite(cv_map) & (cv_map >= cv_min) & (cv_map <= cv_max)
    cv_map = np.where(valid, cv_map, np.nan)
    angles = np.arctan2(dy, dx)
    angles = np.where(valid, angles, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        vx = np.where(valid & (grad_mag > epsilon), dx / (grad_mag ** 2), np.nan)
        vy = np.where(valid & (grad_mag > epsilon), dy / (grad_mag ** 2), np.nan)
    return cv_map, angles, vx, vy


def find_activation_sources(
    activation_map: np.ndarray,
    mask: np.ndarray,
    min_distance_px: int = 5,
) -> np.ndarray:
    """Find local minima (focal sources) in activation map."""
    act_filled = np.where(np.isnan(activation_map) | ~mask, np.inf, activation_map)
    local_min = act_filled == minimum_filter(act_filled, size=min_distance_px)
    local_min &= mask & np.isfinite(activation_map)
    if mask.any():
        threshold = np.nanpercentile(activation_map[mask], 25)
        local_min &= activation_map < threshold
    sources = np.argwhere(local_min)
    return sources


# ---------------------------------------------------------------------------
# Метод PRIMARY: Gradient angular distribution
# ---------------------------------------------------------------------------

def compute_gradient_angular(
    activation_map: np.ndarray,
    mask: np.ndarray,
    pixel_size_mm: float,
    cv_min: float = _CV_MIN_DEFAULT,
    cv_max: float = _CV_MAX_DEFAULT,
    grad_threshold: float = _GRAD_THRESHOLD_DEFAULT,
    smooth_sigma: float = 1.5,
    n_bins: int = 18,
    erode_iterations: int = 3,
) -> Dict:
    """
    CV from gradient angular distribution.

    Логика:
      1. Gaussian smooth activation map (σ=smooth_sigma)
      2. np.gradient → gx, gy → |∇T|
      3. CV = 1/|∇T| где |∇T| > grad_threshold (выкидывает плоские)
      4. Clip CV [cv_min, cv_max]
      5. Erode mask (убрать края)
      6. Angular histogram: 18 бинов по 10°
      7. CVL = max(median CV по бинам) = самое быстрое направление
      8. CVT = perpendicular бин (±90°)
      9. Anisotropy = CVL/CVT

    Возвращает dict со всеми картами и скалярами.
    """
    H, W = activation_map.shape
    nan_map = np.full((H, W), np.nan)

    if not mask.any() or pixel_size_mm <= 0:
        return _empty_result(H, W)

    # 1. Smooth
    act_filled = np.where(mask & np.isfinite(activation_map),
                          activation_map, np.nanmean(activation_map[mask]))
    act = gaussian_filter(act_filled, sigma=smooth_sigma)

    # 2. Gradients via np.gradient (central difference, 2nd-order accurate)
    #    NOTE: Sobel inflates |∇T| by ~√2 due to [1,2,1] kernel → CV underestimated.
    #    np.gradient gives raw central difference — matches old hybrid_structure_tensor.
    gy_px, gx_px = np.gradient(act)
    gy = gy_px / pixel_size_mm  # ms/mm
    gx = gx_px / pixel_size_mm

    grad_mag = np.hypot(gx, gy)  # |∇T| ms/mm

    # 3. CV = 1/|∇T|, only where |∇T| > threshold
    cv_map = np.zeros_like(grad_mag)
    valid_grad = grad_mag > grad_threshold
    cv_map[valid_grad] = 1.0 / grad_mag[valid_grad]
    cv_map[~mask] = np.nan
    cv_map[(cv_map < cv_min) | (cv_map > cv_max)] = np.nan

    # 4. Erode mask (убрать края → градиент на границе ненадёжный)
    inner = binary_erosion(mask, iterations=erode_iterations) & np.isfinite(cv_map)

    # 5. Angles [0, π) — направление распространения
    angles_raw = (np.arctan2(-gy, -gx) % np.pi)
    angles_map = np.where(inner, angles_raw, np.nan)

    # 6. Vector field (direction only, normalized)
    vx = np.where(inner, -gx / (grad_mag + 1e-12), np.nan)
    vy = np.where(inner, -gy / (grad_mag + 1e-12), np.nan)

    # 7. Angular histogram
    angles_inner = angles_raw[inner]
    speeds_inner = cv_map[inner]
    good = (speeds_inner > cv_min) & (speeds_inner < cv_max)
    angles_inner = angles_inner[good]
    speeds_inner = speeds_inner[good]

    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cv_vs_angle = np.full(n_bins, np.nan)

    for i in range(n_bins):
        in_bin = (angles_inner >= bin_edges[i]) & (angles_inner < bin_edges[i + 1])
        if in_bin.sum() > 3:
            cv_vs_angle[i] = np.median(speeds_inner[in_bin])

    # 8. CVL = max direction, CVT = perpendicular
    if np.any(np.isfinite(cv_vs_angle)):
        cvl_idx = int(np.nanargmax(cv_vs_angle))
        cvl = float(cv_vs_angle[cvl_idx])
        cvt_idx = (cvl_idx + n_bins // 2) % n_bins
        cvt_val = cv_vs_angle[cvt_idx]
        cvt = float(cvt_val) if np.isfinite(cvt_val) else float(np.nanmin(cv_vs_angle[np.isfinite(cv_vs_angle)]))
        fiber_angle_rad = float(bin_centers[cvl_idx])
    else:
        cvl = np.nan
        cvt = np.nan
        fiber_angle_rad = np.nan

    aniso = cvl / (cvt + 1e-9) if np.isfinite(cvl) and np.isfinite(cvt) and cvt > 0 else np.nan

    # 9. CVL/CVT maps: только для ST (gradient даёт только скаляры из гистограммы)
    # Глобальное направление волокна не подходит для per-pixel карт при неоднородной ткани.
    # cvl_map/cvt_map для gradient = NaN (используются скаляры cvl_m_s/cvt_m_s)
    cvl_map = nan_map.copy()
    cvt_map = nan_map.copy()
    anisotropy_map = nan_map.copy()

    # 10. Pseudo-coherence
    coherence_map = np.where(mask, 1.0 / (1.0 + grad_mag), np.nan)

    # 11. Sources
    sources = find_activation_sources(activation_map, mask)

    n_valid = int(np.sum(np.isfinite(cv_map) & mask))

    return {
        "method": "gradient_angular",
        "cv_map": cv_map,
        "cvl_map": cvl_map,
        "cvt_map": cvt_map,
        "anisotropy_map": anisotropy_map,
        "fiber_angle_map": angles_map,
        "coherence_map": coherence_map,
        "vx": vx,
        "vy": vy,
        "cvl_m_s": cvl,
        "cvt_m_s": cvt,
        "anisotropy_ratio": aniso,
        "fiber_angle_deg": float(np.degrees(fiber_angle_rad)) if np.isfinite(fiber_angle_rad) else np.nan,
        "fiber_coherence": float(np.nanmedian(coherence_map[inner])) if inner.any() else np.nan,
        "n_valid": n_valid,
        "cv_vs_angle": cv_vs_angle,
        "bin_centers": bin_centers,
        "n_sources": int(len(sources)),
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Метод FALLBACK: Structure tensor
# ---------------------------------------------------------------------------

def compute_structure_tensor(
    activation_map: np.ndarray,
    mask: np.ndarray,
    pixel_size_mm: float,
    cv_min: float = _CV_MIN_DEFAULT,
    cv_max: float = _CV_MAX_DEFAULT,
    local_sigma: float = 2.0,
    integration_sigma: float = 4.0,
    erode_iterations: int = 3,
) -> Dict:
    """
    Per-pixel CV from structure tensor eigenvectors.

    λ_max = быстрый градиент → CVT = 1/√λ_max (медленное проведение)
    λ_min = медленный градиент → CVL = 1/√λ_min (быстрое проведение)

    Требует плавную карту активации. На ступенчатых картах → NaN/∞.
    """
    H, W = activation_map.shape
    nan_map = np.full((H, W), np.nan)

    if not mask.any() or pixel_size_mm <= 0:
        return _empty_result(H, W)

    # Fill NaN + smooth
    fill_val = float(np.nanmean(activation_map[mask]))
    act_filled = np.where(mask & np.isfinite(activation_map), activation_map, fill_val)
    act = gaussian_filter(act_filled, sigma=local_sigma)

    # Gradients
    gy, gx = np.gradient(act, pixel_size_mm)

    # Structure tensor components (smoothed)
    Jxx = gaussian_filter(gx * gx, sigma=integration_sigma)
    Jyy = gaussian_filter(gy * gy, sigma=integration_sigma)
    Jxy = gaussian_filter(gx * gy, sigma=integration_sigma)

    # Eigenvalues
    trace = Jxx + Jyy
    diff = Jxx - Jyy
    discrim = np.sqrt(np.maximum(diff ** 2 + 4 * Jxy ** 2, 0))
    lam_max = 0.5 * (trace + discrim)
    lam_min = 0.5 * (trace - discrim)

    # CV maps: 1/√λ
    with np.errstate(divide="ignore", invalid="ignore"):
        cvt_map_raw = 1.0 / np.sqrt(np.maximum(lam_max, 1e-12))
        cvl_map_raw = 1.0 / np.sqrt(np.maximum(lam_min, 1e-12))

    # Fiber angle
    theta_grad = 0.5 * np.arctan2(2 * Jxy, diff)
    fiber_angle = theta_grad + np.pi / 2

    # Coherence
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.where(
            lam_max + lam_min > 1e-12,
            ((lam_max - lam_min) / (lam_max + lam_min)) ** 2, 0
        )

    # Erode + clip
    inner = binary_erosion(mask, iterations=erode_iterations)
    valid_cv = (cvl_map_raw > cv_min) & (cvl_map_raw < cv_max) & inner

    cvl_map = np.where(valid_cv, cvl_map_raw, np.nan)
    cvt_map = np.where(valid_cv, cvt_map_raw, np.nan)
    fiber_angle_map = np.where(inner, fiber_angle, np.nan)
    coherence_map = np.where(inner, coherence, np.nan)

    # Anisotropy map
    with np.errstate(divide="ignore", invalid="ignore"):
        aniso_map = np.where(
            valid_cv & (cvt_map > 0),
            cvl_map / cvt_map, np.nan
        )

    # Scalar CV map: use cvl_map (fastest direction) as the main CV map
    cv_map = cvl_map.copy()

    # Vector field (direction only, normalized)
    with np.errstate(divide="ignore", invalid="ignore"):
        vx = np.where(valid_cv, gx / (np.sqrt(gx**2 + gy**2) + 1e-12), np.nan)
        vy = np.where(valid_cv, gy / (np.sqrt(gx**2 + gy**2) + 1e-12), np.nan)

    n_valid = int(valid_cv.sum())

    return {
        "method": "structure_tensor",
        "cv_map": cv_map,
        "cvl_map": cvl_map,
        "cvt_map": cvt_map,
        "anisotropy_map": aniso_map,
        "fiber_angle_map": fiber_angle_map,
        "coherence_map": coherence_map,
        "vx": vx,
        "vy": vy,
        "cvl_m_s": float(np.nanmedian(cvl_map_raw[valid_cv])) if n_valid > 0 else np.nan,
        "cvt_m_s": float(np.nanmedian(cvt_map_raw[valid_cv])) if n_valid > 0 else np.nan,
        "anisotropy_ratio": float(np.nanmedian(aniso_map[valid_cv & np.isfinite(aniso_map)])) if n_valid > 0 else np.nan,
        "fiber_angle_deg": float(np.degrees(np.nanmedian(fiber_angle[inner]))) if inner.any() else np.nan,
        "fiber_coherence": float(np.nanmedian(coherence[inner])) if inner.any() else np.nan,
        "n_valid": n_valid,
        "n_sources": 0,
        "sources": np.array([]),
    }


# ---------------------------------------------------------------------------
# Legacy: polynomial Bayly (still used as 2nd opinion)
# ---------------------------------------------------------------------------

def compute_polynomial_bayly(
    activation_map: np.ndarray,
    mask: np.ndarray,
    pixel_size_mm: float,
    window_size: float = 5.0,
    cv_min: float = _CV_MIN_DEFAULT,
    cv_max: float = _CV_MAX_DEFAULT,
) -> tuple:
    """Legacy: Гаусс-сглаженный градиент. Возвращает (cv_map, angles, coherence, vx, vy)."""
    if pixel_size_mm <= 0 or not mask.any():
        H, W = activation_map.shape
        nan = np.full((H, W), np.nan)
        return nan, nan.copy(), nan.copy(), nan.copy(), nan.copy()

    sigma = max(window_size / 3.0, 0.5)
    fill_val = float(np.nanmean(activation_map[mask]))
    filled = np.where(mask & np.isfinite(activation_map), activation_map, fill_val)
    smoothed = gaussian_filter(filled, sigma=sigma)
    dy_px, dx_px = np.gradient(smoothed)
    dy = np.where(mask, dy_px / pixel_size_mm, 0.0)
    dx = np.where(mask, dx_px / pixel_size_mm, 0.0)
    cv_map, angles, vx, vy = _cv_from_gradient(dy, dx, mask, cv_min, cv_max)

    dy_raw, dx_raw = _safe_gradient(activation_map, mask, pixel_size_mm)
    raw_mag = np.sqrt(dx_raw ** 2 + dy_raw ** 2)
    smooth_mag = np.sqrt(dx ** 2 + dy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        coherence = np.where(mask & (raw_mag > 1e-9), np.clip(smooth_mag / (raw_mag + 1e-9), 0, 1), np.nan)

    return cv_map, angles, coherence, vx, vy


# Backwards-compat alias
compute_hybrid_structure_tensor = compute_polynomial_bayly


# ---------------------------------------------------------------------------
# Агрегация статистики
# ---------------------------------------------------------------------------

def estimate_cv_stats(cv_map: np.ndarray, mask: np.ndarray) -> dict:
    """Сводная статистика CV по маске."""
    if not mask.any():
        return _empty_stats(mask)

    vals = cv_map[mask]
    valid = vals[np.isfinite(vals)]
    total = int(mask.sum())
    n_valid = int(len(valid))

    if n_valid == 0:
        return _empty_stats(mask)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p25 = float(np.percentile(valid, 25))
        p50 = float(np.percentile(valid, 50))
        p75 = float(np.percentile(valid, 75))

    return {
        "cv_median_m_per_s": round(p50, 4),
        "cv_mean_m_per_s": round(float(np.mean(valid)), 4),
        "cv_sd_m_per_s": round(float(np.std(valid)), 4),
        "cv_p25_m_per_s": round(p25, 4),
        "cv_p75_m_per_s": round(p75, 4),
        "valid_pixels": n_valid,
        "total_pixels": total,
        "valid_fraction": round(n_valid / total, 4),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Метод selection: выбрать лучший результат из gradient + ST
# ---------------------------------------------------------------------------

def select_best_cv_result(
    grad_res: Dict,
    st_res: Dict,
    mask: np.ndarray,
    min_valid: int = 50,
    valid_frac_margin: float = 0.08,
    n_valid_ratio: float = 1.20,
) -> Dict:
    """
    Сравнивает результаты gradient и structure tensor и возвращает лучший.

    Критерии (по приоритету):
      1. valid_fraction: st_frac > grad_frac + margin → ST
      2. n_valid: st_n > grad_n * ratio → ST
      3. default: gradient (стабильнее на резких картах)

    Если один метод имеет n_valid < min_valid → он проигрывает автоматически.

    Возвращает:
      {
        "result": best_dict,
        "method": "gradient_angular" | "structure_tensor",
        "selection_reason": str,
        "grad_n_valid": int,
        "st_n_valid": int,
        "grad_valid_fraction": float,
        "st_valid_fraction": float,
      }
    """
    total = int(mask.sum()) if mask.any() else 1

    grad_n = grad_res.get("n_valid", 0)
    st_n = st_res.get("n_valid", 0)
    grad_frac = grad_n / total
    st_frac = st_n / total

    # If one method has almost no valid pixels → other wins
    if grad_n < min_valid and st_n >= min_valid:
        return _selection(st_res, "structure_tensor", "grad_n_valid_below_threshold",
                          grad_n, st_n, grad_frac, st_frac)
    if st_n < min_valid and grad_n >= min_valid:
        return _selection(grad_res, "gradient_angular", "st_n_valid_below_threshold",
                          grad_n, st_n, grad_frac, st_frac)
    if grad_n < min_valid and st_n < min_valid:
        # Both bad — return gradient (default)
        return _selection(grad_res, "gradient_angular", "both_below_threshold",
                          grad_n, st_n, grad_frac, st_frac)

    # Criterion 1: valid_fraction
    if st_frac > grad_frac + valid_frac_margin:
        return _selection(st_res, "structure_tensor", "higher_valid_fraction",
                          grad_n, st_n, grad_frac, st_frac)

    # Criterion 2: n_valid ratio
    if st_n > grad_n * n_valid_ratio:
        return _selection(st_res, "structure_tensor", "higher_n_valid",
                          grad_n, st_n, grad_frac, st_frac)

    # Criterion 3: default preference → gradient
    return _selection(grad_res, "gradient_angular", "default_preference",
                      grad_n, st_n, grad_frac, st_frac)


def _selection(result, method, reason, grad_n, st_n, grad_frac, st_frac) -> Dict:
    return {
        "result": result,
        "method": method,
        "selection_reason": reason,
        "grad_n_valid": grad_n,
        "st_n_valid": st_n,
        "grad_valid_fraction": round(grad_frac, 4),
        "st_valid_fraction": round(st_frac, 4),
    }


def _empty_result(H: int, W: int) -> Dict:
    nan = np.full((H, W), np.nan)
    return {
        "method": "none",
        "cv_map": nan.copy(),
        "cvl_map": nan.copy(),
        "cvt_map": nan.copy(),
        "anisotropy_map": nan.copy(),
        "fiber_angle_map": nan.copy(),
        "coherence_map": nan.copy(),
        "vx": nan.copy(),
        "vy": nan.copy(),
        "cvl_m_s": np.nan,
        "cvt_m_s": np.nan,
        "anisotropy_ratio": np.nan,
        "fiber_angle_deg": np.nan,
        "fiber_coherence": np.nan,
        "n_valid": 0,
        "cv_vs_angle": np.nan,
        "bin_centers": np.nan,
        "n_sources": 0,
        "sources": np.array([]),
    }


def _empty_stats(mask: np.ndarray) -> dict:
    return {
        "cv_median_m_per_s": None,
        "cv_mean_m_per_s": None,
        "cv_sd_m_per_s": None,
        "cv_p25_m_per_s": None,
        "cv_p75_m_per_s": None,
        "valid_pixels": 0,
        "total_pixels": int(mask.sum()) if mask is not None else 0,
        "valid_fraction": 0.0,
    }