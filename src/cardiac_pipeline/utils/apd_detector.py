"""
apd_detector.py — Per-pixel Action Potential Duration detection (v3.7 spec).

Architecture (2026-07-09, developer review):

  - Single-pass detection of APD30/50/80 (all levels in one call)
  - VSD inverted polarity (apex = MIN, baseline = 0, signal goes DOWN)
  - Dynamic min_amp threshold: max(100, 3 * sigma_noise)
  - Outlier rejection: APD > max(BCL * 0.9, 300ms) → NaN
  - Outlier rejection: APD < 10ms → NaN
  - Hot mask by configurable percentile (default 50 = top 50% std)
  - Active pixel loop: iterate only over hot_mask, not full H*W grid

Returns:
  - apd_4d: (n_levels, H, W, n_beats) — per-level, per-pixel, per-beat
  - apd_maps: dict {30: (H, W), 50: (H, W), 80: (H, W)} — median over beats
  - hot_mask: (H, W) bool — pixels included in analysis
  - sigma_noise: float — noise floor estimate
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional


# === Defaults ===
DEFAULT_LEVELS = [30, 50, 80]
DEFAULT_MIN_AMP_ABS = 10.0       # absolute floor (raw units, safe for CaT)
DEFAULT_MIN_AMP_NOISE_MULT = 3.0  # multiplier on noise sigma
DEFAULT_HOT_PIXEL_PERCENTILE = 50 # top 50% of masked pixels (keep majority)
DEFAULT_BASELINE_WINDOW = 10      # frames before peak for baseline
DEFAULT_APEX_SEARCH_FRAC = 0.5    # search apex in first half of segment
DEFAULT_UPSTROKE_LEVEL = 0.5      # 50% rising edge
DEFAULT_APD_MIN_MS = 10.0         # APD < 10ms is artifact
DEFAULT_APD_OUTLIER_RATIO = 0.9   # APD > BCL * 0.9 is artifact


def compute_hot_mask(
    preproc: np.ndarray,
    mask: np.ndarray,
    percentile: int = DEFAULT_HOT_PIXEL_PERCENTILE,
) -> np.ndarray:
    """
    Compute hot_mask: top-N% std pixels within tissue mask.

    Args:
        preproc: (T, H, W) video
        mask: (H, W) bool tissue mask
        percentile: keep pixels with std >= this percentile of masked pixels

    Returns:
        hot_mask: (H, W) bool
    """
    T, H, W = preproc.shape
    pixel_std = preproc.reshape(T, -1).std(axis=0).reshape(H, W)
    masked_std = pixel_std[mask]
    if len(masked_std) == 0:
        return np.zeros_like(mask)
    threshold = np.percentile(masked_std, percentile)
    hot_mask = mask & (pixel_std >= threshold)
    return hot_mask


def estimate_noise_sigma(
    preproc: np.ndarray,
    mask: np.ndarray,
    fps: float,
    n_samples: int = 50,
) -> float:
    """
    Estimate per-pixel noise sigma from rest periods between beats.

    Args:
        preproc: (T, H, W) video
        mask: (H, W) bool
        fps: sampling rate
        n_samples: number of pixels to sample

    Returns:
        sigma_noise: float, median per-pixel std of baseline
    """
    T, H, W = preproc.shape
    # Sample n random pixels within mask
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0
    n_samples = min(n_samples, len(ys))
    idx = np.random.default_rng(42).choice(len(ys), size=n_samples, replace=False)

    # For each sample pixel, compute std of first 20 frames (baseline)
    sigmas = []
    for i in idx:
        y, x = ys[i], xs[i]
        baseline_seg = preproc[:20, y, x]
        sigmas.append(baseline_seg.std())

    return float(np.median(sigmas))


def detect_all_apd_levels_pixel(
    preproc: np.ndarray,
    h: int,
    w: int,
    peak_start: int,
    peak_end: int,
    fps: float,
    levels: List[int] = DEFAULT_LEVELS,
    min_amp: float = DEFAULT_MIN_AMP_ABS,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
    apex_search_frac: float = DEFAULT_APEX_SEARCH_FRAC,
    upstroke_level: float = DEFAULT_UPSTROKE_LEVEL,
    invert: bool = True,
) -> Dict[int, float]:
    """
    Detect APD at multiple repolarization levels for one pixel in one beat.

    Polarity:
      VSD inverted (invert=True, dye A):
        - baseline = MAX (rest state, most positive)
        - apex = MIN (most negative during upstroke)
        - signal goes DOWN from baseline to apex, then UP back to baseline
      CaT non-inverted (invert=False, dye B):
        - baseline = MIN (rest state, most negative)
        - apex = MAX (most positive during upstroke)
        - signal goes UP from baseline to apex, then DOWN back to baseline

    Args:
        preproc: (T, H, W) video
        h, w: pixel coordinates
        peak_start: frame of AP upstroke (from consensus_peaks)
        peak_end: frame of NEXT AP upstroke (defines window)
        fps: sampling rate (Hz)
        levels: list of repolarization levels in percent, e.g. [30, 50, 80]
        min_amp: minimum |apex - baseline| to attempt detection
        baseline_window: frames before peak_start for baseline median
        apex_search_frac: search apex in first fraction of segment
        upstroke_level: 50% rising edge (for VSD inverted = 50% falling edge)
        invert: True for VSD (dye A, signal goes down on upstroke),
                False for CaT (dye B, signal goes up on upstroke)

    Returns:
        dict {level: apd_ms} — only levels that were successfully detected
        Missing levels are absent from dict (not NaN) — caller checks with `in`.
    """
    T = preproc.shape[0]

    # 1. Bounds check
    if peak_start < 0 or peak_end <= peak_start or peak_end > T:
        return {}

    # 1.5. Find local AP features per-pixel (not relying on global peak_start).
    #      peak_start from PeakDet is the GLOBAL 50% crossing for mean_tissue.
    #      Per-pixel conduction delay means local AP onset can be ±10 frames off.
    #      We search the full window [peak_start - 30, peak_end - 10] for the
    #      local AP: baseline, apex, upstroke.

    window_start = max(0, peak_start - 30)
    window_end = max(window_start + 10, peak_end - 10)
    full_seg = preproc[window_start:window_end, h, w]

    if len(full_seg) < 20:
        return {}

    if invert:
        # VSD: baseline = max (rest), apex = min (depolarization dip)
        local_baseline = float(np.max(full_seg))
        local_apex_idx = window_start + int(np.argmin(full_seg))
    else:
        # CaT: baseline = min (rest), apex = max (calcium transient peak)
        local_baseline = float(np.min(full_seg))
        local_apex_idx = window_start + int(np.argmax(full_seg))

    local_apex_val = float(preproc[local_apex_idx, h, w])

    # Amplitude
    local_amp = abs(local_baseline - local_apex_val)

    if local_amp < min_amp:
        return {}

    # Use these as our working values
    baseline = local_baseline
    apex_idx = local_apex_idx
    apex_val = local_apex_val
    amp = local_amp

    # Find onset: first frame where signal crosses 20% toward apex
    # For VSD inverted: signal drops below baseline - 0.2*amp
    # For CaT: signal rises above baseline + 0.2*amp
    if invert:
        target_20 = baseline - 0.2 * amp
    else:
        target_20 = baseline + 0.2 * amp

    # Search BEFORE local_apex for the onset of THIS AP
    # Look from window_start to local_apex
    pre_apex_seg = preproc[window_start:local_apex_idx + 1, h, w]
    if invert:
        below = pre_apex_seg < target_20
    else:
        below = pre_apex_seg > target_20
    if below.any():
        up_idx = window_start + int(np.where(below)[0][0])
    else:
        # Fallback: use peak_start (global 50% crossing)
        up_idx = peak_start

    # Override variables for repolarization search
    peak_start = up_idx

    # 7. Compute APD for each level (single pass)
    apd_results = {}
    after_apex = preproc[apex_idx:peak_end, h, w]

    for level in levels:
        level_frac = level / 100.0

        if invert:
            # VSD: repolarization = signal rises from apex back toward baseline
            repol_target = apex_val + level_frac * (baseline - apex_val)
            above = after_apex > repol_target
        else:
            # CaT: repolarization = signal falls from apex back toward baseline
            repol_target = apex_val - level_frac * (apex_val - baseline)
            above = after_apex < repol_target

        if not above.any():
            continue

        repol_idx_local = int(np.where(above)[0][0])
        repol_idx = apex_idx + repol_idx_local

        # APD = repol - upstroke
        apd_frames = repol_idx - up_idx
        apd_ms = apd_frames / fps * 1000.0

        apd_results[level] = apd_ms

    return apd_results

def reject_apd_outliers(
    apd_values: Dict[int, float],
    bcl_ms: float,
    min_apd_ms: float = DEFAULT_APD_MIN_MS,
    outlier_ratio: float = DEFAULT_APD_OUTLIER_RATIO,
) -> Dict[int, float]:
    """
    Filter APD values that are physiologically impossible.

    Rule:
      APD > max(BCL * outlier_ratio, 300ms) → NaN (too long for cycle)
      APD < min_apd_ms → NaN (too short, artifact)

    Args:
        apd_values: {level: apd_ms}
        bcl_ms: basic cycle length (1/freq)
        min_apd_ms: APD below this is artifact
        outlier_ratio: APD above BCL*ratio is artifact

    Returns:
        filtered dict (outliers removed)
    """
    max_apd = max(bcl_ms * outlier_ratio, 300.0)

    filtered = {}
    for level, apd_ms in apd_values.items():
        if apd_ms < min_apd_ms:
            continue
        if apd_ms > max_apd:
            continue
        filtered[level] = apd_ms
    return filtered


def compute_per_pixel_min_amp(
    preproc: np.ndarray,
    mask: np.ndarray,
    sigma_noise: float,
    abs_floor: float = DEFAULT_MIN_AMP_ABS,
    noise_multiplier: float = DEFAULT_MIN_AMP_NOISE_MULT,
) -> float:
    """
    Compute dynamic min_amp threshold: max(abs_floor, noise_multiplier * sigma_noise).

    This protects against weak signals that are within noise floor.
    """
    return max(abs_floor, noise_multiplier * sigma_noise)