"""
activation_detector.py — Per-pixel activation time (TAT) detection (v3.9).

Final logic (Roman 2026-07-10):
  - Global onsets from mean trace: rolling baseline + 50% threshold crossing
    (already done by peak_detection.py → selected_peaks)
  - Per-pixel activation: same 50% threshold crossing, applied to each pixel
    within [onset, next_onset] window.
  - Light spatial Gaussian (σ=1px) on each frame before processing.
  - Optional: parabolic interpolation for subframe precision.
  - No SavGol derivative per pixel. No rolling baseline per pixel.
  - Supports both rising (AP up) and falling (VSD raw) polarity via falling_edge flag.

Vectorized implementation: no pixel-by-pixel Python loop.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def _apply_spatial_gaussian(
    preproc: np.ndarray,
    sigma: float,
    mask: np.ndarray,
) -> np.ndarray:
    """Apply per-frame 2D Gaussian filter, restricted to mask region.

    Args:
        preproc: (T, H, W) video segment
        sigma: Gaussian sigma in pixels (0 = skip)
        mask: (H, W) bool — only filter inside mask

    Returns:
        Filtered (T, H, W) array (same shape, mask respected).
    """
    if sigma <= 0:
        return preproc
    filtered = np.empty_like(preproc, dtype=np.float32)
    for t in range(preproc.shape[0]):
        filtered[t] = gaussian_filter(preproc[t].astype(np.float32), sigma=sigma)
    # Restore non-mask pixels to original (avoid edge bleed)
    for t in range(preproc.shape[0]):
        filtered[t][~mask] = preproc[t][~mask]
    return filtered


def detect_tat_50pct_vectorized(
    preproc: np.ndarray,
    mask: np.ndarray,
    onset: int,
    next_onset: int,
    fps: float,
    min_amp: float = 20.0,
    sigma_spatial: float = 1.0,
    falling_edge: bool = False,
    parabolic_interp: bool = False,
) -> np.ndarray:
    """
    Vectorized per-pixel 50% threshold crossing TAT map.

    For each mask pixel within [onset, next_onset] window:
      - baseline = min(trace) [rising] or max(trace) [falling]
      - peak = max(trace) [rising] or min(trace) [falling]
      - amp = |peak - baseline|
      - if amp < min_amp → NaN
      - threshold = baseline + 0.5*amp [rising] or baseline - 0.5*amp [falling]
      - activation = first frame where signal crosses threshold
      - Optional parabolic interpolation for subframe precision

    Args:
        preproc: (T, H, W) full preprocessed video
        mask: (H, W) bool
        onset: global onset frame (from peak_detection)
        next_onset: next beat onset frame (defines window end)
        fps: sampling rate (Hz)
        min_amp: minimum AP amplitude (a.u.) to accept detection
        sigma_spatial: spatial Gaussian sigma (0 = skip)
        falling_edge: if True, signal goes DOWN on activation (VSD raw polarity)
        parabolic_interp: if True, apply 3-point parabolic interpolation for subframe precision

    Returns:
        tat_map: (H, W) float32 — TAT in ms relative to global onset.
                 NaN where detection failed.
    """
    T, H, W = preproc.shape
    tat_map = np.full((H, W), np.nan, dtype=np.float32)

    # Window bounds (with small margin)
    win_start = max(0, onset - 5)
    win_end = min(T, next_onset + 5)
    if win_end - win_start < 10:
        return tat_map

    # Extract window and apply spatial Gaussian
    window = preproc[win_start:win_end].astype(np.float32)  # (W, H, Wd)
    if sigma_spatial > 0:
        window = _apply_spatial_gaussian(window, sigma_spatial, mask)

    n_win = window.shape[0]

    if falling_edge:
        # Signal goes DOWN on activation: baseline=max, peak=min
        baseline = window.max(axis=0)  # (H, Wd)
        peak = window.min(axis=0)
        amp = baseline - peak  # ≥ 0
        threshold = baseline - 0.5 * amp
        # First frame where signal < threshold (falling crossing)
        below = window < threshold  # (W, H, Wd)
        # argmax on bool gives first True index
        any_below = below.any(axis=0)
        first_cross = np.argmax(below, axis=0).astype(np.float64)  # (H, Wd)
    else:
        # Signal goes UP on activation: baseline=min, peak=max
        baseline = window.min(axis=0)
        peak = window.max(axis=0)
        amp = peak - baseline
        threshold = baseline + 0.5 * amp
        # First frame where signal > threshold (rising crossing)
        above = window > threshold
        any_above = above.any(axis=0)
        first_cross = np.argmax(above, axis=0).astype(np.float64)

    # Amplitude gate
    valid_amp = amp >= min_amp
    valid = valid_amp & mask & (below.any(axis=0) if falling_edge else above.any(axis=0))

    # Parabolic interpolation for subframe precision
    if parabolic_interp and n_win >= 3:
        fc_int = first_cross.astype(int)
        # For each pixel, interpolate around the crossing point
        # parabola: y(t) = a*t^2 + b*t + c, vertex at t = -b/(2a)
        # Using 3 points: [fc-1, fc, fc+1]
        safe_idx = np.clip(fc_int, 1, n_win - 2)
        for h in range(H):
            for w in range(W):
                if not valid[h, w]:
                    continue
                fc = fc_int[h, w]
                if fc < 1 or fc >= n_win - 1:
                    continue
                if falling_edge:
                    # Crossing: signal going down through threshold
                    y_m = float(window[fc - 1, h, w])
                    y_0 = float(window[fc, h, w])
                    y_p = float(window[fc + 1, h, w])
                else:
                    y_m = float(window[fc - 1, h, w])
                    y_0 = float(window[fc, h, w])
                    y_p = float(window[fc + 1, h, w])
                # Parabolic interpolation of the derivative peak
                # Vertex of parabola through (fc-1, y_m), (fc, y_0), (fc+1, y_p)
                denom = (y_m - 2 * y_0 + y_p)
                if abs(denom) > 1e-10:
                    offset = 0.5 * (y_m - y_p) / denom
                    first_cross[h, w] = fc + offset

    # Convert to TAT in ms relative to global onset
    tat_map[valid] = (first_cross[valid] + win_start - onset) / fps * 1000.0

    return tat_map


def detect_tat_map_local(
    preproc: np.ndarray,
    mask: np.ndarray,
    peak_start: int,
    peak_end: int,
    fps: float,
    method: str = "threshold_50pct",
    min_amp: float = 20.0,
    sigma_spatial: float = 1.0,
    falling_edge: bool = False,
    parabolic_interp: bool = False,
) -> np.ndarray:
    """
    Per-pixel TAT map — 50% threshold crossing (v3.9).

    Wrapper around detect_tat_50pct_vectorized for API compatibility.
    The `method` parameter is kept for compatibility but only 'threshold_50pct'
    is implemented (Roman's final decision: no derivative per pixel).

    Args:
        preproc: (T, H, W)
        mask: (H, W) bool
        peak_start: global onset frame (anchor)
        peak_end: next onset frame (window end)
        fps: sampling rate
        method: ignored (always 50% crossing)
        min_amp: minimum AP amplitude
        sigma_spatial: spatial Gaussian sigma (0 = skip)
        falling_edge: True if signal goes DOWN on activation
        parabolic_interp: subframe precision via parabolic interpolation

    Returns:
        tat_map: (H, W) — TAT in ms (relative to peak_start)
    """
    return detect_tat_50pct_vectorized(
        preproc, mask, peak_start, peak_end, fps,
        min_amp=min_amp,
        sigma_spatial=sigma_spatial,
        falling_edge=falling_edge,
        parabolic_interp=parabolic_interp,
    )


def combine_regions_soft(
    tat_per_region: list,
    weights: np.ndarray,
    active_pixels: np.ndarray,
) -> np.ndarray:
    """
    Combine per-region TAT maps using soft weights.

    For each active pixel (h, w):
        tat[h, w] = weighted average over valid regions

    Args:
        tat_per_region: list of n_regions TAT maps (H, W)
        weights: (H, W, n_regions) soft weights (sum=1 per pixel)
        active_pixels: (N, 2) coordinates from np.argwhere(hot_mask)

    Returns:
        tat_combined: (H, W) — soft-weighted TAT map
    """
    H, W = weights.shape[:2]
    n_regions = len(tat_per_region)
    tat_combined = np.full((H, W), np.nan, dtype=np.float32)

    for h, w in active_pixels:
        valid_weights = []
        valid_values = []
        for r in range(n_regions):
            val = tat_per_region[r][h, w]
            wgt = weights[h, w, r]
            if np.isfinite(val) and wgt > 0:
                valid_weights.append(wgt)
                valid_values.append(val)
        if valid_values:
            ws = np.array(valid_weights, dtype=np.float64)
            ws = ws / ws.sum()
            tat_combined[h, w] = float(np.average(valid_values, weights=ws))

    return tat_combined


def consensus_tat_methods(
    preproc: np.ndarray,
    mask: np.ndarray,
    peak_start: int,
    peak_end: int,
    fps: float,
    methods: list = ("threshold_50pct",),
    min_amp: float = 20.0,
    agreement_threshold_ms: float = 5.0,
    sigma_spatial: float = 1.0,
    falling_edge: bool = False,
    parabolic_interp: bool = False,
) -> np.ndarray:
    """
    Run TAT detection. In v3.9 only one method (50% crossing) is used,
    so this is effectively a passthrough to detect_tat_map_local.

    Kept for API compatibility with ActivationAgent.

    Args:
        preproc: (T, H, W)
        mask: (H, W) bool
        peak_start, peak_end: peak window
        fps: sampling rate
        methods: list of method names (only 'threshold_50pct' supported)
        min_amp: minimum AP amplitude
        agreement_threshold_ms: unused (single method)
        sigma_spatial: spatial Gaussian sigma
        falling_edge: True if signal goes DOWN on activation
        parabolic_interp: subframe precision

    Returns:
        tat_map: (H, W) — TAT map
    """
    return detect_tat_map_local(
        preproc, mask, peak_start, peak_end, fps,
        method='threshold_50pct',
        min_amp=min_amp,
        sigma_spatial=sigma_spatial,
        falling_edge=falling_edge,
        parabolic_interp=parabolic_interp,
    )