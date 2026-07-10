"""
peak_detection.py — beat detection for cardiac_pipeline_v3 (v3.6 spec, 2026-07-09).

Algorithm (Roman 2026-07-09, v3.6):
  - Gaussian lowpass (sigma_temporal) on mean tissue trace
  - Rolling baseline: scipy.ndimage.minimum_filter1d with size = 1.0 × BCL
    (mathematically guarantees at least one true diastolic point inside window
     even under linear photobleaching drift)
  - Amplitude = signal − baseline
  - Rising edges at threshold_frac × max_amplitude
  - Min distance filter: 0.6 × BCL between peaks
  - drop_first optional (default False, since startup artifacts should be caught
    by Stage 3 QC, not hardcoded drop)

Design notes:
  - Caller (PeakDetectorAgent) aggregates preproc_video + mask → 1D mean_tissue
    using `preproc[:, mask].mean(axis=(1, 2))` (correct broadcasting).
    This function does NOT touch the 3D video or 2D mask.
  - Polarity: function ASSUMES "AP up" (LoaderAgent inverts VSD via
    should_invert). If AP is inverted (down), caller must flip the sign
    before calling. Function does NOT auto-detect polarity.
  - Upstroke 50%-crossing (rising edge) is the returned peak index.
    APDAgent on Stage 4 will look for argmax locally around pk for apex/amp.

Edge cases (raise ValueError):
  - fps <= 0
  - mean_tissue.ndim != 1
  - T < BCL_frames (recording too short)
  - max_amp < 0.1 * signal.std() (no AP detected — dead tissue / no signal)
  - len(peaks) == 0 (no peaks found at threshold)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d


def detect_beats(mean_tissue: np.ndarray,
                 fps: float,
                 stim_hz: float,
                 sigma_temporal: float = 3.0,
                 threshold_frac: float = 0.5,
                 min_distance_factor: float = 0.6,
                 drop_first: bool = False) -> tuple:
    """Detect beat upstroke 50%-crossing from 1D mean tissue trace.

    Parameters
    ----------
    mean_tissue : np.ndarray (T,)
        1D mean tissue trace (caller aggregates preproc[:, mask].mean(axis=(1,2))).
        MUST be "AP up" (Loader inverts VSD).
    fps : float
        Sampling rate in Hz. Required, must be > 0.
    stim_hz : float
        Stimulation frequency in Hz. Used for BCL calculation.
    sigma_temporal : float, default 3.0
        Gaussian smoothing sigma in frames. Static (NOT dynamic from stim_hz).
    threshold_frac : float, default 0.5
        Fraction of max amplitude for rising-edge threshold (0.5 = upstroke 50%).
    min_distance_factor : float, default 0.6
        Min distance between peaks = BCL_frames × factor. Factor 0.6 gives
        40% safety margin against alternans / early ectopic beats.
    drop_first : bool, default False
        If True and len(peaks) > 2, drop the first peak (startup artifact).
        Default False — startup artifacts are QC's responsibility.

    Returns
    -------
    peaks : np.ndarray (N,) int64
        Frame indices of detected beat upstroke 50%-crossings.
    smoothed : np.ndarray (T,)
        Gaussian-lowpassed mean tissue trace (for debug/visualization).

    Raises
    ------
    ValueError
        - fps <= 0
        - mean_tissue.ndim != 1
        - T < BCL_frames
        - max_amp < 0.1 * signal.std() (no AP detected)
        - len(peaks) == 0 (no peaks at threshold)
    """
    # === Step 1: validate inputs ===
    if fps is None or fps <= 0:
        raise ValueError(
            f"detect_beats(): fps is required and must be > 0 (got {fps})."
        )
    if mean_tissue.ndim != 1:
        raise ValueError(
            f"detect_beats(): mean_tissue must be 1D (got shape {mean_tissue.shape})."
        )

    mt_raw = np.asarray(mean_tissue, dtype=np.float64)
    nt = len(mt_raw)
    if nt == 0:
        raise ValueError("detect_beats(): mean_tissue is empty.")

    BCL_ms = 1000.0 / max(1e-6, stim_hz)
    BCL_frames = max(1, int(BCL_ms * fps / 1000.0))

    if nt < BCL_frames:
        raise ValueError(
            f"Recording too short: T={nt} frames < BCL={BCL_frames} frames "
            f"(stim_hz={stim_hz}, fps={fps})."
        )

    # === Step 2: Gaussian lowpass ===
    if sigma_temporal > 0:
        smoothed = gaussian_filter1d(mt_raw, sigma=sigma_temporal)
    else:
        smoothed = mt_raw.copy()

    # === Step 3: rolling minimum baseline (size = 1.0 × BCL) ===
    # mathematically guaranteed to contain ≥ 1 true diastolic point
    # even under linear photobleaching drift on the BCL timescale.
    baseline = minimum_filter1d(smoothed, size=BCL_frames)

    # === Step 4: amplitude ===
    amp = smoothed - baseline  # ≥ 0 by construction
    max_amp = float(amp.max())

    # === Step 5: amplitude gate (no AP detected) ===
    sig_std = float(mt_raw.std())
    if max_amp < 0.1 * sig_std:
        raise ValueError(
            f"No AP detected: max_amp={max_amp:.2f} < 10% of signal std "
            f"({0.1 * sig_std:.2f}). Tissue may be dead or recording too noisy."
        )

    # === Step 6: threshold flag + rising edges ===
    threshold = max_amp * threshold_frac
    above = amp > threshold
    diff_above = np.diff(above.astype(np.int8))
    rising = np.where(diff_above == 1)[0] + 1  # +1 = index where above becomes True

    # === Step 7: min distance filter (0.6 × BCL safety margin) ===
    min_dist = max(20, int(BCL_frames * min_distance_factor))
    peaks_list: list[int] = []
    last = -min_dist
    for r in rising:
        if r - last >= min_dist:
            peaks_list.append(int(r))
            last = r
    peaks = np.array(peaks_list, dtype=np.int64)

    # === Step 8: drop_first (optional) ===
    if drop_first and len(peaks) > 2:
        peaks = peaks[1:]

    # === Step 9: final gate ===
    if len(peaks) == 0:
        raise ValueError(
            f"No peaks found at threshold_frac={threshold_frac} "
            f"(max_amp={max_amp:.2f}, threshold={threshold:.2f})."
        )

    return peaks, smoothed
