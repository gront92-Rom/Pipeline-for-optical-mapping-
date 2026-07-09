"""
peak_detection.py — beat detection for cardiac_pipeline_v3.

Ported from pipelines/peak_detector_agent.py:126-187 (v3.6 logic).

Algorithm (Roman 2026-06-14, v3.6):
  - Gaussian smooth (sigma_temporal) on mean tissue trace
  - Rolling baseline: min over 60% BCL (min 60ms)
  - 50% threshold crossings of (mt - baseline)
  - Rising edges → peaks
  - Min distance: 0.6*BCL between peaks
  - Drop first beat if >2 peaks (startup artifact)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def detect_beats(data_inv: np.ndarray,
                 mask: np.ndarray,
                 fps: float,
                 stim_hz: float = 6.0,
                 prominence_frac: float = 0.5,
                 sigma_temporal: float = 3.0) -> tuple:
    """Detect beat peak frame indices from mean tissue trace.

    Parameters
    ----------
    data_inv : np.ndarray (T, H, W)
        Preprocessed (inverted, filtered) video.
    mask : np.ndarray (H, W)
        Boolean tissue mask.
    fps : float
        Sampling rate in Hz. Required, must be > 0.
    stim_hz : float
        Stimulation frequency in Hz. Used for BCL calculation.
    prominence_frac : float
        Fraction of max amplitude for threshold crossing (default 0.5).
    sigma_temporal : float
        Gaussian smoothing sigma in frames (default 3.0).

    Returns
    -------
    peaks : np.ndarray (N,) int64
        Frame indices of detected beat peaks.
    mean_trace : np.ndarray (T,)
        Smoothed mean tissue trace (for debugging/visualization).
    """
    if fps is None or fps <= 0:
        raise ValueError(
            f"detect_beats(): fps is required and must be > 0 (got {fps})."
        )

    # Mean tissue trace
    mt_raw = data_inv[:, mask].mean(axis=1)
    if sigma_temporal > 0:
        mt = gaussian_filter1d(mt_raw, sigma=sigma_temporal)
    else:
        mt = mt_raw.copy()

    nt = len(mt)
    BCL_ms = 1000.0 / max(1e-6, stim_hz)

    # Rolling baseline: min over 60% BCL (min 60ms)
    roll_ms = max(60.0, BCL_ms * 0.6)
    roll_frames = max(20, int(roll_ms * fps / 1000.0))
    mt_baseline = np.minimum.reduceat(
        np.r_[np.full(roll_frames, mt.min()), mt],
        np.arange(0, nt, roll_frames),
    )[:nt]
    if len(mt_baseline) < nt:
        mt_baseline = np.r_[mt_baseline,
                            np.full(nt - len(mt_baseline), mt_baseline[-1])]
    mt_amp = mt - mt_baseline

    # Threshold crossings (rising edges)
    above = mt_amp > (mt_amp.max() * prominence_frac)
    rising = np.where(np.diff(above.astype(int)) == 1)[0] + 1

    # Min distance filter: 0.6 * BCL
    min_dist = max(20, int(BCL_ms * 0.6 * fps / 1000.0))
    peaks = []
    last = -min_dist
    for r in rising:
        if r - last >= min_dist:
            peaks.append(int(r))
            last = r
    peaks = np.array(peaks, dtype=np.int64)

    # Drop first beat (startup artifact) if >2 peaks
    if len(peaks) > 2:
        peaks = peaks[1:]

    return peaks, mt