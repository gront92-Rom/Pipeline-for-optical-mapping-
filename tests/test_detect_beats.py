"""
tests/test_detect_beats.py — unit tests for detect_beats() (cardiac_pipeline_v3 v3.6 spec).

7 test cases:
  1. test_clean_signal_detects_all_beats       — perfect AP train, no noise/drift
  2. test_photobleaching_does_not_affect       — linear baseline decay (drift)
  3. test_robust_to_white_noise                — SNR=10 noise
  4. test_recording_too_short_raises           — T < BCL
  5. test_no_ap_raises                         — pure noise (max_amp < 10% std)
  6. test_drop_first_drops_one_peak            — drop_first=True vs False
  7. test_threshold_frac_changes_peak_position — threshold shifts upstroke

Run:
  cd cardiac_pipeline_v3
  pytest tests/test_detect_beats.py -v
"""

import numpy as np
import pytest

from cardiac_pipeline.utils.peak_detection import detect_beats


# === Synthetic AP template ===

def _make_ap_train(T: int, fps: float, stim_hz: float,
                   ap_amp: float = 10.0,
                   ap_width_s: float = 0.05,
                   noise_amp: float = 0.0,
                   baseline_drift: float = 0.0) -> np.ndarray:
    """Generate 1D synthetic AP train (AP up).

    Parameters
    ----------
    T : int
        Number of frames.
    fps : float
        Sampling rate (Hz).
    stim_hz : float
        Stimulation frequency (Hz). One AP every 1/stim_hz seconds.
    ap_amp : float
        Peak amplitude.
    ap_width_s : float
        AP width (Gaussian sigma, seconds).
    noise_amp : float
        White noise amplitude.
    baseline_drift : float
        Linear baseline drift over the whole recording (per second).
    """
    t = np.arange(T) / fps
    ap_period_s = 1.0 / stim_hz
    # Each AP is a Gaussian peak at multiples of ap_period_s + small phase offset
    phase = t - 0.05  # AP starts at t=0.05s (upstroke phase)
    # distance from nearest AP start
    ap_idx = np.floor(phase / ap_period_s).astype(int)
    ap_start = ap_idx * ap_period_s
    dt_to_ap = phase - ap_start
    # Gaussian AP shape: peak at dt = ap_width_s
    ap = ap_amp * np.exp(-((dt_to_ap - ap_width_s) ** 2) / (ap_width_s ** 2))
    ap[ap_idx < 0] = 0.0  # before first AP

    signal = ap
    if noise_amp > 0:
        signal = signal + noise_amp * np.random.randn(T)
    if baseline_drift != 0.0:
        signal = signal + baseline_drift * t
    return signal


# === Test 1: clean signal ===

def test_clean_signal_detects_all_beats():
    """Perfect AP train: 12 beats at 6Hz over 2 sec, no noise, no drift."""
    T, fps, stim_hz = 1000, 500, 6.0
    signal = _make_ap_train(T, fps, stim_hz, ap_amp=10.0, noise_amp=0.0)
    peaks, smoothed = detect_beats(signal, fps, stim_hz)

    # Expected: ~12 beats (T/fps * stim_hz = 2 * 6 = 12)
    assert len(peaks) == 12, f"Expected 12 peaks, got {len(peaks)}: {peaks}"
    # All peaks positive
    assert np.all(peaks > 0)
    # BCL ~ 83 frames; all gaps >= 60 frames (with min_dist factor 0.6)
    diffs = np.diff(peaks)
    assert np.all(diffs >= 60), f"Gaps too small: {diffs}"
    # Smoothed has same length as input
    assert smoothed.shape == (T,)


# === Test 2: photobleaching drift ===

def test_photobleaching_does_not_affect_peak_detection():
    """Linear baseline drift (0 → 1.0) over 2 sec must not affect peak detection.

    minimum_filter1d(size=BCL) tracks the drift, so amplitude (signal - baseline)
    stays constant and peaks are still detected at 50% threshold.
    """
    T, fps, stim_hz = 1000, 500, 6.0
    signal = _make_ap_train(T, fps, stim_hz, ap_amp=10.0,
                            baseline_drift=0.5)  # baseline drops by 1.0 over 2s
    peaks, _ = detect_beats(signal, fps, stim_hz)

    assert len(peaks) == 12, f"Drift broke detection: got {len(peaks)} peaks"
    diffs = np.diff(peaks)
    assert np.all(diffs >= 60)


# === Test 3: white noise ===

def test_robust_to_white_noise():
    """SNR ~10 (signal amp 10, noise std 1.0) — should still find ~12 peaks."""
    T, fps, stim_hz = 1000, 500, 6.0
    rng = np.random.default_rng(42)
    signal = _make_ap_train(T, fps, stim_hz, ap_amp=10.0, noise_amp=1.0)
    peaks, _ = detect_beats(signal, fps, stim_hz)

    # Allow ±2 peaks drift (robust to noise)
    assert 10 <= len(peaks) <= 14, f"Noise broke detection: got {len(peaks)} peaks"


# === Test 4: T < BCL ===

def test_recording_too_short_raises():
    """T=50 frames < BCL=83 frames for 6Hz @ 500fps must raise ValueError."""
    T, fps, stim_hz = 50, 500, 6.0
    signal = np.zeros(T)
    with pytest.raises(ValueError, match="Recording too short"):
        detect_beats(signal, fps, stim_hz)


# === Test 5: no AP ===

def test_no_ap_raises():
    """No AP detected: signal below 10% of std OR no peaks found → raise.

    Two failure modes:
      (a) max_amp < 0.1 * std → "No AP detected"
      (b) threshold crossings yield 0 peaks → "No peaks found"

    For (b) we use DC=0 (no AP, no noise): amp = 0, threshold = 0, no
    rising edges, gate fires. For (a) we'd need a signal where rolling-min
    leaves amp < 10% of overall std, but min_filter1d on any noise creates
    real fluctuations that exceed 10% std. So we test (b) here.
    """
    T, fps, stim_hz = 1000, 500, 6.0
    signal = np.zeros(T)  # DC = 0, no AP, no noise
    with pytest.raises(ValueError, match="No peaks found"):
        detect_beats(signal, fps, stim_hz)


# === Test 6: drop_first ===

def test_drop_first_drops_one_peak():
    """drop_first=True with 12 peaks → 11 peaks (first one removed)."""
    T, fps, stim_hz = 1000, 500, 6.0
    signal = _make_ap_train(T, fps, stim_hz, ap_amp=10.0, noise_amp=0.0)

    peaks_default, _ = detect_beats(signal, fps, stim_hz, drop_first=False)
    peaks_dropped, _ = detect_beats(signal, fps, stim_hz, drop_first=True)

    assert len(peaks_default) == 12
    assert len(peaks_dropped) == 11
    # The first dropped peak should equal peaks_default[0]
    assert np.array_equal(peaks_dropped, peaks_default[1:])


# === Test 7: threshold_frac shifts peak position ===

def test_threshold_frac_changes_peak_position():
    """Lower threshold_frac → earlier rising-edge crossing (upstroke is rising)."""
    T, fps, stim_hz = 1000, 500, 6.0
    signal = _make_ap_train(T, fps, stim_hz, ap_amp=10.0, noise_amp=0.0)

    peaks_low, _ = detect_beats(signal, fps, stim_hz, threshold_frac=0.3)
    peaks_high, _ = detect_beats(signal, fps, stim_hz, threshold_frac=0.7)

    assert len(peaks_low) == 12
    assert len(peaks_high) == 12
    # Lower threshold → earlier crossing (upstroke rises from baseline to apex)
    assert np.mean(peaks_low) < np.mean(peaks_high), \
        f"threshold_frac=0.3 should give earlier peaks than 0.7: {peaks_low} vs {peaks_high}"
    # Shift per peak is bounded by AP width (~25 frames for sigma=0.05s @ 500fps)
    shift = np.mean(peaks_high - peaks_low)
    assert 0 < shift < 20, f"Shift {shift} out of expected range"


# === Bonus: ensure fps <= 0 raises ===

def test_invalid_fps_raises():
    T, fps, stim_hz = 1000, 500, 6.0
    signal = _make_ap_train(T, fps, stim_hz)
    with pytest.raises(ValueError, match="fps"):
        detect_beats(signal, fps=0, stim_hz=stim_hz)
    with pytest.raises(ValueError, match="fps"):
        detect_beats(signal, fps=-1, stim_hz=stim_hz)


# === Bonus: ensure 2D input raises ===

def test_2d_input_raises():
    bad = np.zeros((100, 100))
    with pytest.raises(ValueError, match="1D"):
        detect_beats(bad, fps=500, stim_hz=6.0)
