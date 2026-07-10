"""
tests/test_activation.py — Unit tests for ActivationAgent v3.7.

Covers:
  - detect_tat_pixel_local (single pixel, 2 methods, edge cases)
  - detect_tat_map_local (full map)
  - combine_regions_soft (weighted combine)
  - consensus_tat_methods (multi-method agreement)
"""

import numpy as np
import pytest

from cardiac_pipeline.utils.activation_detector import (
    detect_tat_pixel_local,
    detect_tat_map_local,
    combine_regions_soft,
    consensus_tat_methods,
)


# ============== Helpers ==============

def make_synthetic_video(
    n_pixels: int = 3,
    n_frames: int = 200,
    fps: float = 500.0,
    onset_frame_global: int = 50,
    conduction_delay_frames: tuple = (0, 5, 10),
    noise_std: float = 1.0,
    baseline_val: float = 0.0,
    apex_val: float = -100.0,
) -> np.ndarray:
    """
    Synthetic video with linear conduction wave.

    Pixel i activates at frame = onset_frame_global + conduction_delay_frames[i]
    Then full AP downstroke to apex_val, returns to baseline.

    Returns:
        preproc: (n_frames, n_pixels, 1) — VSD inverted
    """
    rng = np.random.default_rng(42)
    T = n_frames
    # Pad conduction_delay_frames to n_pixels
    if len(conduction_delay_frames) < n_pixels:
        conduction_delay_frames = tuple(list(conduction_delay_frames) + [0] * (n_pixels - len(conduction_delay_frames)))
    preproc = np.full((T, n_pixels, 1), baseline_val, dtype=np.float32)
    for i in range(n_pixels):
        onset_i = onset_frame_global + conduction_delay_frames[i]
        onset_i = max(0, min(onset_i, T - 1))
        for t in range(onset_i, T):
            # Linear upstroke over 10 frames to apex, then return over 80 frames
            if t < onset_i + 10:
                # Upstroke: linear from baseline to apex
                f = (t - onset_i) / 10.0
                val = baseline_val + f * (apex_val - baseline_val)
            elif t < onset_i + 90:
                # Repolarization: return to baseline
                f = (t - onset_i - 10) / 80.0
                val = apex_val + f * (baseline_val - apex_val)
            else:
                val = baseline_val
            preproc[t, i, 0] = val + rng.normal(0, noise_std)
    return preproc


# ============== detect_tat_pixel_local ==============

class TestDetectTatPixelLocal:
    def test_50pct_pixel_with_no_delay(self):
        """Pixel 0 activates at the same time as global reference.
        50% crossing happens 5 frames after onset (upstroke is 10 frames).
        """
        preproc = make_synthetic_video(
            n_pixels=3, conduction_delay_frames=(0, 5, 10)
        )
        T = preproc.shape[0]
        peak_start = 50  # global onset
        peak_end = 150
        fps = 500.0
        tat = detect_tat_pixel_local(
            preproc, 0, 0, peak_start, peak_end, fps, method="threshold_50pct"
        )
        # 50% crossing = onset + 5 frames = 10ms (at 500fps)
        # Allow ±4ms tolerance
        assert 6 < tat < 14, f"Expected TAT ≈ 10ms, got {tat}ms"

    def test_50pct_pixel_with_5_frame_delay(self):
        """Pixel 1 activates 5 frames after global → 50% crossing at onset+5+5 = 10 frames."""
        preproc = make_synthetic_video(
            n_pixels=3, conduction_delay_frames=(0, 5, 10)
        )
        tat = detect_tat_pixel_local(
            preproc, 1, 0, 50, 150, 500.0, method="threshold_50pct"
        )
        # onset=55, 50% crossing=60 → 20ms
        assert 15 < tat < 25, f"Expected TAT ≈ 20ms, got {tat}ms"

    def test_50pct_pixel_with_10_frame_delay(self):
        """Pixel 2 activates 10 frames after global → 50% crossing at onset+10+5 = 15 frames."""
        preproc = make_synthetic_video(
            n_pixels=3, conduction_delay_frames=(0, 5, 10)
        )
        tat = detect_tat_pixel_local(
            preproc, 2, 0, 50, 150, 500.0, method="threshold_50pct"
        )
        # onset=60, 50% crossing=65 → 30ms
        assert 25 < tat < 35, f"Expected TAT ≈ 30ms, got {tat}ms"

    def test_derivative_max_method(self):
        """derivative_max should also detect the onset within tolerance."""
        preproc = make_synthetic_video(
            n_pixels=3, conduction_delay_frames=(0, 5, 10)
        )
        tat_50 = detect_tat_pixel_local(
            preproc, 1, 0, 50, 150, 500.0, method="threshold_50pct"
        )
        tat_deriv = detect_tat_pixel_local(
            preproc, 1, 0, 50, 150, 500.0, method="derivative_max"
        )
        # Both methods should be close (within a few ms)
        assert abs(tat_50 - tat_deriv) < 10, (
            f"Methods disagree: 50pct={tat_50}, deriv={tat_deriv}"
        )

    def test_no_ap_returns_nan(self):
        """Pixel with no AP (flat baseline) should return NaN."""
        preproc = np.zeros((100, 3, 1), dtype=np.float32)  # all zero
        tat = detect_tat_pixel_local(
            preproc, 0, 0, 50, 100, 500.0, method="threshold_50pct", min_amp=50.0
        )
        assert np.isnan(tat), f"Expected NaN for flat signal, got {tat}"

    def test_weak_ap_returns_nan(self):
        """Pixel with amplitude < min_amp should return NaN."""
        preproc = np.zeros((100, 3, 1), dtype=np.float32)
        # Add tiny AP (amp = 5, min_amp=50)
        for t in range(50, 60):
            preproc[t, 0, 0] = -5.0 * (t - 50) / 10
        for t in range(60, 100):
            preproc[t, 0, 0] = -5.0 + 5.0 * (t - 60) / 40
        tat = detect_tat_pixel_local(
            preproc, 0, 0, 50, 100, 500.0, method="threshold_50pct", min_amp=50.0
        )
        assert np.isnan(tat), f"Expected NaN for weak AP, got {tat}"

    def test_invalid_peak_window(self):
        """peak_start >= peak_end or out of range → NaN."""
        preproc = make_synthetic_video(n_pixels=3)
        # peak_start = peak_end (invalid)
        tat = detect_tat_pixel_local(
            preproc, 0, 0, 50, 50, 500.0, method="threshold_50pct"
        )
        assert np.isnan(tat)
        # peak_end out of range
        tat = detect_tat_pixel_local(
            preproc, 0, 0, 50, 10000, 500.0, method="threshold_50pct"
        )
        assert np.isnan(tat)


# ============== detect_tat_map_local ==============

class TestDetectTatMapLocal:
    def test_full_map_shape(self):
        """Output shape should match input mask shape."""
        preproc = make_synthetic_video(n_pixels=10, n_frames=200)
        mask = np.zeros((10, 1), dtype=bool)
        mask[[0, 1, 2, 5, 8], 0] = True
        tat_map = detect_tat_map_local(preproc, mask, 50, 150, 500.0)
        assert tat_map.shape == (10, 1)
        # Only masked pixels should have non-NaN values (or NaN if AP detection failed)
        assert np.isnan(tat_map[~mask]).all()

    def test_only_iterates_masked_pixels(self):
        """Map should be filled for masked pixels (or NaN if AP weak)."""
        preproc = make_synthetic_video(n_pixels=5)
        mask = np.ones((5, 1), dtype=bool)
        tat_map = detect_tat_map_local(preproc, mask, 50, 150, 500.0)
        assert tat_map.shape == (5, 1)
        # All 5 pixels have AP
        assert not np.isnan(tat_map[mask]).any(), (
            f"All masked pixels should have valid TAT, got {tat_map[mask]}"
        )


# ============== combine_regions_soft ==============

class TestCombineRegionsSoft:
    def test_uniform_weights_returns_mean(self):
        """Equal weights → simple average."""
        tat_r0 = np.array([[10.0, 20.0], [30.0, 40.0]])
        tat_r1 = np.array([[50.0, 60.0], [70.0, 80.0]])
        weights = np.zeros((2, 2, 2))
        weights[..., 0] = 0.5
        weights[..., 1] = 0.5
        active = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        out = combine_regions_soft([tat_r0, tat_r1], weights, active)
        expected = (tat_r0 + tat_r1) / 2
        np.testing.assert_array_almost_equal(out, expected)

    def test_full_weight_to_one_region(self):
        """Weight 1.0 on r0, 0.0 on r1 → output = tat_r0."""
        tat_r0 = np.array([[10.0, 20.0]])
        tat_r1 = np.array([[99.0, 99.0]])
        weights = np.zeros((1, 2, 2))
        weights[0, :, 0] = 1.0
        weights[0, :, 1] = 0.0
        active = np.array([[0, 0], [0, 1]])
        out = combine_regions_soft([tat_r0, tat_r1], weights, active)
        np.testing.assert_array_almost_equal(out, tat_r0)

    def test_nan_in_region_excluded(self):
        """If one region has NaN, use only valid regions (renormalize)."""
        tat_r0 = np.array([[10.0, 20.0]])
        tat_r1 = np.array([[np.nan, 60.0]])  # pixel 0 has no detection
        weights = np.zeros((1, 2, 2))
        weights[0, :, 0] = 0.3
        weights[0, :, 1] = 0.7
        active = np.array([[0, 0], [0, 1]])
        out = combine_regions_soft([tat_r0, tat_r1], weights, active)
        # Pixel 0: only r0 valid, weight renormalized to 1.0 → 10.0
        assert out[0, 0] == 10.0
        # Pixel 1: both valid, weighted avg
        expected_1 = 0.3 / 1.0 * 20.0 + 0.7 / 1.0 * 60.0
        assert abs(out[0, 1] - expected_1) < 0.01


# ============== consensus_tat_methods ==============

class TestConsensusTatMethods:
    def test_consensus_shape(self):
        """Output should be (H, W)."""
        preproc = make_synthetic_video(n_pixels=5)
        mask = np.ones((5, 1), dtype=bool)
        tat = consensus_tat_methods(
            preproc, mask, 50, 150, 500.0,
            methods=["threshold_50pct", "derivative_max"]
        )
        assert tat.shape == (5, 1)

    def test_consensus_agrees_on_synthetic(self):
        """On clean synthetic data, both methods should agree."""
        preproc = make_synthetic_video(n_pixels=3)
        mask = np.ones((3, 1), dtype=bool)
        tat = consensus_tat_methods(
            preproc, mask, 50, 150, 500.0,
            methods=["threshold_50pct", "derivative_max"],
            agreement_threshold_ms=10.0,
        )
        # Most pixels should have valid consensus
        valid = tat[mask]
        valid = valid[np.isfinite(valid)]
        assert len(valid) >= 2, "Expected at least 2 valid consensus pixels"
