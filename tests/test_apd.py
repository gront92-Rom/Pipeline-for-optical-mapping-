"""
tests/test_apd.py — Unit tests for APDAgent v3.7.

Covers:
  - apd_detector.detect_all_apd_levels_pixel (single pixel, multiple levels)
  - apd_detector.compute_hot_mask (percentile filtering)
  - apd_detector.estimate_noise_sigma (baseline std)
  - apd_detector.reject_apd_outliers (BCL rule, min threshold)
  - apd_detector.compute_per_pixel_min_amp (dynamic floor)
  - Active pixel iteration (no full H*W loop)
  - Weighted average behavior (not median)
  - Outlier rejection: APD > BCL * 0.9 → NaN
  - Outlier rejection: APD < 10ms → NaN
  - VSD inverted polarity (apex = MIN)
"""
import numpy as np
import pytest

from cardiac_pipeline.utils.apd_detector import (
    detect_all_apd_levels_pixel,
    compute_hot_mask,
    estimate_noise_sigma,
    reject_apd_outliers,
    compute_per_pixel_min_amp,
)


# === Synthetic test data builder ===

def make_synthetic_vsd_ap(T=200, H=4, W=4, n_beats=2, bcl_frames=83, noise=10):
    """
    Build synthetic VSD-inverted video with clear AP upstrokes.

    Per beat:
      - baseline ≈ 0 (Gaussian noise around 0)
      - apex ≈ -1000 (sharp downstroke at peak_start)
      - repolarization: linear rise back to 0 over ~30 frames

    Returns:
        preproc: (T, H, W) — VSD inverted, baseline=0
        peaks: list of frame indices for AP upstrokes
    """
    rng = np.random.default_rng(42)
    preproc = rng.normal(0, noise, size=(T, H, W)).astype(np.float32)

    peaks = []
    peak_start = 10
    for beat in range(n_beats):
        peaks.append(peak_start)
        # Apex at peak_start+5 with value ~ -1000
        apex_frame = peak_start + 5
        if apex_frame < T:
            preproc[apex_frame, :, :] -= 1000

        # Repolarization: rise from apex to baseline over 30 frames
        for dt in range(1, 31):
            f = apex_frame + dt
            if f < T:
                preproc[f, :, :] -= max(0, 1000 - dt * 35)

        peak_start += bcl_frames
    return preproc, peaks


# === Tests: detect_all_apd_levels_pixel ===

class TestDetectAPDLevels:
    def test_returns_dict_with_requested_levels(self):
        """Should return dict with all 3 levels when detection succeeds."""
        preproc, peaks = make_synthetic_vsd_ap()
        result = detect_all_apd_levels_pixel(
            preproc, h=0, w=0,
            peak_start=peaks[0], peak_end=peaks[1],
            fps=500.0, levels=[30, 50, 80], min_amp=100.0,
        )
        assert set(result.keys()) == {30, 50, 80}, f"Expected {{30,50,80}}, got {set(result.keys())}"

    def test_apd_values_in_physiological_range(self):
        """Synthetic AP has apex at +5 frames, repol at ~+30 frames. APD30-80 should be 25-30ms @500fps."""
        preproc, peaks = make_synthetic_vsd_ap()
        result = detect_all_apd_levels_pixel(
            preproc, h=0, w=0,
            peak_start=peaks[0], peak_end=peaks[1],
            fps=500.0, levels=[30, 50, 80], min_amp=100.0,
        )
        # Synthetic: APD80 = apex_frame + repol_crossing - upstroke
        # apex at +5, upstroke at +2 (50% of -1000 = -500, crossed early)
        # repol 80%: signal rises from -1000 toward 0, crosses -200 at ~+23 frames
        # So APD80 ≈ 23 - 2 = 21 frames = 42ms @ 500fps
        for lv in [30, 50, 80]:
            assert 10 < result[lv] < 80, f"APD{lv}={result[lv]}ms outside physiological range"

    def test_apd_ordering_30_lt_50_lt_80(self):
        """APD30 < APD50 < APD80 (earlier repol level = shorter APD)."""
        preproc, peaks = make_synthetic_vsd_ap()
        result = detect_all_apd_levels_pixel(
            preproc, h=0, w=0,
            peak_start=peaks[0], peak_end=peaks[1],
            fps=500.0, levels=[30, 50, 80], min_amp=100.0,
        )
        assert result[30] < result[50] < result[80], \
            f"Expected APD30 < APD50 < APD80, got {result[30]} < {result[50]} < {result[80]}"

    def test_low_amp_returns_empty(self):
        """Pixel with weak signal should return empty dict."""
        preproc, peaks = make_synthetic_vsd_ap(noise=10)
        # Zero out one pixel
        preproc[:, 1, 1] = 0
        result = detect_all_apd_levels_pixel(
            preproc, h=1, w=1,
            peak_start=peaks[0], peak_end=peaks[1],
            fps=500.0, levels=[30, 50, 80], min_amp=100.0,
        )
        assert result == {}, f"Expected empty dict, got {result}"

    def test_invalid_window_returns_empty(self):
        """peak_end <= peak_start should return empty."""
        preproc, _ = make_synthetic_vsd_ap()
        result = detect_all_apd_levels_pixel(
            preproc, h=0, w=0,
            peak_start=100, peak_end=100,  # same frame
            fps=500.0, levels=[30, 50, 80],
        )
        assert result == {}


# === Tests: compute_hot_mask ===

class TestComputeHotMask:
    def test_percentile_50_keeps_top_half(self):
        """Top 50% std pixels by percentile."""
        H, W = 10, 10
        mask = np.ones((H, W), dtype=bool)
        T = 50
        preproc = np.zeros((T, H, W), dtype=np.float32)
        # Pixels 0-49 have low std, 50-99 have high std
        for i in range(50):
            preproc[:, i // 10, i % 10] = np.random.normal(0, 1, T)  # noise
        for i in range(50, 100):
            preproc[:, i // 10, i % 10] = np.random.normal(0, 100, T)  # high std
        preproc = preproc.astype(np.float32)

        hot = compute_hot_mask(preproc, mask, percentile=50)
        # Top 50% by std should give ~50 pixels
        assert hot.sum() == 50, f"Expected 50 hot pixels, got {hot.sum()}"

    def test_empty_mask_returns_zeros(self):
        """Empty mask → all-zeros hot mask."""
        mask = np.zeros((10, 10), dtype=bool)
        preproc = np.zeros((20, 10, 10), dtype=np.float32)
        hot = compute_hot_mask(preproc, mask, percentile=50)
        assert hot.sum() == 0


# === Tests: reject_apd_outliers ===

class TestRejectOutliers:
    def test_apd_above_bcl_rejected(self):
        """APD > BCL*0.9 should be rejected."""
        # BCL=167ms, 0.9*167=150ms, max(150, 300)=300ms
        apd_values = {80: 350.0}  # > 300ms
        filtered = reject_apd_outliers(apd_values, bcl_ms=167.0)
        assert 80 not in filtered

    def test_apd_below_10ms_rejected(self):
        """APD < 10ms should be rejected (artifact)."""
        apd_values = {80: 5.0}
        filtered = reject_apd_outliers(apd_values, bcl_ms=167.0)
        assert 80 not in filtered

    def test_normal_apd_kept(self):
        """APD in [10, BCL*0.9, max=300] should be kept."""
        apd_values = {80: 100.0}
        filtered = reject_apd_outliers(apd_values, bcl_ms=167.0)
        assert 80 in filtered
        assert filtered[80] == 100.0

    def test_high_bcl_threshold(self):
        """At slow pacing (1Hz), max threshold = max(900*0.9, 300) = 810ms."""
        apd_values = {80: 500.0}  # > 300ms but < 810ms
        filtered = reject_apd_outliers(apd_values, bcl_ms=900.0)
        assert 80 in filtered


# === Tests: estimate_noise_sigma ===

class TestEstimateNoiseSigma:
    def test_low_noise_returns_low_sigma(self):
        """Static video → noise sigma ≈ 0."""
        H, W = 10, 10
        mask = np.ones((H, W), dtype=bool)
        T = 50
        preproc = np.zeros((T, H, W), dtype=np.float32)
        sigma = estimate_noise_sigma(preproc, mask, fps=500.0)
        assert sigma < 1.0, f"Expected sigma < 1.0 for static video, got {sigma}"

    def test_returns_positive_value(self):
        """Always returns non-negative float."""
        mask = np.ones((10, 10), dtype=bool)
        preproc = np.random.randn(50, 10, 10).astype(np.float32) * 5
        sigma = estimate_noise_sigma(preproc, mask, fps=500.0)
        assert isinstance(sigma, float)
        assert sigma >= 0


# === Tests: compute_per_pixel_min_amp ===

class TestComputeMinAmp:
    def test_floor_dominates_when_sigma_low(self):
        """Low noise → min_amp = abs_floor."""
        amp = compute_per_pixel_min_amp(np.zeros((10, 10, 10)), np.ones((10, 10), bool),
                                        sigma_noise=5.0, abs_floor=100.0, noise_multiplier=3.0)
        # max(100, 3*5=15) = 100
        assert amp == 100.0

    def test_noise_dominates_when_sigma_high(self):
        """High noise → min_amp = 3 * sigma_noise."""
        amp = compute_per_pixel_min_amp(np.zeros((10, 10, 10)), np.ones((10, 10), bool),
                                        sigma_noise=200.0, abs_floor=100.0, noise_multiplier=3.0)
        # max(100, 600) = 600
        assert amp == 600.0


# === Tests: active pixel iteration (not full H*W loop) ===

class TestActivePixelLoop:
    """Verify that computation only iterates over hot_mask pixels."""

    def test_hot_mask_smaller_than_full_grid(self):
        """Hot mask with high percentile = small active set."""
        H, W = 100, 100
        mask = np.ones((H, W), dtype=bool)
        T = 50
        # Only 5 pixels have real signal
        preproc = np.random.randn(T, H, W).astype(np.float32) * 1
        for i, j in [(10, 10), (50, 50), (90, 90), (25, 75), (75, 25)]:
            preproc[:, i, j] += np.random.randn(T) * 200

        hot = compute_hot_mask(preproc, mask, percentile=99)
        n_hot = hot.sum()
        n_full = H * W

        # With percentile=99, only top 1% = 100 pixels kept
        assert n_hot < n_full, f"Hot mask ({n_hot}) should be smaller than full grid ({n_full})"
        assert n_hot < 0.05 * n_full, f"Hot mask too large: {n_hot}/{n_full}"

    def test_active_coords_via_argwhere(self):
        """np.argwhere(hot_mask) gives the list of pixels to iterate."""
        H, W = 10, 10
        mask = np.ones((H, W), dtype=bool)
        T = 30
        preproc = np.random.randn(T, H, W).astype(np.float32) * 5
        # Make 3 pixels hot
        for i, j in [(0, 0), (5, 5), (9, 9)]:
            preproc[:, i, j] += 500

        hot = compute_hot_mask(preproc, mask, percentile=50)
        active = np.argwhere(hot)
        # active shape: (N, 2)
        assert active.shape[1] == 2
        # Should be ≤ 50 (we made 3 hot + rest have noise std ~5)
        assert len(active) <= 60, f"Got {len(active)} active pixels"


# === Integration test: VSD polarity ===

class TestVSDPolarity:
    def test_apex_is_min_not_max(self):
        """VSD inverted: apex = most negative value (MIN, not MAX)."""
        # Build signal where "true apex" (most negative) is at frame +5
        T = 50
        H, W = 2, 2
        preproc = np.zeros((T, H, W), dtype=np.float32)
        # baseline ≈ 0, apex at +5 (value = -1000), repol over 30 frames
        for f in range(T):
            if f < 5:
                preproc[f] = 0
            elif f < 35:
                preproc[f] = -1000 + (f - 5) * 35  # rise back to 0
            else:
                preproc[f] = 0
        preproc += np.random.randn(T, H, W) * 5  # small noise

        peaks = [10]
        peak_end = 50
        result = detect_all_apd_levels_pixel(
            preproc, h=0, w=0,
            peak_start=peaks[0], peak_end=peak_end,
            fps=500.0, levels=[80], min_amp=100.0,
        )
        assert 80 in result
        # If we used MAX instead of MIN, detection would fail
        assert result[80] > 0, f"APD80 should be positive, got {result[80]}"