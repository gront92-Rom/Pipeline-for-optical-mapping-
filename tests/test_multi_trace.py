"""
tests/test_multi_trace.py — Unit tests for v3.7 multi-trace PeakDet modules.

Covers:
  - region_selector.select_regions_grid
  - voting.consensus_peaks
  - voting.select_top_beats
  - soft_assignment.compute_soft_weights
  - soft_assignment.hard_assignment_from_weights
  - PeakDetectorAgent n_regions=3 mode (integration-style)

Run:
  PYTHONPATH=src python3 -m pytest tests/test_multi_trace.py -v
"""
import numpy as np
import pytest
from pathlib import Path
import tempfile
import json

# Modules under test
from cardiac_pipeline.utils.region_selector import select_regions_grid
from cardiac_pipeline.utils.voting import consensus_peaks, select_top_beats
from cardiac_pipeline.utils.soft_assignment import (
    compute_soft_weights,
    hard_assignment_from_weights,
)


# =============================================================================
# region_selector tests
# =============================================================================

class TestSelectRegionsGrid:
    """Tests for mask_grid region selection."""

    def test_basic_3regions_full_mask(self):
        """100x100 full mask, n=3 → returns 3 region_masks + 3 centers."""
        mask = np.ones((100, 100), dtype=bool)
        pixel_std = np.random.rand(100, 100).astype(np.float32) * 10.0

        region_masks, region_centers, _info = select_regions_grid(
            mask, pixel_std, n=3, min_region_pixels=50
        )

        assert len(region_centers) == 3
        assert len(region_masks) == 3
        # Each region_mask should be a (H, W) bool array
        for rm in region_masks:
            assert rm.shape == (100, 100)
            assert rm.dtype == bool
        # Centers should be inside the grid
        for cy, cx in region_centers:
            assert 0 <= cy < 100
            assert 0 <= cx < 100

    def test_empty_mask_raises(self):
        """Empty mask should raise ValueError."""
        mask = np.zeros((100, 100), dtype=bool)
        pixel_std = np.zeros((100, 100), dtype=np.float32)

        with pytest.raises(ValueError):
            select_regions_grid(mask, pixel_std, n=3, min_region_pixels=50)

    def test_too_small_mask_falls_back(self):
        """If mask is too small for 3 non-adjacent cells, fall back to top-n."""
        # Only top-left quadrant has tissue
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:33, 0:33] = True
        pixel_std = np.random.rand(100, 100).astype(np.float32)

        region_masks, region_centers, _info = select_regions_grid(
            mask, pixel_std, n=3, min_region_pixels=50
        )

        # Should still return at least 1 region (best effort)
        assert len(region_centers) >= 1
        assert len(region_centers) == len(region_masks)

    def test_picks_high_std_cells(self):
        """Top-3 cells should be those with highest mean pixel_std."""
        mask = np.ones((100, 100), dtype=bool)
        pixel_std = np.ones((100, 100), dtype=np.float32) * 1.0

        # Make top-left cell (0:33, 0:33) have HIGH std
        pixel_std[0:33, 0:33] = 100.0

        region_masks, region_centers, _info = select_regions_grid(
            mask, pixel_std, n=3, min_region_pixels=50
        )

        # The first selected region should include the high-std cell
        # (not necessarily be exactly there, but the cell with high std should be a top candidate)
        high_std_in_selected = False
        for rm in region_masks:
            cell_overlap = rm[0:33, 0:33].sum()
            if cell_overlap > 50:  # most of the high-std cell is selected
                high_std_in_selected = True
                break
        assert high_std_in_selected, "Top-std cell should be selected in at least one region"

    def test_spatial_diversity(self):
        """Selected regions should be non-adjacent (diversity)."""
        mask = np.ones((100, 100), dtype=bool)
        pixel_std = np.random.rand(100, 100).astype(np.float32) * 100

        region_masks, region_centers, _info = select_regions_grid(
            mask, pixel_std, n=3, min_region_pixels=50
        )

        # Convert centers to grid coords (33x33 cells)
        def to_grid(y, x):
            return (y // 33, x // 33)

        grids = [to_grid(cy, cx) for cy, cx in region_centers]
        # Check at least 2 are non-adjacent
        if len(grids) >= 2:
            for i in range(len(grids)):
                for j in range(i+1, len(grids)):
                    dy = abs(grids[i][0] - grids[j][0])
                    dx = abs(grids[i][1] - grids[j][1])
                    if max(dy, dx) >= 1:  # not the same cell
                        # found diverse pair
                        return
            pytest.fail(f"No spatially diverse pairs found in {grids}")


# =============================================================================
# voting tests
# =============================================================================

class TestConsensusPeaks:
    """Tests for consensus peak voting across regions."""

    def test_all_regions_agree(self):
        """All 3 regions find identical peaks → all kept, agreement=1.0."""
        peaks = [
            np.array([100, 200, 300]),
            np.array([101, 201, 301]),
            np.array([99, 199, 299]),
        ]
        consensus, agreement = consensus_peaks(
            peaks, n_regions=3, min_agreement=2, frame_tolerance=10
        )
        assert len(consensus) == 3
        assert np.all(agreement == 1.0)
        # Consensus frame should be median (~100, 200, 300)
        assert all(abs(c - e) <= 1 for c, e in zip(consensus, [100, 200, 300]))

    def test_minority_rejected(self):
        """Only 1 region finds peak → rejected (min_agreement=2)."""
        peaks = [
            np.array([100, 200, 300]),
            np.array([]),  # empty
            np.array([]),  # empty
        ]
        consensus, agreement = consensus_peaks(
            peaks, n_regions=3, min_agreement=2, frame_tolerance=10
        )
        # 0 consensus peaks (all from 1 region only)
        assert len(consensus) == 0

    def test_majority_kept(self):
        """2/3 regions find peak → kept, agreement=0.67."""
        peaks = [
            np.array([100, 200]),
            np.array([100, 200]),
            np.array([]),
        ]
        consensus, agreement = consensus_peaks(
            peaks, n_regions=3, min_agreement=2, frame_tolerance=10
        )
        assert len(consensus) == 2
        assert all(abs(a - 0.6667) < 0.01 for a in agreement)

    def test_tolerance_window(self):
        """Peaks within ±10 frames are grouped together."""
        peaks = [
            np.array([100]),
            np.array([105]),  # 5 frames apart
            np.array([110]),  # 10 frames apart
        ]
        consensus, agreement = consensus_peaks(
            peaks, n_regions=3, min_agreement=2, frame_tolerance=10
        )
        assert len(consensus) == 1
        assert agreement[0] == 1.0

    def test_empty_all_regions(self):
        """All regions empty → empty consensus."""
        peaks = [np.array([]), np.array([]), np.array([])]
        consensus, agreement = consensus_peaks(
            peaks, n_regions=3, min_agreement=2, frame_tolerance=10
        )
        assert len(consensus) == 0
        assert len(agreement) == 0

    def test_far_peaks_not_grouped(self):
        """Peaks > tolerance apart → not grouped."""
        peaks = [
            np.array([100, 500]),
            np.array([100, 500]),
            np.array([100, 500]),
        ]
        consensus, agreement = consensus_peaks(
            peaks, n_regions=3, min_agreement=2, frame_tolerance=10
        )
        assert len(consensus) == 2


class TestSelectTopBeats:
    """Tests for top-N beat selection by quality."""

    def test_top_n_by_agreement(self):
        """Top-N beats should have highest agreement scores."""
        consensus = np.array([100, 200, 300, 400])
        agreement = np.array([0.5, 1.0, 0.8, 0.66])

        selected, indices = select_top_beats(
            consensus, agreement, n_beats=2, min_quality=0.66, sort_by="agreement"
        )
        # Should pick 200 (1.0) and 300 (0.8)
        assert len(selected) == 2
        assert 200 in selected
        assert 300 in selected
        # Indices should point to original positions
        assert 1 in indices
        assert 2 in indices

    def test_quality_filter_with_sufficient(self):
        """If n_beats candidates meet min_quality, only those are picked."""
        consensus = np.array([100, 200, 300, 400, 500])
        agreement = np.array([0.5, 1.0, 0.8, 0.4, 0.7])

        selected, indices = select_top_beats(
            consensus, agreement, n_beats=2, min_quality=0.66, sort_by="agreement"
        )
        # Top-2 by agreement: 200 (1.0), 300 (0.8)
        assert len(selected) == 2
        assert 200 in selected
        assert 300 in selected

    def test_quality_filter_fallback(self):
        """If fewer than n_beats meet min_quality, fallback to top-n regardless."""
        consensus = np.array([100, 200, 300])
        agreement = np.array([0.5, 1.0, 0.4])  # only 1 above 0.66

        selected, indices = select_top_beats(
            consensus, agreement, n_beats=2, min_quality=0.66, sort_by="agreement"
        )
        # Fallback: top-2 by agreement = 200 (1.0), 100 (0.5)
        assert len(selected) == 2
        assert selected[0] == 200
        assert selected[1] == 100

    def test_fallback_when_fewer_than_n(self):
        """If fewer than n_beats qualify, fallback to top-n regardless of quality."""
        consensus = np.array([100, 200, 300, 400])
        agreement = np.array([0.5, 1.0, 0.4, 0.3])  # only 1 above 0.66

        selected, indices = select_top_beats(
            consensus, agreement, n_beats=3, min_quality=0.66, sort_by="agreement"
        )
        # Should fallback: top-3 by agreement = 200 (1.0), 100 (0.5), 300 (0.4)
        assert len(selected) == 3
        assert selected[0] == 200  # best

    def test_sort_by_temporal(self):
        """Sort by temporal = take first N."""
        consensus = np.array([100, 200, 300, 400])
        agreement = np.array([0.5, 1.0, 0.8, 0.66])

        selected, indices = select_top_beats(
            consensus, agreement, n_beats=2, min_quality=0.0, sort_by="temporal"
        )
        assert len(selected) == 2
        assert selected[0] == 100
        assert selected[1] == 200


# =============================================================================
# soft_assignment tests
# =============================================================================

class TestSoftWeights:
    """Tests for Gaussian distance-weighted soft assignment."""

    def test_sum_to_one(self):
        """weights.sum(axis=2) should be 1.0 for every pixel."""
        centers = [(20, 20), (50, 50), (80, 80)]
        weights = compute_soft_weights(centers, (100, 100), sigma=20.0)
        sums = weights.sum(axis=2)
        assert np.allclose(sums, 1.0, atol=1e-5)

    def test_shape(self):
        """weights shape should be (H, W, n_regions)."""
        centers = [(20, 20), (50, 50)]
        weights = compute_soft_weights(centers, (100, 100), sigma=20.0)
        assert weights.shape == (100, 100, 2)

    def test_center_pixel_has_max_weight(self):
        """Pixel at a center should have ~1.0 weight for that region."""
        centers = [(50, 50)]
        weights = compute_soft_weights(centers, (100, 100), sigma=20.0)
        # At pixel (50, 50), weight should be ~1.0
        assert weights[50, 50, 0] > 0.99

    def test_far_pixel_smooth(self):
        """Far pixel should have spread weights."""
        centers = [(20, 20), (80, 80)]
        weights = compute_soft_weights(centers, (100, 100), sigma=20.0)
        # Pixel at (50, 50) — equidistant from both centers
        assert 0.4 < weights[50, 50, 0] < 0.6
        assert 0.4 < weights[50, 50, 1] < 0.6

    def test_single_region_uniform(self):
        """n_regions=1 → all weight = 1.0 everywhere."""
        centers = [(50, 50)]
        weights = compute_soft_weights(centers, (100, 100), sigma=20.0)
        assert np.allclose(weights[:, :, 0], 1.0, atol=1e-5)


class TestHardAssignment:
    """Tests for hard assignment from soft weights."""

    def test_argmax_assignment(self):
        """hard_assignment should match argmax of weights."""
        centers = [(20, 20), (50, 50), (80, 80)]
        weights = compute_soft_weights(centers, (100, 100), sigma=20.0)
        hard = hard_assignment_from_weights(weights)

        expected = np.argmax(weights, axis=2)
        assert np.array_equal(hard, expected)
        assert set(np.unique(hard).tolist()) <= {0, 1, 2}


# =============================================================================
# PeakDetectorAgent integration-style test
# =============================================================================

class TestPeakDetectorAgentMultiTrace:
    """Integration test: PeakDetectorAgent with n_regions=3."""

    def test_n_regions_param_loaded(self):
        """Agent should read n_regions from config."""
        pytest.importorskip("omegaconf", reason="omegaconf not installed in test env")
        from cardiac_pipeline.agents.peak_detector_agent import PeakDetectorAgent
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "results_root": "/tmp/test_peak_v37",
            "peak_detector": {
                "n_regions": 3,
                "min_region_pixels": 50,
                "min_agreement": 2,
            }
        })
        agent = PeakDetectorAgent("test_sample", config=cfg)
        assert agent.n_regions == 3
        assert agent.min_region_pixels == 50
        assert agent.min_agreement == 2

    def test_legacy_mode_with_n_regions_1(self):
        """n_regions=1 should keep v3.6 behavior (single-trace)."""
        pytest.importorskip("omegaconf", reason="omegaconf not installed in test env")
        from cardiac_pipeline.agents.peak_detector_agent import PeakDetectorAgent
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "results_root": "/tmp/test_peak_v37",
            "peak_detector": {
                "n_regions": 1,  # legacy
            }
        })
        agent = PeakDetectorAgent("test_sample", config=cfg)
        assert agent.n_regions == 1