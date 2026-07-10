"""
voting.py — consensus peak detection across multiple traces.

v3.7 spec (2026-07-09):
  - Group peaks from all regions within ±tolerance frames
  - Keep groups with ≥ min_agreement unique regions
  - Return consensus frame (median) + agreement fraction

Robust to:
  - One region failing entirely (background / dead tissue)
  - Slight temporal misalignment across regions (conduction delay)
"""
import numpy as np
from typing import List, Tuple


def consensus_peaks(
    peaks_per_region: List[np.ndarray],
    n_regions: int,
    min_agreement: int = 2,
    frame_tolerance: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group peaks across regions, keep those with ≥ min_agreement regions.

    Parameters
    ----------
    peaks_per_region : list of (n_peaks_r,) int64
        Per-region peak frame indices (each may have different length).
    n_regions : int
        Total number of regions (for agreement fraction).
    min_agreement : int
        Minimum unique regions agreeing (default 2).
    frame_tolerance : int
        Max frame difference for "agreement" (default 10 = ±20ms @ 500fps).

    Returns
    -------
    consensus_peaks : (N,) int64
        Median frame index per consensus group.
    agreement : (N,) float32
        Fraction of regions agreeing (n_agreeing / n_regions).

    Algorithm:
      1. Flatten all peaks with region labels: [(frame, region_id), ...]
      2. Sort by frame
      3. Greedy scan: collect all peaks within ±tolerance of current
      4. Count unique regions in group
      5. If ≥ min_agreement: emit (median_frame, n_unique / n_regions)
      6. Skip processed peaks, continue
    """
    # 1. Flatten with labels
    all_peaks = []
    for r, peaks in enumerate(peaks_per_region):
        for p in peaks:
            all_peaks.append((int(p), r))
    all_peaks.sort(key=lambda x: x[0])

    if not all_peaks:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # 2-6. Greedy grouping
    consensus = []
    agreement = []
    i = 0
    n = len(all_peaks)
    while i < n:
        # Start group with peak at i
        group_frame_start = all_peaks[i][0]
        group_frames = [all_peaks[i][0]]
        group_regions = {all_peaks[i][1]}

        # Extend group: peaks within ±tolerance of group_frame_start
        j = i + 1
        while j < n and all_peaks[j][0] - group_frame_start <= frame_tolerance:
            group_frames.append(all_peaks[j][0])
            group_regions.add(all_peaks[j][1])
            j += 1

        # Also extend backwards: peaks within ±tolerance of group median
        # (handles case where later peaks attract earlier ones)
        # Actually our greedy forward scan is fine for monotonic sorting.

        n_agreeing = len(group_regions)
        if n_agreeing >= min_agreement:
            median_frame = int(np.median(group_frames))
            consensus.append(median_frame)
            agreement.append(n_agreeing / n_regions)

        i = j  # skip all processed peaks

    return (
        np.array(consensus, dtype=np.int64),
        np.array(agreement, dtype=np.float32),
    )


def select_top_beats(
    consensus_peaks_arr: np.ndarray,
    agreement: np.ndarray,
    n_beats: int = 3,
    min_quality: float = 0.66,
    sort_by: str = "agreement",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top-n_beats peaks by quality or temporal order.

    Parameters
    ----------
    consensus_peaks_arr : (N,) int64
        Consensus peak frames.
    agreement : (N,) float32
        Per-peak agreement fraction (0-1).
    n_beats : int
        Number of beats to select (default 3).
    min_quality : float
        Minimum agreement to be eligible (default 0.66 = ≥ 2/3 regions).
    sort_by : str
        "agreement" — pick highest-quality peaks (default)
        "temporal" — pick first N peaks in time

    Returns
    -------
    selected_peaks : (n_beats,) int64
        Selected peak frames (padded with -1 if fewer available).
    selected_indices : (n_beats,) int64
        Original indices in consensus_peaks_arr (padded with -1).

    Fallback: if fewer than n_beats meet min_quality, take top available.
    """
    n_total = len(consensus_peaks_arr)

    if n_total == 0:
        return (
            np.full(n_beats, -1, dtype=np.int64),
            np.full(n_beats, -1, dtype=np.int64),
        )

    if sort_by == "agreement":
        # Filter by quality, then sort by agreement DESC
        valid_mask = agreement >= min_quality
        if valid_mask.sum() < n_beats:
            # Fallback: use top-n regardless of quality
            order = np.argsort(-agreement)[:n_beats]
        else:
            valid_indices = np.where(valid_mask)[0]
            order = valid_indices[np.argsort(-agreement[valid_indices])[:n_beats]]
    elif sort_by == "temporal":
        # Take first n_beats in temporal order
        order = np.arange(min(n_beats, n_total))
    else:
        raise ValueError(f"sort_by must be 'agreement' or 'temporal', got '{sort_by}'")

    # Pad with -1 if we have fewer than n_beats
    selected_peaks = np.full(n_beats, -1, dtype=np.int64)
    selected_indices = np.full(n_beats, -1, dtype=np.int64)
    n_selected = min(len(order), n_beats)
    selected_peaks[:n_selected] = consensus_peaks_arr[order[:n_selected]]
    selected_indices[:n_selected] = order[:n_selected]

    return selected_peaks, selected_indices