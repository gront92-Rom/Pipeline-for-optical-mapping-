"""
soft_assignment.py — Gaussian distance-weighted pixel-to-region assignment.

v3.7 spec (2026-07-09):
  - For each pixel, compute weights over n_regions based on Gaussian distance
  - weights[h, w, r] = exp(-dist^2 / (2*sigma^2))
  - Normalize: weights[h, w] /= sum (so sum=1 per pixel)

Used by APDAgent for soft-weighted per-pixel APD computation.
"""
import numpy as np
from typing import List, Tuple


def compute_soft_weights(
    region_centers: List[Tuple[int, int]],
    shape: Tuple[int, int],
    sigma: float = 20.0,
) -> np.ndarray:
    """
    Compute Gaussian distance-weighted soft assignment per pixel.

    Parameters
    ----------
    region_centers : list of (y, x) tuples
        Center of each region.
    shape : (H, W)
        Spatial shape of the mask/video.
    sigma : float
        Gaussian sigma in pixels (default 20 = ~20% of 100x100).

    Returns
    -------
    weights : (H, W, n_regions) float32
        Normalized weights per pixel per region (sum to 1 along last axis).

    Notes:
      - Boundary pixels (far from all centers) get smoother weighting
      - Center pixels of region r get weight ~1 for that region
      - Pixel exactly at a center gets weight = 1 (after normalization)
    """
    H, W = shape
    n_regions = len(region_centers)

    ys, xs = np.mgrid[0:H, 0:W]
    weights = np.zeros((H, W, n_regions), dtype=np.float32)

    for r, (cy, cx) in enumerate(region_centers):
        dist_sq = (ys.astype(np.float32) - float(cy))**2 + \
                  (xs.astype(np.float32) - float(cx))**2
        weights[:, :, r] = np.exp(-dist_sq / (2.0 * float(sigma) ** 2))

    # Normalize per pixel (sum to 1)
    total = weights.sum(axis=2, keepdims=True)
    total = np.maximum(total, 1e-12)  # avoid divide by zero
    weights /= total

    return weights


def hard_assignment_from_weights(weights: np.ndarray) -> np.ndarray:
    """
    Convert soft weights to hard assignment (argmax per pixel).

    Parameters
    ----------
    weights : (H, W, n_regions) float32

    Returns
    -------
    assignments : (H, W) int
        Region index per pixel (argmax of weights).
    """
    return np.argmax(weights, axis=2).astype(np.int32)