"""
region_selector.py — select n regions from mask for multi-trace PeakDet.

v3.7 spec (2026-07-09):
  - Mask-based 3x3 grid selection
  - Score = mean pixel_std weighted by log(n_pixels)
  - Spatial diversity: prefer non-adjacent cells
  - Returns region_masks + region_centers (centroid of high-std pixels)

NOT K-means (deterministic, faster, more predictable).
"""
import numpy as np
from typing import List, Tuple


def select_regions_grid(
    mask: np.ndarray,
    pixel_std: np.ndarray,
    n: int = 3,
    grid_size: int = 3,
    min_region_pixels: int = 50,
    min_std_percentile: float = 50.0,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[dict]]:
    """
    Select n regions from a 3x3 grid based on mask coverage + pixel_std.

    Algorithm:
      1. Divide mask into grid_size x grid_size cells
      2. For each cell, compute:
         - n_pixels = sum of mask in cell
         - mean_std = mean of pixel_std in masked pixels of cell
         - score = mean_std * log1p(n_pixels)
      3. Skip cells with n_pixels < min_region_pixels
      4. Sort cells by score DESC
      5. Select top-n with spatial diversity (skip adjacent if possible)
      6. For each selected cell, compute:
         - region_mask = mask clipped to cell
         - region_center = centroid of HIGH-STD pixels within cell
           (using min_std_percentile threshold)

    Parameters
    ----------
    mask : (H, W) bool
        Tissue mask.
    pixel_std : (H, W) float
        Per-pixel temporal std.
    n : int
        Number of regions to select (default 3).
    grid_size : int
        Grid resolution (default 3 = 3x3 = 9 cells).
    min_region_pixels : int
        Minimum pixels per region (default 50).
    min_std_percentile : float
        Percentile threshold for "high std" pixels used for centroid
        (default 50 = median).

    Returns
    -------
    region_masks : list of (H, W) bool
        Per-region tissue masks.
    region_centers : list of (y, x) tuples
        Per-region center coordinates (centroid of high-std pixels).
    region_info : list of dicts
        Per-region metadata (score, n_pixels, mean_std, cell coords).
    """
    H, W = mask.shape
    cell_h = H // grid_size
    cell_w = W // grid_size

    # 1-3. Compute cell scores
    cells = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0 = gy * cell_h
            y1 = (gy + 1) * cell_h if gy < grid_size - 1 else H
            x0 = gx * cell_w
            x1 = (gx + 1) * cell_w if gx < grid_size - 1 else W

            cell_mask = mask[y0:y1, x0:x1]
            n_pixels = int(cell_mask.sum())
            if n_pixels < min_region_pixels:
                continue

            masked_std = pixel_std[y0:y1, x0:x1][cell_mask]
            if len(masked_std) == 0:
                continue

            mean_std = float(masked_std.mean())
            score = mean_std * np.log1p(n_pixels)

            cells.append({
                "gy": gy, "gx": gx,
                "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                "n_pixels": n_pixels,
                "mean_std": mean_std,
                "score": score,
            })

    if not cells:
        raise ValueError(
            f"No cells with ≥{min_region_pixels} pixels found in {grid_size}x{grid_size} grid. "
            f"Mask coverage: {mask.sum() / mask.size:.1%}. "
            f"Try smaller grid_size or min_region_pixels."
        )

    # Sort by score DESC
    cells.sort(key=lambda c: -c["score"])

    # 4-5. Select with spatial diversity
    selected = []
    used_cells = set()
    for cell in cells:
        if len(selected) >= n:
            break
        gy, gx = cell["gy"], cell["gx"]
        # Skip adjacent cells if we already have n selected... but be lenient
        is_adjacent = any(
            abs(gy - sg["gy"]) <= 1 and abs(gx - sg["gx"]) <= 1
            for sg in selected
        )
        if is_adjacent and len(selected) < n and len(cells) > n:
            # Try to find non-adjacent; otherwise accept
            # Find any non-adjacent with score still high
            non_adj = [c for c in cells if c not in selected and not any(
                abs(c["gy"] - sg["gy"]) <= 1 and abs(c["gx"] - sg["gx"]) <= 1
                for sg in selected
            )]
            if non_adj:
                continue  # skip this adjacent, will pick non-adj later

        selected.append(cell)
        used_cells.add((gy, gx))

    # Fallback: if not enough diverse, take top-n anyway
    if len(selected) < n:
        for cell in cells:
            if cell not in selected:
                selected.append(cell)
                if len(selected) >= n:
                    break

    # 6. Build region_masks + compute centroids of high-std pixels
    region_masks = []
    region_centers = []
    region_info = []

    for cell in selected:
        y0, y1, x0, x1 = cell["y0"], cell["y1"], cell["x0"], cell["x1"]
        cell_mask = mask[y0:y1, x0:x1]
        cell_std = pixel_std[y0:y1, x0:x1]

        # Centroid of HIGH-std pixels within cell
        std_threshold = np.percentile(cell_std[cell_mask], min_std_percentile) \
            if cell_mask.sum() > 0 else 0
        high_std_mask = cell_mask & (cell_std >= std_threshold)

        if high_std_mask.sum() > 0:
            ys, xs = np.where(high_std_mask)
            cy = int(np.mean(ys)) + y0
            cx = int(np.mean(xs)) + x0
        else:
            # Fallback: geometric center of cell
            cy = (y0 + y1) // 2
            cx = (x0 + x1) // 2

        # Full-size region_mask
        rm = np.zeros_like(mask)
        rm[y0:y1, x0:x1] = cell_mask
        region_masks.append(rm)
        region_centers.append((cy, cx))

        info = dict(cell)  # copy
        info["center_y"] = cy
        info["center_x"] = cx
        info["high_std_pixels"] = int(high_std_mask.sum())
        region_info.append(info)

    return region_masks, region_centers, region_info