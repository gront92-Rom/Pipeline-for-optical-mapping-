"""Generate v3.7 multi-trace visualization PNG for 004A.

Layout:
  Row 1 (4 subplots): per-region smoothed traces with peaks marked
  Row 2 (1 wide):     consensus peaks + agreement quality
  Row 3 (3 subplots): spatial layout (region_masks + weights + region_quality)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent / "src"))

SAMPLE = "004A"
RESULTS = Path("results")
MUST = RESULTS / SAMPLE / "must"
DEBUG = RESULTS / SAMPLE / "debug"
DEBUG.mkdir(parents=True, exist_ok=True)

# Load all artifacts
peaks_global = np.load(MUST / "peaks.npy")
peaks_per_region = np.load(MUST / "peaks_per_region.npy")
region_centers = np.load(MUST / "region_centers.npy")
region_masks = np.load(MUST / "region_masks.npy")  # (n_regions, H, W) uint8
region_quality = np.load(MUST / "region_quality.npy")
weights = np.load(MUST / "weights.npy")  # (H, W, n_regions)
consensus_peaks = np.load(MUST / "consensus_peaks.npy")
consensus_agreement = np.load(MUST / "consensus_agreement.npy")
selected_peaks = np.load(MUST / "selected_peaks.npy")
selected_indices = np.load(MUST / "selected_indices.npy")
traces_per_region = np.load(MUST / "traces_per_region.npy")
preproc_video = np.load(MUST / "preproc_video.npy")
mask = np.load(MUST / "mask.npy").astype(bool)
meta = dict(np.load(MUST / "peak_detection_meta.json".replace(".npy", ".json"))) if False else None

import json
with open(MUST / "peak_detection_meta.json") as f:
    meta = json.load(f)

fps = float(meta["fps"])
n_regions = peaks_per_region.shape[0]
T = traces_per_region.shape[1]
t_ms = np.arange(T) / fps * 1000

# Figure
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

# === Row 1: per-region traces ===
for r in range(n_regions):
    ax = fig.add_subplot(gs[0, r])
    ax.plot(t_ms, traces_per_region[r], color=f"C{r}", linewidth=0.8)
    # Mark this region's peaks
    region_peaks = peaks_per_region[r]
    region_peaks = region_peaks[region_peaks >= 0]
    ax.scatter(t_ms[region_peaks], traces_per_region[r][region_peaks],
               color="red", s=30, zorder=5, label=f"{len(region_peaks)} peaks")
    ax.set_title(f"Region {r}: center={tuple(region_centers[r].tolist())}, "
                 f"n_pix={int(region_masks[r].sum())}, q={region_quality[r]:.1f}",
                 fontsize=9)
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.set_ylabel("Fluorescence (a.u.)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

# 4th subplot row 1: pixel_std heatmap with region centers
ax = fig.add_subplot(gs[0, 3])
pixel_std = preproc_video.reshape(T, -1).std(axis=0).reshape(100, 100)
im = ax.imshow(pixel_std, cmap="hot", vmin=pixel_std[mask].min(), vmax=pixel_std[mask].max())
ax.contour(mask, colors="cyan", linewidths=0.5, alpha=0.4)
for r, (cy, cx) in enumerate(region_centers):
    ax.scatter(cx, cy, s=200, c=f"C{r}", marker="o", edgecolors="white", linewidths=2)
    ax.annotate(f"R{r}", (cx, cy), color="white", fontsize=10,
                ha="center", va="center", fontweight="bold")
plt.colorbar(im, ax=ax, label="pixel_std")
ax.set_title("pixel_std + regions", fontsize=10)

# === Row 2: consensus ===
ax = fig.add_subplot(gs[1, :2])
# Average trace across regions
avg_trace = traces_per_region.mean(axis=0)
ax.plot(t_ms, avg_trace, color="black", linewidth=0.8, alpha=0.6, label="avg regions")
# Mark consensus peaks with color = agreement
sc = ax.scatter(t_ms[consensus_peaks], avg_trace[consensus_peaks],
               c=consensus_agreement, cmap="viridis", s=80, zorder=5,
               vmin=0.5, vmax=1.0, edgecolors="red", linewidths=1)
# Highlight selected peaks (top-3)
selected_valid = selected_peaks[selected_peaks >= 0]
ax.scatter(t_ms[selected_valid], avg_trace[selected_valid],
           marker="*", s=300, color="orange", edgecolors="black", linewidths=1.5,
           zorder=6, label=f"selected ({len(selected_valid)})")
plt.colorbar(sc, ax=ax, label="agreement")
ax.set_title(f"Consensus peaks: {len(consensus_peaks)} (top-{len(selected_valid)} selected)",
             fontsize=10)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Fluorescence (a.u.)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Row 2 right: agreement histogram
ax = fig.add_subplot(gs[1, 2:])
ax.bar(range(len(consensus_agreement)), consensus_agreement,
       color=["C0" if a == 1.0 else "C1" for a in consensus_agreement])
ax.axhline(0.66, color="red", linestyle="--", linewidth=1, label="min_quality=0.66")
ax.set_ylim(0, 1.1)
ax.set_xlabel("Consensus peak index")
ax.set_ylabel("Agreement (frac regions)")
ax.set_title("Per-peak agreement quality", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# === Row 3: spatial layout ===
# Subplot (3,0): region assignment (hard)
ax = fig.add_subplot(gs[2, 0])
hard = np.argmax(weights, axis=2)
hard_masked = np.where(mask, hard, -1)
im = ax.imshow(hard_masked, cmap="tab10", vmin=0, vmax=n_regions-1)
ax.contour(mask, colors="white", linewidths=0.5, alpha=0.3)
ax.set_title("Hard region assignment (argmax)", fontsize=10)
plt.colorbar(im, ax=ax, label="region_id")

# Subplot (3,1): weight[0] for region 0
ax = fig.add_subplot(gs[2, 1])
w0 = weights[:, :, 0]
w0_masked = np.where(mask, w0, np.nan)
im = ax.imshow(w0_masked, cmap="Reds", vmin=0, vmax=1)
ax.contour(mask, colors="white", linewidths=0.5, alpha=0.3)
ax.scatter(region_centers[0, 1], region_centers[0, 0], s=200, c="black", marker="+", linewidths=2)
ax.set_title(f"Soft weights → Region 0 (center={tuple(region_centers[0].tolist())})", fontsize=10)
plt.colorbar(im, ax=ax, label="weight")

# Subplot (3,2): Δt per region (bar)
ax = fig.add_subplot(gs[2, 2])
dt_per_region = []
for r in range(n_regions):
    p = peaks_per_region[r]
    p = p[p >= 0]
    if len(p) > 1:
        dt_per_region.append(np.diff(p) / fps * 1000)
    else:
        dt_per_region.append([0])

bp = ax.boxplot(dt_per_region, positions=range(n_regions), widths=0.5, patch_artist=True)
for patch, color in zip(bp['boxes'], [f"C{r}" for r in range(n_regions)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.axhline(1000/5.86, color="red", linestyle="--", linewidth=1, label=f"6Hz BCL={1000/5.86:.1f}ms")
ax.set_xticks(range(n_regions))
ax.set_xticklabels([f"R{r}" for r in range(n_regions)])
ax.set_ylabel("Δt between peaks (ms)")
ax.set_title("Per-region Δt consistency", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Subplot (3,3): region quality bar
ax = fig.add_subplot(gs[2, 3])
colors = [f"C{r}" for r in range(n_regions)]
ax.bar(range(n_regions), region_quality, color=colors, alpha=0.7)
ax.set_xticks(range(n_regions))
ax.set_xticklabels([f"R{r}" for r in range(n_regions)])
ax.set_ylabel("Region quality (mean pixel_std)")
ax.set_title("Region quality scores", fontsize=10)
ax.grid(True, alpha=0.3)

# Title
fig.suptitle(
    f"004A bsl-6Hz | v3.7 multi-trace | "
    f"n_regions={n_regions} | fps={fps:.0f} | stim_hz={meta['stim_hz']:.2f} | "
    f"consensus={len(consensus_peaks)} peaks, "
    f"agreement mean={consensus_agreement.mean():.2f} | "
    f"selected={len(selected_valid)} beats: {selected_valid.tolist()}",
    fontsize=12, fontweight="bold", y=0.99
)

out_path = DEBUG / "peaks_v3_7_multi_trace.png"
plt.savefig(out_path, dpi=100, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")