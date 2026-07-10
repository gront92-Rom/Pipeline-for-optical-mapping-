"""Generate v3.7 APD map visualization PNG for 004A."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

SAMPLE = "004A"
MUST = Path(f"results/{SAMPLE}/must")
DEBUG = Path(f"results/{SAMPLE}/debug")

apd30 = np.load(MUST / "apd30_map.npy")
apd50 = np.load(MUST / "apd50_map.npy")
apd80 = np.load(MUST / "apd80_map.npy")
hot_mask = np.load(DEBUG / "hot_mask.npy")
mask = np.load(MUST / "mask.npy").astype(bool)
apd_4d = np.load(DEBUG / "apd_4d.npy")
preproc = np.load(MUST / "preproc_video.npy")
region_masks = np.load(MUST / "region_masks.npy")
region_centers = np.load(MUST / "region_centers.npy")

with open(MUST / "apd_report.json") as f:
    report = json.load(f)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# === Row 1: APD30 / APD50 / APD80 maps ===
vmin, vmax = 40, 90
for col, (lv, apd_map) in enumerate([(30, apd30), (50, apd50), (80, apd80)]):
    ax = fig.add_subplot(gs[0, col])
    masked_apd = np.where(hot_mask & np.isfinite(apd_map), apd_map, np.nan)
    im = ax.imshow(masked_apd, cmap="jet", vmin=vmin, vmax=vmax)
    ax.contour(mask, colors="white", linewidths=0.3, alpha=0.3)
    # Mark region centers
    for r, (cy, cx) in enumerate(region_centers):
        ax.scatter(cx, cy, s=80, c=f"C{r}", marker="o", edgecolors="white", linewidths=1.5)
    plt.colorbar(im, ax=ax, label=f"APD{lv} (ms)")
    ax.set_title(f"APD{lv} map (median={report[f'apd{lv}_median_ms']:.1f}ms, "
                 f"valid={report[f'apd{lv}_n_valid']})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

# === Row 2: APD80 per-beat comparison ===
for col in range(3):
    ax = fig.add_subplot(gs[1, col])
    beat_apd = apd_4d[2, :, :, col]  # APD80, beat i
    masked = np.where(hot_mask & np.isfinite(beat_apd), beat_apd, np.nan)
    im = ax.imshow(masked, cmap="jet", vmin=vmin, vmax=vmax)
    ax.contour(mask, colors="white", linewidths=0.3, alpha=0.3)
    plt.colorbar(im, ax=ax, label="APD80 (ms)")
    valid = beat_apd[hot_mask & np.isfinite(beat_apd)]
    if len(valid) > 0:
        ax.set_title(f"APD80 beat {col+1} (median={np.median(valid):.1f}ms)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

# === Row 3 left: APD histogram ===
ax = fig.add_subplot(gs[2, 0])
for lv, apd_map, color in [(30, apd30, "C0"), (50, apd50, "C1"), (80, apd80, "C2")]:
    valid = apd_map[hot_mask & np.isfinite(apd_map)]
    ax.hist(valid, bins=30, alpha=0.5, color=color, label=f"APD{lv} (median={np.median(valid):.1f})")
ax.set_xlabel("APD (ms)")
ax.set_ylabel("Pixel count")
ax.set_title("APD distribution", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# === Row 3 middle: Hot mask + APD80 coverage ===
ax = fig.add_subplot(gs[2, 1])
ax.imshow(hot_mask.astype(float), cmap="gray", alpha=0.5)
apd80_masked = np.where(hot_mask & np.isfinite(apd80), apd80, np.nan)
im = ax.imshow(apd80_masked, cmap="jet", vmin=vmin, vmax=vmax, alpha=0.7)
plt.colorbar(im, ax=ax, label="APD80 (ms)")
ax.set_title("Hot mask + APD80", fontsize=10)
ax.set_xticks([])
ax.set_yticks([])

# === Row 3 right: APD trace for top hot pixel ===
ax = fig.add_subplot(gs[2, 2])
ys, xs = np.where(hot_mask)
pixel_std = preproc.reshape(1024, -1).std(axis=0).reshape(100, 100)
hot_stds = pixel_std[ys, xs]
top_idx = np.argmax(hot_stds)
h, w = ys[top_idx], xs[top_idx]
t_ms = np.arange(1024) / 500 * 1000
sig = preproc[:, h, w]
ax.plot(t_ms, sig, color="black", linewidth=0.5)
# Mark selected peaks
selected = np.load(MUST / "selected_peaks.npy")
selected = selected[selected >= 0]
for i, p in enumerate(selected):
    ax.axvline(p / 500 * 1000, color="red", alpha=0.5, linewidth=1)
    ax.annotate(f"#{i+1}", (p/500*1000, sig[p]), color="red", fontsize=9)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Fluorescence (a.u.)")
ax.set_title(f"Top hot pixel ({h}, {w}): std={hot_stds[top_idx]:.1f}", fontsize=10)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"004A bsl-6Hz | v3.7 APD map | "
    f"n_active={report['n_active_pixels']}/{mask.sum()} | "
    f"min_amp={report['min_amp']:.0f} | "
    f"sigma_noise={report['sigma_noise']:.1f} | "
    f"BCL={report['bcl_ms']:.1f}ms",
    fontsize=13, fontweight="bold", y=0.99
)

out = DEBUG / "apd_v3_7_maps.png"
plt.savefig(out, dpi=100, bbox_inches="tight")
plt.close()
print(f"Saved {out}")