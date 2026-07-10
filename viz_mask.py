"""Generate mask overlay visualization."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLE = "004A"
MUST = Path(f"results/{SAMPLE}/must")
DEBUG = Path(f"results/{SAMPLE}/debug")

raw_rsm = np.load(MUST / "raw_rsm.npy").squeeze()
mask = np.load(MUST / "mask.npy").astype(bool)
preproc = np.load(MUST / "preproc_video.npy")
T = preproc.shape[0]
pixel_std = preproc.reshape(T, -1).std(axis=0).reshape(100, 100)

# Also load 1 frame of raw signal
raw_video = np.load(MUST / "raw_video.npy")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 0: background, mask overlay, std
im = axes[0, 0].imshow(raw_rsm, cmap="gray")
plt.colorbar(im, ax=axes[0, 0], label="Fluorescence (a.u.)")
axes[0, 0].set_title(f"raw_rsm background\nrange [{raw_rsm.min():.0f}, {raw_rsm.max():.0f}]")

axes[0, 1].imshow(raw_rsm, cmap="gray", alpha=0.6)
axes[0, 1].contour(mask, colors="red", linewidths=1.5)
axes[0, 1].set_title(f"mask overlay ({mask.sum()} px, {100*mask.sum()/10000:.1f}%)")

im = axes[0, 2].imshow(pixel_std, cmap="viridis")
plt.colorbar(im, ax=axes[0, 2], label="pixel_std")
axes[0, 2].set_title(f"pixel_std (range [{pixel_std.min():.1f}, {pixel_std.max():.1f}])")

# Row 1: histogram, signal frame, std histogram
axes[1, 0].hist(raw_rsm[mask], bins=50, alpha=0.7, label=f"in mask ({mask.sum()})", color="red")
axes[1, 0].hist(raw_rsm[~mask], bins=50, alpha=0.5, label=f"out mask ({(~mask).sum()})", color="blue")
axes[1, 0].set_xlabel("raw_rsm fluorescence")
axes[1, 0].set_ylabel("Pixel count")
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("Raw fluorescence distribution")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

frame_show = raw_video[T // 2] if raw_video.ndim == 3 else raw_video[T // 2, ..., 0]
axes[1, 1].imshow(frame_show, cmap="gray")
axes[1, 1].contour(mask, colors="red", linewidths=1.5)
axes[1, 1].set_title(f"Mid frame ({T//2})")

axes[1, 2].hist(pixel_std[mask], bins=50, alpha=0.7, label="in mask", color="red")
axes[1, 2].hist(pixel_std[~mask], bins=50, alpha=0.5, label="out mask", color="blue")
axes[1, 2].axvline(np.percentile(pixel_std[mask], 50), color="red", linestyle="--",
                    label=f"top-50% thr = {np.percentile(pixel_std[mask], 50):.1f}")
axes[1, 2].set_xlabel("pixel_std")
axes[1, 2].set_ylabel("Pixel count")
axes[1, 2].set_yscale("log")
axes[1, 2].set_title("Signal variation distribution")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle(f"{SAMPLE} — Mask v3.7 (rsm_bg_v3_port, level 0)", fontsize=13, fontweight="bold")
out = DEBUG / "mask_overlay.png"
plt.tight_layout()
plt.savefig(out, dpi=100, bbox_inches="tight")
plt.close()
print(f"Saved {out}")