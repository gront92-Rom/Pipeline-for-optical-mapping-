#!/usr/bin/env python3
"""
sideline_agent.py — Standalone post-pacing analysis agent for optical mapping.

Pipeline:
  1. Load .rsh (or .gsd→.rsh fallback) via optimap → (T, 100, 128) video
  2. Stim extraction: col 2 mean → threshold → onsets/offsets/stim_hz
  3. Detect stim BURSTS — group pulses by inter-pulse gaps
  4. 3×3 ROI (tissue center or frame center) → invert → trim 30ms → ASLS → Butterworth 80Hz
  5. Peak detection on processed trace
  6. For EACH burst:
       - Find last stimulated AP peak in that burst
       - Open post-stim window = min(2 sec, next burst onset − last AP peak)
       - Classify post-stim peaks: amp < 0.5 × stim_amp → DAD, ≥ 0.5 → spontaneous
       - Latency = first spontaneous peak − last stim AP peak (ms)
  7. Table: one row per burst
  8. PNG: 3-panel overview + per-burst zoom
  9. Ask user: "Анализировать карты?"

Usage (CLI):
    PYTHONPATH=src python3 -m cardiac_pipeline.agents.sideline_agent <rsh_or_gsd_file> <output_dir> [--fps 500]
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sideline_agent")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Constants ──────────────────────────────────────────────────────────────

STIM_COL = 2
STIM_BASELINE_MIN = 6000
STIM_DROP_THRESHOLD = 500
STIM_MIN_PULSE_WIDTH = 3      # frames
STIM_MIN_PULSES = 2

STIM_MAX_HZ = 50.0
STIM_MIN_HZ = 0.5

CROP_LEFT = 20
CROP_RIGHT = 8
ASLS_LAM = 1e5
ASLS_P = 0.01
ASLS_NITER = 3
BUTTERWORTH_CUTOFF_HZ = 80.0
TRIM_MS = 30.0

POST_STIM_WINDOW_SEC = 2.0    # max window after last stimulated AP peak
DAD_AMP_FRACTION = 0.5        # < 0.5 × stim median amp → DAD
DEFAULT_FALLBACK_FREQ_HZ = 16.0  # fallback for peak min_dist if stim_hz missing

# Burst detection: gap > BURST_GAP_MULTIPLIER × median_interval → new burst
BURST_GAP_MULTIPLIER = 3.0


# ── Step 1: Load video ─────────────────────────────────────────────────────

def load_video(input_path: str) -> Tuple[np.ndarray, str]:
    """Load .rsh/.gsd via optimap. Returns (video float32, resolved_path_str)."""
    p = Path(input_path)
    ext = p.suffix.lower()

    # .gsd/.gsh → .rsh fallback
    load_path = p
    if ext in ('.gsd', '.gsh'):
        rsh = p.with_suffix('.rsh')
        if rsh.exists():
            logger.info(f"Резолв .{ext} → .rsh: {rsh.name}")
            load_path = rsh
        else:
            raise FileNotFoundError(f"optimap не поддерживает {ext}, .rsh не найден: {rsh}")

    try:
        import optimap as om
    except ImportError:
        raise ImportError("optimap не установлен. pip install opticalmapping")

    logger.info(f"Загрузка через optimap: {load_path.name}")
    video = om.load_video(str(load_path)).astype(np.float32)
    logger.info(f"Loaded: shape={video.shape}, dtype={video.dtype}")
    return video, str(load_path)


def crop_video(video: np.ndarray, crop_left: int = CROP_LEFT, crop_right: int = CROP_RIGHT) -> np.ndarray:
    """Crop columns: remove artifact padding (left=20, right=8 → 128→100)."""
    T, H, W = video.shape
    if crop_left + crop_right >= W:
        logger.warning(f"crop_left={crop_left} + crop_right={crop_right} >= W={W} — кроп пропущен")
        return video
    cropped = video[:, :, crop_left: W - crop_right] if crop_right > 0 else video[:, :, crop_left:]
    logger.info(f"Кроп: W {W} → {cropped.shape[2]} (left={crop_left}, right={crop_right})")
    return cropped


# ── Step 2: Stim extraction ───────────────────────────────────────────────

def extract_stim(video: np.ndarray, fps: float) -> Dict[str, Any]:
    """Extract stim channel from col 2 (before crop). Returns stim info dict."""
    T, H, W = video.shape
    if W < 3:
        logger.warning(f"W={W} < 3, col 2 недоступна")
        return {"stim_trace": None, "pulse_onsets": np.array([], dtype=int),
                "pulse_offsets": np.array([], dtype=int), "is_paced": False,
                "stim_hz": None, "bcl_ms": None, "pulse_width_ms": None,
                "n_pulses": 0, "method": "none"}

    col2_trace = video[:, :, STIM_COL].mean(axis=1).astype(np.float64)
    baseline = np.median(col2_trace)

    if baseline < STIM_BASELINE_MIN:
        logger.info(f"Stim baseline={baseline:.0f} < {STIM_BASELINE_MIN} — нет стимуляции")
        return {"stim_trace": col2_trace, "pulse_onsets": np.array([], dtype=int),
                "pulse_offsets": np.array([], dtype=int), "is_paced": False,
                "stim_hz": None, "bcl_ms": None, "pulse_width_ms": None,
                "n_pulses": 0, "method": "none"}

    threshold = baseline - STIM_DROP_THRESHOLD
    below = col2_trace < threshold
    diff = np.diff(below.astype(int), prepend=0, append=0)
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]

    # Filter: pulse width ≥ 3 frames
    valid = (offsets - onsets) >= STIM_MIN_PULSE_WIDTH
    onsets = onsets[valid]
    offsets = offsets[valid]

    n_pulses = len(onsets)
    is_paced = n_pulses >= STIM_MIN_PULSES

    stim_hz = None
    bcl_ms = None
    if is_paced:
        intervals = np.diff(onsets)
        median_interval = np.median(intervals)
        if median_interval >= 1:
            hz = fps / median_interval
            if STIM_MIN_HZ <= hz <= STIM_MAX_HZ:
                stim_hz = float(hz)
                bcl_ms = float(1000.0 / stim_hz)

    pulse_width_ms = None
    if n_pulses > 0:
        pulse_width_ms = float(np.mean(offsets - onsets) / fps * 1000.0)

    logger.info(
        f"Stim: {n_pulses} pulses, is_paced={is_paced}, "
        f"stim_hz={stim_hz}, bcl_ms={bcl_ms}, pulse_width={pulse_width_ms:.1f}ms"
        if pulse_width_ms else
        f"Stim: {n_pulses} pulses, is_paced={is_paced}"
    )

    return {
        "stim_trace": col2_trace, "pulse_onsets": onsets, "pulse_offsets": offsets,
        "is_paced": is_paced, "stim_hz": stim_hz, "bcl_ms": bcl_ms,
        "pulse_width_ms": pulse_width_ms, "n_pulses": n_pulses,
        "method": "col2_drop" if is_paced else "none",
    }


# ── Step 3: Detect stim bursts ────────────────────────────────────────────

def detect_bursts(stim_info: Dict[str, Any], fps: float) -> List[Dict[str, Any]]:
    """
    Group individual stim pulses into bursts.

    A burst = consecutive pulses where inter-pulse gap ≤ BURST_GAP_MULTIPLIER × median_interval.
    A gap > 3× median_interval → new burst.

    Returns list of burst dicts:
      {burst_id, pulse_onsets, pulse_offsets, n_pulses, start_frame, end_frame}
    """
    onsets = stim_info.get("pulse_onsets", np.array([], dtype=int))
    offsets = stim_info.get("pulse_offsets", np.array([], dtype=int))

    if len(onsets) < 2:
        # Single pulse or no pulses → single burst if 1 pulse
        if len(onsets) == 1:
            return [{"burst_id": 0, "pulse_onsets": onsets, "pulse_offsets": offsets,
                      "n_pulses": 1, "start_frame": int(onsets[0]),
                      "end_frame": int(offsets[0])}]
        return []

    intervals = np.diff(onsets)
    median_interval = float(np.median(intervals))
    gap_threshold = BURST_GAP_MULTIPLIER * median_interval

    # Find burst boundaries: where interval > gap_threshold
    burst_starts = [0]
    for i, interval in enumerate(intervals):
        if interval > gap_threshold:
            burst_starts.append(i + 1)

    # Build burst dicts
    bursts = []
    for bid, start_idx in enumerate(burst_starts):
        end_idx = burst_starts[bid + 1] if bid + 1 < len(burst_starts) else len(onsets)
        b_onsets = onsets[start_idx:end_idx]
        b_offsets = offsets[start_idx:end_idx]
        bursts.append({
            "burst_id": bid,
            "pulse_onsets": b_onsets,
            "pulse_offsets": b_offsets,
            "n_pulses": len(b_onsets),
            "start_frame": int(b_onsets[0]),
            "end_frame": int(b_offsets[-1]),
        })

    logger.info(
        f"Bursts: {len(bursts)} detected "
        f"(gap_threshold={gap_threshold:.0f} frames = {gap_threshold/fps*1000:.0f}ms, "
        f"median_interval={median_interval:.0f} frames = {median_interval/fps*1000:.0f}ms)"
    )
    for b in bursts:
        logger.info(
            f"  Burst {b['burst_id']}: {b['n_pulses']} pulses, "
            f"frames [{b['start_frame']}, {b['end_frame']}] "
            f"({b['start_frame']/fps*1000:.0f}–{b['end_frame']/fps*1000:.0f} ms)"
        )

    return bursts


# ── Step 4: Tissue mask + 3×3 ROI ─────────────────────────────────────────

def find_tissue_center(video: np.ndarray) -> Tuple[int, int]:
    """Find tissue center via temporal variance. Fallback: frame center."""
    T, H, W = video.shape
    if T < 10:
        return H // 2, W // 2

    var_map = np.var(video[:min(T, 2000)], axis=0)
    threshold = np.percentile(var_map, 80)
    mask = var_map > threshold
    if mask.sum() < 10:
        return H // 2, W // 2

    ys, xs = np.where(mask)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    logger.info(f"Tissue center: ({cy}, {cx}), mask pixels={mask.sum()}")
    return cy, cx


def process_3x3_trace(
    video: np.ndarray,
    center: Tuple[int, int],
    fps: float,
    dye: str = "A",
) -> np.ndarray:
    """
    3×3 ROI → invert (dye A) → trim 30ms → ASLS(λ=1e5) → Butterworth 80Hz.
    Returns processed 1D trace.
    """
    T, H, W = video.shape
    cy, cx = center
    y0, y1 = max(0, cy - 1), min(H, cy + 2)
    x0, x1 = max(0, cx - 1), min(W, cx + 2)

    # 1. 3×3 mean
    trace = np.mean(video[:, y0:y1, x0:x1], axis=(1, 2)).astype(np.float32)
    logger.info(f"3×3 ROI: rows [{y0}:{y1}], cols [{x0}:{x1}], T={len(trace)}")

    # 2. Invert for VSD (dye A)
    inverted = isinstance(dye, str) and dye.strip().upper().startswith("A")
    trace = -trace if inverted else trace.copy()

    # 3. Trim 30ms from each edge
    n_trim = max(int(fps * TRIM_MS / 1000.0), 5)
    if len(trace) > 2 * n_trim + 10:
        trace = trace[n_trim:-n_trim]
        logger.info(f"Trimmed {n_trim} frames ({TRIM_MS}ms) from each edge → {len(trace)} frames")

    # 4. ASLS baseline correction
    try:
        from cardiac_pipeline.utils.preprocess import asls_baseline_correct_trace
        trace_bc = asls_baseline_correct_trace(
            trace, lam=ASLS_LAM, p=ASLS_P, niter=ASLS_NITER
        ).astype(np.float32)
        logger.info(f"ASLS: lam={ASLS_LAM}, p={ASLS_P}, niter={ASLS_NITER}")
    except Exception as e:
        logger.warning(f"ASLS failed ({e}), using raw trace")
        trace_bc = trace

    # 5. Butterworth LPF 80 Hz
    if fps and fps > 0:
        try:
            from scipy.signal import filtfilt, butter
            wn = min(BUTTERWORTH_CUTOFF_HZ / (fps / 2.0), 0.99)
            b, a = butter(4, wn, btype='low')
            trace_filt = filtfilt(b, a, trace_bc).astype(np.float32)
            logger.info(f"Butterworth LPF: cutoff={BUTTERWORTH_CUTOFF_HZ}Hz, wn={wn:.4f}")
        except Exception as e:
            logger.warning(f"Butterworth failed ({e}), using ASLS-only")
            trace_filt = trace_bc
    else:
        trace_filt = trace_bc

    return trace_filt


# ── Step 5: Peak detection ────────────────────────────────────────────────

def detect_peaks(
    trace: np.ndarray,
    fps: float,
    stim_hz: Optional[float] = None,
) -> np.ndarray:
    """
    scipy.find_peaks with min_dist based on stim_hz.
    Fallback: 16 Hz → min_dist = 0.6 × fps / 16.
    """
    from scipy.signal import find_peaks

    freq = stim_hz if stim_hz and stim_hz > 0 else DEFAULT_FALLBACK_FREQ_HZ
    min_dist = int(0.6 * fps / freq)
    min_dist = max(min_dist, 3)

    amp_range = np.ptp(trace)
    prominence = max(amp_range * 0.05, 1e-6)

    peaks, props = find_peaks(trace, distance=min_dist, prominence=prominence)
    logger.info(
        f"Peaks: N={len(peaks)}, min_dist={min_dist} frames "
        f"({min_dist/fps*1000:.1f}ms), prominence={prominence:.2f}, "
        f"freq={freq:.1f}Hz"
    )
    return peaks


# ── Step 6: Multi-burst align + classify ──────────────────────────────────

def align_and_classify_multi(
    peaks: np.ndarray,
    trace: np.ndarray,
    bursts: List[Dict[str, Any]],
    fps: float,
    dye: str = "A",
) -> List[Dict[str, Any]]:
    """
    For each burst:
      - Find stimulated AP peaks within burst [start, end]
      - last_stim_ap_peak = last peak in burst
      - Window = [last_stim_ap_peak, min(last_stim_ap_peak + 2sec, next_burst_start)]
      - Classify post-stim peaks as DAD or spontaneous
      - Latency = first spontaneous peak − last_stim_ap_peak

    Returns list of per-burst result dicts.
    """
    if len(peaks) == 0:
        return []

    results = []
    n_bursts = len(bursts)

    for bid, burst in enumerate(bursts):
        burst_start = burst["start_frame"]
        burst_end = burst["end_frame"]

        # Next burst start (for window truncation)
        next_burst_start = bursts[bid + 1]["start_frame"] if bid + 1 < n_bursts else None

        # Stimulated peaks within this burst (with 50ms tolerance after burst end)
        tolerance = int(fps * 0.050)
        stim_mask = (peaks >= burst_start) & (peaks <= burst_end + tolerance)
        burst_stim_peaks = peaks[stim_mask]

        if len(burst_stim_peaks) == 0:
            logger.warning(f"Burst {bid}: no stimulated AP peaks found")
            results.append({
                "burst_id": bid,
                "burst_start_frame": burst_start,
                "burst_end_frame": burst_end,
                "stim_peaks": np.array([], dtype=int),
                "last_stim_ap_peak": None,
                "post_stim_peaks": np.array([], dtype=int),
                "spont_peaks": np.array([], dtype=int),
                "dad_peaks": np.array([], dtype=int),
                "stim_median_amp": None,
                "dad_threshold": None,
                "latency_ms": None,
                "n_stim_peaks": 0, "n_spont": 0, "n_dad": 0,
                "n_post_stim_total": 0,
                "window_start_frame": None,
                "window_end_frame": None,
                "window_ms": None,
                "spont_amps": [], "dad_amps": [],
            })
            continue

        last_stim_ap_peak = int(burst_stim_peaks[-1])

        # Window: [last_stim_ap_peak, min(last_stim_ap_peak + 2sec, next_burst_start)]
        window_frames_max = int(POST_STIM_WINDOW_SEC * fps)
        window_end_ideal = last_stim_ap_peak + window_frames_max

        if next_burst_start is not None and next_burst_start < window_end_ideal:
            # Truncate window at next burst start
            window_end = next_burst_start
            window_ms = (window_end - last_stim_ap_peak) / fps * 1000.0
            logger.info(
                f"Burst {bid}: window truncated to {window_ms:.0f}ms "
                f"(next burst at frame {next_burst_start})"
            )
        else:
            window_end = window_end_ideal
            window_ms = POST_STIM_WINDOW_SEC * 1000.0

        # Stim median amplitude (from this burst's stimulated peaks)
        stim_amps = trace[burst_stim_peaks]
        stim_median_amp = float(np.median(stim_amps))
        dad_threshold = DAD_AMP_FRACTION * stim_median_amp

        # Post-stim peaks in window
        post_mask = (peaks > last_stim_ap_peak) & (peaks <= window_end)
        post_stim_peaks = peaks[post_mask]

        # Classify
        if len(post_stim_peaks) > 0:
            post_amps = trace[post_stim_peaks]
            is_spont = post_amps >= dad_threshold
            is_dad = ~is_spont
            spont_peaks = post_stim_peaks[is_spont]
            dad_peaks = post_stim_peaks[is_dad]
        else:
            spont_peaks = np.array([], dtype=int)
            dad_peaks = np.array([], dtype=int)

        # Latency
        latency_ms = None
        if len(spont_peaks) > 0:
            latency_frames = int(spont_peaks[0]) - last_stim_ap_peak
            latency_ms = float(latency_frames / fps * 1000.0)

        logger.info(
            f"Burst {bid}: stim_peaks={len(burst_stim_peaks)}, "
            f"last_AP_peak={last_stim_ap_peak} ({last_stim_ap_peak/fps*1000:.0f}ms), "
            f"window={window_ms:.0f}ms, "
            f"post_stim={len(post_stim_peaks)} → spont={len(spont_peaks)}, DAD={len(dad_peaks)}, "
            f"latency={'%.1fms' % latency_ms if latency_ms else 'N/A'}"
        )

        results.append({
            "burst_id": bid,
            "burst_start_frame": burst_start,
            "burst_end_frame": burst_end,
            "stim_peaks": burst_stim_peaks,
            "last_stim_ap_peak": last_stim_ap_peak,
            "post_stim_peaks": post_stim_peaks,
            "spont_peaks": spont_peaks,
            "dad_peaks": dad_peaks,
            "stim_median_amp": stim_median_amp,
            "dad_threshold": float(dad_threshold),
            "latency_ms": latency_ms,
            "n_stim_peaks": len(burst_stim_peaks),
            "n_spont": len(spont_peaks),
            "n_dad": len(dad_peaks),
            "n_post_stim_total": len(post_stim_peaks),
            "window_start_frame": last_stim_ap_peak,
            "window_end_frame": window_end,
            "window_ms": round(window_ms, 1),
            "spont_amps": trace[spont_peaks].tolist() if len(spont_peaks) > 0 else [],
            "dad_amps": trace[dad_peaks].tolist() if len(dad_peaks) > 0 else [],
        })

    return results


# ── Step 7: Save table + report ───────────────────────────────────────────

def save_results(
    sample_id: str,
    stim_info: Dict[str, Any],
    bursts: List[Dict[str, Any]],
    burst_results: List[Dict[str, Any]],
    fps: float,
    dye: str,
    output_dir: str,
) -> Tuple[str, str]:
    """Save CSV table (one row per burst) + JSON report."""
    os.makedirs(output_dir, exist_ok=True)

    # CSV: one row per burst
    csv_path = os.path.join(output_dir, "sideline_table.csv")
    rows = []
    for br in burst_results:
        last_peak_ms = None
        if br["last_stim_ap_peak"] is not None:
            last_peak_ms = round(br["last_stim_ap_peak"] / fps * 1000.0, 2)

        burst_start_ms = round(br["burst_start_frame"] / fps * 1000.0, 2)
        burst_end_ms = round(br["burst_end_frame"] / fps * 1000.0, 2)

        rows.append({
            "sample_id": sample_id,
            "burst_id": br["burst_id"],
            "dye": dye,
            "fps": fps,
            "burst_start_ms": burst_start_ms,
            "burst_end_ms": burst_end_ms,
            "n_stim_pulses": bursts[br["burst_id"]]["n_pulses"],
            "stim_hz": stim_info.get("stim_hz"),
            "n_stim_peaks": br["n_stim_peaks"],
            "last_stim_peak_ms": last_peak_ms,
            "window_ms": br["window_ms"],
            "n_post_stim_total": br["n_post_stim_total"],
            "n_spont": br["n_spont"],
            "n_dad": br["n_dad"],
            "latency_ms": round(br["latency_ms"], 2) if br["latency_ms"] else None,
            "stim_median_amp": round(br["stim_median_amp"], 4) if br["stim_median_amp"] else None,
            "dad_threshold": round(br["dad_threshold"], 4) if br["dad_threshold"] else None,
            "spont_amps": json.dumps([round(a, 2) for a in br.get("spont_amps", [])]),
            "dad_amps": json.dumps([round(a, 2) for a in br.get("dad_amps", [])]),
        })

    # Write CSV (overwrite, not append — one file per sample)
    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logger.info(f"Saved: {csv_path} ({len(rows)} rows)")

    # JSON report
    json_path = os.path.join(output_dir, "sideline_report.json")
    report = {
        "sample_id": sample_id,
        "dye": dye,
        "fps": fps,
        "stim": {
            "is_paced": stim_info["is_paced"],
            "stim_hz": stim_info.get("stim_hz"),
            "bcl_ms": stim_info.get("bcl_ms"),
            "pulse_width_ms": stim_info.get("pulse_width_ms"),
            "n_pulses": stim_info["n_pulses"],
            "n_bursts": len(bursts),
            "method": stim_info.get("method"),
        },
        "bursts": [],
        "parameters": {
            "post_stim_window_sec": POST_STIM_WINDOW_SEC,
            "dad_amp_fraction": DAD_AMP_FRACTION,
            "burst_gap_multiplier": BURST_GAP_MULTIPLIER,
            "asls_lam": ASLS_LAM,
            "butterworth_cutoff_hz": BUTTERWORTH_CUTOFF_HZ,
            "fallback_freq_hz": DEFAULT_FALLBACK_FREQ_HZ,
        },
    }
    for br in burst_results:
        last_peak_ms = None
        if br["last_stim_ap_peak"] is not None:
            last_peak_ms = round(br["last_stim_ap_peak"] / fps * 1000.0, 2)
        report["bursts"].append({
            "burst_id": br["burst_id"],
            "n_stim_pulses": bursts[br["burst_id"]]["n_pulses"],
            "n_stim_peaks": br["n_stim_peaks"],
            "last_stim_peak_ms": last_peak_ms,
            "window_ms": br["window_ms"],
            "n_spont": br["n_spont"],
            "n_dad": br["n_dad"],
            "latency_ms": round(br["latency_ms"], 2) if br["latency_ms"] else None,
            "stim_median_amp": br["stim_median_amp"],
            "dad_threshold": br["dad_threshold"],
            "spont_amps": br.get("spont_amps", []),
            "dad_amps": br.get("dad_amps", []),
        })

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return csv_path, json_path


# ── Step 8: PNG ───────────────────────────────────────────────────────────

def save_png(
    stim_info: Dict[str, Any],
    bursts: List[Dict[str, Any]],
    trace: np.ndarray,
    peaks: np.ndarray,
    burst_results: List[Dict[str, Any]],
    fps: float,
    sample_id: str,
    dye: str,
    output_dir: str,
) -> str:
    """
    Multi-panel PNG:
      Panel 1: Stim channel (col 2) with burst shading
      Panel 2: Full processed trace + all classified peaks
      Panel 3..N: Per-burst post-stim zoom (one per burst, up to 4)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    stim_trace = stim_info.get("stim_trace")
    pulse_onsets = stim_info.get("pulse_onsets", np.array([], dtype=int))
    pulse_offsets = stim_info.get("pulse_offsets", np.array([], dtype=int))

    t_ms = np.arange(len(trace), dtype=np.float64) / fps * 1000.0

    # Count total panels: 1 (stim) + 1 (overview) + min(n_bursts, 4) zoom
    n_zoom = min(len(burst_results), 4)
    n_panels = 2 + n_zoom  # stim + overview + zoom panels

    fig, axes = plt.subplots(n_panels, 1, figsize=(22, 3.5 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]

    # Color legend
    legend_elements = [
        Patch(facecolor="dodgerblue", label="Stimulated"),
        Patch(facecolor="orange", label="DAD"),
        Patch(facecolor="red", label="Spontaneous"),
        Patch(facecolor="yellow", alpha=0.3, label="Post-stim window"),
    ]

    # ── Panel 1: Stim channel ──
    ax0 = axes[0]
    if stim_trace is not None and len(stim_trace) > 0:
        t_stim_ms = np.arange(len(stim_trace), dtype=np.float64) / fps * 1000.0
        ax0.plot(t_stim_ms, stim_trace, lw=0.5, color="gray", label="col 2")
        baseline = np.median(stim_trace)
        ax0.axhline(baseline, color="green", ls="--", lw=0.6, alpha=0.5, label=f"baseline={baseline:.0f}")

        # Shade each burst
        burst_colors = ["red", "purple", "brown", "teal"]
        for bid, burst in enumerate(bursts):
            color = burst_colors[bid % len(burst_colors)]
            for on, off in zip(burst["pulse_onsets"], burst["pulse_offsets"]):
                ax0.axvspan(t_stim_ms[on] if on < len(t_stim_ms) else on,
                            t_stim_ms[off] if off < len(t_stim_ms) else off,
                            color=color, alpha=0.2)
            ax0.axvline(t_stim_ms[burst["start_frame"]] if burst["start_frame"] < len(t_stim_ms) else burst["start_frame"],
                        color=color, ls="--", lw=0.8, alpha=0.6, label=f"Burst {bid}")

    ax0.set_ylabel("col 2 mean")
    ax0.set_title(f"Stim channel — {stim_info['n_pulses']} pulses, {len(bursts)} bursts")
    ax0.legend(loc="upper right", fontsize=7)

    # ── Panel 2: Full trace + all peaks ──
    ax1 = axes[1]
    ax1.plot(t_ms, trace, lw=0.5, color="royalblue", label="processed trace")

    # Shade burst regions
    for bid, burst in enumerate(bursts):
        color = burst_colors[bid % len(burst_colors)]
        for on, off in zip(burst["pulse_onsets"], burst["pulse_offsets"]):
            t_on = on / fps * 1000.0
            t_off = off / fps * 1000.0
            ax1.axvspan(t_on, t_off, color=color, alpha=0.08)

    all_stim = []
    all_dad = []
    all_spont = []
    for br in burst_results:
        all_stim.extend(br["stim_peaks"].tolist())
        all_dad.extend(br["dad_peaks"].tolist())
        all_spont.extend(br["spont_peaks"].tolist())

    if all_stim:
        idx = np.array(all_stim, dtype=int)
        ax1.scatter(t_ms[idx], trace[idx], c="dodgerblue", s=20, zorder=5, label=f"stimulated ({len(idx)})")
    if all_dad:
        idx = np.array(all_dad, dtype=int)
        ax1.scatter(t_ms[idx], trace[idx], c="orange", s=25, zorder=5, marker="v", label=f"DAD ({len(idx)})")
    if all_spont:
        idx = np.array(all_spont, dtype=int)
        ax1.scatter(t_ms[idx], trace[idx], c="red", s=30, zorder=5, marker="^", label=f"spontaneous ({len(idx)})")

    # Post-stim windows
    for br in burst_results:
        if br["last_stim_ap_peak"] is not None and br["window_end_frame"] is not None:
            ws = br["last_stim_ap_peak"]
            we = min(br["window_end_frame"], len(t_ms) - 1)
            ax1.axvspan(t_ms[ws], t_ms[we], color="yellow", alpha=0.1)
            ax1.axvline(t_ms[ws], color="green", ls="--", lw=0.8, alpha=0.5)

    ax1.set_ylabel("ΔF/F (a.u.)")
    ax1.set_xlabel("time [ms]")
    ax1.set_title("Full trace — all bursts + post-stim windows")
    ax1.legend(loc="upper right", fontsize=7)

    # ── Panels 3..N: Per-burst zoom ──
    for zoom_idx, br in enumerate(burst_results[:n_zoom]):
        ax = axes[2 + zoom_idx]
        bid = br["burst_id"]
        color = burst_colors[bid % len(burst_colors)]

        if br["last_stim_ap_peak"] is None:
            ax.set_title(f"Burst {bid} — no stimulated peaks")
            ax.text(0.5, 0.5, "No stimulated AP peaks found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
            continue

        last_peak = br["last_stim_ap_peak"]
        we = br["window_end_frame"]

        # Zoom range: 200ms before last peak → window end + 200ms
        zoom_start = max(0, last_peak - int(0.2 * fps))
        zoom_end = min(len(trace) - 1, we + int(0.2 * fps))
        zoom_mask = np.arange(len(trace)) >= zoom_start
        zoom_mask &= np.arange(len(trace)) <= zoom_end

        ax.plot(t_ms[zoom_mask], trace[zoom_mask], lw=1.0, color="royalblue")

        # Shade stim pulses in zoom
        for on, off in zip(bursts[bid]["pulse_onsets"], bursts[bid]["pulse_offsets"]):
            if zoom_start <= on <= zoom_end:
                ax.axvspan(t_ms[on], t_ms[min(off, len(t_ms)-1)], color=color, alpha=0.15)

        # Post-stim window shading
        ax.axvspan(t_ms[last_peak], t_ms[min(we, len(t_ms)-1)], color="yellow", alpha=0.15)
        ax.axvline(t_ms[last_peak], color="green", ls="--", lw=1.0, alpha=0.7, label="last stim AP peak")

        # DAD threshold
        if br["dad_threshold"] is not None:
            ax.axhline(br["dad_threshold"], color="orange", ls=":", lw=0.8, alpha=0.5, label=f"DAD threshold (0.5×stim)")

        # Peaks
        for pk in br["stim_peaks"]:
            if zoom_start <= pk <= zoom_end:
                ax.scatter(t_ms[pk], trace[pk], c="dodgerblue", s=40, zorder=5)
        for pk in br["dad_peaks"]:
            if zoom_start <= pk <= zoom_end:
                ax.scatter(t_ms[pk], trace[pk], c="orange", s=50, zorder=5, marker="v")
        for pk in br["spont_peaks"]:
            if zoom_start <= pk <= zoom_end:
                ax.scatter(t_ms[pk], trace[pk], c="red", s=60, zorder=5, marker="^")

        # Latency annotation
        if br["latency_ms"] is not None and len(br["spont_peaks"]) > 0:
            first_spont = br["spont_peaks"][0]
            if zoom_start <= first_spont <= zoom_end:
                y_range = np.ptp(trace[zoom_mask]) if zoom_mask.sum() > 0 else 1.0
                ax.annotate(
                    f"latency={br['latency_ms']:.1f}ms",
                    xy=(t_ms[first_spont], trace[first_spont]),
                    xytext=(t_ms[first_spont] + 50, trace[first_spont] + 0.1 * y_range),
                    fontsize=8, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
                )

        title = (
            f"Burst {bid}: {br['n_stim_peaks']} stim peaks, "
            f"window={br['window_ms']:.0f}ms, "
            f"spont={br['n_spont']}, DAD={br['n_dad']}, "
            f"latency={'%.1fms' % br['latency_ms'] if br['latency_ms'] else 'N/A'}"
        )
        ax.set_title(title)
        ax.set_xlabel("time [ms]")
        ax.set_ylabel("ΔF/F")
        ax.legend(loc="upper right", fontsize=7)

    # Supertitle
    total_spont = sum(br["n_spont"] for br in burst_results)
    total_dad = sum(br["n_dad"] for br in burst_results)
    fig.suptitle(
        f"{sample_id} | dye={dye} | fps={fps} | bursts={len(bursts)} | "
        f"total spont={total_spont} | total DAD={total_dad}",
        fontsize=11, fontweight="bold"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = os.path.join(output_dir, "sideline_post_pacing.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {png_path}")
    return png_path


# ── Main pipeline ─────────────────────────────────────────────────────────

def run(
    input_path: str,
    output_dir: str,
    fps: float = 500.0,
    dye: str = "A",
    sample_id: str = "",
) -> Dict[str, Any]:
    """Run full sideline post-pacing analysis pipeline (multi-burst)."""

    if not sample_id:
        sample_id = Path(input_path).stem
        parts = sample_id.split("-")
        if parts:
            sample_id = parts[-1]

    logger.info(f"=== SidelineAgent v2 (multi-burst): {sample_id} ===")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"fps={fps}, dye={dye}")

    # Step 1: Load video
    video, resolved_path = load_video(input_path)

    # Step 2: Stim extraction (before crop — col 2 is in padding)
    stim_info = extract_stim(video, fps)

    # Step 2b: Crop (after stim extraction)
    video = crop_video(video)

    # Step 3: Detect bursts
    bursts = detect_bursts(stim_info, fps)

    if not bursts:
        logger.warning("No stim bursts detected — nothing to analyze")
        # Still save trace + empty results
        os.makedirs(output_dir, exist_ok=True)
        return {"sample_id": sample_id, "n_bursts": 0, "png_path": None}

    # Step 4: Tissue center + 3×3 ROI → process trace (on cropped video)
    center = find_tissue_center(video)
    trace = process_3x3_trace(video, center, fps, dye=dye)

    # Step 5: Peak detection
    stim_hz = stim_info.get("stim_hz")
    peaks = detect_peaks(trace, fps, stim_hz=stim_hz)

    # Step 6: Multi-burst align + classify
    burst_results = align_and_classify_multi(peaks, trace, bursts, fps, dye=dye)

    # Step 7: Save table + report
    csv_path, json_path = save_results(
        sample_id, stim_info, bursts, burst_results, fps, dye, output_dir
    )

    # Step 8: PNG
    png_path = save_png(
        stim_info, bursts, trace, peaks, burst_results, fps, sample_id, dye, output_dir
    )

    # Summary
    total_spont = sum(br["n_spont"] for br in burst_results)
    total_dad = sum(br["n_dad"] for br in burst_results)

    logger.info(f"=== DONE: {sample_id} ===")
    logger.info(f"  Bursts: {len(bursts)}")
    for br in burst_results:
        logger.info(
            f"  Burst {br['burst_id']}: stim_peaks={br['n_stim_peaks']}, "
            f"window={br['window_ms']:.0f}ms, "
            f"spont={br['n_spont']}, DAD={br['n_dad']}, "
            f"latency={'%.1fms' % br['latency_ms'] if br['latency_ms'] else 'N/A'}"
        )
    logger.info(f"  Total: spont={total_spont}, DAD={total_dad}")

    return {
        "sample_id": sample_id,
        "n_bursts": len(bursts),
        "burst_results": burst_results,
        "total_spont": total_spont,
        "total_dad": total_dad,
        "csv_path": csv_path,
        "json_path": json_path,
        "png_path": png_path,
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SidelineAgent v2 — post-pacing analysis (multi-burst: stim → 3×3 → peaks → DAD/spont)"
    )
    parser.add_argument("input", help="Path to .rsh or .gsd file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--fps", type=float, default=500.0, help="Sampling rate (Hz)")
    parser.add_argument("--dye", default="A", help="Dye type: A=VSD (voltage), B=CaT (calcium)")
    parser.add_argument("--sample-id", default="", help="Override sample ID")
    args = parser.parse_args()

    result = run(
        input_path=args.input,
        output_dir=args.output_dir,
        fps=args.fps,
        dye=args.dye,
        sample_id=args.sample_id,
    )

    print(f"\n{'='*60}")
    print(f"SidelineAgent v2 — Post-Pacing Analysis (Multi-Burst)")
    print(f"{'='*60}")
    print(f"  Sample:    {result['sample_id']}")
    print(f"  Bursts:    {result['n_bursts']}")
    print(f"  Total spont: {result['total_spont']}")
    print(f"  Total DAD:   {result['total_dad']}")
    for br in result.get("burst_results", []):
        lat = f"{br['latency_ms']:.1f}ms" if br['latency_ms'] else "N/A"
        print(f"  Burst {br['burst_id']}: stim={br['n_stim_peaks']}, "
              f"window={br['window_ms']:.0f}ms, "
              f"spont={br['n_spont']}, DAD={br['n_dad']}, latency={lat}")
    print(f"  PNG:  {result['png_path']}")
    print(f"  CSV:  {result['csv_path']}")
    print(f"  JSON: {result['json_path']}")
    print(f"{'='*60}")
    print(f"\nАнализировать карты?")


if __name__ == "__main__":
    main()