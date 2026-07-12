#!/usr/bin/env python3
"""
post_pacing_agent.py — Detect spontaneous activity after pacing cessation.

Simple flow:
  1. Extract stim channel from .rsd (col 2) → all stim pulses
  2. Find the LAST stim pulse onset
  3. Window = [last_onset, last_onset + 2*fps] (2 seconds)
  4. Extract optical trace (3×3 ROI, col 0-99) in that window
  5. Invert (VSD) / no invert (CaT), ASLS, Butterworth 80Hz
  6. Peak detection → any peaks = spontaneous beats
  7. Report: post_pacing_report.json + PNG

Usage (CLI):
    python3 -m cardiac_pipeline.agents.post_pacing_agent <rsd_path> <output_dir> [--fps 500] [--dye A|B]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import find_peaks, filtfilt, butter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


# ── Constants ──
STIM_COL = 2
STIM_BASELINE_MIN = 6000
STIM_DROP_THRESHOLD = 500
POST_PACING_WINDOW_S = 2.0   # 2 seconds after last stim
ROI_SIZE = 3                   # 3×3 central ROI
ASLS_LAM = 1e5
ASLS_P = 0.01
ASLS_NITER = 3
LP_CUTOFF_HZ = 80.0
PEAK_MIN_DIST_FACTOR = 0.6
PEAK_PROM_FACTOR_IQR = 0.3
PEAK_PROM_FACTOR_STD = 0.2
FALLBACK_STIM_HZ = 16.0


def _read_rsd_col(rsd_path: str, col: int, max_frames: int | None = None) -> tuple[np.ndarray, int]:
    """Read single column from .rsd. Returns (trace, n_frames)."""
    file_size = os.path.getsize(rsd_path)
    total_frames = file_size // (100 * 128 * 2)
    if max_frames:
        total_frames = min(total_frames, max_frames)
    raw = np.fromfile(rsd_path, dtype=np.uint16, count=total_frames * 100 * 128)
    n_frames = raw.size // (100 * 128)
    video = raw[:n_frames * 100 * 128].reshape(n_frames, 100, 128)
    trace = video[:, :, col].mean(axis=1).astype(np.float64)
    return trace, n_frames


def _read_rsd_roi(rsd_path: str, col_start: int, col_end: int,
                  row_start: int, row_end: int,
                  frame_start: int = 0, frame_end: int | None = None) -> np.ndarray:
    """Read ROI from .rsd. Returns (T,) mean trace over the ROI."""
    file_size = os.path.getsize(rsd_path)
    total_frames = file_size // (100 * 128 * 2)
    if frame_end is None:
        frame_end = total_frames
    frame_end = min(frame_end, total_frames)
    n_frames = frame_end - frame_start

    # Read the needed frames
    offset_bytes = frame_start * 100 * 128 * 2
    raw = np.fromfile(rsd_path, dtype=np.uint16, count=n_frames * 100 * 128, offset=offset_bytes)
    n_read = raw.size // (100 * 128)
    video = raw[:n_read * 100 * 128].reshape(n_read, 100, 128)

    # ROI mean
    roi = video[:, row_start:row_end, col_start:col_end].mean(axis=(1, 2)).astype(np.float64)
    return roi


def _detect_stim_pulses(trace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect stim pulses in col 2 trace. Returns (onsets, offsets)."""
    baseline = np.median(trace)
    if baseline < STIM_BASELINE_MIN:
        return np.array([], dtype=int), np.array([], dtype=int)
    threshold = baseline - STIM_DROP_THRESHOLD
    below = trace < threshold
    diff = np.diff(below.astype(int), prepend=0, append=0)
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    valid = (offsets - onsets) >= 1
    return onsets[valid], offsets[valid]


def _asls_baseline(y: np.ndarray, lam: float = ASLS_LAM, p: float = ASLS_P,
                    niter: int = ASLS_NITER) -> np.ndarray:
    """ASLS baseline correction. Returns baseline."""
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L)).tocsr()
    P = (D.T @ D) * lam
    w = np.ones(L)
    baseline = np.zeros(L)
    for _ in range(niter):
        W = sp.diags(w, 0, shape=(L, L)).tocsc()
        Z = W + P
        baseline = spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y <= baseline)
    return baseline


def _butterworth_lp(trace: np.ndarray, fps: float, cutoff: float = LP_CUTOFF_HZ) -> np.ndarray:
    """Butterworth 4-pole zero-phase lowpass."""
    wn = min(cutoff / (fps / 2.0), 0.99)
    b, a = butter(4, wn, btype='low')
    return filtfilt(b, a, trace).astype(np.float32)


def analyze_post_pacing(
    rsd_path: str,
    fps: float = 500.0,
    dye: str = "A",
    window_s: float = POST_PACING_WINDOW_S,
    output_dir: str | None = None,
    sample_name: str = "",
) -> Dict[str, Any]:
    """
    Detect spontaneous beats in a 2-second window after last stim pulse.

    Args:
        rsd_path: Path to .rsd file
        fps: Sampling rate
        dye: "A" (VSD, invert) or "B" (CaT, no invert)
        window_s: Window size in seconds after last stim (default 2.0)
        output_dir: If given, write report + PNG here
        sample_name: Sample name for output files

    Returns:
        Dict with stim info, post-pacing window info, spontaneous beat detection
    """
    result = {
        "sample": sample_name,
        "rsd_path": rsd_path,
        "fps": fps,
        "dye": dye,
        "window_s": window_s,
    }

    # ── 1. Extract stim channel ──
    stim_trace, n_total = _read_rsd_col(rsd_path, STIM_COL)
    onsets, offsets = _detect_stim_pulses(stim_trace)
    n_stim = len(onsets)

    result["n_total_frames"] = n_total
    result["n_stim_pulses"] = n_stim

    if n_stim == 0:
        result["status"] = "no_stim"
        result["has_spontaneous"] = None
        result["reason"] = "No stim pulses detected — cannot define post-pacing window"
        if output_dir:
            _write_output(result, None, None, None, None, output_dir, sample_name)
        return result

    # ── 2. Find LAST stim pulse ──
    last_onset = int(onsets[-1])
    last_offset = int(offsets[-1])
    result["last_stim_onset"] = last_onset
    result["last_stim_offset"] = last_offset
    result["last_stim_time_ms"] = round(last_onset / fps * 1000.0, 2)

    # Stim frequency from all pulses
    if n_stim >= 2:
        intervals = np.diff(onsets)
        median_interval = float(np.median(intervals))
        stim_hz = fps / median_interval if median_interval > 0 else None
        result["stim_hz"] = round(stim_hz, 2) if stim_hz else None
        result["bcl_ms"] = round(1000.0 / stim_hz, 1) if stim_hz else None

    # ── 3. Define post-pacing window ──
    window_frames = int(window_s * fps)
    win_start = last_offset  # start right after last pulse ends
    win_end = min(win_start + window_frames, n_total)
    actual_window_frames = win_end - win_start

    result["window_start_frame"] = win_start
    result["window_end_frame"] = win_end
    result["window_start_ms"] = round(win_start / fps * 1000.0, 2)
    result["window_end_ms"] = round(win_end / fps * 1000.0, 2)
    result["window_actual_s"] = round(actual_window_frames / fps, 3)

    if actual_window_frames < fps * 0.5:
        result["status"] = "window_too_short"
        result["has_spontaneous"] = None
        result["reason"] = f"Only {actual_window_frames} frames ({actual_window_frames/fps:.2f}s) after last stim"
        if output_dir:
            _write_output(result, None, None, None, None, output_dir, sample_name)
        return result

    # ── 4. Extract optical trace in window (3×3 ROI) ──
    H, W = 100, 128  # MiCAM ULTIMA
    cy, cx = H // 2, 50 // 2  # center of active area (cols 0-99)
    # Actually active area is 100x100 (cols 0-99), center = (50, 50)
    cy, cx = 50, 50
    y0, y1 = max(0, cy - 1), min(H, cy + 2)
    x0, x1 = max(0, cx - 1), min(100, cx + 2)  # cols 0-99 = active

    optical_trace = _read_rsd_roi(rsd_path, x0, x1, y0, y1, win_start, win_end)
    result["roi"] = f"{y0}:{y1}, {x0}:{x1}"

    # ── 5. Preprocess: invert → ASLS → Butterworth ──
    # Invert for VSD (dye A): AP peaks point up
    inverted = dye.upper().startswith("A")
    if inverted:
        trace = -optical_trace
    else:
        trace = optical_trace.copy()

    # ASLS baseline correction
    if len(trace) > 10:
        baseline = _asls_baseline(trace, lam=ASLS_LAM, p=ASLS_P, niter=ASLS_NITER)
        trace_bc = trace - baseline
    else:
        trace_bc = trace

    # Butterworth 80 Hz
    if fps > 0 and len(trace_bc) > 10:
        trace_filt = _butterworth_lp(trace_bc, fps)
    else:
        trace_filt = trace_bc

    # ── 6. Peak detection ──
    # Min distance: 0.6 * fps / 16 Hz (fallback stim_hz)
    min_dist = max(int(PEAK_MIN_DIST_FACTOR * fps / FALLBACK_STIM_HZ), 1)
    iqr = np.percentile(trace_filt, 75) - np.percentile(trace_filt, 25)
    std = np.std(trace_filt)
    prominence = max(iqr * PEAK_PROM_FACTOR_IQR, std * PEAK_PROM_FACTOR_STD)

    peaks, props = find_peaks(trace_filt, distance=min_dist, prominence=prominence)

    n_spont = len(peaks)
    has_spont = n_spont > 0

    result["n_spontaneous_beats"] = n_spont
    result["has_spontaneous"] = has_spont
    result["spont_peak_frames"] = peaks.tolist()  # relative to window start
    result["spont_peak_times_ms"] = [round((win_start + p) / fps * 1000.0, 2) for p in peaks]
    result["peak_prominence_threshold"] = round(prominence, 4)
    result["min_distance_frames"] = min_dist

    if n_spont > 0:
        # Intervals between spontaneous beats
        if n_spont >= 2:
            rr = np.diff(peaks) / fps * 1000.0
            result["spont_rr_mean_ms"] = round(float(np.mean(rr)), 2)
            result["spont_rr_std_ms"] = round(float(np.std(rr)), 2)
            result["spont_rr_cv"] = round(float(np.std(rr) / np.mean(rr)), 4) if np.mean(rr) > 0 else None
            result["spont_rate_hz"] = round(1000.0 / float(np.mean(rr)), 2) if np.mean(rr) > 0 else None
        result["status"] = "spontaneous_detected"
    else:
        result["status"] = "quiescent"

    # ── 7. Output ──
    if output_dir:
        _write_output(result, optical_trace, trace_filt, peaks, stim_trace, output_dir, sample_name, win_start, fps)

    return result


def _write_output(
    result: dict,
    optical_raw: np.ndarray | None,
    optical_filt: np.ndarray | None,
    peaks: np.ndarray | None,
    stim_trace: np.ndarray | None,
    output_dir: str,
    sample_name: str,
    win_start: int = 0,
    fps: float = 500.0,
):
    """Write JSON report + PNG."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, f"{sample_name}_post_pacing.json" if sample_name else "post_pacing.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # PNG (only if we have trace data)
    if optical_filt is not None and len(optical_filt) > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=False)

        # Panel 1: stim trace (full, with last pulse marked)
        if stim_trace is not None:
            t_stim = np.arange(len(stim_trace), dtype=np.float64) / fps * 1000.0
            axes[0].plot(t_stim, stim_trace, lw=0.3, color="C3")
            # Mark all stim onsets
            if result.get("n_stim_pulses", 0) > 0:
                last_onset = result.get("last_stim_onset", 0)
                axes[0].axvline(last_onset / fps * 1000.0, color="red", lw=1.5, ls="--",
                               label=f"last stim @ {last_onset/fps*1000:.0f}ms")
                # Mark window
                win_start_ms = result.get("window_start_ms", 0)
                win_end_ms = result.get("window_end_ms", 0)
                axes[0].axvspan(win_start_ms, win_end_ms, color="green", alpha=0.15,
                              label=f"post-pacing window ({result.get('window_s', 2)}s)")
            axes[0].set_ylabel("stim col 2")
            axes[0].set_title(f"Stim channel — {sample_name} | {result.get('n_stim_pulses', 0)} pulses")
            axes[0].legend(fontsize=8)

        # Panel 2: optical trace in window + peaks
        if optical_filt is not None:
            t_ms = np.arange(len(optical_filt), dtype=np.float64) / fps * 1000.0
            axes[1].plot(t_ms, optical_filt, lw=0.7, color="C2", label="optical (filtered)")
            if peaks is not None and len(peaks) > 0:
                axes[1].scatter(t_ms[peaks], optical_filt[peaks], c="red", s=30, zorder=5,
                               label=f"spontaneous beats (N={len(peaks)})")
            status = result.get("status", "?")
            color = "green" if "spontaneous" in status else "red" if "quiescent" in status else "orange"
            axes[1].set_title(
                f"Post-pacing window — {sample_name} | "
                f"N={len(peaks) if peaks is not None else 0} beats | {status}",
                color=color, fontweight="bold",
            )
            axes[1].set_xlabel("time in window [ms]")
            axes[1].set_ylabel("amplitude")
            axes[1].legend(fontsize=8)

        fig.tight_layout()
        png_path = os.path.join(output_dir, f"{sample_name}_post_pacing.png" if sample_name else "post_pacing.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        result["_png_path"] = png_path

    result["_json_path"] = json_path


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect spontaneous activity after pacing cessation"
    )
    parser.add_argument("rsd_path", help="Path to .rsd file (or comma-separated for chunks)")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--fps", type=float, default=500.0, help="Sampling rate (Hz)")
    parser.add_argument("--dye", default="A", choices=["A", "B"], help="Dye type: A=VSD (invert), B=CaT")
    parser.add_argument("--window", type=float, default=2.0, help="Post-pacing window in seconds")
    parser.add_argument("--sample-name", default="", help="Sample name for output files")
    args = parser.parse_args()

    # Each .rsd chunk analyzed independently — stim may stop mid-chunk.
    # For multi-chunk, comma-separated: analyze each, report all.
    paths = [p.strip() for p in args.rsd_path.split(",")] if "," in args.rsd_path else [args.rsd_path]

    results = []
    for i, rsd in enumerate(paths):
        name = args.sample_name
        if len(paths) > 1:
            name = f"{name}_chunk{i}" if name else f"chunk{i}"
        print(f"\n--- Analyzing: {rsd} ---")
        r = analyze_post_pacing(rsd, fps=args.fps, dye=args.dye,
                                window_s=args.window, output_dir=args.output_dir,
                                sample_name=name)
        results.append(r)

        # Per-chunk console summary
        print(f"  Stim pulses:   {r.get('n_stim_pulses', 0)}")
        print(f"  Stim Hz:       {r.get('stim_hz', 'N/A')}")
        print(f"  Last stim:     {r.get('last_stim_time_ms', 'N/A')} ms (frame {r.get('last_stim_onset', 'N/A')})")
        print(f"  Window:        {r.get('window_start_ms', 'N/A')}–{r.get('window_end_ms', 'N/A')} ms "
              f"({r.get('window_actual_s', 'N/A')}s)")
        print(f"  Spont beats:   {r.get('n_spontaneous_beats', 0)}")
        print(f"  Has spont:     {r.get('has_spontaneous', False)}")
        if r.get('spont_rate_hz'):
            print(f"  Spont rate:    {r['spont_rate_hz']} Hz")
            print(f"  Spont RR:      {r.get('spont_rr_mean_ms', 'N/A')}±{r.get('spont_rr_std_ms', 'N/A')} ms")
        print(f"  Status:        {r.get('status', '?')}")

    # Summary across chunks
    any_spont = any(r.get('has_spontaneous') for r in results)
    any_stim = any(r.get('n_stim_pulses', 0) > 0 for r in results)
    print(f"\n{'='*60}")
    print(f"Summary ({len(results)} chunk(s))")
    print(f"{'='*60}")
    print(f"  Any stim found:     {any_stim}")
    print(f"  Any spontaneous:    {any_spont}")
    for i, r in enumerate(results):
        tag = "SPONT" if r.get('has_spontaneous') else ("QUIESC" if r.get('status')=='quiescent' else r.get('status','?'))
        print(f"  chunk{i}: {r.get('n_stim_pulses',0)} stim → {r.get('n_spontaneous_beats',0)} spont [{tag}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()