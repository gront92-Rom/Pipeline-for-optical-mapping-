"""
StimExtractAgent — Extract stimulation channel from MiCAM ULTIMA .rsd files.

Stim signal is encoded in column 2 of the active sensor area (NOT padding cols).
Pattern: baseline ~6516 uint16, drops to ~4888 during stim pulse.
All rows drop simultaneously (uniform across spatial dimension).

Usage (CLI):
    python3 -m cardiac_pipeline.agents.stim_extract_agent <rsd_file> <output_dir>

Usage (import):
    from cardiac_pipeline.agents.stim_extract_agent import extract_stim_channel
    result = extract_stim_channel(rsd_path, fps=500.0)

Returns StimResult dataclass:
    - stim_trace: 1D float array (T,) — col 2 mean across rows
    - pulse_onsets: array of frame indices where stim pulse starts
    - pulse_offsets: array of frame indices where stim pulse ends
    - stim_hz: float — stimulation frequency (None if not detected)
    - is_paced: bool — True if stim pulses detected
    - bcl_ms: float — basic cycle length in ms (1000/stim_hz)
    - pulse_width_ms: float — mean pulse duration in ms
    - n_pulses: int
    - method: str — "col2_drop" or "none"
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────

STIM_COL = 2                  # column index in active area (0-99) containing stim
STIM_BASELINE_MIN = 6000       # baseline must be above this (uint16)
STIM_DROP_THRESHOLD = 500     # frames below baseline - this = stim pulse
STIM_MIN_PULSES = 2            # need at least 2 pulses to confirm stim
STIM_MAX_HZ = 50.0             # sanity: stim above 50 Hz is noise
STIM_MIN_HZ = 0.5              # sanity: stim below 0.5 Hz is likely artifact


# ── Dataclass ─────────────────────────────────────────────────────────────

@dataclass
class StimResult:
    stim_trace: np.ndarray        # (T,) float64 — col 2 mean across rows
    pulse_onsets: np.ndarray      # (N,) int — frame indices of pulse start
    pulse_offsets: np.ndarray      # (N,) int — frame indices of pulse end
    stim_hz: float | None          # stimulation frequency (Hz)
    is_paced: bool                 # True if stim detected
    bcl_ms: float | None           # basic cycle length (ms)
    pulse_width_ms: float | None   # mean pulse width (ms)
    n_pulses: int                  # number of pulses
    method: str                    # detection method used
    fps: float                     # sampling rate
    n_frames: int                 # total frames analyzed


# ── Core extraction ───────────────────────────────────────────────────────

def _read_rsd_col2(rsd_path: str, max_frames: int | None = None) -> tuple[np.ndarray, int]:
    """Read col 2 from .rsd file. Returns (col2_mean_trace, n_frames).
    
    .rsd layout: uint16, shape (T, 100, 128) — 100 rows × 128 cols.
    Col 2 is in the active sensor area (cols 0-99).
    """
    file_size = os.path.getsize(rsd_path)
    total_frames = file_size // (100 * 128 * 2)  # 100 rows × 128 cols × 2 bytes
    
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    
    # Read only col 2: stride through file
    # Each frame = 100*128 uint16 = 25600 bytes
    # Col 2 in each row: offset = row*128 + 2, but we want all rows → col 2
    # Faster: read full file and slice
    raw = np.fromfile(rsd_path, dtype=np.uint16, count=total_frames * 100 * 128)
    n_frames = raw.size // (100 * 128)
    video = raw[:n_frames * 100 * 128].reshape(n_frames, 100, 128)
    
    # Col 2 mean across rows (spatial average)
    col2_trace = video[:, :, STIM_COL].mean(axis=1).astype(np.float64)
    
    return col2_trace, n_frames


def _detect_pulses(trace: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Detect stim pulses in col 2 trace via threshold crossing.
    
    Pulse = contiguous frames where trace < baseline - STIM_DROP_THRESHOLD.
    Returns (onsets, offsets) as arrays of frame indices.
    """
    baseline = np.median(trace)
    
    # Sanity: baseline must be in expected range
    if baseline < STIM_BASELINE_MIN:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    threshold = baseline - STIM_DROP_THRESHOLD
    below = trace < threshold
    
    # Find contiguous below-threshold segments
    diff = np.diff(below.astype(int), prepend=0, append=0)
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    
    # Filter: pulse must be at least 1 frame
    valid = (offsets - onsets) >= 1
    onsets = onsets[valid]
    offsets = offsets[valid]
    
    return onsets, offsets


def _compute_stim_hz(onsets: np.ndarray, fps: float) -> float | None:
    """Compute stim frequency from inter-pulse intervals."""
    if len(onsets) < STIM_MIN_PULSES:
        return None
    
    intervals = np.diff(onsets)
    median_interval = np.median(intervals)
    
    if median_interval < 1:
        return None
    
    stim_hz = fps / median_interval
    
    # Sanity check
    if stim_hz < STIM_MIN_HZ or stim_hz > STIM_MAX_HZ:
        return None
    
    return float(stim_hz)


def extract_stim_channel(
    rsd_path: str,
    fps: float = 500.0,
    max_frames: int | None = None,
) -> StimResult:
    """Extract stimulation channel from MiCAM ULTIMA .rsd file.
    
    Args:
        rsd_path: Path to .rsd file
        fps: Sampling rate (Hz). Default 500 for MiCAM ULTIMA.
        max_frames: Limit frames to read (None = read all)
    
    Returns:
        StimResult with stim trace, pulse times, and frequency.
    """
    col2_trace, n_frames = _read_rsd_col2(rsd_path, max_frames)
    
    onsets, offsets = _detect_pulses(col2_trace, fps)
    n_pulses = len(onsets)
    
    is_paced = n_pulses >= STIM_MIN_PULSES
    stim_hz = _compute_stim_hz(onsets, fps) if is_paced else None
    bcl_ms = (1000.0 / stim_hz) if stim_hz is not None else None
    
    if n_pulses > 0:
        pulse_widths = offsets - onsets
        pulse_width_ms = float(np.mean(pulse_widths) / fps * 1000.0)
    else:
        pulse_width_ms = None
    
    return StimResult(
        stim_trace=col2_trace,
        pulse_onsets=onsets,
        pulse_offsets=offsets,
        stim_hz=stim_hz,
        is_paced=is_paced,
        bcl_ms=bcl_ms,
        pulse_width_ms=pulse_width_ms,
        n_pulses=n_pulses,
        method="col2_drop" if is_paced else "none",
        fps=fps,
        n_frames=n_frames,
    )


def extract_stim_from_chunks(
    rsd_paths: list[str],
    fps: float = 500.0,
    max_frames_per_chunk: int | None = None,
) -> StimResult:
    """Extract stim from multiple .rsd chunks (partitioned recording).
    
    Concatenates col 2 traces from all chunks, then detects pulses.
    """
    traces = []
    total_frames = 0
    
    for path in rsd_paths:
        trace, n = _read_rsd_col2(path, max_frames_per_chunk)
        traces.append(trace)
        total_frames += n
    
    full_trace = np.concatenate(traces)
    
    onsets, offsets = _detect_pulses(full_trace, fps)
    n_pulses = len(onsets)
    is_paced = n_pulses >= STIM_MIN_PULSES
    stim_hz = _compute_stim_hz(onsets, fps) if is_paced else None
    bcl_ms = (1000.0 / stim_hz) if stim_hz is not None else None
    
    if n_pulses > 0:
        pulse_widths = offsets - onsets
        pulse_width_ms = float(np.mean(pulse_widths) / fps * 1000.0)
    else:
        pulse_width_ms = None
    
    return StimResult(
        stim_trace=full_trace,
        pulse_onsets=onsets,
        pulse_offsets=offsets,
        stim_hz=stim_hz,
        is_paced=is_paced,
        bcl_ms=bcl_ms,
        pulse_width_ms=pulse_width_ms,
        n_pulses=n_pulses,
        method="col2_drop" if is_paced else "none",
        fps=fps,
        n_frames=total_frames,
    )


# ── Report ────────────────────────────────────────────────────────────────

def write_report(result: StimResult, output_dir: str, sample_name: str = "") -> dict:
    """Write JSON report + save stim trace as .npy."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save stim trace
    trace_path = os.path.join(output_dir, f"{sample_name}_stim_trace.npy" if sample_name else "stim_trace.npy")
    np.save(trace_path, result.stim_trace)
    
    # JSON report
    report = {
        "sample": sample_name,
        "method": result.method,
        "is_paced": result.is_paced,
        "stim_hz": result.stim_hz,
        "bcl_ms": result.bcl_ms,
        "pulse_width_ms": result.pulse_width_ms,
        "n_pulses": result.n_pulses,
        "n_frames": result.n_frames,
        "fps": result.fps,
        "pulse_onsets": result.pulse_onsets.tolist(),
        "pulse_offsets": result.pulse_offsets.tolist(),
    }
    
    json_path = os.path.join(output_dir, f"{sample_name}_stim_report.json" if sample_name else "stim_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract stim channel from MiCAM ULTIMA .rsd")
    parser.add_argument("rsd_path", help="Path to .rsd file (or comma-separated for chunks)")
    parser.add_argument("output_dir", help="Output directory for report + trace")
    parser.add_argument("--fps", type=float, default=500.0, help="Sampling rate (Hz)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to read")
    parser.add_argument("--sample-name", default="", help="Sample name for output files")
    args = parser.parse_args()
    
    # Handle comma-separated chunk paths
    if "," in args.rsd_path:
        paths = [p.strip() for p in args.rsd_path.split(",")]
        result = extract_stim_from_chunks(paths, fps=args.fps, max_frames_per_chunk=args.max_frames)
    else:
        result = extract_stim_channel(args.rsd_path, fps=args.fps, max_frames=args.max_frames)
    
    report = write_report(result, args.output_dir, sample_name=args.sample_name)
    
    # Console summary
    print(f"\n{'='*60}")
    print(f"StimExtractAgent Results")
    print(f"{'='*60}")
    print(f"  Method:     {result.method}")
    print(f"  Paced:      {result.is_paced}")
    print(f"  Stim Hz:    {result.stim_hz}")
    print(f"  BCL (ms):   {result.bcl_ms}")
    print(f"  Pulse wd:   {result.pulse_width_ms} ms")
    print(f"  N pulses:   {result.n_pulses}")
    print(f"  N frames:   {result.n_frames}")
    print(f"  FPS:        {result.fps}")
    if result.n_pulses > 0:
        print(f"  Onsets:     {result.pulse_onsets[:20].tolist()}{'...' if len(result.pulse_onsets) > 20 else ''}")
    print(f"{'='*60}")
    print(f"  Report:     {os.path.join(args.output_dir, args.sample_name + '_stim_report.json')}")
    print(f"  Trace:      {os.path.join(args.output_dir, args.sample_name + '_stim_trace.npy')}")


if __name__ == "__main__":
    main()