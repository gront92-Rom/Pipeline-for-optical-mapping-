#!/usr/bin/env python3
"""
sideline_agent.py — Post-analysis agent for long optical recordings (sideline mode).

Consumes output from LoaderAgent._handle_sideline():
  - must/sideline_trace.npz   (raw, oriented, bc, filtered, peaks, fps)
  - must/sideline_metrics.json
  - must/sideline_segments.json
  - must/sideline_decision_request.json

Does NOT require raw_video.npy (which is intentionally NOT saved in sideline mode).

Pipeline (consumes loader output, adds analysis):
  1. Load sideline_trace.npz → filtered trace, peaks, fps
  2. Load metadata.json → dye, stim_hz, sample_id
  3. Run stim_extract_agent on .rsd if available → is_paced, stim_hz_measured
  4. Recompute RR stats (if new peaks from stim-aligned detection)
  5. Classify paced vs spontaneous based on stim channel
  6. Save consolidated sideline_report.json with all metrics
  7. Generate enhanced PNG (trace + peaks + stim pulses overlay + regularity)

Usage (CLI):
    python3 -m cardiac_pipeline.agents.sideline_agent <sample_id> --results-root <dir>
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig

logger = logging.getLogger(__name__)


class SidelineAgent(BaseAgent):
    """
    Post-analysis agent for sideline-mode recordings.

    Reads loader output (sideline_trace.npz, metadata.json) and adds:
      - Stim channel extraction (paced/spontaneous classification)
      - Consolidated report
      - Enhanced PNG with stim overlay
    """

    DEPENDS_ON: list = []  # [LoaderAgent] — runs after loader sideline mode
    REQUIRED_INPUTS: list = ["sideline_trace.npz", "metadata.json"]

    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None):
        super().__init__(sample_id, config)
        self.frame_limit = 4096

    # ── Loaders ───────────────────────────────────────────────────────────

    def _load_sideline_npz(self) -> dict:
        """Load sideline_trace.npz from must/ directory."""
        path = self.get_path("sideline_trace.npz", kind="must")
        if not path.exists():
            raise FileNotFoundError(f"sidelive_trace.npz not found at {path}")
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    def _load_metadata(self) -> Dict[str, Any]:
        return self.load_must("metadata.json")

    # ── Stim extraction ──────────────────────────────────────────────────

    def _try_stim_extraction(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to extract stim channel from .rsd files.

        Uses stim_extract_agent if .rsd paths are available in metadata.
        Returns stim info dict (may be empty if no .rsd or no stim found).
        """
        stim_info = {
            "stim_detected": False,
            "is_paced": False,
            "stim_hz_measured": None,
            "bcl_ms": None,
            "pulse_width_ms": None,
            "n_pulses": 0,
            "pulse_onsets": [],
            "method": "none",
        }

        # Find .rsd files from metadata companion_files
        companion_files = metadata.get("companion_files", {})
        rsd_paths = companion_files.get(".rsd", [])

        if isinstance(rsd_paths, str):
            rsd_paths = [rsd_paths]
        if not rsd_paths:
            logger.info("No .rsd files in metadata — stim extraction skipped")
            return stim_info

        # Filter to existing files
        existing = [p for p in rsd_paths if Path(p).exists()]
        if not existing:
            logger.warning(f".rsd paths in metadata but files not found: {rsd_paths}")
            return stim_info

        try:
            from cardiac_pipeline.agents.stim_extract_agent import (
                extract_stim_channel,
                extract_stim_from_chunks,
            )

            fps = metadata.get("fps", 500.0)

            if len(existing) == 1:
                result = extract_stim_channel(existing[0], fps=fps)
            else:
                result = extract_stim_from_chunks(existing, fps=fps)

            stim_info.update({
                "stim_detected": result.is_paced,
                "is_paced": result.is_paced,
                "stim_hz_measured": result.stim_hz,
                "bcl_ms": result.bcl_ms,
                "pulse_width_ms": result.pulse_width_ms,
                "n_pulses": result.n_pulses,
                "pulse_onsets": result.pulse_onsets.tolist(),
                "method": result.method,
            })

            if result.is_paced:
                logger.info(
                    f"Stim detected: {result.stim_hz:.2f} Hz, "
                    f"BCL={result.bcl_ms:.1f} ms, "
                    f"{result.n_pulses} pulses, "
                    f"width={result.pulse_width_ms:.1f} ms"
                )
            else:
                logger.info(f"No stim detected in {len(existing)} .rsd file(s) — likely spontaneous")

        except Exception as e:
            logger.warning(f"Stim extraction failed: {e}")

        return stim_info

    # ── RR recompute ─────────────────────────────────────────────────────

    def _compute_rr_stats(self, peaks: np.ndarray, fps: float) -> Dict[str, Any]:
        """Recompute RR statistics from peak indices."""
        if len(peaks) < 2:
            return {
                "n_peaks": len(peaks),
                "is_regular": False,
                "reason": "too few peaks (< 2)",
            }

        rr = np.diff(peaks) / fps * 1000.0  # ms
        rr_mean = float(np.mean(rr))
        rr_std = float(np.std(rr))
        rr_cv = rr_std / rr_mean if rr_mean > 0 else float("inf")
        rr_median = float(np.median(rr))

        regular_mask = np.abs(rr - rr_median) <= 0.15 * rr_median
        regularity_score = float(np.mean(regular_mask))
        is_regular = regularity_score >= 0.8 and rr_cv < 0.2

        return {
            "n_peaks": len(peaks),
            "rr_mean_ms": round(rr_mean, 2),
            "rr_std_ms": round(rr_std, 2),
            "rr_cv": round(rr_cv, 4),
            "rr_median_ms": round(rr_median, 2),
            "regularity_score": round(regularity_score, 4),
            "is_regular": bool(is_regular),
            "rr_intervals_ms": [round(r, 2) for r in rr.tolist()],
        }

    # ── Enhanced PNG ─────────────────────────────────────────────────────

    def _save_enhanced_png(
        self,
        filtered_trace: np.ndarray,
        peaks: np.ndarray,
        fps: float,
        dye: str,
        rr_stats: Dict[str, Any],
        stim_info: Dict[str, Any],
        stim_trace: Optional[np.ndarray] = None,
    ) -> Path:
        """Generate enhanced PNG: trace + peaks + stim overlay + regularity shading."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t_ms = np.arange(len(filtered_trace), dtype=np.float64) / fps * 1000.0

        n_panels = 3 if (stim_trace is not None and len(stim_trace) > 0) else 2
        fig, axes = plt.subplots(n_panels, 1, figsize=(20, 4 * n_panels), sharex=True)

        # Title
        reg_str = "REGULAR" if rr_stats.get("is_regular", False) else "IRREGULAR"
        fig.suptitle(
            f"Sideline — {self.sample_id} | dye={dye} | fps={fps} | "
            f"N={rr_stats.get('n_peaks', 0)} peaks | "
            f"RR={rr_stats.get('rr_mean_ms', 'N/A')}±{rr_stats.get('rr_std_ms', 'N/A')}ms | "
            f"CV={rr_stats.get('rr_cv', 'N/A')} | {reg_str}",
            fontsize=12, fontweight="bold",
        )

        # Panel 1: filtered trace + peaks
        axes[0].plot(t_ms, filtered_trace, lw=0.5, color="C2", label="filtered trace")
        if len(peaks) > 0:
            axes[0].scatter(
                t_ms[peaks], filtered_trace[peaks],
                c="red", s=20, zorder=5,
                label=f"peaks (N={len(peaks)})",
            )
        axes[0].set_ylabel("amplitude")
        axes[0].legend(loc="upper right")

        # Panel 2: RR intervals
        if len(peaks) >= 2:
            rr = np.diff(peaks) / fps * 1000.0
            rr_t = t_ms[peaks[:-1]] + np.diff(t_ms[peaks]) / 2
            rr_median = float(np.median(rr))
            axes[1].plot(rr_t, rr, "o-", ms=3, lw=0.5, color="C0")
            axes[1].axhline(rr_median, color="green", ls="--", lw=0.8, label=f"median={rr_median:.1f}ms")
            axes[1].axhline(rr_median * 0.85, color="orange", ls=":", lw=0.5, label="±15%")
            axes[1].axhline(rr_median * 1.15, color="orange", ls=":", lw=0.5)
            axes[1].set_ylabel("RR [ms]")
            axes[1].legend(loc="upper right")
        else:
            axes[1].text(0.5, 0.5, "<2 peaks", transform=axes[1].transAxes, ha="center")
            axes[1].set_ylabel("RR [ms]")

        # Panel 3: stim trace (if available)
        if n_panels == 3:
            axes[2].plot(t_ms[:len(stim_trace)], stim_trace, lw=0.5, color="C3")
            # Mark pulse onsets
            onsets = stim_info.get("pulse_onsets", [])
            for onset in onsets:
                axes[2].axvline(onset / fps * 1000.0, color="red", alpha=0.3, lw=0.5)
            if stim_info.get("is_paced"):
                axes[2].set_ylabel(f"stim col2\n{stim_info['stim_hz_measured']:.1f} Hz")
            else:
                axes[2].set_ylabel("stim col2\n(no stim)")
            axes[2].set_xlabel("time [ms]")
        else:
            axes[1].set_xlabel("time [ms]")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        png_path = self.get_path("sideline_enhanced.png", kind="must")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return png_path

    # ── Main run ─────────────────────────────────────────────────────────

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """Run sideline post-analysis on loader output.

        Returns consolidated sideline report dict.
        """
        self.logger.info(f"SidelineAgent: starting post-analysis for {self.sample_id}")

        # 1. Load loader output
        npz = self._load_sideline_npz()
        metadata = self._load_metadata()

        filtered_trace = npz.get("filtered", npz.get("filtered_trace"))
        peaks = npz.get("peaks", np.array([])).astype(int)
        fps = float(npz.get("fps", metadata.get("fps", 500.0)))
        dye = metadata.get("dye", "?")

        self.logger.info(
            f"Loaded: filtered_trace={filtered_trace.shape}, peaks={len(peaks)}, fps={fps}"
        )

        # 2. Stim extraction (from .rsd if available)
        stim_info = self._try_stim_extraction(metadata)

        # 3. Recompute RR stats
        rr_stats = self._compute_rr_stats(peaks, fps)

        # 4. Classification: paced vs spontaneous
        is_paced = stim_info.get("is_paced", False)
        stim_hz_meta = metadata.get("stim_hz") or metadata.get("stim_hz_effective")
        stim_hz_measured = stim_info.get("stim_hz_measured")

        if is_paced:
            rhythm_class = "paced"
            stim_hz_final = stim_hz_measured
        elif stim_hz_meta and stim_hz_meta > 1.0:
            rhythm_class = "paced (metadata)"
            stim_hz_final = stim_hz_meta
        else:
            rhythm_class = "spontaneous"
            stim_hz_final = None

        # 5. Consolidated report
        report = {
            "sample_id": self.sample_id,
            "agent": "SidelineAgent",
            "dye": dye,
            "fps": fps,
            "n_frames": len(filtered_trace),
            "n_peaks": int(len(peaks)),
            "rhythm_class": rhythm_class,
            "is_paced": is_paced,
            "stim_hz_metadata": stim_hz_meta,
            "stim_hz_measured": stim_hz_measured,
            "stim_hz_final": stim_hz_final,
            "bcl_ms": stim_info.get("bcl_ms"),
            "pulse_width_ms": stim_info.get("pulse_width_ms"),
            "n_stim_pulses": stim_info.get("n_pulses", 0),
            "stim_method": stim_info.get("method", "none"),
            "rr_stats": rr_stats,
            "asls_lam": npz.get("asls_lam", 1e5),
            "status": "sideline_complete",
        }

        # 6. Save report
        self.save_must(report, "sideline_report.json")
        self.logger.info(f"[MUST] Saved: sideline_report.json")

        # 7. Enhanced PNG
        stim_trace = None
        if stim_info.get("is_paced"):
            # Reload stim trace for overlay
            try:
                from cardiac_pipeline.agents.stim_extract_agent import _read_rsd_col2
                companion = metadata.get("companion_files", {}).get(".rsd", [])
                if isinstance(companion, str):
                    companion = [companion]
                if companion and Path(companion[0]).exists():
                    stim_trace, _ = _read_rsd_col2(companion[0])
            except Exception:
                pass

        png_path = self._save_enhanced_png(
            filtered_trace, peaks, fps, dye, rr_stats, stim_info, stim_trace
        )
        self.logger.info(f"[MUST] Saved: {png_path.name}")

        # 8. Summary log
        self.logger.info(
            f"SidelineAgent complete: {self.sample_id} | "
            f"rhythm={rhythm_class} | "
            f"N={len(peaks)} peaks | "
            f"RR={rr_stats.get('rr_mean_ms', 'N/A')}ms | "
            f"regular={rr_stats.get('is_regular', False)}"
        )

        return report


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SidelineAgent — post-analysis for long recordings")
    parser.add_argument("sample_id", help="Sample ID (e.g. 030A)")
    parser.add_argument("--results-root", default="results", help="Results root directory")
    parser.add_argument("--force", action="store_true", help="Force re-run")
    args = parser.parse_args()

    cfg = PipelineConfig()
    cfg.results_root = Path(args.results_root)

    agent = SidelineAgent(args.sample_id, config=cfg)
    result = agent.run(force=args.force)

    print(f"\n{'='*60}")
    print(f"SidelineAgent Result")
    print(f"{'='*60}")
    print(f"  Sample:        {result['sample_id']}")
    print(f"  Rhythm:        {result['rhythm_class']}")
    print(f"  Stim Hz:       {result['stim_hz_final']}")
    print(f"  N peaks:       {result['n_peaks']}")
    print(f"  RR mean (ms):  {result['rr_stats'].get('rr_mean_ms', 'N/A')}")
    print(f"  RR CV:         {result['rr_stats'].get('rr_cv', 'N/A')}")
    print(f"  Regular:       {result['rr_stats'].get('is_regular', False)}")
    print(f"  Status:        {result['status']}")
    print(f"{'='*60}")