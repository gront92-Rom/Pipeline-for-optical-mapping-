#!/usr/bin/env python3
"""
ConductionAgent — Conduction Velocity Analysis Agent (Stage: CV)

Рассчитывает карту скорости проведения (CV) по per-beat картам активации.

Стратегия: «Всегда считать оба метода»
  Для каждого бита:
    1. compute_gradient_angular() — primary (gradient + angular histogram)
    2. compute_structure_tensor()  — fallback (structure tensor eigenvectors)
    3. select_best_cv_result()     — выбирает лучший по valid_fraction / n_valid
    4. Результат победителя идёт в агрегацию

Inputs (lazy — запускает ActivationAgent если нужно):
  - must/per_beat_activation.npy   (from ActivationAgent, shape: [N, H, W], мс)
  - must/mask.npy                  (from MaskAgent, bool)
  - must/metadata.json             (from LoaderAgent, содержит pixel_size_mm)

Outputs:
  MUST:
    - cv_mean.npy           — средняя карта CV (м/с)
    - cv_sd.npy             — SD CV по битам (м/с)
    - cv_angles.npy         — медианная карта направлений (рад)
    - cv_vx.npy, cv_vy.npy  — векторное поле (направление, нормированное)
    - cvl_map.npy           — карта продольной CV (м/с) [from ST when available]
    - cvt_map.npy           — карта трансверсальной CV (м/с) [from ST when available]
    - anisotropy_map.npy    — карта анизотропии [from ST when available]
    - fiber_angle_map.npy   — карта направления волокон (рад)
    - coherence_map.npy     — карта когерентности
    - conduction_report.json — вердикт, метрики, метод distribution
  DEBUG:
    - cv_per_beat.npy       — CV для каждого бита [N, H, W]
    - cvl_per_beat.npy      — CVL для каждого бита
    - cvt_per_beat.npy      — CVT для каждого бита
    - anisotropy_per_beat.npy
    - cv_vs_angle.npy       — angular distribution CV
    - conduction_debug.json — детали QC + оба метода по каждому биту
"""

import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig
from cardiac_pipeline.utils.cv_estimators import (
    compute_gradient_angular,
    compute_structure_tensor,
    select_best_cv_result,
    estimate_cv_stats,
)


# ---------------------------------------------------------------------------
# QC / Judge
# ---------------------------------------------------------------------------

def judge_conduction(
    cv_mean: np.ndarray,
    mask: np.ndarray,
    cv_min: float,
    cv_max: float,
    qc_threshold: float = 0.15,
) -> Tuple[str, str, Dict[str, Any]]:
    """Оценивает качество карты CV. Возвращает (verdict, reason, metrics)."""
    if cv_mean is None or not mask.any():
        return "REJECT", "empty cv_mean or empty mask", {}

    stats = estimate_cv_stats(cv_mean, mask)
    metrics: Dict[str, Any] = {**stats}
    metrics["cv_min_config"] = cv_min
    metrics["cv_max_config"] = cv_max
    metrics["qc_threshold"] = qc_threshold

    valid_fraction = stats["valid_fraction"]

    if stats["valid_pixels"] == 0:
        return "REJECT", "all CV pixels are NaN", metrics

    if valid_fraction < qc_threshold:
        return "REJECT", f"valid_fraction={valid_fraction:.3f} < qc_threshold={qc_threshold:.2f}", metrics

    if valid_fraction < 0.5:
        return "WARN", f"valid_fraction={valid_fraction:.3f} < 0.5 (low coverage)", metrics

    return "PASS", "OK", metrics


def classify_phenotype(cvl: float, anisotropy: float) -> str:
    """Классификация фенотипа проведения."""
    if np.isnan(cvl):
        return "unknown"
    if cvl < 0.2:
        return "slowed"
    if np.isfinite(anisotropy):
        if anisotropy > 4.0:
            return "fibrotic"
        if anisotropy < 1.4:
            return "disorganized"
    return "normal"


# ---------------------------------------------------------------------------
# ConductionAgent
# ---------------------------------------------------------------------------

class ConductionAgent(BaseAgent):
    """
    ConductionAgent — всегда считает оба метода, выбирает лучший per-beat.
    """

    DEPENDS_ON: list = []
    REQUIRED_INPUTS: list = ["per_beat_activation.npy", "mask.npy"]

    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None):
        super().__init__(sample_id, config)

        cv_cfg = getattr(self.config, "conduction", {}) or {}

        self.cv_min: float = float(cv_cfg.get("cv_min_m_per_s", 0.05))
        self.cv_max: float = float(cv_cfg.get("cv_max_m_per_s", 2.0))
        self.qc_threshold: float = float(cv_cfg.get("qc_threshold", 0.15))

        # Gradient params
        self.grad_threshold: float = float(cv_cfg.get("grad_threshold", 0.5))
        self.smooth_sigma: float = float(cv_cfg.get("smooth_sigma", 2.0))
        self.n_bins: int = int(cv_cfg.get("n_angle_bins", 18))

        # ST params
        self.st_local_sigma: float = float(cv_cfg.get("st_local_sigma", 2.0))
        self.st_integration_sigma: float = float(cv_cfg.get("st_integration_sigma", 4.0))

        # Selection params
        self.min_valid: int = int(cv_cfg.get("min_valid", 50))
        self.valid_frac_margin: float = float(cv_cfg.get("valid_frac_margin", 0.08))
        self.n_valid_ratio: float = float(cv_cfg.get("n_valid_ratio", 1.20))

        self.metadata: Dict[str, Any] = {}

    # ==================== HELPERS ====================

    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = self.get_path("metadata.json", kind="must")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self.logger.warning("metadata.json not found")
        return self.metadata

    def _get_pixel_size_mm(self) -> float:
        px = self.metadata.get("pixel_size_mm")
        if px is None:
            raise ValueError("pixel_size_mm отсутствует в metadata.json")
        px = float(px)
        if px <= 0:
            raise ValueError(f"pixel_size_mm некорректен ({px})")
        return px

    # ==================== COMPUTATION ====================

    def _compute_beat_cv(
        self,
        beat_map: np.ndarray,
        mask: np.ndarray,
        pixel_size_mm: float,
    ) -> Dict:
        """
        CV для одного бита — всегда оба метода + select_best.

        Возвращает dict:
          primary, grad_res, st_res, method, selection_reason,
          grad_n_valid, st_n_valid, grad_valid_fraction, st_valid_fraction
        """
        # 1. Gradient angular (always)
        grad_res = compute_gradient_angular(
            beat_map, mask, pixel_size_mm,
            cv_min=self.cv_min, cv_max=self.cv_max,
            grad_threshold=self.grad_threshold,
            smooth_sigma=self.smooth_sigma,
            n_bins=self.n_bins,
        )

        # 2. Structure tensor (always)
        st_res = compute_structure_tensor(
            beat_map, mask, pixel_size_mm,
            cv_min=self.cv_min, cv_max=self.cv_max,
            local_sigma=self.st_local_sigma,
            integration_sigma=self.st_integration_sigma,
        )

        # 3. Select best
        selection = select_best_cv_result(
            grad_res, st_res, mask,
            min_valid=self.min_valid,
            valid_frac_margin=self.valid_frac_margin,
            n_valid_ratio=self.n_valid_ratio,
        )

        return {
            "primary": selection["result"],
            "grad_res": grad_res,
            "st_res": st_res,
            "method": selection["method"],
            "selection_reason": selection["selection_reason"],
            "grad_n_valid": selection["grad_n_valid"],
            "st_n_valid": selection["st_n_valid"],
            "grad_valid_fraction": selection["grad_valid_fraction"],
            "st_valid_fraction": selection["st_valid_fraction"],
        }

    def _compute_all_beats(
        self,
        per_beat_activation: np.ndarray,
        mask: np.ndarray,
        pixel_size_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, List[Dict]]:
        """
        Расчёт CV по всем битам. Всегда оба метода + select_best per beat.

        Возвращает:
          cv_mean, cv_sd, cv_angles, cv_vx, cv_vy,
          cvl_mean, cvt_mean, aniso_mean,
          fiber_angle_mean, coherence_mean,
          cv_vs_angle_mean,
          beat_stats
        """
        n_beats = per_beat_activation.shape[0]
        H, W = mask.shape

        # TAT info (for logging)
        all_act = per_beat_activation[:, mask] if mask.any() else np.array([[0]])
        tat = float(np.nanmax(all_act[np.isfinite(all_act)]) - np.nanmin(all_act[np.isfinite(all_act)])) if np.any(np.isfinite(all_act)) else 0.0
        self.logger.info(f"TAT={tat:.1f}ms — computing both methods for all {n_beats} beats")

        beats_cv: List[np.ndarray] = []
        beats_cvl: List[np.ndarray] = []
        beats_cvt: List[np.ndarray] = []
        beats_aniso: List[np.ndarray] = []
        beats_angle: List[np.ndarray] = []
        beats_coh: List[np.ndarray] = []
        beats_vx: List[np.ndarray] = []
        beats_vy: List[np.ndarray] = []
        cv_vs_angles: List[np.ndarray] = []
        beat_stats: List[Dict] = []
        methods_used: List[str] = []

        for i in range(n_beats):
            beat_map = per_beat_activation[i]

            if not np.any(np.isfinite(beat_map[mask])):
                self.logger.warning(f"Beat {i}: all-NaN activation, skipping")
                beat_stats.append({"beat": i, "skipped": True, "reason": "all-NaN"})
                continue

            try:
                beat_cv = self._compute_beat_cv(beat_map, mask, pixel_size_mm)
            except Exception as exc:
                self.logger.warning(f"Beat {i}: CV failed — {exc}")
                beat_stats.append({"beat": i, "skipped": True, "reason": str(exc)})
                continue

            primary = beat_cv["primary"]
            method = beat_cv["method"]
            reason = beat_cv["selection_reason"]
            methods_used.append(method)

            stats_b = estimate_cv_stats(primary["cv_map"], mask)

            beat_stats.append({
                "beat": i,
                "skipped": False,
                "method": method,
                "selection_reason": reason,
                **stats_b,
                "cvl_m_s": primary["cvl_m_s"],
                "cvt_m_s": primary["cvt_m_s"],
                "anisotropy_ratio": primary["anisotropy_ratio"],
                "fiber_angle_deg": primary["fiber_angle_deg"],
                "fiber_coherence": primary["fiber_coherence"],
                "n_sources": primary["n_sources"],
                "n_valid": primary["n_valid"],
                "grad_n_valid": beat_cv["grad_n_valid"],
                "st_n_valid": beat_cv["st_n_valid"],
                "grad_valid_fraction": beat_cv["grad_valid_fraction"],
                "st_valid_fraction": beat_cv["st_valid_fraction"],
            })

            self.logger.info(
                f"Beat {i}: method={method} ({reason}), "
                f"valid={stats_b['valid_fraction']:.3f}, "
                f"cv_median={stats_b['cv_median_m_per_s']}, "
                f"CVL={primary['cvl_m_s']:.3f}, CVT={primary['cvt_m_s']:.3f}, "
                f"aniso={primary['anisotropy_ratio']:.2f} | "
                f"grad_n={beat_cv['grad_n_valid']}, st_n={beat_cv['st_n_valid']}"
            )

            beats_cv.append(primary["cv_map"])
            beats_cvl.append(primary["cvl_map"])
            beats_cvt.append(primary["cvt_map"])
            beats_aniso.append(primary["anisotropy_map"])
            beats_angle.append(primary["fiber_angle_map"])
            beats_coh.append(primary["coherence_map"])
            beats_vx.append(primary["vx"])
            beats_vy.append(primary["vy"])
            if "cv_vs_angle" in primary and np.isfinite(primary["cv_vs_angle"]).any():
                cv_vs_angles.append(primary["cv_vs_angle"])

        if len(beats_cv) == 0:
            nan = np.full((H, W), np.nan)
            return (nan, nan.copy(), nan.copy(), nan.copy(), nan.copy(),
                    nan.copy(), nan.copy(), nan.copy(), nan.copy(), nan.copy(),
                    np.full(self.n_bins, np.nan), beat_stats)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cv_mean = np.nanmean(np.array(beats_cv), axis=0)
            cv_sd = np.nanstd(np.array(beats_cv), axis=0)
            cvl_mean = np.nanmean(np.array(beats_cvl), axis=0)
            cvt_mean = np.nanmean(np.array(beats_cvt), axis=0)
            aniso_mean = np.nanmean(np.array(beats_aniso), axis=0)
            angle_mean = np.nanmedian(np.array(beats_angle), axis=0)
            coh_mean = np.nanmedian(np.array(beats_coh), axis=0)
            vx_mean = np.nanmean(np.array(beats_vx), axis=0)
            vy_mean = np.nanmean(np.array(beats_vy), axis=0)
            cv_vs_angle_mean = np.nanmean(np.array(cv_vs_angles), axis=0) if cv_vs_angles else np.full(self.n_bins, np.nan)

        # Mask
        for arr in [cv_mean, cv_sd, cvl_mean, cvt_mean, aniso_mean,
                    angle_mean, coh_mean, vx_mean, vy_mean]:
            arr[~mask] = np.nan

        # Store methods for report
        self._methods_used = methods_used

        # Store per-beat arrays for DEBUG saving in run()
        self._beats_cv = beats_cv
        self._beats_cvl = beats_cvl
        self._beats_cvt = beats_cvt
        self._beats_aniso = beats_aniso

        return (cv_mean, cv_sd, angle_mean, vx_mean, vy_mean,
                cvl_mean, cvt_mean, aniso_mean, angle_mean, coh_mean,
                cv_vs_angle_mean, beat_stats)

    # ==================== RUN ====================

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        if not force and self.exists("cv_mean.npy"):
            self.logger.info("cv_mean.npy exists, skipping (use force=True)")
            return {"status": "skipped"}

        t0 = time.perf_counter()

        # Lazy dependencies
        from cardiac_pipeline.agents.activation_agent import ActivationAgent
        self.DEPENDS_ON = [ActivationAgent]
        self.ensure_dependencies(force=force)

        # Metadata
        self._load_metadata()
        pixel_size_mm = self._get_pixel_size_mm()

        # Load inputs
        mask = self.load_must("mask.npy").astype(bool)
        per_beat_activation = self.load_must("per_beat_activation.npy")

        if per_beat_activation.ndim == 2:
            per_beat_activation = per_beat_activation[np.newaxis, ...]

        n_beats, H, W = per_beat_activation.shape
        self.logger.info(f"Loaded: mask={mask.shape}, activation={per_beat_activation.shape}, px={pixel_size_mm}")

        if n_beats == 0:
            raise ValueError("per_beat_activation.npy: 0 beats")

        # Compute
        (cv_mean, cv_sd, cv_angles, cv_vx, cv_vy,
         cvl_mean, cvt_mean, aniso_mean, fiber_angle_mean, coh_mean,
         cv_vs_angle_mean, beat_stats) = self._compute_all_beats(
            per_beat_activation, mask, pixel_size_mm
        )

        # QC
        verdict, reason, metrics = judge_conduction(
            cv_mean, mask, cv_min=self.cv_min, cv_max=self.cv_max,
            qc_threshold=self.qc_threshold,
        )
        self.logger.info(f"QC: {verdict} — {reason}")

        if verdict == "REJECT":
            raise ValueError(f"ConductionAgent REJECT: {reason}. Sample {self.sample_id} needs manual review.")

        # Aggregate scalars
        valid_beats = [b for b in beat_stats if not b.get("skipped", False)]
        cvl_scalar = float(np.nanmedian([b.get("cvl_m_s", np.nan) for b in valid_beats])) if valid_beats else np.nan
        cvt_scalar = float(np.nanmedian([b.get("cvt_m_s", np.nan) for b in valid_beats])) if valid_beats else np.nan
        aniso_scalar = float(np.nanmedian([b.get("anisotropy_ratio", np.nan) for b in valid_beats])) if valid_beats else np.nan
        angle_scalar = float(np.nanmedian([b.get("fiber_angle_deg", np.nan) for b in valid_beats])) if valid_beats else np.nan
        coher_scalar = float(np.nanmedian([b.get("fiber_coherence", np.nan) for b in valid_beats])) if valid_beats else np.nan
        n_sources_total = int(np.nansum([b.get("n_sources", 0) for b in valid_beats])) if valid_beats else 0
        phenotype = classify_phenotype(cvl_scalar, aniso_scalar)

        # Method distribution
        method_counts = Counter(self._methods_used) if hasattr(self, '_methods_used') else Counter()
        method_dist = dict(method_counts)
        primary_method = method_counts.most_common(1)[0][0] if method_counts else "unknown"

        # Average valid fractions per method
        grad_fracs = [b.get("grad_valid_fraction", np.nan) for b in valid_beats]
        st_fracs = [b.get("st_valid_fraction", np.nan) for b in valid_beats]
        avg_grad_frac = float(np.nanmean(grad_fracs)) if grad_fracs else np.nan
        avg_st_frac = float(np.nanmean(st_fracs)) if st_fracs else np.nan

        # Save MUST
        self.save_must(cv_mean, "cv_mean.npy")
        self.save_must(cv_sd, "cv_sd.npy")
        self.save_must(cv_angles, "cv_angles.npy")
        self.save_must(cv_vx, "cv_vx.npy")
        self.save_must(cv_vy, "cv_vy.npy")
        self.save_must(cvl_mean, "cvl_map.npy")
        self.save_must(cvt_mean, "cvt_map.npy")
        self.save_must(aniso_mean, "anisotropy_map.npy")
        self.save_must(fiber_angle_mean, "fiber_angle_map.npy")
        self.save_must(coh_mean, "coherence_map.npy")

        elapsed = round(time.perf_counter() - t0, 2)

        report = {
            "sample_id": self.sample_id,
            "pixel_size_mm": pixel_size_mm,
            "n_beats_input": n_beats,
            "n_beats_used": len(valid_beats),
            "verdict": verdict,
            "reason": reason,
            "metrics": metrics,
            "cvl_m_s": round(cvl_scalar, 4) if np.isfinite(cvl_scalar) else None,
            "cvt_m_s": round(cvt_scalar, 4) if np.isfinite(cvt_scalar) else None,
            "anisotropy_ratio": round(aniso_scalar, 3) if np.isfinite(aniso_scalar) else None,
            "fiber_angle_deg": round(angle_scalar, 1) if np.isfinite(angle_scalar) else None,
            "fiber_coherence": round(coher_scalar, 4) if np.isfinite(coher_scalar) else None,
            "n_sources": n_sources_total,
            "phenotype": phenotype,
            "primary_method": primary_method,
            "fallback_used": method_counts.get("structure_tensor", 0) > 0,
            "beats_method_distribution": method_dist,
            "avg_valid_fraction": {
                "gradient_angular": round(avg_grad_frac, 4) if np.isfinite(avg_grad_frac) else None,
                "structure_tensor": round(avg_st_frac, 4) if np.isfinite(avg_st_frac) else None,
            },
            "elapsed_s": elapsed,
        }
        self.save_must(report, "conduction_report.json")

        # Save DEBUG — per-beat stacks (already collected in _compute_all_beats)
        beats_cv = getattr(self, '_beats_cv', [])
        beats_cvl = getattr(self, '_beats_cvl', [])
        beats_cvt = getattr(self, '_beats_cvt', [])
        beats_aniso = getattr(self, '_beats_aniso', [])
        if beats_cv:
            self.save_debug(np.array(beats_cv), "cv_per_beat.npy")
            self.save_debug(np.array(beats_cvl), "cvl_per_beat.npy")
            self.save_debug(np.array(beats_cvt), "cvt_per_beat.npy")
            self.save_debug(np.array(beats_aniso), "anisotropy_per_beat.npy")
        self.save_debug(cv_vs_angle_mean, "cv_vs_angle.npy")
        self.save_debug({
            "beat_stats": beat_stats,
            "verdict": verdict,
            "reason": reason,
            "metrics": metrics,
            "phenotype": phenotype,
            "method_distribution": method_dist,
        }, "conduction_debug.json")

        self._log_metrics({**metrics, "verdict": verdict, "elapsed_s": elapsed})
        self.logger.info(
            f"ConductionAgent done in {elapsed}s — {verdict}, phenotype={phenotype}, "
            f"methods={method_dist}"
        )

        return {
            "status": "success",
            "verdict": verdict,
            "metrics": metrics,
            "cvl_m_s": cvl_scalar,
            "cvt_m_s": cvt_scalar,
            "anisotropy_ratio": aniso_scalar,
            "phenotype": phenotype,
            "primary_method": primary_method,
            "method_distribution": method_dist,
        }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ConductionAgent standalone")
    parser.add_argument("sample_id", help="Sample ID (e.g. 005A)")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig({"results_root": args.results_root})
    agent = ConductionAgent(args.sample_id, config=cfg)
    result = agent.run(force=args.force)
    print(result)