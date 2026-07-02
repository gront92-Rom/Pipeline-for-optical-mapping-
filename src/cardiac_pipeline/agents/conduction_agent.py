#!/usr/bin/env python3
"""
ConductionAgent — Conduction Velocity Analysis Agent (Stage: CV)

Рассчитывает карту скорости проведения (CV) по per-beat картам активации,
используя консенсус двух независимых методов:
  1. compute_hybrid_structure_tensor — прямой градиент (быстро, точно при высоком SNR)
  2. compute_polynomial_bayly       — Гаусс-сглаженный градиент (устойчив к шуму)

Консенсус: пиксель принимается только если оба метода согласуются в пределах
tolerance (15% по умолчанию). Итоговые карты — nanmean и nanstd по битам.

Inputs (lazy — запускает ActivationAgent если нужно):
  - must/per_beat_activation.npy   (from ActivationAgent, shape: [N, H, W], мс)
  - must/mask.npy                  (from MaskAgent, bool)
  - must/metadata.json             (from LoaderAgent, содержит pixel_size_mm)

Outputs:
  MUST:
    - cv_mean.npy           — средняя карта CV по битам (м/с)
    - cv_sd.npy             — стандартное отклонение CV по битам (м/с)
    - cv_angles.npy         — медианная карта направлений (рад)
    - conduction_report.json — вердикт, метрики, параметры
  DEBUG:
    - cv_per_beat.npy       — CV для каждого бита [N, H, W]
    - cv_coherence.npy      — медианная карта когерентности
    - conduction_debug.json — детали QC по каждому биту

Исправления относительно кора (2026-07-02):
  - F5 fix: cv_method_local_fit не импортируется (не существует в conduction_analysis)
  - CV2 fix: pixel_size_mm берётся из metadata.json, без хардкода и дефолта
  - CV4 fix: добавлен judge_conduction() с единым QC и физиологическими границами из конфига
  - CV5 fix: абсолютных путей нет — все пути через self.get_path() / BaseAgent API
  - C1 fix: fps не используется в CV-расчёте (карта активации уже в мс)
  - C2 fix: NaN-CV → REJECT, не тихий SUCCESS
  - SC1 fix: всегда ненулевой exit при REJECT (raise ValueError)
  - SC6 fix: маска берётся только из must/ текущего sample_id
  - SC7 fix: NaN вне маски заполняются до градиента, но результат маскируется обратно
  - np.warnings (deprecated) → warnings.catch_warnings
  - process() → run(force=False) (BaseAgent API)
  - update_status() → raise ValueError / return dict (BaseAgent API)
"""

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig
from cardiac_pipeline.utils.cv_estimators import (
    compute_hybrid_structure_tensor,
    compute_polynomial_bayly,
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
    qc_threshold: float = 0.20,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Оценивает качество карты CV.

    Возвращает (verdict, reason, metrics).
    verdict: PASS / WARN / REJECT

    Правила:
      REJECT — нет ни одного валидного пикселя в маске
      REJECT — valid_fraction < qc_threshold
      WARN   — valid_fraction < 0.5
      PASS   — valid_fraction >= 0.5
    """
    if cv_mean is None or not mask.any():
        return "REJECT", "empty cv_mean or empty mask", {}

    stats = estimate_cv_stats(cv_mean, mask)
    metrics: Dict[str, Any] = {**stats}
    metrics["cv_min_config"] = cv_min
    metrics["cv_max_config"] = cv_max
    metrics["qc_threshold"]  = qc_threshold

    valid_fraction = stats["valid_fraction"]

    if stats["valid_pixels"] == 0:
        return "REJECT", "all CV pixels are NaN", metrics

    if valid_fraction < qc_threshold:
        return (
            "REJECT",
            f"valid_fraction={valid_fraction:.3f} < qc_threshold={qc_threshold:.2f}",
            metrics,
        )

    if valid_fraction < 0.5:
        return (
            "WARN",
            f"valid_fraction={valid_fraction:.3f} < 0.5 (low coverage)",
            metrics,
        )

    return "PASS", "OK", metrics


# ---------------------------------------------------------------------------
# ConductionAgent
# ---------------------------------------------------------------------------

class ConductionAgent(BaseAgent):
    """
    ConductionAgent — рассчитывает карту скорости проведения (CV).

    Консенсус двух методов (hybrid_structure_tensor + polynomial_bayly)
    по каждому биту → nanmean / nanstd по битам.
    """

    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None):
        super().__init__(sample_id, config)

        cv_cfg = getattr(self.config, "conduction", {}) or {}

        # Допустимая относительная разница между методами для консенсуса
        self.tolerance: float = float(cv_cfg.get("tolerance", 0.15))

        # Физиологические границы CV (м/с = мм/мс)
        self.cv_min: float = float(cv_cfg.get("cv_min_m_per_s", 0.05))
        self.cv_max: float = float(cv_cfg.get("cv_max_m_per_s", 2.0))

        # Минимальная доля ткани с валидным CV для PASS
        self.qc_threshold: float = float(cv_cfg.get("qc_threshold", 0.20))

        # Параметр сглаживания для метода Бейли (пикселей)
        self.window_size: float = float(cv_cfg.get("integration_sigma", 4.0))

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
        """
        Извлекает pixel_size_mm из metadata.json.

        CV2 fix: нет хардкода 0.85, нет молчаливого дефолта.
        REJECT если отсутствует или <= 0.
        """
        px = self.metadata.get("pixel_size_mm")
        if px is None:
            raise ValueError(
                "pixel_size_mm отсутствует в metadata.json. "
                "LoaderAgent должен сохранить его заранее."
            )
        px = float(px)
        if px <= 0:
            raise ValueError(
                f"pixel_size_mm некорректен (pixel_size_mm={px}). "
                "Проверьте metadata.json."
            )
        return px

    def _ensure_activation_agent(self) -> None:
        """Запускает ActivationAgent если per_beat_activation.npy отсутствует."""
        if not self.exists("per_beat_activation.npy"):
            self.logger.info("per_beat_activation.npy missing — running ActivationAgent")
            from cardiac_pipeline.agents.activation_agent import ActivationAgent
            ActivationAgent(self.sample_id, self.config).run()

    # ==================== CONSENSUS COMPUTATION ====================

    def _compute_beat_cv(
        self,
        beat_map: np.ndarray,
        mask: np.ndarray,
        pixel_size_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Консенсус CV для одного бита.

        Возвращает (cv_consensus, angles_m1, coherence_m1).
        Пиксели, где методы расходятся > tolerance, → NaN.
        """
        cv1, ang1, coh1 = compute_hybrid_structure_tensor(
            beat_map, mask, pixel_size_mm,
            cv_min=self.cv_min, cv_max=self.cv_max,
        )
        cv2, _ang2, _coh2 = compute_polynomial_bayly(
            beat_map, mask, pixel_size_mm,
            window_size=self.window_size,
            cv_min=self.cv_min, cv_max=self.cv_max,
        )

        # Консенсус: относительная разница <= tolerance
        max_cv = np.fmax(cv1, cv2)  # fmax игнорирует NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = np.abs(cv1 - cv2) / (max_cv + 1e-9)

        consensus_mask = (
            mask
            & np.isfinite(cv1)
            & np.isfinite(cv2)
            & (rel_diff <= self.tolerance)
        )

        cv_consensus = np.where(consensus_mask, (cv1 + cv2) / 2.0, np.nan)

        return cv_consensus, ang1, coh1

    def _compute_consensus_cv(
        self,
        per_beat_activation: np.ndarray,
        mask: np.ndarray,
        pixel_size_mm: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Расчёт консенсусного CV по всем битам.

        Возвращает:
          cv_mean    — nanmean по битам (H, W)
          cv_sd      — nanstd по битам  (H, W)
          cv_angles  — nanmedian углов  (H, W)
          cv_coherence — nanmedian когерентности (H, W)
          beat_stats — список dict с QC по каждому биту
        """
        n_beats = per_beat_activation.shape[0]
        H, W = mask.shape

        beats_cv: List[np.ndarray] = []
        beats_ang: List[np.ndarray] = []
        beats_coh: List[np.ndarray] = []
        beat_stats: List[Dict] = []

        for i in range(n_beats):
            beat_map = per_beat_activation[i]

            # Пропускаем бит если карта пустая
            if not np.any(np.isfinite(beat_map[mask])):
                self.logger.warning(f"Beat {i}: activation map is all-NaN, skipping")
                beat_stats.append({"beat": i, "skipped": True, "reason": "all-NaN activation"})
                continue

            try:
                cv_b, ang_b, coh_b = self._compute_beat_cv(beat_map, mask, pixel_size_mm)
            except Exception as exc:
                self.logger.warning(f"Beat {i}: CV computation failed — {exc}")
                beat_stats.append({"beat": i, "skipped": True, "reason": str(exc)})
                continue

            stats_b = estimate_cv_stats(cv_b, mask)
            beat_stats.append({
                "beat": i,
                "skipped": False,
                **stats_b,
            })
            self.logger.info(
                f"Beat {i}: valid_fraction={stats_b['valid_fraction']:.3f}, "
                f"cv_median={stats_b['cv_median_m_per_s']} m/s"
            )

            beats_cv.append(cv_b)
            beats_ang.append(ang_b)
            beats_coh.append(coh_b)

        if len(beats_cv) == 0:
            nan_map = np.full((H, W), np.nan)
            return nan_map, nan_map.copy(), nan_map.copy(), nan_map.copy(), beat_stats

        beats_cv_arr = np.array(beats_cv)   # [N_valid, H, W]
        beats_ang_arr = np.array(beats_ang)
        beats_coh_arr = np.array(beats_coh)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cv_mean = np.nanmean(beats_cv_arr, axis=0)
            cv_sd   = np.nanstd(beats_cv_arr, axis=0)
            cv_angles  = np.nanmedian(beats_ang_arr, axis=0)
            cv_coherence = np.nanmedian(beats_coh_arr, axis=0)

        # Маскируем вне ткани
        cv_mean  = np.where(mask, cv_mean, np.nan)
        cv_sd    = np.where(mask, cv_sd,   np.nan)
        cv_angles = np.where(mask, cv_angles, np.nan)
        cv_coherence = np.where(mask, cv_coherence, np.nan)

        return cv_mean, cv_sd, cv_angles, cv_coherence, beat_stats

    # ==================== RUN ====================

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Главный метод агента.

        Шаги:
          1. Lazy-запуск ActivationAgent если нужно
          2. Загрузка metadata.json → pixel_size_mm (REJECT если нет)
          3. Загрузка mask.npy + per_beat_activation.npy
          4. Расчёт консенсусного CV по битам
          5. QC / judge → REJECT если valid_fraction < qc_threshold
          6. Сохранение артефактов (MUST + DEBUG)
          7. Возврат report dict
        """
        if not force and self.exists("cv_mean.npy"):
            self.logger.info("cv_mean.npy exists, skipping (use force=True to recompute)")
            return {"status": "skipped"}

        t0 = time.perf_counter()

        # --- 1. Lazy upstream ---
        self._ensure_activation_agent()

        # --- 2. Metadata ---
        self._load_metadata()
        pixel_size_mm = self._get_pixel_size_mm()

        # --- 3. Load inputs ---
        mask = self.load_must("mask.npy").astype(bool)
        per_beat_activation = self.load_must("per_beat_activation.npy")

        # Нормализуем форму: если 2D (один бит) → добавляем ось
        if per_beat_activation.ndim == 2:
            per_beat_activation = per_beat_activation[np.newaxis, ...]

        n_beats, H, W = per_beat_activation.shape
        self.logger.info(
            f"Loaded: mask={mask.shape}, per_beat_activation={per_beat_activation.shape}, "
            f"pixel_size_mm={pixel_size_mm}"
        )

        if n_beats == 0:
            raise ValueError("per_beat_activation.npy содержит 0 битов.")

        # --- 4. Расчёт CV ---
        self.logger.info(
            f"Computing consensus CV: {n_beats} beats, "
            f"tolerance={self.tolerance}, cv_range=[{self.cv_min}, {self.cv_max}] m/s"
        )
        cv_mean, cv_sd, cv_angles, cv_coherence, beat_stats = self._compute_consensus_cv(
            per_beat_activation, mask, pixel_size_mm
        )

        # --- 5. QC ---
        verdict, reason, metrics = judge_conduction(
            cv_mean, mask,
            cv_min=self.cv_min,
            cv_max=self.cv_max,
            qc_threshold=self.qc_threshold,
        )
        self.logger.info(f"QC verdict: {verdict} — {reason}")

        if verdict == "REJECT":
            # C2 fix: REJECT → raise, не тихий exit 0
            raise ValueError(
                f"ConductionAgent REJECT: {reason}. "
                f"Sample {self.sample_id} requires manual review."
            )

        # --- 6. Сохранение ---
        # MUST
        self.save_must(cv_mean,   "cv_mean.npy")
        self.save_must(cv_sd,     "cv_sd.npy")
        self.save_must(cv_angles, "cv_angles.npy")

        elapsed = round(time.perf_counter() - t0, 2)

        report = {
            "sample_id":      self.sample_id,
            "pixel_size_mm":  pixel_size_mm,
            "n_beats_input":  n_beats,
            "n_beats_used":   sum(1 for b in beat_stats if not b.get("skipped", False)),
            "tolerance":      self.tolerance,
            "cv_min_m_per_s": self.cv_min,
            "cv_max_m_per_s": self.cv_max,
            "qc_threshold":   self.qc_threshold,
            "verdict":        verdict,
            "reason":         reason,
            "metrics":        metrics,
            "elapsed_s":      elapsed,
        }
        self.save_must(report, "conduction_report.json")

        # DEBUG
        # Сохраняем per-beat CV stack (может быть большим)
        valid_beats_cv = []
        for i, b in enumerate(beat_stats):
            if not b.get("skipped", False) and i < per_beat_activation.shape[0]:
                bm = per_beat_activation[i]
                if np.any(np.isfinite(bm[mask])):
                    cv_b, _, _ = self._compute_beat_cv(bm, mask, pixel_size_mm)
                    valid_beats_cv.append(cv_b)
        if valid_beats_cv:
            self.save_debug(np.array(valid_beats_cv), "cv_per_beat.npy")
        self.save_debug(cv_coherence, "cv_coherence.npy")
        self.save_debug({
            "beat_stats": beat_stats,
            "verdict":    verdict,
            "reason":     reason,
            "metrics":    metrics,
        }, "conduction_debug.json")

        self._log_metrics({**metrics, "verdict": verdict, "elapsed_s": elapsed})
        self.logger.info(f"ConductionAgent finished in {elapsed}s — {verdict}")

        return {
            "status":  "success",
            "verdict": verdict,
            "metrics": metrics,
        }


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ConductionAgent standalone")
    parser.add_argument("sample_id", help="Sample ID (e.g. 005A)")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--force", action="store_true", help="Recompute even if cv_mean.npy exists")
    args = parser.parse_args()

    cfg = PipelineConfig({"results_root": args.results_root})
    agent = ConductionAgent(args.sample_id, config=cfg)
    result = agent.run(force=args.force)
    print(result)
