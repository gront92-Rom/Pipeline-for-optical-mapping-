#!/usr/bin/env python3
"""
conduction_consensus_agent.py — CLI-агент расчёта скорости проведения (CV).

Использует консенсус двух методов и рассчитывает per-beat статистику (Mean + SD).
Совместим с CLI-архитектурой оркестратора (optical_pipeline_worker.py).

Исправления относительно кора (2026-07-02):
  BUG-1 (F5/ImportError): Убран импорт cv_method_local_fit из conduction_analysis
        (функция не существует). Оба метода теперь из utils/cv_estimators.py.
  BUG-2 (CV2/pixel_size): Убран дефолт --pixel-size=0.85. Аргумент обязателен
        (required=True). Оркестратор ДОЛЖЕН передавать значение из metadata.json.
  BUG-3 (SC1/exit code): sys.exit(0) при REJECT заменён на sys.exit(2).
        Оркестратор различает: 0=SUCCESS, 1=crash, 2=REJECT/QC-fail.
  BUG-4 (SC7/NaN): Консенсусная маска теперь явно включает проверку isfinite
        для обоих методов (не только relative_diff).
  BUG-5 (C2/silent): При пустом результате логируется явный ERROR + exit(2),
        а не молчаливый exit(0).

Outputs (сохраняются в output_dir):
  MUST (при SUCCESS):
    cvl_mean.npy    — средняя продольная CV по битам (м/с)
    cvl_sd.npy      — SD продольной CV по битам (м/с)
    cv_report.json  — QC-метрики, параметры, вердикт
  OPTIONAL (если structure tensor вернул cvt):
    cvt_mean.npy    — средняя поперечная CV по битам (м/с)

Exit codes:
  0 — SUCCESS (файлы сохранены, QC PASS или WARN)
  1 — Crash (исключение, ошибка чтения файлов)
  2 — REJECT (QC ниже порога, файлы НЕ сохраняются)

Пример вызова из оркестратора:
  subprocess.run([
      "python3", "-m", "cardiac_pipeline.agents.conduction_consensus_agent",
      "per_beat_activation.npy", "mask.npy", output_dir,
      "--pixel-size", str(metadata["pixel_size_mm"]),
      "--tolerance", "0.15",
      "--qc-threshold", "0.20",
  ], check=False)
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Импорт из utils/cv_estimators (BUG-1 fix: не из conduction_analysis)
# ---------------------------------------------------------------------------
try:
    # Попытка импорта как пакета (при запуске через -m или из installed package)
    from cardiac_pipeline.utils.cv_estimators import (
        compute_hybrid_structure_tensor,
        compute_polynomial_bayly,
        estimate_cv_stats,
    )
except ImportError:
    # Fallback: добавляем src/ в sys.path (при прямом запуске python3 script.py)
    _here = Path(__file__).resolve()
    _src = _here.parent.parent.parent  # src/
    sys.path.insert(0, str(_src))
    from cardiac_pipeline.utils.cv_estimators import (
        compute_hybrid_structure_tensor,
        compute_polynomial_bayly,
        estimate_cv_stats,
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ConductionConsensusAgent")

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
EXIT_SUCCESS = 0
EXIT_CRASH   = 1
EXIT_REJECT  = 2  # BUG-3 fix: был 0, теперь 2


# ---------------------------------------------------------------------------
# Per-beat consensus
# ---------------------------------------------------------------------------

def _compute_beat_consensus(
    beat_tat: np.ndarray,
    mask: np.ndarray,
    pixel_size_mm: float,
    tolerance: float,
    cv_min: float,
    cv_max: float,
) -> Optional[np.ndarray]:
    """
    Консенсус CV для одного бита.

    Возвращает cv_consensus (H, W) или None при неудаче.
    """
    # Метод 1: hybrid structure tensor
    cv1, _ang1, _coh1 = compute_hybrid_structure_tensor(
        beat_tat, mask, pixel_size_mm,
        cv_min=cv_min, cv_max=cv_max,
    )

    # Метод 2: polynomial Bayly (Гаусс-сглаженный градиент)
    cv2, _ang2, _coh2 = compute_polynomial_bayly(
        beat_tat, mask, pixel_size_mm,
        cv_min=cv_min, cv_max=cv_max,
    )

    # BUG-4 fix: явная проверка isfinite для обоих методов
    both_finite = np.isfinite(cv1) & np.isfinite(cv2)

    max_cv = np.fmax(cv1, cv2)  # fmax игнорирует NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_diff = np.abs(cv1 - cv2) / (max_cv + 1e-9)

    consensus_mask = mask & both_finite & (relative_diff <= tolerance)

    cv_consensus = np.where(consensus_mask, (cv1 + cv2) / 2.0, np.nan)
    return cv_consensus


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate CV using method consensus and per-beat SD."
    )
    parser.add_argument("tat_file",    help="Input per-beat activation map .npy (shape: [n_beats, H, W], ms)")
    parser.add_argument("mask_file",   help="Input mask .npy (bool)")
    parser.add_argument("output_dir",  help="Directory to save results")

    # BUG-2 fix: required=True, нет дефолта 0.85
    parser.add_argument(
        "--pixel-size", type=float, required=True,
        help="Pixel size in mm (from metadata.json — NO default, must be explicit)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.15,
        help="Consensus tolerance — max relative difference between methods (default: 0.15)",
    )
    parser.add_argument(
        "--qc-threshold", type=float, default=0.20,
        help="Min valid tissue fraction for PASS (default: 0.20)",
    )
    parser.add_argument(
        "--cv-min", type=float, default=0.05,
        help="Physiological CV lower bound m/s (default: 0.05)",
    )
    parser.add_argument(
        "--cv-max", type=float, default=2.0,
        help="Physiological CV upper bound m/s (default: 2.0)",
    )

    args = parser.parse_args()

    # Validate pixel_size
    if args.pixel_size <= 0:
        logger.error(f"--pixel-size must be > 0, got {args.pixel_size}")
        return EXIT_CRASH

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Загрузка данных
    try:
        per_beat_tat = np.load(args.tat_file)   # shape: (n_beats, H, W), мс
        mask = np.load(args.mask_file).astype(bool)
    except Exception as e:
        logger.error(f"Failed to load input files: {e}")
        return EXIT_CRASH

    # Нормализуем 2D → 3D (один бит)
    if per_beat_tat.ndim == 2:
        per_beat_tat = per_beat_tat[np.newaxis, ...]

    n_beats = per_beat_tat.shape[0]
    logger.info(
        f"Processing {n_beats} beats | pixel_size={args.pixel_size} mm | "
        f"tolerance={args.tolerance} | qc_threshold={args.qc_threshold}"
    )

    # 2. Per-beat обработка
    valid_beats_cv: List[np.ndarray] = []
    beat_stats = []

    for beat_idx in range(n_beats):
        beat_tat = per_beat_tat[beat_idx]

        # Пропускаем пустые биты
        if not np.any(np.isfinite(beat_tat[mask])):
            logger.warning(f"Beat {beat_idx}: all-NaN activation map, skipping")
            beat_stats.append({"beat": beat_idx, "skipped": True, "reason": "all-NaN activation"})
            continue

        try:
            cv_b = _compute_beat_consensus(
                beat_tat, mask,
                pixel_size_mm=args.pixel_size,
                tolerance=args.tolerance,
                cv_min=args.cv_min,
                cv_max=args.cv_max,
            )
        except Exception as e:
            logger.warning(f"Beat {beat_idx}: CV computation failed — {e}")
            beat_stats.append({"beat": beat_idx, "skipped": True, "reason": str(e)})
            continue

        s = estimate_cv_stats(cv_b, mask)
        logger.info(
            f"Beat {beat_idx}: valid_fraction={s['valid_fraction']:.3f}, "
            f"cv_median={s['cv_median_m_per_s']} m/s"
        )
        beat_stats.append({"beat": beat_idx, "skipped": False, **s})
        valid_beats_cv.append(cv_b)

    # BUG-5 fix: явный REJECT + exit(2) вместо тихого exit(0)
    if not valid_beats_cv:
        logger.error("All beats failed CV calculation — REJECT")
        _save_report(args, beat_stats, verdict="REJECT", reason="all beats failed",
                     acceptance_rate=0.0, n_beats_used=0, n_beats_input=n_beats)
        return EXIT_REJECT

    # 3. Агрегация
    beats_arr = np.array(valid_beats_cv)  # [N_valid, H, W]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cv_mean = np.nanmean(beats_arr, axis=0)
        cv_sd   = np.nanstd(beats_arr, axis=0)

    # 4. QC
    total_tissue = int(np.sum(mask))
    valid_pixels = int(np.sum(np.isfinite(cv_mean) & mask))
    acceptance_rate = valid_pixels / total_tissue if total_tissue > 0 else 0.0

    logger.info(f"QC: consensus reached for {acceptance_rate:.1%} of tissue")

    # BUG-3 fix: exit(2) вместо exit(0) при REJECT
    if acceptance_rate < args.qc_threshold:
        logger.warning(
            f"QC REJECT: accepted tissue ({acceptance_rate:.1%}) "
            f"< threshold ({args.qc_threshold:.1%})"
        )
        _save_report(
            args, beat_stats, verdict="REJECT",
            reason=f"acceptance_rate={acceptance_rate:.3f} < qc_threshold={args.qc_threshold}",
            acceptance_rate=acceptance_rate,
            n_beats_used=len(valid_beats_cv),
            n_beats_input=n_beats,
        )
        return EXIT_REJECT

    # 5. Сохранение
    np.save(os.path.join(args.output_dir, "cvl_mean.npy"), cv_mean)
    np.save(os.path.join(args.output_dir, "cvl_sd.npy"),   cv_sd)

    verdict = "PASS" if acceptance_rate >= 0.5 else "WARN"
    _save_report(
        args, beat_stats, verdict=verdict, reason="OK",
        acceptance_rate=acceptance_rate,
        n_beats_used=len(valid_beats_cv),
        n_beats_input=n_beats,
    )

    logger.info(f"SUCCESS ({verdict}): CV saved to {args.output_dir}")
    return EXIT_SUCCESS


def _save_report(
    args,
    beat_stats: list,
    verdict: str,
    reason: str,
    acceptance_rate: float,
    n_beats_used: int,
    n_beats_input: int,
) -> None:
    """Сохраняет cv_report.json в output_dir (всегда, даже при REJECT)."""
    report = {
        "verdict":          verdict,
        "reason":           reason,
        "pixel_size_mm":    args.pixel_size,
        "tolerance":        args.tolerance,
        "qc_threshold":     args.qc_threshold,
        "cv_min_m_per_s":   args.cv_min,
        "cv_max_m_per_s":   args.cv_max,
        "acceptance_rate":  round(acceptance_rate, 4),
        "n_beats_input":    n_beats_input,
        "n_beats_used":     n_beats_used,
        "beat_stats":       beat_stats,
    }
    path = os.path.join(args.output_dir, "cv_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
