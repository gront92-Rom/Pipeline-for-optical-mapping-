#!/usr/bin/env python3
"""
alternans_agent.py — Stage 7: Детекция и анализ альтернанса.
Версия v1 (2026-07-02).

Архитектура:
  Наследует BaseAgent. Читает параметры из PipelineConfig (config/default.yaml).
  Вся математика вынесена в utils/alternans.py.

Входные данные (lazy — запускает APDAgent если нужно):
  - must/apd_per_beat_3d.npz  — 3D стек APD по биениям (от APDAgent)
  - must/mask.npy              — маска ткани (от MaskAgent)
  - must/metadata.json         — fps, dye (от LoaderAgent)

Выходные данные:
  MUST:
    - alternans_magnitude_ms.npy   — карта амплитуды альтернанса (мс)
    - alternans_phase.npy          — карта фазы (+1 / -1 / NaN)
    - alternans_concordance.npy    — карта индекса конкордантности
    - alternans_report.json        — вердикт, метрики, фенотип
  DEBUG:
    - alternans_spatial_maps.png   — 3 карты: magnitude / phase / concordance
    - alternans_dynamics.png       — эволюция, Пуанкаре, FFT-спектр

Фенотипы:
  "Normal"             — AC_95th < ac_threshold_ms
  "Alternans"          — AC_95th >= ac_threshold_ms, concordance_index >= discordant_threshold
  "Discordant"         — AC_95th >= ac_threshold_ms, concordance_index < discordant_threshold

Коды возврата (CLI-режим):
  0 = SUCCESS
  1 = CRASH
  2 = REJECT (мало биений)

Исправления относительно исходного alternans_agent.py:
  - from utils_alternans import ... → from cardiac_pipeline.utils.alternans import ...
  - compute_poincare_correlation вынесена в utils/alternans.py
  - Параметры (min_beats, ac_threshold, sign_floor_ms, discordant_threshold)
    берутся из PipelineConfig / config.yaml (не из argparse)
  - Все пути через BaseAgent API (must_dir / debug_dir)
  - REJECT → raise ValueError (BaseAgent-контракт), CLI перехватывает → exit(2)
  - PNG-карты в debug/ (не в must/)
  - Добавлен lazy-механизм: auto-run APDAgent если нет apd_per_beat_3d.npz
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig
from cardiac_pipeline.utils.alternans import (
    compute_spatial_alternans,
    compute_concordance_map,
    compute_temporal_spectrum,
    compute_poincare_correlation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AlternansAgent
# ---------------------------------------------------------------------------

class AlternansAgent(BaseAgent):
    """
    Stage 7: Детекция и анализ альтернанса потенциала действия / CaT.

    Потребляет apd_per_beat_3d.npz от APDAgent.
    Генерирует карты амплитуды, фазы, конкордантности и клинический фенотип.
    """

    DEPENDS_ON: list = []  # [APDAgent] — установлен ниже (lazy import)
    REQUIRED_INPUTS: list = ["apd_per_beat_3d.npz", "mask.npy"]

    def __init__(
        self,
        sample_id: str,
        config: Optional[PipelineConfig] = None,
    ):
        super().__init__(sample_id, config)

        alt_cfg = self.config.alternans if isinstance(self.config.alternans, dict) else {}

        # Параметры из конфига (согласованы с config/default.yaml)
        self.min_beats:             int   = int(alt_cfg.get("min_beats",             4))
        self.ac_threshold_ms:       float = float(alt_cfg.get("ac_threshold_ms",     2.0))
        self.sign_floor_ms:         float = float(alt_cfg.get("sign_floor_ms",       0.5))
        self.discordant_threshold:  float = float(alt_cfg.get("discordant_threshold", 0.25))
        self.ac_pct_thresholds:     list  = list(alt_cfg.get("ac_pct_thresholds",    [5, 10, 20]))

        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = self.get_path("metadata.json", kind="must")
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            self.logger.warning("metadata.json not found")
        return self.metadata

    def _get_dye(self) -> str:
        dye = self.metadata.get("dye") or self.metadata.get("recording_mode")
        if dye is None:
            self.logger.warning("dye не найден в metadata.json — используется 'A' (VSD)")
            return "A"
        d = str(dye).upper().strip()
        if d in ("A", "VOLTAGE", "VSD", "AP"):
            return "A"
        if d in ("B", "CALCIUM", "CAT", "CA"):
            return "B"
        self.logger.warning(f"Неизвестный dye='{dye}', используется 'A' (VSD)")
        return "A"

    def _load_apd_3d(self) -> tuple:
        """
        Загружает 3D стек APD из apd_per_beat_3d.npz.
        Возвращает (apd80_3d, metric, n_beats).
        """
        npz_path = self.must_dir / "apd_per_beat_3d.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"apd_per_beat_3d.npz не найден в {self.must_dir}. "
                "Запустите APDAgent."
            )
        data    = np.load(npz_path)
        apd80_3d = data["apd80"]          # (H, W, N_beats)
        metric   = str(data["metric"])
        n_beats  = int(data["n_beats"])
        return apd80_3d, metric, n_beats

    # ------------------------------------------------------------------
    # Главный метод
    # ------------------------------------------------------------------

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Запускает анализ альтернанса.

        Порядок:
          1. Lazy-проверка upstream (APDAgent)
          2. Загрузка метаданных → dye
          3. Загрузка apd_per_beat_3d.npz, mask
          4. QC-гейтинг: достаточно ли биений?
          5. Пространственный анализ: амплитуда + фаза + конкордантность
          6. Временной анализ: tissue mean APD, Пуанкаре, FFT
          7. Агрегация метрик + определение фенотипа
          8. Сохранение артефактов (npy, json, png)
        """
        if not force and self.exists("alternans_report.json"):
            self.logger.info("alternans_report.json exists, skipping")
            return {"status": "skipped"}

        t0 = time.perf_counter()

        # --- Lazy: запускаем APD (→ PeakDetector → Loader → Mask) ---
        from cardiac_pipeline.agents.apd_agent import APDAgent
        self.DEPENDS_ON = [APDAgent]
        self.ensure_dependencies(force=force)

        # --- 2. Метаданные ---
        self._load_metadata()
        dye    = self._get_dye()
        metric = "APD" if dye == "A" else "CaT"

        # --- 3. Загрузка данных ---
        apd80_3d, metric_from_npz, n_beats = self._load_apd_3d()
        metric = metric_from_npz  # Приоритет у метки из npz (уже нормализована APDAgent)
        mask   = self.load_must("mask.npy").astype(bool)

        self.logger.info(
            f"AlternansAgent: apd80_3d={apd80_3d.shape}, "
            f"n_beats={n_beats}, mask_cov={mask.mean():.3f}, metric={metric}"
        )

        # --- 4. QC-гейтинг ---
        if n_beats < self.min_beats:
            reason = (
                f"Найдено {n_beats} биений — меньше минимума {self.min_beats} "
                f"для надёжного анализа альтернанса."
            )
            report = {"status": "REJECT", "reason": reason, "n_beats": n_beats}
            self.save_must(report, "alternans_report.json")
            self.logger.error(f"QC REJECT: {reason}")
            raise ValueError(f"AlternansAgent REJECT: {reason}")

        # --- 5. Пространственный анализ ---
        self.logger.info("Расчёт пространственного альтернанса...")
        ac_ms, ac_pct, phase_map = compute_spatial_alternans(
            apd80_3d, mask, sign_floor_ms=self.sign_floor_ms
        )
        concordance_map = compute_concordance_map(phase_map, mask)

        # --- 6. Временной анализ ---
        # Медиана APD по ткани для каждого биения → временной ряд
        with np.errstate(invalid="ignore"):
            tissue_mean_apd = np.nanmedian(apd80_3d[mask], axis=0)  # (N_beats,)

        temporal_diffs = np.diff(tissue_mean_apd)
        _, _, spectral_purity = compute_temporal_spectrum(temporal_diffs)
        poincare_corr = compute_poincare_correlation(tissue_mean_apd)

        # --- 7. Агрегация метрик ---
        valid_ac   = ac_ms[mask & np.isfinite(ac_ms)]
        ac_95th    = float(np.percentile(valid_ac, 95)) if len(valid_ac) > 0 else 0.0
        ac_median  = float(np.median(valid_ac))         if len(valid_ac) > 0 else 0.0

        valid_pct  = ac_pct[mask & np.isfinite(ac_pct)]
        ac_pct_med = float(np.median(valid_pct))        if len(valid_pct) > 0 else 0.0

        valid_conc  = concordance_map[mask & np.isfinite(concordance_map)]
        conc_median = float(np.median(valid_conc))      if len(valid_conc) > 0 else 0.0

        # Доля ткани с выраженным альтернансом (по каждому порогу из конфига)
        pct_tissue_above = {}
        for thr in self.ac_pct_thresholds:
            frac = float(np.mean(ac_pct[mask & np.isfinite(ac_pct)] >= thr)) if len(valid_pct) > 0 else 0.0
            pct_tissue_above[f"tissue_frac_AC_pct_ge_{thr}"] = round(frac, 4)

        # Фенотип
        if ac_95th >= self.ac_threshold_ms:
            if conc_median < self.discordant_threshold:
                phenotype = "Discordant"   # Пространственно-дискордантный — наиболее опасный
            else:
                phenotype = "Alternans"    # Конкордантный альтернанс
        else:
            phenotype = "Normal"

        elapsed = round(time.perf_counter() - t0, 2)

        report = {
            "sample_id":       self.sample_id,
            "status":          "SUCCESS",
            "phenotype":       phenotype,
            "metric":          metric,
            "dye":             dye,
            "n_beats_analyzed": n_beats,
            "metrics": {
                "AC_median_ms":              round(ac_median,       2),
                "AC_95th_percentile_ms":     round(ac_95th,         2),
                "AC_median_pct":             round(ac_pct_med,      2),
                "concordance_index":         round(conc_median,     3),
                "spectral_purity":           round(spectral_purity, 3),
                "poincare_correlation":      round(poincare_corr,   3),
                **pct_tissue_above,
            },
            "thresholds_used": {
                "ac_threshold_ms":      self.ac_threshold_ms,
                "discordant_threshold": self.discordant_threshold,
                "sign_floor_ms":        self.sign_floor_ms,
                "min_beats":            self.min_beats,
            },
            "elapsed_s": elapsed,
        }

        # --- 8. Сохранение артефактов ---
        self.save_must(ac_ms,           "alternans_magnitude_ms.npy")
        self.save_must(phase_map,       "alternans_phase.npy")
        self.save_must(concordance_map, "alternans_concordance.npy")
        self.save_must(report,          "alternans_report.json")

        # Debug: PNG-карты
        self._save_spatial_maps(ac_ms, phase_map, concordance_map, mask, phenotype, ac_95th)
        self._save_dynamics_plot(tissue_mean_apd, temporal_diffs, poincare_corr,
                                 spectral_purity, metric, n_beats)

        self._log_metrics({
            "alternans_phenotype":   phenotype,
            "AC_95th_ms":            ac_95th,
            "concordance_index":     conc_median,
            "spectral_purity":       spectral_purity,
            "poincare_correlation":  poincare_corr,
            "elapsed_s":             elapsed,
        })

        self.logger.info(
            f"AlternansAgent done in {elapsed}s — Phenotype: {phenotype}. "
            f"AC_95th: {ac_95th:.1f} ms, Concordance: {conc_median:.3f}"
        )

        return {
            "status":    "success",
            "phenotype": phenotype,
            "metrics":   report["metrics"],
        }

    # ------------------------------------------------------------------
    # Вспомогательные методы визуализации
    # ------------------------------------------------------------------

    def _save_spatial_maps(
        self,
        ac_ms: np.ndarray,
        phase_map: np.ndarray,
        concordance_map: np.ndarray,
        mask: np.ndarray,
        phenotype: str,
        ac_95th: float,
    ):
        """Сохраняет 3 пространственные карты (magnitude / phase / concordance) в debug/."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            maps = [
                (ac_ms,          "jet",      "Magnitude (ms)",          0, max(5.0, ac_95th)),
                (phase_map,      "coolwarm", "Phase Map (+1 / -1)",     -1, 1),
                (concordance_map,"viridis",  "Concordance (0 to 1)",     0, 1),
            ]
            for ax, (data_map, cmap, title, vmin, vmax) in zip(axes, maps):
                masked_data = np.ma.masked_where(~mask, data_map)
                im = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=11)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.suptitle(
                f"Spatial Alternans Analysis — Phenotype: {phenotype}",
                fontsize=14, weight="bold",
            )
            plt.tight_layout()
            path = self.debug_dir / "alternans_spatial_maps.png"
            plt.savefig(path, dpi=150)
            plt.close()
            self.logger.info(f"[DEBUG] Saved: alternans_spatial_maps.png")

        except Exception as e:
            self.logger.warning(f"Spatial maps PNG skipped: {e}")

    def _save_dynamics_plot(
        self,
        tissue_mean_apd: np.ndarray,
        temporal_diffs: np.ndarray,
        poincare_corr: float,
        spectral_purity: float,
        metric: str,
        n_beats: int,
    ):
        """Сохраняет диаграмму: эволюция APD + Пуанкаре + FFT-спектр в debug/."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            _, freqs, _ = compute_temporal_spectrum(temporal_diffs)
            spec, _, _  = compute_temporal_spectrum(temporal_diffs)

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            # A: Эволюция медианного APD по биениям
            axes[0].plot(range(1, n_beats + 1), tissue_mean_apd, "k-o", markersize=6)
            axes[0].set_title(f"Tissue Median {metric} — Beat Evolution", fontsize=11)
            axes[0].set_xlabel("Beat Number")
            axes[0].set_ylabel("Duration (ms)")
            axes[0].grid(True, alpha=0.3)

            # B: Диаграмма Пуанкаре
            beat_n  = tissue_mean_apd[:-1]
            beat_n1 = tissue_mean_apd[1:]
            axes[1].scatter(beat_n, beat_n1, c="red", alpha=0.75, s=60, zorder=3)
            lo = min(beat_n.min(), beat_n1.min())
            hi = max(beat_n.max(), beat_n1.max())
            axes[1].plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="Identity")
            axes[1].set_title(f"Poincaré Plot (r = {poincare_corr:.2f})", fontsize=11)
            axes[1].set_xlabel(f"Beat N (ms)")
            axes[1].set_ylabel(f"Beat N+1 (ms)")
            axes[1].legend(fontsize=9)
            axes[1].grid(True, alpha=0.3)

            # C: FFT-спектр
            if spec is not None and freqs is not None:
                axes[2].plot(freqs, spec, "b-", linewidth=1.5)
                axes[2].fill_between(freqs, spec, alpha=0.15, color="blue")
                axes[2].axvline(0.5, color="r", linestyle="--",
                                label=f"Nyquist (Alternans)\nPurity={spectral_purity:.2f}")
                axes[2].set_title("FFT Power Spectrum", fontsize=11)
                axes[2].set_xlabel("Frequency (cycles/beat)")
                axes[2].set_ylabel("Power")
                axes[2].legend(fontsize=9)
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            path = self.debug_dir / "alternans_dynamics.png"
            plt.savefig(path, dpi=150)
            plt.close()
            self.logger.info(f"[DEBUG] Saved: alternans_dynamics.png")

        except Exception as e:
            self.logger.warning(f"Dynamics plot PNG skipped: {e}")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="AlternansAgent standalone — Stage 7")
    parser.add_argument("sample_id", help="Sample ID (e.g. 005A)")
    parser.add_argument("--results-root", default="results", help="Results root directory")
    parser.add_argument("--force", action="store_true", help="Force recompute")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg   = PipelineConfig({"results_root": args.results_root})
    agent = AlternansAgent(args.sample_id, config=cfg)

    try:
        result = agent.run(force=args.force)
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0)
    except ValueError as e:
        logger.error(f"REJECT: {e}")
        sys.exit(2)
    except Exception as e:
        logger.exception(f"CRASH: {e}")
        sys.exit(1)
