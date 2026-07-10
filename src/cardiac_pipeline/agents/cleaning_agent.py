"""
cleaning_agent.py — Финальная стадия пайплайна.

Удаляет большие промежуточные файлы (.npy видео, 4D-массивы) после того,
как все агенты отработали и сохранили результаты как:
  - PNG (картинки: карты, трассы)
  - JSON (числа: отчёты, метрики)
  - Маленькие .npy (2D-карты: apd, cv, activation, alternans — по 40-80K)

Правило: файл > threshold_mb МБ и в blacklist → удалить.
Файлы < threshold_mb всегда остаются (maps, mask, metadata).

CLI:
    python -m cardiac_pipeline.agents.cleaning_agent <sample_id> [--dry-run] [--threshold-mb 1.0]
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig

logger = logging.getLogger(__name__)


class CleaningAgent(BaseAgent):
    """
    Удаляет большие промежуточные файлы после завершения всех стадий.

    Blacklist (всегда удаляются если > threshold_mb):
      must/raw_video.npy
      must/preproc_video.npy
      must/preproc_video_apd.npy
      must/raw_rsm.npy
      debug/apd_4d.npy

    Whitelist (никогда не удаляются — независимо от размера):
      *.png, *.json, *.csv, *.txt
      must/mask.npy, must/metadata.json
      Все 2D-карты < 500K (apd*_map.npy, cvl_map.npy, cvt_map.npy, activation_map.npy, etc.)
    """

    DEPENDS_ON = []  # Не зависит от агентов — запускается вручную после всех
    REQUIRED_INPUTS = []  # Не требует обязательных входов

    # Файлы для удаления (относительно sample_dir/must/ или sample_dir/debug/)
    BLACKLIST_MUST = [
        "raw_video.npy",
        "preproc_video.npy",
        "preproc_video_apd.npy",
        "raw_rsm.npy",
    ]
    BLACKLIST_DEBUG = [
        "apd_4d.npy",
    ]

    # Расширения, которые НИКОГДА не удаляются
    PROTECTED_EXTENSIONS = {".png", ".json", ".csv", ".txt", ".md", ".log"}

    # Файлы, которые никогда не удаляются независимо от размера
    PROTECTED_FILES_MUST = {
        "mask.npy", "metadata.json", "metrics.json",
        "apd30_map.npy", "apd50_map.npy", "apd80_map.npy",
        "activation_map.npy", "cvl_map.npy", "cvt_map.npy",
        "anisotropy_map.npy", "fiber_angle_map.npy",
        "cv_mean.npy", "cv_sd.npy", "cv_angles.npy",
        "cv_vx.npy", "cv_vy.npy",
        "coherence_map.npy",
        "alternans_magnitude_ms.npy", "alternans_concordance.npy",
        "alternans_phase.npy",
        "per_beat_activation.npy", "weights.npy",
        "region_masks.npy", "traces_per_region.npy",
        "apd_per_beat_3d.npz",
        "activation_report.json", "apd_report.json",
        "conduction_report.json", "alternans_report.json",
        "peak_detection_meta.json",
    }
    PROTECTED_FILES_DEBUG = {
        "conduction_debug.json", "preproc_stats.json",
        "hot_mask.npy", "mean_trace.npy", "mean_tissue_raw.npy",
        "tat_per_region.npy",
        "cv_per_beat.npy", "cvl_per_beat.npy", "cvt_per_beat.npy",
        "anisotropy_per_beat.npy",
    }

    def __init__(self, sample_id: str, config: Optional[PipelineConfig] = None,
                 threshold_mb: float = 1.0, dry_run: bool = False):
        super().__init__(sample_id, config=config)
        self.threshold_mb = threshold_mb
        self.dry_run = dry_run

    def _file_size_mb(self, path: Path) -> float:
        """Размер файла в МБ."""
        try:
            return path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0

    def _is_protected(self, path: Path, kind: str) -> bool:
        """Проверяет, защищён ли файл от удаления."""
        # Расширение в protected set
        if path.suffix in self.PROTECTED_EXTENSIONS:
            return True

        name = path.name
        protected_set = (
            self.PROTECTED_FILES_MUST if kind == "must"
            else self.PROTECTED_FILES_DEBUG
        )
        return name in protected_set

    def _is_blacklisted(self, path: Path, kind: str) -> bool:
        """Проверяет, в blacklist ли файл."""
        name = path.name
        blacklist = (
            self.BLACKLIST_MUST if kind == "must"
            else self.BLACKLIST_DEBUG
        )
        return name in blacklist

    def _collect_files(self) -> List[Dict[str, Any]]:
        """Собирает все файлы с метаданными для решения об удалении."""
        files_info = []
        for kind, directory in [("must", self.must_dir), ("debug", self.debug_dir)]:
            if not directory.exists():
                continue
            for path in sorted(directory.iterdir()):
                if not path.is_file():
                    continue
                size_mb = self._file_size_mb(path)
                files_info.append({
                    "path": path,
                    "name": path.name,
                    "kind": kind,
                    "size_mb": round(size_mb, 2),
                    "blacklisted": self._is_blacklisted(path, kind),
                    "protected": self._is_protected(path, kind),
                })
        return files_info

    def _decide_deletion(self, file_info: Dict[str, Any]) -> bool:
        """
        Решение об удалении:
        1. Если в blacklist → удалить (независимо от размера).
        2. Если .npy > threshold_mb И не protected → удалить.
        3. Иначе → оставить.
        """
        if file_info["protected"]:
            return False
        if file_info["blacklisted"]:
            return True
        # Любой .npy больше threshold, не в whitelist → удалить
        if file_info["path"].suffix == ".npy" and file_info["size_mb"] >= self.threshold_mb:
            return True
        return False

    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Удаляет большие промежуточные файлы.

        Returns:
            dict со списками deleted/kept/skipped + total_freed_mb
        """
        self.logger.info(
            f"CleaningAgent started | threshold={self.threshold_mb}MB | dry_run={self.dry_run}"
        )

        files = self._collect_files()
        if not files:
            self.logger.warning("No files found — nothing to clean.")
            return {"status": "skip", "reason": "no files", "deleted": [], "kept": [],
                    "total_freed_mb": 0.0}

        deleted = []
        kept = []
        freed_mb = 0.0

        for fi in files:
            should_delete = self._decide_deletion(fi)
            if should_delete:
                size = fi["size_mb"]
                if not self.dry_run:
                    try:
                        fi["path"].unlink()
                        self.logger.info(
                            f"  [DEL] {fi['kind']}/{fi['name']}  ({size:.2f} MB)"
                        )
                    except OSError as e:
                        self.logger.error(f"  [ERR] {fi['name']}: {e}")
                        kept.append(fi)
                        continue
                else:
                    self.logger.info(
                        f"  [DRY] would delete {fi['kind']}/{fi['name']}  ({size:.2f} MB)"
                    )
                deleted.append(fi["name"])
                freed_mb += size
            else:
                kept.append(fi["name"])

        self.logger.info(
            f"CleaningAgent done | deleted={len(deleted)} | kept={len(kept)} | "
            f"freed={freed_mb:.2f} MB{' (DRY RUN)' if self.dry_run else ''}"
        )

        # Сохраняем отчёт
        report = {
            "status": "ok",
            "dry_run": self.dry_run,
            "threshold_mb": self.threshold_mb,
            "deleted": deleted,
            "kept": kept,
            "total_freed_mb": round(freed_mb, 2),
        }
        report_path = self.must_dir / "cleaning_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Report saved: {report_path}")

        # Метрики
        self._log_metrics({
            "cleaning": {
                "deleted_count": len(deleted),
                "freed_mb": round(freed_mb, 2),
                "dry_run": self.dry_run,
            }
        })

        return report


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(name)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description="CleaningAgent — delete large intermediate files")
    parser.add_argument("sample_id", help="Sample ID (e.g. 004A)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--threshold-mb", type=float, default=1.0,
                        help="Size threshold for .npy deletion (default: 1.0 MB)")
    parser.add_argument("--results-root", default="results", help="Results root directory")
    args = parser.parse_args()

    cfg = PipelineConfig({"results_root": args.results_root})
    agent = CleaningAgent(
        args.sample_id,
        config=cfg,
        threshold_mb=args.threshold_mb,
        dry_run=args.dry_run,
    )
    result = agent.run()
    print("=== CLEANING RESULT ===")
    print(json.dumps(result, indent=2))