"""
report_agent.py — Финальный отчётный агент.

Собирает все JSON-отчёты из всех стадий пайплайна (must/ директория)
и формирует единую плоскую сводную таблицу (одна строка на образец).

Источники (must/):
  - metadata.json            (loader stage)
  - metrics.json             (накопленные метрики всех стадий)
  - peak_detection_meta.json
  - activation_report.json
  - apd_report.json
  - conduction_report.json
  - alternans_report.json
  - cleaning_report.json

Выходные данные:
  MUST:
    - report.csv   — одна строка, плоская таблица (удобно конкатенировать)
    - report.json  — те же данные в JSON (для программного использования)

DEPENDS_ON = [], REQUIRED_INPUTS = [] — агент запускается вручную,
читает уже существующие файлы. Пропущенные JSON заполняются None.

CLI:
  Одиночный режим:
    python -m cardiac_pipeline.agents.report_agent <sample_id> [--results-root results]

  Пакетный режим (--batch): итерирует все поддиректории results-root,
  строит объединённый CSV (одна строка на образец) и сохраняет
  в <results-root>/report_batch.csv:
    python -m cardiac_pipeline.agents.report_agent --batch --results-root results
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cardiac_pipeline.base_agent import BaseAgent, PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Стандартизированный порядок колонок
# ---------------------------------------------------------------------------
COLUMNS: List[str] = [
    # --- идентификация ---
    "sample_id",
    "dye",
    "protocol",
    "model",
    "timepoint",
    "recording_mode",
    # --- acquisition / geometry ---
    "fps",
    "n_frames",
    "width",
    "height",
    "pixel_size_mm",
    # --- stimulation ---
    "stim_hz",
    "stim_hz_effective",
    "bcl_ms",
    # --- peaks / beats ---
    "n_peaks",
    "n_beats",
    "n_beats_selected",
    # --- activation ---
    "n_active_pixels",
    "valid_coverage",
    "tat_max_ms",
    "tat_std_ms",
    # --- APD ---
    "apd30_median_ms",
    "apd50_median_ms",
    "apd80_median_ms",
    # --- conduction ---
    "cvl_m_s",
    "cvt_m_s",
    "anisotropy_ratio",
    "fiber_angle_deg",
    "central_cvl_m_s",
    "central_cvt_m_s",
    "central_anisotropy_ratio",
    "central_cv_m_s",
    # --- alternans ---
    "alternans_phenotype",
    "AC_95th_ms",
    "concordance_index",
    "spectral_purity",
    "poincare_correlation",
    # --- cleaning ---
    "cleaning_deleted_count",
    "cleaning_freed_mb",
]


# JSON-файлы, которые пытаемся прочитать (имя → назначение)
REPORT_FILES = {
    "metadata": "metadata.json",
    "metrics": "metrics.json",
    "peak": "peak_detection_meta.json",
    "activation": "activation_report.json",
    "apd": "apd_report.json",
    "conduction": "conduction_report.json",
    "alternans": "alternans_report.json",
    "cleaning": "cleaning_report.json",
}


def _load_json(path: Path) -> Optional[dict]:
    """Безопасно загружает JSON. Возвращает None при отсутствии/ошибке."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Не удалось прочитать {path}: {e}")
        return None


def _g(d: Optional[dict], *keys, default=None):
    """
    Безопасное вложенное извлечение: _g(d, "metrics", "AC_95th_percentile_ms").
    Возвращает default, если d None или ключ отсутствует.
    """
    if d is None:
        return default
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


# ---------------------------------------------------------------------------
# ReportAgent
# ---------------------------------------------------------------------------
class ReportAgent(BaseAgent):
    """
    Сборочный агент: собирает метрики из всех JSON-отчётов в одну плоскую строку.

    Не зависит от других агентов (DEPENDS_ON=[]), не требует обязательных
    входов (REQUIRED_INPUTS=[]). Читает уже существующие файлы в must/.
    Пропущенные файлы → соответствующие колонки заполняются None.
    """

    DEPENDS_ON: list = []
    REQUIRED_INPUTS: list = []

    def __init__(
        self,
        sample_id: str,
        config: Optional[Union[PipelineConfig, dict]] = None,
    ):
        super().__init__(sample_id, config=config)

    # ------------------------------------------------------------------
    # Чтение источников
    # ------------------------------------------------------------------
    def _load_all_reports(self) -> Dict[str, Optional[dict]]:
        """Загружает все JSON-отчёты из must/. Возвращает dict по ключам REPORT_FILES."""
        reports: Dict[str, Optional[dict]] = {}
        for key, fname in REPORT_FILES.items():
            path = self.must_dir / fname
            reports[key] = _load_json(path)
            if reports[key] is None:
                self.logger.debug(f"[REPORT] отсутствует/пуст: {fname}")
        return reports

    # ------------------------------------------------------------------
    # Извлечение плоской строки
    # ------------------------------------------------------------------
    def build_row(self) -> Dict[str, Any]:
        """
        Извлекает ключевые метрики из всех JSON-отчётов в плоский dict
        со стандартизованными именами колонок (см. COLUMNS).
        """
        r = self._load_all_reports()
        meta = r["metadata"]
        metrics = r["metrics"]
        peak = r["peak"]
        act = r["activation"]
        apd = r["apd"]
        cond = r["conduction"]
        alt = r["alternans"]
        clean = r["cleaning"]

        # alternans: данные вложены в alt["metrics"], но есть дубли в metrics.json
        alt_metrics = _g(alt, "metrics", default={}) or {}

        row: Dict[str, Any] = {}

        # --- идентификация ---
        row["sample_id"] = _g(meta, "sample_id", default=self.sample_id)
        row["dye"] = _g(meta, "dye", default=_g(metrics, "dye"))
        row["protocol"] = _g(meta, "protocol")
        row["model"] = _g(meta, "model")
        row["timepoint"] = _g(meta, "timepoint")
        row["recording_mode"] = _g(
            meta, "recording_mode", default=_g(metrics, "recording_mode")
        )

        # --- acquisition / geometry (из metadata — канонический источник) ---
        row["fps"] = _g(meta, "fps", default=_g(metrics, "fps"))
        row["n_frames"] = _g(meta, "n_frames", default=_g(metrics, "n_frames"))
        row["width"] = _g(meta, "width", default=_g(metrics, "width"))
        row["height"] = _g(meta, "height", default=_g(metrics, "height"))
        row["pixel_size_mm"] = _g(
            meta, "pixel_size_mm", default=_g(cond, "pixel_size_mm")
        )

        # --- stimulation ---
        # В metadata.json stim_hz — частота импульсов (может быть 500.0 = 2ms interval),
        # stim_hz_effective — эффективная частота стимуляции.
        row["stim_hz"] = _g(meta, "stim_hz", default=_g(metrics, "stim_hz"))
        row["stim_hz_effective"] = _g(
            meta, "stim_hz_effective", default=_g(metrics, "stim_hz_effective")
        )
        row["bcl_ms"] = _g(apd, "bcl_ms")

        # --- peaks / beats ---
        row["n_peaks"] = _g(peak, "n_peaks", default=_g(metrics, "n_peaks"))
        # n_beats: активация даёт отобранные биты; apd — n_windows.
        # Канонический n_beats берём из activation_report.
        row["n_beats"] = _g(act, "n_beats", default=_g(apd, "n_beats"))
        row["n_beats_selected"] = _g(metrics, "n_beats_selected")

        # --- activation ---
        row["n_active_pixels"] = _g(
            act, "n_active_pixels", default=_g(apd, "n_active_pixels")
        )
        row["valid_coverage"] = _g(act, "valid_coverage")
        row["tat_max_ms"] = _g(act, "tat_max_ms")
        row["tat_std_ms"] = _g(act, "tat_std_ms")

        # --- APD ---
        row["apd30_median_ms"] = _g(apd, "apd30_median_ms")
        row["apd50_median_ms"] = _g(apd, "apd50_median_ms")
        row["apd80_median_ms"] = _g(apd, "apd80_median_ms")

        # --- conduction ---
        row["cvl_m_s"] = _g(cond, "cvl_m_s")
        row["cvt_m_s"] = _g(cond, "cvt_m_s")
        row["anisotropy_ratio"] = _g(cond, "anisotropy_ratio")
        row["fiber_angle_deg"] = _g(cond, "fiber_angle_deg")

        # --- conduction: central ROI ---
        central = _g(cond, "central_roi", default={}) or {}
        if isinstance(central, dict):
            row["central_cvl_m_s"] = central.get("cvl_m_s")
            row["central_cvt_m_s"] = central.get("cvt_m_s")
            row["central_anisotropy_ratio"] = central.get("anisotropy_ratio")
            row["central_cv_m_s"] = central.get("cv_median_m_per_s")
        else:
            row["central_cvl_m_s"] = None
            row["central_cvt_m_s"] = None
            row["central_anisotropy_ratio"] = None
            row["central_cv_m_s"] = None

        # --- alternans ---
        # alternans_phenotype: в alternans_report.json ключ "phenotype",
        # в metrics.json — "alternans_phenotype".
        row["alternans_phenotype"] = _g(
            alt, "phenotype", default=_g(metrics, "alternans_phenotype")
        )
        # AC_95th_ms: alternans_report → metrics.AC_95th_percentile_ms;
        # metrics.json (top-level) → AC_95th_ms.
        row["AC_95th_ms"] = _g(
            alt_metrics, "AC_95th_percentile_ms",
            default=_g(metrics, "AC_95th_ms"),
        )
        row["concordance_index"] = _g(
            alt_metrics, "concordance_index",
            default=_g(metrics, "concordance_index"),
        )
        row["spectral_purity"] = _g(
            alt_metrics, "spectral_purity",
            default=_g(metrics, "spectral_purity"),
        )
        row["poincare_correlation"] = _g(
            alt_metrics, "poincare_correlation",
            default=_g(metrics, "poincare_correlation"),
        )

        # --- cleaning ---
        # Предпочитаем metrics.cleaning (вычислено CleaningAgent), fallback —
        # прямой разбор cleaning_report.json.
        clean_metrics = _g(metrics, "cleaning", default={}) or {}
        if isinstance(clean_metrics, dict):
            row["cleaning_deleted_count"] = clean_metrics.get(
                "deleted_count",
                len(_g(clean, "deleted", default=[]) or []),
            )
            row["cleaning_freed_mb"] = clean_metrics.get(
                "freed_mb", _g(clean, "total_freed_mb")
            )
        else:
            row["cleaning_deleted_count"] = len(
                _g(clean, "deleted", default=[]) or []
            )
            row["cleaning_freed_mb"] = _g(clean, "total_freed_mb")

        # Гарантируем все колонки присутствуют (None если не извлечено)
        for col in COLUMNS:
            row.setdefault(col, None)

        return row

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------
    def save_row(self, row: Dict[str, Any]) -> Dict[str, Path]:
        """Сохраняет строку как report.csv и report.json в must/."""
        # --- CSV (одна строка) ---
        csv_path = self.must_dir / "report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        self.logger.info(f"[MUST] Saved: {csv_path.name} ({len(COLUMNS)} columns)")

        # --- JSON ---
        json_path = self.must_dir / "report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"[MUST] Saved: {json_path.name}")

        return {"csv": csv_path, "json": json_path}

    # ------------------------------------------------------------------
    # Главный метод
    # ------------------------------------------------------------------
    def run(self, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Собирает сводный отчёт из всех JSON в must/.

        Returns:
            dict со статусом, путями сохранённых файлов и самой строкой row.
        """
        self.logger.info(
            f"ReportAgent started | must_dir={self.must_dir}"
        )

        row = self.build_row()

        # Логируем краткую сводку
        self.logger.info(
            f"[REPORT] sample={row.get('sample_id')} | "
            f"n_peaks={row.get('n_peaks')} n_beats={row.get('n_beats')} | "
            f"APD80={row.get('apd80_median_ms')}ms | "
            f"CVl={row.get('cvl_m_s')} CVt={row.get('cvt_m_s')} | "
            f"central CVl={row.get('central_cvl_m_s')} CVt={row.get('central_cvt_m_s')} aniso={row.get('central_anisotropy_ratio')} | "
            f"phenotype={row.get('alternans_phenotype')}"
        )

        paths = self.save_row(row)

        return {
            "status": "ok",
            "sample_id": self.sample_id,
            "row": row,
            "paths": paths,
        }


# ---------------------------------------------------------------------------
# Пакетный режим
# ---------------------------------------------------------------------------
def run_batch(
    results_root: Union[str, Path],
    config: Optional[Union[PipelineConfig, dict]] = None,
) -> Dict[str, Any]:
    """
    Итерирует все поддиректории results_root, где есть must/metadata.json,
    строит объединённый CSV (одна строка на образец) и сохраняет в
    <results-root>/report_batch.csv.

    Returns:
        dict со статусом, числом обработанных образцов, путём к CSV,
        и списком строк.
    """
    results_root = Path(results_root)
    if not results_root.is_dir():
        raise FileNotFoundError(f"results_root не найден: {results_root}")

    rows: List[Dict[str, Any]] = []
    sample_dirs = sorted(
        d for d in results_root.iterdir()
        if d.is_dir() and (d / "must").is_dir()
    )

    logger.info(f"[BATCH] найдено {len(sample_dirs)} кандидат-директорий в {results_root}")

    for sd in sample_dirs:
        sample_id = sd.name
        # Пропускаем директории без metadata.json (не образцы)
        if not (sd / "must" / "metadata.json").exists():
            logger.debug(f"[BATCH] пропуск {sample_id}: нет must/metadata.json")
            continue
        try:
            cfg = config if config is not None else PipelineConfig(
                {"results_root": str(results_root)}
            )
            agent = ReportAgent(sample_id, config=cfg)
            row = agent.build_row()
            agent.save_row(row)
            rows.append(row)
            logger.info(f"[BATCH] ✓ {sample_id}")
        except Exception as e:
            logger.error(f"[BATCH] ✗ {sample_id}: {e}")

    # Объединённый CSV
    batch_csv = results_root / "report_batch.csv"
    with open(batch_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info(
        f"[BATCH] объединённый CSV: {batch_csv} ({len(rows)} строк, "
        f"{len(COLUMNS)} колонок)"
    )

    return {
        "status": "ok",
        "n_samples": len(rows),
        "batch_csv": str(batch_csv),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="ReportAgent — сводный отчёт по всем стадиям пайплайна"
    )
    parser.add_argument(
        "sample_id",
        nargs="?",
        default=None,
        help="Sample ID (e.g. 004A). Опустите при --batch.",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Корневая директория результатов (по умолчанию: results)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Пакетный режим: обработать все sample-поддиректории в results-root",
    )
    args = parser.parse_args()

    if args.batch:
        result = run_batch(args.results_root)
        print("=== BATCH REPORT ===")
        print(f"n_samples: {result['n_samples']}")
        print(f"batch_csv: {result['batch_csv']}")
        sys.exit(0)

    if not args.sample_id:
        parser.error("sample_id обязателен (или используйте --batch).")

    cfg = PipelineConfig({"results_root": args.results_root})
    agent = ReportAgent(args.sample_id, config=cfg)
    result = agent.run()
    print("=== REPORT ===")
    print(json.dumps(result["row"], indent=2, ensure_ascii=False, default=str))
    print(f"\nCSV: {result['paths']['csv']}")
    print(f"JSON: {result['paths']['json']}")