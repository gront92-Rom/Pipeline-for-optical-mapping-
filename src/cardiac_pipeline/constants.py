"""
cardiac_pipeline.constants — paths, filenames, defaults.

Centralises magic strings. If you find yourself typing the same path twice,
add it here.
"""

from __future__ import annotations

# --- Results layout ---
RESULTS_ROOT_DEFAULT = "results"
"""Root directory for all pipeline outputs. Overridable via CLI flag."""

DATA_ROOT_DEFAULT = "data"
"""Root directory for raw input files."""

MUST_SUBDIR = "must"
"""Per-sample directory for canonical outputs that downstream stages depend on."""

DEBUG_SUBDIR = "debug"
"""Per-sample directory for diagnostics / plots / intermediate files."""

# --- Filenames ---
def manifest_filename(stage_id: str) -> str:
    """Per-stage manifest filename."""
    return f"stage_{stage_id}.json"

SUMMARY_FILENAME = "summary.json"
"""Aggregated run-summary filename (back-compat with run_cardiac.sh)."""

# --- Auto-detect (input file precedence) ---
INPUT_FILE_EXTENSIONS: tuple[str, ...] = (".rsh", ".gsh", ".rsd", ".gsd")
"""Extensions searched by auto_detect_input(), in priority order."""

# --- Logging (deliberately minimal; rich console output is in cli.py) ---
LOG_PREFIX_RUN = "  "
LOG_STATUS_OK = "[green]\u2713[/green] OK"
LOG_STATUS_SKIP = "[yellow]\u2299[/yellow] SKIP"
LOG_STATUS_FAIL = "[red]\u2717[/red] FAIL"
