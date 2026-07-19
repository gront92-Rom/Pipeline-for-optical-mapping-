#!/usr/bin/env bash
# run_cardiac.sh — clean, reliable wrapper to run the full cardiac optical mapping pipeline.
#
# Usage:
#   ./run_cardiac.sh [--debug] <sample_id> [input_file]
#
# Examples:
#   ./run_cardiac.sh 000A
#   ./run_cardiac.sh --debug 004A
#   ./run_cardiac.sh 004A /data/raw/004A/recording.rsh
#
# Modes:
#   Normal mode (default)  — clean output, good for regular runs and CI
#   Debug mode (--debug)   — maximum information for finding errors:
#                              • full traceback in terminal
#                              • debug_report.md with analysis
#                              • list of created files on failure
#
# Exit codes:
#   0 - all stages successful
#   1 - pipeline failed at some stage
#   2 - usage / input error

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="${PIPELINE_ROOT}/results"
DATA_ROOT="${PIPELINE_ROOT}/data"
SRC_PATH="${PIPELINE_ROOT}/src"
TMP_ROOT="${PIPELINE_ROOT}/tmp"
DRIVER_TEMPLATE="${TMP_ROOT}/cardiac_driver_$$.py"

DEBUG_MODE=0

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
print_usage() {
    cat <<EOF
Usage: $(basename "$0") [--debug|-d] <sample_id> [input_file]

  --debug, -d   Enable detailed error reporting (recommended when debugging)
  sample_id     Logical sample name (e.g. 000A, 004A)
  input_file    Optional path to .rsh/.gsh file (auto-detected if omitted)

Data layout expected next to this script:
  data/<sample_id>/*.rsh (or .gsh / .rsd / .gsd) — auto-detected
  src/                  — Python package (added to PYTHONPATH)
  results/              — created automatically
  tmp/                  — temporary driver script (created/removed automatically)

Examples:
  $(basename "$0") 000A
  $(basename "$0") --debug 004A
  $(basename "$0") 004A /full/path/to/recording.rsh
EOF
}

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug|-d)
            DEBUG_MODE=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

SAMPLE_ID="${1:-}"
INPUT_FILE="${2:-}"

if [[ -z "$SAMPLE_ID" ]]; then
    print_usage
    exit 2
fi

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
log()      { echo "$@"; }
error()    { echo "ERROR: $*" >&2; }
die()      { error "$*"; exit 2; }

print_header() {
    cat <<EOF

╔════════════════════════════════════════════════════════════════════════════╗
║                    CARDIAC OPTICAL MAPPING PIPELINE                        ║
╚════════════════════════════════════════════════════════════════════════════╝
EOF
}

print_footer() {
    local exit_code=$1
    cat <<EOF

╔════════════════════════════════════════════════════════════════════════════╗
║                              RUN FINISHED                                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║  Exit code   : $exit_code
║  Sample      : $SAMPLE_ID
║  Log         : $LOG_FILE
║  Summary     : $SUMMARY_JSON
EOF

    if [[ $DEBUG_MODE -eq 1 && -f "$DEBUG_REPORT" ]]; then
        echo "║  Debug report: $DEBUG_REPORT"
    fi
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
}

# =============================================================================
# INPUT FILE AUTO-DETECTION
# =============================================================================
if [[ -z "$INPUT_FILE" ]]; then
    CANDIDATE_DIR="$DATA_ROOT/$SAMPLE_ID"
    if [[ -d "$CANDIDATE_DIR" ]]; then
        INPUT_FILE=$(find "$CANDIDATE_DIR" -maxdepth 1 -type f -name '*.rsh' 2>/dev/null | sort | head -1 || true)
        [[ -z "$INPUT_FILE" ]] && INPUT_FILE=$(find "$CANDIDATE_DIR" -maxdepth 1 -type f -name '*.gsh' 2>/dev/null | sort | head -1 || true)
        [[ -z "$INPUT_FILE" ]] && INPUT_FILE=$(find "$CANDIDATE_DIR" -maxdepth 1 -type f \( -name '*.rsd' -o -name '*.gsd' \) 2>/dev/null | sort | head -1 || true)
    fi
fi

if [[ -z "$INPUT_FILE" || ! -f "$INPUT_FILE" ]]; then
    die "Cannot find input file for sample '$SAMPLE_ID'
  Looked in: $DATA_ROOT/$SAMPLE_ID/
  Provide explicitly: $0 $SAMPLE_ID /full/path/to/file.rsh"
fi

# =============================================================================
# PREPARE OUTPUT
# =============================================================================
mkdir -p "$RESULTS_ROOT/$SAMPLE_ID"
mkdir -p "$TMP_ROOT"  # local temp dir, avoids /tmp portability issues
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$RESULTS_ROOT/$SAMPLE_ID/run_${TS}.log"
SUMMARY_JSON="$RESULTS_ROOT/$SAMPLE_ID/summary.json"
DEBUG_REPORT="$RESULTS_ROOT/$SAMPLE_ID/debug_report.md"
ERROR_TB_FILE="$RESULTS_ROOT/$SAMPLE_ID/error_traceback.txt"

# =============================================================================
# BANNER
# =============================================================================
print_header

MODE_LABEL="NORMAL"
[[ $DEBUG_MODE -eq 1 ]] && MODE_LABEL="DEBUG"

cat <<EOF
  Mode        : $MODE_LABEL
  Sample ID   : $SAMPLE_ID
  Input file  : $INPUT_FILE
  Results     : $RESULTS_ROOT/$SAMPLE_ID
  Started     : $(date '+%Y-%m-%d %H:%M:%S')
EOF

# =============================================================================
# GENERATE PYTHON DRIVER
# =============================================================================
cat > "$DRIVER_TEMPLATE" <<'PYEOF'
#!/usr/bin/env python3
"""
Auto-generated cardiac pipeline driver.
Supports normal and debug modes.
"""
import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.environ.get("PIPELINE_SRC", "src"))

from cardiac_pipeline.base_agent import PipelineConfig
from cardiac_pipeline.agents.loader_agent import LoaderAgent
# SidelineAgent class removed (rewritten as functions in commit 3e9ea1c)
from cardiac_pipeline.agents.mask_agent import MaskAgent
from cardiac_pipeline.agents.peak_detector_agent import PeakDetectorAgent
from cardiac_pipeline.agents.activation_agent import ActivationAgent
from cardiac_pipeline.agents.apd_agent import APDAgent
from cardiac_pipeline.agents.conduction_agent import ConductionAgent
from cardiac_pipeline.agents.alternans_agent import AlternansAgent
from cardiac_pipeline.agents.cleaning_agent import CleaningAgent
from cardiac_pipeline.agents.report_agent import ReportAgent

DEBUG = os.environ.get("DEBUG", "0") == "1"
SAMPLE_ID = os.environ["SAMPLE_ID"]
INPUT_FILE = os.environ["INPUT_FILE"]
RESULTS_ROOT = os.environ["RESULTS_ROOT"]


def generate_debug_report(failed_stage: str, error_msg: str, tb: str, elapsed: float) -> str:
    """Create a human-readable debug report in Markdown."""
    report_path = Path(RESULTS_ROOT) / SAMPLE_ID / "debug_report.md"
    results_dir = Path(RESULTS_ROOT) / SAMPLE_ID

    # List files created so far
    created_files = []
    if results_dir.exists():
        for f in sorted(results_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(results_dir)
                created_files.append(str(rel))

    content = f"""# Debug Report — {SAMPLE_ID}

**Failed at stage:** `{failed_stage}`  
**Time of failure:** {datetime.now(timezone.utc).isoformat()}  
**Elapsed in stage:** {elapsed:.2f} s

## Error Message
```
{error_msg}
```

## Full Traceback
```python
{tb}
```

## Files Created in Results Folder (up to failure)
"""
    if created_files:
        for f in created_files:
            content += f"- `{f}`\n"
    else:
        content += "_No files were created yet._\n"

    content += f"""
## How to investigate
1. Open the stage output folder (if exists)
2. Check the PNG visualizations from the failed stage
3. Look at `summary.json` for structured data
4. Re-run with the same command to reproduce

---
*Generated by run_cardiac.sh in DEBUG mode*
"""
    report_path.write_text(content, encoding="utf-8")
    return str(report_path)


def main() -> int:
    cfg = PipelineConfig({
        "results_root": RESULTS_ROOT,
        "sample_id": SAMPLE_ID,
        "peak_detector": {"n_regions": 3},
    })

    stages = [
        ("1/9 Loader",         LoaderAgent,        {"input_path": INPUT_FILE, "force": True}),
        ("2/9 Mask",           MaskAgent,          {"force": True}),
        ("3/9 PeakDetector",   PeakDetectorAgent,  {"force": True}),
        ("4/9 Activation",     ActivationAgent,    {"force": True}),
        ("5/9 APD",            APDAgent,           {"force": True}),
        ("6/9 Conduction",     ConductionAgent,    {"force": True}),
        ("7/9 Alternans",      AlternansAgent,     {"force": True}),
        ("8/9 Cleaning",       CleaningAgent,      {"force": True}),
        ("9/9 Report",         ReportAgent,        {"force": True}),
    ]

    # --- Sideline stage removed (rewritten as functions, not Agent class) ---
    results = {
        "sample_id": SAMPLE_ID,
        "input_file": INPUT_FILE,
        "mode": "debug" if DEBUG else "normal",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
        "ok": True,
    }

    t_total = time.time()
    overall_ok = True
    failed_stage_name = ""
    last_error = ""
    last_tb = ""
    last_elapsed = 0.0

    for stage_name, AgentClass, kwargs in stages:
        t0 = time.time()
        print(f"\n=== {stage_name} ===", flush=True)

        try:
            agent = AgentClass(SAMPLE_ID, config=cfg)
            result = agent.run(**kwargs)

            elapsed = time.time() - t0
            results["stages"][stage_name] = {
                "ok": True,
                "elapsed_s": round(elapsed, 2),
            }
            print(f"  ✓ OK in {elapsed:.1f}s", flush=True)

        except Exception as exc:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            last_error = str(exc)
            last_tb = tb
            last_elapsed = elapsed
            failed_stage_name = stage_name

            results["stages"][stage_name] = {
                "ok": False,
                "elapsed_s": round(elapsed, 2),
                "error": last_error,
            }

            print(f"  ✗ FAILED in {elapsed:.1f}s", flush=True)
            print(f"    Error: {last_error}", flush=True)

            if DEBUG:
                print("\n" + "="*60, flush=True)
                print("FULL TRACEBACK (DEBUG MODE)", flush=True)
                print("="*60, flush=True)
                traceback.print_exc()
                print("="*60 + "\n", flush=True)

                # Save raw traceback
                tb_path = Path(RESULTS_ROOT) / SAMPLE_ID / "error_traceback.txt"
                tb_path.write_text(tb, encoding="utf-8")

                # Generate nice debug report
                report_path = generate_debug_report(stage_name, last_error, tb, elapsed)
                print(f"Debug report saved to: {report_path}", flush=True)

            overall_ok = False
            break

    results["total_elapsed_s"] = round(time.time() - t_total, 2)
    results["ok"] = overall_ok
    results["finished_at"] = datetime.now(timezone.utc).isoformat()

    if not overall_ok:
        results["failed_stage"] = failed_stage_name
        results["last_error"] = last_error

    # Write summary.json
    summary_path = Path(RESULTS_ROOT) / SAMPLE_ID / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSummary written to: {summary_path}", flush=True)
    status = "SUCCESS ✓" if overall_ok else "FAILED ✗"
    print(f"\nOVERALL STATUS: {status}", flush=True)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
PYEOF

# =============================================================================
# RUN THE PIPELINE
# =============================================================================
export SAMPLE_ID="$SAMPLE_ID"
export INPUT_FILE="$INPUT_FILE"
export RESULTS_ROOT="$RESULTS_ROOT"
export PIPELINE_SRC="$SRC_PATH"
export DEBUG="$DEBUG_MODE"

log "Running pipeline..."
PYTHONPATH="$SRC_PATH" python3 "$DRIVER_TEMPLATE" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=$?

rm -f "$DRIVER_TEMPLATE"
# Clean empty temp dir if nothing left inside
rmdir "$TMP_ROOT" 2>/dev/null || true

# =============================================================================
# FINAL REPORT
# =============================================================================
print_footer "$EXIT_CODE"

if [[ $EXIT_CODE -ne 0 ]]; then
    echo
    echo "To investigate the failure:"
    echo "  • Open: $SUMMARY_JSON"
    if [[ $DEBUG_MODE -eq 1 ]]; then
        echo "  • Read the debug report: $DEBUG_REPORT"
        echo "  • Full traceback: $ERROR_TB_FILE"
    else
        echo "  • Re-run with --debug for full traceback and debug_report.md"
    fi
    echo
fi

exit "$EXIT_CODE"
