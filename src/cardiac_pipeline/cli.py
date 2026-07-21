"""
cardiac_pipeline.cli — Typer-based CLI entry point.

After `pip install -e .`, the `cardiac-pipeline` command is available on PATH
on any OS (Linux, macOS, Windows).

This module is intentionally thin: it parses arguments, delegates to driver
for logic, and uses rich for output formatting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from cardiac_pipeline import constants as C
from cardiac_pipeline.driver import (
    is_cached,
    manifest_path,
    run_pipeline,
)
from cardiac_pipeline.stages import STAGES, get_stage

app = typer.Typer(
    help="Cardiac optical mapping pipeline — single CLI wrapper.",
    no_args_is_help=True,
)
console = Console()


# -----------------------------------------------------------------------------
# version
# -----------------------------------------------------------------------------

@app.command("version")
def version_cmd() -> None:
    """Print version and exit."""
    typer.echo("cardiac-pipeline 0.1.0")


# -----------------------------------------------------------------------------
# run
# -----------------------------------------------------------------------------

@app.command("run")
def run_cmd(
    sample_id: str = typer.Argument(..., help="Sample ID (e.g. 002A, 055A)."),
    input: Optional[str] = typer.Option(None, "--input", "-i", help="Path to .rsh/.gsh/.rsd/.gsd (auto-detect if omitted)."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite all outputs, ignore cache."),
    debug: bool = typer.Option(False, "--debug", "-d", help="Full traceback on error."),
    results_root: str = typer.Option(C.RESULTS_ROOT_DEFAULT, "--results", "-r", help="Results directory."),
) -> None:
    """Run the pipeline for one sample."""
    try:
        result = run_pipeline(
            sample_id=sample_id,
            input_file=input,
            force=force,
            debug=debug,
            results_root=results_root,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(code=2)
    except Exception as exc:
        console.print(f"[red]FATAL[/red] {type(exc).__name__}: {exc}")
        if debug:
            raise
        raise typer.Exit(code=1)

    console.print(f"\n[bold]OVERALL:[/bold] {'[green]\u2713 OK[/green]' if result.ok else '[red]\u2717 FAILED[/red]'}")
    raise typer.Exit(code=0 if result.ok else 1)


# -----------------------------------------------------------------------------
# status
# -----------------------------------------------------------------------------

@app.command("status")
def status_cmd(
    sample_id: str = typer.Argument(..., help="Sample ID."),
    results_root: str = typer.Option(C.RESULTS_ROOT_DEFAULT, "--results", "-r", help="Results directory."),
) -> None:
    """Show pipeline stage status (reads stage_*.json manifests)."""
    table = Table(title=f"Pipeline status: {sample_id}")
    table.add_column("Stage", style="bold")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("When")
    table.add_column("Cache", justify="right")

    n_cached = 0
    n_missing = 0

    for stage in STAGES:
        mp = manifest_path(sample_id, stage.name, Path(results_root))
        cached = is_cached(sample_id, stage.name, Path(results_root))

        if not cached:
            table.add_row(stage.name, "[red]MISSING[/red]", "—", "—", "[red]no[/red]")
            n_missing += 1
            continue

        n_cached += 1
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
        except Exception as exc:
            table.add_row(stage.name, "[red]ERROR[/red]", "—", "—", "[green]yes[/green]")
            console.print(f"  [red]parse error[/red] in {mp}: {exc}")
            continue

        status = data.get("status", "UNKNOWN")
        styled = _color_status(status)
        dur = data.get("elapsed_s", 0.0)
        dur_str = f"{dur:.2f}s" if dur else "—"
        when = (data.get("finished_at") or data.get("started_at") or "")[:19]
        table.add_row(stage.name, styled, dur_str, when, "[green]yes[/green]")

    console.print(table)
    summary = f"{n_cached} cached / {n_missing} missing / {len(STAGES)} total"
    if n_missing:
        console.print(f"\n[yellow]{summary}[/yellow]. Run `cardiac-pipeline run {sample_id}` to fill gaps.")
    else:
        console.print(f"\n[green]{summary}[/green]. All stages present.")


def _color_status(status: str) -> str:
    """Map status string to Rich color markup."""
    return {
        "OK":      "[green]OK[/green]",
        "WARN":    "[yellow]WARN[/yellow]",
        "REJECT":  "[magenta]REJECT[/magenta]",
        "ERROR":   "[red]ERROR[/red]",
        "SKIPPED": "[dim]SKIPPED[/dim]",
    }.get(status, f"[dim]{status}[/dim]")


# -----------------------------------------------------------------------------
# list-agents
# -----------------------------------------------------------------------------

@app.command("list-agents")
def list_agents_cmd() -> None:
    """Show registered pipeline stages."""
    table = Table(title="Registered stages")
    table.add_column("#", justify="right")
    table.add_column("Stage ID", style="bold")
    table.add_column("Agent class")
    table.add_column("Description")

    for i, stage in enumerate(STAGES, 1):
        cls_name = f"{stage.agent_cls.__module__}.{stage.agent_cls.__name__}"
        table.add_row(str(i), stage.name, cls_name, stage.description)

    console.print(table)


# -----------------------------------------------------------------------------
# ingest
# -----------------------------------------------------------------------------

@app.command("ingest")
def ingest_cmd(
    source: str = typer.Argument(..., help="Flat directory with raw MiCAM files."),
    out: str = typer.Option("data", "--out", "-o", help="Output root (default: data/)."),
    copy: bool = typer.Option(False, "--copy", help="Copy instead of move."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only, don't move."),
) -> None:
    """Distribute raw MiCAM files into data/<sample>/<treatment>/ folders.

    Treatment is auto-detected from filename: -bs2 -> bsl, -bleb -> bleb, etc.
    Each sample gets a metadata.json with its full treatment list.
    """
    from cardiac_pipeline.agents.ingest_agent import (
        IngestAgent,
        TREATMENT_PIPELINE_MODES,
    )
    agent = IngestAgent()
    summary = agent.run(source_dir=source, out_dir=out, copy=copy, force=force, dry_run=dry_run)

    # Summary table
    table = Table(title=f"Ingest summary: {source} → {out}/")
    table.add_column("Sample", style="bold")
    table.add_column("Treatments", style="cyan")
    table.add_column("Pipeline mode", style="dim")

    for sid in sorted(summary.samples):
        treatments = summary.samples[sid]
        # Show pipeline mode for first treatment (typically all same)
        first_t = treatments[0]
        mode = " → ".join(TREATMENT_PIPELINE_MODES.get(first_t, ["?"]))
        table.add_row(sid, ", ".join(treatments), mode)

    console.print(table)

    console.print(f"\n[bold]Totals:[/bold]")
    console.print(f"  Files moved/copied: {summary.n_files_moved}")
    console.print(f"  Files skipped (existing): {summary.n_files_skipped}")
    console.print(f"  Samples: {summary.n_samples}")
    console.print(f"  Treatment-instances: {summary.n_treatments}")

    if summary.skipped_files:
        console.print(f"\n[yellow]Skipped (no sample_id in name): {len(summary.skipped_files)}[/yellow]")
        for name in summary.skipped_files[:5]:
            console.print(f"  - {name}")
        if len(summary.skipped_files) > 5:
            console.print(f"  ... and {len(summary.skipped_files) - 5} more")

    if dry_run:
        console.print("\n[yellow]DRY-RUN[/yellow] — no files were moved. Re-run without --dry-run to apply.")
    else:
        console.print(f"\n[green]✓[/green] Ingest complete. Next: cardiac-pipeline run <sample_id> --treatment <bsl|bleb|iso|...>")


# -----------------------------------------------------------------------------
# describe (helper for docs / debugging)
# -----------------------------------------------------------------------------

@app.command("describe")
def describe_cmd(stage_id: str = typer.Argument(..., help="Stage ID to describe.")) -> None:
    """Print detailed info about one stage."""
    try:
        stage = get_stage(stage_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2)

    console.print(f"[bold]{stage.name}[/bold]")
    console.print(f"  Class:     {stage.agent_cls.__module__}.{stage.agent_cls.__name__}")
    console.print(f"  Requires input file: {stage.requires_input}")
    console.print(f"  Description: {stage.description}")


# -----------------------------------------------------------------------------
# entry point
# -----------------------------------------------------------------------------

def main() -> None:
    """Console-script entry point (called by `[project.scripts]`)."""
    app()


if __name__ == "__main__":
    main()
