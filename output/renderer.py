"""Render a `SynthesisResult` to the terminal with rich.

This is the human-facing summary printed at the end of a run: a colour-coded header,
the architecture panel, module/entry-point tables, the contributor quickstart, and —
only when present — a red circular-dependency warning. All output goes through a
single module-level `Console`; tests swap it for a `StringIO`-backed console to
capture and assert on the markup.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ingestion.types import CompressedContext, SynthesisResult

# stderr=False: the rendered report is the primary user-facing output.
console = Console(stderr=False)

MAX_MODULE_ROWS = 10


def _complexity_color(score: int) -> str:
    if score <= 3:
        return "green"
    if score <= 6:
        return "yellow"
    return "red"


def render(
    result: SynthesisResult,
    *,
    repo_name: str,
    compressed: CompressedContext | None = None,
) -> None:
    """Print the full analysis. `compressed` is optional and only drives the footer."""
    color = _complexity_color(result.complexity_score)
    generated = result.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    header = (
        f"[bold cyan]{repo_name}[/]\n"
        f"complexity [{color}]{result.complexity_score}/10[/]   "
        f"[dim]generated {generated}[/]"
    )
    console.print(Panel(header, title="KairoRM", border_style="cyan"))

    console.print(
        Panel(result.architecture_summary or "[dim]no summary[/]", title="Architecture")
    )

    modules_table = Table(title="Modules", expand=True)
    modules_table.add_column("Name", style="bold")
    modules_table.add_column("Path", style="cyan")
    modules_table.add_column("Responsibility")
    for module in result.modules[:MAX_MODULE_ROWS]:
        modules_table.add_row(module.name, module.path, module.responsibility)
    console.print(modules_table)
    if len(result.modules) > MAX_MODULE_ROWS:
        # highlight=False so rich doesn't style the count and split the text span.
        console.print(
            f"[dim]...and {len(result.modules) - MAX_MODULE_ROWS} more[/]", highlight=False
        )

    if result.entry_points:
        entry_table = Table(title="Entry Points", expand=True)
        entry_table.add_column("Name", style="bold")
        entry_table.add_column("File", style="cyan")
        entry_table.add_column("Description")
        for ep in result.entry_points:
            entry_table.add_row(ep.name, ep.file, ep.description)
        console.print(entry_table)

    if result.contributor_quickstart:
        steps = "\n".join(
            f"{i}. {step}" for i, step in enumerate(result.contributor_quickstart, 1)
        )
        console.print(Panel(Markdown(steps), title="Contributor Quickstart"))

    if result.circular_risks:
        console.print(
            Panel(
                "\n".join(f"• {risk}" for risk in result.circular_risks),
                title="⚠ Circular Dependencies",
                border_style="red",
                style="red",
            )
        )

    if compressed is not None:
        footer = f"[dim]Q&A context: {compressed.token_count} tokens[/]"
        if compressed.truncated:
            footer += "  [yellow]⚠ context truncated to fit budget[/]"
        console.print(footer)
