"""Render a `SynthesisResult` to the terminal with rich.

This is the human-facing summary printed at the end of a run: a colour-coded header,
the architecture panel, module/entry-point tables, the contributor quickstart, and —
only when present — a red circular-dependency warning. All output goes through a
single module-level `Console`; tests swap it for a `StringIO`-backed console to
capture and assert on the markup.
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ingestion.types import CompressedContext, SynthesisResult
from output.theme import ACCENT, BORDER, HIGHLIGHT, MUTED, SURFACE, TEXT

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
    color = _complexity_color(result.complexity_score)  # green/yellow/red — semantic
    generated = result.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    header_style = f"{HIGHLIGHT} on {SURFACE}"
    header = (
        f"[bold {ACCENT}]{repo_name}[/]\n"
        f"complexity [{color}]{result.complexity_score}/10[/]   "
        f"[{MUTED}]generated {generated}[/]"
    )
    console.print(
        Panel(header, title=f"[{HIGHLIGHT}]KairoRM[/]", border_style=BORDER, box=box.ROUNDED)
    )

    if result.reading_order:
        lines = [
            f"[{ACCENT}]{i}. {step.path}[/]  [{MUTED}]{step.reason}[/]"
            for i, step in enumerate(result.reading_order, 1)
        ]
        console.print(
            Panel(
                "\n".join(lines),
                title=f"[{HIGHLIGHT}]▶ Start Here[/]",
                border_style=BORDER,
                box=box.ROUNDED,
            )
        )

    console.print(
        Panel(
            f"[{TEXT}]{result.architecture_summary or 'no summary'}[/]",
            title=f"[{HIGHLIGHT}]Architecture[/]",
            border_style=BORDER,
            box=box.ROUNDED,
        )
    )

    modules_table = Table(
        title=f"[{HIGHLIGHT}]Modules[/]",
        border_style=BORDER,
        header_style=header_style,
        box=box.ROUNDED,
        expand=True,
    )
    modules_table.add_column("Name", style=f"bold {TEXT}")
    modules_table.add_column("Path", style=HIGHLIGHT)
    modules_table.add_column("Responsibility", style=TEXT)
    for module in result.modules[:MAX_MODULE_ROWS]:
        modules_table.add_row(module.name, module.path, module.responsibility)
    console.print(modules_table)
    if len(result.modules) > MAX_MODULE_ROWS:
        # highlight=False so rich doesn't style the count and split the text span.
        console.print(
            f"[{MUTED}]...and {len(result.modules) - MAX_MODULE_ROWS} more[/]", highlight=False
        )

    if result.entry_points:
        entry_table = Table(
            title=f"[{HIGHLIGHT}]Entry Points[/]",
            border_style=BORDER,
            header_style=header_style,
            box=box.ROUNDED,
            expand=True,
        )
        entry_table.add_column("Name", style=f"bold {TEXT}")
        entry_table.add_column("File", style=HIGHLIGHT)
        entry_table.add_column("Description", style=TEXT)
        for ep in result.entry_points:
            entry_table.add_row(ep.name, ep.file, ep.description)
        console.print(entry_table)

    if result.key_dependencies:
        deps = ", ".join(f"[{HIGHLIGHT}]{dep}[/]" for dep in result.key_dependencies)
        console.print(
            Panel(
                deps,
                title=f"[{HIGHLIGHT}]Key Dependencies[/]",
                border_style=BORDER,
                box=box.ROUNDED,
            )
        )

    if result.contributor_quickstart:
        steps = "\n".join(
            f"[{HIGHLIGHT}]{i}.[/] [{TEXT}]{step}[/]"
            for i, step in enumerate(result.contributor_quickstart, 1)
        )
        console.print(
            Panel(
                steps,
                title=f"[{HIGHLIGHT}]Contributor Quickstart[/]",
                border_style=BORDER,
                box=box.ROUNDED,
            )
        )

    if result.circular_risks:
        # Kept red: a genuine warning, and asserted by the renderer tests.
        console.print(
            Panel(
                "\n".join(f"• {risk}" for risk in result.circular_risks),
                title="⚠ Circular Dependencies",
                border_style="red",
                style="red",
            )
        )

    if compressed is not None:
        footer = f"[{MUTED}]Q&A context: {compressed.token_count} tokens[/]"
        if compressed.truncated:
            footer += f"  [{HIGHLIGHT}]⚠ context truncated to fit budget[/]"
        console.print(footer)
